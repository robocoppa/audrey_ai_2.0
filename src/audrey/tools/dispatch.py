"""Tool dispatcher — execute one tool_call against the right server.

Given an Ollama tool_call (`{"function": {"name": "...", "arguments": {...}}}`)
and a registry, POST the arguments to the originating server and return the
response body as a JSON-string suitable for inclusion in a `role=tool` message.

Long results are truncated at `agentic.react.max_tool_result_chars` — model
context burns fast otherwise. **A truncated result says how much was lost and
that retrying will not help**, because saying only that a cut happened is worse
than it sounds: on 2026-08-05 a model read a bare `…[truncated]`, correctly
reported it could give only a partial excerpt, and then invented a way to ask
for the rest. Whole list items are dropped in preference to a character cut, so
the body the model receives is still parseable JSON it can count.

Errors (network, 4xx, 5xx) become tool messages too — the model can decide
whether to retry, re-prompt the user, or apologize. We never raise out of
this module so the ReAct loop stays in control.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from audrey.metrics import tool_call_seconds, tool_calls_total
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    name: str                  # tool name (or "?" if unknown)
    call_id: str | None        # tool_call_id from the model, if provided
    content: str               # JSON-string result body (possibly truncated/error)
    elapsed_s: float
    is_error: bool


# Tools that are scoped by the caller's user id. When dispatching these,
# we overwrite any `user` argument the model supplied with the real
# pipeline user_id — prevents the model from querying another user's
# data, and spares it from having to guess its own id.
#
# ADDING A NEW USER-SCOPED TOOL — two places to edit:
#   1. The tools-server route (`tools-server/app.py`) declares the
#      operation and its `user` request-body field.
#   2. This set. Without the name here, the dispatcher will pass the
#      model-supplied `user` argument straight to the network call,
#      and the security invariant is broken.
# If you only edit (1) and forget (2), other users' data is reachable
# from the model. There is no startup check for this today — if you
# add one, look for a `user` property in `ToolSpec.parameters` and
# warn when the name isn't in `_USER_SCOPED_TOOLS`.
_USER_SCOPED_TOOLS: frozenset[str] = frozenset({
    "kb_search",
    "kb_image_search",
    "memory_recall",
    "memory_search",
    "memory_store",
    "chat_history_search",
    # Reaches `POST /v1/files/list`, which is service-token-authenticated and
    # names its target user in the request body. That route has no second
    # defence — this entry is the whole of what stops a prompt naming someone
    # else's address from returning their file list.
    "list_my_files",
    # `POST /v1/files/artifact`, same arrangement and higher stakes: that route
    # returns a user's document text verbatim, not a listing. Added in the same
    # commit as the route, which is the rule `audit_user_scoping` exists to
    # enforce — it only warns, and a warning is not a gate.
    "get_file_text",
})


def audit_user_scoping(registry: ToolRegistry) -> list[str]:
    """Name every discovered tool that takes a `user` argument but isn't scoped.

    The gap this closes: the tools-server and `_USER_SCOPED_TOOLS` are two
    files that must be edited together, and nothing connected them. Ship the
    route without the set entry and the dispatcher forwards the *model-supplied*
    `user` — so a prompt naming another address reads their data, with no error
    anywhere and nothing in a test suite that would notice.

    A **warning, not a failure**, and deliberately so. `user` is an ordinary
    word; a tool could legitimately take one that means something else
    (a GitHub handle, a display name), and refusing to boot over a false
    positive would make the next person delete the check rather than read it.
    Returns the offending names so a caller can assert on them.
    """
    unscoped = [
        spec.name
        for spec in registry.specs()
        if "user" in (spec.parameters.get("properties") or {})
        and spec.name not in _USER_SCOPED_TOOLS
    ]
    if unscoped:
        log.warning(
            "tools: %s take a `user` argument but are NOT in _USER_SCOPED_TOOLS — "
            "the dispatcher will forward whatever the model supplies. If any of "
            "these are per-user, add them to that set in tools/dispatch.py.",
            unscoped,
        )
    return unscoped


# The sentence that exists because a model read `…[truncated]` and concluded it
# should ask again with a bigger `top_k`. It could not have known better: the
# old marker said a cut happened and nothing else, so "ask for more" is a
# perfectly reasonable inference from it — and a completely useless one, since
# the cap is applied to the *response* after the query has already run.
#
# Measured 2026-08-05 against a real transcript search: at 2000 chars a
# `kb_search` keeps ~1.7 hits whether top_k is 5, 10 or 20. The retry costs a
# ReAct round out of three and returns an identical amount of text.
_RETRY_IS_POINTLESS = (
    "This cap is on the size of THIS RESULT, not on your query — re-running "
    "with a larger top_k or limit returns the same amount of text, or less. "
    "A SMALLER top_k can return more, because a response that fits whole is "
    "not trimmed at all. Otherwise use what is here, or narrow the query so "
    "the passages you want rank higher."
)


def _truncate(s: str, limit: int) -> str:
    """Cut a result to `limit` chars, saying how much was lost.

    The bare `…[truncated]` this replaces was the whole problem: a model that
    cannot see how much it is missing has to guess, and on 2026-08-05 one
    guessed out loud — "due to system limitations I can only provide a partial
    excerpt … request it again in a new session". The limitation was real and
    the model was right to report it; it invented a remedy only because nothing
    told it the size of the hole or that no remedy exists.

    Prefer `_truncate_payload`, which keeps JSON parseable. This is the
    fallback for text that has no structure to drop.
    """
    if len(s) <= limit:
        return s
    marker = f"\n…[truncated: showing {{shown:,}} of {len(s):,} chars. {_RETRY_IS_POINTLESS}]"
    # Two passes: the marker's own length depends on the number inside it, so
    # the first pass sizes it and the second fills in the count that survived.
    kept = max(0, limit - len(marker.format(shown=limit)))
    return s[:kept] + marker.format(shown=kept)


def _truncate_payload(payload: Any, content: str, limit: int) -> str:
    """Shrink a tool result to `limit` chars, dropping whole list items first.

    Cutting a JSON body mid-string leaves the model holding **invalid JSON**
    and a half-word, which it then has to interpret. Dropping whole elements
    from the longest list keeps the payload parseable and turns the loss into a
    number the model can report: "showing 2 of 12 results" is something it can
    say truthfully, where a severed brace is something it has to guess about.

    Falls back to a character cut when there is no list to shrink, or when the
    payload is so large without one that dropping every item still does not
    fit.
    """
    if len(content) <= limit:
        return content
    if not isinstance(payload, dict):
        return _truncate(content, limit)

    # The list carrying the weight — `results` for kb_search and web_search,
    # `files` for list_my_files, `hits` for memory. Chosen by serialized size
    # rather than by name so a tool added later needs no entry here.
    key = max(
        (k for k, v in payload.items() if isinstance(v, list) and v),
        key=lambda k: len(json.dumps(payload[k], ensure_ascii=False, default=str)),
        default=None,
    )
    if key is None:
        return _truncate(content, limit)

    items = payload[key]
    for keep in range(len(items) - 1, 0, -1):
        trial = dict(payload)
        trial[key] = items[:keep]
        trial["_truncated"] = (
            f"showing {keep} of {len(items)} {key} — {len(items) - keep} omitted "
            f"to fit a {limit:,}-char cap. {_RETRY_IS_POINTLESS}"
        )
        rendered = json.dumps(trial, ensure_ascii=False, default=str)
        if len(rendered) <= limit:
            return rendered
    # Even one item does not fit. A character cut is all that is left, and the
    # marker still reports the real scale of the loss.
    return _truncate(content, limit)


def _force_user_tag(tags: str, user_id: str) -> str:
    """Strip any existing `user:<anything>` token and append `user:<user_id>`."""
    parts = [t for t in tags.replace(",", " ").split() if not t.startswith("user:")]
    parts.append(f"user:{user_id}")
    return ",".join(parts)


async def dispatch_one(
    client: httpx.AsyncClient,
    registry: ToolRegistry,
    tool_call: dict[str, Any],
    *,
    max_result_chars: int,
    timeout_s: float,
    user_id: str | None = None,
) -> ToolResult:
    """Execute one tool_call. Always returns a ToolResult — never raises.

    Emits `audrey_tool_calls_total{tool,outcome}` for every return path and
    `audrey_tool_call_seconds{tool}` for paths that actually made a network
    call. Outcomes: `ok` (2xx + parsed), `error` (bad args, unknown tool,
    4xx, 5xx, non-timeout transport), `timeout` (httpx.TimeoutException).
    """
    fn = (tool_call.get("function") or {})
    name = str(fn.get("name") or "?")
    call_id = tool_call.get("id")
    # Hold onto the raw arguments value so the JSON-parse error path can
    # always log/echo the original — even if a future edit changes the
    # order or rebinds `args` before the except branch fires.
    raw_args = fn.get("arguments")
    args = raw_args

    # Ollama sometimes passes arguments as a JSON-encoded string instead of a dict.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            log.warning("dispatch: %s arguments not JSON: %r", name, raw_args[:200])
            tool_calls_total.labels(tool=name, outcome="error").inc()
            return ToolResult(
                name=name, call_id=call_id,
                content=json.dumps({"error": "arguments_not_json", "raw": raw_args[:500]}),
                elapsed_s=0.0, is_error=True,
            )
    if args is None:
        args = {}
    if not isinstance(args, dict):
        tool_calls_total.labels(tool=name, outcome="error").inc()
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "arguments_not_object", "got": str(type(args).__name__)}),
            elapsed_s=0.0, is_error=True,
        )

    # Overwrite `user` for user-scoped tools with the real pipeline user_id.
    # Prevents the model from querying or writing to another user's scope,
    # and spares it from having to guess its own id. For `memory_store`, the
    # user lives inside the free-form `tags` string — we enforce it there too.
    if user_id and name in _USER_SCOPED_TOOLS:
        if name == "memory_store":
            args["tags"] = _force_user_tag(str(args.get("tags") or ""), user_id)
        else:
            args["user"] = user_id

    spec = registry.get(name)
    if spec is None:
        log.warning("dispatch: unknown tool %r (registered: %s)", name, registry.names())
        tool_calls_total.labels(tool=name, outcome="error").inc()
        # Deliberately does NOT echo `name` back in the body. A model that
        # invented a tool has that name in context exactly once (its own call);
        # repeating it in the error text reinforces it, and a prompt that names
        # a tool makes models call it. The 2026-07-22 research eval caught this:
        # a worker invented `web_fetch`, got the name handed back, and called it
        # a second time. `available` gives it somewhere real to go instead.
        # (The `role=tool` envelope in `to_tool_message` still carries `name` —
        # that's required for the model to match the reply to its call — so this
        # halves the reinforcement rather than removing it.)
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "unknown_tool", "available": registry.names()}),
            elapsed_s=0.0, is_error=True,
        )

    url = f"{spec.server_url}{spec.path}"
    start = time.monotonic()
    try:
        r = await client.post(url, json=args, timeout=timeout_s)
    except httpx.TimeoutException as e:
        elapsed = round(time.monotonic() - start, 2)
        log.warning("dispatch: %s timeout in %.2fs: %s", name, elapsed, e)
        tool_calls_total.labels(tool=name, outcome="timeout").inc()
        tool_call_seconds.labels(tool=name).observe(elapsed)
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "timeout", "detail": str(e)[:300]}),
            elapsed_s=elapsed, is_error=True,
        )
    except httpx.HTTPError as e:
        elapsed = round(time.monotonic() - start, 2)
        log.warning("dispatch: %s network error in %.2fs: %s", name, elapsed, e)
        tool_calls_total.labels(tool=name, outcome="error").inc()
        tool_call_seconds.labels(tool=name).observe(elapsed)
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "network_error", "detail": str(e)[:300]}),
            elapsed_s=elapsed, is_error=True,
        )

    elapsed = round(time.monotonic() - start, 2)
    tool_call_seconds.labels(tool=name).observe(elapsed)
    if r.status_code >= 400:
        log.warning("dispatch: %s -> %d in %.2fs: %s", name, r.status_code, elapsed, r.text[:200])
        body = {"error": f"http_{r.status_code}", "detail": r.text[:500]}
        tool_calls_total.labels(tool=name, outcome="error").inc()
        return ToolResult(
            name=name, call_id=call_id,
            content=_truncate(json.dumps(body), max_result_chars),
            elapsed_s=elapsed, is_error=True,
        )

    payload: Any = None
    try:
        payload = r.json()
        content = json.dumps(payload, ensure_ascii=False)
    except ValueError:
        content = r.text  # fall back to raw body if not JSON

    # `_truncate_payload` when we have parsed JSON, so the cut drops whole
    # results and stays parseable; the plain char cut only for a raw body.
    truncated = _truncate_payload(payload, content, max_result_chars)
    log.info("dispatch: %s ok in %.2fs (%d chars%s)",
             name, elapsed, len(truncated),
             ", truncated" if len(truncated) != len(content) else "")
    tool_calls_total.labels(tool=name, outcome="ok").inc()
    return ToolResult(
        name=name, call_id=call_id,
        content=truncated, elapsed_s=elapsed, is_error=False,
    )


def to_tool_message(result: ToolResult) -> dict[str, Any]:
    """Build the OpenAI-shaped `role=tool` message for the next ReAct round."""
    msg: dict[str, Any] = {
        "role": "tool",
        "name": result.name,
        "content": result.content,
    }
    if result.call_id:
        msg["tool_call_id"] = result.call_id
    return msg


__all__ = ["dispatch_one", "to_tool_message", "ToolResult", "audit_user_scoping"]
