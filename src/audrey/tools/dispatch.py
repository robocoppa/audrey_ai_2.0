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
from audrey.tools.discovery import ToolRegistry, ToolUserScope
from audrey.user_data_visibility import remote_personal_reads_blocked

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    name: str                  # tool name (or "?" if unknown)
    call_id: str | None        # tool_call_id from the model, if provided
    content: str               # JSON-string result body (possibly truncated/error)
    elapsed_s: float
    is_error: bool


def audit_user_scoping(registry: ToolRegistry) -> list[str]:
    """Report direct-constructed records whose schema and scope disagree.

    Production discovery validates this before registration. This audit remains
    as a compatibility check for registries assembled by tests or extensions.
    """
    invalid: list[str] = []
    for spec in registry.policy_records():
        properties = spec.parameters.get("properties") or {}
        has_user = "user" in properties
        has_tags = "tags" in properties
        if spec.user_scope is ToolUserScope.ARGUMENT and not has_user:
            invalid.append(spec.name)
        elif spec.user_scope is ToolUserScope.TAGS and not has_tags:
            invalid.append(spec.name)
        elif spec.user_scope is ToolUserScope.NONE and has_user:
            invalid.append(spec.name)
    if invalid:
        log.error(
            "tools: capability policy and request schema disagree for %s",
            sorted(invalid),
        )
    return sorted(invalid)


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

    spec = registry.get(name)
    if spec is None:
        log.warning("dispatch: unknown tool %r (registered: %s)", name, registry.names())
        tool_calls_total.labels(tool=name, outcome="error").inc()
        return ToolResult(
            name=name,
            call_id=call_id,
            content=json.dumps({"error": "unknown_tool", "available": registry.names()}),
            elapsed_s=0.0,
            is_error=True,
        )

    # Bind the authenticated identity exactly as the declared policy says.
    # The model never chooses another user for a scoped capability.
    if user_id and spec.user_scope is ToolUserScope.TAGS:
        args["tags"] = _force_user_tag(str(args.get("tags") or ""), user_id)
    elif user_id and spec.user_scope is ToolUserScope.ARGUMENT:
        args["user"] = user_id

    if (
        user_id
        and spec.purge_gated
        and remote_personal_reads_blocked(user_id)
    ):
        tool_calls_total.labels(tool=name, outcome="error").inc()
        return ToolResult(
            name=name,
            call_id=call_id,
            content=json.dumps({
                "error": "personal_data_purge_in_progress",
                "detail": (
                    "Stored memory and chat history are temporarily unavailable "
                    "while their purge cutoff is being installed."
                ),
            }),
            elapsed_s=0.0,
            is_error=True,
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
