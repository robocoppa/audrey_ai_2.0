"""Tool dispatcher — execute one tool_call against the right server.

Given an Ollama tool_call (`{"function": {"name": "...", "arguments": {...}}}`)
and a registry, POST the arguments to the originating server and return the
response body as a JSON-string suitable for inclusion in a `role=tool` message.

Long results are truncated at `agentic.react.max_tool_result_chars` (default
2000) — model context burns fast otherwise. Truncated payloads end with
`…[truncated]` so the model knows the cut happened.

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
_USER_SCOPED_TOOLS: frozenset[str] = frozenset({
    "kb_search",
    "kb_image_search",
    "memory_search",
    "memory_store",
})


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - len("\n…[truncated]")] + "\n…[truncated]"


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
    """Execute one tool_call. Always returns a ToolResult — never raises."""
    fn = (tool_call.get("function") or {})
    name = str(fn.get("name") or "?")
    call_id = tool_call.get("id")
    args = fn.get("arguments")

    # Ollama sometimes passes arguments as a JSON-encoded string instead of a dict.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            log.warning("dispatch: %s arguments not JSON: %r", name, args[:200])
            return ToolResult(
                name=name, call_id=call_id,
                content=json.dumps({"error": "arguments_not_json", "raw": args[:500]}),
                elapsed_s=0.0, is_error=True,
            )
    if args is None:
        args = {}
    if not isinstance(args, dict):
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
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "unknown_tool", "tool": name, "available": registry.names()}),
            elapsed_s=0.0, is_error=True,
        )

    url = f"{spec.server_url}{spec.path}"
    start = time.monotonic()
    try:
        r = await client.post(url, json=args, timeout=timeout_s)
    except httpx.HTTPError as e:
        elapsed = round(time.monotonic() - start, 2)
        log.warning("dispatch: %s network error in %.2fs: %s", name, elapsed, e)
        return ToolResult(
            name=name, call_id=call_id,
            content=json.dumps({"error": "network_error", "detail": str(e)[:300]}),
            elapsed_s=elapsed, is_error=True,
        )

    elapsed = round(time.monotonic() - start, 2)
    if r.status_code >= 400:
        log.warning("dispatch: %s -> %d in %.2fs: %s", name, r.status_code, elapsed, r.text[:200])
        body = {"error": f"http_{r.status_code}", "detail": r.text[:500]}
        return ToolResult(
            name=name, call_id=call_id,
            content=_truncate(json.dumps(body), max_result_chars),
            elapsed_s=elapsed, is_error=True,
        )

    try:
        payload = r.json()
        content = json.dumps(payload, ensure_ascii=False)
    except ValueError:
        content = r.text  # fall back to raw body if not JSON

    truncated = _truncate(content, max_result_chars)
    log.info("dispatch: %s ok in %.2fs (%d chars%s)",
             name, elapsed, len(truncated),
             ", truncated" if len(truncated) != len(content) else "")
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


__all__ = ["dispatch_one", "to_tool_message", "ToolResult"]
