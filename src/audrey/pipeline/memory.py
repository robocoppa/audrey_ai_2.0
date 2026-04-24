"""Per-user memory recall — runs once per request, before classify.

The orchestrator keyword-searches the custom-tools `/memory_search` endpoint
scoped to `state["user_id"]`, then injects any hits into the prompt as a
system message. Writes happen later via the model's own `memory_store` tool
call — this module only reads.

Skipped (no-op) when:
  • `user_id` is empty (no logged-in user — happens with direct curl)
  • `memory_search` isn't in the tool registry (custom-tools unreachable)
  • the last user turn is empty
  • the search returns zero hits

Errors from the search never raise: a best-effort feature should not break
the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from audrey.tools.discovery import ToolRegistry
from audrey.tools.dispatch import dispatch_one

log = logging.getLogger(__name__)

MEMORY_SEARCH_TOOL = "memory_search"
MEMORY_STORE_TOOL = "memory_store"
MAX_QUERY_CHARS = 500          # long prompts degrade SQL LIKE matching, don't help it
DEFAULT_TOP_K = 3              # three hits is usually plenty for context

_MEMORY_STORE_HINT = (
    "If the user states a durable fact about themselves (preferences, goals, "
    "projects, constraints) or explicitly asks you to remember something, "
    "call the `memory_store` tool with: a short descriptive `key`, the fact "
    "as `value`, and `tags=\"user:{user_id}\"` (use exactly that user tag). "
    "Do this silently — do not narrate the tool call in your reply."
)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _format_memory_hint(hits: list[dict[str, Any]]) -> str:
    """Build the system-message body shown to the model."""
    lines = ["[Relevant memories from previous conversations with this user:]"]
    for i, h in enumerate(hits, 1):
        key = h.get("key", "?")
        value = (h.get("value") or "").strip()
        if len(value) > 400:
            value = value[:400].rstrip() + "…"
        lines.append(f"{i}. ({key}) {value}")
    lines.append(
        "Use these facts if they're relevant to the user's question. "
        "Ignore irrelevant ones without mentioning them."
    )
    return "\n".join(lines)


async def recall_for_request(
    registry: ToolRegistry | None,
    *,
    user_id: str,
    messages: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Return recalled memory entries (possibly empty). Never raises."""
    if not user_id:
        return []
    if registry is None or MEMORY_SEARCH_TOOL not in registry.by_name:
        return []
    query = _last_user_text(messages).strip()
    if not query:
        return []
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]

    # Reuse the ReAct dispatcher so errors come back as data, not exceptions.
    call = {
        "function": {
            "name": MEMORY_SEARCH_TOOL,
            "arguments": {"user": user_id, "query": query, "top_k": top_k},
        }
    }
    async with httpx.AsyncClient() as http:
        result = await dispatch_one(
            http, registry, call,
            max_result_chars=10_000,   # don't truncate the JSON body here
            timeout_s=timeout_s,
        )
    if result.is_error:
        log.info("memory: recall skipped (search error: %s)", result.content[:200])
        return []
    try:
        body = json.loads(result.content)
    except json.JSONDecodeError:
        log.warning("memory: search returned non-JSON body")
        return []
    hits = body.get("results") or []
    if not isinstance(hits, list):
        return []
    return hits


def memory_system_message(
    hits: list[dict[str, Any]],
    *,
    user_id: str = "",
    include_store_hint: bool = False,
) -> dict[str, Any] | None:
    """Wrap hits (and optionally the memory_store usage hint) into a system message.

    Returns None when there is nothing to inject — no hits and no store hint.
    """
    parts: list[str] = []
    if hits:
        parts.append(_format_memory_hint(hits))
    if include_store_hint and user_id:
        parts.append(_MEMORY_STORE_HINT.replace("{user_id}", user_id))
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


__all__ = [
    "recall_for_request",
    "memory_system_message",
    "MEMORY_SEARCH_TOOL",
    "MEMORY_STORE_TOOL",
]
