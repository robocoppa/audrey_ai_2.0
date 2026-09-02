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
import time
from typing import Any

import httpx

from audrey.pipeline.messages import last_user_text
from audrey.pipeline.prompts import MEMORY_STORE_HINT, prompt_from_config
from audrey.tools.discovery import ToolRegistry
from audrey.tools.dispatch import dispatch_one

log = logging.getLogger(__name__)

MEMORY_SEARCH_TOOL = "memory_search"
MEMORY_STORE_TOOL = "memory_store"
MAX_QUERY_CHARS = 500          # long prompts dilute the embedding's signal; tighter queries match better
DEFAULT_TOP_K = 3              # three hits is usually plenty for context

# Hint text lives in pipeline/prompts.py. The `{user_id}` placeholder is
# replaced at call time below.
_MEMORY_STORE_HINT = MEMORY_STORE_HINT


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
    http: httpx.AsyncClient,
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
    query = last_user_text(messages).strip()
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
    started = time.monotonic()
    result = await dispatch_one(
        http, registry, call,
        max_result_chars=10_000,   # keep the complete JSON body for parsing
        timeout_s=timeout_s,
    )
    elapsed = time.monotonic() - started
    if result.is_error:
        # WARNING, with the cost attached. Recall is best-effort, but a skip
        # still spends the caller's whole budget on the hot path of a request,
        # and at INFO with no timing it read as free. `%.2fs` against
        # `agentic.memory.timeout_s` is what separates "the embedder stalled"
        # from "custom-tools is down".
        log.warning("memory: recall skipped in %.2fs (search error: %s)",
                    elapsed, result.content[:200])
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
    cfg: Any = None,
) -> dict[str, Any] | None:
    """Wrap hits (and optionally the memory_store usage hint) into a system message.

    Returns None when there is nothing to inject — no hits and no store hint.
    When `cfg` is supplied, `agentic.prompts.memory_store_hint` overrides
    the default hint template; missing/empty falls back to the default.
    """
    parts: list[str] = []
    if hits:
        parts.append(_format_memory_hint(hits))
    if include_store_hint and user_id:
        hint_template = prompt_from_config(cfg, "memory_store_hint", _MEMORY_STORE_HINT)
        parts.append(hint_template.replace("{user_id}", user_id))
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}


__all__ = [
    "recall_for_request",
    "memory_system_message",
    "MEMORY_SEARCH_TOOL",
    "MEMORY_STORE_TOOL",
]
