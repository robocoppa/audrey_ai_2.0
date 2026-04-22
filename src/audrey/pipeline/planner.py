"""Planner — optional sub-task decomposition for deep panel.

For most prompts the panel just answers the user verbatim. But for longer or
multi-part prompts, splitting into 2–3 sub-questions before dispatch gives
each worker a tighter focus and lets synthesis stitch a more complete answer.

Gated on `agentic.planning.min_prompt_tokens` (default 40). Below that it's
not worth the router round-trip — the planner returns `[]` and the panel runs
the full prompt directly.

Failure modes (timeout, parse error, only one subtask returned) all collapse
to `[]` — never raise. The panel always has a fallback path.
"""

from __future__ import annotations

import json
import logging
import re

from audrey.models.ollama import OllamaClient, OllamaError

log = logging.getLogger(__name__)


_PLANNER_SYSTEM = (
    "You decompose a user request into 2 or 3 focused sub-questions that, if "
    "answered separately, would together cover the original request. Output a "
    "JSON object with exactly this shape:\n"
    '  {"subtasks": ["...", "...", "..."]}\n'
    "Rules:\n"
    "- 2 to 3 entries, each a complete question or instruction (≤ 200 chars).\n"
    "- Sub-questions must be independent — no 'first do X then Y' chaining.\n"
    "- If the request is already atomic (one clear ask), return {\"subtasks\": []}.\n"
    "Output ONLY the JSON. No prose, no markdown."
)


async def plan(
    ollama: OllamaClient,
    *,
    planner_model: str,
    user_text: str,
    timeout_s: float,
    max_subtasks: int,
) -> list[str]:
    """Return a list of sub-questions, or `[]` to skip planning.

    Never raises — all failures degrade to `[]` so the deep panel can run the
    original prompt unchanged.
    """
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": user_text[:4000]},
    ]
    try:
        resp = await ollama.chat(
            model=planner_model,
            messages=messages,
            options={"temperature": 0.0},
            timeout_s=timeout_s,
        )
    except OllamaError as e:
        log.warning("planner: ollama error, skipping decomposition: %s", e)
        return []

    body = (resp.get("message", {}) or {}).get("content", "") or ""
    subs = _parse_planner_output(body)
    if len(subs) < 2:
        return []
    return subs[:max_subtasks]


def _parse_planner_output(raw: str) -> list[str]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = obj.get("subtasks", [])
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        s = re.sub(r"\s+", " ", item).strip()
        if s:
            out.append(s[:400])
    return out


__all__ = ["plan"]
