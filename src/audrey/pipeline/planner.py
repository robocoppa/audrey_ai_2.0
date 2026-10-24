"""Planner — optional sub-task decomposition for deep panel.

For most prompts the panel just answers the user verbatim. But for longer or
multi-part prompts, splitting into 2–3 sub-questions before dispatch gives
each worker a tighter focus and lets synthesis stitch a more complete answer.

Gated on `agentic.planning.min_prompt_tokens` (default 40). Below that it's
not worth the router round-trip — the planner returns `[]` and the panel runs
the full prompt directly.

Provider unavailability, parse errors and a single subtask collapse to `[]`.
Protected model, digest and transport-policy refusals still block the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import nullcontext
from typing import Any

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.prompts import PLANNER_SYSTEM, prompt_from_config
from audrey.processing_policy import ProcessingPolicyError

log = logging.getLogger(__name__)


# Prompt centralized in pipeline/prompts.py.
_PLANNER_SYSTEM = PLANNER_SYSTEM


async def plan(
    ollama: OllamaClient,
    *,
    planner_model: str,
    user_text: str,
    timeout_s: float,
    max_subtasks: int,
    cfg: Any = None,
    gate: FairLocalGate | None = None,
    user_id: str | None = None,
) -> list[str]:
    """Return a list of sub-questions, or `[]` to skip planning.

    Provider unavailability degrades to `[]` so the same panel can run the
    original prompt unchanged. Protected trust-policy violations propagate.
    Production callers supply the local GPU gate and authenticated user bucket.
    Waiting and generation share the optional planning deadline; an occupied GPU
    can skip decomposition instead of evicting another user's local generation.

    `cfg` is optional. When supplied, `agentic.prompts.planner` overrides
    the default system prompt; missing/empty falls back to the default.
    """
    system_prompt = prompt_from_config(cfg, "planner", _PLANNER_SYSTEM)
    if len(user_text) > 4000:
        log.debug(
            "planner: user_text truncated from %d to 4000 chars for planning",
            len(user_text),
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text[:4000]},
    ]
    try:
        async with asyncio.timeout(timeout_s):
            async with (gate.acquire(planner_model, location="local", user_id=user_id)
                        if gate is not None else nullcontext()):
                resp = await ollama.chat(
                    model=planner_model,
                    messages=messages,
                    options={"temperature": 0.0},
                    timeout_s=timeout_s,
                )
    except ProcessingPolicyError as exc:
        # The protected client redacts provider timeouts/unavailability into
        # this stable code. Planning is optional: the same verified local
        # panel can answer the original prompt without decomposition. Other
        # policy failures must still block; no alternate provider is tried.
        if exc.code != "private_model_unavailable":
            raise
        log.warning("planner: local provider unavailable, skipping decomposition")
        return []
    except TimeoutError:
        log.warning("planner: optional planning deadline reached, skipping decomposition")
        return []
    except OllamaError as e:
        log.warning("planner: ollama error, skipping decomposition: %s", e)
        return []

    body = (resp.get("message", {}) or {}).get("content", "") or ""
    subs = _parse_planner_output(body)
    if len(subs) < 2:
        if len(subs) == 1:
            log.debug("planner: only 1 subtask returned, treating as no decomposition")
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
