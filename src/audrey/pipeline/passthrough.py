"""Passthrough virtual model — direct forwarding to Ollama.

The pipeline modes (`audrey_fast`, `audrey_deep`, `audrey_auto`, …)
all run the request through a classifier, complexity gate, and either
fast path or planner/panel/synth. The passthrough virtual model skips
all of that: it forwards the chat as-is to a named concrete model.

Why it exists: clients that today talk directly to Ollama (e.g. an
OpenClaw daemon on the LAN) bypass Audrey's fair-scheduling layers
entirely, so they compete with Audrey users inside Ollama's FIFO
queue with no per-user fairness. Routing them through this module
brings them under `FairLocalGate` (round-robin GPU slot) and
`UserInflightRegistry` (per-user request cap) without paying for the
orchestration pipeline they don't need.

The route layer (`routes/openai.py`) handles model-string parsing
(`audrey_passthrough/<concrete>`), the in-flight wrap, validation,
and the OpenAI-shaped response/SSE framing. This module is the thin
seam that holds the GPU gate around the actual Ollama call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from audrey.metrics import dispatch_total
from audrey.models.ollama import OllamaClient
from audrey.pipeline.fair_gate import FairLocalGate

log = logging.getLogger(__name__)


async def passthrough_chat(
    ollama: OllamaClient,
    gate: FairLocalGate,
    *,
    concrete: str,
    location: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    user_id: str,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Non-streaming passthrough: gate-guarded `ollama.chat` forward.

    Returns the raw Ollama response dict. The caller is responsible for
    reshaping it into the OpenAI chat-completion contract.
    """
    dispatch_total.labels(
        model=concrete, task_type="passthrough", path="passthrough",
    ).inc()
    t0 = time.perf_counter()
    async with gate.acquire(concrete, location=location, user_id=user_id):
        resp = await ollama.chat(
            model=concrete, messages=messages, options=options, timeout_s=timeout_s,
        )
    log.info(
        "passthrough.chat model=%s user=%s elapsed=%.2fs",
        concrete, _safe_user(user_id), time.perf_counter() - t0,
    )
    return resp


async def passthrough_stream(
    ollama: OllamaClient,
    gate: FairLocalGate,
    *,
    concrete: str,
    location: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    user_id: str,
    timeout_s: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming passthrough: yield raw Ollama chunks, gate held across all.

    The gate stays held for the entire stream — releasing early would
    let another user's request enter Ollama while ours is still
    generating, and they'd contend at the model level anyway. Holding
    across the stream is the only granularity that gives real
    fairness on local generation.
    """
    dispatch_total.labels(
        model=concrete, task_type="passthrough", path="passthrough_stream",
    ).inc()
    t0 = time.perf_counter()
    async with gate.acquire(concrete, location=location, user_id=user_id):
        async for chunk in ollama.chat_stream(
            model=concrete, messages=messages, options=options, timeout_s=timeout_s,
        ):
            yield chunk
    log.info(
        "passthrough.stream model=%s user=%s elapsed=%.2fs",
        concrete, _safe_user(user_id), time.perf_counter() - t0,
    )


def _safe_user(user_id: str) -> str:
    """Trim email for log lines — same shape as routes/inflight._safe_bucket."""
    if "@" in user_id:
        local, _, _ = user_id.partition("@")
        return f"{local[:8]}…"
    return user_id[:16]


__all__ = ["passthrough_chat", "passthrough_stream"]
