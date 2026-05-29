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
    tools: list[dict[str, Any]] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Non-streaming passthrough: gate-guarded `ollama.chat` forward.

    Returns the raw Ollama response dict. The caller is responsible for
    reshaping it into the OpenAI chat-completion contract.

    `tools` is forwarded verbatim. Agent clients (Hermes, OpenClaw) that
    advertise their own tools to the model rely on this forwarding —
    without it the model sees no schema, can't issue structured
    `tool_calls`, and falls back to emitting tool syntax as plain text.
    """
    dispatch_total.labels(
        model=concrete, task_type="passthrough", path="passthrough",
    ).inc()
    t0 = time.perf_counter()
    async with gate.acquire(concrete, location=location, user_id=user_id):
        resp = await ollama.chat(
            model=concrete, messages=messages, options=options,
            tools=tools, timeout_s=timeout_s,
        )
    msg = resp.get("message") or {}
    log.info(
        "passthrough.chat model=%s user=%s tools=%d elapsed=%.2fs "
        "content_len=%d tool_calls=%d eval_count=%d done_reason=%s",
        concrete, _safe_user(user_id), len(tools or []),
        time.perf_counter() - t0,
        len(str(msg.get("content") or "")),
        len(list(msg.get("tool_calls") or [])),
        int(resp.get("eval_count", 0) or 0),
        resp.get("done_reason", "?"),
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
    tools: list[dict[str, Any]] | None = None,
    timeout_s: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming passthrough: yield raw Ollama chunks, gate held across all.

    The gate stays held for the entire stream — releasing early would
    let another user's request enter Ollama while ours is still
    generating, and they'd contend at the model level anyway. Holding
    across the stream is the only granularity that gives real
    fairness on local generation.

    `tools` is forwarded verbatim. Ollama may populate
    `message.tool_calls` on the final chunk; the caller passes that
    through to the OpenAI SSE shape unchanged.
    """
    dispatch_total.labels(
        model=concrete, task_type="passthrough", path="passthrough_stream",
    ).inc()
    t0 = time.perf_counter()
    total_content_len = 0
    total_tool_calls = 0
    last_eval_count = 0
    last_done_reason = "?"
    chunks_received = 0
    # Dump request shape before sending so we can diagnose empty-response
    # turns. Counts message roles by type; truncates tool/user content
    # heads so the log isn't flooded.
    role_counts: dict[str, int] = {}
    for m in messages:
        r = str(m.get("role") or "?")
        role_counts[r] = role_counts.get(r, 0) + 1
    last_msg = messages[-1] if messages else {}
    last_role = str(last_msg.get("role") or "?")
    last_content_head = str(last_msg.get("content") or "")[:120].replace("\n", "\\n")
    log.info(
        "passthrough.stream.req model=%s user=%s roles=%s last_role=%s last_head=%r",
        concrete, _safe_user(user_id), role_counts, last_role, last_content_head,
    )
    async with gate.acquire(concrete, location=location, user_id=user_id):
        async for chunk in ollama.chat_stream(
            model=concrete, messages=messages, options=options,
            tools=tools, timeout_s=timeout_s,
        ):
            chunks_received += 1
            cmsg = chunk.get("message") or {}
            total_content_len += len(str(cmsg.get("content") or ""))
            total_tool_calls += len(list(cmsg.get("tool_calls") or []))
            if chunk.get("done"):
                last_eval_count = int(chunk.get("eval_count", 0) or 0)
                last_done_reason = chunk.get("done_reason", "?")
            yield chunk
    log.info(
        "passthrough.stream model=%s user=%s tools=%d elapsed=%.2fs "
        "chunks=%d content_len=%d tool_calls=%d eval_count=%d done_reason=%s",
        concrete, _safe_user(user_id), len(tools or []),
        time.perf_counter() - t0,
        chunks_received, total_content_len, total_tool_calls,
        last_eval_count, last_done_reason,
    )


def _safe_user(user_id: str) -> str:
    """Trim email for log lines — same shape as routes/inflight._safe_bucket."""
    if "@" in user_id:
        local, _, _ = user_id.partition("@")
        return f"{local[:8]}…"
    return user_id[:16]


__all__ = ["passthrough_chat", "passthrough_stream"]
