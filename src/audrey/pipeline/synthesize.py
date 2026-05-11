"""Synthesizer — merges worker drafts into the final assistant answer.

The synthesizer writes the user's answer directly in its own voice. A
short `## Caveats` section is appended ONLY when drafts genuinely disagreed
on facts or a tool-grounded draft explicitly noted incomplete evidence —
it's not a required section.

The original conversation's system messages (datetime, memory recall,
OWUI template variables) are forwarded into the synth call so the
synthesizer reasons about "today" from the same temporal floor as the
workers, not from its training cutoff.

If the primary synthesizer fails, we retry once with the configured
`fallback_synth`. If both fail, we degrade to the longest non-empty draft
verbatim — better to ship something than 502 the request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from audrey.config import Config
from audrey.metrics import dispatch_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.deep_panel import _location_of
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.prompts import SYNTH_SYSTEM, prompt_from_config
from audrey.pipeline.state import TaskType, WorkerDraft

log = logging.getLogger(__name__)


# Prompt centralized in pipeline/prompts.py.
_SYNTH_SYSTEM = SYNTH_SYSTEM


def _format_drafts_for_synth(
    user_text: str, drafts: list[WorkerDraft], subtasks: list[str]
) -> str:
    parts: list[str] = []
    parts.append(f"USER REQUEST:\n{user_text.strip()}\n")
    if subtasks:
        parts.append("PLANNED SUB-QUESTIONS:")
        for i, s in enumerate(subtasks, 1):
            parts.append(f"  {i}. {s}")
        parts.append("")
    parts.append("DRAFTS:")
    shown = 0
    for i, d in enumerate(drafts, 1):
        content = (d.get("content") or "").strip()
        err = d.get("error") or ""
        if not content:
            parts.append(f"\n--- draft {i} (model={d.get('model')}) — empty ({err or 'no content'})")
            continue
        shown += 1
        tool_rounds = int(d.get("tool_rounds", 0) or 0)
        tool_tag = f" [tool-grounded: {tool_rounds} rounds]" if tool_rounds > 0 else ""
        parts.append(f"\n--- draft {i} (model={d.get('model')}, "
                     f"elapsed={d.get('elapsed_s', 0)}s){tool_tag} ---\n{content}")
    if shown == 0:
        parts.append("\n[no drafts produced usable output]")
    return "\n".join(parts)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def pick_synthesizer(cfg: Config, *, pool_key: str, task: TaskType) -> tuple[str, str]:
    """Return (primary_synth, fallback_synth). Raises KeyError if missing."""
    pool = cfg.raw.get(pool_key, {}).get(task, {})
    primary = pool.get("synthesizer")
    fallback = pool.get("fallback_synth")
    if not primary:
        raise KeyError(f"No synthesizer configured for {pool_key}/{task}")
    if not fallback:
        fallback = primary
    return primary, fallback


def _build_synth_messages(
    prior_messages: list[dict[str, Any]],
    drafts_block: str,
    *,
    cfg: Config | None = None,
) -> list[dict[str, Any]]:
    """Forward original system messages, then append synth-system + user.

    Workers see all the system messages from the request (datetime from
    `node_datetime`, memory recall hits, OWUI template variables, any
    user-supplied system prompts). The same system messages are forwarded
    into the synth call so the synthesizer reasons with the same context
    — most importantly the datetime (otherwise synth hedges about "today"
    from training cutoff).

    Only `role=system` messages are forwarded. The user/assistant turns
    from the original conversation are already represented in
    `drafts_block` so we don't replay them.

    When `cfg` is supplied, `agentic.prompts.synthesizer` overrides the
    default synth system prompt; missing/empty falls back to the default.
    """
    synth_system = prompt_from_config(cfg, "synthesizer", _SYNTH_SYSTEM)
    system_msgs = [m for m in prior_messages if m.get("role") == "system"]
    return [
        *system_msgs,
        {"role": "system", "content": synth_system},
        {"role": "user", "content": (
            f"Original user request and {drafts_block.count('--- draft ')} drafts follow."
            f" Produce the final answer now.\n\n{drafts_block}"
        )},
    ]


async def _try_synth(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    model: str,
    location: str,
    user_text: str,
    drafts_block: str,
    prior_messages: list[dict[str, Any]],
    timeout_s: float,
    user_id: str | None = None,
    cfg: Config | None = None,
) -> tuple[str, int, int]:
    """Run one synthesizer attempt. Returns (content, prompt_tokens, completion_tokens)."""
    messages = _build_synth_messages(prior_messages, drafts_block, cfg=cfg)
    async with gate.acquire(model, location=location, user_id=user_id):
        resp = await ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.2},
            timeout_s=timeout_s,
        )
    health.record_success(model)
    msg = resp.get("message", {}) or {}
    return (
        msg.get("content", "") or "",
        int(resp.get("prompt_eval_count", 0) or 0),
        int(resp.get("eval_count", 0) or 0),
    )


async def synthesize(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    pool_key: str,
    task: TaskType,
    messages: list[dict[str, Any]],
    drafts: list[WorkerDraft],
    subtasks: list[str],
    timeout_s: float,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Return a dict to merge into PipelineState.

    Keys: content, synthesizer_model, synth_error, prompt_eval_count, eval_count.
    """
    if not drafts or all(not (d.get("content") or "").strip() for d in drafts):
        return {
            "content": "[deep panel produced no usable drafts — all workers failed]",
            "synthesizer_model": "none",
            "synth_error": "no_drafts",
            "prompt_eval_count": 0,
            "eval_count": 0,
        }

    primary, fallback = pick_synthesizer(cfg, pool_key=pool_key, task=task)
    user_text = _last_user_text(messages)
    drafts_block = _format_drafts_for_synth(user_text, drafts, subtasks)

    candidates = [primary] if primary == fallback else [primary, fallback]
    for attempt, model in enumerate(candidates, start=1):
        if not health.is_healthy(model):
            log.warning("synth: %s unhealthy, skipping (attempt %d)", model, attempt)
            continue
        loc = _location_of(model, registry)
        dispatch_total.labels(
            model=model,
            task_type=str(task),
            path="synth_primary" if attempt == 1 else "synth_fallback",
        ).inc()
        start = time.monotonic()
        try:
            content, ptok, etok = await _try_synth(
                ollama, health, gate,
                model=model, location=loc,
                user_text=user_text, drafts_block=drafts_block,
                prior_messages=messages,
                timeout_s=timeout_s,
                user_id=user_id,
                cfg=cfg,
            )
            log.info("synth: %s ok in %.2fs (attempt %d)", model, time.monotonic() - start, attempt)
            if content.strip():
                return {
                    "content": content,
                    "synthesizer_model": model,
                    "synth_error": "",
                    "prompt_eval_count": ptok,
                    "eval_count": etok,
                }
            log.warning("synth: %s returned empty content (attempt %d)", model, attempt)
        except OllamaError as e:
            health.record_failure(model, str(e))
            log.warning("synth: %s failed in %.2fs (attempt %d): %s",
                        model, time.monotonic() - start, attempt, e)

    # Both synthesizers failed. Degrade gracefully: return the longest draft.
    best = max(drafts, key=lambda d: len(d.get("content") or ""))
    log.warning("synth: all synthesizers failed; returning longest draft from %s", best.get("model"))
    return {
        "content": best.get("content", "") or "[no content]",
        "synthesizer_model": "fallback:longest_draft",
        "synth_error": "all_synth_failed",
        "prompt_eval_count": int(best.get("prompt_eval_count", 0)),
        "eval_count": int(best.get("eval_count", 0)),
    }


async def synthesize_stream(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    pool_key: str,
    task: TaskType,
    messages: list[dict[str, Any]],
    drafts: list[WorkerDraft],
    subtasks: list[str],
    timeout_s: float,
    user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming variant of `synthesize`.

    Yields a sequence of events. The route handler interleaves these with
    progress banners and SSE frames; the non-streaming graph keeps using
    `synthesize` unchanged.

    Event shapes:
      {"type": "delta", "text": "..."}
          A content chunk from the active synthesizer.
      {"type": "first_token", "model": str}
          Emitted exactly once, immediately before the first non-empty
          delta. The route uses this to close the Synthesizing banner
          and emit the separator before answer text starts streaming.
      {"type": "fallback_attempt", "model": str, "error": str}
          Primary synth failed before any tokens; fallback is starting.
          Purely informational — the route logs it but doesn't surface
          to the user.
      {"type": "done", "content": str, "synthesizer_model": str,
       "synth_error": str, "prompt_eval_count": int, "eval_count": int}
          Final event. Always emitted exactly once at the end. `content`
          is the concatenation of all deltas (or the degraded fallback
          text). `synth_error` is "" on success, or one of:
            "no_drafts"            — panel produced nothing usable
            "all_synth_failed"     — both synths errored; longest draft
                                     emitted as a single delta first
            "stream_truncated"     — primary synth started streaming then
                                     errored mid-flight; partial content
                                     was emitted, error appended
    """
    # Edge case: no drafts at all → emit a single delta + done, mirroring
    # the non-streaming `synthesize` so the route's separator handling is
    # uniform.
    if not drafts or all(not (d.get("content") or "").strip() for d in drafts):
        msg = "[deep panel produced no usable drafts — all workers failed]"
        yield {"type": "first_token", "model": "none"}
        yield {"type": "delta", "text": msg}
        yield {
            "type": "done",
            "content": msg,
            "synthesizer_model": "none",
            "synth_error": "no_drafts",
            "prompt_eval_count": 0,
            "eval_count": 0,
        }
        return

    primary, fallback = pick_synthesizer(cfg, pool_key=pool_key, task=task)
    user_text = _last_user_text(messages)
    drafts_block = _format_drafts_for_synth(user_text, drafts, subtasks)
    synth_messages = _build_synth_messages(messages, drafts_block, cfg=cfg)

    candidates = [primary] if primary == fallback else [primary, fallback]

    accumulated = ""
    last_chunk: dict[str, Any] = {}
    any_tokens = False

    for attempt, model in enumerate(candidates, start=1):
        if not health.is_healthy(model):
            log.warning("synth: %s unhealthy, skipping (attempt %d)", model, attempt)
            continue
        loc = _location_of(model, registry)
        dispatch_total.labels(
            model=model,
            task_type=str(task),
            path="synth_primary" if attempt == 1 else "synth_fallback",
        ).inc()
        start = time.monotonic()
        attempt_started_tokens = False
        try:
            async with gate.acquire(model, location=loc, user_id=user_id):
                async for chunk in ollama.chat_stream(
                    model=model,
                    messages=synth_messages,
                    options={"temperature": 0.2},
                    timeout_s=timeout_s,
                ):
                    msg_part = chunk.get("message", {}) or {}
                    text = msg_part.get("content", "") or ""
                    if text:
                        if not attempt_started_tokens:
                            attempt_started_tokens = True
                            any_tokens = True
                            yield {"type": "first_token", "model": model}
                        accumulated += text
                        yield {"type": "delta", "text": text}
                    if chunk.get("done"):
                        last_chunk = chunk
                        break
            health.record_success(model)
            log.info("synth: %s ok in %.2fs (attempt %d, streamed=%s)",
                     model, time.monotonic() - start, attempt, attempt_started_tokens)
            if accumulated.strip():
                yield {
                    "type": "done",
                    "content": accumulated,
                    "synthesizer_model": model,
                    "synth_error": "",
                    "prompt_eval_count": int(last_chunk.get("prompt_eval_count", 0) or 0),
                    "eval_count": int(last_chunk.get("eval_count", 0) or 0),
                }
                return
            log.warning("synth: %s returned empty content (attempt %d)", model, attempt)
        except OllamaError as e:
            health.record_failure(model, str(e))
            log.warning("synth: %s failed in %.2fs (attempt %d, streamed=%s): %s",
                        model, time.monotonic() - start, attempt, attempt_started_tokens, e)
            if attempt_started_tokens:
                # Mid-stream failure: partial content already on the wire,
                # can't fall back. Surface the truncation to the user.
                trailing = f"\n\n[synth error mid-stream: {e}]"
                accumulated += trailing
                yield {"type": "delta", "text": trailing}
                yield {
                    "type": "done",
                    "content": accumulated,
                    "synthesizer_model": model,
                    "synth_error": "stream_truncated",
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                }
                return
            # No tokens emitted — safe to try the fallback.
            if attempt < len(candidates):
                yield {"type": "fallback_attempt",
                       "model": candidates[attempt], "error": str(e)}

    # Both candidates failed without producing any output. Degrade to the
    # longest non-empty draft, emitted as a single delta so the route's
    # downstream rendering is uniform.
    best = max(drafts, key=lambda d: len(d.get("content") or ""))
    fallback_text = best.get("content", "") or "[no content]"
    log.warning("synth: all synthesizers failed; returning longest draft from %s",
                best.get("model"))
    if not any_tokens:
        yield {"type": "first_token", "model": "fallback:longest_draft"}
    yield {"type": "delta", "text": fallback_text}
    yield {
        "type": "done",
        "content": fallback_text,
        "synthesizer_model": "fallback:longest_draft",
        "synth_error": "all_synth_failed",
        "prompt_eval_count": int(best.get("prompt_eval_count", 0)),
        "eval_count": int(best.get("eval_count", 0)),
    }


__all__ = ["synthesize", "synthesize_stream", "pick_synthesizer"]
