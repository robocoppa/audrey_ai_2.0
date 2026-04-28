"""Synthesizer — merges worker drafts into the final assistant answer.

Output structure (Markdown):
    ## Approach
    one paragraph naming the strategy the panel converged on
    ## Answer
    the actual response to the user's request
    ## Caveats
    bullets covering disagreements between workers, gaps, things to verify

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
from audrey.pipeline.state import TaskType, WorkerDraft

log = logging.getLogger(__name__)


_SYNTH_SYSTEM = (
    "You are the panel synthesizer. You will receive the original user request "
    "plus several draft answers produced in parallel by different worker models. "
    "Some drafts may be tagged `[tool-grounded: N rounds]` — those workers ran "
    "tool calls (knowledge-base search, web search, etc.) before answering, so "
    "their factual claims are backed by retrieved evidence. When a tool-grounded "
    "draft and a tool-free draft disagree on a factual point, prefer the "
    "tool-grounded draft and note the disagreement in Caveats.\n\n"
    "Your job is to produce ONE coherent final answer using this exact structure:\n\n"
    "## Approach\n"
    "One short paragraph naming the strategy the panel converged on (or the "
    "strongest single approach if drafts disagreed).\n\n"
    "## Answer\n"
    "The actual response to the user's request — substantive, complete, and "
    "directly usable. Pull the strongest passages from the drafts; don't just "
    "average them. If the request was for code, include working code here.\n\n"
    "## Caveats\n"
    "Bullet list. Note where workers disagreed, what's uncertain, what the user "
    "should verify, or what was outside the panel's reach. If there are no real "
    "caveats, write `- none`.\n\n"
    "Rules:\n"
    "- Do NOT mention the worker model names in your answer.\n"
    "- Do NOT label drafts ('Draft 1 said...'). Speak in your own voice.\n"
    "- Preserve any code blocks verbatim from the strongest draft.\n"
)


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


def _build_synth_messages(drafts_block: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _SYNTH_SYSTEM},
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
    timeout_s: float,
    user_id: str | None = None,
) -> tuple[str, int, int]:
    """Run one synthesizer attempt. Returns (content, prompt_tokens, completion_tokens)."""
    messages = _build_synth_messages(drafts_block)
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
                timeout_s=timeout_s,
                user_id=user_id,
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
    """Streaming variant of `synthesize` (Phase 19).

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
    synth_messages = _build_synth_messages(drafts_block)

    candidates = [primary] if primary == fallback else [primary, fallback]

    accumulated = ""
    last_chunk: dict[str, Any] = {}
    chosen_model = ""
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
                            chosen_model = model
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
