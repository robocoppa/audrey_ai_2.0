"""Fast path — single-model generation, optionally with ReAct tool use.

When the chosen model is in `fast_path.tool_capable_models` *and* the tool
registry is non-empty, we run a ReAct loop (`pipeline/react.py`) that lets
the model call tools before answering. For a non-streaming request, the
no-tools alternative is a one-shot `ollama.chat`. Plain streaming instead
uses `stream_fast_path` below so model selection, bounded fallback, thinking
policy, gating, health, metrics, and terminal outcome have one owner; the
route remains responsible only for client framing and banners.

Local calls go through `FairLocalGate`. The non-tools branch holds the
gate around the single `ollama.chat`. The tools branch passes the gate
down into `run_react`, which acquires per-chat so the gate is released
during tool dispatch. Cloud calls bypass the gate entirely
(`gate.acquire` is a no-op when `location != "local"`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from audrey.metrics import dispatch_total, pipeline_seconds, pipeline_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry, ModelSpec, TaskType
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.react import ReactResult, run_react
from audrey.pipeline.run_observations import RunEventToolObserver
from audrey.pipeline.streaming import StreamOutcome, StreamTerminal
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)

async def _think(ollama: Any, model: str, no_thinking: bool) -> bool | None:
    """`False` when this model can be told not to think, else `None`.

    ## Why the fast path turns thinking off

    This is the **user-is-waiting** role, and the flag was measured with tools
    in the request — which is the part that matters, because the fast path's
    job is tool calling and an earlier prose-only probe would have justified
    this for the wrong reason. `qwen3.6:35b`, 2026-08-07, three samples:

        omitted   933c thinking   277 eval tok   right tool 2/3
        true     1090c thinking   333 eval tok   right tool 3/3
        false        0c            45 eval tok   right tool 3/3

    **Tool selection did not degrade — it matched `think=true` and beat the
    current default.** `omitted` is what every non-vision path does today, and
    it was the only state that ever reached for the wrong tool: with reasoning
    on by default the model sometimes talked itself into `list_my_files` for a
    question about a named file's contents.

    6x fewer tokens on round 0, and a ReAct loop pays that per round. Latency
    was 0.6s against ~1.5s steady-state, with the variance gone: every `false`
    run was 0.6-0.7s and 45 tokens exactly.

    ## Why it asks first

    ⚠️ **Sending `think` to a model that does not declare `thinking` is a hard
    error.** Three models that have sat in `fast_path.tool_capable_models` did
    not declare it — `granite4.1:30b`, `qwen2.5-coder:32b`,
    `qwen3-coder-next:latest`. All three are out of the config as of
    2026-08-16, but the hazard is a property of Ollama, not of those names: a
    flat `False` here breaks every chat turn that lands on the next such
    model, and the list changes faster than this docstring. The
    lookup is cached per model in `OllamaClient.thinking_flag`, so it is one
    `/api/show` per model per process, not one per request.
    """
    if not no_thinking:
        return None
    return await ollama.thinking_flag(model, False)


def pick_fast_model(
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
) -> ModelSpec:
    spec = registry.first_healthy(task, health.is_healthy)
    if spec is None:
        raise OllamaError(f"No healthy model available for task={task}")
    return spec


# How many models the non-tools fast path will try before giving up. The
# top healthy model plus one fallback: if the two highest-priority healthy
# models both error mid-request, a third is unlikely to help and the added
# latency is real. Mirrors the deep panel's emergency-fallback cap of 2.
_FAST_FALLBACK_LIMIT = 2


def _healthy_fast_candidates(
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
    limit: int,
) -> list[ModelSpec]:
    """Up to `limit` healthy candidates for `task`, highest priority first.

    Reads health at selection time (same as `pick_fast_model`); a model
    cooled down by an earlier attempt in this same request is skipped on the
    next pass because `_run_one_chat` records the failure before we loop.
    """
    out: list[ModelSpec] = []
    for spec in registry.candidates(task):
        if health.is_healthy(spec.name):
            out.append(spec)
            if len(out) >= limit:
                break
    return out


def resolve_no_thinking_prose(
    no_thinking: bool,
    no_thinking_prose: bool | None,
) -> bool:
    """Resolve the backwards-compatible plain-prose thinking policy."""
    return no_thinking if no_thinking_prose is None else no_thinking_prose


class FastStreamEventType(StrEnum):
    """Client-neutral events emitted by one plain Fast model stream."""

    ATTEMPT = "attempt"
    STARTED = "started"
    TEXT = "text"
    USAGE = "usage"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FastStreamEvent:
    """One observable transition from a plain Fast stream attempt."""

    type: FastStreamEventType
    model: str = ""
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _InlineThinkFilter:
    """Strip inline think tags without buffering an ordinary full response.

    Ollama normally keeps reasoning in message.thinking. A model has also
    emitted literal tags in message.content; in the observed short-answer
    shape the same answer appeared on both sides of a dangling close tag.
    Keep only a marker-sized tail during normal streaming, suppress a real
    think block, and elide an exact post-close replay.

    A dangling close cannot retract text that is already on the wire. If the
    text after it differs, preserve that text; silently discarding a real final
    answer would be worse than exposing the model's malformed prefix.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _TAIL = max(len(_OPEN), len(_CLOSE)) - 1

    def __init__(self) -> None:
        self._mode = "normal"
        self._carry = ""
        self._emitted: list[str] = []
        self._after_close: list[str] = []

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._mode == "after_close":
            self._after_close.append(text)
            return ""

        data = self._carry + text
        self._carry = ""
        out: list[str] = []
        while data:
            if self._mode == "thinking":
                close_at = data.find(self._CLOSE)
                if close_at < 0:
                    self._carry = data[-self._TAIL:]
                    break
                data = data[close_at + len(self._CLOSE):]
                self._mode = "normal"
                continue

            open_at = data.find(self._OPEN)
            close_at = data.find(self._CLOSE)
            marker_at = min(
                (pos for pos in (open_at, close_at) if pos >= 0),
                default=-1,
            )
            if marker_at < 0:
                if len(data) <= self._TAIL:
                    self._carry = data
                else:
                    out.append(data[:-self._TAIL])
                    self._carry = data[-self._TAIL:]
                break

            out.append(data[:marker_at])
            if marker_at == open_at:
                data = data[marker_at + len(self._OPEN):]
                self._mode = "thinking"
                continue

            # A closing tag with no opening tag is the malformed shape seen in
            # OWUI. Text after it is buffered until completion so an exact
            # replay can be dropped instead of rendering the answer twice.
            self._mode = "after_close"
            self._after_close.append(data[marker_at + len(self._CLOSE):])
            data = ""

        emitted = "".join(out)
        if emitted:
            self._emitted.append(emitted)
        return emitted

    def finish(self) -> str:
        if self._mode == "normal":
            tail = self._carry
        elif self._mode == "after_close":
            tail = "".join(self._after_close)
            before = "".join(self._emitted).strip()
            if tail.strip() == before:
                tail = ""
        else:
            # An unclosed opening tag contains reasoning with no final answer.
            tail = ""
        self._carry = ""
        if tail:
            self._emitted.append(tail)
        return tail


async def stream_fast_path(
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    user_id: str | None = None,
    pipeline_started_at: float | None = None,
    no_thinking_prose: bool = False,
    terminal: StreamTerminal | None = None,
) -> AsyncIterator[FastStreamEvent]:
    """Stream one no-tools Fast answer with the non-stream policy contract.

    The top two healthy candidates are eligible. A model swap is allowed only
    before any answer text is emitted; once a token reaches the caller, an
    error terminates the same attempt. The generator owns thinking policy, GPU
    gating, health, dispatch/pipeline metrics, and terminal outcome. Its typed
    events keep banner/SSE rendering outside the model-execution contract.
    """

    terminal = terminal or StreamTerminal()
    started_at = (
        pipeline_started_at
        if pipeline_started_at is not None
        else time.perf_counter()
    )
    candidates = _healthy_fast_candidates(
        registry, health, task=task, limit=_FAST_FALLBACK_LIMIT,
    )
    last_error: OllamaError | None = None

    try:
        if not candidates:
            last_error = OllamaError(f"No healthy model available for task={task}")

        for attempt, candidate in enumerate(candidates, start=1):
            yield FastStreamEvent(FastStreamEventType.ATTEMPT, model=candidate.name)
            log.info(
                "fast_stream task=%s -> %s (attempt %d/%d)",
                task, candidate.name, attempt, len(candidates),
            )
            dispatch_total.labels(
                model=candidate.name, task_type=str(task), path="fast",
            ).inc()

            answer_started = False
            saw_done = False
            content_filter = _InlineThinkFilter()
            try:
                think = await _think(ollama, candidate.name, no_thinking_prose)
                async with gate.acquire(
                    candidate.name,
                    location=candidate.location,
                    user_id=user_id,
                ):
                    async for chunk in ollama.chat_stream(
                        model=candidate.name,
                        messages=messages,
                        options=options or None,
                        timeout_s=timeout_s,
                        think=think,
                    ):
                        message = chunk.get("message", {}) or {}
                        filtered = content_filter.feed(
                            str(message.get("content", "") or "")
                        )
                        if filtered:
                            if not answer_started:
                                answer_started = True
                                yield FastStreamEvent(
                                    FastStreamEventType.STARTED,
                                    model=candidate.name,
                                )
                            yield FastStreamEvent(
                                FastStreamEventType.TEXT,
                                model=candidate.name,
                                text=filtered,
                            )
                        if chunk.get("done"):
                            tail = content_filter.finish()
                            if not answer_started:
                                answer_started = True
                                yield FastStreamEvent(
                                    FastStreamEventType.STARTED,
                                    model=candidate.name,
                                )
                            if tail:
                                yield FastStreamEvent(
                                    FastStreamEventType.TEXT,
                                    model=candidate.name,
                                    text=tail,
                                )
                            yield FastStreamEvent(
                                FastStreamEventType.USAGE,
                                model=candidate.name,
                                prompt_tokens=int(
                                    chunk.get("prompt_eval_count", 0) or 0
                                ),
                                completion_tokens=int(
                                    chunk.get("eval_count", 0) or 0
                                ),
                            )
                            finish_reason = (
                                "length"
                                if chunk.get("done_reason") == "length"
                                else "stop"
                            )
                            health.record_success(candidate.name)
                            terminal.finish(
                                StreamOutcome.OK,
                                finish_reason=finish_reason,
                            )
                            saw_done = True
                            break
                if saw_done:
                    return

                # The upstream iterator ended without Ollama's required done
                # chunk. Preserve any held marker tail. With no visible answer
                # this is still pre-token and can fall back; otherwise truncate.
                tail = content_filter.finish()
                if tail:
                    if not answer_started:
                        answer_started = True
                        yield FastStreamEvent(
                            FastStreamEventType.STARTED,
                            model=candidate.name,
                        )
                    yield FastStreamEvent(
                        FastStreamEventType.TEXT,
                        model=candidate.name,
                        text=tail,
                    )
                missing_done = "Ollama stream ended without done"
                health.record_failure(candidate.name, missing_done)
                if not answer_started:
                    last_error = OllamaError(missing_done)
                    log.warning(
                        "fast_stream: %s ended before answer text "
                        "(attempt %d/%d); trying fallback",
                        candidate.name,
                        attempt,
                        len(candidates),
                    )
                    continue
                terminal.finish(
                    StreamOutcome.TRUNCATED,
                    finish_reason="length",
                )
                return
            except OllamaError as exc:
                health.record_failure(candidate.name, str(exc))
                last_error = exc
                log.warning(
                    "fast_stream: %s failed (attempt %d/%d, started=%s): %s",
                    candidate.name,
                    attempt,
                    len(candidates),
                    answer_started,
                    exc,
                )
                if answer_started:
                    tail = content_filter.finish()
                    if tail:
                        yield FastStreamEvent(
                            FastStreamEventType.TEXT,
                            model=candidate.name,
                            text=tail,
                        )
                    terminal.finish(StreamOutcome.ERROR, finish_reason="stop")
                    yield FastStreamEvent(
                        FastStreamEventType.ERROR,
                        model=candidate.name,
                        text=f"[ollama error: {exc}]",
                    )
                    return

        terminal.finish(StreamOutcome.ERROR, finish_reason="stop")
        yield FastStreamEvent(
            FastStreamEventType.ERROR,
            model=candidates[-1].name if candidates else "",
            text=f"[ollama error: {last_error}]",
        )
    except asyncio.CancelledError:
        terminal.finish_if_unset(StreamOutcome.CANCELLED)
        raise
    except GeneratorExit:
        terminal.finish_if_unset(StreamOutcome.CANCELLED)
        raise
    except BaseException:
        terminal.finish_if_unset(StreamOutcome.ERROR)
        raise
    finally:
        if not terminal.is_final:
            terminal.finish_if_unset(StreamOutcome.CANCELLED)
        pipeline_seconds.labels(mode="fast", task_type=str(task)).observe(
            time.perf_counter() - started_at
        )
        pipeline_total.labels(
            mode="fast",
            task_type=str(task),
            outcome=terminal.outcome.value,
        ).inc()


async def run_fast_path(
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    tools: ToolRegistry | None = None,
    tool_capable_models: set[str] | None = None,
    react_max_rounds: int = 3,
    react_compress_after: int = 2,
    react_max_tool_chars: int = 2000,
    react_dispatch_timeout_s: float = 30.0,
    react_compress_keep_last: int = 1,
    react_max_web_searches: int = 0,
    user_id: str | None = None,
    cfg: Any = None,
    no_thinking: bool = False,
    no_thinking_prose: bool | None = None,
    tool_observer: RunEventToolObserver | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (concrete_model, response_like_dict).

    The returned dict has the shape of an Ollama chat response (`message`,
    `prompt_eval_count`, `eval_count`) so the caller (graph node) doesn't
    care whether ReAct ran or not. When ReAct ran, an extra key
    `_react` carries the loop metadata (rounds, tool calls).

    **Non-tools branch — bounded fallback.** The top healthy model is tried
    first; if it raises `OllamaError`, it's cooled down and the next healthy
    candidate is tried, up to `_FAST_FALLBACK_LIMIT` models. Only when every
    candidate has failed does the error propagate (the graph node turns it
    into a 502). A transient blip on the highest-priority model no longer
    fails the request when a healthy fallback exists.

    **Tools branch — single-shot.** The ReAct branch is *not* retried across
    models: a mid-loop failure can land after tool side effects (e.g.
    `memory_store`), so a blind model-swap could double-apply them. The
    tool-capable model picked first is the one that answers, and an
    `OllamaError` from it propagates as before. (The failure still cools the
    model down, so the *next* request routes around it.)
    """
    spec = pick_fast_model(registry, health, task=task)
    use_tools = bool(
        tools and tools.by_name
        and tool_capable_models is not None
        and spec.name in tool_capable_models
    )

    # ⚠️ TWO BRANCHES, TWO THINKING DECISIONS — they are not the same question.
    # `no_thinking` governs the ReAct branch (tool calling); `no_thinking_prose`
    # governs the plain-chat branch (a direct answer). Measured 2026-08-19:
    # tool selection is 5/5 in ALL THREE thinking states, so thinking buys
    # nothing on the ReAct branch and costs ~0.7s PER ROUND. On prose it buys
    # qwen3.8 +4/125 (116 thinking-on vs 112 off, both suites, --repeat 5) for
    # ~4.0s once (6.25s vs 2.29s mean total), including the 9.11 > 9.9
    # flip it makes 3/5 with thinking off. ⛔ Do not collapse these back into
    # one flag without re-measuring
    # BOTH branches — a prose-only probe justified the wrong thing once already
    # (see `_think`).
    # Defaults to `no_thinking` so a config predating `no_thinking_prose` keeps
    # exactly the behaviour it has.
    prose_no_thinking = resolve_no_thinking_prose(
        no_thinking, no_thinking_prose,
    )

    if not use_tools:
        # Try the top healthy model, then fall back to the next healthy
        # candidate if it errors. `spec` (already chosen above) is the first
        # entry of this list, so the happy path is identical to before:
        # one model, one chat call, one dispatch increment.
        candidates = _healthy_fast_candidates(
            registry, health, task=task, limit=_FAST_FALLBACK_LIMIT,
        )
        last_err: OllamaError | None = None
        for attempt, cand in enumerate(candidates, start=1):
            log.info(
                "fast_path task=%s -> %s (tools=off, attempt %d/%d)",
                task, cand.name, attempt, len(candidates),
            )
            dispatch_total.labels(
                model=cand.name, task_type=str(task), path="fast",
            ).inc()
            try:
                think = await _think(ollama, cand.name, prose_no_thinking)
                async with gate.acquire(cand.name, location=cand.location, user_id=user_id):
                    resp = await ollama.chat(
                        model=cand.name, messages=messages,
                        options=options or None, timeout_s=timeout_s,
                        think=think,
                    )
                health.record_success(cand.name)
                return cand.name, resp
            except OllamaError as e:
                health.record_failure(cand.name, str(e))
                last_err = e
                log.warning(
                    "fast_path: %s failed (attempt %d/%d): %s",
                    cand.name, attempt, len(candidates), e,
                )
        # Every healthy candidate failed. Re-raise the last error (or a
        # no-healthy-model error if the registry had nothing to try).
        raise last_err or OllamaError(f"No healthy model available for task={task}")

    log.info("fast_path task=%s -> %s (tools=on)", task, spec.name)
    dispatch_total.labels(
        model=spec.name, task_type=str(task), path="fast_react",
    ).inc()
    react: ReactResult = await run_react(
        ollama, health, tools,  # type: ignore[arg-type]
        model=spec.name,
        messages=messages,
        options=options,
        timeout_s=timeout_s,
        max_rounds=react_max_rounds,
        compress_after_round=react_compress_after,
        max_tool_result_chars=react_max_tool_chars,
        tool_dispatch_timeout_s=react_dispatch_timeout_s,
        compress_keep_last=react_compress_keep_last,
        max_web_searches=react_max_web_searches,
        user_id=user_id,
        gate=gate,
        location=spec.location,
        cfg=cfg,
        think=await _think(ollama, spec.name, no_thinking),
        tool_observer=tool_observer,
    )
    return spec.name, {
        "message": {"role": "assistant", "content": react.content},
        "prompt_eval_count": react.prompt_eval_count,
        "eval_count": react.eval_count,
        "_react": {
            "tool_rounds": react.tool_rounds,
            "tool_calls": [
                {"name": r.name, "elapsed_s": r.elapsed_s, "is_error": r.is_error}
                for r in react.tool_calls
            ],
        },
    }


__all__ = [
    "FastStreamEvent",
    "FastStreamEventType",
    "pick_fast_model",
    "resolve_no_thinking_prose",
    "run_fast_path",
    "stream_fast_path",
]
