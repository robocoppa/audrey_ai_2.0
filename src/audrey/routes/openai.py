"""OpenAI-compatible routes.

Exposes five virtual models plus `/v1/chat/completions`:
  audrey_deep   — always deep (mixed pool)
  audrey_cloud  — always deep (cloud-only pool)
  audrey_local  — always deep (local-only pool)
  audrey_auto   — adaptive: fast for short prompts, deep for long ones
  audrey_fast   — always fast (no escalation, even on long prompts)

Requests go through the pipeline:
  classify → complexity gate → fast path | deep panel + synth.

Response shape is the OpenAI chat-completion contract so Open WebUI and
any other client can consume it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from audrey import __version__
from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.banners import (
    BANNER_DISPATCHING,
    BANNER_SEPARATOR,
    BANNER_SYNTHESIZING,
    BANNER_THINKING,
    PhaseTicker,
    tool_summary_block,
    worker_fail,
    worker_ok,
)
from audrey.pipeline.classify import classify as classify_fn
from audrey.pipeline.complexity import is_complex
from audrey.pipeline.context import datetime_system_message
from audrey.auth import AuthedUser, require_user
from audrey.pipeline.deep_panel import pool_key_for, run_panel_streaming
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.memory import (
    MEMORY_STORE_TOOL,
    memory_system_message,
    recall_for_request,
)
from audrey.pipeline.planner import plan as planner_plan
from audrey.pipeline.synthesize import synthesize_stream

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

# The three virtual models Audrey exposes. Each is a *pipeline mode*, not a
# real Ollama model. Mapping to concrete models happens inside the pipeline.
VIRTUAL_MODELS = (
    "audrey_deep",   # always deep (mixed pool)
    "audrey_cloud",  # always deep (cloud-only pool)
    "audrey_local",  # always deep (local-only pool)
    "audrey_auto",   # adaptive: fast for short prompts, deep for long ones
    "audrey_fast",   # always fast (no escalation, even on long prompts)
)


# ─── Schemas (OpenAI-compatible subset) ───────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    user: str | None = Field(
        default=None,
        description=(
            "OpenAI-spec passthrough field. Phase 26: Audrey **ignores** this "
            "for identity purposes — the canonical user id comes from the "
            "Authorization header (require_user → AuthedUser.email). Kept in the "
            "schema for OpenAI client compatibility; logged for debugging "
            "client-vs-resolved identity drift but never trusted."
        ),
    )


# ─── /v1/models ───────────────────────────────────────────────────────

@router.get("/models")
async def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": f"audrey-{__version__}",
            }
            for name in VIRTUAL_MODELS
        ],
    }


# ─── /v1/chat/completions ─────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
    app = request.app
    if payload.model not in VIRTUAL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model {payload.model!r}. "
                f"Supported virtual models: {list(VIRTUAL_MODELS)}."
            ),
        )

    # Phase 26: identity comes from the Authorization header via require_user,
    # NOT from payload.user (which is OpenAI-spec passthrough and trusted for
    # nothing). If a client sent a different `user` field, log it once for
    # drift-debugging but otherwise ignore.
    if payload.user and payload.user != me.email:
        log.debug(
            "chat.completions: payload.user=%r ignored (auth user=%r)",
            payload.user, me.email,
        )

    messages = [m.model_dump(exclude_none=True) for m in payload.messages]
    options = _options_from_request(payload)

    if payload.stream:
        return StreamingResponse(
            _stream_via_pipeline(app, payload, messages, options, user_id=me.email),
            media_type="text/event-stream",
        )

    return await _generate_via_pipeline(app, payload, messages, options, user_id=me.email)


async def _run_graph_with_metrics(graph, state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the graph and emit pipeline_seconds + pipeline_total.

    Labels read from the final state (mode/task_type) so the histogram bucket
    matches the path actually taken. On OllamaError we still observe latency
    against the requested virtual model with mode=unknown — the error happened
    before classify could pin a real mode.
    """
    t0 = time.perf_counter()
    try:
        final = await graph.ainvoke(state)
    except OllamaError:
        elapsed = time.perf_counter() - t0
        pipeline_seconds.labels(mode="unknown", task_type="unknown").observe(elapsed)
        pipeline_total.labels(mode="unknown", task_type="unknown", outcome="error").inc()
        raise
    elapsed = time.perf_counter() - t0
    mode = str(final.get("mode") or "unknown")
    task_type = str(final.get("task_type") or "unknown")
    pipeline_seconds.labels(mode=mode, task_type=task_type).observe(elapsed)
    pipeline_total.labels(mode=mode, task_type=task_type, outcome="ok").inc()
    return final


async def _generate_via_pipeline(
    app, payload: ChatCompletionRequest, messages, options, *, user_id: str
):
    """Non-streaming path: invoke the compiled LangGraph and format the result."""
    graph = app.state.graph
    inflight = app.state.inflight
    state = {
        "virtual_model": payload.model,
        "messages": messages,
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "max_tokens": payload.max_tokens,
        "user_id": user_id,
    }
    async with inflight.slot(user_id):
        try:
            final = await _run_graph_with_metrics(graph, state)
        except OllamaError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e

    extra = ""
    if final.get("mode") == "deep":
        drafts = final.get("drafts") or []
        ok = sum(1 for d in drafts if (d.get("content") or "").strip())
        extra = (
            f" pool={final.get('panel_pool')} workers={len(drafts)} ok={ok}"
            f" reflect={final.get('reflect_reason', '?')}"
            f"/attempts={final.get('reflect_attempts', 0)}"
            f" escalated={bool(final.get('escalated_from_fast'))}"
        )
    else:
        rounds = int(final.get("tool_rounds", 0))
        if rounds:
            calls = final.get("tool_calls_log") or []
            names = ",".join(c.get("name", "?") for c in calls) or "-"
            extra = f" tool_rounds={rounds} tool_calls=[{names}]"
    log.info(
        "chat.completions model=%s task=%s(%s, conf=%.2f) mode=%s -> %s%s",
        payload.model,
        final.get("task_type"),
        final.get("classify_reason"),
        final.get("classify_confidence", 0.0),
        final.get("mode"),
        final.get("concrete_model"),
        extra,
    )
    return _to_openai_response(
        virtual=payload.model,
        concrete=final.get("concrete_model", "?"),
        content=final.get("content", "") or "",
        prompt_tokens=int(final.get("prompt_eval_count", 0)),
        completion_tokens=int(final.get("eval_count", 0)),
    )


async def _stream_via_pipeline(
    app, payload: ChatCompletionRequest, messages, options, *, user_id: str
):
    """Streaming path.

    Routing is fixed by the virtual model:
      audrey_deep / audrey_cloud / audrey_local — always deep (banner stream)
      audrey_fast — always fast (token stream from the picked model)
      audrey_auto — adaptive: deep when prompt is complex, fast otherwise

    Deep requests run through `_stream_deep_with_banners` (phase 18). Fast
    requests stream a single model token-by-token; if the chosen model is
    tool-capable, the request goes through the graph for a ReAct loop and
    is emitted as one chunk on completion.
    """
    cfg = app.state.cfg
    ollama: OllamaClient = app.state.ollama
    registry: ModelRegistry = app.state.registry
    health: HealthTracker = app.state.health
    inflight = app.state.inflight
    router_cfg = cfg.router

    async with inflight.slot(user_id):
        user_text = _last_user_text(messages)
        task, reason, conf = await classify_fn(
            ollama,
            router_model=router_cfg.get("model", "qwen3:4b"),
            router_timeout_s=float(router_cfg.get("timeout_s", 20)),
            max_router_strikes=int(router_cfg.get("max_failures_before_fallback", 2)),
            user_text=user_text,
        )
        complex_, n = is_complex(messages, threshold=int(cfg.raw.get("complexity", {}).get("token_threshold", 500)))
        forced_deep = payload.model in ("audrey_deep", "audrey_cloud", "audrey_local")
        forced_fast = payload.model == "audrey_fast"
        if forced_deep:
            use_deep = True
        elif forced_fast:
            use_deep = False
        else:
            use_deep = complex_  # audrey_auto

        log.info(
            "chat.completions (stream) model=%s task=%s(%s, conf=%.2f) tokens=%d mode=%s",
            payload.model, task, reason, conf, n, "deep" if use_deep else "fast",
        )

        if use_deep:
            async for frame in _stream_deep_with_banners(
                app, payload, messages, options, task=task, conf=conf, user_id=user_id,
            ):
                yield frame
            return

        spec = registry.first_healthy(task, health.is_healthy)
        if spec is None:
            async for frame in _emit_single_message(
                payload.model, "none", f"[no healthy model for task={task}]"
            ):
                yield frame
            return

        # If the chosen model is tool-capable and tools are registered, route
        # the streaming request through the graph so the ReAct loop can fire.
        # Mid-stream tool dispatch isn't supported in Phase 7 — we emit one chunk.
        tool_capable = set(cfg.raw.get("fast_path", {}).get("tool_capable_models", []) or [])
        tools_active = bool(app.state.tools.by_name) and spec.name in tool_capable
        if tools_active:
            graph = app.state.graph
            state = {
                "virtual_model": payload.model,
                "messages": messages,
                "temperature": payload.temperature,
                "top_p": payload.top_p,
                "max_tokens": payload.max_tokens,
                "user_id": user_id,
            }
            try:
                final = await _run_graph_with_metrics(graph, state)
            except OllamaError as e:
                async for frame in _emit_single_message(
                    payload.model, "error", f"[ollama error: {e}]"
                ):
                    yield frame
                return
            concrete = final.get("concrete_model", spec.name)
            content = final.get("content", "") or "[empty]"
            # Per-worker tool-usage footer (Phase 28). Fast path is one
            # worker — `tool_calls_log` is its full call list. Skipped when
            # the ReAct loop ran zero tool calls.
            footer = tool_summary_block([
                (concrete, list(final.get("tool_calls_log") or []))
            ])
            if footer:
                content = content + footer
            async for frame in _emit_single_message(payload.model, concrete, content):
                yield frame
            return

        timeout = float(cfg.timeouts.get("fast_path", 180))
        async for frame in _stream_openai(
            ollama, payload.model, spec.name, messages, options,
            timeout_s=timeout, health=health,
            gate=app.state.gate, location=spec.location, user_id=user_id,
        ):
            yield frame


def _last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


# ─── Helpers ──────────────────────────────────────────────────────────

def _options_from_request(req: ChatCompletionRequest) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    return opts


def _to_openai_response(
    *,
    virtual: str,
    concrete: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": virtual,
        "system_fingerprint": f"audrey-{__version__}/{concrete}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_deep_with_banners(
    app, payload: ChatCompletionRequest, messages, options,
    *, task: str, conf: float, user_id: str,
):
    """Streaming deep path with progress banners (Phase 18).

    Bypasses the compiled graph for streaming so we can emit progress banners
    between pipeline phases. Workers run with the same scheduling as the
    non-streaming graph (parallel, gate-bounded for local) — `as_completed`
    just changes reception order so we can banner per-completion.

    Flow:
      Thinking phase   → memory recall + planner (already classified upstream)
      Dispatching      → run_panel_streaming, banner per worker
      Synthesizing     → synth (non-streamed in phase 18; streams in phase 19)
      separator + answer

    The `is_complex` / classify decisions happened in the caller; we receive
    the task type as an arg.
    """
    cfg = app.state.cfg
    ollama: OllamaClient = app.state.ollama
    registry: ModelRegistry = app.state.registry
    health: HealthTracker = app.state.health
    gate = app.state.gate
    tools = app.state.tools
    router_cfg = cfg.router
    agentic = cfg.raw.get("agentic", {}) or {}

    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    concrete = "deep_panel"
    fingerprint = f"audrey-{__version__}/{concrete}"

    def _delta_frame(text: str) -> str:
        frame = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": payload.model, "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        return f"data: {json.dumps(frame)}\n\n"

    def _stop_frame() -> str:
        frame = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": payload.model, "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(frame)}\n\n"

    # Role delta — required first frame per OpenAI streaming spec.
    role = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": payload.model, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role)}\n\n"

    # The banner emitter routes string fragments through a queue so the
    # ticker's background task is decoupled from the route generator's
    # frame emission. Bounded queue: slow consumer backpressures the ticker,
    # never the model.
    banner_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)

    async def emit(text: str) -> None:
        await banner_q.put(text)

    t0 = time.perf_counter()
    pipeline_outcome = "ok"
    drafts: list[dict[str, Any]] = []
    final_content = ""
    synth_model = "deep_panel"

    try:
        # ── Phase 1: Thinking (memory recall + planner) ─────────────────
        memory_cfg = agentic.get("memory", {}) or {}
        memory_enabled = bool(memory_cfg.get("enabled", True))
        memory_top_k = int(memory_cfg.get("top_k", 3))
        memory_timeout_s = float(memory_cfg.get("timeout_s", 5))
        planning_cfg = agentic.get("planning", {}) or {}
        planning_enabled = bool(planning_cfg.get("enabled", True))
        planning_min_tokens = int(planning_cfg.get("min_prompt_tokens", 40))
        planning_max_subtasks = int(planning_cfg.get("max_subtasks", 3))
        complexity_threshold = int(cfg.raw.get("complexity", {}).get("token_threshold", 500))
        _, prompt_tokens = is_complex(messages, threshold=complexity_threshold)

        async with PhaseTicker(BANNER_THINKING, emit):
            think_task = asyncio.create_task(_phase_thinking(
                ollama=ollama, tools=tools, user_id=user_id, messages=messages,
                memory_enabled=memory_enabled, memory_top_k=memory_top_k,
                memory_timeout_s=memory_timeout_s,
                planning_enabled=planning_enabled,
                planning_min_tokens=planning_min_tokens,
                planning_max_subtasks=planning_max_subtasks,
                prompt_tokens=prompt_tokens,
                router_cfg=router_cfg,
            ))
            async for frame in _drain_q_until_task(banner_q, think_task, _delta_frame):
                yield frame
            messages_with_memory, subtasks = think_task.result()
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # ── Phase 2: Dispatching panel ──────────────────────────────────
        pool_key = pool_key_for(payload.model)
        cloud_timeout = float(cfg.timeouts.get("cloud", 120))
        deep_worker_timeout = float(cfg.timeouts.get("deep_worker", 240))
        timeout_s = cloud_timeout if pool_key == "deep_panel_cloud" else deep_worker_timeout
        max_workers_cloud = int(agentic.get("max_deep_workers_cloud", 3))
        fast_path_cfg = cfg.raw.get("fast_path", {}) or {}
        tool_capable_models = set(fast_path_cfg.get("tool_capable_models", []) or [])
        react_cfg = agentic.get("react", {}) or {}
        deep_react_cfg = react_cfg.get("deep_worker", {}) or {}
        deep_react_max_rounds = int(deep_react_cfg.get("max_rounds", 2))
        deep_react_compress_after = int(deep_react_cfg.get("compress_after_round",
            int(react_cfg.get("compress_after_round", 2))))
        deep_react_max_tool_chars = int(deep_react_cfg.get("max_tool_result_chars",
            int(react_cfg.get("max_tool_result_chars", 2000))))
        deep_react_dispatch_timeout = float(deep_react_cfg.get("dispatch_timeout_s",
            float(react_cfg.get("dispatch_timeout_s", 30))))

        async with PhaseTicker(BANNER_DISPATCHING, emit) as ticker:
            panel_task = asyncio.create_task(_phase_dispatch(
                cfg=cfg, ollama=ollama, registry=registry, health=health, gate=gate,
                pool_key=pool_key, task=task, messages=messages_with_memory,
                subtasks=subtasks, options=options,
                timeout_s=timeout_s, max_workers_cloud=max_workers_cloud,
                tools=tools, tool_capable_models=tool_capable_models,
                react_max_rounds=deep_react_max_rounds,
                react_compress_after=deep_react_compress_after,
                react_max_tool_chars=deep_react_max_tool_chars,
                react_dispatch_timeout_s=deep_react_dispatch_timeout,
                user_id=user_id or None,
                ticker=ticker,
            ))
            async for frame in _drain_q_until_task(banner_q, panel_task, _delta_frame):
                yield frame
            drafts = panel_task.result()
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # ── Phase 3: Synthesizing (streaming) ───────────────────────────
        # Phase 19: synth tokens stream live. The Synthesizing banner runs
        # while we wait for the first token; on first_token we close the
        # banner with ✅, emit the separator, and then forward each delta
        # straight to the client. Mid-stream errors are surfaced inline.
        synth_done: dict[str, Any] = {}
        events_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=128)

        async def _run_synth_stream() -> None:
            try:
                async for evt in synthesize_stream(
                    cfg, ollama, registry, health, gate,
                    pool_key=pool_key, task=task,
                    messages=messages_with_memory, drafts=drafts,
                    subtasks=subtasks, timeout_s=deep_worker_timeout,
                    user_id=user_id or None,
                ):
                    await events_q.put(evt)
            finally:
                await events_q.put(None)

        synth_task = asyncio.create_task(_run_synth_stream())
        first_token_seen = False
        try:
            async with PhaseTicker(BANNER_SYNTHESIZING, emit):
                # Pre-first-token: drain banner dots; consume events; on
                # first_token we exit this `async with` so the ticker emits
                # its closing ✅, then we move to raw delta forwarding.
                while not first_token_seen:
                    drained = False
                    while not banner_q.empty():
                        item = banner_q.get_nowait()
                        if item is not None:
                            yield _delta_frame(item)
                            drained = True
                    try:
                        evt = await asyncio.wait_for(events_q.get(), timeout=0.05)
                    except TimeoutError:
                        if not drained and synth_task.done():
                            # Generator finished without ever yielding
                            # first_token (shouldn't happen — generator
                            # always emits one — but guard anyway).
                            break
                        continue
                    if evt is None:
                        # Generator finished pre-first-token.
                        break
                    etype = evt.get("type")
                    if etype == "first_token":
                        first_token_seen = True
                        synth_model = str(evt.get("model") or synth_model)
                        break
                    if etype == "fallback_attempt":
                        log.info("synth: falling back to %s after %s",
                                 evt.get("model"), evt.get("error"))
                        continue
                    if etype == "delta":
                        # Should not arrive before first_token — synth_stream
                        # always emits first_token first. Surface defensively
                        # without losing the text: stash it for emission once
                        # the banner closes.
                        first_token_seen = True
                        text = evt.get("text", "") or ""
                        if text:
                            final_content += text
                        break
                    if etype == "done":
                        synth_done = evt
                        break
            # Banner exited with ✅. Drain the closing fragment.
            async for frame in _drain_q_now(banner_q, _delta_frame):
                yield frame

            # Separator goes between banner and answer body.
            yield _delta_frame(BANNER_SEPARATOR)

            # If a delta was consumed pre-first-token (defensive path),
            # emit it now so we don't drop content.
            if first_token_seen and final_content and not synth_done:
                yield _delta_frame(final_content)

            # If first_token actually arrived, the event we consumed above
            # was just the marker — the next events on the queue are the
            # deltas. Stream them through.
            if first_token_seen and not synth_done:
                while True:
                    evt = await events_q.get()
                    if evt is None:
                        break
                    etype = evt.get("type")
                    if etype == "delta":
                        text = evt.get("text", "") or ""
                        if text:
                            final_content += text
                            yield _delta_frame(text)
                    elif etype == "done":
                        synth_done = evt
                    elif etype == "fallback_attempt":
                        # Can't happen post-first-token (synth_stream only
                        # falls back before any tokens), but log defensively.
                        log.warning("synth: unexpected fallback_attempt mid-stream: %r", evt)

            # Make sure the producer task is finished.
            await synth_task

            if synth_done:
                synth_model = str(synth_done.get("synthesizer_model") or synth_model)
                if not final_content:
                    final_content = synth_done.get("content", "") or "[empty]"
                if synth_done.get("synth_error"):
                    pipeline_outcome = "error"
            elif not final_content:
                final_content = "[empty]"
                pipeline_outcome = "error"

            # Per-worker tool-usage footer (Phase 28). Only renders rows for
            # workers that actually called tools; empty when no tools fired.
            footer = tool_summary_block([
                (str(d.get("model") or "?"), list(d.get("tool_calls") or []))
                for d in drafts
            ])
            if footer:
                yield _delta_frame(footer)

            yield _stop_frame()
            yield "data: [DONE]\n\n"
        finally:
            if not synth_task.done():
                synth_task.cancel()
                try:
                    await synth_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    except asyncio.CancelledError:
        # Client disconnected mid-stream. We can't yield more frames (the
        # response transport is gone), but we still want the metric/log to
        # reflect "cancelled" rather than the misleading "ok" the finally
        # block would otherwise record.
        pipeline_outcome = "cancelled"
        raise
    except OllamaError as e:
        pipeline_outcome = "error"
        log.warning("stream deep: ollama error: %s", e)
        yield _delta_frame(f"\n\n[ollama error: {e}]")
        yield _stop_frame()
        yield "data: [DONE]\n\n"
    except Exception:  # noqa: BLE001 — surface any other failure cleanly
        pipeline_outcome = "error"
        log.exception("stream deep: unexpected error")
        yield _delta_frame("\n\n[internal error]")
        yield _stop_frame()
        yield "data: [DONE]\n\n"
    finally:
        elapsed = time.perf_counter() - t0
        pipeline_seconds.labels(mode="deep", task_type=task).observe(elapsed)
        pipeline_total.labels(mode="deep", task_type=task, outcome=pipeline_outcome).inc()
        log.info(
            "stream deep done model=%s task=%s synth=%s outcome=%s elapsed=%.2fs",
            payload.model, task, synth_model, pipeline_outcome, elapsed,
        )


async def _drain_q_until_task(
    q: asyncio.Queue[str | None],
    task: asyncio.Task[Any],
    delta_frame: Callable[[str], str],
):
    """Yield SSE frames built from queued strings while `task` runs.

    Stops yielding when the task is done AND the queue is empty. Callers
    should run this *inside* the `async with PhaseTicker(...)` block, then
    drain once more *after* the block exits — the ticker's __aexit__ pushes
    the closing fragment (✓/✗ + newline) onto the queue, which lands after
    this generator returns.
    """
    while not task.done() or not q.empty():
        try:
            item = await asyncio.wait_for(q.get(), timeout=0.05)
        except TimeoutError:
            continue
        if item is None:
            continue
        yield delta_frame(item)


async def _drain_q_now(
    q: asyncio.Queue[str | None],
    delta_frame: Callable[[str], str],
):
    """Yield any remaining frames currently in the queue, then return.

    Used right after a PhaseTicker exits to surface the closing ✓/✗ + newline
    that __aexit__ pushed onto the queue.
    """
    while not q.empty():
        item = q.get_nowait()
        if item is None:
            continue
        yield delta_frame(item)


async def _phase_thinking(
    *, ollama, tools, user_id: str, messages,
    memory_enabled, memory_top_k, memory_timeout_s,
    planning_enabled, planning_min_tokens, planning_max_subtasks,
    prompt_tokens, router_cfg,
):
    """Run datetime injection + memory recall + planner. Returns (messages_with_context, subtasks)."""
    # Datetime first so it sits at the top of the system-message stack.
    # Mirrors what node_datetime does for the non-streaming graph.
    msgs = [datetime_system_message(), *messages]
    if memory_enabled and user_id.strip():
        hits = await recall_for_request(
            tools, user_id=user_id, messages=messages,
            top_k=memory_top_k, timeout_s=memory_timeout_s,
        )
        include_store_hint = tools is not None and MEMORY_STORE_TOOL in tools.by_name
        sys_msg = memory_system_message(
            hits, user_id=user_id, include_store_hint=include_store_hint,
        )
        if sys_msg is not None:
            msgs = [msgs[0], sys_msg, *messages]

    subtasks: list[str] = []
    if planning_enabled and prompt_tokens >= planning_min_tokens:
        user_text = _last_user_text(messages)
        subtasks = await planner_plan(
            ollama,
            planner_model=router_cfg.get("model", "qwen3:4b"),
            user_text=user_text,
            timeout_s=float(router_cfg.get("timeout_s", 20)),
            max_subtasks=planning_max_subtasks,
        )
    return msgs, subtasks


async def _phase_dispatch(
    *, cfg, ollama, registry, health, gate,
    pool_key, task, messages, subtasks, options,
    timeout_s, max_workers_cloud,
    tools, tool_capable_models,
    react_max_rounds, react_compress_after,
    react_max_tool_chars, react_dispatch_timeout_s,
    user_id, ticker: PhaseTicker,
):
    """Run the panel and feed per-worker results to the ticker. Returns drafts."""
    drafts: list[dict[str, Any]] = []
    async for evt in run_panel_streaming(
        cfg, ollama, registry, health, gate,
        pool_key=pool_key, task=task, messages=messages,
        subtasks=subtasks, options=options,
        timeout_s=timeout_s, max_workers_cloud=max_workers_cloud,
        tools=tools, tool_capable_models=tool_capable_models,
        react_max_rounds=react_max_rounds,
        react_compress_after=react_compress_after,
        react_max_tool_chars=react_max_tool_chars,
        react_dispatch_timeout_s=react_dispatch_timeout_s,
        user_id=user_id,
    ):
        if evt["type"] == "worker_done":
            ticker.append_tail(worker_ok(evt["model"]) if evt["ok"] else worker_fail(evt["model"]))
        elif evt["type"] == "final":
            drafts = list(evt["drafts"])
    return drafts


async def _stream_openai(
    ollama: OllamaClient,
    virtual: str,
    concrete: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    timeout_s: float | None = None,
    health: HealthTracker | None = None,
    gate: FairLocalGate | None = None,
    location: str = "local",
    user_id: str | None = None,
):
    """Convert Ollama's streaming chunks into OpenAI SSE frames.

    Phase 23: when `gate` is supplied and `location == "local"`, the entire
    token stream is held under the gate. Tokens are GPU-bound the whole way
    through (no tool dispatch in this branch), so a single acquire for the
    full duration is the right granularity here.
    """
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    fingerprint = f"audrey-{__version__}/{concrete}"

    first = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": virtual, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    gate_ctx = (
        gate.acquire(concrete, location=location, user_id=user_id)
        if gate is not None
        else _noop_async_ctx()
    )

    try:
        async with gate_ctx:
            async for chunk in ollama.chat_stream(
                model=concrete, messages=messages, options=options, timeout_s=timeout_s,
            ):
                msg = chunk.get("message", {}) or {}
                content = msg.get("content", "") or ""
                done = bool(chunk.get("done"))
                if content:
                    frame = {
                        "id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": virtual, "system_fingerprint": fingerprint,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(frame)}\n\n"
                if done:
                    final = {
                        "id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": virtual, "system_fingerprint": fingerprint,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    if health is not None:
                        health.record_success(concrete)
                    break
    except OllamaError as e:
        if health is not None:
            health.record_failure(concrete, str(e))
        err = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": virtual, "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {"content": f"\n\n[error: {e}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    yield "data: [DONE]\n\n"


@asynccontextmanager
async def _noop_async_ctx():
    yield


async def _emit_single_message(virtual: str, concrete: str, text: str):
    """One-shot SSE emission: role delta, single content delta, stop."""
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    fingerprint = f"audrey-{__version__}/{concrete}"
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': virtual, 'system_fingerprint': fingerprint, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': virtual, 'system_fingerprint': fingerprint, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': virtual, 'system_fingerprint': fingerprint, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


__all__ = ["router", "VIRTUAL_MODELS"]
