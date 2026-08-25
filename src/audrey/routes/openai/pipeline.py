"""Pipeline request handling — non-streaming + streaming, incl. deep banners.

This is the engine room: it runs the compiled graph for non-streaming
requests (`_generate_via_pipeline`) and hand-rolls the streaming path
(`_stream_via_pipeline` + `_stream_deep_with_banners` + the phase/queue
helpers) so progress banners can interleave with SSE frames.

Depends on `responses` (envelope + tool_call conversion) and `schemas`; never
on `passthrough` (the route layer forks to passthrough before reaching here).
The streaming ordering contract (`first_token` precedes every `delta`) is
documented on `synthesize_stream`; preserve it if you refactor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException

from audrey import __version__
from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.banners import (
    BANNER_DISPATCHING,
    BANNER_FACTCHECKING,
    BANNER_PLANNING,
    BANNER_RESEARCHING,
    BANNER_SEPARATOR,
    BANNER_SYNTHESIZING,
    BANNER_THINKING,
    BANNER_VERIFYING,
    BANNER_WRITING,
    PhaseTicker,
    panel_drafts_block,
    research_trace_block,
    tool_summary_block,
    worker_fail,
    worker_ok,
)
from audrey.pipeline.chat_archive import (
    ChatArchiveClient,
    StreamCollector,
)
from audrey.pipeline.classify import classify_with_registry
from audrey.pipeline.complexity import (
    count_last_user_tokens,
    count_tokens_by_role,
    has_deep_intent,
    is_complex,
    is_owui_task_request,
)
from audrey.pipeline.context import datetime_system_message
from audrey.pipeline.deep_panel import (
    cancel_and_drain,
    pick_panel_timeout,
    pool_key_for,
    run_panel_streaming,
    run_research_pipeline_streaming,
)
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.memory import (
    MEMORY_STORE_TOOL,
    memory_system_message,
    recall_for_request,
)
from audrey.pipeline.messages import has_image_part, last_user_text
from audrey.pipeline.planner import plan as planner_plan
from audrey.pipeline.prompts import (
    compose_system_messages,
    task_role_for,
    without_task_role,
)
from audrey.pipeline.synthesize import synthesize_stream
from audrey.pipeline.vision import describe_enabled, describe_for_text_model
from audrey.routes.openai.responses import _to_openai_response
from audrey.routes.openai.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)


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
    app, payload: ChatCompletionRequest, messages, options,
    *, user_id: str, conversation_id: str, user_turn_text: str,
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

    # Archive after the response is produced. Best-effort: never raises,
    # never delays the response (we await before returning to the client
    # but the tool-server call is bounded by ChatArchiveClient's timeout).
    archive_client: ChatArchiveClient | None = getattr(app.state, "archive_client", None)
    if archive_client is not None:
        await archive_client.archive_turn(
            registry=app.state.tools,
            user_id=user_id,
            conversation_id=conversation_id,
            user_content=user_turn_text,
            assistant_content=str(final.get("content", "") or ""),
            partial=False,
            virtual_model=payload.model,
            concrete_model=str(final.get("concrete_model", "?")),
            prompt_tokens=int(final.get("prompt_eval_count", 0)),
            completion_tokens=int(final.get("eval_count", 0)),
        )

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
    content = final.get("content", "") or ""
    # Debug/eval parity with the streaming path: append the debug block when
    # its flag is on. After the archive write above, so chat history never
    # carries either block. Two mode-specific views: deep panel gets the
    # drafts block (workers write competing answers — draft-vs-synth is the
    # comparison); research gets the staged trace (notes → ledger →
    # fact-check → writer guidance), since its drafts are researcher notes,
    # not candidate answers.
    if final.get("mode") == "deep":
        agentic_cfg = app.state.cfg.raw.get("agentic", {}) or {}
        if final.get("panel_pool") == "deep_panel_research":
            if bool(agentic_cfg.get("debug_research_trace", False)):
                content += research_trace_block(
                    drafts=list(final.get("drafts") or []),
                    ledger=final.get("research_ledger"),
                    factcheck=final.get("research_factcheck_ledger"),
                    critique=str(final.get("research_critique") or ""),
                    corrections=str(final.get("research_factcheck") or ""),
                    dispositions=str(final.get("research_dispositions") or ""),
                )
        elif bool(agentic_cfg.get("debug_panel_drafts", False)):
            content += panel_drafts_block(list(final.get("drafts") or []))
    return _to_openai_response(
        virtual=payload.model,
        concrete=final.get("concrete_model", "?"),
        content=content,
        prompt_tokens=int(final.get("prompt_eval_count", 0)),
        completion_tokens=int(final.get("eval_count", 0)),
    )



async def _stream_via_pipeline(
    app, payload: ChatCompletionRequest, messages, options,
    *, user_id: str, conversation_id: str, user_turn_text: str,
):
    """Streaming path.

    Routing is fixed by the virtual model:
      audrey_deep / audrey_cloud / audrey_local — always deep (banner stream)
      audrey_fast — always fast (token stream from the picked model)
      audrey_auto — adaptive: deep when prompt is complex, fast otherwise

    Deep requests run through `_stream_deep_with_banners`. Fast
    requests stream a single model token-by-token; if the chosen model is
    tool-capable, the request goes through the graph for a ReAct loop and
    is emitted as one chunk on completion.

    Capture: every emitted SSE frame is passed through `StreamCollector`
    so assistant-content deltas accumulate for the post-completion
    archive write. `_stream_deep_with_banners` does its own narrower
    capture (only synth deltas) so progress banners stay out of the
    archive — its return value is ignored here and the archive write
    happens inside that helper instead.
    """
    cfg = app.state.cfg
    ollama: OllamaClient = app.state.ollama
    registry: ModelRegistry = app.state.registry
    health: HealthTracker = app.state.health
    inflight = app.state.inflight
    router_cfg = cfg.router

    archive_client: ChatArchiveClient | None = getattr(app.state, "archive_client", None)
    collector = StreamCollector()
    chosen_concrete: str = "?"
    is_deep_branch = False  # deep handles its own archive write to skip banners

    try:
        async with inflight.slot(user_id):
            user_text = last_user_text(messages)
            # The deep-vs-fast decision uses only cheap local signals — it
            # does NOT need the classifier LLM. (`task` selects the model;
            # the route mode doesn't.) Deciding mode first lets the fast
            # branch put its Thinking banner on the wire *before* paying for
            # classification, so the user sees an ack immediately instead of
            # staring at nothing while the router model runs under GPU load.
            complexity_cfg = cfg.raw.get("complexity", {}) or {}
            # Gate on the request, not on Audrey's own scaffolding — the task
            # role was injected at the route, upstream of here. Mirrors
            # `node_complexity`; see `without_task_role` for what this cost.
            gate_messages = without_task_role(
                messages, task_role_for(payload.model, cfg)
            )
            complex_, n = is_complex(gate_messages, threshold=int(complexity_cfg.get("token_threshold", 500)))
            deep_intent = has_deep_intent(messages, complexity_cfg.get("deep_intent_phrases") or [])
            forced_deep = payload.model in ("audrey_deep", "audrey_cloud", "audrey_local", "audrey_research")
            forced_fast = payload.model == "audrey_fast"
            owui_task = is_owui_task_request(messages)
            image_turn = has_image_part(messages)
            if image_turn and not (forced_deep and describe_enabled(cfg)):
                # An attached image must reach a vision model — force fast.
                # (See the classify branch below: `task` is pinned to "vl".)
                #
                # The exception is an explicit deep/research pick: there the
                # caller named the model they want, so honour it. The image
                # gets transcribed below and the panel they chose reasons
                # over the text instead of being overridden by the vl pool.
                use_deep = False
            elif owui_task:
                # OWUI utility tasks (title gen, tags, follow-up suggestions)
                # always want a short, cheap answer — force fast.
                use_deep = False
            elif forced_deep:
                use_deep = True
            elif forced_fast:
                use_deep = False
            else:
                # audrey_auto: deep if the prompt is long OR explicitly asks for
                # depth (short-but-demanding prompts the length gate misses).
                use_deep = complex_ or deep_intent

            if use_deep and image_turn:
                # Transcribe before the panel runs: every worker and the
                # synthesizer rebuild prompts from `messages`, and none of
                # the text pools can read an `image_url` part. Done after
                # the mode decision so the complexity gate still judges the
                # user's own words, not the description's length.
                messages, _n_described = await describe_for_text_model(
                    messages,
                    ollama=ollama, registry=registry, health=health,
                    gate=app.state.gate, cfg=cfg,
                    target_model="", user_question=user_text, user_id=user_id,
                )

            if use_deep:
                # Deep classification still needs the router LLM (the panel
                # pools are task-keyed). Deep emits its own banner stream, so
                # classify here and hand the task type down.
                task, reason, conf = await classify_with_registry(
                    ollama, user_text=user_text, messages=messages, router_cfg=router_cfg,
                    cfg=cfg, registry=app.state.tools,
                )
                log.info(
                    "chat.completions (stream) model=%s task=%s(%s, conf=%.2f) tokens=%d mode=deep%s%s",
                    payload.model, task, reason, conf, n,
                    " owui_task=1" if owui_task else "",
                    " deep_intent=1" if (deep_intent and not complex_) else "",
                )
                if complexity_cfg.get("log_breakdown", False):
                    by_role = count_tokens_by_role(messages)
                    last_user = count_last_user_tokens(messages)
                    parts = " ".join(f"{r}={by_role[r]}" for r in sorted(by_role))
                    log.info("complexity.breakdown: %s last_user=%d", parts, last_user)
                is_deep_branch = True
                if payload.model == "audrey_research":
                    async for frame in _stream_research_with_banners(
                        app, payload, messages, options, task=task, conf=conf, user_id=user_id,
                        conversation_id=conversation_id, user_turn_text=user_turn_text,
                    ):
                        yield frame
                    return
                async for frame in _stream_deep_with_banners(
                    app, payload, messages, options, task=task, conf=conf, user_id=user_id,
                    conversation_id=conversation_id, user_turn_text=user_turn_text,
                ):
                    yield frame
                return

            # ─── Fast branch: banner first, classify second ───────────────
            # Emit the role frame + Thinking header now, using a
            # model-independent fingerprint (the concrete model isn't known
            # until classification picks it). The model name lands on the
            # closing banner line after classify — same as the deep / tool
            # paths. This is the latency fix: the ack is on the wire before
            # the (possibly slow) router call.
            fast_created = int(time.time())
            fast_cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            fast_fingerprint = f"audrey-{__version__}/{payload.model}"

            def _fast_delta(text: str) -> str:
                frame = {
                    "id": fast_cid, "object": "chat.completion.chunk", "created": fast_created,
                    "model": payload.model, "system_fingerprint": fast_fingerprint,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                return f"data: {json.dumps(frame)}\n\n"

            def _fast_stop() -> str:
                frame = {
                    "id": fast_cid, "object": "chat.completion.chunk", "created": fast_created,
                    "model": payload.model, "system_fingerprint": fast_fingerprint,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                return f"data: {json.dumps(frame)}\n\n"

            role_frame = {
                "id": fast_cid, "object": "chat.completion.chunk", "created": fast_created,
                "model": payload.model, "system_fingerprint": fast_fingerprint,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_frame)}\n\n"
            yield _fast_delta(BANNER_THINKING)

            # Now classify (model selection). Image turns are pinned to vl
            # regardless of wording — the text classifier can't see the image.
            if image_turn:
                task, reason, conf = "vl", "image_turn", 1.0
            else:
                task, reason, conf = await classify_with_registry(
                    ollama, user_text=user_text, messages=messages, router_cfg=router_cfg,
                    cfg=cfg, registry=app.state.tools,
                )
            log.info(
                "chat.completions (stream) model=%s task=%s(%s, conf=%.2f) tokens=%d mode=fast%s%s",
                payload.model, task, reason, conf, n,
                " owui_task=1" if owui_task else "",
                " image=1" if image_turn else "",
            )
            if complexity_cfg.get("log_breakdown", False):
                by_role = count_tokens_by_role(messages)
                last_user = count_last_user_tokens(messages)
                parts = " ".join(f"{r}={by_role[r]}" for r in sorted(by_role))
                log.info("complexity.breakdown: %s last_user=%d", parts, last_user)

            spec = registry.first_healthy(task, health.is_healthy)
            if spec is None:
                # Role frame + Thinking header are already on the wire. Close
                # the banner line and deliver the error as content under the
                # SAME stream identity — a fresh _emit_single_message would
                # emit a second role frame and break the chunk sequence.
                yield _fast_delta(" ❌\n")
                yield _fast_delta(BANNER_SEPARATOR)
                yield _fast_delta(f"[no healthy model for task={task}]")
                yield _fast_stop()
                yield "data: [DONE]\n\n"
                return
            chosen_concrete = spec.name

            # The role frame + Thinking header are already on the wire (emitted
            # before classify). `fast_cid` / `fast_created` / `fast_fingerprint`
            # and `_fast_delta` were defined up there; reuse them so every frame
            # in this response shares one identity.

            # If the chosen model is tool-capable and tools are registered, route
            # the streaming request through the graph so the ReAct loop can fire.
            # Mid-stream tool dispatch isn't supported — we emit one chunk after
            # the loop completes, rather than streaming tokens during ReAct rounds.
            tool_capable = set(cfg.raw.get("fast_path", {}).get("tool_capable_models", []) or [])
            tools_active = bool(app.state.tools.by_name) and spec.name in tool_capable
            if tools_active:
                # Tool-capable path can take 1-3s on a `kb_search` round, so
                # use a full PhaseTicker that emits dots while the graph runs.
                # The Thinking header is already on the wire (emit_header=False)
                # — the ticker just dots the open line and closes it.
                graph = app.state.graph
                state = {
                    "virtual_model": payload.model,
                    "messages": messages,
                    "temperature": payload.temperature,
                    "top_p": payload.top_p,
                    "max_tokens": payload.max_tokens,
                    "user_id": user_id,
                }
                banner_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)

                async def _banner_emit(text: str) -> None:
                    await banner_q.put(text)

                graph_task: asyncio.Task[Any] | None = None
                try:
                    async with PhaseTicker(
                        BANNER_THINKING, _banner_emit, emit_header=False,
                    ) as ticker:
                        graph_task = asyncio.create_task(
                            _run_graph_with_metrics(graph, state)
                        )
                        async for frame in _drain_q_until_task(banner_q, graph_task, _fast_delta):
                            yield frame
                        final = graph_task.result()
                        # Surface the model on the Thinking line, right before
                        # the ticker's closing ✅ — mirrors the deep panel's
                        # per-worker name. The graph's `concrete_model` is the
                        # model that actually ran (health rerouting may differ
                        # from the initial pick); fall back to `spec.name`.
                        ticker.append_tail(f"  {final.get('concrete_model', spec.name)}")
                except OllamaError as e:
                    # The role frame + Thinking header are already on the wire,
                    # and PhaseTicker's __aexit__ queued a closing ❌. Drain it,
                    # then deliver the error as content under the SAME stream
                    # identity (a fresh _emit_single_message would emit a second
                    # role frame and break the chunk sequence).
                    async for frame in _drain_q_now(banner_q, _fast_delta):
                        yield frame
                    yield _fast_delta(BANNER_SEPARATOR)
                    yield _fast_delta(f"[ollama error: {e}]")
                    yield _fast_stop()
                    yield "data: [DONE]\n\n"
                    return
                # Drain the closing ✅\n that PhaseTicker pushed on exit.
                async for frame in _drain_q_now(banner_q, _fast_delta):
                    yield frame

                concrete = final.get("concrete_model", spec.name)
                chosen_concrete = concrete
                content = final.get("content", "") or "[empty]"
                # Per-worker tool-usage footer. Fast path is one worker —
                # `tool_calls_log` is its full call list. Skipped when the ReAct
                # loop ran zero tool calls.
                #
                # ⚠️ …AND the deep workers', because THIS BRANCH CAN GO DEEP.
                # `route_after_fast_path` escalates fast→deep INSIDE the graph,
                # so `final` here may be a synthesized panel answer. It escalates
                # only when `tool_rounds == 0` (graph.py:146), which means
                # `tool_calls_log` is empty by construction on exactly those
                # turns — while the workers that produced the answer did call
                # tools, into `drafts`. Reading only the fast log therefore
                # rendered NO footer at all for every escalated turn:
                # `video-two-file-compare` lost it 53 times out of 53 across the
                # archive, taking `grounded`, `_ungrounded_content` and
                # `no_reasoning_leak` blind with it — all three parse the footer,
                # on the one case built to catch two files being conflated.
                footer = tool_summary_block(
                    [(concrete, list(final.get("tool_calls_log") or []))]
                    + [(str(d.get("model") or "?"), list(d.get("tool_calls") or []))
                       for d in (final.get("drafts") or [])]
                )
                if footer:
                    content = content + footer
                # Debug/eval: an ESCALATED turn was answered by the panel, not
                # by the fast model — `node_mark_escalated` sets `mode: deep`.
                # Both other paths already append the drafts block behind
                # `agentic.debug_panel_drafts`; this branch never checked, so a
                # fast→deep turn produced NO trace of the panel anywhere on the
                # wire. Parity, not a new surface: same existing flag, off by
                # default, nothing user-facing added.
                #
                # ⚠️ Appended to `content` AFTER `collector.feed_text` below
                # takes the raw answer, so the chat archive never carries it —
                # same contract as the footer above and as the deep branch.
                if final.get("mode") == "deep":
                    agentic_cfg = cfg.raw.get("agentic", {}) or {}
                    if bool(agentic_cfg.get("debug_panel_drafts", False)):
                        drafts_debug = panel_drafts_block(list(final.get("drafts") or []))
                        if drafts_debug:
                            content = content + drafts_debug
                # Separator between banner and answer body — matches deep.
                yield _fast_delta(BANNER_SEPARATOR)
                # Tool-capable fast path emits the answer in one chunk under the
                # already-open stream identity (the role frame went out before
                # classify). Feed the answer text directly into the collector —
                # the SSE-frame parser would also catch it, but feeding text is
                # cheaper and unambiguous.
                collector.feed_text(str(final.get("content", "") or ""))
                yield _fast_delta(content)
                yield _fast_stop()
                yield "data: [DONE]\n\n"
                return

            # Plain-chat fast path: no tools, tokens stream within ~200ms of
            # first byte. The Thinking header is already on the wire (emitted
            # before classify); close the line with the model name + ✅ and a
            # separator, then stream the answer. No dot animation — the first
            # token arrives fast enough that dots would never show.
            yield _fast_delta(worker_ok(spec.name) + "\n")
            yield _fast_delta(BANNER_SEPARATOR)

            timeout = float(cfg.timeouts.get("fast_path", 180))
            async for frame in collector.wrap(_stream_openai(
                ollama, payload.model, spec.name, messages, options,
                timeout_s=timeout, health=health,
                gate=app.state.gate, location=spec.location, user_id=user_id,
            )):
                yield frame
    except asyncio.CancelledError:
        # Client disconnect mid-stream. Mark partial; archive what we
        # captured before we re-raise. The deep branch handles its own
        # cancellation accounting inside _stream_deep_with_banners.
        if not is_deep_branch:
            collector.mark_partial()
        raise
    finally:
        # Archive only the fast/tool-capable branches here; deep branch
        # owns its own archive write because banner frames must not be
        # included in the captured assistant text.
        if not is_deep_branch and archive_client is not None:
            await archive_client.archive_turn(
                registry=app.state.tools,
                user_id=user_id,
                conversation_id=conversation_id,
                user_content=user_turn_text,
                assistant_content=collector.text,
                partial=collector.partial,
                virtual_model=payload.model,
                concrete_model=chosen_concrete,
            )



async def _stream_deep_with_banners(
    app, payload: ChatCompletionRequest, messages, options,
    *, task: str, conf: float, user_id: str,
    conversation_id: str = "",
    user_turn_text: str = "",
):
    """Streaming deep path with progress banners.

    Bypasses the compiled graph for streaming so we can emit progress banners
    between pipeline phases. Workers run with the same scheduling as the
    non-streaming graph (parallel, gate-bounded for local) — `as_completed`
    just changes reception order so we can banner per-completion.

    Flow:
      Planning phase   → memory recall + planner (already classified upstream)
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
    owned_tasks: list[asyncio.Task[Any]] = []

    try:
        # ── Stage 1: Planning (memory recall + planner) ─────────────────
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

        async with PhaseTicker(BANNER_PLANNING, emit):
            think_task = asyncio.create_task(_phase_thinking(
                ollama=ollama, tools=tools, user_id=user_id, messages=messages,
                memory_enabled=memory_enabled, memory_top_k=memory_top_k,
                memory_timeout_s=memory_timeout_s,
                planning_enabled=planning_enabled,
                planning_min_tokens=planning_min_tokens,
                planning_max_subtasks=planning_max_subtasks,
                prompt_tokens=prompt_tokens,
                router_cfg=router_cfg,
                cfg=cfg,
            ))
            owned_tasks.append(think_task)
            async for frame in _drain_q_until_task(banner_q, think_task, _delta_frame):
                yield frame
            messages_with_memory, subtasks = think_task.result()
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # ── Stage 2: Dispatching panel ──────────────────────────────────
        pool_key = pool_key_for(payload.model)
        timeout_s = pick_panel_timeout(cfg, pool_key)
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
        deep_react_compress_keep_last = int(deep_react_cfg.get("compress_keep_last",
            int(react_cfg.get("compress_keep_last", 1))))
        deep_react_max_web_searches = int(deep_react_cfg.get("max_web_searches",
            int(react_cfg.get("max_web_searches", 0))))

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
                react_compress_keep_last=deep_react_compress_keep_last,
                react_max_web_searches=deep_react_max_web_searches,
                user_id=user_id or None,
                ticker=ticker,
            ))
            owned_tasks.append(panel_task)
            async for frame in _drain_q_until_task(banner_q, panel_task, _delta_frame):
                yield frame
            drafts = panel_task.result()
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # ── Stage 3: Synthesizing (streaming) ───────────────────────────
        # Synth tokens stream live. The Synthesizing banner runs while we
        # wait for the first token; on first_token we close the banner with
        # ✅, emit the separator, and then forward each delta straight to
        # the client. Mid-stream errors are surfaced inline.
        synth_done: dict[str, Any] = {}
        events_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=128)

        async def _run_synth_stream() -> None:
            async for evt in synthesize_stream(
                cfg, ollama, registry, health, gate,
                pool_key=pool_key, task=task,
                messages=messages_with_memory, drafts=drafts,
                # Pool-aware, matching the panel (`timeout_s` above):
                # cloud-only pools get `timeouts.cloud` rather than the
                # longer `deep_worker` budget.
                subtasks=subtasks, timeout_s=timeout_s,
                user_id=user_id or None,
            ):
                await events_q.put(evt)

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
                        evt = await _queue_get_with_timeout(events_q)
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
                        # Unreachable safety net: `synthesize_stream` documents
                        # `first_token` as preceding every `delta` and enforces
                        # the ordering structurally. We keep this branch so a
                        # future refactor that violates the contract degrades
                        # to text-loss rather than dropping content silently —
                        # the resulting "missing banner ✅" is the visible
                        # symptom that points back to the broken invariant.
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
                    evt = await _queue_get_until_task(events_q, synth_task)
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

            # Per-worker tool-usage footer. Only renders rows for workers
            # that actually called tools; empty when no tools fired.
            footer = tool_summary_block([
                (str(d.get("model") or "?"), list(d.get("tool_calls") or []))
                for d in drafts
            ])
            if footer:
                yield _delta_frame(footer)

            # Debug/eval: append every worker's full draft (opt-in via
            # `agentic.debug_panel_drafts`) so draft-vs-synth quality can be
            # compared from the answer artifact alone. Yielded but NOT folded
            # into `final_content` — the chat archive never carries it.
            if bool(agentic.get("debug_panel_drafts", False)):
                drafts_debug = panel_drafts_block(drafts)
                if drafts_debug:
                    yield _delta_frame(drafts_debug)

            yield _stop_frame()
            yield "data: [DONE]\n\n"
        finally:
            if not synth_task.done():
                synth_task.cancel()
                try:
                    await synth_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 — cleanup path; we just cancelled the task
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
    except Exception:
        pipeline_outcome = "error"
        log.exception("stream deep: unexpected error")
        yield _delta_frame("\n\n[internal error]")
        yield _stop_frame()
        yield "data: [DONE]\n\n"
    finally:
        await cancel_and_drain(owned_tasks)
        elapsed = time.perf_counter() - t0
        pipeline_seconds.labels(mode="deep", task_type=task).observe(elapsed)
        pipeline_total.labels(mode="deep", task_type=task, outcome=pipeline_outcome).inc()
        # `workers/ok/tool_grounded` mirror `graph.py`'s `deep_panel:` line so
        # one grep covers both pipelines. Without them the streaming path —
        # the one OWUI actually uses for questions — reported nothing about
        # the panel at all, and "did the workers use tools" could only be
        # answered by attributing `react:` lines to turns via
        # timestamp-minus-elapsed. Count WORKERS, not zero-tool rounds: the
        # final round of every ReAct loop has `tool_calls=0` by construction.
        ok = sum(1 for d in drafts if (d.get("content") or "").strip())
        grounded = sum(1 for d in drafts if int(d.get("tool_rounds", 0) or 0) > 0)
        log.info(
            "stream deep done model=%s task=%s synth=%s outcome=%s elapsed=%.2fs"
            " workers=%d ok=%d tool_grounded=%d",
            payload.model, task, synth_model, pipeline_outcome, elapsed,
            len(drafts), ok, grounded,
        )
        # Archive whatever synth content actually streamed. `final_content`
        # is built only from synth deltas (banner text never lands here),
        # so progress banners stay out of the archive automatically.
        archive_client: ChatArchiveClient | None = getattr(app.state, "archive_client", None)
        if archive_client is not None and conversation_id:
            await archive_client.archive_turn(
                registry=app.state.tools,
                user_id=user_id,
                conversation_id=conversation_id,
                user_content=user_turn_text,
                assistant_content=final_content,
                partial=(pipeline_outcome == "cancelled"),
                virtual_model=payload.model,
                concrete_model=synth_model,
            )


async def _stream_research_with_banners(
    app, payload: ChatCompletionRequest, messages, options,
    *, task: str, conf: float, user_id: str,
    conversation_id: str = "",
    user_turn_text: str = "",
):
    """Streaming `audrey_research` path: Planning → Researching → Verifying → Writing.

    Same banner machinery as `_stream_deep_with_banners` (PhaseTicker + the
    queue/drain helpers) with two extra stages. The single staged pipeline
    (`run_research_pipeline_streaming`) runs as one background task; its events
    drive the three phase banners and stream the Writer's tokens live as the
    answer. Mirrors the non-streaming `node_research` in `pipeline/graph.py` —
    keep the two in sync.
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
    concrete = "deep_panel_research"
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

    role = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": payload.model, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role)}\n\n"

    banner_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)

    async def emit(text: str) -> None:
        await banner_q.put(text)

    t0 = time.perf_counter()
    pipeline_outcome = "ok"
    drafts: list[dict[str, Any]] = []
    final_content = ""
    writer_model = "deep_panel_research"
    owned_tasks: list[asyncio.Task[Any]] = []

    try:
        # ── Stage 0: Planning (memory recall + planner, reused verbatim) ──
        memory_cfg = agentic.get("memory", {}) or {}
        memory_enabled = bool(memory_cfg.get("enabled", True))
        planning_cfg = agentic.get("planning", {}) or {}
        complexity_threshold = int(cfg.raw.get("complexity", {}).get("token_threshold", 500))
        _, prompt_tokens = is_complex(messages, threshold=complexity_threshold)

        async with PhaseTicker(BANNER_PLANNING, emit):
            think_task = asyncio.create_task(_phase_thinking(
                ollama=ollama, tools=tools, user_id=user_id, messages=messages,
                memory_enabled=memory_enabled,
                memory_top_k=int(memory_cfg.get("top_k", 3)),
                memory_timeout_s=float(memory_cfg.get("timeout_s", 5)),
                planning_enabled=bool(planning_cfg.get("enabled", True)),
                planning_min_tokens=int(planning_cfg.get("min_prompt_tokens", 40)),
                planning_max_subtasks=int(planning_cfg.get("max_subtasks", 3)),
                prompt_tokens=prompt_tokens,
                router_cfg=router_cfg,
                cfg=cfg,
            ))
            owned_tasks.append(think_task)
            async for frame in _drain_q_until_task(banner_q, think_task, _delta_frame):
                yield frame
            # Research workers answer the full prompt; planner subtasks are not
            # used (the research fan-out grounds the whole question).
            messages_with_memory, _subtasks = think_task.result()
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # Drive the whole staged pipeline as one background task; its events
        # feed the three phase banners and the live answer stream.
        events_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
        timeout_s = pick_panel_timeout(cfg, "deep_panel_research")
        max_researchers_cloud = int(agentic.get("max_research_workers_cloud", 2))
        fast_path_cfg = cfg.raw.get("fast_path", {}) or {}
        tool_capable_models = set(fast_path_cfg.get("tool_capable_models", []) or [])

        async def _run_pipeline() -> None:
            async for evt in run_research_pipeline_streaming(
                cfg, ollama, registry, health, gate,
                task=task, messages=messages_with_memory, options=options,
                timeout_s=timeout_s, max_researchers_cloud=max_researchers_cloud,
                tools=tools, tool_capable_models=tool_capable_models,
                user_id=user_id or None,
            ):
                await events_q.put(evt)

        pipe_task = asyncio.create_task(_run_pipeline())
        owned_tasks.append(pipe_task)

        # ── Stage 1: Researching (banner; tails per researcher) ───────────
        async with PhaseTicker(BANNER_RESEARCHING, emit) as ticker:
            while True:
                # Surface any banner dots queued by the ticker.
                while not banner_q.empty():
                    item = banner_q.get_nowait()
                    if item is not None:
                        yield _delta_frame(item)
                try:
                    evt = await _queue_get_with_timeout(events_q)
                except TimeoutError:
                    if pipe_task.done() and events_q.empty():
                        break
                    continue
                if evt is None:
                    break
                etype = evt.get("type")
                if etype == "researcher_done":
                    ticker.append_tail(
                        worker_ok(evt["model"]) if evt["ok"] else worker_fail(evt["model"])
                    )
                elif etype == "findings_ready":
                    break
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # ── Stage 2: Verifying → Stage 3: Fact-checking → Stage 4: Writing ─
        # Event order after findings_ready is: optionally `verify_done`,
        # optionally `factcheck_done`, then zero+ `write_delta`, then `done`.
        # Each banner stays open until the next stage's event arrives. A
        # helper waits through dots until it gets the next "real" event,
        # returning it so the caller can decide which banner to open next.
        # `pending` carries the event that ended a wait so we never drop it.
        done_evt: dict[str, Any] = {}

        async def _next_event():
            """Drain banner dots, return the next pipeline event (or None at end)."""
            while True:
                while not banner_q.empty():
                    item = banner_q.get_nowait()
                    if item is not None:
                        return ("banner", item)
                try:
                    evt = await _queue_get_with_timeout(events_q)
                except TimeoutError:
                    if pipe_task.done() and events_q.empty():
                        return ("event", None)
                    continue
                return ("event", evt)

        # Verifying: open until we see factcheck_done, the first write_delta,
        # or done. verify_done just confirms the stage finished (keep waiting).
        pending: dict[str, Any] | None = None
        async with PhaseTicker(BANNER_VERIFYING, emit):
            while True:
                kind, item = await _next_event()
                if kind == "banner":
                    yield _delta_frame(item)
                    continue
                if item is None:
                    break
                if item.get("type") == "verify_done":
                    continue
                pending = item
                break
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame

        # Fact-checking: only if the verify wait ended on `factcheck_done`.
        # Otherwise the stage was skipped (no grounding / no factchecker) and
        # `pending` is already the first write_delta or done — skip the banner.
        first_write: dict[str, Any] | None = None
        if pending is not None and pending.get("type") == "factcheck_done":
            async with PhaseTicker(BANNER_FACTCHECKING, emit):
                while True:
                    kind, item = await _next_event()
                    if kind == "banner":
                        yield _delta_frame(item)
                        continue
                    if item is None:
                        break
                    first_write = item  # first write_delta or done
                    break
            async for frame in _drain_q_now(banner_q, _delta_frame):
                yield frame
        else:
            first_write = pending

        # Writing banner: it closes as soon as we have the first answer token.
        async with PhaseTicker(BANNER_WRITING, emit):
            # `first_write` already in hand — close immediately. A brief tick
            # may queue; drain it after exit.
            pass
        async for frame in _drain_q_now(banner_q, _delta_frame):
            yield frame
        yield _delta_frame(BANNER_SEPARATOR)

        # Emit the first write chunk (if any), then stream the rest.
        def _consume(evt: dict[str, Any]) -> str:
            nonlocal done_evt
            if evt.get("type") == "write_delta":
                return evt.get("text", "") or ""
            if evt.get("type") == "done":
                done_evt = evt
            return ""

        if first_write is not None:
            text = _consume(first_write)
            if text:
                final_content += text
                yield _delta_frame(text)
        if not done_evt:
            while True:
                evt = await _queue_get_until_task(events_q, pipe_task)
                if evt is None:
                    break
                text = _consume(evt)
                if text:
                    final_content += text
                    yield _delta_frame(text)

        await pipe_task

        if done_evt:
            writer_model = str(done_evt.get("writer_model") or writer_model)
            drafts = list(done_evt.get("drafts") or [])
            if not final_content:
                final_content = done_evt.get("content", "") or "[empty]"
            if done_evt.get("error"):
                pipeline_outcome = "error"
        elif not final_content:
            final_content = "[empty]"
            pipeline_outcome = "error"

        footer = tool_summary_block([
            (str(d.get("model") or "?"), list(d.get("tool_calls") or []))
            for d in drafts
        ])
        if footer:
            yield _delta_frame(footer)

        # Debug/eval: append the staged-pipeline trace (researcher notes,
        # ledger, verifier critique, fact-check verdicts, writer guidance) —
        # opt-in via `agentic.debug_research_trace`, the research counterpart
        # of the deep panel-drafts block. Yielded but NOT folded into
        # `final_content` — the chat archive never carries it.
        if bool(agentic.get("debug_research_trace", False)):
            trace = research_trace_block(
                drafts=drafts,
                ledger=done_evt.get("ledger"),
                factcheck=done_evt.get("factcheck"),
                critique=str(done_evt.get("critique") or ""),
                corrections=str(done_evt.get("corrections") or ""),
                dispositions=str(done_evt.get("dispositions") or ""),
            )
            if trace:
                yield _delta_frame(trace)

        yield _stop_frame()
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        pipeline_outcome = "cancelled"
        raise
    except OllamaError as e:
        pipeline_outcome = "error"
        log.warning("stream research: ollama error: %s", e)
        yield _delta_frame(f"\n\n[ollama error: {e}]")
        yield _stop_frame()
        yield "data: [DONE]\n\n"
    except Exception:
        pipeline_outcome = "error"
        log.exception("stream research: unexpected error")
        yield _delta_frame("\n\n[internal error]")
        yield _stop_frame()
        yield "data: [DONE]\n\n"
    finally:
        await cancel_and_drain(owned_tasks)
        elapsed = time.perf_counter() - t0
        pipeline_seconds.labels(mode="deep", task_type=task).observe(elapsed)
        pipeline_total.labels(mode="deep", task_type=task, outcome=pipeline_outcome).inc()
        # Same fields as the deep path — `drafts` here are the Stage-1
        # researcher notes, which is where research-mode grounding lives.
        ok = sum(1 for d in drafts if (d.get("content") or "").strip())
        grounded = sum(1 for d in drafts if int(d.get("tool_rounds", 0) or 0) > 0)
        log.info(
            "stream research done model=%s task=%s writer=%s outcome=%s elapsed=%.2fs"
            " workers=%d ok=%d tool_grounded=%d",
            payload.model, task, writer_model, pipeline_outcome, elapsed,
            len(drafts), ok, grounded,
        )
        archive_client: ChatArchiveClient | None = getattr(app.state, "archive_client", None)
        if archive_client is not None and conversation_id:
            await archive_client.archive_turn(
                registry=app.state.tools,
                user_id=user_id,
                conversation_id=conversation_id,
                user_content=user_turn_text,
                assistant_content=final_content,
                partial=(pipeline_outcome == "cancelled"),
                virtual_model=payload.model,
                concrete_model=writer_model,
            )


async def _queue_get_with_timeout(q: asyncio.Queue[Any]) -> Any:
    """Poll one queue without parking cancellation on a Queue.get Future."""
    if not q.empty():
        return q.get_nowait()
    await asyncio.sleep(0.05)
    if q.empty():
        raise TimeoutError
    return q.get_nowait()


async def _queue_get_until_task(
    q: asyncio.Queue[Any],
    task: asyncio.Task[Any],
) -> Any | None:
    """Return the next item, or None after the producer settles and drains."""
    while True:
        if task.done() and q.empty():
            return None
        try:
            return await _queue_get_with_timeout(q)
        except TimeoutError:
            continue


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
            item = await _queue_get_with_timeout(q)
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
    prompt_tokens, router_cfg, cfg=None,
):
    """Run datetime injection + memory recall + planner. Returns (messages_with_context, subtasks).

    Used by the streaming deep path. Mirrors the non-streaming graph's
    `node_datetime` → `node_memory_recall` → planner sequence in
    `pipeline/graph.py`. Both paths call the same underlying helpers
    (`datetime_system_message`, `recall_for_request`, `memory_system_message`,
    `compose_system_messages`, `planner_plan`) in the same order; only the
    orchestration shape differs (graph nodes pass state in/out, this returns
    a tuple). If you add a new context-injection step here, add the matching
    node in `pipeline/graph.py` too — otherwise streaming-deep and
    non-streaming runs will silently disagree.
    """
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
            hits, user_id=user_id, include_store_hint=include_store_hint, cfg=cfg,
        )
        chat_history_available = tools is not None and "chat_history_search" in tools.by_name
        composed = compose_system_messages(
            memory_hint=sys_msg,
            chat_history_guidance=chat_history_available,
        )
        if composed:
            msgs = [msgs[0], *composed, *messages]

    subtasks: list[str] = []
    if planning_enabled and prompt_tokens >= planning_min_tokens:
        user_text = last_user_text(messages)
        subtasks = await planner_plan(
            ollama,
            planner_model=router_cfg.get("model", "qwen3:4b"),
            user_text=user_text,
            timeout_s=float(router_cfg.get("timeout_s", 20)),
            max_subtasks=planning_max_subtasks,
            cfg=cfg,
        )
    return msgs, subtasks


async def _phase_dispatch(
    *, cfg, ollama, registry, health, gate,
    pool_key, task, messages, subtasks, options,
    timeout_s, max_workers_cloud,
    tools, tool_capable_models,
    react_max_rounds, react_compress_after,
    react_max_tool_chars, react_dispatch_timeout_s,
    react_compress_keep_last, react_max_web_searches,
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
        react_compress_keep_last=react_compress_keep_last,
        react_max_web_searches=react_max_web_searches,
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

    When `gate` is supplied and `location == "local"`, the entire token
    stream is held under the gate. Tokens are GPU-bound the whole way
    through (no tool dispatch in this branch), so a single acquire for
    the full duration is the right granularity here.
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
