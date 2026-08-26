"""Passthrough family — `audrey_passthrough/<concrete>` → straight to Ollama.

Strips the prefix and forwards directly to Ollama: no classifier, no banners.
Both fair-scheduling layers still fire so a passthrough request shares the GPU
under the same rules as pipeline traffic. The bare `audrey_passthrough` form is
rejected with a 400 (it must name a concrete model).

Depends on `responses` (envelope + tool_call conversion) and `schemas`; never
on the pipeline streaming module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from audrey import __version__
from audrey.auth import AuthedUser
from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.passthrough import passthrough_chat, passthrough_stream
from audrey.pipeline.vision import describe_for_text_model
from audrey.routes.openai.responses import (
    _ollama_to_openai_tool_calls,
    _options_from_request,
    _to_openai_response,
)
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.routes.openai.streaming import StreamOutcome, StreamTerminal

log = logging.getLogger(__name__)

# Passthrough virtual model — `audrey_passthrough/<concrete_model>`
# strips the prefix and forwards directly to Ollama. The bare form
# `audrey_passthrough` is rejected with a 400 (must name a concrete).
PASSTHROUGH_PREFIX = "audrey_passthrough/"
PASSTHROUGH_BARE = "audrey_passthrough"


def _is_passthrough(model: str) -> bool:
    return model == PASSTHROUGH_BARE or model.startswith(PASSTHROUGH_PREFIX)


def _passthrough_concrete(model: str) -> str:
    """Return the concrete model name from `audrey_passthrough/<name>`.

    Raises 400 for the bare form or an empty suffix.
    """
    if not model.startswith(PASSTHROUGH_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Passthrough requires {PASSTHROUGH_PREFIX}<model_name>, "
                f"got {model!r}."
            ),
        )
    concrete = model[len(PASSTHROUGH_PREFIX):].strip()
    if not concrete:
        raise HTTPException(
            status_code=400,
            detail=f"Empty concrete model after {PASSTHROUGH_PREFIX!r}.",
        )
    return concrete


def _resolve_passthrough_model(
    payload_model: str, cfg, registry: ModelRegistry, me: AuthedUser,
) -> tuple[str, str]:
    """Return (concrete_model, location) for a passthrough request.

    Raises 400 (parse error), 403 (disabled / not allowed / role gate).
    """
    pt_cfg = (cfg.raw.get("passthrough") or {})
    if not pt_cfg.get("enabled"):
        raise HTTPException(status_code=403, detail="Passthrough disabled.")
    required_role = pt_cfg.get("require_role")
    if required_role and me.role != required_role:
        raise HTTPException(
            status_code=403,
            detail=f"Passthrough requires role={required_role!r}.",
        )
    concrete = _passthrough_concrete(payload_model)
    allowed = set(pt_cfg.get("allowed_models") or [])
    if allowed and concrete not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Passthrough not allowed for model {concrete!r}.",
        )
    location = registry.location_of(concrete)
    return concrete, location



async def _passthrough_think(
    ollama: OllamaClient, cfg, concrete: str, requested: bool | None = None,
) -> bool | None:
    """Resolve `passthrough.think` for one model. `None` = omit the field.

    Passthrough forwards a client's request verbatim and the OpenAI schema has
    no thinking knob, so every passthrough turn ran in Ollama's `omitted`
    state — whatever each model's template decides. That is defensible for
    serving and useless for comparing: the local bake-off
    (`scripts/eval_prompts_local_models.json`) reaches its models only through
    this route, and on 2026-08-12 all three of its candidates declared
    `thinking`, so their scores were being compared at three different and
    unchosen reasoning budgets.

    ⚠️ Routed through `ollama.thinking_flag`, never sent raw. Ollama HARD
    ERRORS on `think` for a model that does not declare the capability, so a
    bare `False` here would break every non-thinking model in
    `allowed_models` — of which there are several. `thinking_flag` returns
    None for those, which omits the field.

    ⚠️ Default is still None (omit) — absent `requested` and absent config,
    behaviour is unchanged and serving clients see exactly what they saw.

    ▶ **REVERSED 2026-08-19: `requested` (the per-request `think` field) now
    wins over config.** This docstring previously said a per-request field was
    deliberately excluded, because "a client that could ask for thinking would
    make the eval's model column mean something different per caller". That
    concern is real but the prohibition was the wrong fix, and it cost a full
    day of evidence: with the state living ONLY in `PASSTHROUGH_THINK`, the
    arm of a run was recorded nowhere except this container's logs, and
    `docker logs` starts empty after every recreate. On 2026-08-19 two model
    sweeps (`repl-ab`, `repl-gap`) were rebuilt over ~20 minutes later and
    their thinking arm became permanently unrecoverable — the runs are intact,
    labelled, and worthless, because nothing says what they asked for.
    ▶ The answer to "per-caller ambiguity" is RECORDING, not prohibition: the
    harness sets this field and writes the value into its results JSON, so the
    arm travels with the artifact instead of with a container. What the caller
    asked for is also logged below, with its source.
    """
    src = "config"
    want = (cfg.raw.get("passthrough") or {}).get("think")
    if requested is not None:
        want, src = requested, "request"
    if want is None:
        log.info("passthrough.think model=%s resolved=omit src=default", concrete)
        return None
    # ⚠️ `thinking_flag` can still return None here — a model that does not
    # declare `thinking` gets the field omitted no matter who asked. Log the
    # RESOLVED value, not the wanted one, or the A/B records an intent that
    # never reached Ollama.
    flag = await ollama.thinking_flag(concrete, bool(want))
    log.info(
        "passthrough.think model=%s wanted=%s resolved=%s src=%s",
        concrete, bool(want), flag, src,
    )
    return flag


async def _handle_passthrough(
    app, request: Request, payload: ChatCompletionRequest, me: AuthedUser,
):
    """Route a passthrough request: validate, wrap in inflight, forward.

    The inflight wrap mirrors what the pipeline branches do
    (`async with inflight.slot(user_id)` around the whole call). The
    passthrough helpers acquire the GPU gate around the actual Ollama
    request, so both fair-scheduling layers fire.
    """
    cfg = app.state.cfg
    registry: ModelRegistry = app.state.registry
    concrete, location = _resolve_passthrough_model(payload.model, cfg, registry, me)

    messages = [m.model_dump(exclude_none=True) for m in payload.messages]
    options = _options_from_request(payload)
    timeout_s = float(cfg.timeouts.get("medium", 180))
    inflight = app.state.inflight
    ollama: OllamaClient = app.state.ollama
    gate: FairLocalGate = app.state.gate
    think = await _passthrough_think(ollama, cfg, concrete, payload.think)

    # Passthrough forwards verbatim, so an attached image reaches the
    # concrete model as Ollama's `images: [...]`. Text-only targets — most
    # of `passthrough.allowed_models`, including every cloud model — either
    # error on that or answer blind. Transcribe first so the model the
    # client actually named is still the one that answers.
    # Its own inflight slot rather than the request's: the transcription is
    # a separate dispatch, and both branches below open their own slot for
    # the forward itself. Sequential, never nested.
    async with inflight.slot(me.email):
        messages, _n_described = await describe_for_text_model(
            messages,
            ollama=ollama, registry=registry, health=app.state.health, gate=gate,
            cfg=cfg, target_model=concrete, user_question=last_user_text(messages),
            user_id=me.email,
        )

    if payload.stream:
        async def _emit_passthrough_sse():
            t0 = time.perf_counter()
            terminal = StreamTerminal()
            try:
                async with inflight.slot(me.email):
                    async for frame in _passthrough_stream_sse(
                        ollama, gate,
                        virtual=payload.model, concrete=concrete, location=location,
                        messages=messages, options=options,
                        user_id=me.email, tools=payload.tools, timeout_s=timeout_s,
                        think=think,
                        terminal=terminal,
                    ):
                        yield frame
            except asyncio.CancelledError:
                terminal.finish_if_unset(StreamOutcome.CANCELLED)
                raise
            except GeneratorExit:
                terminal.finish_if_unset(StreamOutcome.CANCELLED)
                raise
            except Exception:
                terminal.finish_if_unset(StreamOutcome.ERROR)
                raise
            finally:
                # An exhausted/closed generator without a reported result is
                # incomplete by definition. Inner protocol adapters normally
                # report first; this guards future early-return paths.
                terminal.finish_if_unset(StreamOutcome.TRUNCATED)
                elapsed = time.perf_counter() - t0
                pipeline_seconds.labels(
                    mode="passthrough", task_type="passthrough",
                ).observe(elapsed)
                pipeline_total.labels(
                    mode="passthrough", task_type="passthrough",
                    outcome=terminal.outcome.value,
                ).inc()
        return StreamingResponse(
            _emit_passthrough_sse(), media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    outcome = "ok"
    try:
        async with inflight.slot(me.email):
            resp = await passthrough_chat(
                ollama, gate,
                concrete=concrete, location=location,
                messages=messages, options=options,
                user_id=me.email, tools=payload.tools, timeout_s=timeout_s,
                think=think,
            )
    except OllamaError as e:
        outcome = "error"
        pipeline_seconds.labels(
            mode="passthrough", task_type="passthrough",
        ).observe(time.perf_counter() - t0)
        pipeline_total.labels(
            mode="passthrough", task_type="passthrough", outcome=outcome,
        ).inc()
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e
    pipeline_seconds.labels(
        mode="passthrough", task_type="passthrough",
    ).observe(time.perf_counter() - t0)
    pipeline_total.labels(
        mode="passthrough", task_type="passthrough", outcome=outcome,
    ).inc()

    msg = resp.get("message") or {}
    tool_calls = _ollama_to_openai_tool_calls(msg.get("tool_calls"))
    return _to_openai_response(
        virtual=payload.model,
        concrete=concrete,
        content=str(msg.get("content") or ""),
        prompt_tokens=int(resp.get("prompt_eval_count", 0) or 0),
        completion_tokens=int(resp.get("eval_count", 0) or 0),
        tool_calls=tool_calls,
    )


async def _passthrough_stream_sse(
    ollama: OllamaClient,
    gate: FairLocalGate,
    *,
    virtual: str,
    concrete: str,
    location: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    user_id: str,
    tools: list[dict[str, Any]] | None,
    timeout_s: float | None,
    think: bool | None = None,
    terminal: StreamTerminal | None = None,
):
    """Stream Ollama chunks as OpenAI-shaped SSE frames.

    This route owns the separate raw-model passthrough stream. When `tools`
    are supplied, Ollama typically populates `message.tool_calls` on the
    final chunk (rather than streaming deltas); we translate that
    into an OpenAI `tool_calls` delta in the terminal frame and set
    `finish_reason="tool_calls"` instead of `"stop"` so agent clients
    parsing the SSE stream see the structured calls and don't fall
    back to scraping plain text.
    """
    terminal = terminal or StreamTerminal()
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    fingerprint = f"audrey-{__version__}/{concrete}"

    first = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": virtual, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    # Ollama's streaming protocol emits `message.tool_calls` in a
    # non-final chunk (typically the one *before* `done: true`), not
    # in the final chunk itself. Accumulate any tool_calls seen across
    # the entire stream so we can attach them to the closing delta.
    # Without this, streaming + tools silently produces an empty stream
    # from the client's perspective (no content, no tool_calls).
    accumulated_tool_calls: list[dict[str, Any]] = []
    try:
        async for chunk in passthrough_stream(
            ollama, gate,
            concrete=concrete, location=location,
            messages=messages, options=options,
            user_id=user_id, tools=tools, timeout_s=timeout_s, think=think,
        ):
            msg = chunk.get("message", {}) or {}
            content = msg.get("content", "") or ""
            chunk_tool_calls = msg.get("tool_calls") or []
            if chunk_tool_calls:
                accumulated_tool_calls.extend(chunk_tool_calls)
            done = bool(chunk.get("done"))
            if content:
                frame = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": virtual, "system_fingerprint": fingerprint,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(frame)}\n\n"
            if done:
                tool_calls = _ollama_to_openai_tool_calls(
                    accumulated_tool_calls, streaming=True,
                )
                final_delta: dict[str, Any] = {}
                if tool_calls:
                    final_delta["tool_calls"] = tool_calls
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"
                final = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": virtual, "system_fingerprint": fingerprint,
                    "choices": [{"index": 0, "delta": final_delta, "finish_reason": finish_reason}],
                }
                terminal.finish(
                    StreamOutcome.OK, finish_reason=finish_reason,
                )
                yield f"data: {json.dumps(final)}\n\n"
                break
        if not terminal.is_final:
            # Ollama's iterator ended without its required ``done`` chunk.
            # Tell OpenAI clients the answer is incomplete instead of sending
            # a bare [DONE] that looks successful to both clients and metrics.
            terminal.finish(StreamOutcome.TRUNCATED, finish_reason="length")
            truncated = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": virtual, "system_fingerprint": fingerprint,
                "choices": [{
                    "index": 0, "delta": {}, "finish_reason": "length",
                }],
            }
            yield f"data: {json.dumps(truncated)}\n\n"
    except asyncio.CancelledError:
        terminal.finish_if_unset(StreamOutcome.CANCELLED)
        raise
    except GeneratorExit:
        terminal.finish_if_unset(StreamOutcome.CANCELLED)
        raise
    except OllamaError as e:
        terminal.finish(StreamOutcome.ERROR, finish_reason="stop")
        err = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": virtual, "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {"content": f"\n\n[error: {e}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    yield "data: [DONE]\n\n"
