"""OpenAI-compatible routes.

Exposes three virtual models (`audrey_deep`, `audrey_cloud`, `audrey_local`)
plus `/v1/chat/completions`. Requests go through the pipeline:
  classify → complexity gate → fast path (Phase 5) | deep stub (Phase 6+).

Response shape is the OpenAI chat-completion contract so Open WebUI and
any other client can consume it unchanged.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from audrey import __version__
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.classify import classify as classify_fn
from audrey.pipeline.complexity import is_complex

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

# The three virtual models Audrey exposes. Each is a *pipeline mode*, not a
# real Ollama model. Mapping to concrete models happens inside the pipeline.
VIRTUAL_MODELS = ("audrey_deep", "audrey_cloud", "audrey_local")


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
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    app = request.app
    if payload.model not in VIRTUAL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model {payload.model!r}. "
                f"Supported virtual models: {list(VIRTUAL_MODELS)}."
            ),
        )

    messages = [m.model_dump(exclude_none=True) for m in payload.messages]
    options = _options_from_request(payload)

    if payload.stream:
        return StreamingResponse(
            _stream_via_pipeline(app, payload, messages, options),
            media_type="text/event-stream",
        )

    return await _generate_via_pipeline(app, payload, messages, options)


async def _generate_via_pipeline(app, payload: ChatCompletionRequest, messages, options):
    """Non-streaming path: invoke the compiled LangGraph and format the result."""
    graph = app.state.graph
    state = {
        "virtual_model": payload.model,
        "messages": messages,
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "max_tokens": payload.max_tokens,
    }
    try:
        final = await graph.ainvoke(state)
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


async def _stream_via_pipeline(app, payload: ChatCompletionRequest, messages, options):
    """Streaming path.

    For non-complex audrey_deep prompts, we stream a single fast-path model
    directly (token-by-token). For complex prompts, or for audrey_cloud /
    audrey_local (always deep), we run the full graph non-streamed and emit
    the synthesized answer as one chunk — multi-worker + synth can't be
    coherently token-streamed.
    """
    cfg = app.state.cfg
    ollama: OllamaClient = app.state.ollama
    registry: ModelRegistry = app.state.registry
    health: HealthTracker = app.state.health
    router_cfg = cfg.router

    user_text = _last_user_text(messages)
    task, reason, conf = await classify_fn(
        ollama,
        router_model=router_cfg.get("model", "qwen3:4b"),
        router_timeout_s=float(router_cfg.get("timeout_s", 20)),
        max_router_strikes=int(router_cfg.get("max_failures_before_fallback", 2)),
        user_text=user_text,
    )
    complex_, n = is_complex(messages, threshold=int(cfg.raw.get("complexity", {}).get("token_threshold", 500)))
    forced_deep = payload.model in ("audrey_cloud", "audrey_local")
    use_deep = complex_ or forced_deep

    log.info(
        "chat.completions (stream) model=%s task=%s(%s, conf=%.2f) tokens=%d mode=%s",
        payload.model, task, reason, conf, n, "deep" if use_deep else "fast",
    )

    if use_deep:
        # Run the compiled graph end-to-end, then emit the synthesized answer.
        graph = app.state.graph
        state = {
            "virtual_model": payload.model,
            "messages": messages,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "max_tokens": payload.max_tokens,
        }
        try:
            final = await graph.ainvoke(state)
        except OllamaError as e:
            async for frame in _emit_single_message(
                payload.model, "error", f"[ollama error: {e}]"
            ):
                yield frame
            return
        concrete = final.get("concrete_model", "deep_panel")
        content = final.get("content", "") or "[empty]"
        async for frame in _emit_single_message(payload.model, concrete, content):
            yield frame
        return

    spec = registry.first_healthy(task, health.is_healthy)
    if spec is None:
        async for frame in _emit_single_message(
            payload.model, "none", f"[no healthy model for task={task}]"
        ):
            yield frame
        return

    timeout = float(cfg.timeouts.get("fast_path", 180))
    async for frame in _stream_openai(
        ollama, payload.model, spec.name, messages, options, timeout_s=timeout, health=health,
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


async def _stream_openai(
    ollama: OllamaClient,
    virtual: str,
    concrete: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    timeout_s: float | None = None,
    health: HealthTracker | None = None,
):
    """Convert Ollama's streaming chunks into OpenAI SSE frames."""
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    fingerprint = f"audrey-{__version__}/{concrete}"

    first = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": virtual, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    try:
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
