"""OpenAI-compatible routes.

Phase 4: minimal pass-through. Exposes three virtual models
(`audrey_deep`, `audrey_cloud`, `audrey_local`), plus `/v1/chat/completions`
that picks the highest-priority "general" model from the registry and
forwards the request to Ollama. No classification, no panels, no tools —
those land in Phase 5+.

The response shape is the OpenAI chat-completion contract so Open WebUI
and other clients can consume it unchanged.
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
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry

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
    ollama: OllamaClient = request.app.state.ollama
    registry: ModelRegistry = request.app.state.registry

    concrete = _pick_concrete_model(payload.model, registry)
    if concrete is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model {payload.model!r}. "
                f"Supported virtual models: {list(VIRTUAL_MODELS)}."
            ),
        )

    messages = [m.model_dump(exclude_none=True) for m in payload.messages]
    options = _options_from_request(payload)

    log.info("chat.completions model=%s -> %s stream=%s", payload.model, concrete, payload.stream)

    if payload.stream:
        return StreamingResponse(
            _stream_openai(ollama, payload.model, concrete, messages, options),
            media_type="text/event-stream",
        )

    try:
        result = await ollama.chat(model=concrete, messages=messages, options=options)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e

    return _to_openai_response(payload.model, concrete, result)


# ─── Helpers ──────────────────────────────────────────────────────────

def _pick_concrete_model(virtual: str, registry: ModelRegistry) -> str | None:
    """Phase 4 routing: always pick highest-priority 'general' candidate.

    Cloud vs local filtering is a Phase 6 concern — for now all three
    virtual models map to the same concrete model so we can prove streaming
    and the OpenAI envelope work end-to-end.
    """
    if virtual not in VIRTUAL_MODELS:
        return None
    # `lambda _: True` → no health filtering yet; Phase 5 wires HealthTracker in.
    spec = registry.first_healthy("general", lambda _: True)
    return spec.name if spec else None


def _options_from_request(req: ChatCompletionRequest) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    return opts


def _to_openai_response(virtual: str, concrete: str, ollama_resp: dict[str, Any]) -> dict[str, Any]:
    msg = ollama_resp.get("message", {}) or {}
    content = msg.get("content", "") or ""
    prompt_tokens = int(ollama_resp.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(ollama_resp.get("eval_count", 0) or 0)
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
                "finish_reason": "stop" if ollama_resp.get("done") else "length",
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
):
    """Convert Ollama's streaming chunks into OpenAI SSE frames."""
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    fingerprint = f"audrey-{__version__}/{concrete}"

    # First frame: opening role delta
    first = {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": virtual, "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    try:
        async for chunk in ollama.chat_stream(model=concrete, messages=messages, options=options):
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
                break
    except OllamaError as e:
        err = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": virtual, "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "delta": {"content": f"\n\n[error: {e}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(err)}\n\n"

    yield "data: [DONE]\n\n"


__all__ = ["router", "VIRTUAL_MODELS"]
