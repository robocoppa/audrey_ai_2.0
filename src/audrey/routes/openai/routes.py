"""Route handlers — `/v1/models` and `/v1/chat/completions`.

The thin orchestration layer: validates the request, forks passthrough vs
pipeline, and wires streaming vs non-streaming. The heavy lifting lives in
`pipeline` (graph + streaming) and `passthrough`; response formatting in
`responses`. `router` is defined here and re-exported from the package
`__init__` so `main.py`'s `app.include_router(...)` is unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from audrey import __version__
from audrey.auth import AuthedUser, require_user
from audrey.pipeline.chat_archive import resolve_conversation_id
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.prompts import task_role_for, with_task_role
from audrey.routes.openai.passthrough import (
    PASSTHROUGH_PREFIX,
    _handle_passthrough,
    _is_passthrough,
)
from audrey.routes.openai.pipeline import _generate_via_pipeline, _stream_via_pipeline
from audrey.routes.openai.responses import _options_from_request
from audrey.routes.openai.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])

# The virtual models Audrey exposes. Each is a *pipeline mode*, not a real
# Ollama model. Mapping to concrete models happens inside the pipeline.
# (The count was stated here and went stale twice — don't reintroduce it.)
VIRTUAL_MODELS = (
    "audrey_deep",     # always deep (mixed pool)
    "audrey_cloud",    # always deep (cloud-only pool)
    "audrey_local",    # always deep (local-only pool)
    "audrey_research", # always deep, staged: research → verify → write
    "audrey_auto",     # adaptive: fast for short prompts, deep for long ones
    "audrey_fast",     # always fast (no escalation, even on long prompts)
    "audrey_video",    # adaptive like audrey_auto, plus the video task role
)


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List Audrey's virtual models plus any configured passthrough variants.

    Pipeline virtual models are static (`VIRTUAL_MODELS`). Passthrough
    variants are derived from `passthrough.allowed_models` in config —
    one `audrey_passthrough/<concrete>` id per allowed concrete model,
    so OpenAI-shaped clients can present a dropdown without knowing
    the prefix scheme out of band.
    """
    now = int(time.time())
    entries: list[dict[str, Any]] = [
        {
            "id": name,
            "object": "model",
            "created": now,
            "owned_by": f"audrey-{__version__}",
        }
        for name in VIRTUAL_MODELS
    ]
    cfg = request.app.state.cfg
    pt_cfg = (cfg.raw.get("passthrough") or {})
    if pt_cfg.get("enabled"):
        for concrete in (pt_cfg.get("allowed_models") or []):
            entries.append({
                "id": f"{PASSTHROUGH_PREFIX}{concrete}",
                "object": "model",
                "created": now,
                "owned_by": f"audrey-{__version__}",
            })
    return {"object": "list", "data": entries}


# ─── /v1/chat/completions ─────────────────────────────────────────────

@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
    app = request.app

    # Passthrough branch — bypasses the pipeline entirely. Both fair-
    # scheduling layers still fire so passthrough traffic competes for
    # the GPU on the same terms as pipeline traffic.
    if _is_passthrough(payload.model):
        return await _handle_passthrough(app, request, payload, me)

    if payload.model not in VIRTUAL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model {payload.model!r}. "
                f"Supported virtual models: {list(VIRTUAL_MODELS)}."
            ),
        )

    # Identity comes from the Authorization header via require_user, NOT from
    # payload.user (OpenAI-spec passthrough, trusted for nothing). If a client
    # sent a different `user` field, log once for drift-debugging then ignore.
    if payload.user and payload.user != me.email:
        log.debug(
            "chat.completions: payload.user=%r ignored (auth user=%r)",
            payload.user, me.email,
        )

    identity_messages = [
        message.model_dump(exclude_none=True)
        for message in payload.messages
    ]
    messages = [
        message.model_dump(exclude_none=True, exclude={"metadata"})
        for message in payload.messages
    ]

    # Specialist task role, injected once here so it reaches the streaming and
    # non-streaming paths alike and does not depend on memory being enabled or
    # the user being identified. See `prompts.with_task_role` for why this is
    # not done inside `node_memory_recall`. No-op for every non-specialist
    # model, which is what keeps the A-B comparison against `audrey_auto`
    # honest.
    role_prompt = task_role_for(payload.model, app.state.cfg)
    if role_prompt:
        messages = with_task_role(messages, role_prompt)
        log.info("task_role: %s (%d chars)", payload.model, len(role_prompt))

    debug_cfg = app.state.cfg.raw.get("debug", {}) or {}
    if debug_cfg.get("log_incoming_payload", False):
        shape = [(m.get("role"), len(str(m.get("content") or ""))) for m in messages]
        log.info("incoming.payload: n=%d roles=%s", len(messages), shape)
    if debug_cfg.get("log_incoming_payload_content", False):
        heads = [
            {"role": m.get("role"), "head": str(m.get("content") or "")[:500]}
            for m in messages
        ]
        log.info("incoming.payload.content: %s", heads)
    options = _options_from_request(payload)

    # Resolve once from explicitly modelled client ids and the untouched
    # identity view. The provider view above has metadata removed on purpose.
    raw_payload = {
        "chat_id": payload.chat_id,
        "conversation_id": payload.conversation_id,
        "metadata": payload.metadata,
    }
    conversation_id = resolve_conversation_id(
        user_id=me.email,
        raw_payload=raw_payload,
        messages=identity_messages,
    )
    user_turn_text = last_user_text(messages)

    if payload.stream:
        return StreamingResponse(
            _stream_via_pipeline(
                app, payload, messages, options,
                user_id=me.email,
                conversation_id=conversation_id,
                user_turn_text=user_turn_text,
            ),
            media_type="text/event-stream",
        )

    return await _generate_via_pipeline(
        app, payload, messages, options,
        user_id=me.email,
        conversation_id=conversation_id,
        user_turn_text=user_turn_text,
    )
