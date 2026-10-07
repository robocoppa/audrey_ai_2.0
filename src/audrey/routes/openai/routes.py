"""Route handlers — `/v1/models`, `/v1/chat/completions`, and `/v1/responses`.

The thin orchestration layer: validates the request, forks passthrough vs
pipeline, and wires streaming vs non-streaming. The heavy lifting lives in
`pipeline` (graph + streaming) and `passthrough`; response formatting in
`responses`. `router` is defined here and re-exported from the package
`__init__` so `main.py`'s `app.include_router(...)` is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from audrey import __version__
from audrey.auth import AuthedUser, require_user
from audrey.pipeline.chat_archive import resolve_conversation_id
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.prompts import skill_instruction_for, with_skill_instruction
from audrey.routes.openai.client_tool_generation import (
    generate_client_tool_response,
    prepare_client_tool_model,
    validate_client_tool_prompt,
)
from audrey.routes.openai.client_tools import (
    has_client_tools,
    provider_tools,
    validate_client_tool_request,
)
from audrey.routes.openai.file_inputs import (
    has_file_inputs,
    response_input_messages,
    validate_file_prompt,
)
from audrey.routes.openai.passthrough import (
    PASSTHROUGH_PREFIX,
    _handle_passthrough,
    _is_passthrough,
    _passthrough_think,
    _resolve_passthrough_model,
)
from audrey.routes.openai.pipeline import _generate_via_pipeline, _stream_via_pipeline
from audrey.routes.openai.reasoning import validate_reasoning_request
from audrey.routes.openai.response_storage import (
    expand_stored_request,
    read_response,
    remove_response,
    retain_response,
)
from audrey.routes.openai.responses import _options_from_request, _to_responses_api_response
from audrey.routes.openai.schemas import ChatCompletionRequest, ResponseCreateRequest
from audrey.routes.openai.streaming import OpenAIStreamSession, ResponsesStreamSession
from audrey.routes.openai.structured_outputs import (
    StructuredOutputError,
    response_json_schema,
    structured_output_instruction,
)
from audrey.skills import SkillSelectionError, skill_mode_for_virtual_model

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
    return await _create_chat_completion(payload, request, me)


async def _create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser,
    *,
    stream_session_factory: Callable[..., Any] | None = None,
    response_format: dict[str, Any] | None = None,
):
    """Run the shared authenticated generation path with one wire adapter."""

    validate_reasoning_request(payload.model, effort=payload.reasoning_effort, think=payload.think)
    app = request.app
    requested_stream_session_factory = stream_session_factory
    stream_session_factory = stream_session_factory or OpenAIStreamSession

    # Passthrough branch — bypasses the pipeline entirely. Both fair-
    # scheduling layers still fire so passthrough traffic competes for
    # the GPU on the same terms as pipeline traffic.
    if _is_passthrough(payload.model):
        if payload.skill is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "skill_passthrough_unsupported",
                    "message": "Skills cannot be combined with passthrough models.",
                },
            )
        passthrough_kwargs: dict[str, Any] = {
            "stream_session_factory": requested_stream_session_factory,
        }
        if response_format is not None:
            passthrough_kwargs["response_format"] = response_format
        return await _handle_passthrough(
            app,
            request,
            payload,
            me,
            **passthrough_kwargs,
        )

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

    skill_registry = getattr(app.state, "skills", None)
    if payload.skill is not None and skill_registry is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "skills_unavailable",
                "message": "The skill registry is unavailable.",
            },
        )
    try:
        resolved_skill = (
            skill_registry.resolve(
                explicit_skill=payload.skill,
                virtual_model=payload.model,
                mode=skill_mode_for_virtual_model(payload.model),
            )
            if skill_registry is not None
            else None
        )
    except SkillSelectionError as exc:
        status_code = (
            503
            if exc.code in {"skills_disabled", "skill_unavailable"}
            else 400
        )
        raise HTTPException(status_code=status_code, detail=exc.detail()) from exc
    model_tools = (
        app.state.tools.restrict(resolved_skill.spec.allowed_tools)
        if resolved_skill is not None
        else app.state.tools
    )
    if resolved_skill is not None:
        log.info(
            "skill.selected id=%s version=%d digest=%s reason=%s tools=%s "
            "instruction_chars=%d resource_chars=%d",
            resolved_skill.spec.id,
            resolved_skill.spec.version,
            resolved_skill.spec.digest[:12],
            resolved_skill.reason,
            model_tools.names(),
            len(resolved_skill.spec.instructions),
            sum(len(resource.content) for resource in resolved_skill.spec.resources),
        )

    # Resolve and inject once so streaming and non-streaming see the same
    # immutable instruction even if registry rediscovery happens mid-request.
    # Injection here also does not depend on memory or identity nodes running.
    skill_instruction = skill_instruction_for(
        payload.model,
        app.state.cfg,
        getattr(app.state, "skills", None),
        resolved_skill,
    )
    if skill_instruction:
        messages = with_skill_instruction(messages, skill_instruction)
        log.info(
            "skill_instruction: %s (%d chars)",
            payload.model,
            len(skill_instruction),
        )

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
                skill_instruction=skill_instruction,
                resolved_skill=resolved_skill,
                model_tools=model_tools,
                stream_session_factory=stream_session_factory,
                **(
                    {"response_format": response_format}
                    if response_format is not None
                    else {}
                ),
            ),
            media_type="text/event-stream",
        )

    return await _generate_via_pipeline(
        app, payload, messages, options,
        user_id=me.email,
        conversation_id=conversation_id,
        user_turn_text=user_turn_text,
        skill_instruction=skill_instruction,
        resolved_skill=resolved_skill,
        model_tools=model_tools,
        **(
            {"response_format": response_format}
            if response_format is not None
            else {}
        ),
    )


# --- /v1/responses ----------------------------------------------------

@router.post("/responses")
async def create_response(
    payload: ResponseCreateRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
    """Generate one completed or streamed response through Audrey."""

    payload = await expand_stored_request(payload, request, me)
    result = await _create_response(payload, request, me)
    return await retain_response(result, payload, request, me)


@router.get("/responses/{response_id}")
async def retrieve_response(response_id: str, request: Request, me: AuthedUser = Depends(require_user)):
    """Read one unexpired saved response belonging to the current account."""
    return (await read_response(request, me, response_id))["response"]


@router.delete("/responses/{response_id}")
async def delete_response(response_id: str, request: Request, me: AuthedUser = Depends(require_user)):
    """Delete one owned response and every retained descendant."""
    return await remove_response(request, me, response_id)


async def _create_response(payload: ResponseCreateRequest, request: Request, me: AuthedUser):

    effort = payload.reasoning.effort if payload.reasoning is not None else None
    validate_reasoning_request(payload.model, effort=effort)
    client_tools = has_client_tools(payload)
    unsupported = [
        name
        for name, active in (
            ("background", payload.background),
            ("conversation", payload.conversation is not None),
            ("tools", client_tools and not _is_passthrough(payload.model)),
            ("skill", client_tools and payload.skill is not None),
            (
                "text",
                payload.text is not None
                and (payload.text.format.type == "json_object" or (
                    client_tools and payload.text.format.type != "text"
                )),
            ),
        )
        if active
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "responses_feature_unsupported",
                "message": (
                    "These Responses features are not supported; "
                    f"unsupported fields: {', '.join(unsupported)}."
                ),
            },
        )

    try:
        response_format = response_json_schema(payload)
        format_instruction = structured_output_instruction(payload)
    except StructuredOutputError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "responses_structured_output_invalid",
                "message": str(exc),
            },
        ) from exc

    target = None
    if client_tools:
        validate_client_tool_request(payload)
        await validate_client_tool_prompt(payload)
        target = await prepare_client_tool_model(payload, request, me)
    elif payload.store is True or payload.previous_response_id is not None:
        await validate_client_tool_prompt(payload)
    if effort is not None and not client_tools:
        app = request.app
        concrete, _ = _resolve_passthrough_model(payload.model, app.state.cfg, app.state.registry, me)
        # Reject unsupported controls before owned-file reads or remote fetches.
        # Dispatch repeats the shared resolution against its cached metadata.
        await _passthrough_think(app.state.ollama, app.state.cfg, concrete, effort=effort)
    input_messages = await response_input_messages(payload, request, me)
    messages: list[dict[str, Any]] = []
    if payload.instructions:
        messages.append({"role": "developer", "content": payload.instructions})
    if format_instruction:
        messages.append({"role": "developer", "content": format_instruction})
    messages.extend(input_messages)
    if has_file_inputs(payload) or client_tools or payload.store is True or payload.previous_response_id is not None:
        budget_messages = messages
        if client_tools:
            budget_messages = [*messages, {"content": json.dumps(provider_tools(payload) or [], ensure_ascii=False)}]
        await validate_file_prompt(budget_messages)
    if client_tools:
        assert target is not None
        return await generate_client_tool_response(payload, request, me, messages, target)
    chat_payload = ChatCompletionRequest(
        model=payload.model,
        skill=payload.skill,
        messages=messages,
        stream=payload.stream,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_output_tokens,
        reasoning_effort=effort,
        metadata=payload.metadata,
        user=payload.user,
    )
    if payload.stream:
        stream_kwargs: dict[str, Any] = {
            "stream_session_factory": partial(
                ResponsesStreamSession,
                request=payload,
            ),
        }
        if response_format is not None:
            stream_kwargs["response_format"] = response_format
        stream_response = await _create_chat_completion(
            chat_payload,
            request,
            me,
            **stream_kwargs,
        )
        if not isinstance(stream_response, StreamingResponse):
            raise HTTPException(
                status_code=502,
                detail="Generation returned an invalid streaming response.",
            )
        return stream_response

    if response_format is None:
        chat_response = await chat_completions(chat_payload, request, me)
    else:
        chat_response = await _create_chat_completion(
            chat_payload,
            request,
            me,
            response_format=response_format,
        )
    if not isinstance(chat_response, dict):
        raise HTTPException(status_code=502, detail="Generation returned an invalid response.")
    try:
        return _to_responses_api_response(chat_response, payload)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
