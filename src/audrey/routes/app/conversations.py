"""Owner-bound native conversation and message resources."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audrey.app_state import (
    ApplicationStore,
    ConversationHasActiveRunError,
    ConversationRecord,
    InvalidApplicationStateError,
    MessageRecord,
)
from audrey.auth import require_scope
from audrey.identity import Principal

router = APIRouter(prefix="/conversations", tags=["conversations"])
_conversation_access = require_scope("compat:full")
_Mode = Literal["auto", "fast", "deep", "research", "local", "cloud", "video"]


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=200)
    default_mode: _Mode = "auto"


class ConversationPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    default_mode: _Mode | None = None
    archived: bool | None = None

    @model_validator(mode="after")
    def require_explicit_values(self):
        if not self.model_fields_set:
            raise ValueError("at least one conversation field is required")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("conversation fields cannot be null")
        return self


class ConversationResponse(BaseModel):
    id: str
    title: str
    default_mode: _Mode
    created_at: str
    updated_at: str
    last_message_at: str | None
    archived_at: str | None


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    next_cursor: str | None


class MessageResponse(BaseModel):
    id: str
    run_id: str | None
    sequence: int
    role: Literal["user", "assistant", "tool"]
    status: Literal["in_progress", "completed", "incomplete"]
    content: str
    created_at: str
    updated_at: str


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    next_cursor: str | None


def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application state is not initialized.",
        )
    return store


def _conversation_response(record: ConversationRecord) -> ConversationResponse:
    return ConversationResponse(
        id=record.conversation_id,
        title=record.title,
        default_mode=record.default_mode,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_message_at=record.last_message_at,
        archived_at=record.archived_at,
    )


def _message_response(record: MessageRecord) -> MessageResponse:
    return MessageResponse(
        id=record.message_id,
        run_id=record.run_id,
        sequence=record.sequence_no,
        role=record.role,
        status=record.status,
        content=record.content,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _encode_cursor(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _decode_cursor(value: str) -> dict[str, object]:
    if len(value) > 1000:
        raise HTTPException(status_code=422, detail="Cursor is invalid.")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        decoded = json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Cursor is invalid.") from exc
    if not isinstance(decoded, dict) or decoded.get("v") != 1:
        raise HTTPException(status_code=422, detail="Cursor is invalid.")
    return decoded


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreateRequest,
    request: Request,
    principal: Principal = Depends(_conversation_access),
) -> ConversationResponse:
    try:
        record = await _store(request).conversations.create(
            user_id=principal.user_id,
            title=payload.title,
            default_mode=payload.default_mode,
        )
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _conversation_response(record)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    principal: Principal = Depends(_conversation_access),
    archived: bool = False,
    q: Annotated[str, Query(max_length=200)] = "",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=1000)] = None,
) -> ConversationListResponse:
    search = q.strip()
    before_activity_at: str | None = None
    before_conversation_id: str | None = None
    if cursor is not None:
        decoded = _decode_cursor(cursor)
        before_activity_at = decoded.get("activity")
        before_conversation_id = decoded.get("conversation_id")
        if (
            not isinstance(before_activity_at, str)
            or not isinstance(before_conversation_id, str)
            or not isinstance(decoded.get("archived"), bool)
            or decoded["archived"] != archived
            or decoded.get("search", "") != search
        ):
            raise HTTPException(status_code=422, detail="Cursor is invalid for this view.")
    try:
        records = await _store(request).conversations.list_page(
            user_id=principal.user_id,
            archived=archived,
            limit=limit + 1,
            search=search,
            before_activity_at=before_activity_at,
            before_conversation_id=before_conversation_id,
        )
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    page = records[:limit]
    next_cursor = None
    if len(records) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(
            {
                "v": 1,
                "activity": last.last_message_at or last.created_at,
                "conversation_id": last.conversation_id,
                "archived": archived,
                "search": search,
            }
        )
    return ConversationListResponse(
        items=[_conversation_response(record) for record in page],
        next_cursor=next_cursor,
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    request: Request,
    principal: Principal = Depends(_conversation_access),
) -> ConversationResponse:
    record = await _store(request).conversations.get(
        user_id=principal.user_id,
        conversation_id=conversation_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _conversation_response(record)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    payload: ConversationPatchRequest,
    request: Request,
    principal: Principal = Depends(_conversation_access),
) -> ConversationResponse:
    try:
        record = await _store(request).conversations.update(
            user_id=principal.user_id,
            conversation_id=conversation_id,
            title=payload.title if "title" in payload.model_fields_set else None,
            default_mode=(
                payload.default_mode if "default_mode" in payload.model_fields_set else None
            ),
            archived=payload.archived if "archived" in payload.model_fields_set else None,
        )
    except ConversationHasActiveRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _conversation_response(record)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    request: Request,
    principal: Principal = Depends(_conversation_access),
) -> Response:
    try:
        deleted = await _store(request).conversations.delete(
            user_id=principal.user_id,
            conversation_id=conversation_id,
        )
    except ConversationHasActiveRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    projector = getattr(request.app.state, "archive_projector", None)
    wake = getattr(projector, "wake", None)
    if callable(wake):
        wake()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def list_messages(
    conversation_id: str,
    request: Request,
    principal: Principal = Depends(_conversation_access),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=1000)] = None,
) -> MessageListResponse:
    after_sequence = 0
    if cursor is not None:
        decoded = _decode_cursor(cursor)
        sequence = decoded.get("sequence")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 1
            or decoded.get("conversation_id") != conversation_id
        ):
            raise HTTPException(status_code=422, detail="Cursor is invalid for this conversation.")
        after_sequence = sequence
    try:
        records = await _store(request).conversations.list_message_page(
            user_id=principal.user_id,
            conversation_id=conversation_id,
            after_sequence=after_sequence,
            limit=limit + 1,
        )
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if records is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    page = records[:limit]
    next_cursor = None
    if len(records) > limit:
        next_cursor = _encode_cursor(
            {
                "v": 1,
                "conversation_id": conversation_id,
                "sequence": page[-1].sequence_no,
            }
        )
    return MessageListResponse(
        items=[_message_response(record) for record in page],
        next_cursor=next_cursor,
    )


__all__ = ["router"]
