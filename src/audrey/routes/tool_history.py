"""Service-only, owner-bound reads of saved native conversation text.

The model tool searches the archive for a conversation id, then reads this
route through custom-tools. Its user argument must be replaced by Audrey's
authenticated storage namespace at dispatch. A service token alone does not
provide that binding; both halves of the boundary are required.

Only saved user and assistant content is read. Tool payloads, run prompts,
reasoning fields and provider metadata are outside this projection. Native ids
resolve directly; ids from an imported archive resolve through its owner-bound
provenance. Unimported compatibility archive ids cannot be read here.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from audrey.auth import require_service
from audrey.user_data_visibility import remote_personal_reads_blocked

router = APIRouter(prefix="/v1/chat/history", tags=["chat-history"])


class ConversationReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1, max_length=320)
    conversation_id: str = Field(min_length=1, max_length=200)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=2000, ge=1, le=3000)
    # Added by the trusted sidecar from its durable user/conversation deletion
    # receipts; deliberately absent from the model-visible tool arguments.
    visible_after: str | None = Field(default=None, max_length=64)

    @field_validator("visible_after")
    @classmethod
    def require_aware_cutoff(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _timestamp(value).isoformat(timespec="microseconds")

    @field_validator("user", "conversation_id")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a non-empty value is required")
        return value


class ConversationReadResponse(BaseModel):
    conversation_id: str
    title: str
    text: str
    offset: int
    next_offset: int | None
    total_chars: int


def _connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("history timestamps require a timezone")
    return parsed.astimezone(dt.UTC)


def _purge_cutoff(path: Path, user: str) -> dt.datetime | None:
    connection = _connection(path)
    try:
        row = connection.execute(
            "SELECT MAX(cutoff_at) AS cutoff FROM user_data_purges WHERE user = ?",
            (user,),
        ).fetchone()
        return _timestamp(str(row["cutoff"])) if row and row["cutoff"] else None
    finally:
        connection.close()


def _read_page(
    application_path: Path,
    uploads_path: Path,
    body: ConversationReadRequest,
) -> ConversationReadResponse | None:
    cutoff = _purge_cutoff(uploads_path, body.user)
    archive_cutoff = _timestamp(body.visible_after) if body.visible_after else None
    message_cutoff = max(
        (value for value in (cutoff, archive_cutoff) if value is not None),
        default=None,
    )
    connection = _connection(application_path)
    try:
        # A single read transaction prevents a delete or purge between the
        # owner lookup and content read from producing an inconsistent page.
        connection.execute("BEGIN")
        owner = connection.execute(
            "SELECT user_id FROM app_users WHERE storage_namespace = ? "
            "AND status = 'active' AND NOT EXISTS "
            "(SELECT 1 FROM account_deletion_requests d "
            "WHERE d.user_id = app_users.user_id)",
            (body.user,),
        ).fetchone()
        if owner is None:
            return None
        user_id = str(owner["user_id"])
        conversation = connection.execute(
            "SELECT conversation_id, title, created_at FROM app_conversations "
            "WHERE user_id = ? AND conversation_id = ?",
            (user_id, body.conversation_id),
        ).fetchone()
        if conversation is None:
            conversation = connection.execute(
                "SELECT c.conversation_id, c.title, c.created_at "
                "FROM app_history_import_conversations i "
                "JOIN app_conversations c ON c.conversation_id = i.conversation_id "
                "AND c.user_id = i.user_id "
                "WHERE i.user_id = ? AND i.source = 'audrey_archive_v1' "
                "AND i.source_conversation_id = ?",
                (user_id, body.conversation_id),
            ).fetchone()
        if conversation is None:
            return None
        # Canonical purges delete these rows outright. The durable cutoff also
        # prevents a later import of an old export from restoring readable text.
        if cutoff is not None and _timestamp(str(conversation["created_at"])) <= cutoff:
            return None
        conversation_id = str(conversation["conversation_id"])
        rows = connection.execute(
            "SELECT message_id, sequence_no, role, status, created_at, "
            "length(content) AS content_chars FROM app_messages "
            "WHERE user_id = ? AND conversation_id = ? "
            "AND role IN ('user', 'assistant') "
            "AND status IN ('completed', 'incomplete') ORDER BY sequence_no",
            (user_id, conversation_id),
        )
        page: list[str] = []
        position = 0
        visible_messages = 0
        hidden_messages = False
        page_end = body.offset + body.limit
        for row in rows:
            if message_cutoff is not None and _timestamp(str(row["created_at"])) <= message_cutoff:
                hidden_messages = True
                continue
            visible_messages += 1
            header = f"\n[{row['role']} · {row['status']}]\n"
            header_end = position + len(header)
            if body.offset < header_end and page_end > position:
                page.append(header[max(0, body.offset - position):page_end - position])
            content_chars = int(row["content_chars"])
            content_start = max(0, body.offset - header_end)
            content_end = min(content_chars, page_end - header_end)
            if content_end > content_start:
                # SQL slices only the requested page. A very long stored
                # message never enters the tool result or Python memory whole.
                part = connection.execute(
                    "SELECT substr(content, ?, ?) AS content FROM app_messages "
                    "WHERE message_id = ? AND user_id = ? AND conversation_id = ?",
                    (
                        content_start + 1, content_end - content_start,
                        str(row["message_id"]), user_id, conversation_id,
                    ),
                ).fetchone()
                page.append(str(part["content"]))
            position = header_end + content_chars
        if not visible_messages:
            return None
        text = "".join(page)
        next_offset = body.offset + len(text) if body.offset + len(text) < position else None
        return ConversationReadResponse(
            conversation_id=conversation_id,
            # The title may itself summarize deleted text.
            title="" if hidden_messages else str(conversation["title"]),
            text=text,
            offset=body.offset,
            next_offset=next_offset,
            total_chars=position,
        )
    finally:
        connection.close()


@router.post("/conversation", response_model=ConversationReadResponse)
async def read_conversation(
    body: ConversationReadRequest,
    request: Request,
    _: None = Depends(require_service),
) -> ConversationReadResponse:
    """Return one bounded page of a named owner's saved conversation."""
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="history_unavailable")
    if remote_personal_reads_blocked(body.user):
        raise HTTPException(status_code=409, detail="personal_data_purge_in_progress")
    uploads = getattr(request.app.state, "uploads_db", None)
    uploads_path = getattr(uploads, "_path", None)
    if uploads_path is None:
        raise HTTPException(status_code=503, detail="history_unavailable")
    try:
        result = await asyncio.to_thread(
            _read_page, store.path, uploads_path, body,
        )
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="history_unavailable") from exc
    # Close the local-to-remote purge handoff window even when a purge began
    # while the read ran in the worker thread.
    if remote_personal_reads_blocked(body.user):
        raise HTTPException(status_code=409, detail="personal_data_purge_in_progress")
    if result is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return result
