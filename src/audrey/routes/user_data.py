"""Authenticated current-user controls for Audrey-owned personal data.

The browser never sends a user id. ``require_user`` resolves the bearer token
through the current identity provider and each handler inserts that verified
identity into an internal, service-authenticated custom-tools request. The
sidecar endpoints are absent from its OpenAPI document, so these controls do
not become model-callable tools.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field, ValidationError

from audrey.auth import AuthedUser, require_user

router = APIRouter(prefix="/v1/me", tags=["user-data"])

_MEMORY_HOST_TOOL = "memory_search"
_CHAT_HOST_TOOL = "chat_history_search"
_MEMORY_LIST_PATH = "/user_data/memories/list"
_MEMORY_UPDATE_PATH = "/user_data/memories/update"
_MEMORY_DELETE_PATH = "/user_data/memories/delete"
_CHAT_EXPORT_PATH = "/user_data/chat_history/export"
_CHAT_DELETE_PATH = "/user_data/chat_history/delete"


class MemoryItem(BaseModel):
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str


class MemoryPage(BaseModel):
    items: list[MemoryItem]
    next_cursor: str | None = None


class MemoryCorrection(BaseModel):
    value: Annotated[str, Field(min_length=1, max_length=20_000)]
    tags: Annotated[str | None, Field(max_length=500)] = None


class MemoryDeleteResult(BaseModel):
    key: str
    deleted: bool


class ChatExportMessage(BaseModel):
    message_id: str
    conversation_id: str
    conversation_title: str
    conversation_created_at: str
    conversation_updated_at: str
    role: str
    content: str
    created_at: str
    archived_at: str
    partial: bool
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int


class ChatExportPage(BaseModel):
    schema_version: int
    items: list[ChatExportMessage]
    next_cursor: str | None = None


class ChatDeletionResult(BaseModel):
    conversation_id: str
    requested_at: str
    status: str
    chunks_queued: int
    deletions_pending: int


def _tool_host(request: Request, tool_name: str) -> str:
    registry = getattr(request.app.state, "tools", None)
    spec = registry.get(tool_name) if registry is not None else None
    if spec is None:
        raise HTTPException(status_code=503, detail=f"{tool_name}_not_registered")
    return str(spec.server_url).rstrip("/")


async def _request_backend(
    request: Request,
    *,
    tool_name: str,
    path: str,
    body: dict[str, Any],
    unprocessable_detail: str,
    not_found_detail: str | None = None,
) -> dict[str, Any]:
    client: httpx.AsyncClient | None = getattr(
        request.app.state, "archive_http", None,
    )
    if client is None:
        raise HTTPException(status_code=503, detail="user_data_backend_unavailable")
    service_token = str(
        getattr(request.app.state, "kb_service_token", "") or "",
    )
    if not service_token:
        raise HTTPException(
            status_code=503,
            detail="user_data_service_auth_unavailable",
        )
    try:
        response = await client.post(
            f"{_tool_host(request, tool_name)}{path}",
            json=body,
            headers={"X-Audrey-Service-Token": service_token},
            timeout=10.0,
        )
    except (httpx.HTTPError, TimeoutError) as e:
        raise HTTPException(
            status_code=503,
            detail="user_data_backend_unavailable",
        ) from e
    if response.status_code >= 400:
        # Browser auth already passed. A sidecar 401 is a server-to-server
        # configuration failure, not an expired browser session.
        if response.status_code == 422:
            raise HTTPException(
                status_code=422,
                detail=unprocessable_detail,
            )
        if response.status_code == 404 and not_found_detail is not None:
            raise HTTPException(status_code=404, detail=not_found_detail)
        status_code = (
            503
            if response.status_code >= 500 or response.status_code == 401
            else 502
        )
        raise HTTPException(status_code=status_code, detail="user_data_backend_failed")
    try:
        value = response.json()
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        )
    return value


async def _request_page(
    request: Request,
    *,
    tool_name: str,
    path: str,
    user: str,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"user": user, "limit": limit}
    if cursor is not None:
        body["cursor"] = cursor
    return await _request_backend(
        request,
        tool_name=tool_name,
        path=path,
        body=body,
        unprocessable_detail="invalid_pagination_cursor",
    )


@router.get("/memories", response_model=MemoryPage)
async def list_memories(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    me: AuthedUser = Depends(require_user),
) -> MemoryPage:
    """List only the authenticated account-owned durable memories."""
    value = await _request_page(
        request,
        tool_name=_MEMORY_HOST_TOOL,
        path=_MEMORY_LIST_PATH,
        user=me.email,
        limit=limit,
        cursor=cursor,
    )
    try:
        return MemoryPage.model_validate(value)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e


@router.put("/memories/{key:path}", response_model=MemoryItem)
async def correct_memory(
    request: Request,
    correction: MemoryCorrection,
    key: Annotated[str, Path(min_length=1, max_length=200)],
    me: AuthedUser = Depends(require_user),
) -> MemoryItem:
    """Correct one existing memory owned by the authenticated account."""
    body: dict[str, Any] = {
        "user": me.email,
        "key": key,
        "value": correction.value,
    }
    if correction.tags is not None:
        body["tags"] = correction.tags
    value = await _request_backend(
        request,
        tool_name=_MEMORY_HOST_TOOL,
        path=_MEMORY_UPDATE_PATH,
        body=body,
        unprocessable_detail="invalid_memory_update",
        not_found_detail="memory_not_found",
    )
    try:
        return MemoryItem.model_validate(value)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e


@router.delete("/memories/{key:path}", response_model=MemoryDeleteResult)
async def delete_memory(
    request: Request,
    key: Annotated[str, Path(min_length=1, max_length=200)],
    me: AuthedUser = Depends(require_user),
) -> MemoryDeleteResult:
    """Delete one memory owned by the authenticated account."""
    value = await _request_backend(
        request,
        tool_name=_MEMORY_HOST_TOOL,
        path=_MEMORY_DELETE_PATH,
        body={"user": me.email, "key": key},
        unprocessable_detail="invalid_memory_delete",
        not_found_detail="memory_not_found",
    )
    try:
        return MemoryDeleteResult.model_validate(value)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e


@router.get("/chat-history/export", response_model=ChatExportPage)
async def export_chat_history(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    me: AuthedUser = Depends(require_user),
) -> ChatExportPage:
    """Export one portable page of the authenticated account-owned chat source."""
    value = await _request_page(
        request,
        tool_name=_CHAT_HOST_TOOL,
        path=_CHAT_EXPORT_PATH,
        user=me.email,
        limit=limit,
        cursor=cursor,
    )
    try:
        return ChatExportPage.model_validate(value)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e


@router.delete(
    "/chat-history/{conversation_id}",
    response_model=ChatDeletionResult,
    status_code=202,
)
async def delete_chat_history(
    request: Request,
    conversation_id: Annotated[str, Path(min_length=1, max_length=200)],
    me: AuthedUser = Depends(require_user),
) -> ChatDeletionResult:
    """Durably delete one conversation owned by the authenticated account."""
    value = await _request_backend(
        request,
        tool_name=_CHAT_HOST_TOOL,
        path=_CHAT_DELETE_PATH,
        body={"user": me.email, "conversation_id": conversation_id},
        unprocessable_detail="invalid_chat_history_delete",
        not_found_detail="conversation_not_found",
    )
    try:
        return ChatDeletionResult.model_validate(value)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail="user_data_backend_invalid_response",
        ) from e


__all__ = ["router"]

