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
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError

from audrey.auth import AuthedUser, require_user

router = APIRouter(prefix="/v1/me", tags=["user-data"])

_MEMORY_HOST_TOOL = "memory_search"
_CHAT_HOST_TOOL = "chat_history_search"
_MEMORY_LIST_PATH = "/user_data/memories/list"
_CHAT_EXPORT_PATH = "/user_data/chat_history/export"


class MemoryItem(BaseModel):
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str


class MemoryPage(BaseModel):
    items: list[MemoryItem]
    next_cursor: str | None = None


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


def _tool_host(request: Request, tool_name: str) -> str:
    registry = getattr(request.app.state, "tools", None)
    spec = registry.get(tool_name) if registry is not None else None
    if spec is None:
        raise HTTPException(status_code=503, detail=f"{tool_name}_not_registered")
    return str(spec.server_url).rstrip("/")


async def _request_page(
    request: Request,
    *,
    tool_name: str,
    path: str,
    user: str,
    limit: int,
    cursor: str | None,
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
    body: dict[str, Any] = {"user": user, "limit": limit}
    if cursor is not None:
        body["cursor"] = cursor
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
                detail="invalid_pagination_cursor",
            )
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


__all__ = ["router"]

