"""Owner-scoped native Projects resources."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audrey.app_state import (
    PROJECT_INSTRUCTIONS_MAX_CHARS,
    PROJECT_MAX_FILES,
    PROJECT_NAME_MAX_CHARS,
    ApplicationStore,
    InvalidApplicationStateError,
    InvalidProjectStateError,
    ProjectFileConflictError,
    ProjectFileLimitError,
    ProjectFileRecord,
    ProjectNotFoundError,
    ProjectRecord,
)
from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.routes.app import files as native_files
from audrey.routes.app.conversations import (
    ConversationListResponse,
    ConversationResponse,
    _conversation_response,
    _decode_cursor,
    _encode_cursor,
    _Mode,
    _selected_model,
)

router = APIRouter(prefix="/projects", tags=["projects"])
_project_access = require_scope("compat:full")


class ProjectLimitsResponse(BaseModel):
    max_name_chars: int = PROJECT_NAME_MAX_CHARS
    max_instructions_chars: int = PROJECT_INSTRUCTIONS_MAX_CHARS
    max_files: int = PROJECT_MAX_FILES


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=PROJECT_NAME_MAX_CHARS)
    instructions: str = Field(default="", max_length=PROJECT_INSTRUCTIONS_MAX_CHARS)


class ProjectPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=PROJECT_NAME_MAX_CHARS)
    instructions: str | None = Field(
        default=None,
        max_length=PROJECT_INSTRUCTIONS_MAX_CHARS,
    )

    @model_validator(mode="after")
    def require_explicit_values(self):
        if not self.model_fields_set:
            raise ValueError("at least one project field is required")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("project fields cannot be null")
        return self


class ProjectResponse(BaseModel):
    id: str
    name: str
    instructions: str
    created_at: str
    updated_at: str


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]
    next_cursor: str | None
    limits: ProjectLimitsResponse


class ProjectFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1, max_length=200)


class ProjectFileResponse(BaseModel):
    id: str
    filename: str
    mime: str
    kind: str
    bytes: int
    uploaded_at: str
    added_at: str


class ProjectFileListResponse(BaseModel):
    items: list[ProjectFileResponse]
    limits: ProjectLimitsResponse


class ProjectConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=200)
    default_mode: _Mode | None = None
    model_id: str = Field(default="auto", min_length=1, max_length=200)

    @model_validator(mode="after")
    def reject_ambiguous_model(self):
        if "default_mode" in self.model_fields_set and "model_id" in self.model_fields_set:
            raise ValueError("use model_id or default_mode, not both")
        return self


def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application state is not initialized.",
        )
    return store


def _project_response(record: ProjectRecord) -> ProjectResponse:
    return ProjectResponse(
        id=record.project_id,
        name=record.name,
        instructions=record.instructions,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _project_file_response(
    relation: ProjectFileRecord,
    row: native_files.upload_routes.FileRow,
) -> ProjectFileResponse:
    return ProjectFileResponse(
        id=row.file_id,
        filename=row.filename,
        mime=row.mime,
        kind=native_files._kind(row.mime),
        bytes=row.bytes,
        uploaded_at=row.uploaded_at,
        added_at=relation.added_at,
    )


def _limits() -> ProjectLimitsResponse:
    return ProjectLimitsResponse()


async def _owned_project(
    request: Request,
    principal: Principal,
    project_id: str,
) -> ProjectRecord:
    record = await _store(request).projects.get(
        user_id=principal.user_id,
        project_id=project_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return record


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreateRequest,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ProjectResponse:
    try:
        record = await _store(request).projects.create(
            user_id=principal.user_id,
            name=payload.name,
            instructions=payload.instructions,
        )
    except InvalidProjectStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _project_response(record)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    request: Request,
    principal: Principal = Depends(_project_access),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=1000)] = None,
) -> ProjectListResponse:
    before_updated_at: str | None = None
    before_project_id: str | None = None
    if cursor is not None:
        decoded = _decode_cursor(cursor)
        before_updated_at = decoded.get("updated_at")
        before_project_id = decoded.get("project_id")
        if not isinstance(before_updated_at, str) or not isinstance(
            before_project_id,
            str,
        ):
            raise HTTPException(status_code=422, detail="Cursor is invalid for this view.")
    try:
        records = await _store(request).projects.list_page(
            user_id=principal.user_id,
            limit=limit + 1,
            before_updated_at=before_updated_at,
            before_project_id=before_project_id,
        )
    except InvalidProjectStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    page = records[:limit]
    next_cursor = None
    if len(records) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(
            {
                "v": 1,
                "updated_at": last.updated_at,
                "project_id": last.project_id,
            }
        )
    return ProjectListResponse(
        items=[_project_response(record) for record in page],
        next_cursor=next_cursor,
        limits=_limits(),
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ProjectResponse:
    return _project_response(await _owned_project(request, principal, project_id))


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    payload: ProjectPatchRequest,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ProjectResponse:
    try:
        record = await _store(request).projects.update(
            user_id=principal.user_id,
            project_id=project_id,
            name=payload.name if "name" in payload.model_fields_set else None,
            instructions=(
                payload.instructions if "instructions" in payload.model_fields_set else None
            ),
        )
    except InvalidProjectStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _project_response(record)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> Response:
    deleted = await _store(request).projects.delete(
        user_id=principal.user_id,
        project_id=project_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{project_id}/files", response_model=ProjectFileListResponse)
async def list_project_files(
    project_id: str,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ProjectFileListResponse:
    store = _store(request)
    relations = await store.projects.list_files(
        user_id=principal.user_id,
        project_id=project_id,
    )
    if relations is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    listing = await native_files._list_for_owner(request, principal)
    by_id = {row.file_id: row for row in listing.files}
    items: list[ProjectFileResponse] = []
    for relation in relations:
        row = by_id.get(relation.file_id)
        if row is None or row.status != "ready":
            await store.projects.prune_file(
                user_id=principal.user_id,
                project_id=project_id,
                file_id=relation.file_id,
            )
            continue
        items.append(_project_file_response(relation, row))
    return ProjectFileListResponse(items=items, limits=_limits())


@router.post(
    "/{project_id}/files",
    response_model=ProjectFileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_project_file(
    project_id: str,
    payload: ProjectFileRequest,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ProjectFileResponse:
    row = await native_files._owned_row(request, principal, payload.file_id)
    if row.status != "ready":
        raise HTTPException(status_code=422, detail="Project files must be ready.")
    try:
        relation = await _store(request).projects.add_file(
            user_id=principal.user_id,
            project_id=project_id,
            file_id=payload.file_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc
    except ProjectFileConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProjectFileLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _project_file_response(relation, row)


@router.delete("/{project_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_project_file(
    project_id: str,
    file_id: str,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> Response:
    await native_files._owned_row(request, principal, file_id)
    removed = await _store(request).projects.remove_file(
        user_id=principal.user_id,
        project_id=project_id,
        file_id=file_id,
    )
    if not removed:
        raise HTTPException(status_code=404, detail="Project file not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{project_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project_conversation(
    project_id: str,
    payload: ProjectConversationCreateRequest,
    request: Request,
    principal: Principal = Depends(_project_access),
) -> ConversationResponse:
    await _owned_project(request, principal, project_id)
    requested_model = (
        payload.default_mode if "default_mode" in payload.model_fields_set else payload.model_id
    )
    model = await _selected_model(request, principal, requested_model or "auto")
    try:
        record = await _store(request).conversations.create(
            user_id=principal.user_id,
            title=payload.title,
            default_mode=model.mode,
            default_model_id=model.id,
            project_id=project_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found.") from exc
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _conversation_response(record)


@router.get("/{project_id}/conversations", response_model=ConversationListResponse)
async def list_project_conversations(
    project_id: str,
    request: Request,
    principal: Principal = Depends(_project_access),
    archived: bool = False,
    q: Annotated[str, Query(max_length=200)] = "",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=1000)] = None,
) -> ConversationListResponse:
    await _owned_project(request, principal, project_id)
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
            or decoded.get("project_id") != project_id
            or decoded.get("archived") != archived
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
            project_id=project_id,
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
                "project_id": project_id,
                "archived": archived,
                "search": search,
            }
        )
    return ConversationListResponse(
        items=[_conversation_response(record) for record in page],
        next_cursor=next_cursor,
    )


__all__ = [
    "ProjectCreateRequest",
    "ProjectFileListResponse",
    "ProjectFileRequest",
    "ProjectFileResponse",
    "ProjectLimitsResponse",
    "ProjectListResponse",
    "ProjectPatchRequest",
    "ProjectResponse",
    "router",
]
