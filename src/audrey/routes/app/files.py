"""Owner-bound native file resources backed by Audrey's upload lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from audrey.app_state import AttachmentSnapshot
from audrey.auth import AuthedUser, require_scope
from audrey.identity import Principal
from audrey.kb.extract import is_image_mime, is_video_mime
from audrey.routes import files as upload_routes

router = APIRouter(tags=["application-files"])
_files_access = require_scope("compat:full")


class NativeFileLimits(BaseModel):
    max_upload_bytes: int
    max_user_bytes: int
    allowed_extensions: list[str]
    chunked_max_bytes: int
    part_size: int


class NativeFileRecord(BaseModel):
    id: str
    filename: str
    mime: str
    bytes: int
    uploaded_at: str
    kind: Literal["text", "image", "video"]
    chunks: int
    status: str
    failure_reason: str
    duration_s: float
    summary: str
    source_freed_at: str
    leased_at: str
    source_url: str
    transcript_source: str
    fetch_downloaded_bytes: int
    fetch_total_bytes: int


class NativeFileListResponse(BaseModel):
    items: list[NativeFileRecord]
    total_bytes: int
    server_time: str
    limits: NativeFileLimits


class NativeFileUploadResponse(BaseModel):
    id: str
    filename: str
    mime: str
    bytes: int
    kind: Literal["text", "image", "video"]
    chunks: int
    status: str


class NativeFileDeleteResponse(BaseModel):
    id: str
    deleted: bool
    pending_cleanup: bool


def _compat_user(principal: Principal) -> AuthedUser:
    """Project a native principal into the legacy upload service identity.

    The compatibility service already keys every operation by Audrey's durable
    ``storage_namespace``. Constructing this value server-side prevents a
    browser from selecting an email or another user's namespace.
    """

    return AuthedUser(
        email=principal.storage_namespace,
        role=principal.role,
        owui_id="",
        display_name=principal.display_name,
        principal=principal,
    )


def _kind(mime: str) -> Literal["text", "image", "video"]:
    if is_video_mime(mime):
        return "video"
    if is_image_mime(mime):
        return "image"
    return "text"


def _file_record(row: upload_routes.FileRow) -> NativeFileRecord:
    return NativeFileRecord(
        id=row.file_id,
        filename=row.filename,
        mime=row.mime,
        bytes=row.bytes,
        uploaded_at=row.uploaded_at,
        kind=_kind(row.mime),
        chunks=row.chunks,
        status=row.status,
        failure_reason=row.failure_reason,
        duration_s=row.duration_s,
        summary=row.summary,
        source_freed_at=row.source_freed_at,
        leased_at=row.leased_at,
        source_url=row.source_url,
        transcript_source=row.transcript_source,
        fetch_downloaded_bytes=row.fetch_downloaded_bytes,
        fetch_total_bytes=row.fetch_total_bytes,
    )


def _upload_response(
    result: upload_routes.UploadResponse,
) -> NativeFileUploadResponse:
    return NativeFileUploadResponse(
        id=result.file_id,
        filename=result.filename,
        mime=result.mime,
        bytes=result.bytes,
        kind=result.kind,
        chunks=result.chunks,
        status=result.status,
    )


async def _list_for_owner(
    request: Request,
    principal: Principal,
) -> upload_routes.ListResponse:
    return await upload_routes.list_files(request, _compat_user(principal))


async def resolve_owned_attachments(
    request: Request,
    principal: Principal,
    file_ids: Sequence[str],
) -> tuple[AttachmentSnapshot, ...]:
    """Resolve distinct ready files without revealing cross-owner existence."""

    if not file_ids:
        return ()
    if len(file_ids) > 10 or len(set(file_ids)) != len(file_ids):
        raise HTTPException(
            status_code=422,
            detail="Attachments must name at most 10 distinct ready files owned by you.",
        )
    result = await _list_for_owner(request, principal)
    by_id = {row.file_id: row for row in result.files}
    selected = [by_id.get(file_id) for file_id in file_ids]
    if any(row is None or row.status != "ready" for row in selected):
        raise HTTPException(
            status_code=422,
            detail="Attachments must name at most 10 distinct ready files owned by you.",
        )
    return tuple(
        AttachmentSnapshot(
            file_id=row.file_id,
            filename=row.filename,
            mime=row.mime,
            kind=_kind(row.mime),
            bytes=row.bytes,
        )
        for row in selected
        if row is not None
    )


@router.get("/files", response_model=NativeFileListResponse)
async def list_files(
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileListResponse:
    result = await _list_for_owner(request, principal)
    return NativeFileListResponse(
        items=[_file_record(row) for row in result.files],
        total_bytes=result.total_bytes,
        server_time=result.server_time,
        limits=NativeFileLimits(
            max_upload_bytes=result.limits.max_upload_bytes,
            max_user_bytes=result.limits.max_user_bytes,
            allowed_extensions=result.limits.allowed_extensions,
            chunked_max_bytes=result.limits.chunked_max_bytes,
            part_size=result.limits.part_size,
        ),
    )


@router.get("/files/{file_id}", response_model=NativeFileRecord)
async def get_file(
    file_id: str,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileRecord:
    result = await _list_for_owner(request, principal)
    row = next((item for item in result.files if item.file_id == file_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return _file_record(row)


@router.post("/files", response_model=NativeFileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(_files_access),
) -> NativeFileUploadResponse:
    result = await upload_routes.upload_file(
        request=request,
        me=_compat_user(principal),
        file=file,
    )
    return _upload_response(result)


@router.post(
    "/files/upload-sessions",
    response_model=upload_routes.SessionOpenResponse,
)
async def open_upload_session(
    payload: upload_routes.SessionOpenRequest,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> upload_routes.SessionOpenResponse:
    return await upload_routes.open_upload_session(
        body=payload,
        request=request,
        me=_compat_user(principal),
    )


@router.put(
    "/files/upload-sessions/{upload_id}/parts/{part_no}",
    response_model=upload_routes.PartResponse,
)
async def upload_part(
    upload_id: str,
    part_no: int,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> upload_routes.PartResponse:
    return await upload_routes.upload_part(
        upload_id=upload_id,
        part_no=part_no,
        request=request,
        me=_compat_user(principal),
    )


@router.post(
    "/files/upload-sessions/{upload_id}/complete",
    response_model=NativeFileUploadResponse,
)
async def complete_upload_session(
    upload_id: str,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileUploadResponse:
    result = await upload_routes.complete_upload_session(
        upload_id=upload_id,
        request=request,
        me=_compat_user(principal),
    )
    return _upload_response(result)


@router.delete("/files/{file_id}", response_model=NativeFileDeleteResponse)
async def delete_file(
    file_id: str,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileDeleteResponse:
    result = await upload_routes.delete_file(
        file_id=file_id,
        request=request,
        me=_compat_user(principal),
    )
    if not result.deleted:
        raise HTTPException(status_code=404, detail="File not found.")
    return NativeFileDeleteResponse(
        id=result.file_id,
        deleted=True,
        pending_cleanup=result.pending_cleanup,
    )


__all__ = ["resolve_owned_attachments", "router"]
