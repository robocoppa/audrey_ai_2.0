"""Owner-bound native file resources backed by Audrey's upload lifecycle."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from audrey.app_state import AttachmentSnapshot
from audrey.auth import AuthedUser, require_scope
from audrey.identity import Principal
from audrey.kb.extract import EmptyExtractionError, extract_text, is_image_mime, is_video_mime
from audrey.pipeline.summarise import brief_video_summary
from audrey.pipeline.vision import vision_cfg
from audrey.routes import files as upload_routes

router = APIRouter(tags=["application-files"])
_files_access = require_scope("compat:full")


class NativeFileLimits(BaseModel):
    max_upload_bytes: int
    max_user_bytes: int
    allowed_extensions: list[str]
    chunked_max_bytes: int
    part_size: int
    fetch_hosts: list[str]
    max_images_per_turn: int


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


class NativeFileArtifactResponse(BaseModel):
    id: str
    artifact: Literal["transcript", "visual", "summary"]
    text: str
    offset: int
    next_offset: int | None
    total_chars: int


class NativeFileTextResponse(BaseModel):
    id: str
    text: str
    offset: int
    next_offset: int | None
    total_chars: int


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


def native_image_limit(cfg: object) -> int:
    """Use the same image cap as the vision transcription path."""

    return max(0, int(vision_cfg(cfg).get("max_images_per_turn", 4)))


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
        summary=brief_video_summary(row.summary),
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


async def _owned_row(
    request: Request,
    principal: Principal,
    file_id: str,
) -> upload_routes.FileRow:
    result = await _list_for_owner(request, principal)
    row = next((item for item in result.files if item.file_id == file_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return row


def _source_path(
    request: Request,
    principal: Principal,
    row: upload_routes.FileRow,
) -> Path:
    return upload_routes._source_path(
        request,
        {"user": principal.storage_namespace, "file_id": row.file_id, "filename": row.filename},
    )


def _image_preview(path: Path) -> bytes:
    """Render one bounded frame; do not send untrusted original metadata to the browser."""

    with Image.open(path) as source:
        if source.width * source.height > 50_000_000:
            raise ValueError("Image exceeds preview pixel limit.")
        source.seek(0)
        preview = ImageOps.exif_transpose(source)
        preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        if preview.mode in ("RGBA", "LA") or "transparency" in preview.info:
            rgba = preview.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            preview = flattened
        else:
            preview = preview.convert("RGB")
        output = io.BytesIO()
        preview.save(output, format="JPEG", quality=85)
        return output.getvalue()


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
            fetch_hosts=result.limits.fetch_hosts,
            max_images_per_turn=native_image_limit(getattr(request.app.state, "cfg", None)),
        ),
    )


@router.get("/files/{file_id}", response_model=NativeFileRecord)
async def get_file(
    file_id: str,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileRecord:
    return _file_record(await _owned_row(request, principal, file_id))


@router.get("/files/{file_id}/text", response_model=NativeFileTextResponse)
async def get_file_text(
    file_id: str,
    request: Request,
    response: Response,
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(_files_access),
) -> NativeFileTextResponse:
    """Page through the same extracted document text used by ingestion."""

    row = await _owned_row(request, principal, file_id)
    if _kind(row.mime) != "text" or row.status != "ready":
        raise HTTPException(status_code=422, detail="Text is available for ready documents only.")
    path = _source_path(request, principal, row)
    if row.source_freed_at or not await asyncio.to_thread(path.is_file):
        raise HTTPException(status_code=410, detail="Stored document is unavailable.")
    try:
        content = await asyncio.to_thread(extract_text, path)
    except (EmptyExtractionError, OSError) as exc:
        raise HTTPException(status_code=409, detail="Document text is unavailable.") from exc

    start = min(offset, len(content))
    end = min(start + 4000, len(content))
    response.headers["Cache-Control"] = "private, no-store"
    return NativeFileTextResponse(
        id=row.file_id,
        text=content[start:end],
        offset=start,
        next_offset=end if end < len(content) else None,
        total_chars=len(content),
    )


async def read_owned_image_preview(
    request: Request,
    principal: Principal,
    file_id: str,
) -> bytes:
    """Read bounded JPEG bytes after rechecking current owner and file state."""

    row = await _owned_row(request, principal, file_id)
    if _kind(row.mime) != "image" or row.status != "ready":
        raise HTTPException(status_code=422, detail="Preview is available for ready images only.")
    path = _source_path(request, principal, row)
    if row.source_freed_at or not await asyncio.to_thread(path.is_file):
        raise HTTPException(status_code=410, detail="Stored image is unavailable.")
    try:
        return await asyncio.to_thread(_image_preview, path)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=409, detail="Image preview is unavailable.") from exc


@router.get("/files/{file_id}/image")
async def get_file_image(
    file_id: str,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> Response:
    """Return a resized, metadata-free preview of an owned image."""

    preview = await read_owned_image_preview(request, principal, file_id)
    return Response(
        content=preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/files/{file_id}/artifacts/{artifact}",
    response_model=NativeFileArtifactResponse,
)
async def get_file_artifact(
    file_id: str,
    artifact: Literal["transcript", "visual", "summary"],
    request: Request,
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(_files_access),
) -> NativeFileArtifactResponse:
    """Read one page of a video sidecar by exact owner-bound file ID."""

    row = await _owned_row(request, principal, file_id)
    if _kind(row.mime) != "video":
        raise HTTPException(status_code=422, detail="Artifacts are available for videos only.")

    path = (
        upload_routes._upload_root(request)
        / upload_routes.sanitize_user(principal.storage_namespace)
        / f"{row.file_id}.{upload_routes._ARTIFACT_SIDECARS[artifact]}"
    )
    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except OSError:
        text = ""

    total = len(text)
    start = min(offset, total)
    end = min(start + 4000, total)
    if end < total:
        newline = text.find("\n", end)
        end = total if newline == -1 else newline + 1
    return NativeFileArtifactResponse(
        id=row.file_id,
        artifact=artifact,
        text=text[start:end],
        offset=start,
        next_offset=end if end < total else None,
        total_chars=total,
    )


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


@router.post("/files/from-url", response_model=NativeFileUploadResponse)
async def fetch_file_from_url(
    payload: upload_routes.UrlIngestRequest,
    request: Request,
    principal: Principal = Depends(_files_access),
) -> NativeFileUploadResponse:
    """Queue an allowlisted video URL using the authenticated owner's namespace."""

    result = await upload_routes.ingest_from_url(
        body=payload,
        request=request,
        me=_compat_user(principal),
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


__all__ = [
    "native_image_limit",
    "read_owned_image_preview",
    "resolve_owned_attachments",
    "router",
]
