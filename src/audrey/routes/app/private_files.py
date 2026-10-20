"""Owner-authorized document uploads into Private conversations or Projects."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from audrey.app_state import ProjectFileConflictError, ProjectFileLimitError, ProjectNotFoundError
from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.kb.private_documents import PrivateDocumentError
from audrey.kb.storage_lifecycle import QuotaExceededError
from audrey.kb.user_store import sanitize_user
from audrey.private_processing import prepare_native_private_runtime
from audrey.processing_policy import ProcessingPolicyError, processing_scope
from audrey.routes import files as uploads
from audrey.routes.app.files import NativeFileUploadResponse, _upload_response

router = APIRouter(tags=["application-private-files"])
_access = require_scope("compat:full")


def _create_source(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)


@router.post("/conversations/{conversation_id}/files", response_model=NativeFileUploadResponse)
async def upload_private_document(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(_access),
) -> NativeFileUploadResponse:
    store = request.app.state.application_store
    conversation = await store.conversations.get(
        user_id=principal.user_id, conversation_id=conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if not conversation.private_required or conversation.default_model_id != "private":
        raise HTTPException(
            status_code=409,
            detail="Select Private in a new conversation before uploading Private documents.",
        )
    if conversation.archived_at is not None:
        raise HTTPException(
            status_code=409, detail="Upload into an active Private conversation."
        )
    if conversation.project_id is not None:
        project = await store.projects.get(
            user_id=principal.user_id, project_id=conversation.project_id,
        )
        if project is None or not project.private_required:
            raise HTTPException(status_code=409, detail="Private documents require a Private Project.")
    return await _upload_document(request, file, principal)


@router.post("/projects/{project_id}/uploads", response_model=NativeFileUploadResponse)
async def upload_private_project_document(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(_access),
) -> NativeFileUploadResponse:
    from audrey.routes.app.conversations import _selected_model
    from audrey.routes.app.projects import _owned_project

    project = await _owned_project(request, principal, project_id)
    if not project.private_required:
        raise HTTPException(status_code=409, detail="Use the ordinary uploader for this Project.")
    await _selected_model(request, principal, "private")
    return await _upload_document(request, file, principal, project_id=project_id)


async def _upload_document(
    request: Request, file: UploadFile, principal: Principal, *, project_id: str | None = None,
) -> NativeFileUploadResponse:
    # No browser flag, chosen model name or transport setting establishes the
    # restriction. Both the conversation and owned quota reservation are durable.
    storage = uploads._get_storage_lifecycle(request)
    maximum = uploads._max_upload_bytes(request)
    total = uploads._max_user_bytes(request)
    declared = maximum if file.size is None else int(file.size)
    if not 0 < declared <= maximum:
        raise HTTPException(
            status_code=413, detail="This document exceeds the Private upload limit or is empty."
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {
        ".txt",
        ".log",
        ".md",
        ".rst",
        ".csv",
        ".pdf",
        ".docx",
        ".xlsx",
        ".html",
        ".htm",
    }:
        raise HTTPException(
            status_code=422,
            detail="Private uploads currently support text documents, text PDFs, DOCX and spreadsheets. Images, audio and video are unavailable.",
        )
    try:
        runtime = await prepare_native_private_runtime(request.app)
    except ProcessingPolicyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    user = principal.storage_namespace
    file_id = str(uuid.uuid4())
    dest = uploads._upload_root(request) / sanitize_user(user) / f"{file_id}{suffix}"
    reservation = None
    try:
        with processing_scope(runtime.policy):
            now, expired = uploads._reservation_window(request)
            reservation = await storage.reserve_single_upload(
                reservation_id=file_id,
                user=user,
                bytes_=declared,
                max_user_bytes=total,
                now=now,
                expired_before=expired,
                private_required=True,
            )
            async with request.app.state.file_operation_locks.hold(file_id):
                # Tiny metadata operation, no contents; exclusive mode prevents
                # replacing an existing upload and leaves restrictive permissions.
                _create_source(dest)
                written = await uploads._stream_to_disk(file, dest, limit_bytes=maximum)
                if written <= 0:
                    raise HTTPException(
                        status_code=413,
                        detail="This document exceeds the Private upload limit or is empty.",
                    )
                reservation = await storage.reserve_single_upload(
                    reservation_id=file_id,
                    user=user,
                    bytes_=written,
                    max_user_bytes=total,
                    now=uploads._reservation_window(request)[0],
                    expired_before=expired,
                    private_required=True,
                )
                result = await uploads._ingest_private_document(
                    request,
                    dest,
                    runtime=runtime,
                    storage=storage,
                    reservation=reservation,
                    user=user,
                    file_id=file_id,
                    filename=Path(file.filename or file_id).name,
                    written=written,
                    max_total=total,
                )
                if project_id is not None:
                    try:
                        await uploads._settle_private_persistence(
                            request.app.state.application_store.projects.add_file(
                                user_id=principal.user_id, project_id=project_id,
                                file_id=file_id, file_is_private=True,
                            )
                        )
                    except (ProjectNotFoundError, ProjectFileConflictError, ProjectFileLimitError):
                        # The source is already committed and remains owner-bound.
                        # Project deletion/limits must not erase a successful upload.
                        raise HTTPException(status_code=409, detail={
                            "message": "Upload succeeded, but the Project changed or is full. Your Private document remains in My Files.",
                            "file_id": file_id,
                        }) from None
                return _upload_response(result)
    except QuotaExceededError as exc:
        raise uploads._quota_http_error(exc) from None
    except PrivateDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except ProcessingPolicyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except (HTTPException, asyncio.CancelledError):
        raise
    except Exception:  # noqa: BLE001 - private provider/parser errors must not disclose content
        raise HTTPException(
            status_code=503,
            detail="Private document processing is unavailable. No cloud fallback was used.",
        ) from None
    finally:
        if reservation is not None:
            await storage.release(reservation)
        if await request.app.state.uploads_db.get_upload(file_id) is None:
            uploads._safe_unlink(dest)
