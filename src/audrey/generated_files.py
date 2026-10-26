"""Owned local outputs from native chat; no source ingestion or external editor."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import ValidationError

from audrey.app_state import ProjectFileConflictError, ProjectFileLimitError, ProjectNotFoundError
from audrey.app_state.private_tasks import current_private_task_lease
from audrey.file_rendering import CreateFileRequest, render_file
from audrey.identity import Principal
from audrey.kb.storage_lifecycle import QuotaExceededError
from audrey.processing_policy import ProcessingPolicyError, current_processing_policy
from audrey.routes import files as uploads
from audrey.user_data_visibility import remote_personal_reads_blocked


class GeneratedFileError(ValueError):
    """Fixed safe feedback, without model content or host paths."""

    def __init__(self, code: str, detail: str) -> None:
        self.code, self.detail = code, detail
        super().__init__(detail)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


async def create_generated_file(
    app: Any, principal: Principal, conversation_id: str, run_id: str, arguments: dict,
) -> dict:
    """Render a new immutable file, or return the same file on an identical retry.

    Identity comes from the native execution scope. Content changes create a new
    file; no output can overwrite an upload or a previous version. Disk and
    catalog handovers settle before cancellation cleanup, as with Private uploads.
    """
    try:
        if len(json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode()) > 262144:
            raise ValueError("too large")
        body = CreateFileRequest.model_validate(arguments)
    except (ValidationError, TypeError, ValueError, RecursionError):
        return {"error": "arguments_invalid", "detail": (
            "Use a safe filename matching format (txt, md, csv, json, docx or xlsx) "
            "and only its content, data, rows or sheets field. Keep total input under 256 KiB."
        )}
    request = Request({"type": "http", "app": app})
    store = app.state.application_store
    db = uploads._get_uploads_db(request)
    storage = uploads._get_storage_lifecycle(request)
    policy = current_processing_policy()

    async def guard():
        if current_processing_policy() is not policy:
            raise GeneratedFileError("file_creation_unavailable", "The processing scope changed.")
        if policy is not None:
            policy.assert_tool_allowed("create_file")
        current = await store.get_principal(user_id=principal.user_id)
        conversation = await store.conversations.get(
            user_id=principal.user_id, conversation_id=conversation_id,
        )
        run = await store.conversations.get_run(user_id=principal.user_id, run_id=run_id)
        if (current is None or current.status != "active"
                or current.storage_namespace != principal.storage_namespace
                or conversation is None or conversation.archived_at is not None
                or run is None or run.conversation_id != conversation_id or run.status != "running"
                or bool(conversation.private_required) != (policy is not None)
                or run.private_required != conversation.private_required
                or remote_personal_reads_blocked(principal.storage_namespace)):
            raise GeneratedFileError("file_creation_unavailable", "This conversation can no longer create files.")
        lease = current_private_task_lease()
        if lease is not None:
            if (not lease.work_active or lease.user_id != principal.user_id
                    or lease.conversation_id != conversation_id):
                raise GeneratedFileError("file_creation_unavailable", "This review step has ended.")
            await lease.verify()
        if conversation.project_id:
            project = await store.projects.get(user_id=principal.user_id, project_id=conversation.project_id)
            if project is None or project.private_required != conversation.private_required:
                raise GeneratedFileError("file_creation_unavailable", "The Project changed. Retry from the current conversation.")
        return conversation

    reservation = None
    paths: list[Path] = []
    file_id = ""
    try:
        conversation = await guard()
        normalized = body.model_dump(mode="json", exclude_none=True)
        identity = json.dumps([principal.storage_namespace, conversation_id,
                               bool(conversation.private_required), normalized],
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(identity.encode()).hexdigest()))
        async with app.state.file_operation_locks.hold(file_id):
            # Tombstones are durable: an inherited retry cannot resurrect a
            # deleted output. A deliberate new filename creates a new identity.
            if file_id in await db.file_deletion_ids(principal.storage_namespace):
                raise GeneratedFileError("generated_file_deleted", "This output was deleted. Choose a new filename to create it again.")
            existing = await db.get_upload(file_id)
            created = existing is None
            if existing is not None:
                if (existing["user"] != principal.storage_namespace or existing.get("origin") != "generated"
                        or bool(existing["private_required"]) != bool(conversation.private_required)):
                    raise GeneratedFileError("file_creation_unavailable", "The output could not be saved.")
                if not await asyncio.to_thread(uploads._source_path(request, existing).is_file):
                    raise GeneratedFileError("generated_file_missing", "The saved file is unavailable. Choose a new filename to create a new copy.")
            else:
                rendered = await asyncio.to_thread(render_file, body)
                current = await guard()
                if current.project_id != conversation.project_id:
                    raise GeneratedFileError("file_creation_unavailable", "The Project changed. Retry from the current conversation.")
                now, expired = uploads._reservation_window(request)
                reservation = await storage.reserve_single_upload(
                    reservation_id=file_id, user=principal.storage_namespace,
                    bytes_=len(rendered.data), max_user_bytes=uploads._max_user_bytes(request),
                    now=now, expired_before=expired, private_required=conversation.private_required,
                )
                source = uploads._source_path(request, {
                    "user": principal.storage_namespace, "file_id": file_id, "filename": body.filename,
                })
                sidecar = source.with_name(f"{file_id}.ocr.txt")
                paths = [source, sidecar]
                # A crash before catalog publication may leave these exact
                # unowned staged paths. File lock and deterministic id make
                # cleaning them safe; committed sources are never replaced.
                for path in paths:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                await uploads._settle_private_persistence(asyncio.to_thread(_write_new, source, rendered.data))
                await uploads._settle_private_persistence(asyncio.to_thread(_write_new, sidecar, rendered.text.encode("utf-8")))
                await guard()
                await uploads._settle_private_persistence(storage.commit_upload(
                    reservation, file_id=file_id, filename=body.filename,
                    mime=rendered.mime_type, bytes_=len(rendered.data), kind="text", collection="",
                    chunks=0, uploaded_at=now, status="ready", max_user_bytes=uploads._max_user_bytes(request),
                    origin="generated",
                ))
            linked = None
            notice = ""
            current = await guard()
            if current.project_id == conversation.project_id and current.project_id:
                try:
                    await uploads._settle_private_persistence(store.projects.add_file(
                        user_id=principal.user_id, project_id=current.project_id,
                        file_id=file_id, file_is_private=current.private_required,
                        file_origin="generated",
                    ))
                    linked = current.project_id
                except ProjectFileConflictError:
                    linked = current.project_id
                except (ProjectNotFoundError, ProjectFileLimitError):
                    notice = "File saved in My files; the Project changed or reached its file limit."
            elif current.project_id != conversation.project_id:
                notice = "File saved in My files; the conversation's Project changed."
            return {
                "file_id": file_id, "filename": body.filename, "created": created,
                "origin": "generated", "download_url": f"/api/files/{file_id}/download",
                "project_id": linked, "notice": notice,
                "persistence": "Saved in My files. This is a generated working output, not independent evidence.",
            }
    except QuotaExceededError:
        return {"error": "file_quota_exceeded", "detail": "Your file storage is full. Delete unneeded files before creating another."}
    except GeneratedFileError as exc:
        return {"error": exc.code, "detail": exc.detail}
    except ProcessingPolicyError:
        return {"error": "file_creation_unavailable", "detail": "File creation is unavailable in this processing scope."}
    except (ValueError, OSError):
        return {"error": "file_creation_unavailable", "detail": "The file could not be saved. No previous file was replaced."}
    finally:
        # A concurrent identical retry can acquire the file lock as this call
        # unwinds. Serialize reservation release and orphan cleanup with it;
        # never unlink bytes that a retry is about to publish.
        if file_id:
            async with app.state.file_operation_locks.hold(file_id):
                if reservation is not None:
                    await storage.release(reservation)
                if paths and await db.get_upload(file_id) is None:
                    for path in paths:
                        await asyncio.to_thread(path.unlink, missing_ok=True)
