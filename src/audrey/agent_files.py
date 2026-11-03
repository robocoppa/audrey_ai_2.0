"""Immutable owner-bound working files from one live saved Agent Work step.

Durable content-free intents bridge the application and upload databases. A
retry reconciles the deterministic catalog identity; it never overwrites bytes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import ValidationError

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.agents import AgentOutputLimitError, AgentTaskError, _stamp
from audrey.app_state.projects import PROJECT_MAX_FILES
from audrey.app_state.transactions import write_transaction
from audrey.file_rendering import CreateFileRequest, render_file
from audrey.generated_files import GeneratedFileError, _output_ids, _write_new
from audrey.identity import Principal
from audrey.kb.storage_lifecycle import QuotaExceededError
from audrey.model_catalog import _apply_policy, _apply_profile, _model_allows, configured_models
from audrey.processing_policy import current_processing_policy
from audrey.routes import files as uploads
from audrey.user_data_visibility import remote_personal_reads_blocked


async def recover_agent_output_staging_for_startup(app: Any, *, batch_size: int = 100) -> int:
    """Release unregistered crash staging before any Agent worker is started.

    Cataloged outputs are immutable owned files, including the cross-database
    crash case where the application receipt is still pending. Retain those
    files for an authorized retry to adopt. Never run this sweep during work.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 200:
        raise ValueError("Invalid Agent output recovery batch size.")
    runtime = getattr(app.state, "agent_runtime", None)
    if runtime is not None and getattr(runtime, "_worker", None) is not None:
        raise RuntimeError("Agent output staging recovery requires startup.")
    store = app.state.application_store
    db = app.state.uploads_db
    request = Request({"type": "http", "app": app})
    cursor, cleaned = "", 0

    def batch():
        with store._lock:
            return [dict(row) for row in store._conn.execute(
                "SELECT o.file_id,o.filename,u.storage_namespace AS user FROM app_agent_outputs o "
                "JOIN app_users u ON u.user_id=o.user_id WHERE o.status='reserved' AND o.file_id>? "
                "ORDER BY o.file_id LIMIT ?", (cursor, batch_size),
            ).fetchall()]

    try:
        while rows := await asyncio.to_thread(batch):
            for row in rows:
                file_id = row["file_id"]
                if str(uuid.UUID(file_id)) != file_id or Path(row["filename"]).suffix.lower() not in {
                    ".txt", ".md", ".csv", ".json", ".docx", ".xlsx",
                }:
                    raise ValueError("Invalid stored Agent output identity.")
                async with app.state.file_operation_locks.hold(file_id):
                    if await db.get_upload(file_id) is not None:
                        continue
                    source = uploads._source_path(request, row)
                    await asyncio.to_thread(source.unlink, missing_ok=True)
                    await asyncio.to_thread(source.with_name(f"{file_id}.ocr.txt").unlink, missing_ok=True)
                    await db.release_storage_reservation(file_id, user=row["user"])
                    cleaned += 1
            cursor = rows[-1]["file_id"]
    except (OSError, sqlite3.Error, ValueError):
        raise RuntimeError("Agent output staging recovery failed.") from None
    return cleaned


@dataclass(slots=True)
class _Binding:
    app: Any
    principal: Principal
    task: dict
    files: list[dict]
    active: bool = True


_BINDING: ContextVar[_Binding | None] = ContextVar("audrey_agent_files", default=None)


def agent_files_available(app: Any) -> bool:
    return all(getattr(app.state, name, None) is not None for name in (
        "application_store", "uploads_db", "storage_lifecycle", "file_operation_locks",
    ))


@contextmanager
def agent_file_scope(*, app: Any, principal: Principal, task: dict, files: list[dict]) -> Iterator[None]:
    if (_BINDING.get() is not None or not isinstance(principal, Principal)
            or task.get("user_id") != principal.user_id or task.get("current_step") != "research"
            or current_processing_policy() is not None):
        raise AgentDispatchStoppedError("Agent file creation is unavailable.")
    binding = _Binding(app, principal, task, files)
    token = _BINDING.set(binding)
    try:
        yield
    finally:
        binding.active = False
        _BINDING.reset(token)


def agent_file_handler(*, user_id: str | None):
    binding = _BINDING.get()
    if binding is None:
        return None

    def guard():
        if (not binding.active or _BINDING.get() is not binding
                or user_id != binding.principal.storage_namespace):
            raise AgentDispatchStoppedError("The agent attempt has stopped.")

    guard()

    async def create(arguments: dict):
        guard()
        result = await create_agent_file(binding.app, binding.principal, binding.task, arguments, scope_guard=guard)
        guard()
        if not result.get("error") and not any(item["file_id"] == result["file_id"] for item in binding.files):
            binding.files.append({key: result[key] for key in ("file_id", "filename", "download_url", "created_at")})
        return result

    return create


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


def _authority(app, principal, task, scope_guard=None):
    """Synchronous final authority; caller holds the shared application lock."""
    from audrey.agent_executor import agent_settings

    if scope_guard is not None:
        scope_guard()
    store = app.state.application_store
    if current_processing_policy() is not None or not agent_settings(app).enabled:
        raise AgentDispatchStoppedError("Agents are unavailable in this processing mode.")
    try:
        store.agents.fence_execution(**_lease(task))
        current = store._get_principal_sync(principal.user_id)
        record = store.agents._lease(**_lease(task))
    except AgentTaskError:
        raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
    if (current is None or current.status != "active" or current.role == "bot" or "bots" in current.groups
            or current.storage_namespace != principal.storage_namespace or record["user_id"] != principal.user_id
            or store.agents._internal(record)["current_step"] != "research"
            or remote_personal_reads_blocked(principal.storage_namespace)):
        raise AgentDispatchStoppedError("Agent file creation is unavailable.")
    model = next((item for item in configured_models(app.state.cfg) if item.id == "local"), None)
    policy = next((item for item in store._list_model_access_policies_sync() if item.model_id == "local"), None)
    profile = next((item for item in store._list_model_publication_profiles_sync() if item.model_id == "local"), None)
    if model is None:
        raise AgentDispatchStoppedError("Local model access is unavailable.")
    model = _apply_profile(_apply_policy(model, policy), profile)
    if not model.enabled or not _model_allows(model, current):
        raise AgentDispatchStoppedError("Local model access is unavailable.")
    return record


def _project_link(store, record, file_id, now):
    """Use the caller's application transaction rather than nest BEGIN writes."""
    context = record["project_context"]
    if not context:
        return None, ""
    project_id = context["project_id"]
    duplicate = store._conn.execute(
        "SELECT 1 FROM app_project_files WHERE project_id=? AND file_id=?", (project_id, file_id),
    ).fetchone()
    if duplicate:
        return project_id, ""
    count = store._conn.execute("SELECT COUNT(*) FROM app_project_files WHERE project_id=?", (project_id,)).fetchone()[0]
    if count >= PROJECT_MAX_FILES:
        return None, "File saved in My files; the Project reached its file limit."
    store._conn.execute(
        "INSERT INTO app_project_files(project_id,file_id,added_at,origin) VALUES (?,?,?,'generated')",
        (project_id, file_id, now),
    )
    store._conn.execute("UPDATE app_projects SET updated_at=? WHERE project_id=? AND user_id=?",
                        (now, project_id, record["user_id"]))
    return project_id, ""


async def create_agent_file(app, principal, task, arguments, *, scope_guard=None):
    """Render, reconcile and publish without storing model arguments in receipts."""
    if not agent_files_available(app):
        return {"error": "file_creation_unavailable", "detail": "Agent file storage is unavailable."}
    store = app.state.application_store

    def guard():
        with store._lock:
            return _authority(app, principal, task, scope_guard)

    guard()
    try:
        if len(json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode()) > 262144:
            raise ValueError
        body = CreateFileRequest.model_validate(arguments)
    except (ValidationError, TypeError, ValueError, RecursionError):
        return {"error": "arguments_invalid", "detail": (
            "Use a safe filename matching format (txt, md, csv, json, docx or xlsx) "
            "and only its content, data, rows or sheets field. Keep total input under 256 KiB."
        )}
    identities = _output_ids(principal, "agent:" + task["id"], False, body)
    file_id = identities[-1]
    identity = hashlib.sha256(json.dumps(
        [file_id, body.model_dump(mode="json", exclude_none=True)], ensure_ascii=False, sort_keys=True,
    ).encode()).hexdigest()
    request = Request({"type": "http", "app": app})
    db = uploads._get_uploads_db(request)
    storage = uploads._get_storage_lifecycle(request)
    reservation = None
    paths: list[Path] = []
    try:
        await store.agents.reserve_output(**_lease(task), file_id=file_id, filename=body.filename, identity_sha256=identity)
        guard()
        async with app.state.file_operation_locks.hold(file_id):
            deleted = await db.file_deletion_ids(principal.storage_namespace)
            guard()
            existing = await db.get_upload(file_id)
            guard()
            if file_id in deleted or (existing is None and any(old in deleted for old in identities[:-1])):
                raise GeneratedFileError("generated_file_deleted", "This output was deleted. Choose a new filename to create it again.")
            created = existing is None
            if existing is not None:
                if (existing["user"] != principal.storage_namespace or existing.get("origin") != "generated"
                        or existing["private_required"] or existing["filename"] != body.filename):
                    raise GeneratedFileError("file_creation_unavailable", "The output could not be saved.")
                if not await asyncio.to_thread(uploads._source_path(request, existing).is_file):
                    raise GeneratedFileError("generated_file_missing", "The saved file is unavailable. Choose a new filename to create a new copy.")
                guard()
                rendered = None
            else:
                rendered = await asyncio.to_thread(render_file, body)
                guard()
                now, expired = uploads._reservation_window(request)
                reservation = await storage.reserve_single_upload(
                    reservation_id=file_id, user=principal.storage_namespace, bytes_=len(rendered.data),
                    max_user_bytes=uploads._max_user_bytes(request), now=now, expired_before=expired,
                )
                guard()
                source = uploads._source_path(request, {
                    "user": principal.storage_namespace, "file_id": file_id, "filename": body.filename,
                })
                paths = [source, source.with_name(f"{file_id}.ocr.txt")]
                for path in paths:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                    guard()
                await uploads._settle_private_persistence(asyncio.to_thread(_write_new, paths[0], rendered.data))
                guard()
                await uploads._settle_private_persistence(asyncio.to_thread(_write_new, paths[1], rendered.text.encode("utf-8")))
                guard()

            def publish():
                # Global lock order is application -> uploads. There is no
                # await between the final lease fence, publication and receipt.
                inputs = store.agents._file_input_service
                with (store._lock, db._lock, write_transaction(store._conn), db._transaction_locked(),
                      inputs.catalog_lock_already_held() if inputs else nullcontext()):
                    record = _authority(app, principal, task, scope_guard)
                    receipt = store._conn.execute(
                        "SELECT * FROM app_agent_outputs WHERE file_id=? AND task_id=? AND user_id=?",
                        (file_id, task["id"], principal.user_id),
                    ).fetchone()
                    if receipt is None or receipt["identity_sha256"] != identity:
                        raise AgentDispatchStoppedError("The agent output intent changed.")
                    for candidate in identities if existing is None else [file_id]:
                        if db._conn.execute(
                            "SELECT 1 FROM file_deletions WHERE file_id=? AND user=?",
                            (candidate, principal.storage_namespace),
                        ).fetchone():
                            raise GeneratedFileError("generated_file_deleted", "This output was deleted. Choose a new filename to create it again.")
                    if existing is not None:
                        catalog = db._conn.execute(
                            "SELECT user,origin,private_required,status FROM uploads WHERE file_id=?", (file_id,),
                        ).fetchone()
                        if (catalog is None or catalog["user"] != principal.storage_namespace
                                or catalog["origin"] != "generated" or catalog["private_required"] or catalog["status"] != "ready"):
                            raise GeneratedFileError("generated_file_missing", "The saved file is unavailable. Choose a new filename to create a new copy.")
                    now = _stamp(store.agents._clock())
                    if rendered is not None:
                        decision = db._commit_reserved_upload_locked(
                            reservation.reservation_id, reservation.kind, principal.storage_namespace,
                            file_id, body.filename, rendered.mime_type, len(rendered.data), "text", "", 0,
                            now, "ready", uploads._max_user_bytes(request), "", "generated",
                        )
                        storage._require(decision)
                    linked, notice = _project_link(store, record, file_id, now)
                    if receipt["status"] != "ready":
                        store._conn.execute("UPDATE app_agent_outputs SET status='ready',committed_at=? WHERE file_id=?", (now, file_id))
                        store.agents._save(record)
                    return {"file_id": file_id, "filename": body.filename,
                            "download_url": f"/api/files/{file_id}/download", "created_at": receipt["created_at"],
                            "created": created, "project_id": linked, "notice": notice, "origin": "generated",
                            "persistence": "Saved in My files. This is a generated working output, not independent evidence."}

            result = await uploads._settle_private_persistence(asyncio.to_thread(publish))
            guard()
            return result
    except AgentOutputLimitError as exc:
        return {"error": "generated_file_limit", "detail": str(exc)}
    except AgentTaskError:
        raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
    except QuotaExceededError:
        return {"error": "file_quota_exceeded", "detail": "Your file storage is full. Delete unneeded files before creating another."}
    except GeneratedFileError as exc:
        return {"error": exc.code, "detail": exc.detail}
    except (ValueError, OSError):
        return {"error": "file_creation_unavailable", "detail": "The file could not be saved. No previous file was replaced."}
    finally:
        async with app.state.file_operation_locks.hold(file_id):
            if reservation is not None:
                await storage.release(reservation)
            if paths and await db.get_upload(file_id) is None:
                for path in paths:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
