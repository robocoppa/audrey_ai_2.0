"""Owner-scoped approvals, leases, and immutable document derivations."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Mapping

from audrey.app_state.records import (
    DocumentApprovalRecord,
    DocumentJobRecord,
    FileVersionRecord,
)

_ARGUMENTS_MAX_BYTES = 16_000
_ARGUMENTS_MAX_DEPTH = 6
_ARGUMENTS_MAX_NODES = 256
_ARGUMENT_STRING_MAX_CHARS = 4_000
_DEFAULT_APPROVAL_MINUTES = 15
_MAX_APPROVAL_HOURS = 24
_OPERATION_RE = re.compile(r"\A[a-z][a-z0-9_]{0,99}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_TERMINAL_JOB_STATES = frozenset({
    "succeeded", "rejected", "expired", "cancelled", "failed",
})


class InvalidDocumentOperationError(ValueError):
    """A proposed document operation is malformed or outside its bounds."""


class DocumentConflictError(RuntimeError):
    """A document request conflicts with durable state or an expected version."""


class DocumentApprovalError(RuntimeError):
    """An approval cannot be decided or consumed in its current state."""


class DocumentApprovalExpiredError(DocumentApprovalError):
    """An approval expired before it could be used."""


class DocumentLeaseError(RuntimeError):
    """A worker lease is absent, stale, or belongs to a different worker."""


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _required(value: str, label: str, *, maximum: int = 200) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise InvalidDocumentOperationError(f"{label} is required")
    if len(cleaned) > maximum:
        raise InvalidDocumentOperationError(
            f"{label} must be at most {maximum} characters"
        )
    if "\x00" in cleaned or "\r" in cleaned or "\n" in cleaned:
        raise InvalidDocumentOperationError(f"{label} contains invalid characters")
    return cleaned


def _sha256(value: str, label: str = "content sha256") -> str:
    cleaned = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(cleaned):
        raise InvalidDocumentOperationError(f"{label} must be 64 lowercase hex characters")
    return cleaned


def _instant(value: str | None, *, default_minutes: int = 0) -> tuple[str, dt.datetime]:
    if value is None:
        moment = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=default_minutes)
        return moment.isoformat(), moment
    try:
        moment = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidDocumentOperationError("timestamp must be ISO 8601") from exc
    if moment.tzinfo is None:
        raise InvalidDocumentOperationError("timestamp must include a timezone")
    normalized = moment.astimezone(dt.UTC)
    return normalized.isoformat(), normalized


def _normalize_value(value: object, *, depth: int, count: list[int]) -> object:
    count[0] += 1
    if count[0] > _ARGUMENTS_MAX_NODES:
        raise InvalidDocumentOperationError("document arguments contain too many values")
    if depth > _ARGUMENTS_MAX_DEPTH:
        raise InvalidDocumentOperationError("document arguments are nested too deeply")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidDocumentOperationError("document arguments require finite numbers")
        return value
    if isinstance(value, str):
        if len(value) > _ARGUMENT_STRING_MAX_CHARS:
            raise InvalidDocumentOperationError(
                "one document argument exceeds the text limit"
            )
        return value
    if isinstance(value, list | tuple):
        return [
            _normalize_value(item, depth=depth + 1, count=count)
            for item in value
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 100:
                raise InvalidDocumentOperationError(
                    "document argument names must be 1 to 100 characters"
                )
            normalized[key] = _normalize_value(
                item,
                depth=depth + 1,
                count=count,
            )
        return normalized
    raise InvalidDocumentOperationError(
        "document arguments may contain only JSON-compatible values"
    )


def normalize_document_arguments(arguments: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Return a validated object and its stable compact JSON representation."""

    if not isinstance(arguments, Mapping):
        raise InvalidDocumentOperationError("document arguments must be an object")
    normalized = _normalize_value(arguments, depth=0, count=[0])
    assert isinstance(normalized, dict)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidDocumentOperationError("document arguments are not valid JSON") from exc
    if len(encoded.encode("utf-8")) > _ARGUMENTS_MAX_BYTES:
        raise InvalidDocumentOperationError("document arguments exceed the size limit")
    return normalized, encoded


def document_operation_digest(
    *,
    operation: str,
    input_version_id: str,
    arguments: Mapping[str, object],
    output_mime: str,
) -> str:
    """Hash the complete normalized operation that an approval authorizes."""

    operation = _required(operation, "operation", maximum=100)
    if not _OPERATION_RE.fullmatch(operation):
        raise InvalidDocumentOperationError("operation must use lowercase snake case")
    input_version_id = _required(input_version_id, "input version id")
    output_mime = _required(output_mime, "output mime")
    normalized, _ = normalize_document_arguments(arguments)
    envelope = json.dumps(
        {
            "arguments": normalized,
            "input_version_id": input_version_id,
            "operation": operation,
            "output_mime": output_mime,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(envelope.encode("utf-8")).hexdigest()


class DocumentToolsRepository:
    """Short SQLite transactions for document approval and publication state."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def register_source_version(
        self,
        *,
        user_id: str,
        file_id: str,
        filename: str,
        mime: str,
        bytes_count: int,
        content_sha256: str,
    ) -> FileVersionRecord:
        return await asyncio.to_thread(
            self._register_source_version_sync,
            user_id,
            file_id,
            filename,
            mime,
            bytes_count,
            content_sha256,
        )

    def _register_source_version_sync(
        self,
        user_id: str,
        file_id: str,
        filename: str,
        mime: str,
        bytes_count: int,
        content_sha256: str,
    ) -> FileVersionRecord:
        user_id = _required(user_id, "user id")
        file_id = _required(file_id, "file id")
        filename = _required(filename, "filename", maximum=255)
        mime = _required(mime, "mime")
        digest = _sha256(content_sha256)
        if bytes_count < 0:
            raise InvalidDocumentOperationError("bytes must be nonnegative")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    "SELECT * FROM app_file_versions "
                    "WHERE user_id = ? AND file_id = ? AND content_sha256 = ?",
                    (user_id, file_id, digest),
                ).fetchone()
                if existing is not None:
                    self._conn.commit()
                    return _version_from_row(existing)
                latest = self._conn.execute(
                    "SELECT COALESCE(MAX(version_number), 0) FROM app_file_versions "
                    "WHERE user_id = ? AND file_id = ?",
                    (user_id, file_id),
                ).fetchone()
                version_number = int(latest[0]) + 1
                parent = self._conn.execute(
                    "SELECT version_id FROM app_file_versions "
                    "WHERE user_id = ? AND file_id = ? "
                    "ORDER BY version_number DESC LIMIT 1",
                    (user_id, file_id),
                ).fetchone()
                version_id = f"fver_{uuid.uuid4().hex}"
                now = _utc_now()
                self._conn.execute(
                    "INSERT INTO app_file_versions "
                    "(version_id, user_id, file_id, version_number, parent_version_id, "
                    "filename, mime, bytes, content_sha256, origin, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'upload', ?)",
                    (
                        version_id,
                        user_id,
                        file_id,
                        version_number,
                        str(parent[0]) if parent is not None else None,
                        filename,
                        mime,
                        bytes_count,
                        digest,
                        now,
                    ),
                )
                row = self._version_row_locked(user_id, version_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _version_from_row(row)

    async def get_version(
        self,
        *,
        user_id: str,
        version_id: str,
    ) -> FileVersionRecord | None:
        return await asyncio.to_thread(self._get_version_sync, user_id, version_id)

    def _get_version_sync(self, user_id: str, version_id: str) -> FileVersionRecord | None:
        user_id = _required(user_id, "user id")
        version_id = _required(version_id, "version id")
        with self._lock:
            row = self._version_row_locked(user_id, version_id)
        return _version_from_row(row) if row is not None else None

    async def create_request(
        self,
        *,
        user_id: str,
        input_version_id: str,
        operation: str,
        arguments: Mapping[str, object],
        output_mime: str,
        summary: str,
        preview: str,
        idempotency_key: str,
        requested_by_kind: str,
        requested_by_id: str,
        expires_at: str | None = None,
        bot_allowed_operations: frozenset[str] = frozenset(),
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord]:
        return await asyncio.to_thread(
            self._create_request_sync,
            user_id,
            input_version_id,
            operation,
            arguments,
            output_mime,
            summary,
            preview,
            idempotency_key,
            requested_by_kind,
            requested_by_id,
            expires_at,
            bot_allowed_operations,
        )

    def _create_request_sync(
        self,
        user_id: str,
        input_version_id: str,
        operation: str,
        arguments: Mapping[str, object],
        output_mime: str,
        summary: str,
        preview: str,
        idempotency_key: str,
        requested_by_kind: str,
        requested_by_id: str,
        expires_at: str | None,
        bot_allowed_operations: frozenset[str],
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord]:
        user_id = _required(user_id, "user id")
        input_version_id = _required(input_version_id, "input version id")
        operation = _required(operation, "operation", maximum=100)
        if not _OPERATION_RE.fullmatch(operation):
            raise InvalidDocumentOperationError("operation must use lowercase snake case")
        output_mime = _required(output_mime, "output mime")
        summary = _required(summary, "summary", maximum=500)
        preview = str(preview).strip()
        if len(preview) > 2_000:
            raise InvalidDocumentOperationError("preview must be at most 2000 characters")
        idempotency_key = _required(idempotency_key, "idempotency key")
        requested_by_kind = _required(requested_by_kind, "requester kind", maximum=20)
        if requested_by_kind not in {"user", "model", "bot"}:
            raise InvalidDocumentOperationError("requester kind is invalid")
        requested_by_id = _required(requested_by_id, "requester id")
        if requested_by_kind == "bot" and operation not in bot_allowed_operations:
            raise InvalidDocumentOperationError("document operation is not allowlisted for bots")
        normalized, arguments_json = normalize_document_arguments(arguments)
        digest = document_operation_digest(
            operation=operation,
            input_version_id=input_version_id,
            arguments=normalized,
            output_mime=output_mime,
        )
        now_text, now = _instant(_utc_now())
        expiry_text, expiry = _instant(
            expires_at,
            default_minutes=_DEFAULT_APPROVAL_MINUTES,
        )
        if expiry <= now or expiry > now + dt.timedelta(hours=_MAX_APPROVAL_HOURS):
            raise InvalidDocumentOperationError(
                "approval expiry must be in the next 24 hours"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                version = self._version_row_locked(user_id, input_version_id)
                if version is None:
                    raise DocumentConflictError("input version not found")
                if not self._is_current_version_locked(version):
                    raise DocumentConflictError("input version is stale")
                existing = self._conn.execute(
                    "SELECT * FROM app_document_jobs "
                    "WHERE user_id = ? AND idempotency_key = ?",
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if not hmac.compare_digest(str(existing["operation_digest"]), digest):
                        raise DocumentConflictError(
                            "idempotency key was already used for another operation"
                        )
                    approval = self._approval_for_job_locked(
                        user_id,
                        str(existing["job_id"]),
                    )
                    assert approval is not None
                    self._conn.commit()
                    return _job_from_row(existing), _approval_from_row(approval)
                job_id = f"djob_{uuid.uuid4().hex}"
                approval_id = f"appr_{uuid.uuid4().hex}"
                self._conn.execute(
                    "INSERT INTO app_document_jobs "
                    "(job_id, user_id, input_version_id, operation, arguments_json, "
                    "operation_digest, output_mime, summary, preview, "
                    "requested_by_kind, requested_by_id, idempotency_key, status, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "'awaiting_approval', ?, ?)",
                    (
                        job_id,
                        user_id,
                        input_version_id,
                        operation,
                        arguments_json,
                        digest,
                        output_mime,
                        summary,
                        preview,
                        requested_by_kind,
                        requested_by_id,
                        idempotency_key,
                        now_text,
                        now_text,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO app_document_approvals "
                    "(approval_id, job_id, user_id, operation_digest, expires_at, "
                    "decision, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                    (
                        approval_id,
                        job_id,
                        user_id,
                        digest,
                        expiry_text,
                        now_text,
                    ),
                )
                job = self._job_row_locked(user_id, job_id)
                approval = self._approval_for_job_locked(user_id, job_id)
                assert job is not None and approval is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _job_from_row(job), _approval_from_row(approval)

    async def get_job(
        self,
        *,
        user_id: str,
        job_id: str,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        return await asyncio.to_thread(self._get_job_sync, user_id, job_id)

    def _get_job_sync(
        self,
        user_id: str,
        job_id: str,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        user_id = _required(user_id, "user id")
        job_id = _required(job_id, "job id")
        with self._lock:
            job = self._job_row_locked(user_id, job_id)
            if job is None:
                return None
            approval = self._approval_for_job_locked(user_id, job_id)
            assert approval is not None
        return _job_from_row(job), _approval_from_row(approval)

    async def list_jobs(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> tuple[tuple[DocumentJobRecord, DocumentApprovalRecord], ...]:
        return await asyncio.to_thread(self._list_jobs_sync, user_id, limit)

    def _list_jobs_sync(
        self,
        user_id: str,
        limit: int,
    ) -> tuple[tuple[DocumentJobRecord, DocumentApprovalRecord], ...]:
        user_id = _required(user_id, "user id")
        if limit < 1 or limit > 100:
            raise InvalidDocumentOperationError("limit must be between 1 and 100")
        with self._lock:
            rows = self._conn.execute(
                "SELECT j.*, a.approval_id AS a_approval_id, "
                "a.operation_digest AS a_operation_digest, a.expires_at AS a_expires_at, "
                "a.decision AS a_decision, a.actor_user_id AS a_actor_user_id, "
                "a.created_at AS a_created_at, a.decided_at AS a_decided_at, "
                "a.used_at AS a_used_at FROM app_document_jobs AS j "
                "JOIN app_document_approvals AS a "
                "ON a.job_id = j.job_id AND a.user_id = j.user_id "
                "WHERE j.user_id = ? ORDER BY j.created_at DESC, j.job_id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(
            (_job_from_row(row), _approval_from_joined_row(row))
            for row in rows
        )

    async def decide(
        self,
        *,
        user_id: str,
        job_id: str,
        actor_user_id: str,
        operation_digest: str,
        decision: str,
        now: str | None = None,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        return await asyncio.to_thread(
            self._decide_sync,
            user_id,
            job_id,
            actor_user_id,
            operation_digest,
            decision,
            now,
        )

    def _decide_sync(
        self,
        user_id: str,
        job_id: str,
        actor_user_id: str,
        operation_digest: str,
        decision: str,
        now: str | None,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        user_id = _required(user_id, "user id")
        job_id = _required(job_id, "job id")
        actor_user_id = _required(actor_user_id, "actor user id")
        if actor_user_id != user_id:
            raise DocumentApprovalError("only the document owner can decide approval")
        operation_digest = _sha256(operation_digest, "operation digest")
        if decision not in {"approved", "rejected"}:
            raise InvalidDocumentOperationError("decision must be approved or rejected")
        now_text, now_value = _instant(now or _utc_now())
        outcome_error: DocumentApprovalError | None = None
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                job = self._job_row_locked(user_id, job_id)
                if job is None:
                    self._conn.commit()
                    return None
                approval = self._approval_for_job_locked(user_id, job_id)
                assert approval is not None
                if str(approval["decision"]) != "pending" or str(job["status"]) != "awaiting_approval":
                    raise DocumentApprovalError("approval is no longer pending")
                if not hmac.compare_digest(
                    str(approval["operation_digest"]),
                    operation_digest,
                ) or not hmac.compare_digest(
                    str(job["operation_digest"]),
                    operation_digest,
                ):
                    raise DocumentApprovalError("approved operation digest does not match")
                _, expiry = _instant(str(approval["expires_at"]))
                if expiry <= now_value:
                    self._conn.execute(
                        "UPDATE app_document_approvals SET decision = 'expired', "
                        "decided_at = ? WHERE approval_id = ?",
                        (now_text, approval["approval_id"]),
                    )
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = 'expired', error_code = "
                        "'approval_expired', updated_at = ?, completed_at = ? "
                        "WHERE job_id = ?",
                        (now_text, now_text, job_id),
                    )
                    outcome_error = DocumentApprovalExpiredError("approval expired")
                elif decision == "approved" and not self._is_current_version_id_locked(
                    user_id,
                    str(job["input_version_id"]),
                ):
                    self._conn.execute(
                        "UPDATE app_document_approvals SET decision = 'cancelled', "
                        "actor_user_id = ?, decided_at = ? WHERE approval_id = ?",
                        (actor_user_id, now_text, approval["approval_id"]),
                    )
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = 'failed', error_code = "
                        "'stale_input', updated_at = ?, completed_at = ? WHERE job_id = ?",
                        (now_text, now_text, job_id),
                    )
                    outcome_error = DocumentApprovalError("input version is stale")
                else:
                    next_status = "queued" if decision == "approved" else "rejected"
                    completed_at = None if decision == "approved" else now_text
                    self._conn.execute(
                        "UPDATE app_document_approvals SET decision = ?, "
                        "actor_user_id = ?, decided_at = ? WHERE approval_id = ?",
                        (decision, actor_user_id, now_text, approval["approval_id"]),
                    )
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = ?, updated_at = ?, "
                        "completed_at = ? WHERE job_id = ?",
                        (next_status, now_text, completed_at, job_id),
                    )
                job = self._job_row_locked(user_id, job_id)
                approval = self._approval_for_job_locked(user_id, job_id)
                assert job is not None and approval is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        if outcome_error is not None:
            raise outcome_error
        return _job_from_row(job), _approval_from_row(approval)

    async def cancel(
        self,
        *,
        user_id: str,
        job_id: str,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        return await asyncio.to_thread(self._cancel_sync, user_id, job_id)

    def _cancel_sync(
        self,
        user_id: str,
        job_id: str,
    ) -> tuple[DocumentJobRecord, DocumentApprovalRecord] | None:
        user_id = _required(user_id, "user id")
        job_id = _required(job_id, "job id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                job = self._job_row_locked(user_id, job_id)
                if job is None:
                    self._conn.commit()
                    return None
                if str(job["status"]) in _TERMINAL_JOB_STATES:
                    if str(job["status"]) != "cancelled":
                        raise DocumentConflictError("completed document job cannot be cancelled")
                else:
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = 'cancelled', "
                        "lease_id = '', lease_expires_at = NULL, updated_at = ?, "
                        "completed_at = ? WHERE job_id = ?",
                        (now, now, job_id),
                    )
                    self._conn.execute(
                        "UPDATE app_document_approvals SET decision = 'cancelled', "
                        "decided_at = COALESCE(decided_at, ?) WHERE job_id = ? "
                        "AND decision IN ('pending', 'approved')",
                        (now, job_id),
                    )
                job = self._job_row_locked(user_id, job_id)
                approval = self._approval_for_job_locked(user_id, job_id)
                assert job is not None and approval is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _job_from_row(job), _approval_from_row(approval)

    async def claim_next(
        self,
        *,
        lease_id: str,
        lease_seconds: int,
        max_attempts: int = 3,
        now: str | None = None,
        operation: str | None = None,
    ) -> DocumentJobRecord | None:
        return await asyncio.to_thread(
            self._claim_next_sync,
            lease_id,
            lease_seconds,
            max_attempts,
            now,
            operation,
        )

    def _claim_next_sync(
        self,
        lease_id: str,
        lease_seconds: int,
        max_attempts: int,
        now: str | None,
        operation: str | None,
    ) -> DocumentJobRecord | None:
        lease_id = _required(lease_id, "lease id")
        if operation is not None:
            operation = _required(operation, "operation", maximum=100)
            if not _OPERATION_RE.fullmatch(operation):
                raise InvalidDocumentOperationError(
                    "operation filter must use lowercase snake case"
                )
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise InvalidDocumentOperationError("lease seconds must be between 1 and 86400")
        if max_attempts < 1 or max_attempts > 20:
            raise InvalidDocumentOperationError("max attempts must be between 1 and 20")
        now_text, now_value = _instant(now or _utc_now())
        lease_expires = (now_value + dt.timedelta(seconds=lease_seconds)).isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                expired = self._conn.execute(
                    "SELECT job_id FROM app_document_approvals WHERE decision = 'pending' "
                    "AND expires_at <= ?",
                    (now_text,),
                ).fetchall()
                if expired:
                    ids = [str(row[0]) for row in expired]
                    self._conn.executemany(
                        "UPDATE app_document_approvals SET decision = 'expired', "
                        "decided_at = ? WHERE job_id = ?",
                        [(now_text, job_id) for job_id in ids],
                    )
                    self._conn.executemany(
                        "UPDATE app_document_jobs SET status = 'expired', "
                        "error_code = 'approval_expired', updated_at = ?, "
                        "completed_at = ? WHERE status = 'awaiting_approval' "
                        "AND job_id = ?",
                        [(now_text, now_text, job_id) for job_id in ids],
                    )
                self._conn.execute(
                    "UPDATE app_document_jobs SET status = 'failed', error_code = "
                    "'attempts_exhausted', lease_id = '', lease_expires_at = NULL, "
                    "updated_at = ?, completed_at = ? WHERE status = 'running' "
                    "AND lease_expires_at <= ? AND attempts >= ?",
                    (now_text, now_text, now_text, max_attempts),
                )
                self._conn.execute(
                    "UPDATE app_document_jobs SET status = 'queued', lease_id = '', "
                    "lease_expires_at = NULL, updated_at = ? WHERE status = 'running' "
                    "AND lease_expires_at <= ? AND attempts < ?",
                    (now_text, now_text, max_attempts),
                )
                while True:
                    if operation is None:
                        row = self._conn.execute(
                            "SELECT j.* FROM app_document_jobs AS j "
                            "JOIN app_document_approvals AS a ON a.job_id = j.job_id "
                            "AND a.user_id = j.user_id WHERE j.status = 'queued' "
                            "AND a.decision IN ('approved', 'used') "
                            "ORDER BY j.created_at, j.job_id LIMIT 1",
                        ).fetchone()
                    else:
                        row = self._conn.execute(
                            "SELECT j.* FROM app_document_jobs AS j "
                            "JOIN app_document_approvals AS a ON a.job_id = j.job_id "
                            "AND a.user_id = j.user_id WHERE j.status = 'queued' "
                            "AND a.decision IN ('approved', 'used') "
                            "AND j.operation = ? "
                            "ORDER BY j.created_at, j.job_id LIMIT 1",
                            (operation,),
                        ).fetchone()
                    if row is None:
                        self._conn.commit()
                        return None
                    if not self._is_current_version_id_locked(
                        str(row["user_id"]),
                        str(row["input_version_id"]),
                    ):
                        self._conn.execute(
                            "UPDATE app_document_jobs SET status = 'failed', "
                            "error_code = 'stale_input', updated_at = ?, completed_at = ? "
                            "WHERE job_id = ?",
                            (now_text, now_text, row["job_id"]),
                        )
                        continue
                    self._conn.execute(
                        "UPDATE app_document_approvals SET decision = 'used', used_at = ? "
                        "WHERE job_id = ? AND decision = 'approved'",
                        (now_text, row["job_id"]),
                    )
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = 'running', attempts = "
                        "attempts + 1, lease_id = ?, lease_expires_at = ?, updated_at = ? "
                        "WHERE job_id = ? AND status = 'queued'",
                        (lease_id, lease_expires, now_text, row["job_id"]),
                    )
                    claimed = self._conn.execute(
                        "SELECT * FROM app_document_jobs WHERE job_id = ?",
                        (row["job_id"],),
                    ).fetchone()
                    assert claimed is not None
                    self._conn.commit()
                    return _job_from_row(claimed)
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    async def publish(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_id: str,
        output_file_id: str,
        filename: str,
        mime: str,
        bytes_count: int,
        content_sha256: str,
        worker_version: str,
    ) -> tuple[DocumentJobRecord, FileVersionRecord]:
        return await asyncio.to_thread(
            self._publish_sync,
            user_id,
            job_id,
            lease_id,
            output_file_id,
            filename,
            mime,
            bytes_count,
            content_sha256,
            worker_version,
        )

    def _publish_sync(
        self,
        user_id: str,
        job_id: str,
        lease_id: str,
        output_file_id: str,
        filename: str,
        mime: str,
        bytes_count: int,
        content_sha256: str,
        worker_version: str,
    ) -> tuple[DocumentJobRecord, FileVersionRecord]:
        user_id = _required(user_id, "user id")
        job_id = _required(job_id, "job id")
        lease_id = _required(lease_id, "lease id")
        output_file_id = _required(output_file_id, "output file id")
        filename = _required(filename, "filename", maximum=255)
        mime = _required(mime, "mime")
        digest = _sha256(content_sha256)
        worker_version = _required(worker_version, "worker version")
        if bytes_count < 0:
            raise InvalidDocumentOperationError("bytes must be nonnegative")
        now = _utc_now()
        _, now_value = _instant(now)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                job = self._job_row_locked(user_id, job_id)
                if job is None:
                    raise DocumentLeaseError("document job not found")
                if str(job["status"]) == "succeeded":
                    output = self._version_row_locked(
                        user_id,
                        str(job["output_version_id"]),
                    )
                    assert output is not None
                    if (
                        str(output["file_id"]) == output_file_id
                        and str(output["filename"]) == filename
                        and str(output["mime"]) == mime
                        and int(output["bytes"]) == bytes_count
                        and hmac.compare_digest(str(output["content_sha256"]), digest)
                    ):
                        self._conn.commit()
                        return _job_from_row(job), _version_from_row(output)
                    raise DocumentConflictError("document job already published another output")
                lease_expires_at = job["lease_expires_at"]
                if (
                    str(job["status"]) != "running"
                    or not hmac.compare_digest(str(job["lease_id"]), lease_id)
                    or lease_expires_at is None
                    or _instant(str(lease_expires_at))[1] <= now_value
                ):
                    raise DocumentLeaseError("document worker lease is stale")
                if mime != str(job["output_mime"]):
                    raise DocumentConflictError("output mime differs from the approved operation")
                if not self._is_current_version_id_locked(
                    user_id,
                    str(job["input_version_id"]),
                ):
                    self._conn.execute(
                        "UPDATE app_document_jobs SET status = 'failed', "
                        "error_code = 'stale_input', lease_id = '', "
                        "lease_expires_at = NULL, updated_at = ?, completed_at = ? "
                        "WHERE job_id = ?",
                        (now, now, job_id),
                    )
                    self._conn.commit()
                    raise DocumentConflictError("input version became stale")
                latest = self._conn.execute(
                    "SELECT COALESCE(MAX(version_number), 0) FROM app_file_versions "
                    "WHERE user_id = ? AND file_id = ?",
                    (user_id, output_file_id),
                ).fetchone()
                version_number = int(latest[0]) + 1
                version_id = f"fver_{uuid.uuid4().hex}"
                self._conn.execute(
                    "INSERT INTO app_file_versions "
                    "(version_id, user_id, file_id, version_number, parent_version_id, "
                    "filename, mime, bytes, content_sha256, origin, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'derivation', ?)",
                    (
                        version_id,
                        user_id,
                        output_file_id,
                        version_number,
                        job["input_version_id"],
                        filename,
                        mime,
                        bytes_count,
                        digest,
                        now,
                    ),
                )
                self._conn.execute(
                    "UPDATE app_document_jobs SET status = 'succeeded', "
                    "output_version_id = ?, lease_id = '', lease_expires_at = NULL, "
                    "updated_at = ?, completed_at = ? WHERE job_id = ?",
                    (version_id, now, now, job_id),
                )
                self._conn.execute(
                    "INSERT INTO app_file_derivations "
                    "(output_version_id, input_version_id, job_id, user_id, "
                    "operation_digest, worker_version, verified_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        version_id,
                        job["input_version_id"],
                        job_id,
                        user_id,
                        job["operation_digest"],
                        worker_version,
                        now,
                    ),
                )
                job = self._job_row_locked(user_id, job_id)
                output = self._version_row_locked(user_id, version_id)
                assert job is not None and output is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _job_from_row(job), _version_from_row(output)

    async def fail(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_id: str,
        error_code: str,
    ) -> DocumentJobRecord:
        return await asyncio.to_thread(
            self._fail_sync,
            user_id,
            job_id,
            lease_id,
            error_code,
        )

    def _fail_sync(
        self,
        user_id: str,
        job_id: str,
        lease_id: str,
        error_code: str,
    ) -> DocumentJobRecord:
        user_id = _required(user_id, "user id")
        job_id = _required(job_id, "job id")
        lease_id = _required(lease_id, "lease id")
        error_code = _required(error_code, "error code", maximum=100)
        if not _OPERATION_RE.fullmatch(error_code):
            raise InvalidDocumentOperationError("error code must use lowercase snake case")
        now = _utc_now()
        _, now_value = _instant(now)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                job = self._job_row_locked(user_id, job_id)
                lease_expires_at = job["lease_expires_at"] if job is not None else None
                if (
                    job is None
                    or str(job["status"]) != "running"
                    or not hmac.compare_digest(str(job["lease_id"]), lease_id)
                    or lease_expires_at is None
                    or _instant(str(lease_expires_at))[1] <= now_value
                ):
                    raise DocumentLeaseError("document worker lease is stale")
                self._conn.execute(
                    "UPDATE app_document_jobs SET status = 'failed', error_code = ?, "
                    "lease_id = '', lease_expires_at = NULL, updated_at = ?, "
                    "completed_at = ? WHERE job_id = ?",
                    (error_code, now, now, job_id),
                )
                job = self._job_row_locked(user_id, job_id)
                assert job is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _job_from_row(job)

    def _job_row_locked(self, user_id: str, job_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM app_document_jobs WHERE user_id = ? AND job_id = ?",
            (user_id, job_id),
        ).fetchone()

    def _approval_for_job_locked(
        self,
        user_id: str,
        job_id: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM app_document_approvals WHERE user_id = ? AND job_id = ?",
            (user_id, job_id),
        ).fetchone()

    def _version_row_locked(self, user_id: str, version_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM app_file_versions WHERE user_id = ? AND version_id = ?",
            (user_id, version_id),
        ).fetchone()

    def _is_current_version_id_locked(self, user_id: str, version_id: str) -> bool:
        version = self._version_row_locked(user_id, version_id)
        return version is not None and self._is_current_version_locked(version)

    def _is_current_version_locked(self, version: sqlite3.Row) -> bool:
        latest = self._conn.execute(
            "SELECT version_id FROM app_file_versions WHERE user_id = ? AND file_id = ? "
            "ORDER BY version_number DESC LIMIT 1",
            (version["user_id"], version["file_id"]),
        ).fetchone()
        return latest is not None and str(latest[0]) == str(version["version_id"])


def _version_from_row(row: sqlite3.Row) -> FileVersionRecord:
    return FileVersionRecord(
        version_id=str(row["version_id"]),
        user_id=str(row["user_id"]),
        file_id=str(row["file_id"]),
        version_number=int(row["version_number"]),
        parent_version_id=(
            str(row["parent_version_id"]) if row["parent_version_id"] is not None else None
        ),
        filename=str(row["filename"]),
        mime=str(row["mime"]),
        bytes=int(row["bytes"]),
        content_sha256=str(row["content_sha256"]),
        origin=str(row["origin"]),
        created_at=str(row["created_at"]),
    )


def _job_from_row(row: sqlite3.Row) -> DocumentJobRecord:
    raw_arguments = json.loads(str(row["arguments_json"]))
    assert isinstance(raw_arguments, dict)
    return DocumentJobRecord(
        job_id=str(row["job_id"]),
        user_id=str(row["user_id"]),
        input_version_id=str(row["input_version_id"]),
        operation=str(row["operation"]),
        arguments=raw_arguments,
        operation_digest=str(row["operation_digest"]),
        output_mime=str(row["output_mime"]),
        summary=str(row["summary"]),
        preview=str(row["preview"]),
        requested_by_kind=str(row["requested_by_kind"]),
        requested_by_id=str(row["requested_by_id"]),
        idempotency_key=str(row["idempotency_key"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        lease_id=str(row["lease_id"]),
        lease_expires_at=(
            str(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
        ),
        output_version_id=(
            str(row["output_version_id"]) if row["output_version_id"] is not None else None
        ),
        error_code=str(row["error_code"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=(str(row["completed_at"]) if row["completed_at"] is not None else None),
    )


def _approval_from_row(row: sqlite3.Row) -> DocumentApprovalRecord:
    return DocumentApprovalRecord(
        approval_id=str(row["approval_id"]),
        job_id=str(row["job_id"]),
        user_id=str(row["user_id"]),
        operation_digest=str(row["operation_digest"]),
        expires_at=str(row["expires_at"]),
        decision=str(row["decision"]),
        actor_user_id=(
            str(row["actor_user_id"]) if row["actor_user_id"] is not None else None
        ),
        created_at=str(row["created_at"]),
        decided_at=(str(row["decided_at"]) if row["decided_at"] is not None else None),
        used_at=str(row["used_at"]) if row["used_at"] is not None else None,
    )


def _approval_from_joined_row(row: sqlite3.Row) -> DocumentApprovalRecord:
    return DocumentApprovalRecord(
        approval_id=str(row["a_approval_id"]),
        job_id=str(row["job_id"]),
        user_id=str(row["user_id"]),
        operation_digest=str(row["a_operation_digest"]),
        expires_at=str(row["a_expires_at"]),
        decision=str(row["a_decision"]),
        actor_user_id=(
            str(row["a_actor_user_id"]) if row["a_actor_user_id"] is not None else None
        ),
        created_at=str(row["a_created_at"]),
        decided_at=(
            str(row["a_decided_at"]) if row["a_decided_at"] is not None else None
        ),
        used_at=str(row["a_used_at"]) if row["a_used_at"] is not None else None,
    )


__all__ = [
    "DocumentApprovalError",
    "DocumentApprovalExpiredError",
    "DocumentConflictError",
    "DocumentLeaseError",
    "DocumentToolsRepository",
    "InvalidDocumentOperationError",
    "document_operation_digest",
    "normalize_document_arguments",
]
