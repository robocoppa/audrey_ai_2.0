#!/usr/bin/env python3
"""Verify schema 20 approval and lease state around an Audrey restart."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from audrey.app_state import ApplicationStore, DocumentConflictError, DocumentLeaseError

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DATABASE_PATH = Path(os.getenv("AUDREY_APPLICATION_DB", "/data/audrey_app.sqlite"))
SNAPSHOT_PATH = Path(os.getenv(
    "AUDREY_DOCUMENT_APPROVALS_SNAPSHOT_PATH",
    "/data/c3-document-approvals-smoke.json",
))
READY_TIMEOUT_SECONDS = float(os.getenv("AUDREY_DOCUMENT_SMOKE_TIMEOUT_SECONDS", "180"))
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


class SmokeError(RuntimeError):
    """A deployed document approval contract was not satisfied."""


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


def _json_request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json", **_headers(token)}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - operator controls the smoke base URL
        f"{BASE_URL}{path}", data=body, headers=headers, method=method,
    )
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            status, content = response.status, response.read()
    except HTTPError as exc:
        status, content = exc.code, exc.read()
    if status not in expected:
        raise SmokeError(
            f"{method} {path}: HTTP {status}: {content.decode(errors='replace')[:500]}"
        )
    if not content:
        return status, {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"{method} {path} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"{method} {path} returned non-object JSON")
    return status, value


def _wait_ready() -> dict[str, str]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last = ""
    while time.monotonic() < deadline:
        try:
            _, health = _json_request("/health", token=USER_TOKEN)
            _, capabilities = _json_request("/api/capabilities", token=USER_TOKEN)
            if health.get("status") == "ok" and capabilities.get("status") == "ready":
                return {"health": "ok", "capabilities": "ready"}
            last = f"health={health.get('status')!r}, capabilities={capabilities.get('status')!r}"
        except Exception as exc:  # noqa: BLE001 - startup failures are retried
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise SmokeError(f"Audrey did not become ready within {READY_TIMEOUT_SECONDS:g}s: {last}")


def _snapshot_write(value: dict[str, Any]) -> None:
    temporary = SNAPSHOT_PATH.with_suffix(SNAPSHOT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(SNAPSHOT_PATH)


def _snapshot_read() -> dict[str, Any]:
    try:
        value = json.loads(SNAPSHOT_PATH.read_text())
    except FileNotFoundError as exc:
        raise SmokeError(f"snapshot does not exist: {SNAPSHOT_PATH}; run capture first") from exc
    except json.JSONDecodeError as exc:
        raise SmokeError("document approval snapshot is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise SmokeError("document approval snapshot schema is unsupported")
    return value


async def _create_request(
    store: ApplicationStore,
    *,
    user_id: str,
    version_id: str,
    suffix: str,
    label: str,
):
    return await store.document_tools.create_request(
        user_id=user_id,
        input_version_id=version_id,
        operation="template_to_docx",
        arguments={"fields": {"probe": label, "suffix": suffix}},
        output_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        summary=f"Create the {label} restart probe document.",
        preview=f"Probe: {label}",
        idempotency_key=f"smoke-{suffix}-{label}",
        requested_by_kind="model",
        requested_by_id=user_id,
    )


def _decision(job_id: str, digest: str, decision: str, *, expected=frozenset({200})):
    return _json_request(
        f"/api/document-jobs/{quote(job_id, safe='')}/decision",
        token=USER_TOKEN,
        method="POST",
        payload={"decision": decision, "operation_digest": digest},
        expected=expected,
    )


def _cleanup(snapshot: dict[str, Any]) -> dict[str, int]:
    jobs = snapshot.get("jobs", {})
    if not isinstance(jobs, dict):
        raise SmokeError("snapshot omitted document jobs")
    job_ids = [str(row.get("id") or "") for row in jobs.values() if isinstance(row, dict)]
    user_id = str(snapshot.get("user_id") or "")
    versions = [str(snapshot.get("version_id") or "")]
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for job_id in job_ids:
            row = connection.execute(
                "SELECT output_version_id FROM app_document_jobs WHERE user_id = ? AND job_id = ?",
                (user_id, job_id),
            ).fetchone()
            if row is not None and row[0]:
                versions.append(str(row[0]))
        deleted_jobs = sum(max(0, connection.execute(
            "DELETE FROM app_document_jobs WHERE user_id = ? AND job_id = ?",
            (user_id, job_id),
        ).rowcount) for job_id in job_ids)
        deleted_versions = sum(max(0, connection.execute(
            "DELETE FROM app_file_versions WHERE user_id = ? AND version_id = ?",
            (user_id, version_id),
        ).rowcount) for version_id in versions if version_id)
        connection.commit()
    return {"jobs": deleted_jobs, "versions": deleted_versions}


def capture() -> dict[str, Any]:
    if SNAPSHOT_PATH.exists():
        raise SmokeError(f"snapshot already exists: {SNAPSHOT_PATH}; run cleanup first")
    readiness = _wait_ready()
    _, user = _json_request("/api/me", token=USER_TOKEN)
    _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
    user_id = str(user.get("id") or "")
    if not user_id or user_id == str(admin.get("id") or ""):
        raise SmokeError("smoke credentials must resolve to two distinct accounts")
    suffix = uuid.uuid4().hex[:10]
    base_time = dt.datetime.now(dt.UTC)
    snapshot: dict[str, Any] = {"schema": 1, "user_id": user_id, "jobs": {}}
    try:
        store = ApplicationStore(DATABASE_PATH)
        try:
            if store.schema_version != 20:
                raise SmokeError(f"expected schema 20, found {store.schema_version}")
            source = asyncio.run(store.document_tools.register_source_version(
                user_id=user_id,
                file_id=f"smoke_document_{suffix}",
                filename=f"document-approval-smoke-{suffix}.txt",
                mime="text/plain",
                bytes_count=len(suffix),
                content_sha256=_hash(f"source-{suffix}"),
            ))
            snapshot["version_id"] = source.version_id
            pending, pending_approval = asyncio.run(_create_request(
                store, user_id=user_id, version_id=source.version_id,
                suffix=suffix, label="pending",
            ))
            running, running_approval = asyncio.run(_create_request(
                store, user_id=user_id, version_id=source.version_id,
                suffix=suffix, label="running",
            ))
            snapshot["jobs"] = {
                "pending": {"id": pending.job_id, "digest": pending_approval.operation_digest},
                "running": {"id": running.job_id, "digest": running_approval.operation_digest},
            }
        finally:
            store.close()

        altered_http, _ = _decision(
            pending.job_id, "0" * 64, "approved", expected=frozenset({409}),
        )
        cross_owner_http, _ = _json_request(
            f"/api/document-jobs/{quote(pending.job_id, safe='')}",
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )
        _decision(running.job_id, running_approval.operation_digest, "approved")

        store = ApplicationStore(DATABASE_PATH)
        try:
            claimed = asyncio.run(store.document_tools.claim_next(
                lease_id=f"smoke-before-{suffix}",
                lease_seconds=1,
                now=base_time.isoformat(),
            ))
            if claimed is None or claimed.job_id != running.job_id:
                raise SmokeError("approved running probe was not claimed")
            queued, queued_approval = asyncio.run(_create_request(
                store, user_id=user_id, version_id=str(snapshot["version_id"]),
                suffix=suffix, label="queued",
            ))
            snapshot["jobs"]["queued"] = {
                "id": queued.job_id, "digest": queued_approval.operation_digest,
            }
        finally:
            store.close()
        _decision(queued.job_id, queued_approval.operation_digest, "approved")
        snapshot.update({
            "recovery_now": (base_time + dt.timedelta(seconds=2)).isoformat(),
            "lease_before": f"smoke-before-{suffix}",
            "lease_after": f"smoke-after-{suffix}",
            "output_file_id": f"smoke_document_output_{suffix}",
            "output_sha256": _hash(f"output-{suffix}"),
        })
        _snapshot_write(snapshot)
        return {
            "schema": 1,
            "status": "captured",
            "snapshot": str(SNAPSHOT_PATH),
            "readiness": readiness,
            "database_schema": 20,
            "states": {"pending": "awaiting_approval", "queued": "queued", "running": "running"},
            "guards": {"altered_digest_http": altered_http, "cross_owner_http": cross_owner_http},
            "next": "restart Audrey, then run verify from this same checkout",
        }
    except Exception:
        if snapshot.get("jobs"):
            _cleanup(snapshot)
        raise


def verify() -> dict[str, Any]:
    snapshot = _snapshot_read()
    readiness = _wait_ready()
    jobs = snapshot.get("jobs", {})
    if not isinstance(jobs, dict):
        raise SmokeError("snapshot omitted document jobs")
    pending, queued, running = (jobs.get(name, {}) for name in ("pending", "queued", "running"))
    for label, row, expected in (
        ("pending", pending, "awaiting_approval"),
        ("queued", queued, "queued"),
        ("running", running, "running"),
    ):
        _, body = _json_request(
            f"/api/document-jobs/{quote(str(row.get('id') or ''), safe='')}",
            token=USER_TOKEN,
        )
        if body.get("status") != expected:
            raise SmokeError(f"{label} job did not survive restart as {expected}")

    store = ApplicationStore(DATABASE_PATH)
    primary_error: Exception | None = None
    result: dict[str, Any] = {}
    try:
        recovered = asyncio.run(store.document_tools.claim_next(
            lease_id=str(snapshot["lease_after"]),
            lease_seconds=60,
            now=str(snapshot["recovery_now"]),
        ))
        if recovered is None or recovered.job_id != str(running["id"]):
            raise SmokeError("expired running job was not reclaimed first")
        if recovered.attempts != 2:
            raise SmokeError(f"recovered job attempts changed: {recovered.attempts}")
        try:
            asyncio.run(store.document_tools.publish(
                user_id=str(snapshot["user_id"]), job_id=str(running["id"]),
                lease_id=str(snapshot["lease_before"]), output_file_id="stale-output",
                filename="stale.docx", mime=recovered.output_mime, bytes_count=1,
                content_sha256=_hash("stale"), worker_version="smoke/1",
            ))
        except DocumentLeaseError:
            pass
        else:
            raise SmokeError("pre-restart worker lease could still publish")
        succeeded, output = asyncio.run(store.document_tools.publish(
            user_id=str(snapshot["user_id"]), job_id=str(running["id"]),
            lease_id=str(snapshot["lease_after"]),
            output_file_id=str(snapshot["output_file_id"]),
            filename="restart-proof.docx", mime=recovered.output_mime, bytes_count=512,
            content_sha256=str(snapshot["output_sha256"]), worker_version="smoke/1",
        ))
        repeated, repeated_output = asyncio.run(store.document_tools.publish(
            user_id=str(snapshot["user_id"]), job_id=str(running["id"]),
            lease_id=str(snapshot["lease_after"]),
            output_file_id=str(snapshot["output_file_id"]),
            filename="restart-proof.docx", mime=recovered.output_mime, bytes_count=512,
            content_sha256=str(snapshot["output_sha256"]), worker_version="smoke/1",
        ))
        if repeated.output_version_id != output.version_id or repeated_output != output:
            raise SmokeError("idempotent publication returned a different version")
        try:
            asyncio.run(store.document_tools.publish(
                user_id=str(snapshot["user_id"]), job_id=str(running["id"]),
                lease_id=str(snapshot["lease_after"]),
                output_file_id=str(snapshot["output_file_id"]), filename="changed.docx",
                mime=recovered.output_mime, bytes_count=513,
                content_sha256=_hash("changed"), worker_version="smoke/1",
            ))
        except DocumentConflictError:
            pass
        else:
            raise SmokeError("completed job accepted a changed output")
        queued_claim = asyncio.run(store.document_tools.claim_next(
            lease_id=f"queued-{uuid.uuid4().hex}", lease_seconds=60,
        ))
        if queued_claim is None or queued_claim.job_id != str(queued["id"]):
            raise SmokeError("queued job was not claimable after recovered publication")
        _, cancelled = _json_request(
            f"/api/document-jobs/{quote(str(queued['id']), safe='')}/cancel",
            token=USER_TOKEN, method="POST",
        )
        _, rejected = _decision(str(pending["id"]), str(pending["digest"]), "rejected")
        _, completed = _json_request(
            f"/api/document-jobs/{quote(str(running['id']), safe='')}", token=USER_TOKEN,
        )
        if completed.get("status") != "succeeded":
            raise SmokeError("published job was not visible as succeeded through the API")
        result = {
            "schema": 1, "status": "passed", "snapshot": str(SNAPSHOT_PATH),
            "readiness": readiness,
            "persistence": {"pending": True, "queued": True, "running": True},
            "recovery": {"attempts": recovered.attempts, "stale_worker_blocked": True},
            "publication": {
                "status": succeeded.status, "output_version_prefix": "fver_",
                "single_version": True, "changed_output_blocked": True,
            },
            "terminal": {"pending": rejected.get("status"), "queued": cancelled.get("status")},
        }
    except Exception as exc:  # noqa: BLE001 - cleanup follows every failure
        primary_error = exc
    finally:
        store.close()
    cleanup_error: Exception | None = None
    try:
        result["cleanup"] = _cleanup(snapshot)
        SNAPSHOT_PATH.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - report cleanup with the main failure
        cleanup_error = exc
    if primary_error is not None and cleanup_error is not None:
        raise SmokeError(f"{primary_error}; cleanup also failed: {cleanup_error}")
    if primary_error is not None:
        raise primary_error
    if cleanup_error is not None:
        raise SmokeError(f"cleanup failed: {cleanup_error}")
    return result


def cleanup() -> dict[str, Any]:
    cleaned = _cleanup(_snapshot_read())
    SNAPSHOT_PATH.unlink(missing_ok=True)
    return {"schema": 1, "status": "cleaned", "cleanup": cleaned}


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if READY_TIMEOUT_SECONDS <= 0:
        print("AUDREY_DOCUMENT_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2
    if len(sys.argv) != 2 or sys.argv[1] not in {"capture", "verify", "cleanup"}:
        print("usage: smoke_native_document_approvals.py capture|verify|cleanup", file=sys.stderr)
        return 2
    try:
        action = sys.argv[1]
        result = capture() if action == "capture" else verify() if action == "verify" else cleanup()
    except Exception as exc:  # noqa: BLE001 - emit one structured smoke failure
        print(json.dumps({
            "schema": 1, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
