"""Contract tests for the two-stage document-approval smoke."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parent / "smoke"
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

import smoke_native_document_approvals as smoke  # noqa: E402


def test_readiness_uses_backend_health_and_ui_capabilities(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_json_request(path, *, token, absolute_url=""):
        calls.append((path, token, absolute_url))
        if absolute_url:
            return 200, {"status": "ok"}
        return 200, {"status": "ready"}

    monkeypatch.setattr(smoke, "BACKEND_HEALTH_URL", "http://audrey:8000/health")
    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke._wait_ready() == {
        "health": "ok",
        "capabilities": "ready",
    }
    assert calls == [
        ("", smoke.USER_TOKEN, "http://audrey:8000/health"),
        ("/api/capabilities", smoke.USER_TOKEN, ""),
    ]


def test_readiness_fails_immediately_when_access_assertion_is_rejected(monkeypatch):
    calls: list[str] = []

    def fake_json_request(path, *, token, absolute_url=""):
        calls.append(absolute_url or path)
        if absolute_url:
            return 200, {"status": "ok"}
        raise smoke.SmokeCredentialError(
            "HTTP 401; refresh application assertions in .env.smoke.local"
        )

    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    with pytest.raises(smoke.SmokeCredentialError, match=r"\.env\.smoke\.local"):
        smoke._wait_ready()

    assert calls == [
        smoke.BACKEND_HEALTH_URL,
        "/api/capabilities",
    ]


def test_cleanup_removes_derived_output_before_its_source(tmp_path, monkeypatch):
    database_path = tmp_path / "app.sqlite"
    store = smoke.ApplicationStore(database_path)
    owner = asyncio.run(store.resolve_external_identity(
        provider="owui",
        subject="cleanup-owner",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace="alice@example.com",
    ))
    source = asyncio.run(store.document_tools.register_source_version(
        user_id=owner.user_id,
        file_id="smoke_cleanup_source",
        filename="source.txt",
        mime="text/plain",
        bytes_count=6,
        content_sha256=smoke._hash("source"),
    ))
    job, approval = asyncio.run(smoke._create_request(
        store,
        user_id=owner.user_id,
        version_id=source.version_id,
        suffix="cleanup",
        label="published",
    ))
    asyncio.run(store.document_tools.decide(
        user_id=owner.user_id,
        job_id=job.job_id,
        actor_user_id=owner.user_id,
        operation_digest=approval.operation_digest,
        decision="approved",
    ))
    claimed = asyncio.run(store.document_tools.claim_next(
        lease_id="cleanup-worker",
        lease_seconds=60,
    ))
    assert claimed is not None
    _completed, output = asyncio.run(store.document_tools.publish(
        user_id=owner.user_id,
        job_id=job.job_id,
        lease_id="cleanup-worker",
        output_file_id="smoke_cleanup_output",
        filename="output.docx",
        mime=job.output_mime,
        bytes_count=12,
        content_sha256=smoke._hash("output"),
        worker_version="smoke-test/1",
    ))
    assert output.parent_version_id == source.version_id
    store.close()

    monkeypatch.setattr(smoke, "DATABASE_PATH", database_path)
    result = smoke._cleanup({
        "schema": 1,
        "user_id": owner.user_id,
        "version_id": source.version_id,
        "jobs": {"running": {"id": job.job_id}},
    })

    assert result == {"jobs": 1, "versions": 2}
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM app_document_jobs"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM app_file_versions"
        ).fetchone()[0] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
