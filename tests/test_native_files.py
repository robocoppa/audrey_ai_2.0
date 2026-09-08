"""Native file resources keep Audrey's proven storage lifecycle owner-bound."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey.identity import Principal
from audrey.kb.uploads_db import QuotaUsage
from audrey.routes import files as upload_routes
from audrey.routes.app import files as native_files
from audrey.routes.app import router


def _principal() -> Principal:
    return Principal(
        user_id="usr_123",
        storage_namespace="private-storage-123",
        provider="cloudflare_access",
        provider_subject="provider-subject",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        status="active",
        auth_method="cloudflare_access",
    )


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.storage_lifecycle = SimpleNamespace(usage=_usage)
    app.dependency_overrides[native_files._files_access] = _principal
    return app


async def _usage(user: str) -> QuotaUsage:
    assert user == "private-storage-123"
    return QuotaUsage(
        stored_bytes=42,
        chunk_declared_bytes=0,
        chunk_part_bytes=0,
        chunk_part_overage_bytes=0,
        url_fetch_bytes=0,
        single_shot_bytes=0,
    )


def _listing() -> upload_routes.ListResponse:
    return upload_routes.ListResponse(
        user="private-storage-123",
        files=[
            upload_routes.FileRow(
                file_id="file_123",
                filename="field-notes.txt",
                mime="text/plain",
                bytes=42,
                uploaded_at="2026-09-07T12:00:00+00:00",
                chunks=1,
                status="ready",
            ),
        ],
        total_bytes=42,
        server_time="2026-09-07T12:01:00+00:00",
        limits=upload_routes.Limits(
            max_upload_bytes=50_000_000,
            max_user_bytes=1_000_000_000,
            allowed_extensions=[".txt"],
            chunked_max_bytes=2_000_000_000,
            part_size=8_000_000,
            fetch_hosts=["example.com"],
        ),
    )


def test_native_list_uses_server_owned_namespace_and_hides_compat_fields(monkeypatch):
    captured = SimpleNamespace(user="")

    async def fake_list(request, me):
        captured.user = me.email
        return _listing()

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)

    response = TestClient(_app()).get("/api/files")

    assert response.status_code == 200
    assert captured.user == "private-storage-123"
    assert response.json() == {
        "items": [
            {
                "id": "file_123",
                "filename": "field-notes.txt",
                "mime": "text/plain",
                "bytes": 42,
                "uploaded_at": "2026-09-07T12:00:00+00:00",
                "kind": "text",
                "chunks": 1,
                "status": "ready",
                "failure_reason": "",
                "duration_s": 0.0,
                "summary": "",
                "source_freed_at": "",
                "leased_at": "",
                "source_url": "",
                "transcript_source": "",
                "fetch_downloaded_bytes": 0,
                "fetch_total_bytes": 0,
            },
        ],
        "total_bytes": 42,
        "server_time": "2026-09-07T12:01:00+00:00",
        "limits": {
            "max_upload_bytes": 50_000_000,
            "max_user_bytes": 1_000_000_000,
            "allowed_extensions": [".txt"],
            "chunked_max_bytes": 2_000_000_000,
            "part_size": 8_000_000,
        },
    }
    assert "private-storage-123" not in response.text
    assert "collection" not in response.text
    assert "fetch_hosts" not in response.text


def test_native_get_returns_only_a_file_in_the_owner_listing(monkeypatch):
    async def fake_list(request, me):
        return _listing()

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    client = TestClient(_app())

    assert client.get("/api/files/file_123").status_code == 200
    missing = client.get("/api/files/another-users-file")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "File not found."}


async def test_attachment_resolution_is_ready_owner_bound_and_indistinguishable(
    monkeypatch,
):
    listing = _listing()
    listing.files.append(
        upload_routes.FileRow(
            file_id="file_processing",
            filename="still-processing.mp4",
            mime="video/mp4",
            bytes=100,
            uploaded_at="2026-09-07T12:00:00+00:00",
            chunks=0,
            status="processing",
        )
    )

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    request = SimpleNamespace()
    resolved = await native_files.resolve_owned_attachments(
        request,
        _principal(),
        ["file_123"],
    )
    assert resolved[0].filename == "field-notes.txt"
    assert resolved[0].kind == "text"

    for invalid in (
        ["another-users-file"],
        ["file_processing"],
        ["file_123", "file_123"],
    ):
        with pytest.raises(HTTPException) as caught:
            await native_files.resolve_owned_attachments(
                request,
                _principal(),
                invalid,
            )
        assert caught.value.status_code == 422
        assert "owned by you" in caught.value.detail


def test_native_upload_delegates_bytes_with_server_owned_identity(monkeypatch):
    captured = SimpleNamespace(user="", content=b"")

    async def fake_upload(*, request, me, file):
        captured.user = me.email
        captured.content = await file.read()
        return upload_routes.UploadResponse(
            file_id="file_uploaded",
            filename=file.filename,
            mime="text/plain",
            bytes=len(captured.content),
            kind="text",
            collection="private-internal-collection",
            chunks=1,
            status="ready",
        )

    monkeypatch.setattr(native_files.upload_routes, "upload_file", fake_upload)

    response = TestClient(_app()).post(
        "/api/files",
        files={"file": ("notes.txt", b"native bytes", "text/plain")},
    )

    assert response.status_code == 200
    assert captured.user == "private-storage-123"
    assert captured.content == b"native bytes"
    assert response.json() == {
        "id": "file_uploaded",
        "filename": "notes.txt",
        "mime": "text/plain",
        "bytes": 12,
        "kind": "text",
        "chunks": 1,
        "status": "ready",
    }
    assert "collection" not in response.text


def test_native_delete_preserves_deferred_cleanup_and_uses_404_for_unknown(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_delete(*, file_id, request, me):
        calls.append((file_id, me.email))
        return upload_routes.DeleteResponse(
            file_id=file_id,
            deleted=file_id == "known",
            pending_cleanup=file_id == "known",
        )

    monkeypatch.setattr(native_files.upload_routes, "delete_file", fake_delete)
    client = TestClient(_app())

    deferred = client.delete("/api/files/known")
    missing = client.delete("/api/files/unknown")

    assert deferred.status_code == 200
    assert deferred.json() == {
        "id": "known",
        "deleted": True,
        "pending_cleanup": True,
    }
    assert missing.status_code == 404
    assert calls == [
        ("known", "private-storage-123"),
        ("unknown", "private-storage-123"),
    ]


def test_native_files_require_authentication():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).get("/api/files")

    assert response.status_code == 401
