"""Durable file restrictions block model reuse, preserving owned browser views."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from audrey.app_state import ApplicationStore
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes.app import files as native_files
from audrey.routes.app import projects as native_projects
from audrey.routes.app import router


async def _record_upload(db: UploadsDB, *, private_required: bool, **values) -> None:
    if not private_required:
        await db.record_upload(**values)
        return
    stamp = values["uploaded_at"]
    quota = 1_000_000
    reservation = await db.reserve_single_upload(
        reservation_id=values["file_id"], user=values["user"],
        bytes_=values["bytes_"], max_user_bytes=quota, now=stamp,
        expired_before="2026-10-18T12:00:00+00:00", private_required=True,
    )
    assert reservation.accepted
    status = values.pop("status", "ready")
    committed = await db.commit_reserved_upload(
        **values, reservation_id=values["file_id"], reservation_kind="single_shot",
        max_user_bytes=quota, status=status,
    )
    assert committed.accepted


@pytest.fixture
async def resources(tmp_path):
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    store = ApplicationStore(tmp_path / "app.sqlite")
    principal = await store.resolve_external_identity(
        provider="owui", subject="private-file-owner", email="owner@example.com",
        display_name="Owner", role="user", auth_method="owui_bearer",
        legacy_storage_namespace="owner@example.com",
    )
    root = tmp_path / "uploads"
    directory = root / sanitize_user(principal.storage_namespace)
    directory.mkdir(parents=True)
    for file_id, protected in (("protected-doc", True), ("ordinary-doc", False)):
        content = "An owned document with a confirmed invoice total of 42."
        (directory / f"{file_id}.txt").write_text(content, encoding="utf-8")
        (directory / f"{file_id}.summary.txt").write_text("A short summary.", encoding="utf-8")
        await _record_upload(uploads,
            file_id=file_id, user=principal.storage_namespace,
            filename=f"{file_id}.txt", mime="text/plain", bytes_=len(content),
            kind="text", collection="test-text", chunks=1,
            uploaded_at="2026-10-19T12:00:00+00:00", private_required=protected,
        )
    for file_id, protected in (("protected-image", True), ("ordinary-image", False)):
        path = directory / f"{file_id}.png"
        Image.new("RGB", (20, 20), "red").save(path)
        await _record_upload(uploads,
            file_id=file_id, user=principal.storage_namespace,
            filename=f"{file_id}.png", mime="image/png", bytes_=path.stat().st_size,
            kind="image", collection="test-images", chunks=1,
            uploaded_at="2026-10-19T12:00:00+00:00", private_required=protected,
        )
    app = FastAPI()
    app.include_router(router)
    app.state.application_store = store
    app.state.uploads_db = uploads
    app.state.cfg = SimpleNamespace(raw={"kb": {"upload_root": str(root)}})
    app.dependency_overrides[native_files._files_access] = lambda: principal
    app.dependency_overrides[native_projects._project_access] = lambda: principal
    try:
        yield SimpleNamespace(
            app=app, request=SimpleNamespace(app=app), uploads=uploads,
            store=store, principal=principal,
        )
    finally:
        uploads.close()
        store.close()


def _assert_private_refusal(error: HTTPException) -> None:
    assert error.status_code == 409
    assert error.detail == {
        "error": "private_processing_unavailable",
        "message": "Private processing is not available yet for this operation.",
    }


@pytest.mark.parametrize("active_policy", [False, True])
@pytest.mark.parametrize(
    ("file_id", "reader", "core_reader"),
    [
        ("protected-doc", "read_owned_document_text", "_read_document_text"),
        ("protected-image", "read_owned_image_preview", "_read_image_preview"),
    ],
)
async def test_model_reader_refuses_before_disk_work_even_in_policy_scope(
    resources, monkeypatch, active_policy, file_id, reader, core_reader,
):
    core = AsyncMock(side_effect=AssertionError("Protected bytes reached a reader"))
    path = Mock(side_effect=AssertionError("Protected source path was accessed"))
    monkeypatch.setattr(native_files, core_reader, core)
    monkeypatch.setattr(native_files, "_source_path", path)
    policy = ProcessingPolicy({"test-local:latest": "a" * 64}, "http://ollama:11434")
    with processing_scope(policy if active_policy else None):
        with pytest.raises(HTTPException) as caught:
            await getattr(native_files, reader)(
                resources.request, resources.principal, file_id,
            )
    _assert_private_refusal(caught.value)
    core.assert_not_called()
    path.assert_not_called()


@pytest.mark.parametrize("file_id", ["protected-doc", "protected-image"])
async def test_protected_attachment_never_becomes_a_snapshot(resources, file_id):
    with pytest.raises(HTTPException) as caught:
        await native_files.resolve_owned_attachments(
            resources.request, resources.principal, ["ordinary-doc", file_id],
        )
    _assert_private_refusal(caught.value)
    admitted = await native_files.resolve_owned_attachments(
        resources.request, resources.principal, ["ordinary-doc", "ordinary-image"],
    )
    assert [item.file_id for item in admitted] == ["ordinary-doc", "ordinary-image"]


async def test_ordinary_model_readers_still_read_owned_evidence(resources):
    document = await native_files.read_owned_document_text(
        resources.request, resources.principal, "ordinary-doc",
    )
    image = await native_files.read_owned_image_preview(
        resources.request, resources.principal, "ordinary-image",
    )
    assert "invoice total of 42" in document.text
    assert image.startswith(b"\xff\xd8")


async def test_owner_browser_can_view_and_download_protected_document(resources, monkeypatch):
    model_reader = AsyncMock(side_effect=AssertionError("Viewer entered model admission"))
    monkeypatch.setattr(native_files, "read_owned_document_text", model_reader)
    with TestClient(resources.app) as client:
        listing = client.get("/api/files")
        assert listing.status_code == 200
        assert "protected-doc" in {item["id"] for item in listing.json()["items"]}
        metadata = client.get("/api/files/protected-doc")
        assert metadata.status_code == 200
        assert "private_required" not in metadata.json()
        text = client.get("/api/files/protected-doc/text")
        assert text.status_code == 200
        assert "invoice total of 42" in text.json()["text"]
        assert text.headers["cache-control"] == "private, no-store"
        download = client.get("/api/files/protected-doc/download")
        assert download.status_code == 200
        assert "invoice total of 42" in download.text
        summary = client.get("/api/files/protected-doc/artifacts/summary")
        assert summary.status_code == 200
        assert summary.json()["text"] == "A short summary."
        artifact = client.get("/api/files/protected-doc/artifacts/summary/download")
        assert artifact.status_code == 200
        assert artifact.text == "A short summary."
    model_reader.assert_not_called()


async def test_owner_browser_can_view_and_download_protected_image(resources, monkeypatch):
    model_reader = AsyncMock(side_effect=AssertionError("Viewer entered model admission"))
    monkeypatch.setattr(native_files, "read_owned_image_preview", model_reader)
    with TestClient(resources.app) as client:
        preview = client.get("/api/files/protected-image/image")
        assert preview.status_code == 200
        assert preview.content.startswith(b"\xff\xd8")
        assert preview.headers["cache-control"] == "private, no-store"
        download = client.get("/api/files/protected-image/download")
        assert download.status_code == 200
        assert download.content.startswith(b"\x89PNG")
    model_reader.assert_not_called()


async def test_foreign_owner_cannot_view_or_reuse_protected_files(resources):
    foreign = await resources.store.resolve_external_identity(
        provider="owui", subject="foreign-file-owner", email="foreign@example.com",
        display_name="Foreign", role="user", auth_method="owui_bearer",
        legacy_storage_namespace="foreign@example.com",
    )
    resources.app.dependency_overrides[native_files._files_access] = lambda: foreign
    with TestClient(resources.app) as client:
        for suffix in ("", "/text", "/download", "/artifacts/summary"):
            assert client.get(f"/api/files/protected-doc{suffix}").status_code == 404
        assert client.get("/api/files/protected-image/image").status_code == 404
    with pytest.raises(HTTPException) as caught:
        await native_files.read_owned_document_text(resources.request, foreign, "protected-doc")
    assert caught.value.status_code == 404


@pytest.mark.parametrize("status", ["ready", "pending", "processing"])
async def test_protected_file_cannot_be_linked_to_ordinary_project(resources, status):
    file_id = f"protected-{status}"
    await _record_upload(resources.uploads,
        file_id=file_id, user=resources.principal.storage_namespace,
        filename="protected.txt", mime="text/plain", bytes_=42,
        kind="text", collection="test-text", chunks=1,
        uploaded_at="2026-10-19T12:00:00+00:00", status=status,
        private_required=True,
    )
    project = await resources.store.projects.create(
        user_id=resources.principal.user_id, name="Ordinary research",
    )
    url = f"/api/projects/{project.project_id}/files"
    with TestClient(resources.app) as client:
        refused = client.post(url, json={"file_id": file_id})
        assert refused.status_code == 409
        assert refused.json()["detail"]["error"] == "private_processing_unavailable"
        listed = client.get(url)
        assert listed.status_code == 200
        assert listed.json()["items"] == []
        ordinary = client.post(url, json={"file_id": "ordinary-doc"})
        assert ordinary.status_code == 201
        assert ordinary.json()["id"] == "ordinary-doc"
