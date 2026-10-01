"""Native file resources keep Audrey's proven storage lifecycle owner-bound."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

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

    app = _app()
    app.state.cfg = SimpleNamespace(raw={"vision": {"max_images_per_turn": 2}})
    response = TestClient(app).get("/api/files")

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
            "fetch_hosts": ["example.com"],
            "max_images_per_turn": 2,
        },
    }
    assert "private-storage-123" not in response.text
    assert "collection" not in response.text
    assert "fetch_hosts" in response.text


def test_native_file_list_briefs_an_existing_verbose_video_summary(monkeypatch):
    listing = _listing()
    listing.files[0].mime = "video/mp4"
    listing.files[0].summary = (
        "Let me analyze this video. "
        "A Minecraft tutorial demonstrates a way to grow trees close together. "
        "The player places saplings beside a compact structure for easier harvesting. "
        "The method helps collect wood efficiently in a small area. "
        "The streamer is wearing earphones and a dark shirt."
    )

    async def fake_list(request, me):
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)

    response = TestClient(_app()).get("/api/files")

    assert response.status_code == 200
    assert response.json()["items"][0]["summary"] == (
        "A Minecraft tutorial demonstrates a way to grow trees close together. "
        "The player places saplings beside a compact structure for easier harvesting. "
        "The method helps collect wood efficiently in a small area."
    )


def test_native_get_returns_only_a_file_in_the_owner_listing(monkeypatch):
    async def fake_list(request, me):
        return _listing()

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    client = TestClient(_app())

    assert client.get("/api/files/file_123").status_code == 200
    missing = client.get("/api/files/another-users-file")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "File not found."}


def test_native_original_download_is_owner_bound_attachment_with_ranges(monkeypatch, tmp_path):
    listing = _listing()
    listing.files[0].filename = "field notes ü.txt"
    content = b"private original bytes"
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_123.txt").write_bytes(content)

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    response = client.get("/api/files/file_123/download")
    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert (
        "filename*=utf-8''field%20notes%20%C3%BC.txt"
        in response.headers["content-disposition"]
    )
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"

    partial = client.get("/api/files/file_123/download", headers={"Range": "bytes=8-15"})
    assert partial.status_code == 206
    assert partial.content == content[8:16]
    assert partial.headers["content-range"] == f"bytes 8-15/{len(content)}"

    foreign = client.get("/api/files/another-users-file/download")
    assert foreign.status_code == 404
    assert foreign.json() == {"detail": "File not found."}


def test_native_original_download_reports_reclaimed_or_missing_source(monkeypatch, tmp_path):
    listing = _listing()

    async def fake_list(request, me):
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    missing = client.get("/api/files/file_123/download")
    assert missing.status_code == 410
    assert missing.json() == {"detail": "Stored original is unavailable."}

    listing.files[0].source_freed_at = "2026-09-01T00:00:00+00:00"
    reclaimed = client.get("/api/files/file_123/download")
    assert reclaimed.status_code == 410
    assert reclaimed.json() == {"detail": "Stored original is unavailable."}


def test_native_document_text_is_owner_bound_and_paged_without_gaps(monkeypatch, tmp_path):
    listing = _listing()
    content = "A" * 3999 + "é" + "B" * 17
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_123.txt").write_text(content, encoding="utf-8")
    foreign_dir = tmp_path / upload_routes.sanitize_user("another-owner")
    foreign_dir.mkdir()
    (foreign_dir / "file_foreign.txt").write_text("private", encoding="utf-8")

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    first = client.get("/api/files/file_123/text")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "private, no-store"
    assert first.json() == {
        "id": "file_123",
        "text": content[:4000],
        "offset": 0,
        "next_offset": 4000,
        "total_chars": len(content),
    }
    rest = client.get("/api/files/file_123/text?offset=4000")
    assert rest.json()["text"] == content[4000:]
    assert rest.json()["next_offset"] is None
    assert first.json()["text"] + rest.json()["text"] == content
    assert client.get("/api/files/file_foreign/text").status_code == 404
    assert client.get("/api/files/file_123/text?offset=-1").status_code == 422

    (owner_dir / "file_123.txt").unlink()
    assert client.get("/api/files/file_123/text").status_code == 410
    listing.files[0].mime = "image/png"
    assert client.get("/api/files/file_123/text").status_code == 422


def test_native_scanned_pdf_reader_uses_derived_ocr_text(monkeypatch, tmp_path):
    listing = _listing()
    listing.files[0].filename = "scan.pdf"
    listing.files[0].mime = "application/pdf"
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_123.pdf").write_bytes(b"%PDF image-only fixture")
    (owner_dir / "file_123.ocr.txt").write_text(
        "--- Page 1 ---\nRecognized invoice total: $42",
        encoding="utf-8",
    )

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)

    response = TestClient(_app()).get("/api/files/file_123/text")

    assert response.status_code == 200
    assert response.json()["text"] == "--- Page 1 ---\nRecognized invoice total: $42"


def test_native_image_preview_is_owner_bound_resized_and_strips_metadata(
    monkeypatch, tmp_path,
):
    listing = _listing()
    listing.files[0].filename = "portrait.png"
    listing.files[0].mime = "image/png"
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private", "secret-image-metadata")
    Image.new("RGBA", (2000, 1000), (255, 0, 0, 128)).save(
        owner_dir / "file_123.png", pnginfo=metadata,
    )

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    preview = client.get("/api/files/file_123/image")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.headers["cache-control"] == "private, no-store"
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert b"secret-image-metadata" not in preview.content
    with Image.open(io.BytesIO(preview.content)) as image:
        assert image.size == (1600, 800)
        assert image.mode == "RGB"
    assert client.get("/api/files/file_foreign/image").status_code == 404

    (owner_dir / "file_123.png").unlink()
    assert client.get("/api/files/file_123/image").status_code == 410
    listing.files[0].status = "processing"
    assert client.get("/api/files/file_123/image").status_code == 422


def test_native_video_artifact_uses_exact_owned_id_and_pages_on_lines(monkeypatch, tmp_path):
    listing = _listing()
    listing.files[0].file_id = "file_first"
    listing.files[0].filename = "same-name.mp4"
    listing.files[0].mime = "video/mp4"
    listing.files.append(
        upload_routes.FileRow(
            file_id="file_second",
            filename="same-name.mp4",
            mime="video/mp4",
            bytes=100,
            uploaded_at="2026-09-07T13:00:00+00:00",
            chunks=1,
            status="ready",
        )
    )

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_first.transcript.txt").write_text("A" * 4000 + "\nsecond line")
    (owner_dir / "file_second.transcript.txt").write_text("different file")
    (owner_dir / "file_foreign.transcript.txt").write_text("private")

    client = TestClient(_app())
    first = client.get("/api/files/file_first/artifacts/transcript")
    assert first.status_code == 200
    assert first.json() == {
        "id": "file_first",
        "artifact": "transcript",
        "text": "A" * 4000 + "\n",
        "offset": 0,
        "next_offset": 4001,
        "total_chars": 4012,
    }
    second_page = client.get("/api/files/file_first/artifacts/transcript?offset=4001")
    assert second_page.status_code == 200
    assert second_page.json()["text"] == "second line"
    assert second_page.json()["next_offset"] is None

    second = client.get("/api/files/file_second/artifacts/transcript")
    assert second.json()["text"] == "different file"
    foreign = client.get("/api/files/file_foreign/artifacts/transcript")
    assert foreign.status_code == 404
    assert foreign.json() == {"detail": "File not found."}


@pytest.mark.parametrize(
    ("filename", "mime"),
    [
        ("interview.mp3", "audio/mpeg"),
        ("interview.wav", "audio/x-wav"),
        ("interview.m4a", "audio/x-m4a"),
        ("interview.flac", "audio/flac"),
    ],
)
def test_native_audio_artifacts_offer_transcript_and_summary_but_not_visual(
    monkeypatch, tmp_path, filename, mime,
):
    listing = _listing()
    listing.files[0].filename = filename
    listing.files[0].mime = mime
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_123.transcript.txt").write_text("Spoken words.", encoding="utf-8")
    (owner_dir / "file_123.summary.txt").write_text("A short interview.", encoding="utf-8")

    async def fake_list(request, me):
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    assert client.get("/api/files").json()["items"][0]["kind"] == "audio"
    assert client.get("/api/files/file_123/artifacts/transcript").json()["text"] == (
        "Spoken words."
    )
    assert client.get("/api/files/file_123/artifacts/summary").json()["text"] == (
        "A short interview."
    )
    assert client.get("/api/files/file_123/artifacts/visual").status_code == 422
    assert client.get(
        "/api/files/file_123/artifacts/transcript/download"
    ).status_code == 200


def test_native_pdf_offers_summary_as_an_artifact(monkeypatch, tmp_path):
    listing = _listing()
    listing.files[0].filename = "inspection.pdf"
    listing.files[0].mime = "application/pdf"
    listing.files[0].summary = "The inspection recommends replacing the western seal."
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    (owner_dir / "file_123.summary.txt").write_text(
        listing.files[0].summary,
        encoding="utf-8",
    )

    async def fake_list(request, me):
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    summary = client.get("/api/files/file_123/artifacts/summary")
    assert summary.status_code == 200
    assert summary.json()["text"] == listing.files[0].summary
    assert client.get("/api/files/file_123/artifacts/transcript").status_code == 422
    assert client.get("/api/files/file_123/artifacts/visual").status_code == 422


def test_native_video_artifact_downloads_existing_sidecars_after_source_reclamation(
    monkeypatch, tmp_path,
):
    listing = _listing()
    listing.files[0].filename = "field notes ü.mp4"
    listing.files[0].mime = "video/mp4"
    listing.files[0].source_freed_at = "2026-09-01T00:00:00+00:00"

    async def fake_list(request, me):
        assert me.email == "private-storage-123"
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    owner_dir = tmp_path / upload_routes.sanitize_user("private-storage-123")
    owner_dir.mkdir()
    artifacts = {
        "transcript": ("transcript", "What was said."),
        "visual": ("visual-notes", "What was shown."),
        "summary": ("summary", "What the video covers."),
    }
    for artifact, text_content in {
        "transcript.txt": artifacts["transcript"][1],
        "frames.txt": artifacts["visual"][1],
        "summary.txt": artifacts["summary"][1],
    }.items():
        (owner_dir / f"file_123.{artifact}").write_text(text_content, encoding="utf-8")

    client = TestClient(_app())
    for artifact, (suffix, expected_text) in artifacts.items():
        response = client.get(f"/api/files/file_123/artifacts/{artifact}/download")
        assert response.status_code == 200
        assert response.content == expected_text.encode()
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.headers["content-disposition"].startswith("attachment;")
        assert (
            f"filename*=utf-8''field%20notes%20%C3%BC.{suffix}.txt"
            in response.headers["content-disposition"]
        )
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"

    foreign = client.get("/api/files/file_foreign/artifacts/transcript/download")
    assert foreign.status_code == 404
    assert foreign.json() == {"detail": "File not found."}

    (owner_dir / "file_123.summary.txt").unlink()
    unavailable = client.get("/api/files/file_123/artifacts/summary/download")
    assert unavailable.status_code == 404
    assert unavailable.json() == {"detail": "Artifact is unavailable."}

    (owner_dir / "file_123.transcript.txt").write_text("", encoding="utf-8")
    empty = client.get("/api/files/file_123/artifacts/transcript/download")
    assert empty.status_code == 404
    assert empty.json() == {"detail": "Artifact is unavailable."}


def test_native_video_artifact_absence_and_validation(monkeypatch, tmp_path):
    listing = _listing()
    listing.files[0].mime = "video/mp4"

    async def fake_list(request, me):
        return listing

    monkeypatch.setattr(native_files.upload_routes, "list_files", fake_list)
    monkeypatch.setattr(native_files.upload_routes, "_upload_root", lambda request: tmp_path)
    client = TestClient(_app())

    absent = client.get("/api/files/file_123/artifacts/visual")
    assert absent.status_code == 200
    assert absent.json() == {
        "id": "file_123",
        "artifact": "visual",
        "text": "",
        "offset": 0,
        "next_offset": None,
        "total_chars": 0,
    }
    assert client.get("/api/files/file_123/artifacts/unknown").status_code == 422
    assert client.get("/api/files/file_123/artifacts/transcript?offset=-1").status_code == 422
    listing.files[0].mime = "text/plain"
    assert client.get("/api/files/file_123/artifacts/transcript").status_code == 422
    assert client.get("/api/files/file_123/artifacts/transcript/download").status_code == 422


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


def test_native_url_fetch_queues_for_authenticated_owner_without_compat_fields(monkeypatch):
    captured = SimpleNamespace(user="", url="")

    async def fake_fetch(*, body, request, me):
        captured.user = me.email
        captured.url = body.url
        return upload_routes.UploadResponse(
            file_id="file_video",
            filename="video-id",
            mime="",
            bytes=0,
            kind="video",
            collection="private-internal-collection",
            chunks=0,
            status="fetch_pending",
        )

    monkeypatch.setattr(native_files.upload_routes, "ingest_from_url", fake_fetch)
    response = TestClient(_app()).post(
        "/api/files/from-url",
        json={"url": "https://example.com/watch?v=video-id", "user": "someone-else"},
    )

    assert response.status_code == 200
    assert captured.user == "private-storage-123"
    assert captured.url == "https://example.com/watch?v=video-id"
    assert response.json() == {
        "id": "file_video",
        "filename": "video-id",
        "mime": "",
        "bytes": 0,
        "kind": "video",
        "chunks": 0,
        "status": "fetch_pending",
    }
    assert "collection" not in response.text


def test_native_url_fetch_preserves_duplicate_and_host_rejections(monkeypatch):
    async def rejected(*, body, request, me):
        raise HTTPException(
            status_code=409 if "duplicate" in body.url else 403,
            detail="Already queued" if "duplicate" in body.url else "Host not allowed",
        )

    monkeypatch.setattr(native_files.upload_routes, "ingest_from_url", rejected)
    client = TestClient(_app())
    duplicate = client.post("/api/files/from-url", json={"url": "https://example.com/duplicate"})
    blocked = client.post("/api/files/from-url", json={"url": "https://evil.test/video"})

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Already queued"
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "Host not allowed"


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

    client = TestClient(app)
    assert client.get("/api/files").status_code == 401
    assert client.get("/api/files/file_123/download").status_code == 401
    assert client.get("/api/files/file_123/artifacts/transcript/download").status_code == 401
    assert client.post("/api/files/from-url", json={"url": "https://example.com/video"}).status_code == 401
