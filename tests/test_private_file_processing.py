"""Protected uploads cannot enter the unfinished ordinary processing paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey.auth import AuthedUser, require_service, require_user
from audrey.kb.storage_lifecycle import StorageLifecycle, StorageReservation
from audrey.kb.uploads_db import UploadsDB
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes import files

USER = "alice@example.com"
STAMP = "2026-10-19T10:00:00+00:00"
OLD = "2026-10-19T09:00:00+00:00"


@pytest.fixture
def db(tmp_path):
    index = UploadsDB(tmp_path / "uploads.sqlite")
    yield index
    index.close()


def _app(db, tmp_path):
    app = FastAPI()
    app.include_router(files.router)
    app.state.uploads_db = db
    app.state.qdrant = AsyncMock()
    app.state.text_embedder = AsyncMock()
    app.state.image_embedder = AsyncMock()
    app.state.cfg = SimpleNamespace(raw={"kb": {
        "upload_root": str(tmp_path / "uploads"),
        "video": {"lease_minutes": 30, "max_attempts": 3},
    }})
    app.dependency_overrides[require_service] = lambda: None
    app.dependency_overrides[require_user] = lambda: AuthedUser(
        email=USER, role="user", display_name="Alice",
    )
    return app


async def _seed(db, *, private=True, file_id="protected", filename="private-notes.txt",
                kind="text", mime="text/plain", status="processing", stamp=STAMP):
    decision = await db.reserve_single_upload(
        reservation_id=file_id, user=USER, bytes_=12, max_user_bytes=10_000,
        now=stamp, expired_before=OLD, private_required=private,
    )
    assert decision.accepted
    decision = await db.commit_reserved_upload(
        reservation_id=file_id, reservation_kind="single_shot", user=USER,
        file_id=file_id, filename=filename, mime=mime, bytes_=12, kind=kind,
        collection="", chunks=0, uploaded_at=stamp, status=status,
        max_user_bytes=10_000,
    )
    assert decision.accepted


def _assert_refused(response):
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "private_processing_unavailable"
    assert "private-notes" not in response.text


@pytest.mark.parametrize("kind", ["single_shot", "chunked"])
async def test_ingestion_reads_durable_flag_before_extraction_or_embedding(
    db, tmp_path, monkeypatch, kind,
):
    storage = StorageLifecycle(db)
    if kind == "single_shot":
        reserved = await storage.reserve_single_upload(
            reservation_id="reserved", user=USER, bytes_=12, max_user_bytes=100,
            now=STAMP, expired_before=OLD, private_required=True,
        )
    else:
        reserved = await storage.open_chunk_session(
            upload_id="reserved", user=USER, filename="private-notes.txt",
            total_bytes=12, part_size=12, parts_total=1, max_user_bytes=100,
            now=STAMP, expired_before=OLD, private_required=True,
        )
    assert reserved.private_required is True
    # A stale caller object does not authorize ordinary processing.
    forged = StorageReservation("reserved", USER, kind, 12, private_required=False)
    sniff = Mock(side_effect=AssertionError("must not extract protected bytes"))
    monkeypatch.setattr(files, "sniff_mime", sniff)
    with pytest.raises(HTTPException) as error:
        await files._validate_and_ingest(
            SimpleNamespace(), tmp_path / "not-read.txt", user=USER,
            file_id="file", filename="private-notes.txt", written=12,
            max_total=100, qdrant=AsyncMock(), text_embedder=AsyncMock(),
            image_embedder=AsyncMock(), storage=storage, reservation=forged,
            text_col="text", image_col="image",
        )
    assert error.value.status_code == 409
    sniff.assert_not_called()
    assert await storage.reservation_private_required(forged) is True


@pytest.mark.parametrize("route", ["part", "complete"])
async def test_private_session_refuses_before_part_write_or_assembly(db, tmp_path, route):
    await db.reserve_upload_session(
        upload_id="protected-session", user=USER, filename="private-notes.txt",
        total_bytes=12, part_size=12, parts_total=1, max_user_bytes=100,
        now=STAMP, expired_before=OLD, private_required=True,
    )
    app = _app(db, tmp_path)
    with TestClient(app) as client:
        if route == "part":
            response = client.put(
                "/v1/files/upload-sessions/protected-session/parts/0", content=b"private data",
            )
        else:
            response = client.post("/v1/files/upload-sessions/protected-session/complete")
    _assert_refused(response)
    assert await db.session_progress("protected-session") == (0, 0)
    app.state.qdrant.ensure_collection.assert_not_called()
    assert not (tmp_path / "uploads").exists()


@pytest.mark.parametrize("route,body", [
    ("ocr-result", {"lease_id": "L", "text": "Synthetic private content", "pages": 1, "language": "eng"}),
    ("ingest-result", {"lease_id": "L", "segments": [{"t_start": 0, "t_end": 1, "text": "Synthetic private content"}]}),
    ("ingest-failed", {"lease_id": "L", "reason": "Synthetic private content"}),
    ("requeue?force=true", {}),
    ("fetch/protected/result", {"lease_id": "L", "filename": "private-notes.mp4"}),
    ("fetch/protected/failed", {"lease_id": "L", "reason": "Synthetic private content"}),
    ("fetch/protected/progress", {"lease_id": "L", "title": "Synthetic private content", "downloaded_bytes": 1}),
])
async def test_ordinary_worker_callbacks_refuse_before_any_content_work(
    db, tmp_path, monkeypatch, route, body,
):
    await _seed(db)
    app = _app(db, tmp_path)
    summary = AsyncMock(side_effect=AssertionError("must not summarize private content"))
    ingest = AsyncMock(side_effect=AssertionError("must not embed private content"))
    monkeypatch.setattr(files, "summarise_document", summary)
    monkeypatch.setattr(files, "summarise_video", summary)
    monkeypatch.setattr(files, "ingest_user_text_content", ingest)
    monkeypatch.setattr(files, "ingest_transcript_segments", ingest)
    before = await db.get_upload("protected")
    path = f"/v1/files/{route}" if route.startswith("fetch/") else f"/v1/files/protected/{route}"
    with TestClient(app) as client:
        response = client.post(path, json=body)
    _assert_refused(response)
    assert "Synthetic private content" not in response.text
    assert await db.get_upload("protected") == before
    assert app.state.qdrant.mock_calls == []
    summary.assert_not_called()
    ingest.assert_not_called()
    assert not (tmp_path / "uploads").exists()


async def test_service_listing_hides_private_metadata_before_stat_calls(db, tmp_path, monkeypatch):
    await _seed(db)
    await _seed(db, private=False, file_id="ordinary", filename="ordinary.txt")
    available = Mock(return_value={})
    monkeypatch.setattr(files, "_available_artifacts", available)
    app = _app(db, tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/files/list", json={"user": USER})
    assert response.status_code == 200
    assert [row["filename"] for row in response.json()["files"]] == ["ordinary.txt"]
    assert "private-notes" not in response.text
    seen_rows = available.call_args.args[1]
    assert [row["file_id"] for row in seen_rows] == ["ordinary"]


async def test_artifact_refuses_protected_newest_duplicate_and_hides_alternatives(
    db, tmp_path, monkeypatch,
):
    await _seed(db, private=False, file_id="ordinary", filename="private-notes.txt", stamp=OLD)
    await _seed(db, stamp=STAMP)
    extract = Mock(side_effect=AssertionError("must not substitute an older file"))
    monkeypatch.setattr(files, "extract_uploaded_text", extract)
    app = _app(db, tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/files/artifact", json={"user": USER, "filename": "private-notes.txt"})
        _assert_refused(response)
        response = client.post("/v1/files/artifact", json={"user": USER, "filename": "missing.txt"})
        # The same name legitimately belongs to an ordinary upload too.
        assert response.status_code == 404
    extract.assert_not_called()


async def test_missing_artifact_notice_does_not_name_protected_only_files(db, tmp_path):
    await _seed(db)
    app = _app(db, tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/files/artifact", json={"user": USER, "filename": "missing.txt"})
    assert response.status_code == 404
    assert "private-notes" not in response.text


async def test_independent_document_summary_refuses_before_fail_soft_handler(db, tmp_path, monkeypatch):
    await _seed(db)
    summary = AsyncMock(side_effect=AssertionError("must not invoke summarizer"))
    monkeypatch.setattr(files, "summarise_document", summary)
    app = _app(db, tmp_path)
    with pytest.raises(HTTPException) as error:
        await files._store_document_summary(
            SimpleNamespace(app=app), "Synthetic private content", user=USER,
            file_id="protected", filename="private-notes.txt", mime="application/pdf",
            source_bytes=12, collection="text", stamp=STAMP,
            qdrant=app.state.qdrant, text_embedder=app.state.text_embedder,
        )
    assert error.value.status_code == 409
    summary.assert_not_called()


@pytest.mark.parametrize("operation", ["single", "session", "url"])
async def test_ordinary_upload_entry_points_cannot_shed_an_active_policy(operation):
    policy = ProcessingPolicy(local_models={"local:latest": "a" * 64}, ollama_base_url="http://ollama:11434")
    request = SimpleNamespace()  # Any attempt to access app state before refusal fails.
    with processing_scope(policy), pytest.raises(HTTPException) as error:
        if operation == "single":
            await files.upload_file(request, me=SimpleNamespace(email=USER), file=None)
        elif operation == "session":
            await files.open_upload_session(None, request, me=SimpleNamespace(email=USER))
        else:
            await files.ingest_from_url(None, request, me=SimpleNamespace(email=USER))
    assert error.value.status_code == 409


async def test_missing_reservation_cannot_default_to_ordinary_processing(db, tmp_path, monkeypatch):
    dest = tmp_path / "uncommitted.txt"
    dest.write_text("Synthetic upload", encoding="utf-8")
    sniff = Mock(side_effect=AssertionError("must refuse before extracting"))
    monkeypatch.setattr(files, "sniff_mime", sniff)
    with pytest.raises(HTTPException) as error:
        await files._validate_and_ingest(
            SimpleNamespace(), dest, user=USER, file_id="new-file",
            filename="uncommitted.txt", written=16, max_total=100,
            qdrant=AsyncMock(), text_embedder=AsyncMock(), image_embedder=AsyncMock(),
            storage=StorageLifecycle(db),
            reservation=StorageReservation("expired", USER, "single_shot", 16),
            text_col="text", image_col="image",
        )
    assert error.value.status_code == 409
    assert not dest.exists()
    sniff.assert_not_called()
    assert await db.get_upload("new-file") is None
