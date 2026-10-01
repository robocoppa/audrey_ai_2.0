"""Scanned-PDF upload, indexing, and lease handoff contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey.kb.extract import EmptyExtractionError
from audrey.kb.ingest import ingest_user_text_content
from audrey.kb.storage_lifecycle import StorageReservation
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.routes import files as files_routes
from audrey.routes.files import _validate_and_ingest, router

SECRET = "ocr-test-service-token"  # noqa: S105


class _Storage:
    def __init__(self) -> None:
        self.commits: list[dict] = []

    async def commit_upload(self, reservation, **kwargs) -> None:
        self.commits.append({"reservation": reservation, **kwargs})


def _request(tmp_path: Path, *, enabled: bool = True):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                cfg=SimpleNamespace(
                    raw={
                        "kb": {
                            "upload_root": str(tmp_path / "uploads"),
                            "ocr": {"enabled": enabled},
                        },
                    },
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_image_only_pdf_becomes_a_pending_text_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dest = tmp_path / "scan.pdf"
    dest.write_bytes(b"%PDF image-only fixture")
    storage = _Storage()
    reservation = StorageReservation("r1", "alice@example.com", "single_shot", 23)

    monkeypatch.setattr(files_routes, "sniff_mime", lambda _path: "application/pdf")

    async def no_text(*args, **kwargs):
        raise EmptyExtractionError("no text layer")

    monkeypatch.setattr(files_routes, "ingest_user_text_file", no_text)

    response = await _validate_and_ingest(
        _request(tmp_path),
        dest,
        user="alice@example.com",
        file_id="doc1",
        filename="scan.pdf",
        written=23,
        max_total=1000,
        qdrant=object(),
        text_embedder=object(),
        image_embedder=None,
        storage=storage,
        reservation=reservation,
        text_col="kb_user_text",
        image_col="kb_user_images",
    )

    assert response.status == "pending"
    assert response.kind == "text"
    assert response.collection == ""
    assert response.chunks == 0
    assert dest.is_file()
    assert storage.commits[0]["status"] == "pending"
    assert storage.commits[0]["kind"] == "text"


@pytest.mark.asyncio
async def test_text_layer_pdf_commits_its_summary_with_the_ready_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dest = tmp_path / "report.pdf"
    dest.write_bytes(b"%PDF text fixture")
    storage = _Storage()
    reservation = StorageReservation("r1", "alice@example.com", "single_shot", 17)

    monkeypatch.setattr(files_routes, "sniff_mime", lambda _path: "application/pdf")

    async def ingest_text(*args, **kwargs):
        return 2

    async def store_summary(*args, **kwargs):
        assert args[1] == "Extracted report text."
        return "The report identifies the maintenance priorities.", 1

    monkeypatch.setattr(files_routes, "ingest_user_text_file", ingest_text)
    monkeypatch.setattr(
        files_routes,
        "extract_uploaded_text",
        lambda _path: "Extracted report text.",
    )
    monkeypatch.setattr(files_routes, "_store_document_summary", store_summary)

    response = await _validate_and_ingest(
        _request(tmp_path),
        dest,
        user="alice@example.com",
        file_id="doc1",
        filename="report.pdf",
        written=17,
        max_total=1000,
        qdrant=object(),
        text_embedder=object(),
        image_embedder=None,
        storage=storage,
        reservation=reservation,
        text_col="kb_user_text",
        image_col="kb_user_images",
    )

    assert response.status == "ready"
    assert response.chunks == 3
    assert storage.commits[0]["summary"] == (
        "The report identifies the maintenance priorities."
    )


@pytest.mark.asyncio
async def test_ocr_kill_switch_restores_the_immediate_empty_pdf_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dest = tmp_path / "scan.pdf"
    dest.write_bytes(b"%PDF image-only fixture")
    storage = _Storage()
    reservation = StorageReservation("r1", "alice@example.com", "single_shot", 23)

    monkeypatch.setattr(files_routes, "sniff_mime", lambda _path: "application/pdf")

    async def no_text(*args, **kwargs):
        raise EmptyExtractionError("no text layer")

    monkeypatch.setattr(files_routes, "ingest_user_text_file", no_text)

    with pytest.raises(HTTPException) as caught:
        await _validate_and_ingest(
            _request(tmp_path, enabled=False),
            dest,
            user="alice@example.com",
            file_id="doc1",
            filename="scan.pdf",
            written=23,
            max_total=1000,
            qdrant=object(),
            text_embedder=object(),
            image_embedder=None,
            storage=storage,
            reservation=reservation,
            text_col="kb_user_text",
            image_col="kb_user_images",
        )

    assert caught.value.status_code == 422
    assert not dest.exists()
    assert storage.commits == []


class _Embedder:
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _text in texts]


class _Qdrant:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str, str]] = []
        self.points = []

    async def delete_by_file_id(self, file_id: str, *, user: str, collection: str):
        self.deleted.append((file_id, user, collection))

    async def has_sparse(self, collection: str) -> bool:
        return False

    async def upsert_text(self, points, *, collection: str):
        self.points.extend(points)


@pytest.mark.asyncio
async def test_ocr_index_payload_keeps_owner_and_original_pdf_accounting(tmp_path: Path):
    sidecar = tmp_path / "doc1.ocr.txt"
    sidecar.write_text("Recognized invoice total forty two.", encoding="utf-8")
    qdrant = _Qdrant()

    count = await ingest_user_text_content(
        sidecar.read_text(encoding="utf-8"),
        source=sidecar,
        source_bytes=9876,
        qdrant=qdrant,
        embedder=_Embedder(),
        collection="kb_user_text_alice",
        user="alice@example.com",
        file_id="doc1",
        filename="scan.pdf",
        mime="application/pdf",
        uploaded_at="2026-09-30T12:00:00+00:00",
    )

    assert count == 1
    assert qdrant.deleted == [("doc1", "alice@example.com", "kb_user_text_alice")]
    payload = qdrant.points[0].payload
    assert payload["user"] == "alice@example.com"
    assert payload["file_id"] == "doc1"
    assert payload["filename"] == "scan.pdf"
    assert payload["mime"] == "application/pdf"
    assert payload["bytes"] == 9876
    assert "artifact" not in payload
    assert payload["source"].endswith("doc1.ocr.txt")


async def _add_pdf(db: UploadsDB, file_id: str = "doc1") -> None:
    await db.record_upload(
        file_id=file_id,
        user="alice@example.com",
        filename="scan.pdf",
        mime="application/pdf",
        bytes_=9876,
        kind="text",
        collection="",
        chunks=0,
        uploaded_at="2026-09-30T12:00:00+00:00",
        status="pending",
    )


def _app(db: UploadsDB, tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.uploads_db = db
    app.state.qdrant = object()
    app.state.text_embedder = object()
    app.state.image_embedder = object()
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={
            "kb": {
                "upload_root": str(tmp_path / "uploads"),
                "video": {"lease_minutes": 30, "max_attempts": 3},
                "ocr": {
                    "enabled": True,
                    "language": "eng",
                    "dpi": 240,
                    "max_pages": 12,
                    "max_chars": 3456,
                    "timeout_s": 78,
                },
            },
        },
    )
    return app


def _headers() -> dict[str, str]:
    return {"X-Audrey-Service-Token": SECRET}


@pytest.mark.asyncio
async def test_claim_and_result_complete_a_scanned_pdf_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _add_pdf(db)
    owner_dir = tmp_path / "uploads" / sanitize_user("alice@example.com")
    owner_dir.mkdir(parents=True)
    source = owner_dir / "doc1.pdf"
    source.write_bytes(b"%PDF image-only fixture")
    captured: dict = {}

    async def collections(_qdrant, user):
        assert user == "alice@example.com"
        return "kb_user_text_alice", "kb_user_images_alice"

    async def ingest(text, **kwargs):
        captured.update(text=text, **kwargs)
        assert kwargs["source"].read_text(encoding="utf-8") == text
        return 2

    async def store_summary(*args, **kwargs):
        assert args[1].startswith("--- Page 1 ---")
        return "The invoice records a total of forty-two dollars.", 1

    monkeypatch.setattr(files_routes, "ensure_user_collections", collections)
    monkeypatch.setattr(files_routes, "ingest_user_text_content", ingest)
    monkeypatch.setattr(files_routes, "_store_document_summary", store_summary)
    client = TestClient(_app(db, tmp_path))

    claim = client.post("/v1/files/jobs/claim", headers=_headers())
    assert claim.status_code == 200
    job = claim.json()
    assert job["kind"] == "text"
    assert job["mime"] == "application/pdf"
    assert job["path"].endswith("doc1.pdf")
    assert job["transcript"] is None
    assert job["ocr"] == {
        "language": "eng",
        "dpi": 240,
        "max_pages": 12,
        "max_chars": 3456,
        "timeout_s": 78.0,
    }

    result = client.post(
        "/v1/files/doc1/ocr-result",
        headers=_headers(),
        json={
            "lease_id": job["lease_id"],
            "text": "--- Page 1 ---\nRecognized invoice total: $42",
            "pages": 1,
            "language": "eng",
        },
    )

    assert result.status_code == 200
    assert result.json() == {"file_id": "doc1", "status": "ready", "chunks": 3}
    assert captured["source_bytes"] == 9876
    assert captured["user"] == "alice@example.com"
    assert captured["file_id"] == "doc1"
    assert captured["filename"] == "scan.pdf"
    assert captured["mime"] == "application/pdf"
    assert source.with_suffix(".ocr.txt").is_file()
    row = await db.get_upload("doc1")
    assert (row["status"], row["collection"], row["chunks"]) == (
        "ready",
        "kb_user_text_alice",
        3,
    )
    assert row["bytes"] == 9876
    assert row["summary"] == "The invoice records a total of forty-two dollars."
    db.close()


@pytest.mark.asyncio
async def test_ocr_result_requires_service_auth_and_the_current_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _add_pdf(db)
    app = _app(db, tmp_path)
    client = TestClient(app)
    body = {
        "lease_id": "not-the-current-lease",
        "text": "--- Page 1 ---\nPrivate text",
        "pages": 1,
        "language": "eng",
    }

    assert client.post("/v1/files/doc1/ocr-result", json=body).status_code == 401
    job = client.post("/v1/files/jobs/claim", headers=_headers()).json()

    async def must_not_run(*args, **kwargs):
        pytest.fail("a stale lease must be rejected before indexing")

    monkeypatch.setattr(files_routes, "ensure_user_collections", must_not_run)
    stale = client.post("/v1/files/doc1/ocr-result", headers=_headers(), json=body)

    assert stale.status_code == 409
    assert job["lease_id"] != body["lease_id"]
    sidecar = (
        tmp_path
        / "uploads"
        / sanitize_user("alice@example.com")
        / "doc1.ocr.txt"
    )
    assert not sidecar.exists()
    assert (await db.get_upload("doc1"))["status"] == "processing"
    db.close()


@pytest.mark.asyncio
async def test_whitespace_ocr_result_cannot_complete_a_ready_row(tmp_path: Path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _add_pdf(db)
    client = TestClient(_app(db, tmp_path))
    job = client.post("/v1/files/jobs/claim", headers=_headers()).json()

    result = client.post(
        "/v1/files/doc1/ocr-result",
        headers=_headers(),
        json={
            "lease_id": job["lease_id"],
            "text": "   \n\t",
            "pages": 1,
            "language": "eng",
        },
    )

    assert result.status_code == 422
    assert result.json() == {"detail": "OCR result contains no text."}
    row = await db.get_upload("doc1")
    assert row["status"] == "processing"
    db.close()
