"""Atomic storage quota reservations across every upload transport."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.auth import AuthedUser, require_user
from audrey.kb.storage_lifecycle import (
    QuotaExceededError,
    StorageLifecycle,
    StorageReservation,
)
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.routes.files import router as files_router

NOW = "2026-08-26T12:00:00+00:00"
EXPIRED_BEFORE = "2026-08-26T10:00:00+00:00"
USER = "alice@example.com"


async def _stored(db: UploadsDB, *, file_id: str, bytes_: int) -> None:
    await db.record_upload(
        file_id=file_id,
        user=USER,
        filename=f"{file_id}.txt",
        mime="text/plain",
        bytes_=bytes_,
        kind="text",
        collection="kb_user_text_alice",
        chunks=1,
        uploaded_at=NOW,
    )


@pytest.mark.asyncio
async def test_mixed_concurrent_reservations_admit_exactly_what_fits(
    tmp_path: Path,
) -> None:
    """Separate connections must still share one transactional quota decision."""
    path = tmp_path / "uploads.sqlite"
    seed = UploadsDB(path)
    await _stored(seed, file_id="stored", bytes_=20)
    seed.close()

    dbs = [UploadsDB(path) for _ in range(3)]
    services = [StorageLifecycle(db) for db in dbs]
    start = asyncio.Event()

    async def single() -> StorageReservation:
        await start.wait()
        return await services[0].reserve_single_upload(
            reservation_id="single-1",
            user=USER,
            bytes_=40,
            max_user_bytes=100,
            now=NOW,
            expired_before=EXPIRED_BEFORE,
        )

    async def chunked() -> StorageReservation:
        await start.wait()
        return await services[1].open_chunk_session(
            upload_id="chunk-1",
            user=USER,
            filename="video.mp4",
            total_bytes=40,
            part_size=10,
            parts_total=4,
            max_user_bytes=100,
            now=NOW,
            expired_before=EXPIRED_BEFORE,
        )

    async def fetched() -> StorageReservation:
        await start.wait()
        return await services[2].reserve_url_fetch(
            file_id="fetch-1",
            user=USER,
            source_url="https://example.com/video",
            filename="video",
            ceiling_bytes=40,
            max_user_bytes=100,
            now=NOW,
            expired_before=EXPIRED_BEFORE,
        )

    tasks = [
        asyncio.create_task(single()),
        asyncio.create_task(chunked()),
        asyncio.create_task(fetched()),
    ]
    start.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    accepted = [r for r in results if isinstance(r, StorageReservation)]
    refused = [r for r in results if isinstance(r, QuotaExceededError)]
    assert len(accepted) == 2
    assert len(refused) == 1

    usage = await dbs[0].quota_usage(USER)
    assert usage.stored_bytes == 20
    assert usage.total_bytes == 100
    assert usage.total_bytes <= 100
    for db in dbs:
        db.close()


@pytest.mark.asyncio
async def test_quota_usage_accounts_for_declared_sessions_and_landed_parts(
    tmp_path: Path,
) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _stored(db, file_id="stored", bytes_=10)
    await db.open_session(
        upload_id="legacy-session",
        user=USER,
        filename="video.mp4",
        total_bytes=30,
        part_size=20,
        parts_total=2,
        now=NOW,
    )
    await db.record_part(
        upload_id="legacy-session", part_no=0, bytes_=20, now=NOW,
    )
    await db.record_part(
        upload_id="legacy-session", part_no=1, bytes_=15, now=NOW,
    )

    usage = await db.quota_usage(USER)

    assert usage.stored_bytes == 10
    assert usage.chunk_declared_bytes == 30
    assert usage.chunk_part_bytes == 35
    assert usage.chunk_part_overage_bytes == 5
    assert usage.total_bytes == 45
    db.close()


@pytest.mark.asyncio
async def test_single_reservation_converts_to_committed_bytes_atomically(
    tmp_path: Path,
) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    reservation = await storage.reserve_single_upload(
        reservation_id="single-1",
        user=USER,
        bytes_=30,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert (await db.quota_usage(USER)).single_shot_bytes == 30

    await storage.commit_upload(
        reservation,
        file_id="single-1",
        filename="notes.txt",
        mime="text/plain",
        bytes_=25,
        kind="text",
        collection="kb_user_text_alice",
        chunks=1,
        uploaded_at=NOW,
        status="ready",
        max_user_bytes=100,
    )

    usage = await db.quota_usage(USER)
    assert usage.stored_bytes == 25
    assert usage.single_shot_bytes == 0
    assert usage.total_bytes == 25
    await storage.release(reservation)  # repeated cleanup is a no-op
    assert (await db.quota_usage(USER)).total_bytes == 25
    db.close()


@pytest.mark.asyncio
async def test_chunk_session_commit_and_ttl_release_are_idempotent(
    tmp_path: Path,
) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    reservation = await storage.open_chunk_session(
        upload_id="chunk-1",
        user=USER,
        filename="video.mp4",
        total_bytes=40,
        part_size=20,
        parts_total=2,
        max_user_bytes=100,
        now="2026-08-26T09:00:00+00:00",
        expired_before="2026-08-26T08:00:00+00:00",
    )
    await storage.record_chunk_part(
        reservation, part_no=0, bytes_=20, now="2026-08-26T09:01:00+00:00",
    )
    assert (await db.quota_usage(USER)).total_bytes == 40

    expired = await storage.expire_chunk_sessions(
        older_than="2026-08-26T10:00:00+00:00",
    )
    assert [row["upload_id"] for row in expired] == ["chunk-1"]
    assert await storage.expire_chunk_sessions(
        older_than="2026-08-26T10:00:00+00:00",
    ) == []
    assert (await db.quota_usage(USER)).total_bytes == 0
    await storage.release(reservation)
    db.close()


@pytest.mark.asyncio
async def test_url_fetch_failure_releases_its_ceiling_once(tmp_path: Path) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    await storage.reserve_url_fetch(
        file_id="fetch-1",
        user=USER,
        source_url="https://example.com/video",
        filename="video",
        ceiling_bytes=60,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert (await db.quota_usage(USER)).url_fetch_bytes == 60

    claimed = await db.claim_fetch(lease_id="lease-1", now=NOW)
    assert claimed is not None
    assert await db.fail_job(
        file_id="fetch-1",
        lease_id="lease-1",
        reason="download failed",
        stage="fetching",
    )
    assert (await db.quota_usage(USER)).url_fetch_bytes == 0
    assert not await db.fail_job(
        file_id="fetch-1",
        lease_id="lease-1",
        reason="again",
        stage="fetching",
    )
    assert (await db.quota_usage(USER)).total_bytes == 0
    db.close()


@pytest.mark.asyncio
async def test_url_fetch_success_converts_ceiling_to_actual_bytes(tmp_path: Path) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    await storage.reserve_url_fetch(
        file_id="fetch-1",
        user=USER,
        source_url="https://example.com/video",
        filename="video",
        ceiling_bytes=60,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert await db.claim_fetch(lease_id="fetch-lease", now=NOW) is not None

    assert await db.complete_fetch(
        file_id="fetch-1",
        lease_id="fetch-lease",
        filename="video.mp4",
        mime="video/mp4",
        bytes_=25,
    )

    usage = await db.quota_usage(USER)
    assert usage.stored_bytes == 25
    assert usage.url_fetch_bytes == 0
    assert usage.total_bytes == 25
    db.close()


@pytest.mark.asyncio
async def test_refetch_ceiling_counts_before_row_moves_to_fetch_queue(
    tmp_path: Path,
) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    await storage.reserve_url_fetch(
        file_id="fetch-1",
        user=USER,
        source_url="https://example.com/video",
        filename="video",
        ceiling_bytes=60,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert await db.claim_fetch(lease_id="fetch-lease", now=NOW) is not None
    assert await db.complete_fetch(
        file_id="fetch-1",
        lease_id="fetch-lease",
        filename="video.mp4",
        mime="video/mp4",
        bytes_=25,
    )
    assert await db.claim_job(lease_id="ingest-lease", now=NOW) is not None
    assert await db.complete_job(
        file_id="fetch-1",
        lease_id="ingest-lease",
        collection="kb_user_text_alice",
        chunks=1,
        completed_at=NOW,
    )
    assert await db.mark_source_freed("fetch-1", freed_at=NOW)

    await storage.reserve_url_refetch(
        file_id="fetch-1",
        user=USER,
        ceiling_bytes=25,
        max_user_bytes=100,
        expired_before=EXPIRED_BEFORE,
    )

    row = await db.get_upload("fetch-1")
    usage = await db.quota_usage(USER)
    assert row is not None and row["status"] == "ready"
    assert usage.stored_bytes == 0
    assert usage.url_fetch_bytes == 25
    assert usage.total_bytes == 25
    db.close()


@pytest.mark.asyncio
async def test_final_fetch_lease_expiry_releases_its_ceiling(tmp_path: Path) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    await storage.reserve_url_fetch(
        file_id="fetch-1",
        user=USER,
        source_url="https://example.com/video",
        filename="video",
        ceiling_bytes=60,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert (
        await db.claim_fetch(
            lease_id="fetch-lease",
            now="2026-08-26T09:00:00+00:00",
        )
        is not None
    )

    swept = await db.sweep_expired_leases(
        expired_before="2026-08-26T10:00:00+00:00",
        max_attempts=1,
        stage="fetching",
    )

    assert swept == {"requeued": 0, "failed": 1}
    assert (await db.quota_usage(USER)).total_bytes == 0
    db.close()


@pytest.mark.asyncio
async def test_rejected_chunk_retry_preserves_the_previous_part(tmp_path: Path) -> None:
    db = UploadsDB(tmp_path / "uploads.sqlite")
    storage = StorageLifecycle(db)
    await storage.open_chunk_session(
        upload_id="chunk-1",
        user=USER,
        filename="video.mp4",
        total_bytes=15,
        part_size=10,
        parts_total=2,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )

    app = FastAPI()
    app.include_router(files_router)
    app.dependency_overrides[require_user] = lambda: AuthedUser(
        email=USER,
        role="user",
        owui_id="user-1",
    )
    app.state.uploads_db = db
    app.state.cfg = SimpleNamespace(
        raw={"kb": {"upload_root": str(tmp_path / "uploads")}},
    )
    app.state.storage_lifecycle = storage

    with TestClient(app) as client:
        assert (
            client.put(
                "/v1/files/upload-sessions/chunk-1/parts/0",
                content=b"a" * 5,
            ).status_code
            == 200
        )
        assert (
            client.put(
                "/v1/files/upload-sessions/chunk-1/parts/1",
                content=b"b" * 10,
            ).status_code
            == 200
        )
        rejected = client.put(
            "/v1/files/upload-sessions/chunk-1/parts/0",
            content=b"c" * 10,
        )

    session_dir = tmp_path / "uploads" / sanitize_user(USER) / ".sessions" / "chunk-1"
    assert rejected.status_code == 409
    assert (session_dir / "000000.part").read_bytes() == b"a" * 5
    assert list(session_dir.glob(".*.incoming")) == []
    assert await db.session_progress("chunk-1") == (2, 15)
    db.close()


@pytest.mark.asyncio
async def test_startup_repairs_stranded_and_legacy_url_reservations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "uploads.sqlite"
    db = UploadsDB(path)
    storage = StorageLifecycle(db)
    await storage.reserve_url_fetch(
        file_id="stranded-refetch",
        user=USER,
        source_url="https://example.com/stranded",
        filename="video",
        ceiling_bytes=60,
        max_user_bytes=100,
        now=NOW,
        expired_before=EXPIRED_BEFORE,
    )
    assert await db.claim_fetch(lease_id="fetch-lease", now=NOW) is not None
    assert await db.complete_fetch(
        file_id="stranded-refetch",
        lease_id="fetch-lease",
        filename="video.mp4",
        mime="video/mp4",
        bytes_=25,
    )
    assert await db.claim_job(lease_id="ingest-lease", now=NOW) is not None
    assert await db.complete_job(
        file_id="stranded-refetch",
        lease_id="ingest-lease",
        collection="kb_user_text_alice",
        chunks=1,
        completed_at=NOW,
    )
    assert await db.mark_source_freed("stranded-refetch", freed_at=NOW)
    await storage.reserve_url_refetch(
        file_id="stranded-refetch",
        user=USER,
        ceiling_bytes=25,
        max_user_bytes=100,
        expired_before=EXPIRED_BEFORE,
    )
    await db.record_url_fetch(
        file_id="legacy-pending",
        user=USER,
        source_url="https://example.com/legacy",
        filename="legacy",
        uploaded_at=NOW,
    )
    db.close()

    reopened = UploadsDB(path)
    recovered = await StorageLifecycle(reopened).restore_pending_url_fetches(
        ceiling_bytes=60,
    )

    assert recovered.released_stranded == 1
    assert recovered.restored_pending == 1
    usage = await reopened.quota_usage(USER)
    assert usage.url_fetch_bytes == 60
    assert usage.total_bytes == 60

    repeated = await StorageLifecycle(reopened).restore_pending_url_fetches(
        ceiling_bytes=60,
    )
    assert repeated.released_stranded == 0
    assert repeated.restored_pending == 0
    reopened.close()
