"""Durable file deletion across Qdrant, disk, restart, and reconcile."""

from __future__ import annotations

import asyncio
from pathlib import Path

from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.kb.user_store import (
    sanitize_user,
    user_image_collection,
    user_text_collection,
)

USER = "alice@example.com"
FILE_ID = "file-1"


def _point(file_id: str = FILE_ID) -> dict:
    return {
        "user": USER,
        "file_id": file_id,
        "filename": "notes.txt",
        "mime": "text/plain",
        "bytes": 12,
        "kind": "text",
        "uploaded_at": "2026-08-28T00:00:00+00:00",
    }


class _FakeQdrant:
    def __init__(self) -> None:
        self.points: dict[str, list[dict]] = {
            user_text_collection(USER): [_point()],
        }
        self.delete_failures = 0
        self.delete_calls: list[tuple[str, str, str]] = []

    async def delete_by_file_id(
        self,
        file_id: str,
        *,
        user: str,
        collection: str,
    ) -> None:
        self.delete_calls.append((file_id, user, collection))
        if self.delete_failures:
            self.delete_failures -= 1
            raise RuntimeError("injected Qdrant delete failure")
        self.points[collection] = [
            point
            for point in self.points.get(collection, [])
            if not (point.get("file_id") == file_id and point.get("user") == user)
        ]

    async def list_collections(self) -> list[str]:
        return list(self.points)

    async def collection_exists(self, collection: str) -> bool:
        return collection in self.points

    async def scroll_collection(self, collection: str) -> list[tuple[str, dict]]:
        return [
            (f"point-{index}", point) for index, point in enumerate(self.points.get(collection, []))
        ]

    async def list_user_files(self, *, user: str, collection: str) -> list[dict]:
        seen: set[str] = set()
        rows: list[dict] = []
        for point in self.points.get(collection, []):
            if point.get("user") != user or point["file_id"] in seen:
                continue
            seen.add(str(point["file_id"]))
            rows.append({"file_id": point["file_id"]})
        return rows


async def _record(db: UploadsDB, file_id: str = FILE_ID) -> None:
    await db.record_upload(
        file_id=file_id,
        user=USER,
        filename="notes.txt",
        mime="text/plain",
        bytes_=12,
        kind="text",
        collection=user_text_collection(USER),
        chunks=1,
        uploaded_at="2026-08-28T00:00:00+00:00",
    )


def _worker(db: UploadsDB, qdrant: _FakeQdrant, root: Path, locks=None):
    return FileDeletionWorker(
        db=db,
        qdrant=qdrant,
        upload_root=root,
        locks=locks or FileOperationLocks(),
        retry_interval_s=60,
        batch_size=50,
    )


async def test_qdrant_failure_stays_hidden_and_converges_after_restart(tmp_path: Path):
    db_path = tmp_path / "uploads.sqlite"
    upload_root = tmp_path / "uploads"
    user_dir = upload_root / sanitize_user(USER)
    stage_dir = upload_root / ".staging"
    user_dir.mkdir(parents=True)
    stage_dir.mkdir()
    (user_dir / f"{FILE_ID}.txt").write_text("source", encoding="utf-8")
    (user_dir / f"{FILE_ID}.summary.txt").write_text("summary", encoding="utf-8")
    (stage_dir / f"{FILE_ID}.part").write_text("partial", encoding="utf-8")

    db = UploadsDB(db_path)
    await _record(db)
    qdrant = _FakeQdrant()
    qdrant.delete_failures = 2
    first = await _worker(db, qdrant, upload_root).request(FILE_ID, user=USER)

    assert first.known is True
    assert first.completed is False
    assert await db.list_user(USER) == []
    assert await db.file_deletion_ids(USER) == {FILE_ID}
    assert len(await db.pending_file_deletions()) == 1
    assert list(user_dir.iterdir()) == []
    assert list(stage_dir.iterdir()) == []

    await reconcile_with_qdrant(db, qdrant)
    assert await db.list_user(USER) == []
    db.close()

    restarted = UploadsDB(db_path)
    qdrant.delete_failures = 0
    worker = _worker(restarted, qdrant, upload_root)
    assert await worker.drain_once() == 1
    assert await restarted.pending_file_deletions() == []
    assert await restarted.list_user(USER) == []
    assert qdrant.points[user_text_collection(USER)] == []

    qdrant.points[user_text_collection(USER)] = [_point()]
    await reconcile_with_qdrant(restarted, qdrant)
    assert await restarted.list_user(USER) == []
    restarted.close()


async def test_disk_failure_leaves_tombstone_retryable(tmp_path: Path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _record(db)
    qdrant = _FakeQdrant()
    user_dir = tmp_path / "uploads" / sanitize_user(USER)
    blocked = user_dir / f"{FILE_ID}.sidecar"
    blocked.mkdir(parents=True)
    worker = _worker(db, qdrant, tmp_path / "uploads")

    result = await worker.request(FILE_ID, user=USER)
    assert result.completed is False
    assert "IsADirectoryError" in result.error
    assert await db.list_user(USER) == []

    blocked.rmdir()
    assert await worker.drain_once() == 1
    assert await db.pending_file_deletions() == []
    assert await db.list_user(USER) == []
    db.close()


async def test_delete_waits_for_inflight_operation_on_same_file(tmp_path: Path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _record(db)
    qdrant = _FakeQdrant()
    locks = FileOperationLocks()
    worker = _worker(db, qdrant, tmp_path / "uploads", locks)

    async with locks.hold(FILE_ID):
        task = asyncio.create_task(worker.request(FILE_ID, user=USER))
        await asyncio.sleep(0.01)
        assert qdrant.delete_calls == []

    result = await task
    assert result.completed is True
    assert qdrant.delete_calls == [
        (FILE_ID, USER, user_text_collection(USER)),
        (FILE_ID, USER, user_image_collection(USER)),
    ]
    db.close()


async def test_unknown_file_does_not_create_a_tombstone(tmp_path: Path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    worker = _worker(db, _FakeQdrant(), tmp_path / "uploads")

    result = await worker.request("missing", user=USER)

    assert result.known is False
    assert await db.file_deletion_keys() == set()
    db.close()


async def test_full_failed_batch_waits_instead_of_hammering_qdrant(tmp_path: Path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    await _record(db)
    await db.request_file_deletion(
        FILE_ID,
        user=USER,
        requested_at="2026-08-28T00:00:00+00:00",
    )
    qdrant = _FakeQdrant()
    qdrant.delete_failures = 1_000
    worker = FileDeletionWorker(
        db=db,
        qdrant=qdrant,
        upload_root=tmp_path / "uploads",
        locks=FileOperationLocks(),
        retry_interval_s=60,
        batch_size=1,
    )

    await worker.start()
    await asyncio.sleep(0.05)
    await worker.stop()

    assert len(qdrant.delete_calls) == 2
    assert len(await db.pending_file_deletions()) == 1
    db.close()
