"""Durable account-wide purge coordination across Audrey-owned stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.user_data_purge import UserDataPurgeCoordinator
from audrey.user_data_visibility import (
    remote_personal_reads_blocked,
    unblock_remote_personal_reads,
)

ALICE = "alice@example.com"
BOB = "bob@example.com"
OLD = "2020-01-01T00:00:00.000000+00:00"
NEW = "2030-01-01T00:00:00.000000+00:00"


class _FileWorker:
    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


class _ArchiveQueue:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str, str]] = []

    async def purge_user_before(
        self,
        *,
        user_id: str,
        cutoff_at: str,
        purge_id: str,
    ) -> int:
        self.calls.append((user_id, cutoff_at, purge_id))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected local archive failure")
        return 0


class _Transport:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []

    async def request_user_purge(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else {"status": "completed"}
        if isinstance(outcome, Exception):
            raise outcome
        outcome = dict(outcome)
        outcome.setdefault("purge_id", kwargs["purge_id"])
        outcome.setdefault("cutoff_at", kwargs["cutoff_at"])
        return outcome


def _coordinator(
    db: UploadsDB,
    root: Path,
    *,
    queue: _ArchiveQueue,
    transport: _Transport,
    files: _FileWorker | None = None,
) -> UserDataPurgeCoordinator:
    return UserDataPurgeCoordinator(
        db=db,
        file_deletions=files or _FileWorker(),  # type: ignore[arg-type]
        archive_queue=queue,  # type: ignore[arg-type]
        archive_transport=transport,  # type: ignore[arg-type]
        registry=object(),  # type: ignore[arg-type]
        upload_root=root,
        retry_interval_s=60,
    )


async def _record(db: UploadsDB, *, file_id: str, user: str, uploaded_at: str) -> None:
    await db.record_upload(
        file_id=file_id,
        user=user,
        filename=f"{file_id}.txt",
        mime="text/plain",
        bytes_=10,
        kind="text",
        collection="collection",
        chunks=1,
        uploaded_at=uploaded_at,
    )


async def test_purge_tombstones_snapshot_cleans_paths_and_preserves_new_activity(
    tmp_path: Path,
):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    root = tmp_path / "uploads"
    files = _FileWorker()
    queue = _ArchiveQueue()
    transport = _Transport([{"status": "pending"}, {"status": "completed"}])
    coordinator = _coordinator(
        db,
        root,
        queue=queue,
        transport=transport,
        files=files,
    )
    try:
        await _record(db, file_id="alice-old", user=ALICE, uploaded_at=OLD)
        await _record(db, file_id="alice-new", user=ALICE, uploaded_at=NEW)
        await _record(db, file_id="bob-old", user=BOB, uploaded_at=OLD)
        with db._lock, db._transaction_locked():
            db._conn.execute(
                "INSERT INTO upload_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("session-old", ALICE, "large.bin", 20, 10, 2, OLD, OLD),
            )
            db._conn.execute(
                "INSERT INTO upload_parts VALUES (?, ?, ?)",
                ("session-old", 0, 10),
            )
            db._conn.execute(
                "INSERT INTO storage_reservations VALUES (?, ?, ?, ?, ?, ?)",
                ("single-old", ALICE, "single_shot", 10, OLD, OLD),
            )

        user_dir = root / sanitize_user(ALICE)
        session_dir = user_dir / ".sessions" / "session-old"
        session_dir.mkdir(parents=True)
        (session_dir / "0.part").write_bytes(b"partial")
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "single-old.txt").write_text("partial", encoding="utf-8")

        receipt = await coordinator.request(user=ALICE, purge_id="purge-1")

        assert receipt["status"] == "pending"
        assert receipt["files"]["pending"] == 1
        assert receipt["paths"] == {
            "pending": 0,
            "attempts": 2,
            "with_error": 0,
            "completed": 2,
        }
        assert receipt["local_delivery"]["completed"] is True
        assert receipt["sidecar"]["completed"] is False
        assert receipt["sidecar"]["acknowledged"] is True
        assert remote_personal_reads_blocked(ALICE) is False
        assert [row["file_id"] for row in await db.list_user(ALICE)] == ["alice-new"]
        assert [row["file_id"] for row in await db.list_user(BOB)] == ["bob-old"]
        assert not session_dir.exists()
        assert not (user_dir / "single-old.txt").exists()
        assert files.wakes == 1

        repeated = await coordinator.request(user=ALICE, purge_id="purge-1")
        assert repeated["purge_id"] == "purge-1"
        assert repeated["sidecar"]["completed"] is True
        assert len(queue.calls) == 1
        assert await coordinator.status(user=BOB, purge_id="purge-1") is None

        await db.complete_file_deletion(
            "alice-old",
            user=ALICE,
            completed_at=NEW,
        )
        await coordinator.drain_once()
        completed = await coordinator.status(user=ALICE, purge_id="purge-1")
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["files"]["pending"] == 0
    finally:
        db.close()


async def test_purge_receipt_retries_local_and_remote_failures_after_restart(
    tmp_path: Path,
):
    db_path = tmp_path / "uploads.sqlite"
    root = tmp_path / "uploads"
    first_db = UploadsDB(db_path)
    first = _coordinator(
        first_db,
        root,
        queue=_ArchiveQueue(failures=1),
        transport=_Transport([RuntimeError("sidecar down")]),
    )
    try:
        receipt = await first.request(user=ALICE, purge_id="purge-restart")
        assert receipt["status"] == "pending"
        assert receipt["local_delivery"]["with_error"] is True
        assert receipt["sidecar"]["with_error"] is True
        assert receipt["sidecar"]["acknowledged"] is False
        assert remote_personal_reads_blocked(ALICE) is True
    finally:
        first_db.close()

    restarted_db = UploadsDB(db_path)
    restarted = _coordinator(
        restarted_db,
        root,
        queue=_ArchiveQueue(),
        transport=_Transport([{"status": "completed"}]),
    )
    try:
        await restarted.drain_once()
        completed = await restarted.status(
            user=ALICE,
            purge_id="purge-restart",
        )
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["local_delivery"] == {
            "completed": True,
            "attempts": 2,
            "with_error": False,
        }
        assert completed["sidecar"] == {
            "acknowledged": True,
            "completed": True,
            "status": "completed",
            "attempts": 2,
            "with_error": False,
        }
        assert remote_personal_reads_blocked(ALICE) is False
    finally:
        restarted_db.close()

async def test_coordinator_start_restores_unacknowledged_privacy_gate(
    tmp_path: Path,
):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    coordinator = _coordinator(
        db,
        tmp_path / "uploads",
        queue=_ArchiveQueue(),
        transport=_Transport([RuntimeError("sidecar remains down")]),
    )
    purge_id = "purge-startup-gate"
    try:
        await db.request_user_data_purge(
            purge_id=purge_id,
            user=ALICE,
            cutoff_at=OLD,
            requested_at=OLD,
        )

        await coordinator.start()

        assert remote_personal_reads_blocked(ALICE) is True
    finally:
        await coordinator.stop()
        unblock_remote_personal_reads(user=ALICE, purge_id=purge_id)
        db.close()

async def test_uploads_db_migrates_existing_purge_acknowledgement(
    tmp_path: Path,
):
    db_path = tmp_path / "uploads.sqlite"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE user_data_purges (
          purge_id TEXT PRIMARY KEY,
          user TEXT NOT NULL,
          cutoff_at TEXT NOT NULL,
          requested_at TEXT NOT NULL,
          local_delivery_completed_at TEXT NOT NULL DEFAULT "",
          local_delivery_attempts INTEGER NOT NULL DEFAULT 0,
          local_delivery_last_error TEXT NOT NULL DEFAULT "",
          sidecar_completed_at TEXT NOT NULL DEFAULT "",
          sidecar_status TEXT NOT NULL DEFAULT "pending",
          sidecar_attempts INTEGER NOT NULL DEFAULT 0,
          sidecar_last_error TEXT NOT NULL DEFAULT "",
          completed_at TEXT NOT NULL DEFAULT ""
        );
        INSERT INTO user_data_purges
          (purge_id, user, cutoff_at, requested_at, sidecar_completed_at)
        VALUES
          ("legacy-purge", "alice@example.com", "2020-01-01T00:00:00+00:00",
           "2020-01-01T00:00:00+00:00", "2020-01-01T00:01:00+00:00");
        """
    )
    legacy.close()

    db = UploadsDB(db_path)
    try:
        row = await db.get_user_data_purge("legacy-purge")
        assert row is not None
        assert row["sidecar_acknowledged_at"] == "2020-01-01T00:01:00+00:00"
        assert await db.unacknowledged_user_data_purges() == []
    finally:
        db.close()


async def test_admin_repair_stats_aggregate_without_returning_user_rows(
    tmp_path: Path,
):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    try:
        await _record(db, file_id="alice-file", user=ALICE, uploaded_at=OLD)
        await _record(db, file_id="bob-file", user=BOB, uploaded_at=OLD)
        await db.request_file_deletion(
            "alice-file",
            user=ALICE,
            requested_at=OLD,
        )
        await db.request_file_deletion(
            "bob-file",
            user=BOB,
            requested_at=OLD,
        )
        await db.begin_file_deletion_attempt(
            "alice-file",
            user=ALICE,
            attempted_at=OLD,
        )
        await db.fail_file_deletion(
            "alice-file",
            user=ALICE,
            error="private qdrant detail",
        )
        await db.complete_file_deletion(
            "bob-file",
            user=BOB,
            completed_at=NEW,
        )

        assert await db.file_deletion_stats() == {
            "pending": 1,
            "attempts": 1,
            "with_error": 1,
            "exhausted": 0,
            "completed": 1,
        }

        for purge_id, user in (("purge-a", ALICE), ("purge-b", BOB)):
            await db.request_user_data_purge(
                purge_id=purge_id,
                user=user,
                cutoff_at="1900-01-01T00:00:00+00:00",
                requested_at=OLD,
            )
            await db.begin_user_data_purge_component(
                purge_id,
                component="local_delivery",
                attempted_at=OLD,
            )
            await db.finish_user_data_purge_component(
                purge_id,
                component="local_delivery",
                completed_at=NEW,
            )
            await db.begin_user_data_purge_component(
                purge_id,
                component="sidecar",
                attempted_at=OLD,
            )

        await db.fail_user_data_purge_component(
            "purge-a",
            component="sidecar",
            error="private sidecar detail",
        )
        await db.finish_user_data_purge_component(
            "purge-b",
            component="sidecar",
            completed_at=NEW,
        )
        assert await db.finalize_user_data_purge("purge-b", completed_at=NEW)

        stats = await db.data_purge_stats()
        assert stats == {
            "pending": 1,
            "attempts": 4,
            "with_error": 1,
            "exhausted": 0,
            "completed": 1,
        }
        assert ALICE not in str(stats)
        assert BOB not in str(stats)
        assert "private" not in str(stats)
    finally:
        db.close()
