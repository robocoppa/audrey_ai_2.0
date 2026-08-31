"""Durable, idempotent cleanup for per-user file deletion."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import (
    sanitize_user,
    user_image_collection,
    user_text_collection,
)
from audrey.metrics import file_deletion_events_total, file_deletion_pending

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class FileOperationLocks:
    """Per-file locks shared by ingest handovers and deletion cleanup."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @contextlib.asynccontextmanager
    async def hold(self, file_id: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._entries.get(file_id)
            if entry is None:
                entry = _LockEntry(asyncio.Lock())
                self._entries[file_id] = entry
            entry.users += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(file_id, None)


@dataclass(frozen=True, slots=True)
class FileDeletionResult:
    known: bool
    completed: bool
    error: str = ""


def delete_disk_files(upload_root: Path, file_id: str, user: str) -> list[str]:
    """Remove source, sidecars, and fetch staging files for one file id."""
    errors: list[str] = []
    directories = (
        upload_root / sanitize_user(user),
        upload_root / ".staging",
    )
    for directory in directories:
        try:
            candidates = [
                path
                for path in directory.iterdir()
                if path.name == file_id or path.name.startswith(f"{file_id}.")
            ]
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"list {directory}: {type(exc).__name__}: {exc}")
            continue
        for path in candidates:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"unlink {path}: {type(exc).__name__}: {exc}")
    return errors


class FileDeletionWorker:
    """Own retryable Qdrant/disk cleanup for durable SQLite tombstones."""

    def __init__(
        self,
        *,
        db: UploadsDB,
        qdrant,
        upload_root: Path,
        locks: FileOperationLocks,
        retry_interval_s: float = 30.0,
        batch_size: int = 50,
    ) -> None:
        if retry_interval_s <= 0:
            raise ValueError("file deletion retry_interval_s must be positive")
        if batch_size < 1:
            raise ValueError("file deletion batch_size must be positive")
        self._db = db
        self._qdrant = qdrant
        self._upload_root = upload_root
        self._locks = locks
        self._retry_interval_s = retry_interval_s
        self._batch_size = batch_size
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        pending = await self._db.pending_file_deletion_count()
        file_deletion_pending.set(pending)
        log.info(
            "file_deletion: worker ready interval=%.1fs batch=%d pending=%d",
            self._retry_interval_s,
            self._batch_size,
            pending,
        )
        self._task = asyncio.create_task(
            self._run(),
            name="audrey.file_deletion",
        )
        self._wake.set()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def request(self, file_id: str, *, user: str) -> FileDeletionResult:
        known = await self._db.request_file_deletion(
            file_id,
            user=user,
            requested_at=_now_iso(),
        )
        if not known:
            file_deletion_events_total.labels(result="not_found").inc()
            return FileDeletionResult(known=False, completed=False)
        file_deletion_events_total.labels(result="requested").inc()
        result = await self._attempt(file_id, user=user)
        if not result.completed:
            self._wake.set()
        file_deletion_pending.set(await self._db.pending_file_deletion_count())
        return result

    def wake(self) -> None:
        """Wake the durable worker after another owner queues bulk deletions."""
        self._wake.set()

    async def drain_once(self) -> int:
        rows = await self._db.pending_file_deletions(limit=self._batch_size)
        for row in rows:
            try:
                await self._attempt(str(row["file_id"]), user=str(row["user"]))
            except Exception as exc:  # Keep the lifecycle owner alive.
                log.exception(
                    "file_deletion: unexpected attempt failure file_id=%s: %s",
                    row["file_id"],
                    exc,
                )
        file_deletion_pending.set(await self._db.pending_file_deletion_count())
        return len(rows)

    async def _run(self) -> None:
        while True:
            try:
                self._wake.clear()
                pending_before = await self._db.pending_file_deletion_count()
                attempted = await self.drain_once()
                pending_after = await self._db.pending_file_deletion_count()
                if attempted >= self._batch_size and pending_after < pending_before:
                    continue
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._retry_interval_s,
                    )
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Keep the lifecycle owner alive.
                log.exception("file_deletion: worker iteration failed: %s", exc)
                await asyncio.sleep(self._retry_interval_s)

    async def _attempt(self, file_id: str, *, user: str) -> FileDeletionResult:
        async with self._locks.hold(file_id):
            attempted_at = _now_iso()
            if not await self._db.begin_file_deletion_attempt(
                file_id,
                user=user,
                attempted_at=attempted_at,
            ):
                return FileDeletionResult(known=True, completed=True)

            collections = (
                user_text_collection(user),
                user_image_collection(user),
            )
            outcomes = await asyncio.gather(
                *(
                    self._qdrant.delete_by_file_id(
                        file_id,
                        user=user,
                        collection=collection,
                    )
                    for collection in collections
                ),
                return_exceptions=True,
            )
            errors = [
                f"qdrant {collection}: {type(outcome).__name__}: {outcome}"
                for collection, outcome in zip(collections, outcomes, strict=True)
                if isinstance(outcome, BaseException)
            ]
            disk_errors = await asyncio.to_thread(
                delete_disk_files,
                self._upload_root,
                file_id,
                user,
            )
            errors.extend(disk_errors)
            if not disk_errors:
                await self._db.mark_source_freed(file_id, freed_at=attempted_at)

            if errors:
                error = "; ".join(errors)[:500]
                await self._db.fail_file_deletion(file_id, user=user, error=error)
                file_deletion_events_total.labels(result="retry").inc()
                log.warning(
                    "file_deletion: deferred file_id=%s user=%s error=%s",
                    file_id,
                    user,
                    error,
                )
                return FileDeletionResult(known=True, completed=False, error=error)

            completed = await self._db.complete_file_deletion(
                file_id,
                user=user,
                completed_at=_now_iso(),
            )
            if completed:
                file_deletion_events_total.labels(result="completed").inc()
            file_deletion_pending.set(await self._db.pending_file_deletion_count())
            log.info(
                "file_deletion: completed file_id=%s user=%s",
                file_id,
                user,
            )
            return FileDeletionResult(known=True, completed=completed)


__all__ = [
    "delete_disk_files",
    "FileDeletionResult",
    "FileDeletionWorker",
    "FileOperationLocks",
]
