"""Durable coordinator for authenticated account-wide personal-data purge.

The uploads SQLite database owns the authoritative receipt. Every other store
is a retryable component of that receipt: upload files and Qdrant points drain
through ``FileDeletionWorker``; abandoned disk paths have their own outbox;
the local archive queue installs a cutoff; and custom-tools owns the
memory/chat source tombstone. A process restart therefore loses no work.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from audrey.kb.file_deletion import FileDeletionWorker, delete_disk_files
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.pipeline.chat_archive import ChatArchiveClient, ChatArchiveQueue
from audrey.tools.discovery import ToolRegistry
from audrey.user_data_visibility import (
    block_remote_personal_reads,
    unblock_remote_personal_reads,
)

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _safe_object_id(value: str) -> str:
    """Reject a corrupted cleanup token before it can become a path."""
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("invalid purge cleanup token")
    return value


class UserDataPurgeCoordinator:
    """Own one retry loop spanning Audrey-local and sidecar data stores."""

    def __init__(
        self,
        *,
        db: UploadsDB,
        file_deletions: FileDeletionWorker,
        archive_queue: ChatArchiveQueue | None,
        archive_transport: ChatArchiveClient,
        registry: ToolRegistry,
        upload_root: Path,
        retry_interval_s: float = 30.0,
        batch_size: int = 50,
    ) -> None:
        if retry_interval_s <= 0:
            raise ValueError("user data purge retry interval must be positive")
        if batch_size < 1:
            raise ValueError("user data purge batch size must be positive")
        self._db = db
        self._file_deletions = file_deletions
        self._archive_queue = archive_queue
        self._archive_transport = archive_transport
        self._registry = registry
        self._upload_root = upload_root
        self._retry_interval_s = retry_interval_s
        self._batch_size = batch_size
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        for row in await self._db.unacknowledged_user_data_purges():
            block_remote_personal_reads(
                user=str(row["user"]),
                purge_id=str(row["purge_id"]),
            )
        self._task = asyncio.create_task(
            self._run(),
            name="audrey.user_data_purge",
        )
        self._wake.set()
        log.info(
            "user_data_purge: worker ready interval=%.1fs batch=%d",
            self._retry_interval_s,
            self._batch_size,
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def request(self, *, user: str, purge_id: str = "") -> dict[str, Any]:
        """Create an idempotent receipt, install local tombstones, and retry now."""
        if not user:
            raise ValueError("account purge requires a non-empty user")
        purge_id = purge_id or str(uuid.uuid4())
        existing = await self._db.get_user_data_purge(purge_id)
        if existing is not None:
            if str(existing["user"]) != user:
                raise ValueError("purge_id is already bound to another user")
            cutoff_at = str(existing["cutoff_at"])
            requested_at = str(existing["requested_at"])
        else:
            cutoff_at = _now_iso()
            requested_at = cutoff_at
        await self._db.request_user_data_purge(
            purge_id=purge_id,
            user=user,
            cutoff_at=cutoff_at,
            requested_at=requested_at,
        )
        row = await self._db.get_user_data_purge(purge_id)
        if row is None:
            raise RuntimeError("account purge disappeared after creation")
        if not row.get("sidecar_acknowledged_at"):
            block_remote_personal_reads(user=user, purge_id=purge_id)
        await self._attempt(row)
        self._file_deletions.wake()
        self._wake.set()
        result = await self.status(user=user, purge_id=purge_id)
        if result is None:
            raise RuntimeError("account purge disappeared after processing")
        return result

    async def status(self, *, user: str, purge_id: str) -> dict[str, Any] | None:
        return await self._db.user_data_purge_status(
            user=user,
            purge_id=purge_id,
        )

    async def drain_once(self) -> int:
        rows = await self._db.pending_user_data_purges(limit=self._batch_size)
        for row in rows:
            try:
                await self._attempt(row)
            except Exception as exc:
                log.exception(
                    "user_data_purge: unexpected pass failure purge=%s: %s",
                    row.get("purge_id", ""),
                    exc,
                )
        if rows:
            self._file_deletions.wake()
        return len(rows)

    async def _attempt(self, row: dict[str, Any]) -> None:
        purge_id = str(row["purge_id"])
        user = str(row["user"])
        cutoff_at = str(row["cutoff_at"])

        if not row.get("local_delivery_completed_at"):
            await self._db.begin_user_data_purge_component(
                purge_id,
                component="local_delivery",
                attempted_at=_now_iso(),
            )
            try:
                if self._archive_queue is not None:
                    await self._archive_queue.purge_user_before(
                        user_id=user,
                        cutoff_at=cutoff_at,
                        purge_id=purge_id,
                    )
            except Exception as exc:  # noqa: BLE001 — retry local SQLite failures
                await self._db.fail_user_data_purge_component(
                    purge_id,
                    component="local_delivery",
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                await self._db.finish_user_data_purge_component(
                    purge_id,
                    component="local_delivery",
                    completed_at=_now_iso(),
                )

        await self._clean_paths(purge_id)

        latest = await self._db.get_user_data_purge(purge_id)
        if latest is not None and not latest.get("sidecar_completed_at"):
            await self._db.begin_user_data_purge_component(
                purge_id,
                component="sidecar",
                attempted_at=_now_iso(),
            )
            try:
                remote = await self._archive_transport.request_user_purge(
                    registry=self._registry,
                    user=user,
                    purge_id=purge_id,
                    cutoff_at=cutoff_at,
                )
                if (
                    str(remote.get("purge_id") or "") != purge_id
                    or str(remote.get("cutoff_at") or "") != cutoff_at
                ):
                    raise RuntimeError(
                        "sidecar purge returned a mismatched receipt"
                    )
                remote_status = str(remote.get("status") or "pending")
                acknowledged_at = _now_iso()
                await self._db.acknowledge_user_data_purge_sidecar(
                    purge_id,
                    acknowledged_at=acknowledged_at,
                    status=remote_status,
                )
                unblock_remote_personal_reads(
                    user=user,
                    purge_id=purge_id,
                )
                if remote_status == "completed":
                    await self._db.finish_user_data_purge_component(
                        purge_id,
                        component="sidecar",
                        completed_at=acknowledged_at,
                        status="completed",
                    )
            except Exception as exc:  # noqa: BLE001 — retry remote cleanup
                await self._db.fail_user_data_purge_component(
                    purge_id,
                    component="sidecar",
                    error=f"{type(exc).__name__}: {exc}",
                )

        await self._db.finalize_user_data_purge(
            purge_id,
            completed_at=_now_iso(),
        )

    async def _clean_paths(self, purge_id: str) -> None:
        rows = await self._db.pending_user_data_purge_paths(
            purge_id,
            limit=self._batch_size,
        )
        for row in rows:
            kind = str(row["kind"])
            object_id = str(row["object_id"])
            user = str(row["user"])
            await self._db.begin_user_data_purge_path(
                purge_id,
                kind=kind,
                object_id=object_id,
                attempted_at=_now_iso(),
            )
            try:
                errors = await asyncio.to_thread(
                    self._delete_path,
                    user,
                    kind,
                    object_id,
                )
                if errors:
                    raise OSError("; ".join(errors))
            except Exception as exc:  # noqa: BLE001 — disk failures stay retryable
                await self._db.fail_user_data_purge_path(
                    purge_id,
                    kind=kind,
                    object_id=object_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            await self._db.finish_user_data_purge_path(
                purge_id,
                kind=kind,
                object_id=object_id,
                completed_at=_now_iso(),
            )

    def _delete_path(self, user: str, kind: str, object_id: str) -> list[str]:
        token = _safe_object_id(object_id)
        if kind == "session":
            path = self._upload_root / sanitize_user(user) / ".sessions" / token
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                return [f"remove session: {type(exc).__name__}: {exc}"]
            return []
        if kind == "file_prefix":
            return delete_disk_files(self._upload_root, token, user)
        raise ValueError("unknown purge cleanup kind")

    async def _run(self) -> None:
        while True:
            try:
                self._wake.clear()
                attempted = await self.drain_once()
                if attempted >= self._batch_size:
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
            except Exception as exc:
                log.exception("user_data_purge: worker iteration failed: %s", exc)
                await asyncio.sleep(self._retry_interval_s)


__all__ = ["UserDataPurgeCoordinator"]
