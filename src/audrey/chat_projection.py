"""Promote canonical conversation turns into the durable search outbox."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from audrey.app_state import (
    ApplicationStore,
    ChatProjectionDeletionRecord,
    ChatProjectionRecord,
)
from audrey.pipeline.chat_archive import ChatArchiveQueue

log = logging.getLogger(__name__)


class ChatProjectionPromoter:
    """Move due canonical receipts into the existing archive delivery queue.

    The canonical transaction owns the receipt; the compatibility archive
    queue owns delivery. A crash on either side of the handoff is safe because
    both use the stable projection id and both writes are idempotent.
    """

    def __init__(
        self,
        *,
        store: ApplicationStore,
        archive_queue: ChatArchiveQueue,
        retry_interval_s: float = 30.0,
        batch_size: int = 50,
    ) -> None:
        if retry_interval_s <= 0:
            raise ValueError("chat projection retry interval must be positive")
        if not 1 <= batch_size <= 200:
            raise ValueError("chat projection batch size must be between 1 and 200")
        self._store = store
        self._archive_queue = archive_queue
        self._retry_interval_s = retry_interval_s
        self._batch_size = batch_size
        self._wake = asyncio.Event()
        self._pass_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._started = False
        self._accepting = False

    async def start(self, *, run_worker: bool = True) -> None:
        if self._started:
            raise RuntimeError("chat projection promoter already started")
        self._started = True
        self._accepting = run_worker
        if run_worker:
            self._worker = asyncio.create_task(
                self._run(),
                name="canonical-chat-projection",
            )
            self.wake()
        stats = await self.repair_stats()
        log.info(
            "chat_projection: ready pending=%d worker=%s",
            stats["pending"],
            "on" if run_worker else "off",
        )

    async def stop(self) -> None:
        self._accepting = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._started = False

    def wake(self) -> None:
        if self._accepting:
            self._wake.set()

    async def retry_now(self, *, limit: int = 200) -> int:
        made_due = await self._store.chat_projections.retry_now(limit=limit)
        made_due += await self._store.chat_projections.retry_deletions_now(
            limit=limit
        )
        if self._accepting:
            await self.run_once()
            self.wake()
        return made_due

    async def rebuild(self) -> int:
        reset = await self._store.chat_projections.reset_all()
        if self._accepting:
            await self.run_once()
            self.wake()
        return reset

    async def repair_stats(self) -> dict[str, int | str]:
        writes, deletions = await asyncio.gather(
            self._store.chat_projections.stats(),
            self._store.chat_projections.deletion_stats(),
        )
        return combine_repair_stats(writes, deletions)

    async def user_stats(self, user_id: str) -> dict[str, int | str]:
        writes, deletions = await asyncio.gather(
            self._store.chat_projections.stats(user_id=user_id),
            self._store.chat_projections.deletion_stats(user_id=user_id),
        )
        return combine_repair_stats(writes, deletions)

    async def run_once(self) -> int:
        if not self._accepting:
            return 0
        async with self._pass_lock:
            rows = await self._store.chat_projections.due(limit=self._batch_size)
            for row in rows:
                await self._promote(row)
            deletions = await self._store.chat_projections.due_deletions(
                limit=self._batch_size
            )
            for deletion in deletions:
                await self._promote_deletion(deletion)
            return len(rows) + len(deletions)

    async def _promote(self, row: ChatProjectionRecord) -> None:
        try:
            accepted = await self._archive_queue.archive_turn(
                registry=None,
                user_id=row.storage_namespace,
                conversation_id=row.conversation_id,
                user_content=row.user_content,
                assistant_content=row.assistant_content,
                partial=row.partial,
                virtual_model=row.virtual_model,
                concrete_model=row.concrete_model,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                archive_id=row.projection_id,
                created_at=row.created_at,
            )
        except Exception as exc:  # noqa: BLE001 - durable receipt retains work
            accepted = False
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = "archive queue rejected projection"
        if accepted:
            await self._store.chat_projections.mark_enqueued(
                projection_id=row.projection_id
            )
            return
        await self._store.chat_projections.mark_failed(
            projection_id=row.projection_id,
            error=error,
            retry_interval_s=self._retry_interval_s,
        )
        log.warning(
            "chat_projection: deferred projection=%s conversation=%s",
            row.projection_id,
            row.conversation_id,
        )

    async def _promote_deletion(
        self,
        deletion: ChatProjectionDeletionRecord,
    ) -> None:
        try:
            accepted = await self._archive_queue.request_conversation_deletion(
                user_id=deletion.storage_namespace,
                conversation_id=deletion.conversation_id,
            )
        except Exception as exc:  # noqa: BLE001 - durable tombstone retains work
            accepted = False
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = "archive queue rejected conversation deletion"
        if accepted:
            await self._store.chat_projections.mark_deletion_completed(
                deletion_id=deletion.deletion_id
            )
            return
        await self._store.chat_projections.mark_deletion_failed(
            deletion_id=deletion.deletion_id,
            error=error,
            retry_interval_s=self._retry_interval_s,
        )
        log.warning(
            "chat_projection: deferred deletion=%s conversation=%s",
            deletion.deletion_id,
            deletion.conversation_id,
        )

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._retry_interval_s,
                )
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                processed = await self.run_once()
            except Exception:
                log.exception("chat_projection: maintenance pass failed")
                continue
            if processed >= self._batch_size:
                self.wake()


def combine_repair_stats(*values: dict[str, Any] | None) -> dict[str, int | str]:
    """Combine adjacent durable handoff queues without exposing payload data."""

    present = [value for value in values if value is not None]
    if not present:
        return {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
            "oldest_created_at": "",
        }
    oldest = sorted(
        str(value.get("oldest_created_at") or "")
        for value in present
        if value.get("oldest_created_at")
    )
    return {
        "pending": sum(int(value.get("pending", 0)) for value in present),
        "attempts": sum(int(value.get("attempts", 0)) for value in present),
        "with_error": sum(int(value.get("with_error", 0)) for value in present),
        "exhausted": sum(int(value.get("exhausted", 0)) for value in present),
        "completed": sum(int(value.get("completed", 0)) for value in present),
        "oldest_created_at": oldest[0] if oldest else "",
    }


__all__ = ["ChatProjectionPromoter", "combine_repair_stats"]
