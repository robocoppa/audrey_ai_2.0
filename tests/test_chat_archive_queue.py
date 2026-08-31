"""Failure-injection contracts for Audrey's durable archive outbox."""

from __future__ import annotations

import asyncio
from pathlib import Path

from audrey.metrics import chat_archive_queue_events_total
from audrey.pipeline.chat_archive import (
    ArchiveDelivery,
    ArchiveJob,
    ChatArchiveQueue,
)


class _Client:
    def __init__(
        self,
        *,
        outcome: ArchiveDelivery = ArchiveDelivery.DELIVERED,
        error: str = "",
        block: bool = False,
    ) -> None:
        self.outcome = outcome
        self.error = error
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[ArchiveJob] = []

    def host_url(self, _registry) -> str:
        return "http://custom-tools:8001"

    async def deliver_job(self, *, registry, job: ArchiveJob):
        del registry
        self.calls.append(job)
        self.started.set()
        if self.block:
            await self.release.wait()
        return self.outcome, self.error


def _queue(
    path: Path,
    client: _Client,
    *,
    maxsize: int = 8,
    retry_interval_s: float = 60.0,
) -> ChatArchiveQueue:
    return ChatArchiveQueue(
        client=client,  # type: ignore[arg-type] — intentional transport fake
        registry=object(),  # type: ignore[arg-type] — queue passes it through
        sqlite_path=path,
        maxsize=maxsize,
        retry_interval_s=retry_interval_s,
    )


async def _enqueue(
    queue: ChatArchiveQueue,
    *,
    archive_id: str,
    user_id: str = "alice@example.com",
) -> None:
    await queue.archive_turn(
        registry=None,
        user_id=user_id,
        conversation_id="conversation-1",
        user_content="question",
        assistant_content=f"answer {archive_id}",
        virtual_model="audrey_fast",
        concrete_model="qwen-test",
        archive_id=archive_id,
        created_at="2026-08-28T12:00:00.000000+00:00",
    )


async def _wait_for_pending(queue: ChatArchiveQueue, expected: int) -> None:
    for _ in range(200):
        if await queue.pending_count() == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"archive queue did not reach pending={expected}")


async def _wait_for_attempt(queue: ChatArchiveQueue) -> None:
    for _ in range(200):
        if (await queue.stats())["attempts"] >= 1:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("archive queue did not attempt delivery")


async def test_response_enqueue_does_not_wait_for_remote_delivery(tmp_path: Path):
    client = _Client(block=True)
    queue = _queue(tmp_path / "archive-outbox.sqlite", client)
    await queue.start()
    try:
        await _enqueue(queue, archive_id="job-1")
        await asyncio.wait_for(client.started.wait(), timeout=1)

        assert await queue.pending_count() == 1
        assert not client.release.is_set()

        client.release.set()
        await _wait_for_pending(queue, 0)
    finally:
        client.release.set()
        await queue.stop()


async def test_full_wake_queue_keeps_durable_rows_for_restart(
    tmp_path: Path,
    caplog,
):
    path = tmp_path / "archive-outbox.sqlite"
    blocked = _Client(block=True)
    queue = _queue(path, blocked, maxsize=1)
    overflow = chat_archive_queue_events_total.labels(result="overflow")
    before = overflow._value.get()
    await queue.start()
    await asyncio.sleep(0)

    try:
        await _enqueue(queue, archive_id="job-1")
        await asyncio.wait_for(blocked.started.wait(), timeout=1)
        await _enqueue(queue, archive_id="job-2")
        await _enqueue(queue, archive_id="job-3")

        assert await queue.pending_count() == 3
        assert overflow._value.get() >= before + 1
        assert "durable source retained" in caplog.text
    finally:
        await queue.stop()

    recovered = _Client()
    restarted = _queue(path, recovered, maxsize=1)
    await restarted.start()
    try:
        await _wait_for_pending(restarted, 0)
        assert {job.archive_id for job in recovered.calls} == {
            "job-1",
            "job-2",
            "job-3",
        }
    finally:
        await restarted.stop()


async def test_failed_delivery_retries_after_restart_with_same_identity(
    tmp_path: Path,
):
    path = tmp_path / "archive-outbox.sqlite"
    failed = _Client(outcome=ArchiveDelivery.RETRY, error="upstream unavailable")
    queue = _queue(path, failed)
    await queue.start()
    try:
        await _enqueue(queue, archive_id="stable-job")
        await _wait_for_attempt(queue)
        for _ in range(200):
            if (await queue.stats())["last_error"] == "upstream unavailable":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("archive queue did not persist the delivery error")
        assert await queue.pending_count() == 1
    finally:
        await queue.stop()

    recovered = _Client()
    restarted = _queue(path, recovered)
    await restarted.start()
    try:
        await _wait_for_pending(restarted, 0)
        assert recovered.calls[0].archive_id == "stable-job"
        assert recovered.calls[0].created_at == "2026-08-28T12:00:00.000000+00:00"
    finally:
        await restarted.stop()


async def test_user_stats_do_not_mix_delivery_backlogs(tmp_path: Path):
    client = _Client(block=True)
    queue = _queue(tmp_path / "archive-outbox.sqlite", client)
    await queue.start()
    try:
        await _enqueue(queue, archive_id="alice-job")
        await asyncio.wait_for(client.started.wait(), timeout=1)
        await _enqueue(queue, archive_id="bob-job", user_id="bob")

        alice = await queue.user_stats("alice@example.com")
        bob = await queue.user_stats("bob")
        missing = await queue.user_stats("missing")

        assert alice["pending"] == 1
        assert bob["pending"] == 1
        assert missing["pending"] == 0
        assert alice["pending"] + bob["pending"] == await queue.pending_count()
    finally:
        client.release.set()
        await queue.stop()
