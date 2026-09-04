"""Failure-injection contracts for Audrey's durable archive outbox."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from audrey.metrics import chat_archive_queue_events_total
from audrey.pipeline.chat_archive import (
    ArchiveDelivery,
    ArchiveJob,
    ChatArchiveClient,
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
        self.deletion_calls: list[tuple[str, str]] = []

    def host_url(self, _registry) -> str:
        return "http://custom-tools:8001"

    async def deliver_job(self, *, registry, job: ArchiveJob):
        del registry
        self.calls.append(job)
        self.started.set()
        if self.block:
            await self.release.wait()
        return self.outcome, self.error

    async def request_conversation_deletion(
        self,
        *,
        registry,
        user: str,
        conversation_id: str,
    ) -> bool:
        del registry
        self.deletion_calls.append((user, conversation_id))
        return True


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
    conversation_id: str = "conversation-1",
    created_at: str = "2026-08-28T12:00:00.000000+00:00",
) -> None:
    await queue.archive_turn(
        registry=None,
        user_id=user_id,
        conversation_id=conversation_id,
        user_content="question",
        assistant_content=f"answer {archive_id}",
        virtual_model="audrey_fast",
        concrete_model="qwen-test",
        archive_id=archive_id,
        created_at=created_at,
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


async def test_start_migrates_legacy_outbox_conversation_selector(tmp_path: Path):
    path = tmp_path / "archive-outbox.sqlite"
    job = ArchiveJob.create(
        user_id="alice@example.com",
        conversation_id="conversation-legacy",
        user_content="question",
        assistant_content="answer",
        archive_id="legacy-job",
        created_at="2026-08-28T12:00:00.000000+00:00",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE archive_write_outbox (
                archive_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                next_attempt_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO archive_write_outbox "
            "(archive_id, payload_json, created_at, next_attempt_at) "
            "VALUES (?, ?, ?, ?)",
            (
                job.archive_id,
                json.dumps(asdict(job)),
                job.created_at,
                job.created_at,
            ),
        )

    queue = _queue(path, _Client())
    await queue.start(run_worker=False)
    await queue.stop()

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT user_id, conversation_id FROM archive_write_outbox "
            "WHERE archive_id = 'legacy-job'"
        ).fetchone()
    assert row == ("alice@example.com", "conversation-legacy")


async def test_conversation_delete_waits_for_inflight_write_and_discards_queued_writes(
    tmp_path: Path,
):
    client = _Client(block=True)
    queue = _queue(tmp_path / "archive-outbox.sqlite", client)
    await queue.start()
    try:
        await _enqueue(
            queue,
            archive_id="unrelated-inflight",
            conversation_id="conversation-other",
        )
        await asyncio.wait_for(client.started.wait(), timeout=1)
        await _enqueue(
            queue,
            archive_id="deleted-conversation-write",
            conversation_id="conversation-delete",
        )
        deletion = asyncio.create_task(
            queue.request_conversation_deletion(
                user_id="alice@example.com",
                conversation_id="conversation-delete",
            )
        )
        await asyncio.sleep(0)
        assert client.deletion_calls == []

        client.release.set()
        assert await asyncio.wait_for(deletion, timeout=1) is True
        await _wait_for_pending(queue, 0)

        assert [job.archive_id for job in client.calls] == ["unrelated-inflight"]
        assert client.deletion_calls == [
            ("alice@example.com", "conversation-delete")
        ]
    finally:
        client.release.set()
        await queue.stop()


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


async def test_user_purge_removes_old_rows_blocks_late_delivery_and_survives_restart(
    tmp_path: Path,
):
    path = tmp_path / "archive-outbox.sqlite"
    blocked = _Client(block=True)
    queue = _queue(path, blocked)
    await queue.start()
    try:
        await _enqueue(queue, archive_id="alice-old")
        await asyncio.wait_for(blocked.started.wait(), timeout=1)
        await _enqueue(queue, archive_id="bob-old", user_id="bob")
        await _enqueue(
            queue,
            archive_id="alice-new",
            created_at="2026-09-01T12:00:00.000000+00:00",
        )

        deleted = await queue.purge_user_before(
            user_id="alice@example.com",
            cutoff_at="2026-08-29T00:00:00.000000+00:00",
            purge_id="purge-1",
        )

        assert deleted == 1
        assert (await queue.user_stats("alice@example.com"))["pending"] == 1
        assert (await queue.user_stats("bob"))["pending"] == 1
        await _enqueue(queue, archive_id="alice-late-old")
        assert (await queue.user_stats("alice@example.com"))["pending"] == 1
    finally:
        blocked.release.set()
        await queue.stop()

    restarted_client = _Client(block=True)
    restarted = _queue(path, restarted_client)
    await restarted.start()
    try:
        before = await restarted.pending_count()
        await _enqueue(restarted, archive_id="alice-restart-late")
        assert await restarted.pending_count() == before
        await _enqueue(
            restarted,
            archive_id="alice-restart-new",
            created_at="2026-09-02T12:00:00.000000+00:00",
        )
        assert (await restarted.user_stats("alice@example.com"))["pending"] >= 1
    finally:
        restarted_client.release.set()
        await restarted.stop()

async def test_maintenance_only_queue_purges_without_delivering(
    tmp_path: Path,
):
    path = tmp_path / "archive-outbox.sqlite"
    blocked = _Client(block=True)
    active = _queue(path, blocked)
    await active.start()
    try:
        await _enqueue(active, archive_id="old-job")
        await asyncio.wait_for(blocked.started.wait(), timeout=1)
    finally:
        await active.stop()

    maintenance_client = _Client()
    maintenance = _queue(path, maintenance_client)
    await maintenance.start(run_worker=False)
    try:
        await asyncio.sleep(0)
        assert maintenance_client.calls == []
        assert await maintenance.pending_count() == 1

        deleted = await maintenance.purge_user_before(
            user_id="alice@example.com",
            cutoff_at="2026-08-29T00:00:00.000000+00:00",
            purge_id="maintenance-purge",
        )

        assert deleted == 1
        assert await maintenance.pending_count() == 0
        assert maintenance_client.calls == []
    finally:
        await maintenance.stop()


async def test_admin_retry_makes_bounded_delivery_due_and_sanitizes_stats(
    tmp_path: Path,
):
    client = _Client(
        outcome=ArchiveDelivery.RETRY,
        error="private upstream detail",
    )
    queue = _queue(
        tmp_path / "archive-outbox.sqlite",
        client,
        retry_interval_s=3600.0,
    )
    await queue.start()
    try:
        await _enqueue(queue, archive_id="repair-job")
        await _wait_for_attempt(queue)
        for _ in range(200):
            status = await queue.repair_stats()
            if status["with_error"] == 1:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("archive queue did not persist repair status")

        assert status == {
            "pending": 1,
            "attempts": 1,
            "with_error": 1,
            "exhausted": 0,
            "completed": 0,
        }
        assert "private upstream detail" not in str(status)

        client.outcome = ArchiveDelivery.DELIVERED
        client.error = ""
        assert await queue.retry_now(limit=1) == 1
        await _wait_for_pending(queue, 0)
    finally:
        await queue.stop()


async def test_admin_sidecar_controls_use_hidden_paths_and_service_token():
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {"indexing": {}, "deletions": {}},
    )
    http = SimpleNamespace(post=AsyncMock(return_value=response))
    credential = "test-value"
    client = ChatArchiveClient(
        http,  # type: ignore[arg-type]
        service_token=credential,
    )
    registry = SimpleNamespace(
        get=lambda _name: SimpleNamespace(server_url="http://custom-tools:8001"),
    )

    await client.repair_status(registry=registry)
    await client.repair(registry=registry)

    assert [call.args[0] for call in http.post.await_args_list] == [
        "http://custom-tools:8001/user_data/repair/status",
        "http://custom-tools:8001/user_data/repair/run",
    ]
    for call in http.post.await_args_list:
        assert call.kwargs["headers"] == {
            "X-Audrey-Service-Token": credential,
        }
        assert call.kwargs["json"] == {}


async def test_conversation_deletion_client_uses_service_auth_and_accepts_absence():
    response = SimpleNamespace(
        status_code=202,
        json=lambda: {"conversation_id": "conversation-1"},
    )
    http = SimpleNamespace(post=AsyncMock(return_value=response))
    credential = "test-value"
    client = ChatArchiveClient(
        http,  # type: ignore[arg-type]
        service_token=credential,
    )
    registry = SimpleNamespace(
        get=lambda _name: SimpleNamespace(server_url="http://custom-tools:8001"),
    )

    assert await client.request_conversation_deletion(
        registry=registry,
        user="alice@example.com",
        conversation_id="conversation-1",
    )
    call = http.post.await_args
    assert call.args[0] == (
        "http://custom-tools:8001/user_data/chat_history/delete"
    )
    assert call.kwargs["headers"] == {
        "X-Audrey-Service-Token": credential,
    }
    assert call.kwargs["json"] == {
        "user": "alice@example.com",
        "conversation_id": "conversation-1",
    }

    http.post.return_value = SimpleNamespace(status_code=404)
    assert await client.request_conversation_deletion(
        registry=registry,
        user="alice@example.com",
        conversation_id="conversation-missing",
    )

    http.post.return_value = SimpleNamespace(status_code=503, text="private")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await client.request_conversation_deletion(
            registry=registry,
            user="alice@example.com",
            conversation_id="conversation-1",
        )
