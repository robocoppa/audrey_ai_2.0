"""Durable canonical-to-search projection handoff contracts."""

from __future__ import annotations

import asyncio
from typing import Any

from audrey.app_state import ApplicationStore
from audrey.chat_projection import ChatProjectionPromoter, combine_repair_stats


class _ArchiveQueue:
    def __init__(self, *, accepted: bool) -> None:
        self.accepted = accepted
        self.calls: list[dict[str, Any]] = []
        self.deletion_calls: list[dict[str, str]] = []

    async def archive_turn(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return self.accepted

    async def request_conversation_deletion(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> bool:
        self.deletion_calls.append(
            {"user_id": user_id, "conversation_id": conversation_id}
        )
        return self.accepted


async def _wait_for(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for projection worker")


async def _terminal_turn(store: ApplicationStore):
    owner = await store.resolve_external_identity(
        provider="owui",
        subject="owui-alice",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace="alice@example.com",
    )
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Canonical question",
    )
    assert started is not None
    await store.conversations.finish_run(
        user_id=owner.user_id,
        run_id=started.run.run_id,
        outcome="succeeded",
        assistant_content="Canonical answer",
        finish_reason="stop",
        virtual_model="audrey_fast",
        concrete_model="qwen-test",
        prompt_tokens=5,
        completion_tokens=3,
    )
    return owner, conversation, started


async def test_promoter_retries_failed_handoff_and_rebuilds_with_stable_id(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, conversation, started = await _terminal_turn(store)
    queue = _ArchiveQueue(accepted=False)
    promoter = ChatProjectionPromoter(
        store=store,
        archive_queue=queue,  # type: ignore[arg-type]
        retry_interval_s=60,
        batch_size=50,
    )
    try:
        await promoter.start()
        await _wait_for(lambda: len(queue.calls) == 1)
        failed = await promoter.user_stats(owner.user_id)
        assert failed["pending"] == 1
        assert failed["attempts"] == 1
        assert failed["with_error"] == 1

        queue.accepted = True
        assert await promoter.retry_now() == 1
        await _wait_for(lambda: len(queue.calls) == 2)
        delivered = await promoter.repair_stats()
        assert delivered["pending"] == 0
        assert delivered["completed"] == 1

        assert await promoter.rebuild() == 1
        await _wait_for(lambda: len(queue.calls) == 3)
        rebuilt = await promoter.repair_stats()
        assert rebuilt["pending"] == 0
        assert rebuilt["completed"] == 1

        expected_id = f"native:{started.run.run_id}"
        assert {call["archive_id"] for call in queue.calls} == {expected_id}
        assert {call["user_id"] for call in queue.calls} == {
            owner.storage_namespace
        }
        assert {call["conversation_id"] for call in queue.calls} == {
            conversation.conversation_id
        }
        assert queue.calls[-1]["user_content"] == "Canonical question"
        assert queue.calls[-1]["assistant_content"] == "Canonical answer"
    finally:
        await promoter.stop()
        store.close()


async def test_promoter_disabled_mode_does_not_process_or_double_start(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    await _terminal_turn(store)
    queue = _ArchiveQueue(accepted=True)
    promoter = ChatProjectionPromoter(
        store=store,
        archive_queue=queue,  # type: ignore[arg-type]
    )
    try:
        await promoter.start(run_worker=False)
        try:
            await promoter.start(run_worker=False)
        except RuntimeError as exc:
            assert str(exc) == "chat projection promoter already started"
        else:
            raise AssertionError("a second promoter start must fail")
        assert await promoter.run_once() == 0
        assert queue.calls == []
    finally:
        await promoter.stop()
        store.close()


async def test_promoter_retries_native_conversation_deletion_tombstone(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, conversation, _started = await _terminal_turn(store)
    queue = _ArchiveQueue(accepted=True)
    promoter = ChatProjectionPromoter(
        store=store,
        archive_queue=queue,  # type: ignore[arg-type]
        retry_interval_s=60,
    )
    try:
        await promoter.start()
        await _wait_for(lambda: len(queue.calls) == 1)
        assert await store.conversations.delete(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        queue.accepted = False
        promoter.wake()
        await _wait_for(lambda: len(queue.deletion_calls) == 1)
        failed = await promoter.user_stats(owner.user_id)
        assert failed["pending"] == 1
        assert failed["attempts"] == 1
        assert failed["with_error"] == 1

        queue.accepted = True
        assert await promoter.retry_now() == 1
        await _wait_for(lambda: len(queue.deletion_calls) == 2)
        repaired = await promoter.user_stats(owner.user_id)
        assert repaired["pending"] == 0
        # The write receipt cascades with its deleted canonical conversation;
        # only the durable deletion handoff remains as completed history.
        assert repaired["completed"] == 1
        assert queue.deletion_calls == [
            {
                "user_id": owner.storage_namespace,
                "conversation_id": conversation.conversation_id,
            },
            {
                "user_id": owner.storage_namespace,
                "conversation_id": conversation.conversation_id,
            },
        ]
    finally:
        await promoter.stop()
        store.close()


def test_combined_repair_stats_preserve_oldest_pending_receipt():
    combined = combine_repair_stats(
        {
            "pending": 2,
            "attempts": 3,
            "with_error": 1,
            "exhausted": 0,
            "completed": 4,
            "oldest_created_at": "2026-02-01T00:00:00+00:00",
        },
        {
            "pending": 1,
            "attempts": 2,
            "with_error": 1,
            "exhausted": 0,
            "completed": 5,
            "oldest_created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    assert combined == {
        "pending": 3,
        "attempts": 5,
        "with_error": 2,
        "exhausted": 0,
        "completed": 9,
        "oldest_created_at": "2026-01-01T00:00:00+00:00",
    }
