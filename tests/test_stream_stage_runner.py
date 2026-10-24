"""Contracts for the shared deep/research stream stage runner."""

from __future__ import annotations

import asyncio

import pytest

from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.pipeline.streaming import (
    StreamArchiveTarget,
    StreamOutcome,
    StreamQueueWake,
    StreamSignalQueue,
    StreamStageRunner,
)


class _Archive:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def archive_turn(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _runner(
    *,
    task_type: str,
    archive: _Archive | None = None,
    conversation_id: str = "",
) -> StreamStageRunner:
    return StreamStageRunner(
        mode="deep",
        task_type=task_type,
        archive=StreamArchiveTarget(
            client=archive,
            registry="registry",
            user_id="alice@example.com",
            conversation_id=conversation_id,
            user_content="question",
            virtual_model="audrey_deep",
        ),
    )


def _histogram_count(histogram) -> float:
    return next(
        sample.value
        for sample in histogram.collect()[0].samples
        if sample.name.endswith("_count")
    )


async def test_event_channel_is_bounded_and_drains_to_clean_completion():
    async def events():
        for value in range(3):
            yield value

    runner = _runner(task_type="stage-runner-bounded")
    channel = runner.start_events(events(), maxsize=1, name="test-bounded-producer")

    await asyncio.sleep(0)
    assert not channel.producer_task.done()

    assert [await channel.receive(), await channel.receive(), await channel.receive()] == [
        0,
        1,
        2,
    ]
    assert await channel.receive() is None

    runner.terminal.finish(StreamOutcome.OK, finish_reason="stop")
    await runner.finalize(assistant_content="answer", concrete_model="writer")


async def test_finalization_cancels_and_drains_a_blocked_producer():
    settled = asyncio.Event()

    async def events():
        try:
            yield "ready"
            await asyncio.Event().wait()
        finally:
            settled.set()

    runner = _runner(task_type="stage-runner-cancel")
    channel = runner.start_events(events(), maxsize=1, name="test-cancel-producer")
    assert await channel.receive() == "ready"

    runner.terminal.finish(StreamOutcome.CANCELLED)
    await runner.finalize(assistant_content="", concrete_model="writer")

    assert settled.is_set()
    assert channel.producer_task.done()


@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
async def test_event_channel_propagates_the_producer_exception_after_buffered_events(error_type):
    async def events():
        yield "before failure"
        raise error_type("producer failed")

    runner = _runner(task_type="stage-runner-error")
    channel = runner.start_events(events(), maxsize=1, name="test-error-producer")

    assert await channel.receive() == "before failure"
    with pytest.raises(error_type, match="producer failed"):
        await channel.receive()

    runner.terminal.finish(StreamOutcome.ERROR, finish_reason="stop")
    await runner.finalize(
        assistant_content="before failure",
        concrete_model="writer",
    )


async def test_terminal_finalization_records_one_metric_and_partial_archive():
    archive = _Archive()
    task_type = "stage-runner-finalize"
    total = pipeline_total.labels(
        mode="deep",
        task_type=task_type,
        outcome="truncated",
    )
    seconds = pipeline_seconds.labels(mode="deep", task_type=task_type)
    total_before = total._value.get()
    count_before = _histogram_count(seconds)

    runner = _runner(
        task_type=task_type,
        archive=archive,
        conversation_id="conversation-1",
    )
    runner.terminal.finish(StreamOutcome.TRUNCATED, finish_reason="stop")
    elapsed = await runner.finalize(
        assistant_content="partial answer",
        concrete_model="writer",
    )

    assert elapsed >= 0
    assert total._value.get() == total_before + 1
    assert _histogram_count(seconds) == count_before + 1
    assert archive.calls == [{
        "registry": "registry",
        "user_id": "alice@example.com",
        "conversation_id": "conversation-1",
        "user_content": "question",
        "assistant_content": "partial answer",
        "partial": True,
        "virtual_model": "audrey_deep",
        "concrete_model": "writer",
        "skill_id": "",
        "skill_version": 0,
        "skill_digest": "",
        "skill_reason": "",
    }]

    with pytest.raises(RuntimeError, match="already finalized"):
        await runner.finalize(
            assistant_content="duplicate",
            concrete_model="writer",
        )
    assert total._value.get() == total_before + 1
    assert len(archive.calls) == 1



def _pending_ready_waiters() -> list[asyncio.Task]:
    return [
        task for task in asyncio.all_tasks()
        if task.get_name() == "stream-queue-ready" and not task.done()
    ]


async def test_signal_queue_tracks_buffered_data_without_consuming_it():
    queue = StreamSignalQueue[str](maxsize=1)
    waiting = asyncio.create_task(queue.wait_ready())
    await asyncio.sleep(0)
    assert not waiting.done()

    queue.put_nowait("first")
    await asyncio.wait_for(waiting, timeout=1)
    with pytest.raises(asyncio.QueueFull):
        queue.put_nowait("does not fit")
    await queue.wait_ready()
    assert queue.get_nowait() == "first"

    waiting = asyncio.create_task(queue.wait_ready())
    await asyncio.sleep(0)
    assert not waiting.done()
    await queue.put("second")
    await asyncio.wait_for(waiting, timeout=1)
    assert queue.get_nowait() == "second"
    assert queue.empty()


async def test_banner_wakes_poll_without_consuming_banner_or_cancelling_producer():
    release = asyncio.Event()

    async def events():
        await release.wait()
        yield "token"

    runner = _runner(task_type="stage-runner-banner")
    channel = runner.start_events(events(), maxsize=1, name="test-banner-producer")
    banners = StreamSignalQueue[str]()
    waiting = asyncio.create_task(channel.poll(wake_queue=banners))
    await asyncio.sleep(0)
    assert not waiting.done()
    banners.put_nowait("working")

    with pytest.raises(StreamQueueWake):
        await asyncio.wait_for(waiting, timeout=1)
    assert not channel.producer_task.done()
    assert banners.get_nowait() == "working"
    assert not _pending_ready_waiters()

    release.set()
    assert await channel.receive() == "token"
    assert await channel.receive() is None
    await runner.cancel_and_drain()


async def test_cancelling_poll_preserves_simultaneous_data_and_does_not_orphan_waiters():
    release = asyncio.Event()

    async def events():
        await release.wait()
        yield "token"

    runner = _runner(task_type="stage-runner-cancel-wait")
    channel = runner.start_events(events(), maxsize=1, name="test-cancel-wait-producer")
    banners = StreamSignalQueue[str]()
    waiting = asyncio.create_task(channel.poll(wake_queue=banners))
    await asyncio.sleep(0)
    assert _pending_ready_waiters()

    banners.put_nowait("working")
    release.set()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiting, timeout=1)

    assert not channel.producer_task.cancelled()
    assert not _pending_ready_waiters()
    assert banners.get_nowait() == "working"
    assert await channel.receive() == "token"
    assert await channel.receive() is None
    await runner.cancel_and_drain()


@pytest.mark.parametrize("failed", [False, True])
async def test_buffered_event_then_completion_precede_simultaneous_banner(failed):
    async def events():
        yield "token"
        if failed:
            raise ValueError("producer failed")

    runner = _runner(task_type="stage-runner-simultaneous")
    channel = runner.start_events(events(), maxsize=1, name="test-simultaneous-producer")
    await asyncio.gather(channel.producer_task, return_exceptions=True)
    banners = StreamSignalQueue[str]()
    banners.put_nowait("working")

    assert await channel.poll(wake_queue=banners) == "token"
    if failed:
        with pytest.raises(ValueError, match="producer failed"):
            await channel.poll(wake_queue=banners)
    else:
        assert await channel.poll(wake_queue=banners) is None
    assert banners.get_nowait() == "working"
    assert not _pending_ready_waiters()
    await runner.cancel_and_drain()


async def test_empty_producer_wakes_waiting_receiver_on_clean_exit():
    release = asyncio.Event()

    async def events():
        await release.wait()
        if False:
            yield "unreachable"

    runner = _runner(task_type="stage-runner-empty")
    channel = runner.start_events(events(), maxsize=1, name="test-empty-producer")
    waiting = asyncio.create_task(channel.receive())
    await asyncio.sleep(0)
    assert not waiting.done()
    release.set()

    assert await asyncio.wait_for(waiting, timeout=1) is None
    assert not _pending_ready_waiters()
    await runner.cancel_and_drain()


async def test_finalization_cancels_producer_when_delivery_queue_is_full():
    second_offered = asyncio.Event()

    async def events():
        yield "buffered"
        second_offered.set()
        yield "blocked"

    runner = _runner(task_type="stage-runner-full-finalize")
    channel = runner.start_events(events(), maxsize=1, name="test-full-producer")
    await second_offered.wait()
    assert channel._queue.full()
    assert not channel.producer_task.done()

    runner.terminal.finish(StreamOutcome.CANCELLED)
    await asyncio.wait_for(
        runner.finalize(assistant_content="", concrete_model="writer"), timeout=1,
    )

    assert channel.producer_task.cancelled()
    assert channel._queue.get_nowait() == "buffered"
    assert not _pending_ready_waiters()
