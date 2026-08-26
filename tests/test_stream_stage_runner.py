"""Contracts for the shared deep/research stream stage runner."""

from __future__ import annotations

import asyncio

import pytest

from audrey.metrics import pipeline_seconds, pipeline_total
from audrey.pipeline.streaming import (
    StreamArchiveTarget,
    StreamOutcome,
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


async def test_event_channel_propagates_the_producer_exception_after_buffered_events():
    async def events():
        yield "before failure"
        raise ValueError("producer failed")

    runner = _runner(task_type="stage-runner-error")
    channel = runner.start_events(events(), maxsize=1, name="test-error-producer")

    assert await channel.receive() == "before failure"
    with pytest.raises(ValueError, match="producer failed"):
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
    }]

    with pytest.raises(RuntimeError, match="already finalized"):
        await runner.finalize(
            assistant_content="duplicate",
            concrete_model="writer",
        )
    assert total._value.get() == total_before + 1
    assert len(archive.calls) == 1

