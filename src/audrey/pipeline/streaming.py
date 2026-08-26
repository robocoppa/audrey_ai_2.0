"""Client-neutral terminal state shared by streamed pipeline owners.

Async generators cannot return a value to their async-for caller. A
StreamTerminal is therefore passed into the generator: the inner stream reports
what actually happened, and the outer adapter records metrics, archive state,
and the protocol-specific finish frame exactly once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from audrey.metrics import pipeline_seconds, pipeline_total


class StreamOutcome(StrEnum):
    """Bounded terminal labels used by stream metrics and persistence."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TRUNCATED = "truncated"


@dataclass(slots=True)
class StreamTerminal:
    """One-shot result channel from an inner stream to its outer owner."""

    _outcome: StreamOutcome | None = None
    _finish_reason: str | None = None

    @property
    def is_final(self) -> bool:
        return self._outcome is not None

    @property
    def outcome(self) -> StreamOutcome:
        if self._outcome is None:
            raise RuntimeError("stream terminal outcome has not been reported")
        return self._outcome

    @property
    def finish_reason(self) -> str | None:
        if self._outcome is None:
            raise RuntimeError("stream terminal outcome has not been reported")
        return self._finish_reason

    def finish(
        self,
        outcome: StreamOutcome,
        *,
        finish_reason: str | None = None,
    ) -> None:
        """Report the terminal result once; duplicate ownership is a bug."""
        if self._outcome is not None:
            raise RuntimeError(
                "stream terminal outcome already reported as "
                f"{self._outcome.value!r}"
            )
        self._outcome = outcome
        self._finish_reason = finish_reason

    def finish_if_unset(
        self,
        outcome: StreamOutcome,
        *,
        finish_reason: str | None = None,
    ) -> None:
        """Outer-owner fallback for cancellation or an escaping exception."""
        if self._outcome is None:
            self.finish(outcome, finish_reason=finish_reason)


@dataclass(frozen=True, slots=True)
class StreamArchiveTarget:
    """Stable archive identity for one streamed request."""

    client: Any | None
    registry: Any
    user_id: str
    conversation_id: str
    user_content: str
    virtual_model: str

    async def write(
        self,
        *,
        assistant_content: str,
        concrete_model: str,
        partial: bool,
    ) -> None:
        """Persist the answer body when chat archiving is enabled."""
        if self.client is None or not self.conversation_id:
            return
        await self.client.archive_turn(
            registry=self.registry,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            user_content=self.user_content,
            assistant_content=assistant_content,
            partial=partial,
            virtual_model=self.virtual_model,
            concrete_model=concrete_model,
        )


@dataclass(slots=True)
class StreamEventChannel[EventT]:
    """Bounded delivery channel tied to its one producer task.

    Completion is derived from the producer task instead of a sentinel put.
    That matters when a disconnected client leaves the queue full: cancelling
    the producer is enough to settle it, without first making queue space.
    """

    _queue: asyncio.Queue[EventT]
    producer_task: asyncio.Task[None]

    async def poll(self) -> EventT | None:
        """Poll once while leaving time for an independent banner queue."""
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self.producer_task.done():
            await self.producer_task
            return None
        await asyncio.sleep(0.05)
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self.producer_task.done():
            await self.producer_task
            return None
        raise TimeoutError

    async def receive(self) -> EventT | None:
        """Return the next event, or None after a clean producer exit."""
        while True:
            try:
                return await self.poll()
            except TimeoutError:
                continue


class StreamStageRunner:
    """Own shared mechanics for one deep or research stream.

    The route branches retain stage policy and banner interpretation. This
    owner is deliberately smaller: child-task lifetime, bounded producer
    delivery, one terminal outcome, metrics, and answer-only archiving.
    """

    def __init__(
        self,
        *,
        mode: str,
        task_type: str,
        archive: StreamArchiveTarget,
    ) -> None:
        self.mode = mode
        self.task_type = task_type
        self.archive = archive
        self.terminal = StreamTerminal()
        self._started_at = time.perf_counter()
        self._owned_tasks: list[asyncio.Task[Any]] = []
        self._finalized = False

    def own[ResultT](
        self,
        coroutine: Coroutine[Any, Any, ResultT],
        *,
        name: str,
    ) -> asyncio.Task[ResultT]:
        """Create and register one request-owned child task."""
        task = asyncio.create_task(coroutine, name=name)
        self._owned_tasks.append(task)
        return task

    def start_events[EventT](
        self,
        source: AsyncIterable[EventT],
        *,
        maxsize: int,
        name: str,
    ) -> StreamEventChannel[EventT]:
        """Pump an async event source into a bounded request-owned channel."""
        if maxsize < 1:
            raise ValueError("stream event channel maxsize must be positive")
        queue: asyncio.Queue[EventT] = asyncio.Queue(maxsize=maxsize)

        async def _pump() -> None:
            async for event in source:
                await queue.put(event)

        return StreamEventChannel(queue, self.own(_pump(), name=name))

    async def cancel_and_drain(self) -> None:
        """Cancel unfinished children and retrieve every terminal result."""
        for task in self._owned_tasks:
            if not task.done():
                task.cancel()
        if self._owned_tasks:
            await asyncio.gather(*self._owned_tasks, return_exceptions=True)

    async def finalize(
        self,
        *,
        assistant_content: str,
        concrete_model: str,
    ) -> float:
        """Settle children and record terminal state exactly once."""
        if self._finalized:
            raise RuntimeError("stream stage runner already finalized")
        self._finalized = True

        await self.cancel_and_drain()
        self.terminal.finish_if_unset(StreamOutcome.ERROR, finish_reason="stop")
        elapsed = time.perf_counter() - self._started_at
        outcome = self.terminal.outcome
        pipeline_seconds.labels(mode=self.mode, task_type=self.task_type).observe(elapsed)
        pipeline_total.labels(
            mode=self.mode,
            task_type=self.task_type,
            outcome=outcome.value,
        ).inc()
        await self.archive.write(
            assistant_content=assistant_content,
            concrete_model=concrete_model,
            partial=(outcome is not StreamOutcome.OK),
        )
        return elapsed


__all__ = [
    "StreamArchiveTarget",
    "StreamEventChannel",
    "StreamOutcome",
    "StreamStageRunner",
    "StreamTerminal",
]
