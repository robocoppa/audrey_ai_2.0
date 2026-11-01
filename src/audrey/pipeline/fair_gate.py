"""Local generations favor interactive work while preserving per-user fairness.

Running generations finish normally. At the next free slot, interactive
waiters pass unaged background work, including their own user's backlog.
Background work waiting long enough joins the interactive priority tier;
round-robin users and FIFO eligible requests then prevent starvation.

Admission and enqueue share one lock so a release cannot strand a waiter
between checking availability and entering the queue. Cancelled waiters
are swept before dispatch; cancellation after a grant transfers that slot
again even if the context body never starts. Cloud calls bypass the gate.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from audrey.metrics import gpu_gate_wait_seconds, local_admission_wait_seconds
from audrey.scheduling import (
    ANON_USER_BUCKET,
    current_background_queued_at,
    current_local_workload,
)


@dataclass(eq=False, slots=True)
class _Waiter:
    future: asyncio.Future[None]
    workload: Literal["interactive", "background"]
    enqueued_at: float


class FairLocalGate:
    """Per-user fair gate around local GPU slots with background aging."""

    def __init__(
        self, *, concurrency: int = 1, background_aging_seconds: float = 30.0,
    ) -> None:
        if concurrency < 1:
            concurrency = 1
        if not math.isfinite(background_aging_seconds) or background_aging_seconds < 0:
            raise ValueError("background_aging_seconds must be finite and nonnegative")
        self._concurrency = concurrency
        self._available = concurrency
        self._background_aging_seconds = background_aging_seconds
        self._waiters: OrderedDict[str, deque[_Waiter]] = OrderedDict()
        # Separate pointers preserve round-robin order within each eligible
        # tier, including user buckets joining after an earlier grant.
        self._last_granted: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @asynccontextmanager
    async def acquire(self, model: str, *, location: str, user_id: str | None = None):
        """Acquire the gate iff `location == 'local'`. Cloud is a no-op."""
        if location != "local":
            yield
            return

        bucket = (user_id or "").strip() or ANON_USER_BUCKET
        workload = current_local_workload()
        t0 = time.perf_counter()
        waiter: _Waiter | None = None

        # Availability and enqueue must be one transition: the holder could
        # otherwise release between two separately locked operations.
        async with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                self._last_granted[workload] = bucket
            else:
                queued_at = current_background_queued_at() if workload == "background" else None
                waiter = _Waiter(
                    asyncio.get_running_loop().create_future(), workload,
                    time.monotonic() if queued_at is None else queued_at,
                )
                dq = self._waiters.get(bucket)
                if dq is None:
                    dq = deque()
                    self._waiters[bucket] = dq
                dq.append(waiter)

        if waiter is not None:
            fut = waiter.future
            try:
                await fut
                await self._after_waiter_granted()
            except asyncio.CancelledError:
                # Cancellation of the queued await cancels its Future. If
                # dispatch already set a result, we own a slot and must
                # transfer it even though the context body never started.
                async with self._lock:
                    was_granted = fut.done() and not fut.cancelled()
                    dq = self._waiters.get(bucket)
                    if dq is not None:
                        try:
                            dq.remove(waiter)
                        except ValueError:
                            pass
                        if not dq:
                            self._waiters.pop(bucket, None)
                if was_granted:
                    await self._release()
                raise

        waited = time.perf_counter() - t0
        gpu_gate_wait_seconds.observe(waited)
        local_admission_wait_seconds.labels(stage="gpu", workload=workload).observe(waited)
        try:
            yield
        finally:
            await self._release()

    async def _after_waiter_granted(self) -> None:
        """Non-suspending production seam for grant-before-resume cancellation tests."""

    async def _release(self) -> None:
        """Transfer one released slot to the next eligible round-robin user."""
        async with self._lock:
            # Selection can reach past a queue's head; sweep all cancelled
            # Futures before choosing either workload class.
            for bucket, queued in list(self._waiters.items()):
                dq = deque(waiter for waiter in queued if not waiter.future.done())
                if not dq:
                    self._waiters.pop(bucket, None)
                else:
                    self._waiters[bucket] = dq

            if not self._waiters:
                self._available += 1
                return

            now = time.monotonic()
            # First eligible request per user preserves FIFO within each
            # workload and among promoted/interactive requests.
            eligible = {
                bucket: next(
                    (waiter for waiter in dq if self._is_interactive_priority(waiter, now)),
                    None,
                )
                for bucket, dq in self._waiters.items()
            }
            priority = "interactive"
            eligible = {bucket: waiter for bucket, waiter in eligible.items() if waiter is not None}
            if not eligible:
                priority = "background"
                eligible = {bucket: dq[0] for bucket, dq in self._waiters.items()}

            pick = next(
                (bucket for bucket in eligible if bucket != self._last_granted.get(priority)),
                next(iter(eligible)),
            )
            dq = self._waiters[pick]
            waiter = eligible[pick]
            assert waiter is not None
            dq.remove(waiter)
            if dq:
                self._waiters.move_to_end(pick)
            else:
                self._waiters.pop(pick, None)
            self._last_granted[priority] = pick
            waiter.future.set_result(None)

    def _is_interactive_priority(self, waiter: _Waiter, now: float) -> bool:
        return (
            waiter.workload == "interactive"
            or now - waiter.enqueued_at >= self._background_aging_seconds
        )

    def snapshot(self) -> dict[str, int]:
        """Best-effort queued counts per user; diagnostics only, never public labels."""
        return {bucket: len(dq) for bucket, dq in list(self._waiters.items())}

    def pressure_snapshot(self) -> dict[str, int]:
        """Aggregate gate pressure without exposing queue bucket identities."""
        waiters = list(self._waiters.values())
        live = [waiter for queue in waiters for waiter in queue if not waiter.future.done()]
        now = time.monotonic()
        return {
            "capacity": self._concurrency,
            "in_use": max(0, self._concurrency - self._available),
            "waiting": len(live),
            "waiting_users": sum(
                any(not waiter.future.done() for waiter in queue) for queue in waiters
            ),
            "waiting_interactive": sum(waiter.workload == "interactive" for waiter in live),
            "waiting_background": sum(waiter.workload == "background" for waiter in live),
            "aged_background": sum(
                waiter.workload == "background" and self._is_interactive_priority(waiter, now)
                for waiter in live
            ),
        }


__all__ = ["FairLocalGate", "ANON_USER_BUCKET"]
