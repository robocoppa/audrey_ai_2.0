"""Per-user in-flight cap.

Bounds the number of concurrent pipeline requests one user can have
running at once. When a user is at the cap, request N+1 *waits* — no
429, no error. The user's request "takes longer to start" but
eventually runs.

Why this exists: layer 1 (`FairLocalGate`) gives round-robin fairness
at the local-GPU level, but a user with a runaway client could still
push hundreds of requests into the scheduler and bloat memory. This
layer protects the scheduler itself.

Per-user priority semaphores are lazy-created on first use. Interactive
requests precede queued background model calls, while background work
joins the interactive FIFO after an aging interval to prevent starvation.
Running requests are never preempted. The
registry caps total tracked users (`max_tracked_users`) and evicts
LRU when it grows past the cap. Idle users (zero pending or held
slots) get evicted first; users with any pending or held slot stay.

Single-process: this is in-memory state on the uvicorn event loop.
Process restart drops everything (in-flight requests die anyway).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from audrey.metrics import (
    inflight_cap_breached_total,
    local_admission_wait_seconds,
    user_inflight_blocked_seconds,
)
from audrey.scheduling import ANON_USER_BUCKET, current_local_workload

log = logging.getLogger(__name__)


class BackgroundPermit:
    """Cooperatively yield admission before a background generation starts.

    The adapter races ``give_way`` against GPU admission, then calls
    ``mark_running`` synchronously before dispatch. A successful transition
    protects an actual generation from later interactive arrivals.
    """

    def __init__(self, *, enqueued_at: float):
        self.give_way = asyncio.Event()
        self.enqueued_at = enqueued_at
        self._running = False

    def mark_running(self) -> bool:
        if self.give_way.is_set():
            return False
        self._running = True
        return True


@dataclass
class _Waiter:
    future: asyncio.Future[None]
    workload: Literal["interactive", "background"]
    enqueued_at: float


class _PrioritySemaphore:
    """Event-loop-local semaphore with priority only at slot transfer."""

    def __init__(self, value: int, *, aging_seconds: float, clock: Callable[[], float]):
        self._value = value
        self._aging_seconds = aging_seconds
        self._clock = clock
        self._waiters: list[_Waiter] = []
        self._permits: list[BackgroundPermit] = []

    async def acquire(
        self, workload: Literal["interactive", "background"] = "interactive",
        *, queued_at: float | None = None,
    ) -> bool:
        # Wake older queued callers before a new arrival can use a free slot.
        self._wake_next()
        if self._value > 0:
            self._value -= 1
            return True

        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiter = _Waiter(future, workload, self._clock() if queued_at is None else queued_at)
        self._waiters.append(waiter)
        if workload == "interactive":
            self._notify_background_permits()
        try:
            await future
            await self._after_waiter_granted()
        except BaseException:
            # Cancellation of a queued await cancels its Future. A completed,
            # non-cancelled Future means release already transferred a slot,
            # even if the caller has not resumed or entered its context yet.
            granted = future.done() and not future.cancelled()
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            if granted:
                self.release()
            raise
        return True

    async def _after_waiter_granted(self) -> None:
        """Non-suspending production seam for cancellation-boundary tests."""

    def release(self) -> None:
        self._value += 1
        self._wake_next()

    def register_permit(self, permit: BackgroundPermit) -> None:
        self._permits.append(permit)
        # A newly granted background waiter can resume after an interactive
        # arrival queued. Registration must honor that arrival too.
        self._notify_background_permits()

    def unregister_permit(self, permit: BackgroundPermit) -> None:
        self._permits.remove(permit)

    def _notify_background_permits(self) -> None:
        if not any(
            waiter.workload == "interactive" and not waiter.future.done()
            for waiter in self._waiters
        ):
            return
        now = self._clock()
        for permit in self._permits:
            if not permit._running and now - permit.enqueued_at < self._aging_seconds:
                permit.give_way.set()

    def _wake_next(self) -> None:
        # Cancelled futures can still be queued until their coroutine resumes.
        self._waiters[:] = [waiter for waiter in self._waiters if not waiter.future.done()]
        while self._value > 0 and self._waiters:
            now = self._clock()
            chosen = next(
                (
                    waiter for waiter in self._waiters
                    if waiter.workload == "interactive"
                    or now - waiter.enqueued_at >= self._aging_seconds
                ),
                self._waiters[0],
            )
            self._waiters.remove(chosen)
            self._value -= 1
            chosen.future.set_result(None)


class UserInflightRegistry:
    """LRU map of per-user priority semaphores with bounded size."""

    def __init__(
        self, *, max_inflight_per_user: int, max_tracked_users: int = 1024,
        background_aging_seconds: float = 30,
    ) -> None:
        if max_inflight_per_user < 1:
            max_inflight_per_user = 1
        if not math.isfinite(background_aging_seconds) or background_aging_seconds < 0:
            raise ValueError("background_aging_seconds must be finite and nonnegative")
        self._max_per_user = max_inflight_per_user
        self._max_tracked = max_tracked_users
        self._background_aging_seconds = background_aging_seconds
        self._clock = time.monotonic
        # Insertion order = LRU order (most recently touched at the tail).
        self._sems: OrderedDict[str, _PrioritySemaphore] = OrderedDict()
        # Reservations = waiters parked on sem.acquire() + holders inside
        # the slot. Counted from inside the registry lock so eviction can
        # tell whether a bucket is truly idle. `_inuse` (holders only) is
        # kept for `snapshot()`.
        self._reservations: dict[str, int] = {}
        self._inuse: dict[str, int] = {}
        self._lock = asyncio.Lock()

    @property
    def max_per_user(self) -> int:
        return self._max_per_user

    async def _reserve(self, bucket: str) -> _PrioritySemaphore:
        """Reserve a slot for `bucket`: return its semaphore, increment its
        reservation count, and evict idle users to honor the soft cap.

        Callers MUST `await sem.acquire()` after this returns. The
        reservation counter pins the bucket against eviction during the
        window before the semaphore acquire resolves.
        """
        async with self._lock:
            sem = self._sems.get(bucket)
            if sem is None:
                # Evict LRU users with no reservations (truly idle) until
                # we're under the cap. Reserved users (pending or held)
                # skip eviction.
                while len(self._sems) >= self._max_tracked:
                    evicted = False
                    for k in list(self._sems.keys()):
                        if self._reservations.get(k, 0) == 0:
                            self._sems.pop(k, None)
                            self._reservations.pop(k, None)
                            self._inuse.pop(k, None)
                            evicted = True
                            break
                    if not evicted:
                        # Every tracked user has a reservation — accept
                        # the overflow rather than block. The cap is a
                        # soft guideline, not a hard ceiling.
                        inflight_cap_breached_total.inc()
                        log.warning(
                            "inflight: tracking %d active users, exceeding cap %d",
                            len(self._sems), self._max_tracked,
                        )
                        break
                sem = _PrioritySemaphore(
                    self._max_per_user, aging_seconds=self._background_aging_seconds,
                    clock=self._clock,
                )
                self._sems[bucket] = sem
                self._reservations[bucket] = 0
                self._inuse[bucket] = 0
            else:
                self._sems.move_to_end(bucket)
            self._reservations[bucket] = self._reservations.get(bucket, 0) + 1
            return sem

    @asynccontextmanager
    async def slot(
        self, user_id: str | None, *, yield_to_interactive: bool = False,
        queued_at: float | None = None,
    ):
        """Hold one in-flight slot for `user_id` for the duration of the block.

        Background adapters may opt into a cooperative permit while awaiting
        the GPU. ``queued_at`` preserves their original monotonic enqueue time
        across retries; normal callers retain the existing None yield value.
        Both admission histograms also observe immediate admissions.
        """
        bucket = (user_id or "").strip() or ANON_USER_BUCKET
        workload = current_local_workload()
        enqueued_at = self._clock() if queued_at is None else queued_at
        sem = await self._reserve(bucket)
        t0 = time.perf_counter()
        acquired = False
        counted = False
        permit: BackgroundPermit | None = None
        try:
            await sem.acquire(workload, queued_at=enqueued_at)
            acquired = True
            wait_s = time.perf_counter() - t0
            user_inflight_blocked_seconds.observe(wait_s)
            local_admission_wait_seconds.labels(stage="inflight", workload=workload).observe(wait_s)
            if wait_s > 1.0:
                log.info("inflight: user=%s waited %.2fs for slot", _safe_bucket(bucket), wait_s)
            # No suspension between a successful acquire and bookkeeping, or
            # during cleanup. All registry mutation runs on one event loop;
            # reservation creation alone needs its lock for the LRU operation.
            self._inuse[bucket] = self._inuse.get(bucket, 0) + 1
            counted = True
            if yield_to_interactive and workload == "background":
                permit = BackgroundPermit(enqueued_at=enqueued_at)
                sem.register_permit(permit)
            yield permit
        finally:
            if permit is not None:
                sem.unregister_permit(permit)
            if acquired:
                sem.release()
            if counted:
                self._inuse[bucket] = max(0, self._inuse[bucket] - 1)
            self._reservations[bucket] = max(0, self._reservations[bucket] - 1)

    def snapshot(self) -> dict[str, int]:
        """Best-effort: per-user in-flight count. For debug logs only.

        Materializes via `list()` first so concurrent mutation can't
        raise "dictionary changed size during iteration".
        """
        return dict(list(self._inuse.items()))

    def pressure_snapshot(self) -> dict[str, int]:
        """Aggregate in-flight pressure without returning user buckets."""
        reservations = dict(list(self._reservations.items()))
        in_use = dict(list(self._inuse.items()))
        return {
            "max_per_user": self._max_per_user,
            "tracked_users": len(self._sems),
            "active_users": sum(value > 0 for value in reservations.values()),
            "saturated_users": sum(
                in_use.get(bucket, 0) >= self._max_per_user
                for bucket in reservations
            ),
            "in_use": sum(in_use.values()),
            "waiting": sum(
                max(0, value - in_use.get(bucket, 0))
                for bucket, value in reservations.items()
            ),
        }


def _safe_bucket(bucket: str) -> str:
    """Don't dump full email addresses to logs — log a short prefix."""
    if "@" in bucket:
        local, _, _ = bucket.partition("@")
        return f"{local[:8]}…"
    return bucket[:16]


__all__ = ["BackgroundPermit", "UserInflightRegistry", "ANON_USER_BUCKET"]
