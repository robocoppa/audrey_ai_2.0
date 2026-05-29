"""Per-user in-flight cap.

Bounds the number of concurrent pipeline requests one user can have
running at once. When a user is at the cap, request N+1 *waits* — no
429, no error. The user's request "takes longer to start" but
eventually runs.

Why this exists: layer 1 (`FairLocalGate`) gives round-robin fairness
at the local-GPU level, but a user with a runaway client could still
push hundreds of requests into the scheduler and bloat memory. This
layer protects the scheduler itself.

Per-user `asyncio.Semaphore` is lazy-created on first use. The
registry caps total tracked users (`max_tracked_users`) and evicts
LRU when it grows past the cap. Idle users (zero pending or held
slots) get evicted first; users with any pending or held slot stay.

Single-process: this is in-memory state on the uvicorn event loop.
Process restart drops everything (in-flight requests die anyway).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from audrey.metrics import inflight_cap_breached_total, user_inflight_blocked_seconds
from audrey.scheduling import ANON_USER_BUCKET

log = logging.getLogger(__name__)


class UserInflightRegistry:
    """LRU map of `user_id -> Semaphore` with bounded size."""

    def __init__(self, *, max_inflight_per_user: int, max_tracked_users: int = 1024) -> None:
        if max_inflight_per_user < 1:
            max_inflight_per_user = 1
        self._max_per_user = max_inflight_per_user
        self._max_tracked = max_tracked_users
        # Insertion order = LRU order (most recently touched at the tail).
        self._sems: OrderedDict[str, asyncio.Semaphore] = OrderedDict()
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

    async def _reserve(self, bucket: str) -> asyncio.Semaphore:
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
                sem = asyncio.Semaphore(self._max_per_user)
                self._sems[bucket] = sem
                self._reservations[bucket] = 0
                self._inuse[bucket] = 0
            else:
                self._sems.move_to_end(bucket)
            self._reservations[bucket] = self._reservations.get(bucket, 0) + 1
            return sem

    async def _drop_reservation(self, bucket: str) -> None:
        async with self._lock:
            if bucket in self._reservations:
                self._reservations[bucket] = max(0, self._reservations[bucket] - 1)

    @asynccontextmanager
    async def slot(self, user_id: str | None):
        """Hold one in-flight slot for `user_id` for the duration of the block.

        Times the wait via `audrey_user_inflight_blocked_seconds`. The
        observation only lands if we actually waited — fast path
        (slot immediately available) records a 0-bucket sample.
        """
        bucket = (user_id or "").strip() or ANON_USER_BUCKET
        sem = await self._reserve(bucket)
        t0 = time.perf_counter()
        try:
            await sem.acquire()
        except BaseException:
            # Cancelled or otherwise abandoned before acquiring — release
            # the reservation so eviction can reclaim the bucket.
            await self._drop_reservation(bucket)
            raise
        wait_s = time.perf_counter() - t0
        user_inflight_blocked_seconds.observe(wait_s)
        if wait_s > 1.0:
            log.info("inflight: user=%s waited %.2fs for slot", _safe_bucket(bucket), wait_s)
        async with self._lock:
            self._inuse[bucket] = self._inuse.get(bucket, 0) + 1
        try:
            yield
        finally:
            sem.release()
            async with self._lock:
                if bucket in self._inuse:
                    self._inuse[bucket] = max(0, self._inuse[bucket] - 1)
                if bucket in self._reservations:
                    self._reservations[bucket] = max(0, self._reservations[bucket] - 1)

    def snapshot(self) -> dict[str, int]:
        """Best-effort: per-user in-flight count. For debug logs only.

        Materializes via `list()` first so concurrent mutation can't
        raise "dictionary changed size during iteration".
        """
        return dict(list(self._inuse.items()))


def _safe_bucket(bucket: str) -> str:
    """Don't dump full email addresses to logs — log a short prefix."""
    if "@" in bucket:
        local, _, _ = bucket.partition("@")
        return f"{local[:8]}…"
    return bucket[:16]


__all__ = ["UserInflightRegistry", "ANON_USER_BUCKET"]
