"""Per-user fair local-GPU gate (Phase 20).

Replaces the global-FIFO `GpuGate`. Same purpose — serialize local Ollama
generations against the PSU/VRAM budget — but instead of strict FIFO, we
round-robin across users so U2's first request slips ahead of U1's
queries 2–10.

Design:

- One `concurrency` slot total (default 1, same as before).
- Per-user FIFO of waiters. When the slot frees, we pick the next user
  in round-robin order, then dequeue their head waiter.
- Cloud calls (location != "local") bypass entirely.
- Anonymous / no-user-id requests share a synthetic `__anon__` bucket.
  They round-robin against authed users like everyone else.

Round-robin pointer logic uses an `OrderedDict[str, deque[Future]]`:
- A user joining the queue is appended (move-to-end via `move_to_end` on
  re-entry isn't strictly needed; we pop the head user's first waiter
  and demote them to the tail when they still have remaining waiters).
- After each grant, the granted user is moved to the back of the order
  if they still have queued waiters; if their deque is empty, the user
  entry is removed entirely.

Cancellation safety: when a waiter's coroutine is cancelled (e.g. client
disconnected), their Future is marked cancelled. The release path checks
`fut.done()` before resolving and skips the dead waiter, popping the
next.

Same `acquire(model, *, location, user_id=None)` interface as `GpuGate`,
plus the `user_id` kwarg. Callers that pass nothing get bucketed as
`__anon__`.

Usage:
    gate = FairLocalGate(concurrency=cfg.gpu_concurrency)
    async with gate.acquire(model, location="local", user_id="bart"):
        await ollama.chat(...)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

from audrey.metrics import gpu_gate_wait_seconds

log = logging.getLogger(__name__)

ANON_USER_BUCKET = "__anon__"


class FairLocalGate:
    """Per-user fair gate. Same surface as `GpuGate` plus `user_id` kwarg."""

    def __init__(self, *, concurrency: int = 1) -> None:
        if concurrency < 1:
            concurrency = 1
        self._concurrency = concurrency
        self._available = concurrency
        # OrderedDict iteration order = round-robin order. The head of the
        # dict is "next user in line"; after a grant we pop that user's
        # head waiter and either remove the user (deque empty) or move
        # them to the tail.
        self._waiters: OrderedDict[str, deque[asyncio.Future[None]]] = OrderedDict()
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
        t0 = time.perf_counter()
        granted_directly = False

        # Fast path: a slot is free *and* nobody else is queued. Take it
        # without parking a Future.
        async with self._lock:
            if self._available > 0 and not self._waiters:
                self._available -= 1
                granted_directly = True

        if not granted_directly:
            # Slow path: park a waiter under our user bucket.
            fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            async with self._lock:
                dq = self._waiters.get(bucket)
                if dq is None:
                    dq = deque()
                    self._waiters[bucket] = dq
                dq.append(fut)
            try:
                await fut
            except asyncio.CancelledError:
                # Clean ourselves out of the queue if we were cancelled
                # before being granted. The grant loop also tolerates
                # already-done futures, so this is belt+suspenders.
                async with self._lock:
                    dq = self._waiters.get(bucket)
                    if dq is not None:
                        try:
                            dq.remove(fut)
                        except ValueError:
                            pass
                        if not dq:
                            self._waiters.pop(bucket, None)
                raise

        gpu_gate_wait_seconds.observe(time.perf_counter() - t0)
        try:
            yield
        finally:
            await self._release(bucket)

    async def _release(self, _released_bucket: str) -> None:
        """Free one slot and grant it to the next round-robin waiter."""
        async with self._lock:
            # Find the next non-empty user bucket. Pop dead/cancelled
            # futures along the way so they don't keep their user "first
            # in line" forever.
            granted = False
            while self._waiters and not granted:
                bucket, dq = next(iter(self._waiters.items()))
                while dq and dq[0].done():
                    dq.popleft()
                if not dq:
                    self._waiters.pop(bucket, None)
                    continue
                fut = dq.popleft()
                if dq:
                    # User has more queued — demote to tail of round-robin.
                    self._waiters.move_to_end(bucket)
                else:
                    self._waiters.pop(bucket, None)
                fut.set_result(None)
                granted = True
            if not granted:
                self._available += 1

    # ── Introspection (used by metrics endpoint, smoke tests) ─────────

    def snapshot(self) -> dict[str, int]:
        """Best-effort snapshot of current queue depth per user.

        Not synchronized — values may shift mid-iteration. Intended for
        debugging logs, not anything that requires consistency.
        """
        return {bucket: len(dq) for bucket, dq in self._waiters.items()}


__all__ = ["FairLocalGate", "ANON_USER_BUCKET"]
