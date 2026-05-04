"""Per-user fair local-GPU gate.

Serializes local Ollama generations against the PSU/VRAM budget. Instead
of strict FIFO across all callers, this round-robins across users so a
new arrival's first request slips ahead of an existing user's backlog —
fairness rather than first-come-first-served.

Design:

- One `concurrency` slot total (default 1, same as before).
- Per-user FIFO of waiters. When the slot frees, we pick the next user
  in round-robin order, then dequeue their head waiter.
- Cloud calls (location != "local") bypass entirely.
- Anonymous / no-user-id requests share a synthetic `__anon__` bucket.
  They round-robin against authed users like everyone else.

Round-robin pointer logic uses an `OrderedDict[str, deque[Future]]` plus
a `_last_granted` bucket name:
- A user joining the queue is appended.
- On release, we walk the OrderedDict from the head looking for the next
  bucket to grant. We *skip* the most-recently-granted bucket if any
  other bucket has waiters — this is the "round-robin to a different
  user" behavior. If the last-granted bucket is the only one with
  waiters (or no other has waiters), we grant to it again.
- After each grant, the granted user is moved to the back of the order
  if they still have queued waiters; if their deque is empty, the user
  entry is removed entirely.

Why the explicit `_last_granted` is needed: with only OrderedDict
ordering, a single user with many queued waiters stays at the head
indefinitely (their deque never goes empty so we never `pop` them and
never reinsert them at the tail in a way that lets a new arrival
overtake). A new bucket arriving mid-queue gets appended to the tail
and would only be granted after the head user's queue drained — that's
FIFO-by-bucket, not round-robin. Tracking the last granted bucket and
skipping it on the next pick gives true alternation.

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
        # Bucket name of the most recently granted slot. Used in `_release`
        # to skip back-to-back grants to the same user when another user
        # has waiters queued.
        self._last_granted: str | None = None
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
                self._last_granted = bucket
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
        """Free one slot and grant it to the next round-robin waiter.

        Skips the bucket that just released (`self._last_granted`) when
        any other bucket has waiters, so the slot alternates across users
        instead of letting one user's backlog hog the head of the dict.
        Falls back to re-granting the last bucket only when it's the
        sole queued user.
        """
        async with self._lock:
            # First sweep: garbage-collect cancelled/dead futures and
            # empty buckets. Keeps the picker logic below clean.
            for bucket in list(self._waiters.keys()):
                dq = self._waiters[bucket]
                while dq and dq[0].done():
                    dq.popleft()
                if not dq:
                    self._waiters.pop(bucket, None)

            if not self._waiters:
                self._available += 1
                return

            # Pick: prefer any bucket *other* than the one we just granted.
            # Fall back to the last-granted bucket only if it's the only
            # one with waiters.
            pick: str | None = None
            for bucket in self._waiters:
                if bucket != self._last_granted:
                    pick = bucket
                    break
            if pick is None:
                # All remaining waiters are from the last-granted bucket.
                pick = next(iter(self._waiters))

            dq = self._waiters[pick]
            fut = dq.popleft()
            if dq:
                # User still has queued waiters — demote them so the next
                # release prefers a different bucket if one exists.
                self._waiters.move_to_end(pick)
            else:
                self._waiters.pop(pick, None)
            self._last_granted = pick
            fut.set_result(None)

    # ── Introspection (used by metrics endpoint, smoke tests) ─────────

    def snapshot(self) -> dict[str, int]:
        """Best-effort snapshot of current queue depth per user.

        Not synchronized — values may shift mid-iteration. Intended for
        debugging logs, not anything that requires consistency.
        """
        return {bucket: len(dq) for bucket, dq in self._waiters.items()}


__all__ = ["FairLocalGate", "ANON_USER_BUCKET"]
