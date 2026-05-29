"""Tests for UserInflightRegistry.

Covers the per-user concurrency cap, LRU eviction, the reservation
counter that pins buckets against eviction during the
sem.acquire() window, cancellation cleanup, and the soft-cap
breach counter.

Asyncio bookkeeping with no Ollama or network involved.
"""

import asyncio

import pytest

from audrey.metrics import inflight_cap_breached_total
from audrey.routes.inflight import ANON_USER_BUCKET, UserInflightRegistry

# ─── Per-user cap ──────────────────────────────────────────────────────

async def test_cap_serializes_per_user_requests():
    # max_inflight_per_user=1: the second concurrent request for alice
    # must wait until the first releases.
    reg = UserInflightRegistry(max_inflight_per_user=1)
    inside = 0
    max_concurrent = 0
    log: list[int] = []

    async def run(idx: int) -> None:
        nonlocal inside, max_concurrent
        async with reg.slot("alice"):
            inside += 1
            max_concurrent = max(max_concurrent, inside)
            log.append(idx)
            await asyncio.sleep(0)
            inside -= 1

    await asyncio.gather(run(1), run(2), run(3))
    assert max_concurrent == 1
    assert sorted(log) == [1, 2, 3]


async def test_cap_independent_across_users():
    # alice and bob each get their own semaphore. With cap=1 per user,
    # alice and bob can each hold one slot simultaneously.
    reg = UserInflightRegistry(max_inflight_per_user=1)

    a_started = asyncio.Event()
    a_release = asyncio.Event()
    b_started = asyncio.Event()
    b_release = asyncio.Event()

    async def hold(user: str, started: asyncio.Event, release: asyncio.Event) -> None:
        async with reg.slot(user):
            started.set()
            await release.wait()

    a_task = asyncio.create_task(hold("alice", a_started, a_release))
    b_task = asyncio.create_task(hold("bob", b_started, b_release))
    await a_started.wait()
    await b_started.wait()
    # Both inside at once — independent caps.
    assert reg.snapshot() == {"alice": 1, "bob": 1}

    a_release.set()
    b_release.set()
    await asyncio.gather(a_task, b_task)


# ─── Anon bucket ───────────────────────────────────────────────────────

async def test_none_user_id_funnels_to_anon_bucket():
    reg = UserInflightRegistry(max_inflight_per_user=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with reg.slot(None):
            started.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await started.wait()
    assert reg.snapshot().get(ANON_USER_BUCKET) == 1
    release.set()
    await task


async def test_blank_user_id_funnels_to_anon_bucket():
    reg = UserInflightRegistry(max_inflight_per_user=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with reg.slot("   "):
            started.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await started.wait()
    assert reg.snapshot().get(ANON_USER_BUCKET) == 1
    release.set()
    await task


# ─── LRU eviction ──────────────────────────────────────────────────────

async def test_idle_user_evicted_when_cap_reached():
    # max_tracked_users=2, one user finishes (idle), then a third arrives.
    # The idle user should be evicted to make room.
    reg = UserInflightRegistry(max_inflight_per_user=1, max_tracked_users=2)

    async def quick(user: str) -> None:
        async with reg.slot(user):
            pass

    await quick("alice")
    await quick("bob")
    # Registry now tracks alice and bob, both idle.
    # carol arrives — one of alice/bob is evicted.
    await quick("carol")
    tracked = set(reg._sems.keys())
    assert "carol" in tracked
    assert len(tracked) == 2


async def test_reserved_user_not_evicted_during_acquire_window():
    # Regression for the race where eviction inspected only `_inuse`
    # (incremented after sem.acquire()) and could evict a bucket whose
    # waiter was parked. The fix: a reservation counter incremented
    # under the registry lock before sem.acquire(), checked by eviction.
    #
    # Setup: cap=1, max_tracked=1. alice holds her slot. bob arrives;
    # eviction would normally evict alice (who is "in use"), but with
    # the reservation counter pinning her, bob's reserve must NOT evict
    # her — we accept the soft-cap breach instead.
    reg = UserInflightRegistry(max_inflight_per_user=1, max_tracked_users=1)

    a_started = asyncio.Event()
    a_release = asyncio.Event()

    async def hold_alice() -> None:
        async with reg.slot("alice"):
            a_started.set()
            await a_release.wait()

    async def grab_bob() -> None:
        async with reg.slot("bob"):
            pass

    a_task = asyncio.create_task(hold_alice())
    await a_started.wait()

    # Bob arrives — alice is in_use=1, reservation=1; she shouldn't
    # be evicted. The registry should accept the overflow.
    b_task = asyncio.create_task(grab_bob())
    # Let bob's reserve run.
    for _ in range(5):
        await asyncio.sleep(0)
    # Bob's semaphore exists; alice's is still tracked too.
    assert "alice" in reg._sems
    assert "bob" in reg._sems

    a_release.set()
    await asyncio.gather(a_task, b_task)


async def test_soft_cap_breach_increments_metric():
    # Same setup as the eviction test, but assert the breach counter
    # advances by 1 when the registry admits a user past its cap.
    reg = UserInflightRegistry(max_inflight_per_user=1, max_tracked_users=1)

    before = inflight_cap_breached_total._value.get()

    a_started = asyncio.Event()
    a_release = asyncio.Event()

    async def hold_alice() -> None:
        async with reg.slot("alice"):
            a_started.set()
            await a_release.wait()

    async def grab_bob() -> None:
        async with reg.slot("bob"):
            pass

    a_task = asyncio.create_task(hold_alice())
    await a_started.wait()
    b_task = asyncio.create_task(grab_bob())
    for _ in range(5):
        await asyncio.sleep(0)

    after = inflight_cap_breached_total._value.get()
    assert after == before + 1

    a_release.set()
    await asyncio.gather(a_task, b_task)


# ─── Cancellation cleanup ──────────────────────────────────────────────

async def test_cancelled_waiter_releases_reservation():
    # If a waiter is cancelled before acquiring the per-user semaphore,
    # its reservation must be dropped so the bucket can be evicted
    # later.
    reg = UserInflightRegistry(max_inflight_per_user=1, max_tracked_users=10)

    a_started = asyncio.Event()
    a_release = asyncio.Event()

    async def hold() -> None:
        async with reg.slot("alice"):
            a_started.set()
            await a_release.wait()

    async def park() -> None:
        async with reg.slot("alice"):
            pass

    a_task = asyncio.create_task(hold())
    await a_started.wait()
    # alice has 1 in_use + 1 reservation.

    parked = asyncio.create_task(park())
    # Give the parked task time to call _reserve and start awaiting
    # sem.acquire().
    for _ in range(5):
        await asyncio.sleep(0)
    assert reg._reservations["alice"] == 2

    # Cancel the parked waiter.
    parked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parked
    # Reservation should be back down to 1 (alice's holder).
    assert reg._reservations["alice"] == 1

    a_release.set()
    await a_task
    # After alice releases, both counters land at 0.
    assert reg._reservations["alice"] == 0
    assert reg._inuse["alice"] == 0


# ─── Constructor defenses ──────────────────────────────────────────────

async def test_max_inflight_below_one_clamped():
    reg = UserInflightRegistry(max_inflight_per_user=0)
    assert reg.max_per_user == 1
    reg = UserInflightRegistry(max_inflight_per_user=-3)
    assert reg.max_per_user == 1


# ─── Shared anon constant ──────────────────────────────────────────────

def test_anon_bucket_constant_is_shared_with_fair_gate():
    # Finding 5: both modules used to define `ANON_USER_BUCKET` separately.
    # They now import from `audrey.scheduling`, so the constants must
    # refer to the same string object.
    from audrey.pipeline.fair_gate import ANON_USER_BUCKET as A
    from audrey.routes.inflight import ANON_USER_BUCKET as B
    from audrey.scheduling import ANON_USER_BUCKET as C

    assert A is B is C
