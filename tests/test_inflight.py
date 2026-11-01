"""Tests for UserInflightRegistry.

Covers the per-user concurrency cap, LRU eviction, the reservation
counter that pins buckets against eviction during the
sem.acquire() window, cancellation cleanup, and the soft-cap
breach counter.

Asyncio bookkeeping with no Ollama or network involved.
"""

import asyncio

import pytest

from audrey.metrics import (
    inflight_cap_breached_total,
    local_admission_wait_seconds,
    user_inflight_blocked_seconds,
)
from audrey.routes.inflight import ANON_USER_BUCKET, UserInflightRegistry
from audrey.scheduling import local_workload

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


async def _wait_for_reservations(reg, count):
    for _ in range(20):
        if sum(reg._reservations.values()) == count:
            return
        await asyncio.sleep(0)
    assert sum(reg._reservations.values()) == count


async def _record_work(reg, name, workload, order):
    with local_workload(workload):
        async with reg.slot("alice"):
            order.append(name)


async def test_interactive_precedes_queued_background_without_preempting_holder():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    with local_workload("background"):
        holder = reg.slot("alice")
        await holder.__aenter__()
    background = asyncio.create_task(_record_work(reg, "background", "background", order))
    await _wait_for_reservations(reg, 2)
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
    await _wait_for_reservations(reg, 3)

    # Enqueuing interactive work never interrupts an active background call.
    assert order == []
    assert reg.snapshot() == {"alice": 1}
    assert reg.pressure_snapshot()["waiting"] == 2
    await holder.__aexit__(None, None, None)
    await asyncio.gather(background, interactive)
    assert order == ["interactive", "background"]
    assert reg._sems["alice"]._value == 1
    assert reg._reservations["alice"] == 0


async def test_background_waiters_stay_fifo_when_no_interactive_work_waits():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    async with reg.slot("alice"):
        first = asyncio.create_task(_record_work(reg, "first", "background", order))
        await _wait_for_reservations(reg, 2)
        second = asyncio.create_task(_record_work(reg, "second", "background", order))
        await _wait_for_reservations(reg, 3)
    await asyncio.gather(first, second)
    assert order == ["first", "second"]


async def test_aged_background_joins_interactive_fifo_with_fake_clock(monkeypatch):
    reg = UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=30)
    now = 0.0
    monkeypatch.setattr(reg, "_clock", lambda: now)
    order = []
    async with reg.slot("alice"):
        older_interactive = asyncio.create_task(
            _record_work(reg, "older interactive", "interactive", order),
        )
        await _wait_for_reservations(reg, 2)
        now = 5.0
        aged_background = asyncio.create_task(
            _record_work(reg, "aged background", "background", order),
        )
        await _wait_for_reservations(reg, 3)
        now = 10.0
        newer_interactive = asyncio.create_task(
            _record_work(reg, "newer interactive", "interactive", order),
        )
        await _wait_for_reservations(reg, 4)
        now = 35.0  # No wall-clock wait: exactly the aging interval has elapsed.
    await asyncio.gather(older_interactive, aged_background, newer_interactive)
    assert order == ["older interactive", "aged background", "newer interactive"]


async def test_unaged_background_remains_behind_interactive_with_fake_clock(monkeypatch):
    reg = UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=30)
    now = 0.0
    monkeypatch.setattr(reg, "_clock", lambda: now)
    order = []
    async with reg.slot("alice"):
        background = asyncio.create_task(_record_work(reg, "background", "background", order))
        await _wait_for_reservations(reg, 2)
        interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
        await _wait_for_reservations(reg, 3)
        now = 29.9
    await asyncio.gather(background, interactive)
    assert order == ["interactive", "background"]


async def test_zero_aging_preserves_fifo_for_all_workloads():
    reg = UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=0)
    order = []
    async with reg.slot("alice"):
        background = asyncio.create_task(_record_work(reg, "background", "background", order))
        await _wait_for_reservations(reg, 2)
        interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
        await _wait_for_reservations(reg, 3)
    await asyncio.gather(background, interactive)
    assert order == ["background", "interactive"]


async def test_admission_wait_observes_workload_and_preserves_legacy_histogram():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    before = {
        sample.name: sample.value
        for metric in user_inflight_blocked_seconds.collect()
        for sample in metric.samples if sample.name.endswith("_count")
    }
    workload_histogram = local_admission_wait_seconds.labels(stage="inflight", workload="background")
    workload_before = next(
        sample.value for metric in workload_histogram.collect()
        for sample in metric.samples if sample.name.endswith("_count")
    )
    with local_workload("background"):
        async with reg.slot("alice"):
            pass
    after = {
        sample.name: sample.value
        for metric in user_inflight_blocked_seconds.collect()
        for sample in metric.samples if sample.name.endswith("_count")
    }
    assert all(after[name] == count + 1 for name, count in before.items())
    workload_after = next(
        sample.value for metric in workload_histogram.collect()
        for sample in metric.samples if sample.name.endswith("_count")
    )
    assert workload_after == workload_before + 1


async def test_pre_gpu_background_permit_gives_way_to_new_interactive_waiter():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    with local_workload("background"):
        holder = reg.slot("alice", yield_to_interactive=True)
        permit = await holder.__aenter__()
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
    await _wait_for_reservations(reg, 2)
    assert permit.give_way.is_set()
    assert not permit.mark_running()
    await holder.__aexit__(None, None, None)
    await interactive
    assert order == ["interactive"]
    assert reg._sems["alice"]._permits == []
    assert reg._sems["alice"]._value == 1


async def test_running_background_permit_never_gives_way():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    with local_workload("background"):
        holder = reg.slot("alice", yield_to_interactive=True)
        permit = await holder.__aenter__()
    assert permit.mark_running()
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
    await _wait_for_reservations(reg, 2)
    assert not permit.give_way.is_set()
    assert order == []
    await holder.__aexit__(None, None, None)
    await interactive
    assert reg._sems["alice"]._permits == []


@pytest.mark.parametrize("age,should_yield", [(29.9, True), (30.0, False), (100.0, False)])
async def test_background_permit_aging_boundary(monkeypatch, age, should_yield):
    reg = UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=30)
    now = 10.0
    monkeypatch.setattr(reg, "_clock", lambda: now)
    with local_workload("background"):
        holder = reg.slot("alice", yield_to_interactive=True, queued_at=10.0)
        permit = await holder.__aenter__()
    now += age
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", []))
    await _wait_for_reservations(reg, 2)
    assert permit.give_way.is_set() is should_yield
    assert permit.mark_running() is not should_yield
    await holder.__aexit__(None, None, None)
    await interactive


async def test_background_permit_registration_honors_already_queued_interactive(monkeypatch):
    reg = UserInflightRegistry(max_inflight_per_user=2)
    first_holder = reg.slot("alice")
    second_holder = reg.slot("alice")
    await first_holder.__aenter__()
    await second_holder.__aenter__()
    granted = asyncio.Event()
    resume = asyncio.Event()
    registered = asyncio.Event()
    release = asyncio.Event()
    permits = []

    async def pause_once():
        if not granted.is_set():
            granted.set()
            await resume.wait()

    monkeypatch.setattr(reg._sems["alice"], "_after_waiter_granted", pause_once)

    async def background():
        with local_workload("background"):
            async with reg.slot("alice", yield_to_interactive=True) as permit:
                permits.append(permit)
                registered.set()
                await release.wait()

    background_task = asyncio.create_task(background())
    await _wait_for_reservations(reg, 3)
    await first_holder.__aexit__(None, None, None)
    await granted.wait()
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", []))
    await _wait_for_reservations(reg, 3)
    resume.set()
    await registered.wait()
    assert permits[0].give_way.is_set()
    assert not permits[0].mark_running()
    release.set()
    await second_holder.__aexit__(None, None, None)
    await asyncio.gather(background_task, interactive)
    assert reg._sems["alice"]._permits == []
    assert reg._sems["alice"]._value == 2


async def test_retried_background_permit_keeps_original_age(monkeypatch):
    reg = UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=30)
    now = 0.0
    monkeypatch.setattr(reg, "_clock", lambda: now)
    order = []
    with local_workload("background"):
        holder = reg.slot("alice", yield_to_interactive=True, queued_at=0.0)
        first_permit = await holder.__aenter__()
    started = asyncio.Event()
    release = asyncio.Event()

    async def interactive_holder():
        async with reg.slot("alice"):
            started.set()
            await release.wait()

    now = 5.0
    interactive = asyncio.create_task(interactive_holder())
    await _wait_for_reservations(reg, 2)
    assert first_permit.give_way.is_set()
    await holder.__aexit__(None, None, None)
    await started.wait()
    now = 30.0

    async def background_retry():
        with local_workload("background"):
            async with reg.slot("alice", yield_to_interactive=True, queued_at=0.0) as permit:
                assert not permit.give_way.is_set()
                assert permit.mark_running()
                order.append("background retry")

    retry = asyncio.create_task(background_retry())
    await _wait_for_reservations(reg, 2)
    newer_interactive = asyncio.create_task(_record_work(reg, "newer interactive", "interactive", order))
    await _wait_for_reservations(reg, 3)
    release.set()
    await asyncio.gather(interactive, retry, newer_interactive)
    assert order == ["background retry", "newer interactive"]
    assert reg._sems["alice"]._permits == []
    assert reg._reservations["alice"] == 0


async def test_cancelled_background_permit_unregisters_and_restores_capacity():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    started = asyncio.Event()
    never_release = asyncio.Event()
    permits = []

    async def background():
        with local_workload("background"):
            async with reg.slot("alice", yield_to_interactive=True) as permit:
                permits.append(permit)
                started.set()
                await never_release.wait()

    task = asyncio.create_task(background())
    await started.wait()
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", []))
    await _wait_for_reservations(reg, 2)
    assert permits[0].give_way.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await interactive
    assert reg._sems["alice"]._permits == []
    assert reg._sems["alice"]._value == 1
    assert reg._reservations["alice"] == 0
    assert reg._inuse["alice"] == 0


async def test_normal_slots_keep_none_yield_value():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    async with reg.slot("alice", yield_to_interactive=True) as interactive_value:
        assert interactive_value is None
    with local_workload("background"):
        async with reg.slot("alice") as background_value:
            assert background_value is None


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


async def test_cancelled_future_is_skipped_before_waiter_cleanup_runs():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    holder = reg.slot("alice")
    await holder.__aenter__()
    cancelled = asyncio.create_task(_record_work(reg, "cancelled", "interactive", order))
    await _wait_for_reservations(reg, 2)
    survivor = asyncio.create_task(_record_work(reg, "survivor", "background", order))
    await _wait_for_reservations(reg, 3)
    cancelled.cancel()
    # Release runs synchronously before the cancellation handler can remove its
    # dead Future. The survivor must get the slot without InvalidStateError.
    await holder.__aexit__(None, None, None)
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await survivor
    assert order == ["survivor"]
    assert reg._sems["alice"]._value == 1
    assert reg._reservations["alice"] == 0


@pytest.mark.parametrize("pause_after_grant", [False, True])
async def test_cancelled_just_granted_waiter_returns_slot(monkeypatch, pause_after_grant):
    reg = UserInflightRegistry(max_inflight_per_user=1)
    order = []
    holder = reg.slot("alice")
    await holder.__aenter__()
    sem = reg._sems["alice"]
    granted = asyncio.Event()
    never_release = asyncio.Event()

    async def pause_once():
        if not granted.is_set():
            granted.set()
            await never_release.wait()

    if pause_after_grant:
        monkeypatch.setattr(sem, "_after_waiter_granted", pause_once)
    cancelled = asyncio.create_task(_record_work(reg, "cancelled", "interactive", order))
    await _wait_for_reservations(reg, 2)
    survivor = asyncio.create_task(_record_work(reg, "survivor", "background", order))
    await _wait_for_reservations(reg, 3)
    await holder.__aexit__(None, None, None)
    if pause_after_grant:
        await granted.wait()
    # False: Future resolved, task has never resumed. True: acquire resumed,
    # but its context body has not started. Both own a transferred slot.
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await survivor
    assert order == ["survivor"]
    assert reg._sems["alice"]._value == 1
    assert reg._sems["alice"]._waiters == []
    assert reg.snapshot() == {"alice": 0}
    assert reg.pressure_snapshot()["waiting"] == 0
    assert reg._reservations["alice"] == 0


async def test_holder_cancellation_releases_capacity_and_reservation():
    reg = UserInflightRegistry(max_inflight_per_user=1)
    started = asyncio.Event()
    never_release = asyncio.Event()
    order = []

    async def hold():
        with local_workload("background"):
            async with reg.slot("alice"):
                started.set()
                await never_release.wait()

    holder = asyncio.create_task(hold())
    await started.wait()
    interactive = asyncio.create_task(_record_work(reg, "interactive", "interactive", order))
    await _wait_for_reservations(reg, 2)
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder
    await interactive
    assert order == ["interactive"]
    assert reg._reservations["alice"] == 0
    assert reg._inuse["alice"] == 0
    assert reg._sems["alice"]._value == 1


async def test_priority_does_not_expand_per_user_capacity():
    reg = UserInflightRegistry(max_inflight_per_user=2)
    inside = 0
    maximum = 0
    release = asyncio.Event()

    async def hold(workload):
        nonlocal inside, maximum
        with local_workload(workload):
            async with reg.slot("alice"):
                inside += 1
                maximum = max(inside, maximum)
                await release.wait()
                inside -= 1

    tasks = [asyncio.create_task(hold(workload)) for workload in (
        "background", "background", "background", "interactive", "interactive",
    )]
    await _wait_for_reservations(reg, 5)
    assert reg.snapshot() == {"alice": 2}
    assert reg.pressure_snapshot()["waiting"] == 3
    release.set()
    await asyncio.gather(*tasks)
    assert maximum == 2
    assert reg._sems["alice"]._value == 2


# ─── Constructor defenses ──────────────────────────────────────────────

async def test_max_inflight_below_one_clamped():
    reg = UserInflightRegistry(max_inflight_per_user=0)
    assert reg.max_per_user == 1
    reg = UserInflightRegistry(max_inflight_per_user=-3)
    assert reg.max_per_user == 1


@pytest.mark.parametrize("aging_seconds", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_background_aging_is_rejected(aging_seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        UserInflightRegistry(max_inflight_per_user=1, background_aging_seconds=aging_seconds)


# ─── Shared anon constant ──────────────────────────────────────────────

def test_anon_bucket_constant_is_shared_with_fair_gate():
    # Finding 5: both modules used to define `ANON_USER_BUCKET` separately.
    # They now import from `audrey.scheduling`, so the constants must
    # refer to the same string object.
    from audrey.pipeline.fair_gate import ANON_USER_BUCKET as A
    from audrey.routes.inflight import ANON_USER_BUCKET as B
    from audrey.scheduling import ANON_USER_BUCKET as C

    assert A is B is C
