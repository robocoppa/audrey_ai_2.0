"""Tests for FairLocalGate (Phase 20 + Phase 23 round-robin fix).

Asyncio fairness with no Ollama or network involved. Uses
`asyncio.Event` for cross-task ordering so the tests are deterministic
on any host — never `asyncio.sleep` for synchronization.

The Phase 23 starvation regression (`_release` always picked the
OrderedDict head, so a new bucket joining the queue was appended
behind the head and waited for that head's deque to fully drain) is
covered by `test_two_user_round_robin_skips_last_granted`.
"""

import asyncio

import pytest

from audrey.pipeline.fair_gate import ANON_USER_BUCKET, FairLocalGate


async def _hold_gate(
    gate: FairLocalGate,
    *,
    user_id: str,
    started: asyncio.Event,
    release: asyncio.Event,
    log_into: list[str],
    label: str,
    location: str = "local",
) -> None:
    """Acquire the gate, signal that we're inside, then wait for release.

    Logs `label` into `log_into` on entry — gives tests a way to assert
    on grant order without sleep-tuning.
    """
    async with gate.acquire("model-x", location=location, user_id=user_id):
        log_into.append(label)
        started.set()
        await release.wait()


# ─── Cloud bypass ──────────────────────────────────────────────────────

async def test_cloud_location_bypasses_gate_entirely():
    # Hold the only local slot under one task, then verify a cloud
    # acquire returns immediately rather than queueing.
    gate = FairLocalGate(concurrency=1)

    local_started = asyncio.Event()
    local_release = asyncio.Event()
    log: list[str] = []
    local_task = asyncio.create_task(_hold_gate(
        gate, user_id="alice",
        started=local_started, release=local_release,
        log_into=log, label="local",
    ))
    await local_started.wait()

    # The local slot is now occupied. A cloud acquire should still pass
    # through without waiting.
    async with gate.acquire("cloud-model", location="cloud", user_id="bob"):
        log.append("cloud")

    assert log == ["local", "cloud"]

    local_release.set()
    await local_task


# ─── Single-user serialization ─────────────────────────────────────────

async def test_single_user_three_acquires_serialize():
    # With concurrency=1, three back-to-back same-user acquires must
    # complete strictly in order. No two inside the gate at once.
    gate = FairLocalGate(concurrency=1)
    inside = 0
    max_concurrent = 0
    log: list[int] = []

    async def acquire_and_run(idx: int) -> None:
        nonlocal inside, max_concurrent
        async with gate.acquire("m", location="local", user_id="alice"):
            inside += 1
            max_concurrent = max(max_concurrent, inside)
            log.append(idx)
            # Yield to the loop so any incorrectly-granted second waiter
            # would get a chance to enter and bump `max_concurrent`.
            await asyncio.sleep(0)
            inside -= 1

    await asyncio.gather(
        acquire_and_run(1),
        acquire_and_run(2),
        acquire_and_run(3),
    )

    assert max_concurrent == 1
    assert sorted(log) == [1, 2, 3]


# ─── Round-robin (Phase 23 fix) ────────────────────────────────────────

async def test_two_user_round_robin_skips_last_granted():
    # Phase 23 regression case. Alice fires 3 acquires (a1, a2, a3) and
    # holds the slot. Bob arrives mid-flight with one acquire (b1).
    # Expected grant order on releases: a1, b1, a2, a3 — bob slips into
    # slot 2 because `_release` skips the just-granted alice bucket
    # when bob has waiters. Pre-fix order would have been a1, a2, a3, b1
    # (FIFO-by-bucket starvation).
    gate = FairLocalGate(concurrency=1)
    log: list[str] = []

    # Helper that records label, then immediately exits the gate. We
    # don't want any waits inside; the test is about pick order, not
    # hold-time.
    async def grab(label: str, user_id: str) -> None:
        async with gate.acquire("m", location="local", user_id=user_id):
            log.append(label)

    # Pre-load alice's three acquires before bob arrives. We need them
    # all *queued* before bob, so park them as Tasks with predictable
    # scheduling: the first will grant immediately, the next two will
    # park as waiters under bucket "alice".
    a1_started = asyncio.Event()
    a1_release = asyncio.Event()
    a1_task = asyncio.create_task(_hold_gate(
        gate, user_id="alice",
        started=a1_started, release=a1_release,
        log_into=log, label="a1",
    ))
    await a1_started.wait()  # a1 is inside the gate

    a2_task = asyncio.create_task(grab("a2", "alice"))
    a3_task = asyncio.create_task(grab("a3", "alice"))
    # Yield twice so a2 and a3 reach `await fut` and are queued.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Bob arrives now. Should be queued as a *separate* bucket.
    b1_task = asyncio.create_task(grab("b1", "bob"))
    await asyncio.sleep(0)

    # Drain by releasing a1; the gate then dispatches the rest itself.
    a1_release.set()
    await asyncio.gather(a1_task, a2_task, a3_task, b1_task)

    assert log[0] == "a1"
    # Phase 23 fix: bob slips into slot 2.
    assert log[1] == "b1", f"expected bob to be granted next; got order {log!r}"
    # Then alice's remaining waiters drain — bucket order doesn't matter
    # here, only that both are present.
    assert set(log[2:]) == {"a2", "a3"}


# ─── Cancellation ──────────────────────────────────────────────────────

async def test_cancelled_waiter_is_skipped_on_release():
    # If a waiter's coroutine is cancelled before being granted, the
    # release path must skip the dead future and grant the next live
    # one. Otherwise the slot leaks (granted to a task that already
    # bailed out).
    gate = FairLocalGate(concurrency=1)
    log: list[str] = []

    a1_started = asyncio.Event()
    a1_release = asyncio.Event()
    a1_task = asyncio.create_task(_hold_gate(
        gate, user_id="alice",
        started=a1_started, release=a1_release,
        log_into=log, label="a1",
    ))
    await a1_started.wait()

    async def grab(label: str, user_id: str) -> None:
        async with gate.acquire("m", location="local", user_id=user_id):
            log.append(label)

    a2_task = asyncio.create_task(grab("a2", "alice"))
    a3_task = asyncio.create_task(grab("a3", "alice"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Cancel a2 while it's parked. The release of a1 should skip a2
    # and grant a3 instead.
    a2_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a2_task

    a1_release.set()
    await asyncio.gather(a1_task, a3_task)

    assert log == ["a1", "a3"]


# ─── Anonymous bucket ──────────────────────────────────────────────────

async def test_no_user_id_funnels_into_anon_bucket():
    # Two acquires with no user_id should share one bucket — verifiable
    # via snapshot() while both are queued.
    gate = FairLocalGate(concurrency=1)

    held_started = asyncio.Event()
    held_release = asyncio.Event()
    log: list[str] = []
    held_task = asyncio.create_task(_hold_gate(
        gate, user_id="alice",
        started=held_started, release=held_release,
        log_into=log, label="held",
    ))
    await held_started.wait()

    async def grab(label: str) -> None:
        async with gate.acquire("m", location="local", user_id=None):
            log.append(label)

    anon1 = asyncio.create_task(grab("a1"))
    anon2 = asyncio.create_task(grab("a2"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    snap = gate.snapshot()
    assert snap.get(ANON_USER_BUCKET) == 2, f"expected 2 anon waiters, got {snap!r}"

    held_release.set()
    await asyncio.gather(held_task, anon1, anon2)


async def test_empty_string_user_id_funnels_into_anon_bucket():
    # Same as above but with `user_id=""` (programmatic clients that
    # forward an empty string instead of None).
    gate = FairLocalGate(concurrency=1)

    held_started = asyncio.Event()
    held_release = asyncio.Event()
    log: list[str] = []
    held_task = asyncio.create_task(_hold_gate(
        gate, user_id="alice",
        started=held_started, release=held_release,
        log_into=log, label="held",
    ))
    await held_started.wait()

    async def grab(label: str) -> None:
        async with gate.acquire("m", location="local", user_id="   "):
            log.append(label)

    t = asyncio.create_task(grab("anon"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert gate.snapshot().get(ANON_USER_BUCKET) == 1

    held_release.set()
    await asyncio.gather(held_task, t)


# ─── Concurrency knob ──────────────────────────────────────────────────

async def test_concurrency_below_one_clamped_to_one():
    # Defensive: the constructor should not allow zero or negative
    # concurrency (would deadlock the first acquire).
    gate = FairLocalGate(concurrency=0)
    assert gate.concurrency == 1
    gate = FairLocalGate(concurrency=-5)
    assert gate.concurrency == 1
