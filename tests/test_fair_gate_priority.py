"""Foreground admission and aging use controlled task events and a fake clock."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from audrey.metrics import gpu_gate_wait_seconds, local_admission_wait_seconds
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.scheduling import local_workload


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "audrey.pipeline.fair_gate.time",
        SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter),
    )
    return now


async def _enqueue(
    gate, log, label, *, user="alice", workload="interactive", entered=None, release=None,
):
    queued = asyncio.Event()

    async def run():
        with local_workload(workload):
            queued.set()
            async with gate.acquire("local-model", location="local", user_id=user):
                log.append(label)
                if entered:
                    entered.set()
                if release:
                    await release.wait()

    task = asyncio.create_task(run())
    # Setting the event does not suspend run(): it continues until it is
    # parked inside the gate before this caller resumes.
    await queued.wait()
    return task


@pytest.mark.parametrize("foreground_owner", ["alice", "bob"])
async def test_interactive_passes_unaged_background_even_for_the_same_owner(clock, foreground_owner):
    gate = FairLocalGate()
    log = []
    async with gate.acquire("active", location="local", user_id="alice"):
        background = await _enqueue(gate, log, "agent", workload="background")
        foreground = await _enqueue(gate, log, "chat", user=foreground_owner)
        assert gate.pressure_snapshot() == {
            "capacity": 1, "in_use": 1, "waiting": 2, "waiting_users": len({"alice", foreground_owner}),
            "waiting_interactive": 1, "waiting_background": 1, "aged_background": 0,
        }
        assert log == []
    await asyncio.gather(background, foreground)
    assert log == ["chat", "agent"]
    assert gate._available == 1


async def test_round_robin_foreground_users_and_fifo_within_workload(clock):
    gate = FairLocalGate()
    log = []
    async with gate.acquire("active", location="local", user_id="alice"):
        tasks = [await _enqueue(gate, log, "agent", workload="background")]
        for user, label in [("alice", "a1"), ("alice", "a2"), ("bob", "b1"), ("bob", "b2")]:
            tasks.append(await _enqueue(gate, log, label, user=user))
    await asyncio.gather(*tasks)
    assert log == ["b1", "a1", "b2", "a2", "agent"]


async def test_round_robin_background_users_when_no_interactive_waits(clock):
    gate = FairLocalGate()
    log = []
    with local_workload("background"):
        async with gate.acquire("active", location="local", user_id="alice"):
            tasks = []
            for user, label in [("alice", "a1"), ("alice", "a2"), ("bob", "b1"), ("bob", "b2")]:
                tasks.append(await _enqueue(gate, log, label, user=user, workload="background"))
    await asyncio.gather(*tasks)
    assert log == ["b1", "a1", "b2", "a2"]


async def test_aging_prevents_starvation_under_same_owner_chat_load(clock):
    gate = FairLocalGate(background_aging_seconds=30)
    log = []
    chat1_entered, chat2_entered = asyncio.Event(), asyncio.Event()
    chat1_release, chat2_release = asyncio.Event(), asyncio.Event()
    async with gate.acquire("active", location="local", user_id="alice"):
        agent = await _enqueue(gate, log, "agent", workload="background")
        chat1 = await _enqueue(gate, log, "chat1", entered=chat1_entered, release=chat1_release)
        chat2 = await _enqueue(gate, log, "chat2", entered=chat2_entered, release=chat2_release)
        chat3 = await _enqueue(gate, log, "chat3")
    await chat1_entered.wait()
    assert log == ["chat1"]
    clock[0] = 29.99
    chat1_release.set()
    await chat2_entered.wait()
    assert log == ["chat1", "chat2"]
    clock[0] = 30
    assert gate.pressure_snapshot()["aged_background"] == 1
    # Advancing time never interrupts chat2: promotion matters only at
    # the next generation boundary.
    assert log == ["chat1", "chat2"]
    chat2_release.set()
    await asyncio.gather(agent, chat1, chat2, chat3)
    assert log == ["chat1", "chat2", "agent", "chat3"]


async def test_aged_background_shares_user_round_robin_with_interactive(clock):
    gate = FairLocalGate(background_aging_seconds=30)
    log = []
    async with gate.acquire("active", location="local", user_id="alice"):
        tasks = [
            await _enqueue(gate, log, "a-chat"),
            await _enqueue(gate, log, "b-agent", user="bob", workload="background"),
            await _enqueue(gate, log, "a-chat2"),
        ]
        clock[0] = 30
    await asyncio.gather(*tasks)
    assert log == ["b-agent", "a-chat", "a-chat2"]


async def test_zero_aging_makes_older_same_user_background_eligible_immediately(clock):
    gate = FairLocalGate(background_aging_seconds=0)
    log = []
    async with gate.acquire("active", location="local", user_id="alice"):
        background = await _enqueue(gate, log, "agent", workload="background")
        foreground = await _enqueue(gate, log, "chat")
    await asyncio.gather(background, foreground)
    assert log == ["agent", "chat"]


@pytest.mark.parametrize("cancelled_workload", ["interactive", "background"])
async def test_cancelled_interior_waiter_is_not_selected(clock, cancelled_workload):
    gate = FairLocalGate()
    log = []
    async with gate.acquire("active", location="local", user_id="alice"):
        background = await _enqueue(gate, log, "agent", workload="background")
        cancelled = await _enqueue(gate, log, "cancelled", workload=cancelled_workload)
        foreground = await _enqueue(gate, log, "chat")
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    await asyncio.gather(background, foreground)
    assert log == ["chat", "agent"]
    assert gate._available == 1
    assert gate.snapshot() == {}


async def test_cancel_promoted_background_after_grant_passes_slot_to_foreground(clock, monkeypatch):
    gate = FairLocalGate()
    log = []
    paused = asyncio.Event()
    hook_calls = 0

    async def pause_first_grantee():
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            paused.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(gate, "_after_waiter_granted", pause_first_grantee)
    async with gate.acquire("active", location="local", user_id="alice"):
        background = await _enqueue(gate, log, "cancelled", user="bob", workload="background")
        foreground = await _enqueue(gate, log, "chat", user="carol")
        clock[0] = 30
    await paused.wait()
    background.cancel()
    with pytest.raises(asyncio.CancelledError):
        await background
    await foreground
    assert log == ["chat"]
    assert gate._available == 1
    assert gate.snapshot() == {}


async def test_priority_preserves_two_active_slots_and_cloud_bypass(clock):
    gate = FairLocalGate(concurrency=2)
    log = []
    foreground_entered = asyncio.Event()
    foreground_release = asyncio.Event()
    with local_workload("background"):
        async with gate.acquire("active1", location="local", user_id="alice"):
            async with gate.acquire("active2", location="local", user_id="bob"):
                background = await _enqueue(gate, log, "agent", workload="background")
                foreground = await _enqueue(
                    gate, log, "chat", entered=foreground_entered, release=foreground_release,
                )
                async with gate.acquire("cloud", location="cloud", user_id="carol"):
                    assert gate.pressure_snapshot()["in_use"] == 2
                    assert gate.pressure_snapshot()["waiting"] == 2
            await foreground_entered.wait()
            assert log == ["chat"]
            assert gate.pressure_snapshot()["in_use"] == 2
            assert gate.pressure_snapshot()["waiting_background"] == 1
            foreground_release.set()
        await asyncio.gather(background, foreground)
    assert log == ["chat", "agent"]
    assert gate._available == 2


@pytest.mark.parametrize("aging", [-1, float("inf"), float("nan")])
async def test_invalid_aging_is_rejected(aging):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FairLocalGate(background_aging_seconds=aging)


async def test_release_between_admission_and_wait_cannot_strand_a_generation():
    gate = FairLocalGate()
    paused = asyncio.Event()
    resume = asyncio.Event()

    class PauseAfterAdmissionLock:
        """Force the holder to release just after the contender's admission check."""

        def __init__(self):
            self.lock = asyncio.Lock()
            self.did_pause = False

        async def __aenter__(self):
            await self.lock.acquire()

        async def __aexit__(self, *_args):
            self.lock.release()
            if asyncio.current_task().get_name() == "contender" and not self.did_pause:
                self.did_pause = True
                paused.set()
                await resume.wait()

    gate._lock = PauseAfterAdmissionLock()
    entered = []

    async def contend():
        async with gate.acquire("next", location="local", user_id="bob"):
            entered.append("next")

    async with gate.acquire("active", location="local", user_id="alice"):
        contender = asyncio.create_task(contend(), name="contender")
        await paused.wait()
        # The original split fast/slow-path code had not enqueued yet at
        # this point; the release would miss it and leave a free slot.
        assert gate.snapshot() == {"bob": 1}
    assert gate._available == 0
    resume.set()
    await contender
    assert entered == ["next"]
    assert gate._available == 1


async def test_gpu_wait_observes_workload_and_preserves_legacy_histogram():
    def observations(histogram):
        return next(
            sample.value for metric in histogram.collect() for sample in metric.samples
            if sample.name.endswith("_count")
        )

    labeled = local_admission_wait_seconds.labels(stage="gpu", workload="background")
    legacy_before = observations(gpu_gate_wait_seconds)
    workload_before = observations(labeled)
    gate = FairLocalGate()
    with local_workload("background"):
        async with gate.acquire("local", location="local", user_id="alice"):
            pass
        async with gate.acquire("cloud", location="cloud", user_id="alice"):
            pass
    assert observations(gpu_gate_wait_seconds) == legacy_before + 1
    assert observations(labeled) == workload_before + 1
