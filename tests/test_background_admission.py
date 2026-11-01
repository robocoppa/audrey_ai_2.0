"""The paired queues yield pre-generation background permits without lost slots."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from audrey.background_admission import BackgroundGenerationGate
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.scheduling import current_background_queued_at, current_local_workload


@pytest.fixture
def setup(monkeypatch):
    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter)
    monkeypatch.setattr("audrey.background_admission.time", clock)
    monkeypatch.setattr("audrey.pipeline.fair_gate.time", clock)
    gate = FairLocalGate()
    registry = UserInflightRegistry(max_inflight_per_user=1)
    registry._clock = clock.monotonic
    return now, gate, registry, BackgroundGenerationGate(gate, registry)


async def _until(check):
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0)
    assert check()


async def _chat(gate, registry, order, *, entered=None, release=None, reserved=None):
    async with registry.slot("alice"):
        if reserved:
            reserved.set()
        async with gate.acquire("chat", location="local", user_id="alice"):
            assert current_local_workload() == "interactive"
            order.append("chat")
            if entered:
                entered.set()
            if release:
                await release.wait()


async def _agent(adapter, order, *, entered=None, release=None, ages=None):
    caller = asyncio.current_task()
    async with adapter.acquire("agent", location="local", user_id="alice"):
        assert asyncio.current_task() is caller
        assert current_local_workload() == "background"
        if ages is not None:
            ages.append(current_background_queued_at())
        order.append("agent")
        if entered:
            entered.set()
        if release:
            await release.wait()


def _assert_idle(gate, registry):
    assert gate.pressure_snapshot()["in_use"] == 0
    assert gate.snapshot() == {}
    assert registry.pressure_snapshot()["in_use"] == 0
    assert registry.pressure_snapshot()["waiting"] == 0
    assert current_local_workload() == "interactive"
    assert current_background_queued_at() is None


async def test_chat_passes_same_owner_background_already_waiting_for_gpu(setup):
    _, gate, registry, adapter = setup
    order, ages = [], []
    chat_reserved = asyncio.Event()
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order, ages=ages))
        await _until(lambda: gate.snapshot() == {"alice": 1})
        assert registry.snapshot() == {"alice": 1}
        chat = asyncio.create_task(_chat(gate, registry, order, reserved=chat_reserved))
        await chat_reserved.wait()
        assert gate.pressure_snapshot()["waiting_interactive"] == 1
        assert gate.pressure_snapshot()["waiting_background"] == 0
        assert order == []
    await asyncio.gather(agent, chat)
    assert order == ["chat", "agent"]
    assert ages == [0.0]
    _assert_idle(gate, registry)


async def test_aged_background_keeps_its_permit_and_runs_before_new_chat(setup):
    now, gate, registry, adapter = setup
    order = []
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order))
        await _until(lambda: gate.snapshot() == {"alice": 1})
        now[0] = 30
        chat = asyncio.create_task(_chat(gate, registry, order))
        await _until(lambda: registry.pressure_snapshot()["waiting"] == 1)
        assert gate.pressure_snapshot()["aged_background"] == 1
        assert not registry._sems["alice"]._permits[0].give_way.is_set()
    await asyncio.gather(agent, chat)
    assert order == ["agent", "chat"]
    _assert_idle(gate, registry)


async def test_retry_retains_original_age_until_background_promoted(setup):
    now, gate, registry, adapter = setup
    order, ages = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order, ages=ages))
        await _until(lambda: gate.snapshot() == {"alice": 1})
        now[0] = 29
        chat = asyncio.create_task(_chat(gate, registry, order, entered=entered, release=release))
        await _until(lambda: gate.pressure_snapshot()["waiting_interactive"] == 1)
        await _until(lambda: registry.pressure_snapshot()["waiting"] == 1)
        assert registry._sems["alice"]._waiters[0].enqueued_at == 0
    await entered.wait()
    now[0] = 30
    later_chat = asyncio.create_task(_chat(gate, registry, order))
    await _until(lambda: registry.pressure_snapshot()["waiting"] == 2)
    release.set()
    await asyncio.gather(agent, chat, later_chat)
    assert order == ["chat", "agent", "chat"]
    assert ages == [0.0]
    _assert_idle(gate, registry)


@pytest.mark.parametrize("handoff", [False, True])
async def test_cancellation_during_gpu_wait_or_handoff_releases_both_queues(setup, handoff):
    _, gate, registry, adapter = setup
    order = []
    chat = None
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order))
        await _until(lambda: gate.snapshot() == {"alice": 1})
        if handoff:
            permit = registry._sems["alice"]._permits[0]
            chat = asyncio.create_task(_chat(gate, registry, order))
            await permit.give_way.wait()
        agent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await agent
        assert gate.pressure_snapshot()["waiting_background"] == 0
    if chat:
        await chat
    assert order == (["chat"] if handoff else [])
    _assert_idle(gate, registry)


async def test_simultaneous_gpu_grant_and_give_way_does_not_run_revoked_background(setup):
    _, gate, registry, adapter = setup
    order = []
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order))
        await _until(lambda: gate.snapshot() == {"alice": 1})
        permit = registry._sems["alice"]._permits[0]
        chat = asyncio.create_task(_chat(gate, registry, order))
        await permit.give_way.wait()
        # Releasing Bob grants the background GPU future in the same loop
        # turn that its permit is revoked. mark_running must reject it.
    await asyncio.gather(agent, chat)
    assert order == ["chat", "agent"]
    _assert_idle(gate, registry)


async def test_running_generation_is_never_interrupted_by_chat(setup):
    _, gate, registry, adapter = setup
    order = []
    entered, release = asyncio.Event(), asyncio.Event()
    agent = asyncio.create_task(_agent(adapter, order, entered=entered, release=release))
    await entered.wait()
    chat = asyncio.create_task(_chat(gate, registry, order))
    await _until(lambda: registry.pressure_snapshot()["waiting"] == 1)
    permit = registry._sems["alice"]._permits[0]
    assert permit._running
    assert not permit.give_way.is_set()
    assert order == ["agent"] and not agent.done()
    release.set()
    await asyncio.gather(agent, chat)
    assert order == ["agent", "chat"]
    _assert_idle(gate, registry)


async def test_cancel_after_gpu_grant_before_adapter_resume_returns_slot(setup, monkeypatch):
    _, gate, registry, adapter = setup
    order = []
    paused = asyncio.Event()

    async def pause_granted_waiter():
        paused.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(gate, "_after_waiter_granted", pause_granted_waiter)
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order))
        await _until(lambda: gate.snapshot() == {"alice": 1})
    await paused.wait()
    agent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent
    assert order == []
    _assert_idle(gate, registry)


async def test_repeated_cancellation_cannot_interrupt_gpu_child_cleanup(setup, monkeypatch):
    _, gate, registry, adapter = setup
    granted, cleanup_started, allow_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order = []

    async def delayed_cancel_cleanup():
        granted.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await allow_cleanup.wait()
            raise

    monkeypatch.setattr(gate, "_after_waiter_granted", delayed_cancel_cleanup)
    async with gate.acquire("bob", location="local", user_id="bob"):
        agent = asyncio.create_task(_agent(adapter, order))
        await _until(lambda: gate.snapshot() == {"alice": 1})
    await granted.wait()
    agent.cancel()
    await cleanup_started.wait()
    agent.cancel()
    # The second runtime cancel must not reach the already-cancelling child.
    await asyncio.sleep(0)
    assert not agent.done()
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await agent
    assert order == []
    _assert_idle(gate, registry)
