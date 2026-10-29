"""Durable agent recovery and cancellation without models, network or GPUs."""

from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 10, 28, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture
async def state(tmp_path):
    path = tmp_path / "application.sqlite"
    store = ApplicationStore(path)
    owner = await store.resolve_external_identity(
        provider="test",
        subject="alice",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="test",
    )
    clock = Clock()
    store.agents._clock = clock
    value = SimpleNamespace(store=store, owner=owner, clock=clock, path=path)
    yield value
    value.store.close()


async def _create(state, *, request_id="request-1", **kwargs):
    return await state.store.agents.create(
        user_id=state.owner.user_id,
        objective="Compare the public release notes.",
        request_id=request_id,
        **kwargs,
    )


async def _get(state, task):
    return await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])


async def _control(state, task, action):
    current = await _get(state, task)
    return await state.store.agents.control(
        user_id=state.owner.user_id,
        task_id=task["id"],
        expected_revision=current["revision"],
        action=action,
    )


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


def _runtime(state, executor, **kwargs):
    return AgentRuntime(state.store.agents, executor, poll_seconds=0.01, **kwargs)


async def test_commits_the_plan_research_and_review_as_one_saved_result(state):
    task = await _create(state)
    seen = []

    async def execute(current, step_id):
        seen.append(step_id)
        assert current["operation_id"]
        assert current["current_step"] == step_id
        for earlier in current["steps"][: len(seen) - 1]:
            assert earlier["status"] == "completed"
            assert earlier["result"] == f"Saved {earlier['id']}"
        return f"Saved {step_id}"

    runtime = _runtime(state, execute)
    assert await runtime.run_once() is True
    saved = await _get(state, task)
    assert seen == ["plan", "research", "review"]
    assert saved["status"] == "completed" and saved["result"] == "Saved review"
    assert saved["attempts"] == 1
    assert await runtime.run_once() is False


async def test_restart_after_an_expired_lease_reuses_committed_steps(state):
    task = await _create(state)
    claimed = await state.store.agents.claim(lease_seconds=2)
    lease = _lease(claimed)
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    await state.store.agents.checkpoint(**lease, step_id="plan", result="Committed plan")
    await state.store.agents.reserve_operation(**lease, step_id="research")
    # Simulate the process disappearing after dispatch, before its next commit.
    state.store.close()
    state.clock.advance(3)
    state.store = ApplicationStore(state.path)
    state.store.agents._clock = state.clock
    seen = []

    async def execute(current, step_id):
        seen.append(step_id)
        assert current["steps"][0]["result"] == "Committed plan"
        return f"Recovered {step_id}"

    assert await _runtime(state, execute).run_once() is True
    saved = await _get(state, task)
    assert seen == ["research", "review"]
    assert saved["status"] == "completed" and saved["attempts"] == 2
    row = state.store._conn.execute(
        "SELECT operations,active_seconds FROM app_agent_tasks WHERE task_id=?",
        (task["id"],),
    ).fetchone()
    assert row["operations"] == 4 and row["active_seconds"] == 2


async def test_lost_checkpoint_acknowledgement_does_not_repeat_committed_work(state, monkeypatch):
    task = await _create(state)
    checkpoint = state.store.agents.checkpoint
    seen = []

    async def committed_then_disconnected(**kwargs):
        saved = await checkpoint(**kwargs)
        if kwargs["step_id"] == "plan":
            raise RuntimeError("Simulated lost acknowledgement")
        return saved

    async def execute(_task, step_id):
        seen.append(step_id)
        return f"Saved {step_id}"

    monkeypatch.setattr(state.store.agents, "checkpoint", committed_then_disconnected)
    runtime = _runtime(state, execute)
    await runtime.run_once()
    assert (await _get(state, task))["status"] == "queued"
    await runtime.run_once()
    assert seen == ["plan", "research", "review"]
    assert (await _get(state, task))["status"] == "completed"


async def test_lost_completion_acknowledgement_preserves_the_completed_result(state, monkeypatch):
    task = await _create(state)
    complete = state.store.agents.complete
    seen = []

    async def complete_then_disconnect(**kwargs):
        await complete(**kwargs)
        raise RuntimeError("Simulated lost delivery acknowledgement")

    async def execute(_task, step_id):
        seen.append(step_id)
        return f"Saved {step_id}"

    monkeypatch.setattr(state.store.agents, "complete", complete_then_disconnect)
    runtime = _runtime(state, execute)
    assert await runtime.run_once() is True
    assert await runtime.run_once() is False
    saved = await _get(state, task)
    assert saved["status"] == "completed" and saved["result"] == "Saved review"
    assert seen == ["plan", "research", "review"]


@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_controls_fence_a_result_returning_before_the_next_heartbeat(state, action):
    task = await _create(state)
    started, release = asyncio.Event(), asyncio.Event()

    async def execute(_task, _step):
        started.set()
        await release.wait()
        return "This late result must not be saved."

    attempt = asyncio.create_task(_runtime(state, execute).run_once())
    await asyncio.wait_for(started.wait(), timeout=2)
    await _control(state, task, action)
    release.set()
    await asyncio.wait_for(attempt, timeout=2)
    saved = await _get(state, task)
    assert saved["status"] == {"pause": "paused", "cancel": "cancelled"}[action]
    assert saved["result"] == "" and all(step["result"] == "" for step in saved["steps"])


@pytest.mark.parametrize("revoke_owner", [False, True])
async def test_heartbeat_cancels_work_after_pause_or_owner_revocation(state, revoke_owner):
    task = await _create(state)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def execute(_task, _step):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    attempt = asyncio.create_task(_runtime(state, execute).run_once())
    await asyncio.wait_for(started.wait(), timeout=2)
    if revoke_owner:
        with state.store._lock:
            state.store._conn.execute(
                "UPDATE app_users SET status='disabled' WHERE user_id=?",
                (state.owner.user_id,),
            )
            state.store._conn.commit()
    else:
        await _control(state, task, "pause")
    await asyncio.wait_for(attempt, timeout=2)
    assert stopped.is_set()
    row = state.store._conn.execute(
        "SELECT result FROM app_agent_tasks WHERE task_id=?",
        (task["id"],),
    ).fetchone()
    assert row["result"] == ""


async def test_shutdown_requeues_and_the_next_runtime_continues_after_saved_plan(state):
    task = await _create(state)
    started, stopped = asyncio.Event(), asyncio.Event()
    seen = []

    async def execute(_task, step_id):
        seen.append(step_id)
        if step_id == "research":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return f"Saved {step_id}"

    runtime = _runtime(state, execute)
    await runtime.start()
    await runtime.start()  # Lifecycle startup is idempotent.
    await asyncio.wait_for(started.wait(), timeout=2)
    await asyncio.wait_for(runtime.stop(), timeout=2)
    assert stopped.is_set()
    saved = await _get(state, task)
    assert saved["status"] == "queued" and saved["steps"][0]["status"] == "completed"

    async def recover(_task, step_id):
        seen.append(step_id)
        return f"Recovered {step_id}"

    await _runtime(state, recover).run_once()
    assert seen == ["plan", "research", "research", "review"]
    saved = await _get(state, task)
    assert saved["status"] == "completed" and saved["attempts"] == 2


async def test_shutdown_does_not_wait_indefinitely_for_a_cancellation_resistant_provider(state):
    task = await _create(state)
    provider_tasks = []
    started, cancelled, release, returned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def execute(_task, _step):
        provider_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            # Some provider libraries finish their cleanup before returning.
            # Ignore repeated cancellation to exercise the bounded drain.
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        returned.set()
        return "Late provider result"

    runtime = _runtime(state, execute)
    await runtime.start()
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        await asyncio.wait_for(runtime.stop(), timeout=2)
        assert cancelled.is_set() and not returned.is_set()
        assert (await _get(state, task))["status"] == "queued"
    finally:
        release.set()
        await asyncio.wait_for(returned.wait(), timeout=2)
        # Let the old work attempt its fenced checkpoint and settle.
        await asyncio.wait_for(asyncio.gather(*provider_tasks, return_exceptions=True), timeout=2)
    saved = await _get(state, task)
    assert all(step["result"] == "" for step in saved["steps"])


async def test_ordinary_failures_are_bounded_and_do_not_persist_provider_errors(state):
    task = await _create(state)
    calls = 0

    async def execute(_task, _step):
        nonlocal calls
        calls += 1
        raise RuntimeError("credential=secret and private provider content")

    runtime = _runtime(state, execute, max_attempts=2)
    assert await runtime.run_once() is True
    assert await runtime.run_once() is True
    assert await runtime.run_once() is False
    saved = await _get(state, task)
    assert calls == 2 and saved["attempts"] == 2 and saved["status"] == "failed"
    assert "credential" not in str(saved) and "private provider" not in str(saved)


async def test_pausing_and_resuming_cannot_reset_the_attempt_allowance(state):
    task = await _create(state)
    calls = 0

    async def execute(_task, _step):
        nonlocal calls
        calls += 1
        await _control(state, task, "pause")
        return "Revoked result"

    runtime = _runtime(state, execute, max_attempts=2)
    for _ in range(2):
        assert await runtime.run_once() is True
        assert (await _get(state, task))["status"] == "paused"
        await _control(state, task, "resume")
    assert await runtime.run_once() is False
    saved = await _get(state, task)
    assert calls == 2 and saved["attempts"] == 2 and saved["status"] == "failed"


async def test_dispatch_policy_stops_are_terminal_without_retrying(state):
    task = await _create(state)

    async def execute(_task, _step):
        raise AgentDispatchStoppedError("Public research is unavailable for this account.")

    runtime = _runtime(state, execute)
    assert await runtime.run_once() is True
    assert await runtime.run_once() is False
    saved = await _get(state, task)
    assert saved["attempts"] == 1 and saved["status"] == "failed"
    assert saved["error"] == "Public research is unavailable for this account."


@pytest.mark.parametrize("result", ["", " ", None, "x" * 65, "界" * 22000])
async def test_invalid_or_excessive_step_outputs_stop_without_retrying(state, result):
    task = await _create(state)

    async def execute(_task, _step):
        return result

    await _runtime(state, execute, max_output_chars=64 if result == "x" * 65 else 64000).run_once()
    saved = await _get(state, task)
    assert saved["status"] == "failed" and saved["attempts"] == 1
    assert saved["result"] == "" and all(step["result"] == "" for step in saved["steps"])


async def test_step_timeout_cancels_the_executor_and_saves_a_safe_terminal_error(state):
    task = await _create(state)
    stopped = asyncio.Event()

    async def execute(_task, _step):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    await _runtime(state, execute, step_timeout_seconds=0.02).run_once()
    saved = await _get(state, task)
    assert stopped.is_set() and saved["status"] == "failed"
    assert saved["error"] == "Agent step exceeded its time limit."


async def test_server_operation_allowance_stops_before_the_next_model_step(state):
    task = await _create(state, limits={"max_operations": 1})
    seen = []

    async def execute(_task, step_id):
        seen.append(step_id)
        return "Saved plan"

    await _runtime(state, execute).run_once()
    saved = await _get(state, task)
    assert seen == ["plan"] and saved["status"] == "failed"
    assert saved["steps"][0]["result"] == "Saved plan"


async def test_server_active_time_allowance_cancels_inflight_work(state):
    task = await _create(state, limits={"max_active_seconds": 1})
    stopped = asyncio.Event()

    async def execute(_task, _step):
        state.clock.advance(2)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    await _runtime(state, execute).run_once()
    saved = await _get(state, task)
    assert stopped.is_set() and saved["status"] == "failed"
    assert saved["result"] == ""
    assert await _runtime(state, execute).run_once() is False


async def test_concurrent_dispatch_calls_execute_only_one_task_at_a_time(state):
    first = await _create(state)
    second = await _create(state, request_id="request-2")
    active = maximum = 0

    async def execute(_task, step_id):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return step_id

    runtime = _runtime(state, execute)
    assert await asyncio.gather(runtime.run_once(), runtime.run_once()) == [True, True]
    assert maximum == 1
    assert (await _get(state, first))["status"] == "completed"
    assert (await _get(state, second))["status"] == "completed"
