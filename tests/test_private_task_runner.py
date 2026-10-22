"""A bounded Private step cannot outlive its tax-write authority."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from test_private_tasks import PINS, _claim, _control, _create
from test_private_tasks import state as state

from audrey.app_state.private_tasks import (
    PrivateTaskConflictError,
    PrivateTaskLeaseExpiredError,
    private_task_scope,
)
from audrey.app_state.tax_cases import TaxCaseConflictError
from audrey.private_task_runner import (
    PreparedStep,
    PrivateStepBudgetError,
    PrivateTaskStepRunner,
    StepResult,
)
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.private_dispatch import private_tool_handler


def _runner(state, *, handlers=None, guard=None):
    policy = ProcessingPolicy(
        local_models=PINS["model_digests"],
        ollama_base_url="http://ollama:11434",
        allowed_tools=frozenset(handlers or {}),
        tool_servers=frozenset({"http://custom-tools:8001"}),
    )

    async def check():
        await state.store.tax_cases.get(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id
        )
        if guard is not None:
            await guard()

    @asynccontextmanager
    async def prepare(user_id, conversation_id):
        assert user_id == state.owner.user_id
        assert conversation_id == state.conversation.conversation_id
        with processing_scope(policy):
            yield PreparedStep(
                SimpleNamespace(), PINS, state.owner.storage_namespace, handlers or {}, check
            )

    return PrivateTaskStepRunner(state.store.private_tasks, prepare, poll_seconds=0.01)


async def _fact(state, **changes):
    return await state.store.tax_cases.update(
        user_id=state.owner.user_id,
        conversation_id=state.conversation.conversation_id,
        expected_revision=state.case["revision"],
        operation="propose_fact",
        data={"label": "Filing status", "value": "single", "kind": "user_statement"},
        **changes,
    )


async def test_case_write_and_task_expected_revision_advance_atomically(state):
    task = await _create(state)

    async def work(_app, step):
        state.case = await _fact(state)
        current = await step.check()
        assert current["case_revision"] == state.case["revision"]
        await step.tokens_used(17)
        return StepResult(
            {
                "stage": "facts",
                "summary": "One fact remains proposed.",
                "fact_ids": [state.case["facts"][0]["id"]],
            },
            "waiting_for_user",
        )

    saved = await _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    assert saved["status"] == "waiting_for_user"
    assert saved["usage"]["model_tokens"] == 17
    assert saved["checkpoint_sequence"] == 1
    assert state.case["facts"][0]["status"] == "proposed"


@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_paused_or_cancelled_worker_cannot_mutate_tax_case(state, action):
    running = await _claim(state, await _create(state))
    with private_task_scope(state.store.private_tasks, running):
        await _control(state, running, action)
        with pytest.raises(TaxCaseConflictError):
            await _fact(state)
    assert (
        await state.store.tax_cases.get(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id
        )
    )["facts"] == []


async def test_expired_worker_cannot_mutate_tax_case(state):
    running = await _claim(state, await _create(state), lease_seconds=2)
    with private_task_scope(state.store.private_tasks, running):
        state.clock.advance(3)
        with pytest.raises(TaxCaseConflictError):
            await _fact(state)
    assert state.store._conn.execute("SELECT revision FROM app_tax_cases").fetchone()[0] == 1


async def test_task_revision_failure_rolls_back_case_write(state, monkeypatch):
    running = await _claim(state, await _create(state))

    def reject(_record):
        raise RuntimeError("Simulated storage interruption")

    with private_task_scope(state.store.private_tasks, running), monkeypatch.context() as patch:
        patch.setattr(state.store.private_tasks, "_write", reject)
        with pytest.raises(RuntimeError):
            await _fact(state)
    assert state.store._conn.execute("SELECT revision FROM app_tax_cases").fetchone()[0] == 1
    assert (
        await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=running["id"])
        == running
    )


async def test_owner_cancel_stops_waiting_callback_and_preserves_cancelled_state(state):
    task = await _create(state)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def work(_app, _step):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    attempt = asyncio.create_task(
        _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    )
    await started.wait()
    current = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
    cancelled = await _control(state, current, "cancel")
    with pytest.raises(PrivateTaskConflictError):
        await asyncio.wait_for(attempt, 2)
    assert stopped.is_set()
    assert (
        await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
        == cancelled
    )


async def test_lease_without_observed_progress_expires_and_is_paused(state):
    task = await _create(state)
    started = asyncio.Event()

    async def work(_app, _step):
        started.set()
        await asyncio.Event().wait()

    attempt = asyncio.create_task(
        _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    )
    await started.wait()
    state.clock.advance(121)
    with pytest.raises(PrivateTaskLeaseExpiredError):
        await asyncio.wait_for(attempt, 2)
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == "paused" and saved["stop_reason"] == "lease_expired"
    assert saved["usage"]["steps"] == 1


async def test_step_timeout_retains_partial_usage_without_false_completion(state):
    task = await _create(state, limits={"max_wall_seconds": 1})

    async def work(_app, step):
        await step.tokens_used(9)
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == "paused" and saved["stop_reason"] == "step_timeout"
    assert saved["usage"]["steps"] == 1 and saved["usage"]["model_tokens"] == 9
    assert saved["checkpoint"] is None


async def test_wrapped_tools_charge_failed_calls_and_block_calls_over_budget(state):
    seen = []

    async def calculate(args):
        seen.append(args)
        return {"result": "2"}

    task = await _create(state, limits={"max_tool_calls": 1})

    async def work(_app, _step):
        handler = private_tool_handler("calculate", user_id=state.owner.storage_namespace)
        await handler({"expression": "1+1"})
        await handler({"expression": "2+2"})

    with pytest.raises(PrivateStepBudgetError):
        await _runner(state, handlers={"calculate": calculate}).execute(
            user_id=state.owner.user_id, task_id=task["id"], work=work
        )
    assert seen == [{"expression": "1+1"}]
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["usage"]["tool_calls"] == 1 and saved["status"] == "paused"


async def test_inherited_callback_loses_write_authority_when_step_exits(state):
    task = await _create(state)
    release = asyncio.Event()
    children = []

    async def late_write():
        await release.wait()
        await _fact(state)

    async def work(_app, _step):
        children.append(asyncio.create_task(late_write()))
        return StepResult({"stage": "inspect", "summary": "Inventory saved."}, "completed")

    await _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    release.set()
    with pytest.raises(TaxCaseConflictError):
        await children[0]
    assert state.store._conn.execute("SELECT revision FROM app_tax_cases").fetchone()[0] == 1


async def test_observed_usage_is_durable_before_checkpoint_and_not_counted_twice(state):
    task = await _create(state)

    async def work(_app, step):
        await step.tokens_used(23)
        current = await state.store.private_tasks.get(
            user_id=state.owner.user_id, task_id=task["id"]
        )
        assert current["status"] == "running" and current["checkpoint"] is None
        assert current["usage"]["model_tokens"] == 23
        return StepResult(
            {"stage": "review", "summary": "A partial finding is saved."}, "completed"
        )

    saved = await _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work)
    assert saved["usage"]["model_tokens"] == 23 and saved["usage"]["steps"] == 1


async def test_token_budget_allows_exact_completion_but_preserves_overrun_as_partial(state):
    task = await _create(state, limits={"max_model_tokens": 10})

    async def exact(_app, step):
        await step.tokens_used(10)
        return StepResult(
            {"stage": "review", "summary": "The limited review is saved."}, "completed"
        )

    saved = await _runner(state).execute(
        user_id=state.owner.user_id, task_id=task["id"], work=exact
    )
    assert saved["status"] == "completed" and saved["usage"]["model_tokens"] == 10
    replacement = await _create(state, limits={"max_model_tokens": 10})

    async def overrun(_app, step):
        await step.tokens_used(11)
        pytest.fail("The overrun must stop this callback")

    with pytest.raises(PrivateStepBudgetError):
        await _runner(state).execute(
            user_id=state.owner.user_id, task_id=replacement["id"], work=overrun
        )
    partial = await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=replacement["id"]
    )
    assert partial["status"] == "paused" and partial["stop_reason"] == "budget_exhausted"
    assert partial["usage"]["model_tokens"] == 11 and partial["usage"]["steps"] == 1


async def test_live_admission_refusal_happens_before_claim_or_callback(state):
    task = await _create(state)

    async def guard():
        raise PermissionError("Access was withdrawn")

    async def work(_app, _step):
        pytest.fail("A denied task must not start")

    with pytest.raises(PermissionError):
        await _runner(state, guard=guard).execute(
            user_id=state.owner.user_id, task_id=task["id"], work=work
        )
    assert (
        await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"]) == task
    )


async def test_failed_tool_dispatch_is_charged_and_preserves_partial_state(state):
    task = await _create(state)

    async def failed(_arguments):
        raise ValueError("Synthetic tool failure")

    async def work(_app, _step):
        await private_tool_handler("calculate", user_id=state.owner.storage_namespace)({})

    with pytest.raises(ValueError):
        await _runner(state, handlers={"calculate": failed}).execute(
            user_id=state.owner.user_id, task_id=task["id"], work=work
        )
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == "paused" and saved["stop_reason"] == "step_failed"
    assert saved["usage"]["tool_calls"] == 1 and saved["usage"]["steps"] == 1


async def test_slow_cancellation_cleanup_cannot_hold_step_open_or_write_late(state):
    task = await _create(state, limits={"max_wall_seconds": 1})
    release, late_write_blocked = asyncio.Event(), asyncio.Event()
    callbacks = []

    async def work(_app, _step):
        callbacks.append(asyncio.current_task())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            with pytest.raises(TaxCaseConflictError):
                await _fact(state)
            late_write_blocked.set()

    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                _runner(state).execute(user_id=state.owner.user_id, task_id=task["id"], work=work),
                4,
            )
        assert callbacks and not callbacks[0].done()
    finally:
        release.set()
        if callbacks:
            await asyncio.wait_for(callbacks[0], 2)
    assert late_write_blocked.is_set()
    assert state.store._conn.execute("SELECT revision FROM app_tax_cases").fetchone()[0] == 1


async def test_worker_scope_cannot_impersonate_user_confirmation(state):
    state.case = await _fact(state)
    running = await _claim(state, await _create(state))
    with private_task_scope(state.store.private_tasks, running):
        with pytest.raises(TaxCaseConflictError):
            await state.store.tax_cases.update(
                user_id=state.owner.user_id,
                conversation_id=state.conversation.conversation_id,
                expected_revision=state.case["revision"],
                operation="confirm_fact",
                data={"fact_id": state.case["facts"][0]["id"]},
                actor="user",
            )
    current = await state.store.tax_cases.get(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id
    )
    assert current["facts"][0]["status"] == "proposed"


async def test_worker_scope_cannot_write_another_conversation_owned_by_same_user(state):
    other = await state.store.conversations.create(
        user_id=state.owner.user_id, private_required=True
    )
    other_case = await state.store.tax_cases.create(
        user_id=state.owner.user_id,
        conversation_id=other.conversation_id,
        tax_year=2025,
        goal="Other case",
        reference_release="rel_current",
    )
    running = await _claim(state, await _create(state))
    with private_task_scope(state.store.private_tasks, running):
        with pytest.raises(TaxCaseConflictError):
            await state.store.tax_cases.update(
                user_id=state.owner.user_id,
                conversation_id=other.conversation_id,
                expected_revision=other_case["revision"],
                operation="propose_fact",
                data={"label": "Filing status", "value": "single", "kind": "user_statement"},
            )
    assert (
        await state.store.tax_cases.get(
            user_id=state.owner.user_id, conversation_id=other.conversation_id
        )
        == other_case
    )


async def test_child_cannot_write_between_callback_completion_and_checkpoint(state):
    task = await _create(state)
    work_finished, release_child, child_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children = []

    async def guard():
        if work_finished.is_set():
            release_child.set()
            await asyncio.wait_for(child_finished.wait(), 2)

    async def child():
        await release_child.wait()
        try:
            with pytest.raises(TaxCaseConflictError):
                await _fact(state)
        finally:
            child_finished.set()

    async def work(_app, _step):
        children.append(asyncio.create_task(child()))
        work_finished.set()
        return StepResult({"stage": "inspect", "summary": "Inspection saved."}, "completed")

    saved = await _runner(state, guard=guard).execute(
        user_id=state.owner.user_id, task_id=task["id"], work=work
    )
    await children[0]
    assert saved["status"] == "completed"
    assert state.store._conn.execute("SELECT revision FROM app_tax_cases").fetchone()[0] == 1
