"""Durable agent ownership, recovery, resource reservations and write failures."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.agents import (
    AgentTaskAccessError,
    AgentTaskConflictError,
    AgentTaskError,
)
from audrey.app_state.titles import normalize_agent_title


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 10, 28, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


async def _owner(store, name="alice"):
    return await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com",
        display_name=name.title(), role="user", auth_method="test",
    )


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, other = await _owner(store), await _owner(store, "bob")
    clock = Clock()
    store.agents._clock = clock
    yield SimpleNamespace(store=store, owner=owner, other=other, clock=clock)
    store.close()


async def _create(state, **changes):
    kwargs = dict(user_id=state.owner.user_id, objective="Compare public research.", request_id="request-1")
    kwargs.update(changes)
    return await state.store.agents.create(**kwargs)


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def _step(state, task, step_id, result="Observed result."):
    await state.store.agents.reserve_operation(**_lease(task), step_id=step_id)
    return await state.store.agents.checkpoint(**_lease(task), step_id=step_id, result=result)


async def _control(state, task, action, **changes):
    kwargs = dict(user_id=state.owner.user_id, task_id=task["id"],
                  expected_revision=task["revision"], action=action)
    kwargs.update(changes)
    return await state.store.agents.control(**kwargs)


async def test_create_idempotent_and_public_records_have_no_worker_credentials(state):
    first, second = await asyncio.gather(_create(state), _create(state))
    assert first == second
    assert set(first) == {
        "id", "objective", "title", "status", "revision", "created_at", "updated_at", "steps",
        "result", "error", "attempts",
    }
    assert [step["id"] for step in first["steps"]] == ["plan", "research", "review"]
    assert all(step["status"] == "pending" for step in first["steps"])
    with pytest.raises(AgentTaskConflictError):
        await _create(state, objective="A different objective.")
    assert await state.store.agents.list(user_id=state.owner.user_id) == [first]
    assert await state.store.agents.list(user_id=state.other.user_id) == []


async def test_agent_title_fallback_is_short_and_never_rewrites_the_objective(state):
    objective = "Explain the history of this detailed public research topic. " * 20
    task = await _create(state, objective=objective)
    assert task["objective"] == objective.strip()
    assert len(task["title"]) <= 72 and task["title"].endswith("…")
    assert task["title"] != task["objective"]
    assert "title_attempted" not in task and "title_generated" not in task
    summary = (await state.store.agents.list(user_id=state.owner.user_id, summary=True))[0]
    assert summary["title"] == task["title"]


async def test_title_is_saved_only_by_a_current_reserved_lease_and_survives_rerun(state):
    await _create(state)
    task = await state.store.agents.claim()
    lease = _lease(task)
    operation = await state.store.agents.reserve_title_operation(**lease, step_id="plan")
    assert operation and await state.store.agents.reserve_title_operation(**lease, step_id="plan") is None
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.save_title(**lease, step_id="plan", operation_id="foreign", title="Public Research Release Notes")
    title = await state.store.agents.save_title(**lease, step_id="plan", operation_id=operation, title='Title: "Public Research Release Notes"')
    assert title == "Public Research Release Notes"
    for step_id in ("plan", "research", "review"):
        task = await _step(state, task, step_id)
    completed = await state.store.agents.complete(**_lease(task), result="Report.")
    assert completed["title"] == title
    queued = await _control(state, completed, "rerun")
    assert queued["title"] == title and queued["steps"][0] == completed["steps"][0]
    reclaimed = await state.store.agents.claim()
    assert reclaimed["title_generated"] is True
    assert await state.store.agents.reserve_title_operation(**_lease(reclaimed), step_id="research") is None
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.save_title(**lease, step_id="research", operation_id=operation, title="Stale Old Worker Title")


async def test_title_attempt_preserves_last_operation_and_missing_title_retries_only_after_rerun(state):
    await _create(state, limits={"max_operations": 2})
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    assert await state.store.agents.reserve_title_operation(**_lease(task), step_id="plan") is None
    row = state.store._conn.execute("SELECT title,title_attempted,operations FROM app_agent_tasks").fetchone()
    assert tuple(row) == ("", 1, 1)
    failed = await state.store.agents.fail(**_lease(task), error="Useful work failed.")
    await _control(state, failed, "rerun")
    claimed = await state.store.agents.claim()
    assert claimed["title_attempted"] == 0 and not claimed["title_generated"]


async def test_title_write_cannot_cross_account_purge(state):
    await _create(state)
    task = await state.store.agents.claim()
    operation = await state.store.agents.reserve_title_operation(**_lease(task), step_id="plan")
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.save_title(**_lease(task), step_id="plan", operation_id=operation, title="Deleted Private Research Title")
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0] == 0


async def test_title_additive_migration_repairs_partial_schema_without_losing_saved_titles(state):
    task = await _create(state)
    task = await state.store.agents.claim()
    task = await _step(state, task, "plan", result="Previously committed plan survives migration.")
    with state.store._lock, state.store._conn:
        state.store._conn.execute("UPDATE app_agent_tasks SET title='Saved Public Research Title' WHERE task_id=?", (task["id"],))
        state.store._conn.execute("ALTER TABLE app_agent_tasks DROP COLUMN title_attempted")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=28")
    upgraded = ApplicationStore(state.store.path)
    try:
        assert upgraded.schema_version == 29
        current = await upgraded.agents.get(user_id=state.owner.user_id, task_id=task["id"])
        assert current["title"] == "Saved Public Research Title" and current["objective"] == task["objective"]
        assert current["steps"][0] == task["steps"][0]
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        upgraded.close()


@pytest.mark.parametrize("value,expected", [
    ('Title: "Public Research Release Notes"', "Public Research Release Notes"),
    ("```\n# Local Orchard Gate Review\n```", "Local Orchard Gate Review"),
    ("One Two", ""), ("One Two Three Four Five Six Seven Eight", ""),
    ("<script> Never Render This", ""), ("", ""),
    ("Public \ud800 Research", ""), ("Public \x00 Research", ""),
])
def test_generated_agent_titles_are_short_plain_labels(value, expected):
    assert normalize_agent_title(value) == expected


async def test_foreign_and_missing_ids_are_indistinguishable(state):
    task = await _create(state)
    messages = []
    for task_id in (task["id"], "missing"):
        with pytest.raises(AgentTaskAccessError) as error:
            await state.store.agents.get(user_id=state.other.user_id, task_id=task_id)
        messages.append(str(error.value))
        with pytest.raises(AgentTaskAccessError):
            await _control(state, task, "cancel", user_id=state.other.user_id, task_id=task_id)
    assert messages[0] == messages[1]


async def test_summary_listing_omits_all_report_bodies_but_preserves_public_steps(state):
    await _create(state)
    task = await state.store.agents.claim()
    task = await _step(state, task, "plan", result="Saved plan report body.")
    partial = await state.store.agents.list(user_id=state.owner.user_id, summary=True)
    assert [step["status"] for step in partial[0]["steps"]] == ["completed", "running", "pending"]
    assert "report body" not in repr(partial)
    for step_id in ("research", "review"):
        task = await _step(state, task, step_id, result=f"Saved {step_id} report body.")
    await state.store.agents.complete(**_lease(task), result="Saved final report body.")
    full = await state.store.agents.list(user_id=state.owner.user_id)
    summary = await state.store.agents.list(user_id=state.owner.user_id, summary=True)
    assert len(summary) == 1
    assert set(summary[0]) == set(full[0]) | {"summary"}
    assert summary[0]["summary"] is True
    assert summary[0]["result"] == ""
    assert summary[0]["status"] == "completed"
    assert summary[0]["steps"] == [
        {"id": "plan", "label": "Plan", "status": "completed", "result": ""},
        {"id": "research", "label": "Research", "status": "completed", "result": ""},
        {"id": "review", "label": "Review", "status": "completed", "result": ""},
    ]
    assert full[0]["result"] == "Saved final report body."
    assert full[0]["steps"][0]["result"] == "Saved plan report body."
    assert "report body" not in repr(summary)
    assert await state.store.agents.list(user_id=state.other.user_id, summary=True) == []


async def test_active_owner_required_on_create_read_control_and_dispatch(state):
    await _create(state)
    task = await state.store.agents.claim()
    with state.store._lock, state.store._conn:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?",
                                  (state.owner.user_id,))
    for call in (
        _create(state, request_id="request-2"),
        state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]),
        state.store.agents.list(user_id=state.owner.user_id),
        _control(state, task, "cancel"),
        state.store.agents.validate_lease(**_lease(task)),
        state.store.agents.heartbeat(**_lease(task)),
        state.store.agents.reserve_operation(**_lease(task), step_id="plan"),
        state.store.agents.checkpoint(**_lease(task), step_id="plan", result="No write."),
        state.store.agents.fail(**_lease(task), error="No write."),
    ):
        with pytest.raises(AgentTaskAccessError):
            await call
    assert await state.store.agents.claim() is None


async def test_deletion_request_fences_even_an_active_owner(state):
    await _create(state)
    task = await state.store.agents.claim()
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO account_deletion_requests(user_id,purge_id,storage_namespace,requested_at) "
            "VALUES (?,?,?,?)", (state.owner.user_id, "purge", "alice", state.clock().isoformat()),
        )
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.validate_lease(**_lease(task))
    with pytest.raises(AgentTaskAccessError):
        await _create(state, request_id="request-2")


async def test_claim_is_atomic_and_one_running_task_per_owner(state):
    await _create(state)
    await _create(state, request_id="request-2")
    claims = await asyncio.gather(*(state.store.agents.claim() for _ in range(5)))
    assert sum(task is not None for task in claims) == 1
    await _create(state, request_id="other", user_id=state.other.user_id)
    assert (await state.store.agents.claim())["user_id"] == state.other.user_id


async def test_claim_is_atomic_between_separate_store_connections(state):
    await _create(state)
    second = ApplicationStore(state.store.path)
    second.agents._clock = state.clock
    try:
        claims = await asyncio.gather(state.store.agents.claim(), second.agents.claim())
        assert sum(task is not None for task in claims) == 1
    finally:
        second.close()


async def test_steps_require_reservations_and_ordered_commits(state):
    await _create(state)
    task = await state.store.agents.claim()
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Unreserved.")
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.reserve_operation(**_lease(task), step_id="review")
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.complete(**_lease(task), result="Too early.")
    task = await _step(state, task, "plan")
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Overwrite.")
    task = await _step(state, task, "research")
    task = await _step(state, task, "review")
    assert task["current_step"] == ""
    completed = await state.store.agents.complete(**_lease(task), result="Saved final report.")
    assert completed["status"] == "completed"
    assert completed["result"] == "Saved final report."
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.complete(**_lease(task), result="Late result.")


async def test_expiry_recovers_steps_and_rejects_old_worker(state):
    await _create(state)
    first = await state.store.agents.claim(lease_seconds=10)
    first = await _step(state, first, "plan", result="Saved plan.")
    await state.store.agents.reserve_operation(**_lease(first), step_id="research")
    state.clock.advance(11)
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.heartbeat(**_lease(first))
    second = await state.store.agents.claim(lease_seconds=10)
    assert second["id"] == first["id"]
    assert second["attempts"] == 2
    assert second["fence"] > first["fence"]
    assert second["lease_token"] != first["lease_token"]
    assert second["current_step"] == "research"
    assert second["steps"][0]["result"] == "Saved plan."
    assert second["operations"] == 2
    assert second["active_seconds"] == 10
    for call in (
        state.store.agents.validate_lease(**_lease(first)),
        state.store.agents.checkpoint(**_lease(first), step_id="research", result="Stale result."),
        state.store.agents.fail(**_lease(first), error="Stale failure."),
    ):
        with pytest.raises(AgentTaskConflictError):
            await call
    assert await state.store.agents.validate_lease(**_lease(second))


async def test_restart_preserves_live_lease_and_recovers_only_after_expiry(state):
    await _create(state)
    first = await state.store.agents.claim(lease_seconds=20)
    first = await _step(state, first, "plan")
    restarted = ApplicationStore(state.store.path)
    restarted.agents._clock = state.clock
    try:
        assert await restarted.agents.claim() is None
        assert await restarted.agents.validate_lease(**_lease(first))
        state.clock.advance(21)
        recovered = await restarted.agents.claim()
        assert recovered["current_step"] == "research"
        assert recovered["steps"][0]["status"] == "completed"
    finally:
        restarted.close()


async def test_heartbeat_charges_incrementally_without_churning_control_revision(state):
    await _create(state)
    task = await state.store.agents.claim(lease_seconds=10)
    state.clock.advance(4)
    first = await state.store.agents.heartbeat(**_lease(task), lease_seconds=10)
    assert first["active_seconds"] == 4
    assert first["revision"] == task["revision"]
    state.clock.advance(4)
    second = await state.store.agents.heartbeat(**_lease(task), lease_seconds=10)
    assert second["active_seconds"] == 8
    assert second["revision"] == task["revision"]


async def test_clock_backstep_does_not_double_charge_or_contract_the_lease(state):
    await _create(state)
    task = await state.store.agents.claim(lease_seconds=20)
    state.clock.advance(5)
    task = await state.store.agents.heartbeat(**_lease(task), lease_seconds=20)
    expiry = task["lease_expires_at"]
    state.clock.advance(-3)
    task = await state.store.agents.heartbeat(**_lease(task), lease_seconds=20)
    assert task["active_seconds"] == 5
    assert task["lease_expires_at"] == expiry
    state.clock.advance(4)
    task = await state.store.agents.heartbeat(**_lease(task), lease_seconds=20)
    assert task["active_seconds"] == 6


@pytest.mark.parametrize("action,expected", [("pause", "paused"), ("cancel", "cancelled")])
async def test_user_controls_fence_inflight_writes(state, action, expected):
    await _create(state)
    task = await state.store.agents.claim()
    task = await _step(state, task, "plan")
    state.clock.advance(5)
    controlled = await _control(state, task, action)
    assert controlled["status"] == expected
    assert controlled["steps"][0]["status"] == "completed"
    assert controlled["steps"][1]["status"] == "pending"
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.checkpoint(**_lease(task), step_id="research", result="Stale result.")
    assert await state.store.agents.claim() is None


async def test_pause_waiting_time_does_not_consume_active_allowance_or_reset_usage(state):
    await _create(state)
    task = await state.store.agents.claim()
    task = await _step(state, task, "plan")
    state.clock.advance(5)
    paused = await _control(state, task, "pause")
    state.clock.advance(50000)
    await _control(state, paused, "resume")
    resumed = await state.store.agents.claim()
    assert resumed["active_seconds"] == 5
    assert resumed["operations"] == 1
    assert resumed["attempts"] == 2
    assert resumed["current_step"] == "research"


async def test_optimistic_revision_rejects_a_stale_control(state):
    original = await _create(state)
    task = await state.store.agents.claim()
    with pytest.raises(AgentTaskConflictError):
        await _control(state, original, "pause")
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["status"] == "running"


async def test_retry_and_expired_attempts_have_one_cumulative_limit(state):
    task = await _create(state)
    first = await state.store.agents.claim(max_attempts=2)
    await state.store.agents.fail(**_lease(first), error="Temporary failure.", retry=True)
    second = await state.store.agents.claim(lease_seconds=5, max_attempts=2)
    assert second["attempts"] == 2
    state.clock.advance(6)
    assert await state.store.agents.claim(max_attempts=2) is None
    failed = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert failed["status"] == "failed"
    assert failed["attempts"] == 2
    with pytest.raises(AgentTaskConflictError):
        await _control(state, failed, "resume")
    again = await _create(state, request_id="new-history")
    assert again["id"] != failed["id"]
    assert again["attempts"] == 0


async def test_operation_reservations_are_durable_and_cannot_reset_on_resume(state):
    await _create(state, limits={"max_operations": 2})
    task = await state.store.agents.claim()
    first = await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    second = await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    assert first != second
    await state.store.agents.fail(**_lease(task), error="Interrupted.", retry=True)
    resumed = await state.store.agents.claim()
    assert resumed["operations"] == 2
    with pytest.raises(AgentTaskError, match="operation allowance"):
        await state.store.agents.reserve_operation(**_lease(resumed), step_id="plan")
    count = state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0]
    assert count == 2


async def test_active_wall_budget_blocks_dispatch_and_recovery(state):
    task = await _create(state, limits={"max_active_seconds": 5})
    claimed = await state.store.agents.claim(lease_seconds=10)
    state.clock.advance(5)
    with pytest.raises(AgentTaskError, match="active execution allowance"):
        await state.store.agents.validate_lease(**_lease(claimed))
    with pytest.raises(AgentTaskError):
        await state.store.agents.reserve_operation(**_lease(claimed), step_id="plan")
    state.clock.advance(6)
    assert await state.store.agents.claim() is None
    failed = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert failed["status"] == "failed"


async def test_content_and_queue_bounds_are_enforced_before_write(state):
    for changes in (
        {"objective": " "}, {"objective": "x" * 8001}, {"request_id": ""},
        {"objective": "invalid surrogate \ud800"},
        {"limits": {"max_operations": True}}, {"limits": {"untrusted_limit": 1}},
    ):
        with pytest.raises(AgentTaskError):
            await _create(state, **changes)
    await _create(state, limits={"max_queued_per_user": 1})
    with pytest.raises(AgentTaskError, match="capacity"):
        await _create(state, request_id="excess", limits={"max_queued_per_user": 1})
    claimed = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(claimed), step_id="plan")
    with pytest.raises(AgentTaskError):
        await state.store.agents.checkpoint(**_lease(claimed), step_id="plan", result="é" * 32001)
    with pytest.raises(AgentTaskError, match="saved step storage allowance"):
        await state.store.agents.checkpoint(**_lease(claimed), step_id="plan", result="x" + "\0" * 40000)
    assert not state.store._conn.in_transaction
    assert state.store._conn.execute("SELECT status FROM app_agent_operations").fetchone()[0] == "reserved"


async def test_global_queue_bound_includes_other_owners_and_paused_tasks(state):
    first = await _create(state, limits={"max_queued_global": 1})
    await _control(state, first, "pause")
    with pytest.raises(AgentTaskError, match="capacity"):
        await _create(state, user_id=state.other.user_id, limits={"max_queued_global": 1})


async def test_failed_checkpoint_rolls_back_result_and_receipt_then_connection_recovers(state):
    await _create(state)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    state.store._conn.execute(
        "CREATE TEMP TRIGGER reject_agent_write BEFORE UPDATE ON app_agent_tasks "
        "BEGIN SELECT RAISE(ABORT,'injected checkpoint failure'); END"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected checkpoint"):
            await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Must roll back.")
        assert not state.store._conn.in_transaction
        receipt = state.store._conn.execute("SELECT status FROM app_agent_operations").fetchone()[0]
        assert receipt == "reserved"
    finally:
        state.store._conn.execute("DROP TRIGGER reject_agent_write")
    current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert current["steps"][0]["result"] == ""
    recovered = await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Recovered.")
    assert recovered["steps"][0]["result"] == "Recovered."


async def test_local_purge_erases_tasks_receipts_and_rejects_delayed_worker(state):
    await _create(state)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0] == 0
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Cannot resurrect.")


async def test_account_delete_cascades_agent_records(state):
    await _create(state)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DELETE FROM app_users WHERE user_id=?", (state.owner.user_id,))
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_tasks").fetchone()[0] == 0
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0] == 0
    assert state.store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_delete_terminal_history_erases_receipts_and_fences_delayed_writes(state):
    await _create(state)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    cancelled = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    cancelled = await _control(state, cancelled, "cancel")
    assert await state.store.agents.delete(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"],
    )
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0] == 0
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Cannot resurrect.")


async def test_delete_refuses_foreign_missing_active_and_stale_revision(state):
    task = await _create(state)
    for task_id in (task["id"], "missing"):
        with pytest.raises(AgentTaskAccessError, match="unavailable"):
            await state.store.agents.delete(
                user_id=state.other.user_id, task_id=task_id, expected_revision=task["revision"],
            )
    with pytest.raises(AgentTaskConflictError, match="Cancel active"):
        await state.store.agents.delete(
            user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"],
        )
    cancelled = await _control(state, task, "cancel")
    with pytest.raises(AgentTaskConflictError, match="changed"):
        await state.store.agents.delete(
            user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"],
        )
    assert await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == cancelled


async def test_saved_history_bound_and_deletion_free_capacity(state):
    first = None
    for number in range(200):
        task = await _create(state, request_id=f"saved-{number}")
        task = await _control(state, task, "cancel")
        if first is None:
            first = task
    with pytest.raises(AgentTaskError, match="capacity"):
        await _create(state, request_id="over-capacity")
    rerun = await _control(state, first, "rerun")
    assert rerun["id"] == first["id"]
    first = await _control(state, rerun, "cancel")
    await state.store.agents.delete(
        user_id=state.owner.user_id, task_id=first["id"], expected_revision=first["revision"],
    )
    replacement = await _create(state, request_id="over-capacity")
    assert replacement["status"] == "queued"
    assert len(await state.store.agents.list(user_id=state.owner.user_id, limit=200)) == 200


async def _terminal_task(state, completed_steps=1, status="failed", **changes):
    await _create(state, **changes)
    task = await state.store.agents.claim()
    for step_id in ("plan", "research", "review")[:completed_steps]:
        task = await _step(state, task, step_id, result=f"Saved {step_id}")
    if task["current_step"]:
        await state.store.agents.reserve_operation(**_lease(task), step_id=task["current_step"])
    state.clock.advance(2)
    if status == "cancelled":
        terminal = await _control(
            state,
            await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]),
            "cancel",
        )
    elif status == "completed":
        terminal = await state.store.agents.complete(**_lease(task), result="Previous report")
    else:
        terminal = await state.store.agents.fail(**_lease(task), error="Research stopped.")
    return task, terminal


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("completed_steps", [0, 1, 2])
async def test_rerun_keeps_the_same_task_and_all_completed_checkpoints(
    state, status, completed_steps
):
    _, terminal = await _terminal_task(state, completed_steps, status)
    receipts = [
        tuple(row)
        for row in state.store._conn.execute("SELECT * FROM app_agent_operations ORDER BY rowid")
    ]
    rerun = await _control(state, terminal, "rerun")
    assert rerun["id"] == terminal["id"] and rerun["created_at"] == terminal["created_at"]
    assert rerun["status"] == "queued" and rerun["error"] == ""
    assert rerun["steps"] == terminal["steps"]
    assert rerun["attempts"] == terminal["attempts"]
    assert len(await state.store.agents.list(user_id=state.owner.user_id)) == 1
    assert [
        tuple(row)
        for row in state.store._conn.execute("SELECT * FROM app_agent_operations ORDER BY rowid")
    ] == receipts
    claimed = await state.store.agents.claim()
    assert claimed["current_step"] == ("plan", "research", "review")[completed_steps]
    assert claimed["operations"] == completed_steps + 1
    assert claimed["active_seconds"] == 2
    assert claimed["attempts"] == 2


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_rerun_of_fully_saved_work_reuses_plan_and_refreshes_report(state, status):
    _, terminal = await _terminal_task(state, 3, status)
    rerun = await _control(state, terminal, "rerun")
    assert rerun["id"] == terminal["id"] and rerun["steps"][0] == terminal["steps"][0]
    assert rerun["result"] == "" and rerun["error"] == ""
    assert all(step["status"] == "pending" and step["result"] == "" for step in rerun["steps"][1:])
    claimed = await state.store.agents.claim()
    assert claimed["current_step"] == "research" and claimed["operations"] == 3
    assert claimed["active_seconds"] == 2 and claimed["attempts"] == 2
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_operations").fetchone()[0] == 3


async def test_rerun_rejects_foreign_missing_stale_and_duplicate_actions(state):
    _, terminal = await _terminal_task(state)
    for task_id in (terminal["id"], "missing"):
        with pytest.raises(AgentTaskAccessError):
            await _control(state, terminal, "rerun", task_id=task_id, user_id=state.other.user_id)
    results = await asyncio.gather(
        _control(state, terminal, "rerun"),
        _control(state, terminal, "rerun"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, AgentTaskConflictError) for result in results) == 1
    assert len(await state.store.agents.list(user_id=state.owner.user_id)) == 1


@pytest.mark.parametrize("status", ["queued", "running", "paused"])
async def test_rerun_does_not_restart_active_or_paused_work(state, status):
    task = await _create(state)
    if status != "queued":
        task = await state.store.agents.claim()
    if status == "paused":
        task = await _control(state, task, "pause")
    with pytest.raises(AgentTaskConflictError):
        await _control(state, task, "rerun")
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))[
        "status"
    ] == status


async def test_rerun_fences_every_delayed_old_worker_write(state):
    old, terminal = await _terminal_task(state)
    await _control(state, terminal, "rerun")
    new = await state.store.agents.claim()
    assert new["fence"] > old["fence"] and new["lease_token"] != old["lease_token"]
    for action in (
        state.store.agents.heartbeat(**_lease(old)),
        state.store.agents.reserve_operation(**_lease(old), step_id="research"),
        state.store.agents.checkpoint(**_lease(old), step_id="research", result="Late report"),
        state.store.agents.complete(**_lease(old), result="Late report"),
        state.store.agents.fail(**_lease(old), error="Late failure"),
    ):
        with pytest.raises(AgentTaskConflictError):
            await action
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=new["id"])
    assert saved["status"] == "running" and saved["steps"][0]["result"] == "Saved plan"


@pytest.mark.parametrize("limit", ["attempts", "operations", "time"])
async def test_rerun_rejects_exhausted_allowances_without_erasing_saved_work(state, limit):
    limits = (
        {"max_operations": 2}
        if limit == "operations"
        else {"max_active_seconds": 2}
        if limit == "time"
        else {}
    )
    _, terminal = await _terminal_task(state, limits=limits)
    with pytest.raises(AgentTaskError, match="allowance exhausted"):
        await _control(state, terminal, "rerun", max_attempts=1 if limit == "attempts" else 6)
    assert (
        await state.store.agents.get(user_id=state.owner.user_id, task_id=terminal["id"])
        == terminal
    )


@pytest.mark.parametrize("scope", ["owner", "global"])
async def test_rerun_obeys_current_queue_limits_and_retains_report_on_rejection(state, scope):
    _, terminal = await _terminal_task(state, 3, "completed")
    await _create(
        state,
        request_id="occupying-queue",
        user_id=state.other.user_id if scope == "global" else state.owner.user_id,
    )
    limits = {"max_queued_global": 1} if scope == "global" else {"max_queued_per_user": 1}
    with pytest.raises(AgentTaskError, match="capacity"):
        await _control(state, terminal, "rerun", admission_limits=limits)
    assert (
        await state.store.agents.get(user_id=state.owner.user_id, task_id=terminal["id"])
        == terminal
    )


async def test_rerun_persists_same_identity_checkpoints_and_cumulative_usage_after_restart(state):
    _, terminal = await _terminal_task(state)
    await _control(state, terminal, "rerun")
    path = state.store.path
    state.store.close()
    state.store = ApplicationStore(path)
    state.store.agents._clock = state.clock
    claimed = await state.store.agents.claim()
    assert claimed["id"] == terminal["id"] and claimed["current_step"] == "research"
    assert claimed["steps"][0]["result"] == "Saved plan"
    assert claimed["operations"] == 2 and claimed["active_seconds"] == 2
    assert claimed["attempts"] == 2
