"""Observed Agent metadata survives recovery without creating new authority."""

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
    AgentTaskContextError,
    AgentTaskError,
)


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 11, 3, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds=1):
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    users = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    clock = Clock()
    store.agents._clock = clock
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], clock=clock)
    store.close()


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def _claim(state, **changes):
    values = {"user_id": state.owner.user_id, "request_id": "task-1", "objective": "Prepare a household checklist."}
    values.update(changes)
    await state.store.agents.create(**values)
    return await state.store.agents.claim()


async def _events(state, task, **changes):
    values = {"user_id": state.owner.user_id, "task_id": task["id"]}
    values.update(changes)
    return await state.store.agents.events(**values)


async def _control(state, task, action):
    current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    return await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action=action,
    )


async def test_legacy_tasks_have_no_invented_events_and_public_records_stay_compact(state):
    task = await _claim(state)
    assert await _events(state, task) == {"items": [], "has_more": False, "truncated": False}
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    assert seq == 1
    for public in (
        await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]),
        (await state.store.agents.list(user_id=state.owner.user_id, summary=True))[0],
    ):
        assert "events" not in public


async def test_event_start_finish_is_metadata_only_stable_and_does_not_mutate_controls(state):
    task = await _claim(state)
    before = dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone())
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    event = (await _events(state, task))["items"][0]
    assert event == {
        "seq": 1, "attempt": 1, "stage": "plan", "kind": "model", "state": "started", "tool": None,
        "created_at": state.clock().isoformat(), "finished_at": None, "duration_ms": None,
        "input_tokens": None, "output_tokens": None,
    }
    state.clock.advance(2)
    await state.store.agents.finish_event(
        **_lease(task), seq=seq, duration_ms=1950, input_tokens=120, output_tokens=40,
    )
    finished = (await _events(state, task))["items"][0]
    assert finished["created_at"] == event["created_at"] and finished["seq"] == seq
    assert finished["finished_at"] == state.clock().isoformat()
    assert (finished["state"], finished["duration_ms"], finished["input_tokens"], finished["output_tokens"]) == (
        "completed", 1950, 120, 40,
    )
    assert dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone()) == before
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {"items": [], "unread_count": 0}
    # The revision read before either event is still a valid control revision.
    paused = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"], action="pause",
    )
    assert paused["status"] == "paused"


async def test_stage_events_commit_with_checkpoint_and_do_not_duplicate_model_reservations(state):
    task = await _claim(state)
    lease = _lease(task)
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    rows = (await _events(state, task))["items"]
    assert len(rows) == 1 and rows[0]["kind"] == "stage" and rows[0]["state"] == "started"
    state.clock.advance()
    await state.store.agents.checkpoint(**lease, step_id="plan", result="Use a concise checklist.")
    assert (await _events(state, task))["items"][0]["state"] == "completed"
    await state.store.agents.reserve_operation(**lease, step_id="research")
    current = (await _events(state, task))["items"][0]
    assert current["stage"] == "research" and current["seq"] == 2 and current["state"] == "started"


async def test_checkpoint_failure_rolls_back_stage_outcome(state):
    task = await _claim(state)
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    with state.store._conn:
        state.store._conn.execute(
            "CREATE TRIGGER reject_saved_task BEFORE UPDATE ON app_agent_tasks "
            "BEGIN SELECT RAISE(ABORT,'fixture failure'); END",
        )
    with pytest.raises(sqlite3.IntegrityError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Unsaved plan.")
    assert (await _events(state, task))["items"][0]["state"] == "started"


@pytest.mark.parametrize("action", ["pause", "cancel", "fail", "retry", "question"])
async def test_revoking_lease_interrupts_unfinished_events_and_rejects_callbacks(state, action):
    task = await _claim(state)
    lease = _lease(task)
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    seq = await state.store.agents.start_event(**lease, stage="plan", kind="model")
    if action in {"pause", "cancel"}:
        await _control(state, task, action)
    elif action == "question":
        await state.store.agents.request_input(**lease, question="How many rooms?")
    else:
        await state.store.agents.fail(**lease, error="Trusted fixture failure.", retry=action == "retry")
    page = await _events(state, task)
    assert all(row["state"] == "interrupted" and row["finished_at"] is not None for row in page["items"])
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.finish_event(**lease, seq=seq)
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.start_event(**lease, stage="plan", kind="model")
    assert await _events(state, task) == page


async def test_expired_events_survive_restart_as_unknown_and_new_claim_interrupts_them(state):
    task = await _claim(state)
    first = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    state.clock.advance(61)
    reopened = ApplicationStore(state.store.path)
    reopened.agents._clock = state.clock
    try:
        history = await reopened.agents.events(user_id=state.owner.user_id, task_id=task["id"])
        assert history["items"][0]["state"] == "started"
        current = await reopened.agents.claim()
        assert current["attempts"] == 2 and current["fence"] > task["fence"]
        history = await reopened.agents.events(user_id=state.owner.user_id, task_id=task["id"])
        assert history["items"][0]["state"] == "interrupted"
        second = await reopened.agents.start_event(**_lease(current), stage="plan", kind="model")
        assert second == first + 1
        history = await reopened.agents.events(user_id=state.owner.user_id, task_id=task["id"])
        assert [row["attempt"] for row in history["items"]] == [2, 1]
        with pytest.raises(AgentTaskConflictError):
            await reopened.agents.finish_event(**_lease(task), seq=first)
    finally:
        reopened.close()


async def test_completed_history_and_same_task_rerun_preserve_monotonic_sequences(state):
    task = await _claim(state)
    lease = _lease(task)
    for stage in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=stage)
        await state.store.agents.checkpoint(**lease, step_id=stage, result=f"Saved {stage}.")
    done = await state.store.agents.complete(**lease, result="Checklist complete.")
    saved = await _events(state, task)
    assert len(saved["items"]) == 3 and all(row["state"] == "completed" for row in saved["items"])
    await _control(state, done, "rerun")
    current = await state.store.agents.claim()
    assert current["id"] == task["id"] and current["current_step"] == "research"
    await state.store.agents.reserve_operation(**_lease(current), step_id="research")
    history = await _events(state, task)
    assert history["items"][1:] == saved["items"]
    assert history["items"][0]["seq"] == 4 and history["items"][0]["attempt"] == 2


async def test_retention_pagination_and_late_pruned_finish_do_not_fail_useful_work(state):
    task = await _claim(state, limits={"max_operations": 1000})
    lease = _lease(task)
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    stage_seq = (await _events(state, task))["items"][0]["seq"]
    first = await state.store.agents.start_event(**lease, stage="plan", kind="model")
    for _ in range(204):
        seq = await state.store.agents.start_event(**lease, stage="plan", kind="model")
        await state.store.agents.finish_event(**lease, seq=seq, duration_ms=1)
    page = await _events(state, task, limit=50)
    assert page["truncated"] is True and page["has_more"] is True
    assert len(page["items"]) == 50 and page["items"][0]["seq"] == 206
    before = page["items"][-1]["seq"]
    rest = []
    while page["has_more"]:
        page = await _events(state, task, before=before, limit=50)
        rest.extend(page["items"])
        before = page["items"][-1]["seq"]
    assert len(rest) == 150 and rest[-1]["seq"] == 7
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == 200
    await state.store.agents.finish_event(**lease, seq=first)
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    assert not any(row["kind"] == "stage" for row in (await _events(state, task))["items"])
    await state.store.agents.checkpoint(**lease, step_id="plan", result="The retained history cap does not stop work.")
    assert stage_seq == 1


async def test_concurrent_event_starts_allocate_unique_per_task_sequences(state):
    task = await _claim(state)
    sequences = await asyncio.gather(*[
        state.store.agents.start_event(**_lease(task), stage="plan", kind="model") for _ in range(20)
    ])
    assert sorted(sequences) == list(range(1, 21))
    other = await _claim(state, user_id=state.other.user_id, request_id="other-task")
    assert await state.store.agents.start_event(**_lease(other), stage="plan", kind="model") == 1


async def test_owner_and_response_fences_reject_foreign_mutated_and_removed_rows(state):
    task = await _claim(state)
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    page = await _events(state, task)
    with pytest.raises(AgentTaskAccessError):
        await _events(state, task, user_id=state.other.user_id)
    with pytest.raises(AgentTaskAccessError):
        state.store.agents.fence_events_response(user_id=state.other.user_id, task_id=task["id"], page=page)
    await state.store.agents.finish_event(**_lease(task), seq=seq)
    with pytest.raises(AgentTaskConflictError):
        state.store.agents.fence_events_response(user_id=state.owner.user_id, task_id=task["id"], page=page)
    page = await _events(state, task)
    for _ in range(200):
        await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    with pytest.raises(AgentTaskConflictError):
        state.store.agents.fence_events_response(user_id=state.owner.user_id, task_id=task["id"], page=page)


@pytest.mark.parametrize("remove", ["task", "account"])
async def test_deletion_erases_events_and_invalidates_saved_pages(state, remove):
    task = await _claim(state)
    await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    page = await _events(state, task)
    if remove == "account":
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    else:
        current = await _control(state, task, "cancel")
        await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == 0
    with pytest.raises(AgentTaskAccessError):
        state.store.agents.fence_events_response(user_id=state.owner.user_id, task_id=task["id"], page=page)


@pytest.mark.parametrize("values", [
    {"stage": "secret input", "kind": "model"}, {"stage": [], "kind": "model"},
    {"stage": "plan", "kind": {}}, {"stage": "title", "kind": "stage"},
    {"stage": "plan", "kind": "tool", "tool": "web_fetch"},
    {"stage": "research", "kind": "tool", "tool": "secret input"},
    {"stage": "research", "kind": "tool", "tool": []},
    {"stage": "plan", "kind": "model", "tool": "web_fetch"},
])
async def test_only_known_metadata_enums_are_accepted(state, values):
    task = await _claim(state)
    with pytest.raises(AgentTaskError):
        await state.store.agents.start_event(**_lease(task), **values)
    assert (await _events(state, task))["items"] == []


@pytest.mark.parametrize("values", [
    {"seq": True}, {"seq": 0}, {"state": "interrupted"}, {"state": {}},
    {"duration_ms": -1}, {"duration_ms": True}, {"input_tokens": 2.5},
    {"input_tokens": False}, {"output_tokens": -1}, {"output_tokens": 2**63},
])
async def test_invalid_outcomes_and_counts_are_rejected_without_finishing(state, values):
    task = await _claim(state)
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    with pytest.raises(AgentTaskError):
        await state.store.agents.finish_event(**_lease(task), **{"seq": seq, **values})
    assert (await _events(state, task))["items"][0]["state"] == "started"


async def test_tool_metadata_follows_live_stages_and_non_models_cannot_claim_token_usage(state):
    task = await _claim(state)
    lease = _lease(task)
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.start_event(**lease, stage="research", kind="tool", tool="web_fetch")
    await state.store.agents.reserve_operation(**lease, step_id="plan")
    await state.store.agents.checkpoint(**lease, step_id="plan", result="Make a checklist file.")
    seq = await state.store.agents.start_event(**lease, stage="research", kind="tool", tool="create_file")
    with pytest.raises(AgentTaskError):
        await state.store.agents.finish_event(**lease, seq=seq, output_tokens=3)
    await state.store.agents.finish_event(**lease, seq=seq, state="failed", duration_ms=12)
    first = await _events(state, task)
    await state.store.agents.finish_event(**lease, seq=seq, state="failed", duration_ms=12)
    assert await _events(state, task) == first
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.finish_event(**lease, seq=seq, state="completed")


async def test_review_records_native_tools_and_refuses_web_activity(state):
    task = await _claim(state)
    lease = _lease(task)
    for stage in ("plan", "research"):
        await state.store.agents.reserve_operation(**lease, step_id=stage)
        await state.store.agents.checkpoint(**lease, step_id=stage, result="Saved work.")
    for tool in ("read_selected_file", "create_file"):
        seq = await state.store.agents.start_event(**lease, stage="review", kind="tool", tool=tool)
        await state.store.agents.finish_event(**lease, seq=seq, duration_ms=4)
    before = await _events(state, task)
    assert [(item["stage"], item["tool"]) for item in before["items"][:2]] == [
        ("review", "create_file"), ("review", "read_selected_file"),
    ]
    for tool in ("web_search", "web_fetch"):
        with pytest.raises(AgentTaskError):
            await state.store.agents.start_event(**lease, stage="review", kind="tool", tool=tool)
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.start_event(**lease, stage="research", kind="tool", tool="create_file")
    assert await _events(state, task) == before


async def test_budget_and_inactive_owner_fence_event_callbacks(state):
    task = await _claim(state, limits={"max_active_seconds": 2})
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    state.clock.advance(2)
    with pytest.raises(AgentTaskError):
        await state.store.agents.finish_event(**_lease(task), seq=seq)
    with pytest.raises(AgentTaskError):
        await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    state.clock.advance(-2)
    with state.store._conn:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.finish_event(**_lease(task), seq=seq)


async def test_changed_project_inputs_fence_callbacks_but_not_history_or_cleanup(state):
    project = await state.store.projects.create(
        user_id=state.owner.user_id, name="Household", instructions="Use a compact checklist.",
    )
    task = await _claim(state, project_id=project.project_id)
    seq = await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    page = await _events(state, task)
    await state.store.projects.update(
        user_id=state.owner.user_id, project_id=project.project_id, instructions="Use a different scope.",
    )
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.finish_event(**_lease(task), seq=seq)
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.start_event(**_lease(task), stage="plan", kind="model")
    assert await _events(state, task) == page
    state.store.agents.fence_events_response(user_id=state.owner.user_id, task_id=task["id"], page=page)
    await state.store.agents.fail(**_lease(task), error="Selected context changed.")
    assert (await _events(state, task))["items"][0]["state"] == "interrupted"


@pytest.mark.parametrize("values", [{"limit": True}, {"limit": 0}, {"limit": 101}, {"before": True}, {"before": -1}])
async def test_history_pagination_rejects_invalid_bounds(state, values):
    task = await _claim(state)
    with pytest.raises(AgentTaskError):
        await _events(state, task, **values)


async def test_schema_upgrade_keeps_saved_tasks_and_enforces_owner_foreign_key(state):
    task = await _claim(state)
    with state.store._conn:
        state.store._conn.execute("DROP TABLE app_agent_events")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version>=36")
    upgraded = ApplicationStore(state.store.path)
    upgraded.agents._clock = state.clock
    try:
        assert upgraded.schema_version == 39
        assert (await upgraded.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["objective"] == task["objective"]
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        with upgraded._conn, pytest.raises(sqlite3.IntegrityError):
            upgraded._conn.execute(
                "INSERT INTO app_agent_events(task_id,user_id,seq,attempt,fence,stage,kind,state,created_at) "
                "VALUES (?,?,1,1,1,'plan','model','started',?)",
                (task["id"], state.other.user_id, state.clock().isoformat()),
            )
    finally:
        upgraded.close()
