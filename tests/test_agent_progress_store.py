"""Owned generated Work drafts persist without changing stage or source authority."""

import json

import pytest
from test_agent_store import _create, _lease, _step
from test_agent_store import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import (
    AgentTaskAccessError,
    AgentTaskConflictError,
    AgentTaskContextError,
    AgentTaskError,
)


async def _work(state, **changes):
    await _create(state, **changes)
    task = await state.store.agents.claim()
    task = await _step(state, task, "plan", result="The saved Plan.")
    await state.store.agents.reserve_operation(**_lease(task), step_id="research")
    return task


async def _get(state, task):
    return await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])


async def _act(state, task, action):
    current = await _get(state, task)
    return await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=current["revision"], action=action)


async def test_progress_replaces_latest_note_idempotently_without_completing_or_committing_work(state):
    task = await _work(state)
    before = await _get(state, task)
    calls = []

    def guard():
        assert state.store._conn.in_transaction
        calls.append(True)

    progress = await state.store.agents.save_work_progress(**_lease(task), note="  Draft outline saved. \n", scope_guard=guard)
    assert progress == {"note": "Draft outline saved.", "saved_at": state.clock().isoformat()}
    saved = await _get(state, task)
    assert saved["revision"] == before["revision"] + 1
    assert saved["steps"][0] == before["steps"][0]
    assert saved["steps"][1]["status"] == "running" and saved["steps"][1]["result"] == ""
    assert saved["steps"][1]["progress"] == progress and saved["status"] == "running"
    row = dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone())
    operations = [tuple(row) for row in state.store._conn.execute("SELECT step_id,status FROM app_agent_operations ORDER BY step_id")]
    assert operations == [("plan", "committed"), ("research", "reserved")]
    assert row["operations"] == 2
    state.clock.advance(1)
    assert await state.store.agents.save_work_progress(**_lease(task), note="Draft outline saved.", scope_guard=guard) == progress
    assert dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone()) == row
    state.clock.advance(1)
    latest = await state.store.agents.save_work_progress(**_lease(task), note="First section drafted.", scope_guard=guard)
    assert latest["saved_at"] == state.clock().isoformat() and latest != progress
    saved = await _get(state, task)
    assert saved["revision"] == before["revision"] + 2 and saved["steps"][1]["progress"] == latest
    assert "Draft outline saved." not in state.store._conn.execute("SELECT steps_json FROM app_agent_tasks").fetchone()[0]
    assert calls == [True, True, True]
    assert [tuple(row) for row in state.store._conn.execute("SELECT step_id,status FROM app_agent_operations ORDER BY step_id")] == operations


async def test_identical_note_still_runs_final_scope_guard_and_rolls_back_on_rejection(state):
    task = await _work(state)
    await state.store.agents.save_work_progress(**_lease(task), note="One saved draft.")
    before = dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone())

    def expired():
        assert state.store._conn.in_transaction
        raise AgentDispatchStoppedError("The progress scope expired.")

    for note in ("One saved draft.", "A new draft."):
        with pytest.raises(AgentDispatchStoppedError):
            await state.store.agents.save_work_progress(**_lease(task), note=note, scope_guard=expired)
        assert dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone()) == before


@pytest.mark.parametrize("note", ["", " \n\t", "a" * 4001, "é" * 2001, "\ud800", None, 42])
async def test_progress_requires_nonblank_bounded_utf8_text(state, note):
    task = await _work(state)
    before = await _get(state, task)
    with pytest.raises(AgentTaskError):
        await state.store.agents.save_work_progress(**_lease(task), note=note)
    assert await _get(state, task) == before


async def test_progress_accepts_exact_utf8_boundary_and_never_saves_a_note_on_other_stages(state):
    await _create(state)
    task = await state.store.agents.claim()
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.save_work_progress(**_lease(task), note="Too early.")
    task = await _step(state, task, "plan")
    progress = await state.store.agents.save_work_progress(**_lease(task), note="é" * 2000)
    assert len(progress["note"].encode("utf-8")) == 4000
    task = await _step(state, task, "research")
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.save_work_progress(**_lease(task), note="Too late.")
    assert (await _get(state, task))["steps"][1]["progress"] == progress


async def test_progress_survives_pause_question_restart_and_unfinished_rerun_but_finished_rerun_clears_it(state):
    task = await _work(state)
    note = await state.store.agents.save_work_progress(**_lease(task), note="Outline drafted; waiting for the chosen tone.")
    await _act(state, task, "pause")
    assert (await _get(state, task))["steps"][1]["progress"] == note
    await _act(state, task, "resume")
    task = await state.store.agents.claim()
    waiting = await state.store.agents.request_input(**_lease(task), step_id="research", question="Formal or casual?")
    assert waiting["steps"][1]["progress"] == note
    second = ApplicationStore(state.store.path)
    second.agents._clock = state.clock
    try:
        assert (await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["steps"][1]["progress"] == note
        await second.agents.answer_input(user_id=state.owner.user_id, task_id=task["id"],
            request_id=waiting["input_requests"][0]["id"], expected_revision=waiting["revision"], answer="Casual")
        task = await second.agents.claim()
        assert task["steps"][1]["progress"] == note
        failed = await second.agents.fail(**_lease(task), error="A safe interrupted outcome.")
        await second.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=failed["revision"], action="rerun")
        task = await second.agents.claim()
        assert task["steps"][1]["progress"] == note and task["current_step"] == "research"
        for stage in ("research", "review"):
            await second.agents.reserve_operation(**_lease(task), step_id=stage)
            task = await second.agents.checkpoint(**_lease(task), step_id=stage, result=f"Saved {stage}.")
        finished = await second.agents.complete(**_lease(task), result="The final report.")
        assert finished["steps"][1]["progress"] == note
        queued = await second.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=finished["revision"], action="rerun")
        assert queued["steps"][0] == finished["steps"][0]
        assert "progress" not in queued["steps"][1] and queued["steps"][1]["status"] == "pending"
    finally:
        second.close()


async def test_progress_revisions_refresh_full_responses_and_both_summary_paths_omit_note_body(state):
    task = await _work(state)
    await state.store.agents.save_work_progress(**_lease(task), note="First working draft marker.")
    captured = await _get(state, task)
    summary = (await state.store.agents.list(user_id=state.owner.user_id, summary=True))[0]
    assert all("progress" not in step for step in summary["steps"])
    await state.store.agents.save_work_progress(**_lease(task), note="Latest working draft marker.")
    with pytest.raises(AgentTaskConflictError):
        state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[captured])
    refreshed = state.store.agents.response_snapshot(user_id=state.owner.user_id, tasks=[captured])[0]
    assert refreshed["steps"][1]["progress"]["note"] == "Latest working draft marker."
    refreshed_summary = state.store.agents.response_snapshot(user_id=state.owner.user_id, tasks=[summary])[0]
    assert all("progress" not in step for step in refreshed_summary["steps"])
    assert "working draft marker" not in json.dumps(refreshed_summary)
    assert refreshed_summary["revision"] == refreshed["revision"]


@pytest.mark.parametrize("change", ["lease", "time", "source", "owner"])
async def test_progress_revalidates_current_task_limits_owner_and_sources(state, change):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Draft inputs", instructions="Use three sections.")
    task = await _work(state, project_id=project.project_id)
    progress = await state.store.agents.save_work_progress(**_lease(task), note="The first section is drafted.")
    if change == "lease":
        await _act(state, task, "pause")
    elif change == "time":
        with state.store._conn:
            state.store._conn.execute("UPDATE app_agent_tasks SET active_seconds=1800 WHERE task_id=?", (task["id"],))
    elif change == "source":
        await state.store.projects.update(user_id=state.owner.user_id, project_id=project.project_id, instructions="A changed scope.")
    else:
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    error = AgentTaskAccessError if change == "owner" else AgentTaskContextError if change == "source" else AgentTaskError
    with pytest.raises(error):
        await state.store.agents.save_work_progress(**_lease(task), note="A late changed note.")
    if change != "owner":
        retained = await _get(state, task)
        assert retained["steps"][1]["progress"] == progress
        if change == "source":
            assert retained["project_context"]["current"] is False


@pytest.mark.parametrize("remove", ["task", "account"])
async def test_deletion_erases_saved_progress_body(state, remove):
    task = await _work(state)
    await state.store.agents.save_work_progress(**_lease(task), note="A disposable working draft marker.")
    if remove == "account":
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    else:
        cancelled = await _act(state, task, "cancel")
        await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"])
    assert state.store._conn.execute("SELECT steps_json FROM app_agent_tasks WHERE task_id=?", (task["id"],)).fetchone() is None
    assert not state.store._conn.execute("PRAGMA foreign_key_check").fetchall()
