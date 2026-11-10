"""Project research freezes identities, never copies canonical input bodies."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agent_projects import token_count
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskContextError


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(provider="test", subject="alice", email="alice@example.com", display_name="Alice", role="user", auth_method="test")
    project = await store.projects.create(user_id=owner.user_id, name="Public releases", instructions="Use stable releases only.")
    yield SimpleNamespace(store=store, owner=owner, project=project)
    store.close()


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def create(state, **changes):
    return await state.store.agents.create(**{**dict(user_id=state.owner.user_id, objective="Compare public releases.", request_id="task", project_id=state.project.project_id), **changes})


async def note(state, **changes):
    return await state.store.knowledge.create(**{**dict(user_id=state.owner.user_id, project_id=state.project.project_id, kind="decision", text="Only stable releases count."), **changes})


async def test_manifest_contains_only_pins_and_survives_restart_without_input_copies(state):
    entry = await note(state)
    task = await create(state)
    row = dict(state.store._conn.execute("SELECT * FROM app_agent_project_contexts").fetchone())
    assert row["project_id"] == state.project.project_id
    assert state.project.instructions not in json.dumps(row) and entry["text"] not in json.dumps(row)
    assert json.loads(row["knowledge_json"])[0]["id"] == entry["id"]
    second = ApplicationStore(state.store.path)
    try:
        saved = await second.agents.get(user_id=state.owner.user_id, task_id=task["id"])
        assert saved == task
        claimed = await second.agents.claim()
        context = await second.agents.resolve_project_context(**lease(claimed))
        assert context["instructions"] == state.project.instructions and context["knowledge"][0]["text"] == entry["text"]
    finally:
        second.close()


async def test_context_selects_only_approved_sourceless_notes_and_bounds_text(state):
    await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id, instructions="stable " * 570)
    for index in range(15):
        await note(state, text=f"Release {index}. " + "stable releases " * 300)
    linked = await note(state, text="Linked source must not become an alternate file read.", pinned=True,
        sources=[{"source_id": "file:missing:text", "revision": "old"}])
    task = await create(state)
    claimed = await state.store.agents.claim()
    context = await state.store.agents.resolve_project_context(**lease(claimed))
    assert 0 < len(context["knowledge"]) <= 12 and linked["id"] not in {pin["id"] for pin in task["project_context"]["knowledge"]}
    assert sum(token_count(entry["text"]) for entry in context["knowledge"]) <= 1000
    assert token_count(context["instructions"]) <= 2000


async def test_rename_added_note_and_unselected_file_membership_leave_frozen_context_valid(state):
    selected = await note(state, pinned=True)
    task = await create(state)
    await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id, name="Renamed")
    await note(state, text="A later note is not silently adopted.", pinned=True)
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id="future-file")
    current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert current["project_context"]["current"] and current["project_context"]["project_name"] == "Renamed"
    assert current["project_context"]["knowledge"] == [{"id": selected["id"], "revision": 1}]
    assert await state.store.agents.claim() is not None


@pytest.mark.parametrize("change", ["instructions", "note", "delete_note", "sources", "project"])
async def test_context_changes_fence_dispatch_checkpoints_and_completion_but_allow_cleanup(state, change):
    entry = await note(state)
    task = await create(state)
    claimed = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**lease(claimed), step_id="plan")
    if change == "project":
        await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    elif change == "instructions":
        await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id, instructions="A changed scope.")
    elif change == "delete_note":
        await state.store.knowledge.delete(user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1)
    else:
        changes = {"text": "A new decision."} if change == "note" else {"sources": [{"source_id": "file:changed:text", "revision": "new"}]}
        await state.store.knowledge.update(user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1, **changes)
    for call in (
        state.store.agents.validate_lease(**lease(claimed)),
        state.store.agents.checkpoint(**lease(claimed), step_id="plan", result="A late saved Plan."),
        state.store.agents.request_input(**lease(claimed), question="A late question?"),
        state.store.agents.complete(**lease(claimed), result="A late report."),
    ):
        with pytest.raises(AgentTaskContextError):
            await call
    with pytest.raises(AgentTaskContextError):
        state.store.agents.fence_execution(**lease(claimed))
    failed = await state.store.agents.fail(**lease(claimed), error="Project context changed.")
    assert failed["id"] == task["id"] and not failed["project_context"]["current"]
    assert await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=failed["revision"])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_project_contexts").fetchone()[0] == 0


async def test_invalid_queued_project_context_creates_failure_notice_without_dispatch(state):
    task = await create(state)
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    assert await state.store.agents.claim() is None
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == "failed" and "Project context changed" in saved["error"] and saved["attempts"] == 0
    assert (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]["kind"] == "failure"


async def test_saved_plan_rerun_keeps_context_and_usage_while_new_project_notes_are_not_adopted(state):
    entry = await note(state)
    task = await create(state)

    async def execute(_task, step):
        return f"Saved {step}."

    assert await AgentRuntime(state.store.agents, execute).run_once()
    completed = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    await note(state, text="A later unselected note.")
    queued = await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=completed["revision"], action="rerun")
    assert queued["steps"][0] == completed["steps"][0] and queued["attempts"] == completed["attempts"]
    assert queued["project_context"]["knowledge"] == [{"id": entry["id"], "revision": 1}]


async def test_project_creation_rejection_rolls_back_task_and_account_purge_erases_manifest(state):
    private = await state.store.projects.create(user_id=state.owner.user_id, name="Protected", private_required=True)
    with pytest.raises(AgentTaskAccessError):
        await create(state, project_id=private.project_id)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    await create(state)
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DELETE FROM app_users WHERE user_id=?", (state.owner.user_id,))
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_project_contexts").fetchone()[0] == 0


async def test_schema_32_preserves_legacy_tasks_without_implicitly_adopting_project_context(state):
    task = await create(state, project_id=None)
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DROP TABLE app_agent_project_contexts")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=32")
    upgraded = ApplicationStore(state.store.path)
    try:
        assert upgraded.schema_version == 40
        assert await upgraded.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == task
        assert upgraded._conn.execute("SELECT COUNT(*) FROM app_agent_project_contexts").fetchone()[0] == 0
    finally:
        upgraded.close()
