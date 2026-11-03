"""Knowledge work explicitly starts one fenced Project research agent."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agent_projects import token_count
from audrey.app_state.agents import (
    AgentTaskAccessError,
    AgentTaskConflictError,
    AgentTaskContextError,
    AgentTaskError,
)
from audrey.content_sources import ContentSourceReader


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    project = await store.projects.create(user_id=owners[0].user_id, name="Release decisions")
    yield SimpleNamespace(
        store=store, owner=owners[0], other=owners[1], project=project,
        reader=ContentSourceReader(store, None, tmp_path),
    )
    store.close()


async def note(state, **changes):
    return await state.store.knowledge.create(**{
        "user_id": state.owner.user_id, "project_id": state.project.project_id,
        "kind": "action", "text": "Compare the public stable release support windows.", **changes,
    })


async def create(state, entry, **changes):
    return await state.store.agents.create(**{
        "user_id": state.owner.user_id, "project_id": state.project.project_id,
        "objective": "Compare the public support windows.", "request_id": "research-work",
        "knowledge_entry_id": entry["id"], "knowledge_entry_revision": entry["revision"], **changes,
    })


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def message(state, *, private=False):
    project = state.project
    if private:
        project = await state.store.projects.create(
            user_id=state.owner.user_id, name="Private source", private_required=True,
        )
    conversation = await state.store.conversations.create(
        user_id=state.owner.user_id, title="Source title", project_id=project.project_id,
        private_required=private,
    )
    mid = uuid.uuid4().hex
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,role,status,"
            "content,created_at,updated_at) VALUES (?,?,?,1,'user','completed',?,?,?)",
            (mid, conversation.conversation_id, state.owner.user_id,
             "SOURCE BODY: public support evidence.", "2026-11-02T12:00:00+00:00", "2026-11-02T12:00:00+00:00"),
        )
    document = await state.reader.resolve(state.owner.user_id, state.owner.storage_namespace, f"message:{mid}")
    return document, conversation.conversation_id


def ref(document):
    return {"source_id": document.source_id, "revision": document.revision}


@pytest.mark.parametrize("kind", ["action", "question"])
async def test_explicit_origin_stays_open_after_research_and_is_persisted_as_metadata(state, kind):
    entry = await note(state, kind=kind)
    created = await create(state, entry)
    assert created["project_context"]["origin_entry"] == {"id": entry["id"], "revision": 1}
    row = state.store._conn.execute("SELECT * FROM app_agent_project_contexts WHERE task_id=?", (created["id"],)).fetchone()
    pins = json.loads(row["knowledge_json"])
    assert pins == [{"id": entry["id"], "revision": 1, "chars": len(entry["text"]), "origin": True}]
    assert entry["text"] not in json.dumps(dict(row))

    async def execute(_task, step):
        return f"Saved {step}."

    assert await AgentRuntime(state.store.agents, execute).run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert saved["status"] == "completed" and saved["project_context"]["origin_entry"] == created["project_context"]["origin_entry"]
    assert await state.store.knowledge.get(user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"]) == entry


async def test_origin_precedes_pinned_relevant_notes_and_keeps_existing_context_bounds(state):
    entry = await note(state, text="Investigate unsupported legacy versions. " * 180)
    for index in range(15):
        await note(state, kind="decision", text=f"Pinned public support window {index}.", pinned=True)
    task = await create(state, entry)
    assert task["project_context"]["knowledge"][0] == {"id": entry["id"], "revision": 1}
    assert len(task["project_context"]["knowledge"]) <= 12
    claimed = await state.store.agents.claim()
    context = await state.store.agents.resolve_project_context(**lease(claimed))
    assert context["knowledge"][0]["id"] == entry["id"] and entry["text"].startswith(context["knowledge"][0]["text"])
    assert sum(token_count(item["text"]) for item in context["knowledge"]) <= 1000
    manifest = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone()[0]
    assert len(manifest) <= 4000
    assert sum(pin.get("origin") is True for pin in json.loads(manifest)) == 1


async def test_short_origin_and_eight_reference_notes_fit_twelve_pin_manifest(state):
    sources = [ref((await message(state))[0]) for _ in range(8)]
    entry = await note(state, sources=sources)
    for index in range(12):
        await note(state, kind="decision", text=f"Pinned decision {index}.", pinned=True, sources=sources)
    task = await create(state, entry)
    pins = task["project_context"]["knowledge"]
    assert len(pins) == 12 and pins[0] == {"id": entry["id"], "revision": 1}
    manifest = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone()[0]
    assert len(manifest) <= 4000 and json.loads(manifest)[0]["origin"] is True


@pytest.mark.parametrize("invalid", ["missing", "foreign", "other_project", "fact", "decision", "closed", "resolved", "proposed", "rejected", "superseded", "file", "mixed_file", "stale_source", "private_source"])
async def test_ineligible_origins_do_not_create_tasks_or_change_knowledge(state, invalid):
    changes = {}
    if invalid in {"foreign", "other_project"}:
        owner = state.other if invalid == "foreign" else state.owner
        project = await state.store.projects.create(user_id=owner.user_id, name="Other")
        changes.update(user_id=owner.user_id, project_id=project.project_id)
    if invalid in {"fact", "decision"}:
        changes["kind"] = invalid
    elif invalid in {"closed", "resolved"}:
        changes["kind"] = "action" if invalid == "closed" else "question"
    elif invalid in {"file", "mixed_file", "stale_source", "private_source"}:
        source, _ = await message(state, private=invalid == "private_source")
        changes["sources"] = [ref(source)]
        if invalid == "file":
            changes["sources"] = [{"source_id": "file:missing:text", "revision": "old"}]
        elif invalid == "mixed_file":
            changes["sources"].append({"source_id": "file:missing:text", "revision": "old"})
        elif invalid == "stale_source":
            changes["sources"][0]["revision"] = "old"
    entry = await note(state, **changes)
    if invalid in {"closed", "resolved"}:
        entry = await state.store.knowledge.update(
            user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1,
            work_status="completed" if invalid == "closed" else "resolved",
        )
    if invalid in {"proposed", "rejected", "superseded"}:
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_project_knowledge SET approval_state=? WHERE entry_id=?", (invalid, entry["id"]))
    if invalid == "missing":
        entry = {**entry, "id": "missing"}
    before = list(state.store._conn.execute("SELECT entry_id,revision,approval_state,work_status,text FROM app_project_knowledge"))
    with pytest.raises(AgentTaskAccessError):
        await create(state, entry)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_project_contexts").fetchone()[0] == 0
    assert list(state.store._conn.execute("SELECT entry_id,revision,approval_state,work_status,text FROM app_project_knowledge")) == before


async def test_private_project_origin_cannot_enter_public_research(state):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Private", private_required=True)
    entry = await note(state, project_id=project.project_id)
    with pytest.raises(AgentTaskAccessError):
        await create(state, entry, project_id=project.project_id)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []


@pytest.mark.parametrize("changes", [
    {"knowledge_entry_id": None}, {"knowledge_entry_revision": None}, {"project_id": None},
    {"knowledge_entry_id": " "}, {"knowledge_entry_id": "x" * 301},
    {"knowledge_entry_revision": True}, {"knowledge_entry_revision": 0},
    {"knowledge_entry_revision": 1.0}, {"knowledge_entry_revision": "1"},
])
async def test_origin_pair_is_strict_and_requires_project(state, changes):
    entry = await note(state)
    with pytest.raises(AgentTaskError):
        await create(state, entry, **changes)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []


async def test_origin_revision_is_checked_at_capture_and_failed_capture_rolls_back(state, monkeypatch):
    entry = await note(state)
    capture = state.store.agents._project_contexts.capture

    def changed_before_capture(**kwargs):
        state.store._conn.execute("UPDATE app_project_knowledge SET revision=revision+1 WHERE entry_id=?", (entry["id"],))
        return capture(**kwargs)

    monkeypatch.setattr(state.store.agents._project_contexts, "capture", changed_before_capture)
    with pytest.raises(AgentTaskConflictError, match="Project knowledge changed"):
        await create(state, entry)
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_project_contexts").fetchone()[0] == 0


@pytest.mark.parametrize("change", ["edit", "close", "delete", "source_edit", "source_detach"])
async def test_changed_origin_fences_dispatch_and_commit_and_replays_same_creation(state, change):
    source, cid = await message(state)
    entry = await note(state, sources=[ref(source)])
    task = await create(state, entry)
    claimed = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**lease(claimed), step_id="plan")
    if change == "delete":
        await state.store.knowledge.delete(user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1)
    elif change in {"edit", "close"}:
        await state.store.knowledge.update(
            user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1,
            **({"text": "Changed research objective."} if change == "edit" else {"work_status": "completed"}),
        )
    else:
        with state.store._lock, state.store._conn:
            if change == "source_edit":
                state.store._conn.execute("UPDATE app_messages SET content='Changed source',updated_at='later' WHERE message_id=?", (source.target["message_id"],))
            else:
                state.store._conn.execute("UPDATE app_conversations SET project_id=NULL WHERE conversation_id=?", (cid,))
    with pytest.raises(AgentTaskContextError):
        state.store.agents.fence_execution(**lease(claimed))
    for operation in (
        state.store.agents.checkpoint(**lease(claimed), step_id="plan", result="A late Plan."),
        state.store.agents.complete(**lease(claimed), result="A late report."),
        state.store.agents.request_input(**lease(claimed), question="A late question?"),
    ):
        with pytest.raises(AgentTaskContextError):
            await operation
    replay = await create(state, entry)
    assert replay["id"] == task["id"] and not replay["project_context"]["current"]
    assert replay["project_context"]["origin_entry"] == {"id": entry["id"], "revision": 1}
    assert len(await state.store.agents.list(user_id=state.owner.user_id)) == 1
    paused = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=replay["revision"], action="pause",
    )
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.control(
            user_id=state.owner.user_id, task_id=task["id"], expected_revision=paused["revision"], action="resume",
        )
    cancelled = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=paused["revision"], action="cancel",
    )
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.control(
            user_id=state.owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"], action="rerun",
        )


async def test_origin_and_source_proofs_survive_restart_without_text_copies(state):
    source, _ = await message(state)
    entry = await note(state, kind="question", sources=[ref(source)])
    task = await create(state, entry)
    second = ApplicationStore(state.store.path)
    try:
        assert await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == task
        claimed = await second.agents.claim()
        context = await second.agents.resolve_project_context(**lease(claimed))
        assert context["knowledge"][0]["text"] == entry["text"]
        assert "SOURCE BODY" not in json.dumps(context)
    finally:
        second.close()


async def test_approved_report_proposal_can_start_research_without_changing_the_proposal(state):
    source_task = await state.store.agents.create(
        user_id=state.owner.user_id, objective="Compare public release evidence.", request_id="source-report",
    )

    async def execute(_task, step):
        return f"SOURCE REPORT BODY: saved {step}."

    assert await AgentRuntime(state.store.agents, execute).run_once()
    source_task = await state.store.agents.get(user_id=state.owner.user_id, task_id=source_task["id"])
    source = await state.reader.resolve(state.owner.user_id, state.owner.storage_namespace, f"agent:{source_task['id']}:report")
    entry = await note(
        state, origin="agent_report", source_task_id=source_task["id"], source_task_revision=source_task["revision"],
        sources=[ref(source)],
    )
    with pytest.raises(AgentTaskAccessError):
        await create(state, entry)
    entry = await state.store.knowledge.decide(
        user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"], expected_revision=1,
        action="approve",
    )
    task = await create(state, entry)
    assert task["project_context"]["origin_entry"] == {"id": entry["id"], "revision": entry["revision"]}
    assert task["id"] != source_task["id"] and task["project_context"]["current"]
    assert await state.store.knowledge.get(user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"]) == entry


async def test_request_identity_includes_origin_pair_without_adopting_legacy_pins(state):
    first = await note(state)
    other = await note(state, kind="question", text="Which public release should we compare?")
    ordinary = await create(state, first, knowledge_entry_id=None, knowledge_entry_revision=None)
    assert "origin_entry" not in ordinary["project_context"]
    with pytest.raises(AgentTaskConflictError):
        await create(state, first)
    linked = await create(state, first, request_id="explicit")
    for changes in (
        {"knowledge_entry_id": None, "knowledge_entry_revision": None},
        {"knowledge_entry_id": other["id"], "knowledge_entry_revision": 1},
        {"knowledge_entry_revision": 2},
    ):
        with pytest.raises(AgentTaskConflictError):
            await create(state, first, request_id="explicit", **changes)
    assert (await create(state, first, request_id="explicit"))["id"] == linked["id"]
