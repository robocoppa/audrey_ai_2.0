"""Ordinary Agent notes pin canonical DB provenance without copying its text."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskContextError
from audrey.content_sources import ContentSourceReader

STAMP = "2026-11-02T12:00:00+00:00"


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com",
        display_name=name, role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    project = await store.projects.create(user_id=owners[0].user_id, name="Stable releases")
    reader = ContentSourceReader(store, None, tmp_path)
    yield SimpleNamespace(store=store, owner=owners[0], other=owners[1], project=project, reader=reader)
    store.close()


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def message(state, *, owner=None, project=None, private=False, status="completed", text="SOURCE BODY: stable release evidence."):
    owner = owner or state.owner
    project = project or state.project
    conversation = await state.store.conversations.create(
        user_id=owner.user_id, title="SOURCE TITLE", project_id=project.project_id,
        private_required=private,
    )
    mid = f"msg_{uuid.uuid4().hex}"
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,"
            "role,status,content,created_at,updated_at) VALUES (?,?,?,1,'user',?,?,?,?)",
            (mid, conversation.conversation_id, owner.user_id, status, text, STAMP, STAMP),
        )
    source = await state.reader.resolve(owner.user_id, owner.storage_namespace, f"message:{mid}")
    return source, conversation.conversation_id


async def report(state, *, owner=None, project=None):
    owner = owner or state.owner
    task = await state.store.agents.create(
        user_id=owner.user_id, objective="SOURCE OBJECTIVE: compare stable releases.",
        request_id=uuid.uuid4().hex, project_id=project.project_id if project else None,
    )

    async def execute(_task, step):
        return f"SOURCE REPORT BODY: saved {step}."

    assert await AgentRuntime(state.store.agents, execute).run_once()
    source = await state.reader.resolve(owner.user_id, owner.storage_namespace, f"agent:{task['id']}:report")
    return source, await state.store.agents.get(user_id=owner.user_id, task_id=task["id"])


def ref(source):
    return {"source_id": source.source_id, "revision": str(source.revision)}


async def note(state, sources, **changes):
    return await state.store.knowledge.create(**{
        "user_id": state.owner.user_id, "project_id": state.project.project_id,
        "kind": "decision", "text": "APPROVED NOTE: stable releases count.",
        "sources": sources, **changes,
    })


async def create(state):
    return await state.store.agents.create(
        user_id=state.owner.user_id, objective="Compare stable releases.",
        request_id=uuid.uuid4().hex, project_id=state.project.project_id,
    )


@pytest.mark.parametrize("kind", ["message", "report", "mixed", "incomplete_message"])
async def test_current_sources_include_only_note_text_and_restart_safe_hash_proof(state, kind):
    source, _ = await message(state, status="incomplete" if kind == "incomplete_message" else "completed")
    report_source, _ = await report(state)
    sources = [ref(source)] if kind != "report" else [ref(report_source)]
    if kind == "mixed":
        sources.append(ref(report_source))
    entry = await note(state, sources)
    task = await create(state)
    assert task["project_context"]["knowledge"] == [{"id": entry["id"], "revision": 1}]
    manifest = dict(state.store._conn.execute("SELECT * FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone())
    pin = json.loads(manifest["knowledge_json"])[0]
    assert set(pin) == {"id", "revision", "chars", "sources_sha256"}
    assert len(pin["sources_sha256"]) == 64 and len(manifest["knowledge_json"]) <= 4000
    for marker in ("SOURCE BODY", "SOURCE TITLE", "SOURCE OBJECTIVE", "SOURCE REPORT BODY", "APPROVED NOTE", source.source_id, report_source.source_id):
        assert marker not in json.dumps(manifest)
    second = ApplicationStore(state.store.path)
    try:
        assert await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == task
        claimed = await second.agents.claim()
        context = await second.agents.resolve_project_context(**lease(claimed))
        assert context["knowledge"][0]["text"] == entry["text"]
        assert "SOURCE BODY" not in json.dumps(context) and "SOURCE REPORT BODY" not in json.dumps(context)
    finally:
        second.close()


@pytest.mark.parametrize("invalid", ["stale", "foreign", "other_project", "private", "file", "mixed_file", "unfinished_report", "empty_report", "oversized", "malformed"])
async def test_invalid_or_unsupported_sources_are_not_selected(state, invalid):
    if invalid in {"unfinished_report", "empty_report"}:
        source, task = await report(state)
        with state.store._lock, state.store._conn:
            state.store._conn.execute(
                "UPDATE app_agent_tasks SET status=?,result=? WHERE task_id=?",
                ("failed" if invalid == "unfinished_report" else "completed", source.text if invalid == "unfinished_report" else "", task["id"]),
            )
    else:
        owner, project = state.owner, state.project
        if invalid == "foreign":
            owner = state.other
            project = await state.store.projects.create(user_id=owner.user_id, name="Foreign")
        elif invalid in {"other_project", "private"}:
            project = await state.store.projects.create(user_id=owner.user_id, name="Other", private_required=invalid == "private")
        source, _ = await message(state, owner=owner, project=project, private=invalid == "private")
    sources = [ref(source)]
    if invalid == "stale":
        sources[0]["revision"] = "old"
    elif invalid in {"file", "mixed_file"}:
        sources = ([*sources] if invalid == "mixed_file" else []) + [{"source_id": "file:missing:text", "revision": "old"}]
    entry = await note(state, sources, pinned=True)
    if invalid in {"oversized", "malformed"}:
        with state.store._lock, state.store._conn:
            if invalid == "oversized":
                state.store._conn.execute("UPDATE app_messages SET content=? WHERE message_id=?", ("x" * 500_001, source.target["message_id"]))
            else:
                state.store._conn.execute("UPDATE app_project_knowledge SET sources_json=? WHERE entry_id=?", ('[{"source_id":42,"revision":"x"}]', entry["id"]))
    fallback = await note(state, [], text="A valid sourceless fallback.")
    task = await create(state)
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]


@pytest.mark.parametrize("change", ["edit", "delete", "detach", "archive", "title", "privacy", "role", "status", "report_rerun", "report_delete", "report_status", "report_empty"])
async def test_source_changes_fence_every_saved_write_and_final_dispatch(state, change):
    if change.startswith("report"):
        source, source_task = await report(state)
    else:
        source, cid = await message(state)
    await note(state, [ref(source)])
    task = await create(state)
    claimed = await state.store.agents.claim()
    scope = lease(claimed)
    if change == "report_rerun":
        await state.store.agents.control(user_id=state.owner.user_id, task_id=source_task["id"], expected_revision=source_task["revision"], action="rerun")
    elif change == "report_delete":
        await state.store.agents.delete(user_id=state.owner.user_id, task_id=source_task["id"], expected_revision=source_task["revision"])
    else:
        sql, parameters = {
            "edit": ("UPDATE app_messages SET content='Edited source',updated_at='later' WHERE message_id=?", (source.target.get("message_id"),)),
            "delete": ("DELETE FROM app_messages WHERE message_id=?", (source.target.get("message_id"),)),
            "detach": ("UPDATE app_conversations SET project_id=NULL WHERE conversation_id=?", (locals().get("cid"),)),
            "archive": ("UPDATE app_conversations SET archived_at='later' WHERE conversation_id=?", (locals().get("cid"),)),
            "title": ("UPDATE app_conversations SET title='Renamed' WHERE conversation_id=?", (locals().get("cid"),)),
            "privacy": ("UPDATE app_conversations SET project_id=NULL,private_required=1 WHERE conversation_id=?", (locals().get("cid"),)),
            "role": ("UPDATE app_messages SET role='tool' WHERE message_id=?", (source.target.get("message_id"),)),
            "status": ("UPDATE app_messages SET status='in_progress' WHERE message_id=?", (source.target.get("message_id"),)),
            "report_status": ("UPDATE app_agent_tasks SET status='failed' WHERE task_id=?", (source.target.get("task_id"),)),
            "report_empty": ("UPDATE app_agent_tasks SET result='' WHERE task_id=?", (source.target.get("task_id"),)),
        }[change]
        with state.store._lock, state.store._conn:
            state.store._conn.execute(sql, parameters)
    for call in (
        state.store.agents.validate_lease(**scope),
        state.store.agents.reserve_operation(**scope, step_id="plan"),
        state.store.agents.checkpoint(**scope, step_id="plan", result="Late Plan."),
        state.store.agents.request_input(**scope, question="Late question?"),
        state.store.agents.complete(**scope, result="Late report."),
    ):
        with pytest.raises(AgentTaskContextError):
            await call
    with pytest.raises(AgentTaskContextError):
        state.store.agents.fence_execution(**scope)
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"]
    assert state.store._conn.execute("SELECT operations FROM app_agent_tasks WHERE task_id=?", (task["id"],)).fetchone()[0] == 0
    assert saved["result"] == "" and saved["input_requests"] == []
    assert saved["steps"][0]["result"] == ""


async def test_metadata_fences_never_read_source_bodies_and_duplicate_capture_reads_once(state, monkeypatch):
    from audrey.app_state import agent_projects

    source, _ = await message(state)
    for index in range(4):
        await note(state, [ref(source)], text=f"Stable note {index}.")
    original, reads = agent_projects.database_source_rows, []

    def observed(*args, **kwargs):
        reads.append(kwargs["include_text"])
        return original(*args, **kwargs)

    monkeypatch.setattr(agent_projects, "database_source_rows", observed)
    task = await create(state)
    assert reads.count(True) == 1
    reads.clear()
    claimed = await state.store.agents.claim()
    await state.store.agents.resolve_project_context(**lease(claimed))
    state.store.agents.fence_execution(**lease(claimed))
    assert reads and not any(reads)
    assert task["project_context"]["current"]


async def test_stale_sourced_candidates_do_not_hide_sourceless_notes_and_capture_is_bounded(state, monkeypatch):
    from audrey.app_state import agent_projects

    for index in range(20):
        await note(state, [{"source_id": f"message:missing_{index}", "revision": "old"}], pinned=True)
    fallback = await note(state, [], text="A valid sourceless fallback.")
    original, reads = agent_projects.database_source_rows, []

    def observed(*args, **kwargs):
        reads.append(kwargs["include_text"])
        return original(*args, **kwargs)

    monkeypatch.setattr(agent_projects, "database_source_rows", observed)
    task = await create(state)
    assert reads.count(True) == 12
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]


async def test_twelve_notes_with_eight_sources_fit_existing_manifest_bound(state):
    sources = [ref((await message(state))[0]) for _ in range(8)]
    for index in range(12):
        await note(state, sources, text=f"Stable note {index}.")
    task = await create(state)
    assert len(task["project_context"]["knowledge"]) == 12
    row = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone()
    assert len(row[0]) <= 4000 and len(json.loads(row[0])) == 12


@pytest.mark.parametrize("malformed", ["{}", "[null]", '[{"source_id":"message:x","revision":"old","extra":1}]', json.dumps([{"source_id": "message:" + chr(0xD800), "revision": "old"}])])
async def test_malformed_canonical_refs_fail_closed_after_capture(state, malformed):
    source, _ = await message(state)
    entry = await note(state, [ref(source)])
    task = await create(state)
    claimed = await state.store.agents.claim()
    with state.store._lock, state.store._conn:
        state.store._conn.execute("UPDATE app_project_knowledge SET sources_json=? WHERE entry_id=?", (malformed, entry["id"]))
    with pytest.raises(AgentTaskContextError):
        state.store.agents.fence_execution(**lease(claimed))
    assert not (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["project_context"]["current"]


async def test_source_backed_legacy_pin_without_proof_is_not_silently_adopted(state):
    source, _ = await message(state)
    await note(state, [ref(source)])
    task = await create(state)
    with state.store._lock, state.store._conn:
        row = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone()
        pins = json.loads(row[0])
        pins[0].pop("sources_sha256")
        state.store._conn.execute("UPDATE app_agent_project_contexts SET knowledge_json=? WHERE task_id=?", (json.dumps(pins), task["id"]))
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"]
    assert await state.store.agents.claim() is None


async def test_report_project_rename_and_later_notes_are_safe_but_source_project_deletion_changes_proof(state):
    original_project = await state.store.projects.create(user_id=state.owner.user_id, name="Report origin")
    source, _ = await report(state, project=original_project)
    entry = await note(state, [ref(source)])
    task = await create(state)
    await state.store.projects.update(user_id=state.owner.user_id, project_id=original_project.project_id, name="Renamed origin")
    await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id, name="Renamed current")
    await note(state, [], text="Later unselected note.")
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["project_context"]["current"] and saved["project_context"]["knowledge"] == [{"id": entry["id"], "revision": 1}]
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=original_project.project_id)
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"]
    # The separately retained report itself remains available to Search.
    fresh = await state.reader.resolve(state.owner.user_id, state.owner.storage_namespace, source.source_id)
    assert fresh is not None
    refreshed_note = await note(state, [ref(fresh)], text="Explicitly reviewed retained report note.")
    new_task = await create(state)
    assert new_task["project_context"]["current"]
    assert {"id": refreshed_note["id"], "revision": 1} in new_task["project_context"]["knowledge"]
    assert entry["id"] not in {pin["id"] for pin in new_task["project_context"]["knowledge"]}
