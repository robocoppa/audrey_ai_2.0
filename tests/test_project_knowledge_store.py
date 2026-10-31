"""Project knowledge ownership, approval, concurrency and erasure boundaries."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.knowledge import (
    KnowledgeAccessError,
    KnowledgeConflictError,
    KnowledgeError,
    KnowledgeLimitError,
)


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    users = []
    for name in ("alice", "bob"):
        users.append(await store.resolve_external_identity(
            provider="test", subject=name, email=f"{name}@example.com",
            display_name=name.title(), role="user", auth_method="test",
        ))
    project = await store.projects.create(user_id=users[0].user_id, name="House plans")
    other_project = await store.projects.create(user_id=users[1].user_id, name="Other")
    yield SimpleNamespace(
        store=store, owner=users[0], other=users[1], project=project, other_project=other_project,
    )
    store.close()


async def _create(state, **changes):
    values = dict(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        kind="decision", text="Use the north entrance.",
    )
    values.update(changes)
    return await state.store.knowledge.create(**values)


def _key(state, entry, **changes):
    values = dict(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        entry_id=entry["id"], expected_revision=entry["revision"],
    )
    values.update(changes)
    return values


async def _report(state):
    task = await state.store.agents.create(
        user_id=state.owner.user_id, objective="Research entrances.", request_id=uuid.uuid4().hex,
    )
    task = await state.store.agents.claim()
    lease = {key: task[key] for key in ("lease_token", "fence")}
    lease["task_id"] = task["id"]
    for step in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=step)
        await state.store.agents.checkpoint(**lease, step_id=step, result="Saved report.")
    return await state.store.agents.complete(**lease, result="The south entrance is sheltered.")


async def _proposal(state, **changes):
    report = await _report(state)
    return await _create(
        state, text="Use the south entrance.", origin="agent_report", source_task_id=report["id"],
        sources=[{"source_id": f"agent:{report['id']}:report", "revision": "report-v1"}], **changes,
    )


async def _run(state):
    conversation = await state.store.conversations.create(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    )
    run = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=conversation.conversation_id,
        user_content="Which entrance?",
    )
    return conversation, run.run.run_id


async def test_manual_entries_approved_persistent_and_history_is_immutable(state):
    first = await _create(state, pinned=True)
    assert first["approval_state"] == "approved" and first["approved_at"]
    assert first["revision"] == 1 and first["pinned"] is True
    updated = await state.store.knowledge.update(**_key(state, first), text="Use the west entrance.")
    assert updated["revision"] == 2
    history = await state.store.knowledge.history(
        user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=first["id"],
    )
    assert [entry["text"] for entry in history] == ["Use the west entrance.", first["text"]]
    with state.store._lock:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            state.store._conn.execute(
                "UPDATE app_project_knowledge_revisions SET record_json='{}' WHERE entry_id=?",
                (first["id"],),
            )
        state.store._conn.rollback()
    second = ApplicationStore(state.store.path)
    try:
        assert await second.knowledge.get(
            user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=first["id"],
        ) == updated
    finally:
        second.close()


async def test_foreign_project_and_entry_reads_are_indistinguishable_from_missing(state):
    entry = await _create(state)
    for entry_id in (entry["id"], "missing"):
        with pytest.raises(KnowledgeAccessError, match="unavailable"):
            await state.store.knowledge.get(
                user_id=state.other.user_id, project_id=state.project.project_id, entry_id=entry_id,
            )
        with pytest.raises(KnowledgeAccessError):
            await state.store.knowledge.update(
                **_key(state, entry, user_id=state.other.user_id, entry_id=entry_id), text="No.",
            )
    with pytest.raises(KnowledgeAccessError):
        await _create(state, project_id=state.other_project.project_id)
    with pytest.raises(KnowledgeAccessError):
        await state.store.knowledge.list(user_id=state.other.user_id, project_id=state.project.project_id)


@pytest.mark.parametrize("fence", ["disabled", "deleting"])
async def test_account_status_and_deletion_request_fence_all_paths(state, fence):
    entry = await _create(state)
    with state.store._lock, state.store._conn:
        if fence == "disabled":
            state.store._conn.execute(
                "UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,),
            )
        else:
            state.store._conn.execute(
                "INSERT INTO account_deletion_requests VALUES (?,?,?,?)",
                (state.owner.user_id, "purge", state.owner.storage_namespace, "2026-10-29"),
            )
    for call in (
        _create(state),
        state.store.knowledge.get(
            user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"],
        ),
        state.store.knowledge.list(user_id=state.owner.user_id, project_id=state.project.project_id),
        state.store.knowledge.list_approved(user_id=state.owner.user_id, project_id=state.project.project_id),
        state.store.knowledge.update(**_key(state, entry), text="No write."),
        state.store.knowledge.delete(**_key(state, entry)),
    ):
        with pytest.raises(KnowledgeAccessError):
            await call


async def test_agent_report_proposal_stays_inactive_until_explicit_user_approval(state):
    proposal = await _proposal(state)
    assert proposal["approval_state"] == "proposed" and proposal["approved_at"] is None
    assert await state.store.knowledge.list_approved(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    ) == []
    edited = await state.store.knowledge.update(**_key(state, proposal), text="Shelter the south entrance.")
    assert edited["approval_state"] == "proposed"
    approved = await state.store.knowledge.decide(**_key(state, edited), action="approve")
    assert approved["approval_state"] == "approved" and approved["approved_at"]
    assert approved["work_status"] == "open"
    assert await state.store.knowledge.list_approved(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    ) == [approved]
    with pytest.raises(KnowledgeConflictError):
        await state.store.knowledge.decide(**_key(state, approved), action="approve")


async def test_replacement_approval_supersedes_target_atomically_and_checks_base_revision(state):
    existing = await _create(state)
    proposal = await _proposal(state, replaces_id=existing["id"], base_revision=existing["revision"])
    edited = await state.store.knowledge.update(**_key(state, existing), text="Keep the north entrance.")
    with pytest.raises(KnowledgeConflictError):
        await state.store.knowledge.decide(**_key(state, proposal), action="approve")
    assert state.store.knowledge.get_sync(
        user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=existing["id"],
    )["approval_state"] == "approved"
    rejected = await state.store.knowledge.decide(**_key(state, proposal), action="reject")
    assert rejected["approval_state"] == "rejected"
    report = await _report(state)
    replacement = await _create(
        state, origin="agent_report", source_task_id=report["id"], text="Use west.",
        replaces_id=edited["id"], base_revision=edited["revision"],
    )
    approved = await state.store.knowledge.decide(**_key(state, replacement), action="approve")
    target = state.store.knowledge.get_sync(
        user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=edited["id"],
    )
    assert target["approval_state"] == "superseded" and target["revision"] == 3
    assert (await state.store.knowledge.list_approved(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    )) == [approved]


async def test_optimistic_writes_serialize_between_store_connections(state):
    entry = await _create(state)
    second = ApplicationStore(state.store.path)
    try:
        outcomes = await asyncio.gather(
            state.store.knowledge.update(**_key(state, entry), text="First change."),
            second.knowledge.update(**_key(state, entry), text="Second change."),
            return_exceptions=True,
        )
        assert sum(isinstance(value, dict) for value in outcomes) == 1
        assert sum(isinstance(value, KnowledgeConflictError) for value in outcomes) == 1
    finally:
        second.close()


async def test_history_write_failure_rolls_back_current_entry_and_clears_transaction(state):
    entry = await _create(state)
    state.store._conn.execute(
        "CREATE TEMP TRIGGER reject_knowledge_revision BEFORE INSERT ON app_project_knowledge_revisions "
        "BEGIN SELECT RAISE(ABORT,'injected failure'); END",
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        await state.store.knowledge.update(**_key(state, entry), text="Must roll back.")
    assert not state.store._conn.in_transaction
    assert state.store.knowledge.get_sync(
        user_id=state.owner.user_id, project_id=state.project.project_id, entry_id=entry["id"],
    ) == entry
    state.store._conn.execute("DROP TRIGGER reject_knowledge_revision")
    assert (await state.store.knowledge.update(**_key(state, entry), text="Successful retry."))["revision"] == 2


async def test_work_status_applies_to_kind_and_approval_does_not_complete_actions(state):
    action = await _create(state, kind="action")
    action = await state.store.knowledge.update(**_key(state, action), work_status="completed")
    question = await state.store.knowledge.update(**_key(state, action), kind="question")
    assert question["work_status"] == "open"
    resolved = await state.store.knowledge.update(**_key(state, question), work_status="resolved")
    with pytest.raises(KnowledgeError):
        await state.store.knowledge.update(**_key(state, resolved), work_status="completed")
    assert (await state.store.knowledge.update(**_key(state, resolved), kind="fact"))["work_status"] == "open"


@pytest.mark.parametrize("changes", [
    {"text": " "}, {"text": "x" * 8001}, {"kind": "instruction"}, {"pinned": 1},
    {"sources": [{"source_id": "file:x:text", "revision": "v1", "excerpt": "secret"}]},
    {"sources": [{"source_id": "url:https://example.com", "revision": "v1"}]},
    {"sources": [{"source_id": "message:x", "revision": "v1"}] * 2},
    {"sources": [{"source_id": f"message:{i}", "revision": "v1"} for i in range(9)]},
    {"origin": "agent_report"}, {"base_revision": 1},
])
async def test_invalid_writes_are_bounded_and_never_persist(state, changes):
    with pytest.raises(KnowledgeError):
        await _create(state, **changes)
    assert await state.store.knowledge.list(user_id=state.owner.user_id, project_id=state.project.project_id) == []


async def test_create_request_is_idempotent_but_conflicting_payload_fails(state):
    entries = await asyncio.gather(*[_create(state, request_id="same-request") for _ in range(3)])
    assert entries[0] == entries[1] == entries[2]
    with pytest.raises(KnowledgeConflictError):
        await _create(state, request_id="same-request", text="Other text.")


async def test_entry_revision_and_listing_limits(state):
    entry = await _create(state)
    for revision in range(2, 101):
        entry = await state.store.knowledge.update(**_key(state, entry), text=f"Revision {revision}.")
    with pytest.raises(KnowledgeLimitError):
        await state.store.knowledge.update(**_key(state, entry), text="Over allowance.")
    for index in range(199):
        await _create(state, text=f"Entry {index}.")
    with pytest.raises(KnowledgeLimitError):
        await _create(state)
    page = await state.store.knowledge.list(
        user_id=state.owner.user_id, project_id=state.project.project_id, limit=100,
    )
    next_page = await state.store.knowledge.list(
        user_id=state.owner.user_id, project_id=state.project.project_id, limit=100,
        before_updated_at=page[-1]["updated_at"], before_id=page[-1]["id"],
    )
    assert len(page) == len(next_page) == 100
    assert not {item["id"] for item in page} & {item["id"] for item in next_page}
    with pytest.raises(KnowledgeError):
        await state.store.knowledge.list(user_id=state.owner.user_id, project_id=state.project.project_id, limit=101)


async def test_run_snapshot_is_bounded_owned_approved_current_and_immutable(state):
    entry = await _create(state)
    _conversation, run_id = await _run(state)
    kwargs = dict(user_id=state.owner.user_id, project_id=state.project.project_id, run_id=run_id)
    references = [{"id": entry["id"], "revision": entry["revision"]}]
    await state.store.knowledge.record_run_snapshot(**kwargs, entries=references)
    await state.store.knowledge.record_run_snapshot(**kwargs, entries=references)
    assert await state.store.knowledge.get_run_snapshot(**kwargs) == references
    with pytest.raises(KnowledgeConflictError):
        await state.store.knowledge.record_run_snapshot(**kwargs, entries=[])
    with pytest.raises(KnowledgeAccessError):
        await state.store.knowledge.get_run_snapshot(**{**kwargs, "user_id": state.other.user_id})
    await state.store.knowledge.update(**_key(state, entry), text="Changed after dispatch.")
    assert await state.store.knowledge.get_run_snapshot(**kwargs) == references
    with pytest.raises(KnowledgeConflictError):
        await state.store.knowledge.record_run_snapshot(**kwargs, entries=references)


async def test_empty_run_snapshot_is_also_immutable(state):
    _conversation, run_id = await _run(state)
    kwargs = dict(user_id=state.owner.user_id, project_id=state.project.project_id, run_id=run_id)
    await state.store.knowledge.record_run_snapshot(**kwargs, entries=[])
    await state.store.knowledge.record_run_snapshot(**kwargs, entries=[])
    entry = await _create(state)
    with pytest.raises(KnowledgeConflictError):
        await state.store.knowledge.record_run_snapshot(
            **kwargs, entries=[{"id": entry["id"], "revision": 1}],
        )


@pytest.mark.parametrize("erase", ["entry", "project", "purge"])
async def test_deletion_erases_entries_history_and_run_references(state, erase):
    entry = await _create(state)
    _conversation, run_id = await _run(state)
    await state.store.knowledge.record_run_snapshot(
        user_id=state.owner.user_id, project_id=state.project.project_id, run_id=run_id,
        entries=[{"id": entry["id"], "revision": 1}],
    )
    if erase == "entry":
        await state.store.knowledge.delete(**_key(state, entry))
    elif erase == "project":
        await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    else:
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    for query in (
        "SELECT COUNT(*) FROM app_project_knowledge",
        "SELECT COUNT(*) FROM app_project_knowledge_revisions",
        "SELECT COUNT(*) FROM app_run_project_knowledge",
    ):
        assert state.store._conn.execute(query).fetchone()[0] == 0
    assert not state.store._conn.execute("PRAGMA foreign_key_check").fetchall()


async def test_migration_is_additive_reopen_safe_and_preserves_existing_project(state):
    assert state.store.schema_version == 30
    second = ApplicationStore(state.store.path)
    try:
        assert second.schema_version == 30
        assert await second.projects.get(
            user_id=state.owner.user_id, project_id=state.project.project_id,
        ) == state.project
        assert second._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        second.close()
