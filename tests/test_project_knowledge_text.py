"""The plain Project document preserves old notes and existing context fences."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_project_knowledge_store import _create, _key, _proposal

from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskContextError
from audrey.app_state.knowledge import KnowledgeAccessError, KnowledgeConflictError, KnowledgeError


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "application.sqlite")
    users = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    project = await store.projects.create(user_id=users[0].user_id, name="Garden")
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], project=project)
    store.close()


def owner(state, **changes):
    return {"user_id": state.owner.user_id, "project_id": state.project.project_id, **changes}


async def read(state, **changes):
    return await state.store.knowledge.get_text(**owner(state, **changes))


async def save(state, text, revision=None, **changes):
    revision = revision or (await read(state))["revision"]
    return await state.store.knowledge.save_text(
        **owner(state, **changes), text=text, expected_revision=revision,
    )


async def approved(state):
    return await state.store.knowledge.list_approved(**owner(state))


async def test_read_preserves_full_legacy_text_and_does_not_promote_or_mutate_suggestions(state):
    first = await _create(state, text="A" * 6000, sources=[{"source_id": "message:old", "revision": "v1"}])
    second = await _create(state, kind="action", text="B" * 6000)
    proposal = await _proposal(state)
    rejection = await _proposal(state)
    rejected = await state.store.knowledge.decide(**_key(state, rejection), action="reject")
    before = {entry["id"]: entry for entry in (first, second, proposal, rejected)}

    snapshot = await read(state)
    assert snapshot["text"] == first["text"] + "\n\n" + second["text"]
    assert len(snapshot["text"]) > snapshot["max_text_chars"] == 8000
    assert len(snapshot["revision"]) == 64
    assert await read(state) == snapshot
    with pytest.raises(KnowledgeError, match="8000"):
        await save(state, snapshot["text"], snapshot["revision"])
    for entry_id, entry in before.items():
        assert await state.store.knowledge.get(**owner(state), entry_id=entry_id) == entry

    result, changed = await save(state, "  Keep this user-reviewed note.\n", snapshot["revision"])
    assert result["text"] == "  Keep this user-reviewed note.\n"
    current = await approved(state)
    assert len(current) == 1 and current[0]["text"] == result["text"]
    assert current[0]["kind"] == "fact" and current[0]["pinned"]
    assert current[0]["origin"] == "user" and current[0]["sources"] == []
    assert set(changed) == {first["id"], second["id"], current[0]["id"]}
    for entry in (first, second):
        old = await state.store.knowledge.get(**owner(state), entry_id=entry["id"])
        assert old["approval_state"] == "superseded" and old["revision"] == 2
        assert old["sources"] == entry["sources"] and old["text"] == entry["text"]
        history = await state.store.knowledge.history(**owner(state), entry_id=entry["id"])
        assert history[-1] == entry
    for entry in (proposal, rejected):
        assert await state.store.knowledge.get(**owner(state), entry_id=entry["id"]) == entry


async def test_clearing_reopening_and_identical_retry_preserve_identity_without_new_revisions(state):
    initial = await read(state)
    document, touched = await save(state, "First note.", initial["revision"])
    entry = (await approved(state))[0]
    assert touched == [entry["id"]]
    assert await save(state, "First note.", initial["revision"]) == (document, [])
    assert (await approved(state))[0]["revision"] == 1

    updated, _ = await save(state, "Different note.", document["revision"])
    with pytest.raises(KnowledgeConflictError):
        await save(state, "First note.", document["revision"])
    assert await read(state) == updated
    cleared, changed = await save(state, " \n\t", updated["revision"])
    assert cleared["text"] == "" and changed == [entry["id"]] and await approved(state) == []
    assert await save(state, "", updated["revision"]) == (cleared, [])
    restored, _ = await save(state, "Restored note.", cleared["revision"])
    final = (await approved(state))[0]
    assert restored["text"] == final["text"] and final["id"] == entry["id"]
    assert final["revision"] == 4
    reopened = ApplicationStore(state.store.path)
    try:
        assert await reopened.knowledge.get_text(**owner(state)) == restored
    finally:
        reopened.close()


async def test_two_connections_serialize_document_saves_and_preserve_losing_draft(state):
    initial = await read(state)
    second = ApplicationStore(state.store.path)
    try:
        results = await asyncio.gather(
            save(state, "First editor.", initial["revision"]),
            second.knowledge.save_text(**owner(state), text="Second editor.", expected_revision=initial["revision"]),
            return_exceptions=True,
        )
        assert sum(isinstance(result, tuple) for result in results) == 1
        assert sum(isinstance(result, KnowledgeConflictError) for result in results) == 1
        assert (await read(state))["text"] in {"First editor.", "Second editor."}
        assert len(await approved(state)) == 1
    finally:
        second.close()


async def test_document_edits_continue_past_100_while_legacy_history_is_preserved(state):
    legacy = await _create(state)
    for revision in range(2, 101):
        legacy = await state.store.knowledge.update(**_key(state, legacy), text=f"Legacy revision {revision}.")
    snapshot, _ = await save(state, "Document revision 1.")
    legacy_current = await state.store.knowledge.get(**owner(state), entry_id=legacy["id"])
    assert legacy_current["approval_state"] == "superseded" and legacy_current["revision"] == 101
    assert state.store._conn.execute(
        "SELECT COUNT(*) FROM app_project_knowledge_revisions WHERE entry_id=?", (legacy["id"],),
    ).fetchone()[0] == 101
    for revision in range(2, 107):
        snapshot, _ = await save(state, f"Document revision {revision}.", snapshot["revision"])
    entry = (await approved(state))[0]
    assert entry["revision"] == 106
    history = await state.store.knowledge.history(**owner(state), entry_id=entry["id"])
    assert len(history) == 100 and history[0]["revision"] == 106 and history[-1]["revision"] == 7
    assert state.store._conn.execute(
        "SELECT COUNT(*) FROM app_project_knowledge_revisions WHERE entry_id=?", (legacy["id"],),
    ).fetchone()[0] == 101


async def test_consolidation_at_legacy_entry_limit_keeps_every_saved_entry(state):
    for index in range(200):
        await _create(state, text=f"Legacy note {index}.")
    result, changed = await save(state, "The consolidated document.")
    assert result["text"] == "The consolidated document." and len(changed) == 201
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_project_knowledge").fetchone()[0] == 201
    assert len(await approved(state)) == 1


async def test_failed_history_write_rolls_back_superseding_and_document_creation(state):
    legacy = await _create(state)
    before = await read(state)
    state.store._conn.execute(
        "CREATE TEMP TRIGGER reject_text_save BEFORE INSERT ON app_project_knowledge_revisions "
        "BEGIN SELECT RAISE(ABORT,'injected document failure'); END",
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected document failure"):
        await save(state, "Must not replace the old note.", before["revision"])
    assert await read(state) == before and (await approved(state)) == [legacy]
    assert not state.store._conn.in_transaction


@pytest.mark.parametrize("fence", ["foreign", "missing", "disabled", "deleting", "deleted_project"])
async def test_plain_text_read_and_write_keep_owner_and_deletion_fences(state, fence):
    snapshot, _ = await save(state, "Owner-only note.")
    args = owner(state)
    if fence == "foreign":
        args["user_id"] = state.other.user_id
    elif fence == "missing":
        args["project_id"] = "missing"
    elif fence == "deleted_project":
        await state.store.projects.delete(**owner(state))
    else:
        with state.store._lock, state.store._conn:
            if fence == "disabled":
                state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
            else:
                state.store._conn.execute("INSERT INTO account_deletion_requests VALUES (?,?,?,?)",
                    (state.owner.user_id, "purge", state.owner.storage_namespace, "2026-11-04"))
    for operation in (state.store.knowledge.get_text(**args), state.store.knowledge.save_text(
        **args, text="Unauthorized replacement.", expected_revision=snapshot["revision"],
    )):
        with pytest.raises(KnowledgeAccessError):
            await operation


async def test_private_project_text_stays_owned_without_granting_agent_access(state):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Private", private_required=True)
    args = owner(state, project_id=project.project_id)
    snapshot = await state.store.knowledge.get_text(**args)
    document, _ = await state.store.knowledge.save_text(**args, text="Private notes.", expected_revision=snapshot["revision"])
    assert (await state.store.knowledge.get_text(**args)) == document
    with pytest.raises(KnowledgeAccessError):
        await state.store.knowledge.get_text(**args | {"user_id": state.other.user_id})
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.create(**args, objective="Read private notes.", request_id="private-agent")


async def test_document_edits_fence_existing_agent_versions_and_keep_new_context_body_free(state):
    snapshot, _ = await save(state, "The garden label is Fern Circle.")
    task = await state.store.agents.create(**owner(state), objective="Use the garden label.", request_id="plain-context")
    claimed = await state.store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    context = await state.store.agents.resolve_project_context(**lease)
    assert [entry["text"] for entry in context["knowledge"]] == [snapshot["text"]]
    manifest = dict(state.store._conn.execute("SELECT * FROM app_agent_project_contexts WHERE task_id=?", (task["id"],)).fetchone())
    assert snapshot["text"] not in json.dumps(manifest)
    assert await save(state, snapshot["text"], snapshot["revision"]) == (snapshot, [])
    assert await state.store.agents.resolve_project_context(**lease) == context
    await save(state, "The garden label is Moss Circle.", snapshot["revision"])
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.resolve_project_context(**lease)


@pytest.mark.parametrize("text,revision", [("x" * 8001, "a" * 64), (42, "a" * 64), ("text", "bad"), ("text", None)])
async def test_document_write_rejects_invalid_bounds_and_versions(state, text, revision):
    with pytest.raises(KnowledgeError):
        await state.store.knowledge.save_text(**owner(state), text=text, expected_revision=revision)
