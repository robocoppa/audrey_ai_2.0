"""Approved Project knowledge stays bounded, attributable and revocable."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from test_native_private_workflow import native  # noqa: F401
from test_private_document_ingestion import setup as document_setup  # noqa: F401

from audrey.app_state import ApplicationStore
from audrey.content_sources import ContentSourceReader
from audrey.kb.qdrant import KBHit
from audrey.kb.uploads_db import UploadsDB
from audrey.knowledge_sources import KnowledgeSourceResolver
from audrey.models.ollama import _report_model_call
from audrey.processing_policy import processing_scope
from audrey.project_context import (
    PROJECT_KNOWLEDGE_MAX_TOKENS,
    PROJECT_RETRIEVAL_MAX_TOKENS,
    ProjectContextUnavailableError,
    ProjectFileSnapshot,
    _bounded_passages,
    _token_count,
    assert_project_knowledge_current,
    project_context_system_message,
    resolve_project_context,
)
from audrey.routes.app.runs import NativeRunManager, _record_project_knowledge
from audrey.routes.openai.schemas import ChatCompletionRequest


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="test", subject="alice", email="alice@example.com",
        display_name="Alice", role="user", auth_method="test",
    )
    project = await store.projects.create(user_id=owner.user_id, name="Launch")
    conversation = await store.conversations.create(
        user_id=owner.user_id, project_id=project.project_id,
    )
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    reader = ContentSourceReader(store, uploads, tmp_path / "uploads")
    resolver = KnowledgeSourceResolver(store, reader)
    app = SimpleNamespace(state=SimpleNamespace(
        application_store=store, uploads_db=uploads,
        cfg=SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path / "uploads")}}),
        knowledge_source_resolver=resolver,
    ))
    yield SimpleNamespace(
        store=store, owner=owner, project=project, conversation=conversation,
        uploads=uploads, reader=reader, resolver=resolver, app=app,
    )
    uploads.close()
    store.close()


async def note(state, text="The launch uses a blue label.", **kwargs):
    return await state.store.knowledge.create(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        kind="decision", text=text, **kwargs,
    )


async def context(state, query="Which label is selected?"):
    return await resolve_project_context(
        SimpleNamespace(app=state.app), store=state.store, principal=state.owner,
        conversation=state.conversation, query=query,
    )


async def test_approved_notes_are_quoted_with_revision_and_combined_budget(state):
    entry = await note(state, pinned=True)
    await note(state, "Long background detail. " * 300)
    snapshot = await context(state)
    assert snapshot.knowledge[0].entry_id == entry["id"]
    assert snapshot.knowledge[0].revision == 1
    used = sum(_token_count(item.text) for item in snapshot.knowledge)
    assert 0 < used <= PROJECT_KNOWLEDGE_MAX_TOKENS
    rendered = project_context_system_message(snapshot)["content"]
    assert "approval does not verify truth" in rendered
    assert '"origin":"user"' in rendered and '"work_status":"open"' in rendered
    hits = [KBHit(score=1, source="local", kind="text", chunk_idx=0,
                  text="File evidence. " * 3000, payload={"file_id": "f"})]
    passages = _bounded_passages(
        hits, {"f": ProjectFileSnapshot("f", "evidence.txt", "text/plain", "text")},
        max_tokens=PROJECT_RETRIEVAL_MAX_TOKENS - used,
    )
    assert used + sum(_token_count(item.text) for item in passages) <= PROJECT_RETRIEVAL_MAX_TOKENS


async def test_changed_or_deleted_entry_cannot_keep_grounding_from_old_snapshot(state):
    entry = await note(state)
    snapshot = await context(state)
    assert_project_knowledge_current(state.app, snapshot)
    await state.store.knowledge.update(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        entry_id=entry["id"], expected_revision=1, text="Use a green label.",
    )
    with pytest.raises(ProjectContextUnavailableError):
        assert_project_knowledge_current(state.app, snapshot)
    current = await context(state)
    assert current.knowledge[0].revision == 2
    await state.store.knowledge.delete(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        entry_id=entry["id"], expected_revision=2,
    )
    assert not state.resolver.is_current(current.knowledge_proofs[0])


async def test_source_message_revision_and_project_detachment_fence_notes(state):
    started = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="The launch date is November 3.",
    )
    source = await state.reader.resolve(
        state.owner.user_id, state.owner.storage_namespace,
        f"message:{started.user_message.message_id}",
    )
    entry = await note(state, sources=[{"source_id": source.source_id, "revision": str(source.revision)}])
    snapshot = await context(state)
    assert snapshot.knowledge[0].entry_id == entry["id"]
    await state.store.conversations.update(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        project_id=None, update_project=True,
    )
    assert not state.resolver.is_current(snapshot.knowledge_proofs[0])
    proof = await state.resolver.resolve(state.owner, state.project, entry)
    assert proof.stale and "detached" in proof.warnings[0]


async def test_missing_source_does_not_become_alternate_grounding(state):
    await note(state, "Retained user note with unavailable source.", sources=[{
        "source_id": "message:msg_missing", "revision": "old",
    }])
    assert (await context(state)).knowledge == ()
    # The user-authored note remains editable; only its automatic reuse is fenced.
    assert len(await state.store.knowledge.list_approved(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    )) == 1


async def test_message_reference_survives_new_turn_but_not_source_deletion(state):
    first = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="The blue label was selected.",
    )
    await state.store.conversations.finish_run(
        user_id=state.owner.user_id, run_id=first.run.run_id,
        outcome="succeeded", assistant_content="Recorded.",
    )
    source = await state.reader.resolve(
        state.owner.user_id, state.owner.storage_namespace,
        f"message:{first.user_message.message_id}",
    )
    await note(state, sources=[{"source_id": source.source_id, "revision": str(source.revision)}])
    snapshot = await context(state)
    second = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="What did we decide?",
    )
    assert_project_knowledge_current(state.app, snapshot)
    fresh = await state.reader.resolve(state.owner.user_id, state.owner.storage_namespace, source.source_id)
    assert fresh.revision == source.revision
    await state.store.conversations.finish_run(
        user_id=state.owner.user_id, run_id=second.run.run_id,
        outcome="succeeded", assistant_content="The blue label.",
    )
    assert await state.store.conversations.delete(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
    )
    assert not state.resolver.is_current(snapshot.knowledge_proofs[0])


async def test_search_proof_cache_eviction_does_not_invalidate_native_context(state):
    started = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="A source decision.",
    )
    source = await state.reader.resolve(
        state.owner.user_id, state.owner.storage_namespace,
        f"message:{started.user_message.message_id}",
    )
    await note(state, sources=[{"source_id": source.source_id, "revision": str(source.revision)}])
    state.app.state.content_search = SimpleNamespace(reader=state.reader)
    snapshot = await context(state)
    assert snapshot.knowledge_resolver.reader is not state.reader
    with state.reader._proof_lock:
        state.reader._proofs.clear()
    assert not state.reader.is_current(source)
    assert_project_knowledge_current(state.app, snapshot)


async def test_source_changes_during_later_await_drop_earlier_proof(state, monkeypatch):
    first = await note(state, "First pinned decision", pinned=True)
    await note(state, "Second decision")
    original = state.resolver.resolve
    calls = 0

    async def changing(*args):
        nonlocal calls
        proof = await original(*args)
        calls += 1
        if calls == 2:
            await state.store.knowledge.update(
                user_id=state.owner.user_id, project_id=state.project.project_id,
                entry_id=first["id"], expected_revision=1, text="Changed decision",
            )
        return proof

    monkeypatch.setattr(state.resolver, "resolve", changing)
    monkeypatch.setattr("audrey.project_context.get_knowledge_resolver", lambda *args, **kwargs: state.resolver)
    snapshot = await context(state)
    assert first["id"] not in {entry.entry_id for entry in snapshot.knowledge}


@pytest.mark.parametrize("swallow_refusal", [False, True])
async def test_native_run_saves_revision_ids_and_refuses_changed_context_before_dispatch(state, swallow_refusal):
    entry = await note(state)
    snapshot = await context(state)
    started = await state.store.conversations.begin_run(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="What did we decide?",
    )
    await _record_project_knowledge(SimpleNamespace(app=state.app), state.owner, started, snapshot)
    assert await state.store.knowledge.get_run_snapshot(
        user_id=state.owner.user_id, run_id=started.run.run_id, project_id=state.project.project_id,
    ) == [{"id": entry["id"], "revision": 1}]
    await state.store.knowledge.update(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        entry_id=entry["id"], expected_revision=1, text="A later decision.",
    )
    dispatched = []

    async def stream(*args, **kwargs):
        try:
            _report_model_call("local-test")
        except ProjectContextUnavailableError:
            if not swallow_refusal:
                raise
            emitter = kwargs["event_context"].emitter
            emitter.run_started()
            emitter.message_started()
            emitter.message_finished(status="completed")
            emitter.run_finished(status="succeeded", finish_reason="stop")
            yield "refused"
            return
        dispatched.append(True)
        yield "never"

    manager = NativeRunManager(app=state.app, store=state.store, stream_factory=stream)
    await manager.launch(
        principal=state.owner, started=started,
        payload=ChatCompletionRequest(model="audrey_fast", messages=[{"role": "user", "content": "Question"}]),
        messages=[], options={}, project_context=snapshot,
    )
    await asyncio.wait_for(manager._runs[started.run.run_id].task, 2)
    run = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=started.run.run_id)
    assert dispatched == []
    assert run.status == "failed" and run.error_code == "project_context_changed"
    terminal = manager._runs[started.run.run_id].terminal_event
    assert terminal.status == "failed" and terminal.error_code == "project_context_changed"


async def test_project_deletion_invalidates_note_proofs(state):
    await note(state)
    snapshot = await context(state)
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    assert not state.resolver.is_current(snapshot.knowledge_proofs[0])


async def test_disabled_account_cannot_reuse_already_resolved_notes(state):
    await note(state)
    snapshot = await context(state)
    with state.store._lock, state.store._conn:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    assert not state.resolver.is_current(snapshot.knowledge_proofs[0])


async def test_private_project_notes_use_only_verified_local_runtime(native):  # noqa: F811
    s = native
    project = await s.store.projects.create(
        user_id=s.owner.user_id, name="Private decisions", private_required=True,
    )
    conversation = await s.store.conversations.create(
        user_id=s.owner.user_id, project_id=project.project_id,
    )
    with processing_scope(s.env.runtime.policy):
        source_run = await s.store.conversations.begin_run(
            user_id=s.owner.user_id, conversation_id=conversation.conversation_id,
            user_content="Owner confirmed the blue decision.", expected_private_required=True,
        )
    await s.store.conversations.finish_run(
        user_id=s.owner.user_id, run_id=source_run.run.run_id,
        outcome="succeeded", assistant_content="Confirmed.",
    )
    reader = ContentSourceReader(s.store, s.app.state.uploads_db, s.app.state.cfg.raw["kb"]["upload_root"])
    source = await reader.resolve(s.owner.user_id, s.owner.storage_namespace, f"message:{source_run.user_message.message_id}")
    entry = await s.store.knowledge.create(
        user_id=s.owner.user_id, project_id=project.project_id,
        kind="decision", text="Use the blue launch label.",
        sources=[{"source_id": source.source_id, "revision": str(source.revision)}],
    )
    created = await s.client.post(
        f"/api/conversations/{conversation.conversation_id}/runs",
        json={"content": "Which label did we choose?", "model_id": "private"},
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["id"]
    await asyncio.wait_for(s.manager._runs[run_id].task, 3)
    assert "Use the blue launch label." in str(s.captured)
    assert all(request.url.host == "ollama" for request in s.env.seen)
    assert await s.store.knowledge.get_run_snapshot(
        user_id=s.owner.user_id, run_id=run_id, project_id=project.project_id,
    ) == [{"id": entry["id"], "revision": 1}]
    saved = await s.store.conversations.list_messages(
        user_id=s.owner.user_id, conversation_id=conversation.conversation_id,
    )
    assert all("Use the blue launch label." not in message.content for message in saved)
