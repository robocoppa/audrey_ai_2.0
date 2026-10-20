"""Private Project context, ownership and immutable conversation binding."""
from __future__ import annotations

# ruff: noqa: F811 - shared pytest fixtures are deliberately named parameters
from types import SimpleNamespace

import pytest
from test_native_private_workflow import native  # noqa: F401
from test_private_document_ingestion import PRIVATE_TEXT, USER, _ingest, _land
from test_private_document_ingestion import setup as document_setup  # noqa: F401

from audrey.app_state import ApplicationStore, ConversationProjectChangedError
from audrey.kb.private_documents import private_text_collection
from audrey.kb.qdrant import KBHit
from audrey.processing_policy import ProcessingPolicyError, processing_scope
from audrey.project_context import (
    PROJECT_RETRIEVAL_MAX_TOKENS,
    project_context_system_message,
    resolve_project_context,
)


async def _owner(store):
    return await store.resolve_external_identity(
        provider="owui", subject="private-project-owner", email="owner@example.com",
        display_name="Owner", role="user", auth_method="owui_bearer",
        legacy_storage_namespace=USER,
    )


async def _project_setup(env, tmp_path):
    store = ApplicationStore(tmp_path / "project-app.sqlite")
    owner = await _owner(store)
    project = await store.projects.create(
        user_id=owner.user_id, name="Private evidence", private_required=True,
        instructions="Use only supported facts; explain missing evidence.",
    )
    conversation = await store.conversations.create(
        user_id=owner.user_id, project_id=project.project_id,
    )
    file_id, path, reservation = await _land(env)
    await _ingest(env, file_id, path, reservation)
    with processing_scope(env.runtime.policy):
        await store.projects.add_file(
            user_id=owner.user_id, project_id=project.project_id,
            file_id=file_id, file_is_private=True,
        )
    return store, owner, project, conversation, file_id


async def test_protected_context_uses_exact_private_collection_and_bound_local_embedder(
    document_setup, tmp_path,
):
    env = document_setup
    store, owner, project, conversation, file_id = await _project_setup(env, tmp_path)
    hit = KBHit(
        score=.8, source="local", kind="text", chunk_idx=0, text=PRIVATE_TEXT,
        payload={"user": USER, "file_id": file_id, "private_required": True,
                 "filename": "example.txt", "artifact": "document"},
    )
    env.qdrant.collection_exists.return_value = True
    env.qdrant.search_hybrid.return_value = ([hit], [hit])
    try:
        async with env.runtime.scope(env.app) as app:
            snapshot = await resolve_project_context(
                SimpleNamespace(app=app), store=store, principal=owner,
                conversation=conversation, query="Summarize the evidence.",
            )
        assert snapshot.project_id == project.project_id
        assert snapshot.updated_at == (await store.projects.get(
            user_id=owner.user_id, project_id=project.project_id,
        )).updated_at
        assert snapshot.files[0].file_id == file_id
        assert snapshot.passages[0].text == PRIVATE_TEXT
        assert snapshot.sources[0].title == "example.txt"
        assert "untrusted data" in project_context_system_message(snapshot)["content"]
        call = env.qdrant.search_hybrid.call_args
        assert call.kwargs["collection"] == private_text_collection(USER)
        assert call.kwargs["scope"].user == USER
        assert call.kwargs["scope"].file_ids == [file_id]
        assert call.args[1] == "Summarize the evidence."
        assert any(request.url.path == "/api/embed" for request in env.seen)
        assert all(request.url.host == "ollama" for request in env.seen)
        assert PROJECT_RETRIEVAL_MAX_TOKENS == 3000
    finally:
        store.close()


@pytest.mark.parametrize("forgery", ["owner", "file", "privacy"])
async def test_private_project_refuses_forged_retriever_identity(
    document_setup, tmp_path, forgery,
):
    env = document_setup
    store, owner, _, conversation, file_id = await _project_setup(env, tmp_path)
    payload = {"user": USER, "file_id": file_id, "private_required": True}
    payload[{"owner": "user", "file": "file_id", "privacy": "private_required"}[forgery]] = (
        False if forgery == "privacy" else "not-authorized"
    )
    hit = KBHit(score=1, source="local", kind="text", chunk_idx=0,
                text="FORGED PRIVATE CONTENT", payload=payload)
    env.qdrant.collection_exists.return_value = True
    env.qdrant.search_hybrid.return_value = ([hit], [])
    try:
        async with env.runtime.scope(env.app) as app:
            with pytest.raises(ProcessingPolicyError) as error:
                await resolve_project_context(
                    SimpleNamespace(app=app), store=store, principal=owner,
                    conversation=conversation, query="Evidence?",
                )
        assert "FORGED" not in str(error.value)
    finally:
        store.close()


async def test_private_project_empty_manifest_never_searches_global_collection(document_setup, tmp_path):
    env = document_setup
    store = ApplicationStore(tmp_path / "empty.sqlite")
    owner = await _owner(store)
    project = await store.projects.create(user_id=owner.user_id, name="Empty", private_required=True)
    conversation = await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
    try:
        async with env.runtime.scope(env.app) as app:
            snapshot = await resolve_project_context(
                SimpleNamespace(app=app), store=store, principal=owner,
                conversation=conversation, query="Any facts?",
            )
        assert snapshot.files == snapshot.passages == snapshot.sources == ()
        env.qdrant.search_hybrid.assert_not_called()
        assert all(request.url.path in {"/api/tags", "/api/show"} for request in env.seen)
    finally:
        store.close()


@pytest.mark.parametrize("change", ["instructions", "add", "remove", "prune"])
async def test_private_project_revision_change_refuses_before_persisting_question(document_setup, tmp_path, change):
    env = document_setup
    store = ApplicationStore(tmp_path / "revision.sqlite")
    owner = await _owner(store)
    project = await store.projects.create(user_id=owner.user_id, name="Tax case", private_required=True)
    conversation = await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
    with processing_scope(env.runtime.policy):
        if change in {"remove", "prune"}:
            await store.projects.add_file(
                user_id=owner.user_id, project_id=project.project_id,
                file_id="synthetic-source", file_is_private=True,
            )
        project = await store.projects.get(user_id=owner.user_id, project_id=project.project_id)
        if change == "instructions":
            await store.projects.update(user_id=owner.user_id, project_id=project.project_id, instructions="Changed")
        elif change == "add":
            await store.projects.add_file(
                user_id=owner.user_id, project_id=project.project_id,
                file_id="synthetic-source", file_is_private=True,
            )
        elif change == "remove":
            await store.projects.remove_file(
                user_id=owner.user_id, project_id=project.project_id, file_id="synthetic-source",
            )
        else:
            await store.projects.prune_file(user_id=owner.user_id, file_id="synthetic-source")
    try:
        with processing_scope(env.runtime.policy):
            with pytest.raises(ConversationProjectChangedError):
                await store.conversations.begin_run(
                    user_id=owner.user_id, conversation_id=conversation.conversation_id,
                    user_content="Read project", expected_private_required=True,
                    expected_project_id=project.project_id, enforce_project_id=True,
                    expected_project_updated_at=project.updated_at,
                )
        assert await store.conversations.list_messages(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
        ) == ()
    finally:
        store.close()


async def test_generic_conversation_creation_infers_private_project_and_refuses_explicit_other_model(native):
    project = await native.store.projects.create(
        user_id=native.owner.user_id, name="Private project", private_required=True,
    )
    created = await native.client.post("/api/conversations", json={"project_id": project.project_id})
    assert created.status_code == 201, created.text
    assert created.json()["private_required"] is True
    assert created.json()["default_model_id"] == "private"
    assert created.json()["default_mode"] == "local"
    for model in ("auto", "cloud", "local"):
        denied = await native.client.post(
            "/api/conversations", json={"project_id": project.project_id, "model_id": model},
        )
        assert denied.status_code == 409, denied.text
    missing = await native.client.post("/api/conversations", json={"project_id": "project-foreign"})
    assert missing.status_code == 404


async def test_private_conversation_attach_detach_and_project_delete_preserve_restriction(native):
    project = await native.store.projects.create(
        user_id=native.owner.user_id, name="Protected", private_required=True,
    )
    ordinary = await native.store.projects.create(user_id=native.owner.user_id, name="Ordinary")
    created = await native.client.post("/api/conversations", json={"model_id": "private"})
    conversation_id = created.json()["id"]
    path = f"/api/conversations/{conversation_id}"
    attached = await native.client.patch(path, json={"project_id": project.project_id})
    assert attached.status_code == 200, attached.text
    denied = await native.client.patch(path, json={"project_id": ordinary.project_id})
    assert denied.status_code == 409, denied.text
    detached = await native.client.patch(path, json={"project_id": None})
    assert detached.status_code == 200
    assert detached.json()["private_required"] is True
    assert detached.json()["default_model_id"] == "private"
    await native.client.patch(path, json={"project_id": project.project_id})
    await native.store.projects.delete(user_id=native.owner.user_id, project_id=project.project_id)
    retained = (await native.client.get(path)).json()
    assert retained["project_id"] is None and retained["private_required"] is True


async def test_empty_ordinary_chat_can_enter_private_project_but_populated_chat_cannot(native):
    project = await native.store.projects.create(
        user_id=native.owner.user_id, name="Protected", private_required=True,
    )
    empty = await native.client.post("/api/conversations", json={})
    path = f"/api/conversations/{empty.json()['id']}"
    moved = await native.client.patch(path, json={"project_id": project.project_id})
    assert moved.status_code == 200, moved.text
    assert moved.json()["private_required"] and moved.json()["default_model_id"] == "private"
    ordinary = await native.store.conversations.create(user_id=native.owner.user_id)
    started = await native.store.conversations.begin_run(
        user_id=native.owner.user_id, conversation_id=ordinary.conversation_id, user_content="Existing question",
    )
    await native.store.conversations.finish_run(
        user_id=native.owner.user_id, run_id=started.run.run_id,
        outcome="succeeded", assistant_content="Existing answer",
    )
    denied = await native.client.patch(
        f"/api/conversations/{ordinary.conversation_id}", json={"project_id": project.project_id},
    )
    assert denied.status_code == 422, denied.text
    retained = (await native.client.get(f"/api/conversations/{ordinary.conversation_id}")).json()
    assert retained["project_id"] is None and retained["private_required"] is False


async def test_private_project_metadata_updates_do_not_depend_on_available_model_catalog(native, monkeypatch):
    from audrey.routes.app import conversations

    project = await native.store.projects.create(
        user_id=native.owner.user_id, name="Protected", private_required=True,
    )
    conversation = await native.store.conversations.create(
        user_id=native.owner.user_id, project_id=project.project_id,
    )

    async def unavailable(*_args, **_kwargs):
        pytest.fail("Renaming an existing Private chat must not resolve a model")

    monkeypatch.setattr(conversations, "_selected_model", unavailable)
    renamed = await native.client.patch(
        f"/api/conversations/{conversation.conversation_id}", json={"title": "Renamed protected chat"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["private_required"] is True
    assert renamed.json()["title"] == "Renamed protected chat"


@pytest.mark.parametrize("lexical_match", [False, True])
async def test_private_project_grounding_requires_dense_or_lexical_evidence(
    document_setup, tmp_path, lexical_match,
):
    env = document_setup
    store, owner, _, conversation, file_id = await _project_setup(env, tmp_path)
    # These are the deployed corpus floors, not thresholds tuned for the test.
    env.app.state.cfg.raw["kb"].update({
        "min_score": .53, "hybrid": {"rrf_k": 60, "min_term_overlap": .7},
    })
    query = "eligible solar equipment"
    text = "Eligible solar equipment is listed on this invoice." if lexical_match else "Unrelated boxing registration form."
    hit = KBHit(
        score=.01, source="local", kind="text", chunk_idx=0, text=text,
        payload={"user": USER, "file_id": file_id, "private_required": True,
                 "filename": "example.txt", "artifact": "document"},
    )
    env.qdrant.collection_exists.return_value = True
    env.qdrant.search_hybrid.return_value = ([hit], [hit])
    try:
        async with env.runtime.scope(env.app) as app:
            snapshot = await resolve_project_context(
                SimpleNamespace(app=app), store=store, principal=owner,
                conversation=conversation, query=query,
            )
        assert snapshot.files[0].file_id == file_id
        if lexical_match:
            # An exact lexical match survives even below the dense cutoff.
            assert snapshot.passages[0].text == text
            assert snapshot.sources[0].title == "example.txt"
        else:
            # Nearest available vectors and a BM25 rank do not make a source.
            assert snapshot.passages == snapshot.sources == ()
            prompt = project_context_system_message(snapshot)["content"]
            assert text not in prompt
            assert "No relevant text passage was retrieved" in prompt
    finally:
        store.close()
