"""Durable local-only restrictions before the Private workflow is exposed."""

from __future__ import annotations

import sqlite3

import pytest

import audrey.app_state.store as store_module
from audrey.app_state import (
    ApplicationStore,
    ConversationHasActiveRunError,
    InvalidApplicationStateError,
    InvalidProjectStateError,
    ProjectNotFoundError,
)
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    processing_scope,
)


async def _owner(store: ApplicationStore, name: str = "alice"):
    return await store.resolve_external_identity(
        provider="owui",
        subject=f"private-state-{name}",
        email=f"{name}@example.com",
        display_name=name.title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=f"{name}@example.com",
    )


def _policy() -> ProcessingPolicy:
    return ProcessingPolicy(
        local_models={"local-test:latest": "a" * 64},
        ollama_base_url="http://ollama:11434",
    )


async def _finish(store: ApplicationStore, owner_id: str, run_id: str):
    return await store.conversations.finish_run(
        user_id=owner_id,
        run_id=run_id,
        outcome="succeeded",
        assistant_content="Recorded answer.",
    )


async def test_schema_21_upgrade_preserves_existing_state_without_claiming_privacy(
    tmp_path, monkeypatch,
):
    path = tmp_path / "app.sqlite"
    migrations = store_module.MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(
            store_module, "MIGRATIONS",
            tuple(entry for entry in migrations if entry[0] <= 21),
        )
        legacy = ApplicationStore(path)
        try:
            owner = await _owner(legacy)
        finally:
            legacy.close()
    # Seed the old schema directly: current repositories correctly require the
    # new columns, which do not exist until this migration has been applied.
    stamp = "2026-10-18T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO app_projects "
            "(project_id, user_id, name, instructions, created_at, updated_at) "
            "VALUES ('proj_existing', ?, 'Existing', '', ?, ?)",
            (owner.user_id, stamp, stamp),
        )
        connection.execute(
            "INSERT INTO app_conversations "
            "(conversation_id, user_id, title, default_mode, default_model_id, "
            "created_at, updated_at, last_message_at, archived_at, project_id) "
            "VALUES ('con_existing', ?, 'Existing conversation', 'auto', "
            "'direct/local-test:latest', ?, ?, ?, NULL, 'proj_existing')",
            (owner.user_id, stamp, stamp, stamp),
        )
        connection.execute(
            "INSERT INTO app_runs "
            "(run_id, conversation_id, user_id, mode, requested_model_id, status, "
            "started_at, completed_at, finish_reason, virtual_model, concrete_model, "
            "prompt_tokens, completion_tokens) "
            "VALUES ('run_existing', 'con_existing', ?, 'auto', "
            "'direct/local-test:latest', 'succeeded', ?, ?, 'stop', '', '', 0, 0)",
            (owner.user_id, stamp, stamp),
        )
        connection.executemany(
            "INSERT INTO app_messages "
            "(message_id, conversation_id, user_id, run_id, sequence_no, role, "
            "status, content, created_at, updated_at) "
            "VALUES (?, 'con_existing', ?, 'run_existing', ?, ?, 'completed', ?, ?, ?)",
            [
                ("msg_existing_user", owner.user_id, 1, "user",
                 "Previously saved prompt.", stamp, stamp),
                ("msg_existing_assistant", owner.user_id, 2, "assistant",
                 "Recorded answer.", stamp, stamp),
            ],
        )
        connection.execute(
            "INSERT INTO app_chat_projections "
            "(projection_id, user_id, conversation_id, user_message_id, "
            "assistant_message_id, created_at, next_attempt_at) "
            "VALUES ('native:run_existing', ?, 'con_existing', "
            "'msg_existing_user', 'msg_existing_assistant', ?, ?)",
            (owner.user_id, stamp, stamp),
        )

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 26
        after = await upgraded.conversations.get(
            user_id=owner.user_id, conversation_id="con_existing",
        )
        assert after is not None
        assert after.title == "Existing conversation"
        assert after.default_model_id == "direct/local-test:latest"
        assert after.project_id == "proj_existing"
        assert after.private_required is False
        saved_project = await upgraded.projects.get(
            user_id=owner.user_id, project_id="proj_existing",
        )
        assert saved_project is not None and saved_project.private_required is False
        run = await upgraded.conversations.get_run(
            user_id=owner.user_id, run_id="run_existing",
        )
        assert run is not None and run.private_required is False
        messages = await upgraded.conversations.list_messages(
            user_id=owner.user_id, conversation_id="con_existing",
        )
        assert messages is not None
        assert [message.content for message in messages] == [
            "Previously saved prompt.", "Recorded answer.",
        ]
        assert len(await upgraded.chat_projections.due()) == 1
    finally:
        upgraded.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_private_project_inheritance_is_owner_bound_and_survives_restart(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    try:
        alice = await _owner(store)
        bob = await _owner(store, "bob")
        project = await store.projects.create(
            user_id=alice.user_id, name="Protected", private_required=True,
        )
        conversation = await store.conversations.create(
            user_id=alice.user_id, project_id=project.project_id,
        )
        assert project.private_required is True
        assert conversation.private_required is True
        snapshot = await store.projects.snapshot_for_conversation(
            user_id=alice.user_id, conversation_id=conversation.conversation_id,
        )
        assert snapshot is not None and snapshot[0].private_required is True
        assert await store.projects.get(
            user_id=bob.user_id, project_id=project.project_id,
        ) is None
        assert await store.conversations.get(
            user_id=bob.user_id, conversation_id=conversation.conversation_id,
        ) is None
        with pytest.raises(ProjectNotFoundError):
            await store.conversations.create(user_id=bob.user_id, project_id=project.project_id)
        assert await store.conversations.update(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
            title="Foreign edit",
        ) is None
    finally:
        store.close()

    reopened = ApplicationStore(path)
    try:
        projects = await reopened.projects.list_page(user_id=alice.user_id, limit=10)
        conversations = await reopened.conversations.list_for_user(user_id=alice.user_id)
        assert len(projects) == len(conversations) == 1
        assert projects[0].private_required is True
        assert conversations[0].private_required is True
    finally:
        reopened.close()


async def test_private_restriction_survives_allowed_moves_and_refuses_public_downgrades(tmp_path):
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    from audrey.auth import require_principal
    from audrey.routes.app.conversations import router

    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        protected = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        second_protected = await store.projects.create(
            user_id=owner.user_id, name="Another protected project", private_required=True,
        )
        ordinary = await store.projects.create(user_id=owner.user_id, name="Ordinary")
        conversation = await store.conversations.create(
            user_id=owner.user_id, project_id=protected.project_id,
        )
        assert conversation.default_model_id == "private"
        with pytest.raises(InvalidApplicationStateError):
            await store.conversations.update(
                user_id=owner.user_id, conversation_id=conversation.conversation_id,
                project_id=ordinary.project_id, update_project=True,
            )
        retained = await store.conversations.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
        )
        assert retained == conversation

        app = FastAPI()
        app.state.application_store = store
        app.state.cfg = SimpleNamespace(raw={"private_processing": {"enabled": True}})
        app.include_router(router, prefix="/api")
        app.dependency_overrides[require_principal] = lambda: owner
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = f"/api/conversations/{conversation.conversation_id}"
            for forbidden in (
                {"project_id": ordinary.project_id},
                {"model_id": "cloud"},
            ):
                refused = await client.patch(path, json=forbidden)
                assert refused.status_code == 409, refused.text
                unchanged = await store.conversations.get(
                    user_id=owner.user_id, conversation_id=conversation.conversation_id,
                )
                assert unchanged == conversation
            for patch in (
                {"project_id": second_protected.project_id},
                {"project_id": None},
                {"archived": True},
                {"archived": False},
                {"title": "Renamed"},
            ):
                changed = await client.patch(path, json=patch)
                assert changed.status_code == 200, changed.text
                current = changed.json()
                assert current["private_required"] is True
                assert current["default_model_id"] == "private"
                if "project_id" in patch:
                    assert current["project_id"] == patch["project_id"]
                if "archived" in patch:
                    assert (current["archived_at"] is not None) is patch["archived"]
                if "title" in patch:
                    assert current["title"] == patch["title"]
        edited_project = await store.projects.update(
            user_id=owner.user_id, project_id=protected.project_id, name="Renamed project",
        )
        assert edited_project is not None and edited_project.private_required is True
    finally:
        store.close()


async def test_deleting_private_project_retains_conversation_restriction(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        project = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        conversation = await store.conversations.create(
            user_id=owner.user_id, project_id=project.project_id,
        )
        assert await store.projects.delete(user_id=owner.user_id, project_id=project.project_id)
        retained = await store.conversations.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
        )
        assert retained is not None
        assert retained.project_id is None
        assert retained.private_required is True
    finally:
        store.close()


async def test_empty_project_move_inherits_and_stale_run_expectation_is_atomic(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    other = ApplicationStore(path)
    try:
        owner = await _owner(store)
        project = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        original = await store.conversations.create(user_id=owner.user_id)
        assert original.private_required is False
        changed = await other.conversations.update(
            user_id=owner.user_id,
            conversation_id=original.conversation_id,
            project_id=project.project_id,
            update_project=True,
        )
        assert changed is not None and changed.private_required is True
        with processing_scope(_policy()), pytest.raises(InvalidApplicationStateError):
            await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=original.conversation_id,
                user_content="Must not be saved with stale policy.",
                expected_private_required=original.private_required,
            )
        assert await store.conversations.list_messages(
            user_id=owner.user_id, conversation_id=original.conversation_id,
        ) == ()
        assert await store.chat_projections.due() == ()
    finally:
        other.close()
        store.close()


@pytest.mark.parametrize("active", [False, True])
async def test_existing_standard_turn_cannot_be_relabelled_private(tmp_path, active):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        project = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Already accepted as an ordinary turn.",
        )
        assert started is not None
        if not active:
            await _finish(store, owner.user_id, started.run.run_id)
        with pytest.raises((InvalidApplicationStateError, ConversationHasActiveRunError)):
            await store.conversations.update(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                project_id=project.project_id,
                update_project=True,
            )
        retained = await store.conversations.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
        )
        assert retained is not None
        assert retained.project_id is None and retained.private_required is False
        assert len(await store.chat_projections.due()) == (0 if active else 1)
    finally:
        store.close()


async def test_private_run_requires_active_server_scope_and_matching_expectation(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        conversation = await store.conversations.create(
            user_id=owner.user_id, private_required=True,
        )
        with pytest.raises(InvalidApplicationStateError):
            await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Omitted expectation.",
            )
        with pytest.raises(ProcessingPolicyError) as refusal:
            await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Missing server scope.",
                expected_private_required=True,
            )
        assert refusal.value.code == "private_processing_unavailable"
        assert await store.conversations.list_messages(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
        ) == ()
        with processing_scope(_policy()):
            started = await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Scoped protected turn.",
                expected_private_required=True,
            )
        assert started is not None
        assert started.conversation.private_required is True
        assert started.run.private_required is True
        ordinary = await store.conversations.create(user_id=owner.user_id)
        with processing_scope(_policy()), pytest.raises(InvalidApplicationStateError):
            await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=ordinary.conversation_id,
                user_content="Wrong positive expectation.",
                expected_private_required=True,
            )
        assert await store.conversations.list_messages(
            user_id=owner.user_id, conversation_id=ordinary.conversation_id,
        ) == ()
    finally:
        store.close()


@pytest.mark.parametrize("recover", [False, True])
async def test_private_turns_never_enter_shared_history_projection(tmp_path, recover):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _owner(store)
    ordinary = await store.conversations.create(user_id=owner.user_id)
    private = await store.conversations.create(user_id=owner.user_id, private_required=True)
    try:
        normal_run = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=ordinary.conversation_id,
            user_content="Ordinary prompt.",
        )
        with processing_scope(_policy()):
            private_run = await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=private.conversation_id,
                user_content="Private prompt.",
                expected_private_required=True,
            )
        assert normal_run is not None and private_run is not None
        if not recover:
            await _finish(store, owner.user_id, normal_run.run.run_id)
            await _finish(store, owner.user_id, private_run.run.run_id)
        else:
            store.close()
            store = ApplicationStore(path)
            assert await store.conversations.recover_interrupted_runs() == 2
        receipts = await store.chat_projections.due()
        assert len(receipts) == 1
        assert receipts[0].conversation_id == ordinary.conversation_id
        assert receipts[0].user_content == "Ordinary prompt."
        assert "Private prompt." not in repr(receipts)
        run = await store.conversations.get_run(
            user_id=owner.user_id, run_id=private_run.run.run_id,
        )
        assert run is not None and run.private_required is True
    finally:
        store.close()


async def test_private_project_file_link_refuses_unprotected_ingestion(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        project = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        with processing_scope(_policy()), pytest.raises(ProcessingPolicyError) as refusal:
            await store.projects.add_file(
                user_id=owner.user_id, project_id=project.project_id, file_id="ordinary-file",
            )
        assert refusal.value.code == "private_processing_unavailable"
        assert await store.projects.list_files(
            user_id=owner.user_id, project_id=project.project_id,
        ) == ()
    finally:
        store.close()


@pytest.mark.parametrize("table,id_column", [
    ("app_conversations", "conversation_id"),
    ("app_projects", "project_id"),
    ("app_runs", "run_id"),
])
async def test_sql_rejects_downgrade_or_run_snapshot_mutation(tmp_path, table, id_column):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    try:
        owner = await _owner(store)
        project = await store.projects.create(
            user_id=owner.user_id, name="Protected", private_required=True,
        )
        conversation = await store.conversations.create(
            user_id=owner.user_id, project_id=project.project_id,
        )
        with processing_scope(_policy()):
            started = await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Protected prompt.",
                expected_private_required=True,
            )
        assert started is not None
        ids = {
            "app_projects": project.project_id,
            "app_conversations": conversation.conversation_id,
            "app_runs": started.run.run_id,
        }
        with sqlite3.connect(path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE {table} SET private_required = 0 WHERE {id_column} = ?",  # noqa: S608 - fixed identifiers above
                    (ids[table],),
                )
            connection.rollback()
            row = connection.execute(
                f"SELECT private_required FROM {table} WHERE {id_column} = ?",  # noqa: S608 - fixed identifiers above
                (ids[table],),
            ).fetchone()
            assert row == (1,)
    finally:
        store.close()


@pytest.mark.parametrize("value", [0, 1, None, "true"])
@pytest.mark.parametrize("resource", ["project", "conversation"])
async def test_private_creation_rejects_non_boolean_restrictions(tmp_path, resource, value):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _owner(store)
        if resource == "project":
            with pytest.raises(InvalidProjectStateError):
                await store.projects.create(
                    user_id=owner.user_id, name="Invalid", private_required=value,
                )
            assert await store.projects.list_page(user_id=owner.user_id, limit=10) == ()
        else:
            with pytest.raises(InvalidApplicationStateError):
                await store.conversations.create(user_id=owner.user_id, private_required=value)
            assert await store.conversations.list_for_user(user_id=owner.user_id) == ()
    finally:
        store.close()
