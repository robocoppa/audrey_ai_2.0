"""Contracts for Audrey's canonical preferences, conversations, messages, and runs."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3

import pytest

from audrey.app_state import (
    ApplicationStore,
    ConversationArchivedError,
    ConversationHasActiveRunError,
    InvalidApplicationStateError,
    PersonalTokenAuthenticationError,
    RunAlreadyTerminalError,
)


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str = "owui-alice",
    email: str = "alice@example.com",
):
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


async def test_v2_upgrade_backfills_preferences_without_changing_identity_or_token(tmp_path):
    path = tmp_path / "app.sqlite"
    current = ApplicationStore(path)
    owner = await _resolve(current)
    issued = await current.create_personal_token(
        user_id=owner.user_id,
        name="Migration token",
        scopes=["account:read"],
        expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
    )
    current.close()

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE app_messages")
        conn.execute("DROP TABLE app_runs")
        conn.execute("DROP TABLE app_conversations")
        conn.execute("DROP TABLE user_preferences")
        conn.execute("DELETE FROM app_schema_migrations WHERE version = 3")
        conn.commit()

    upgraded = ApplicationStore(path)
    try:
        after = await _resolve(upgraded)
        preferences = await upgraded.preferences.get(user_id=owner.user_id)
        assert upgraded.schema_version == 4
        assert after.user_id == owner.user_id
        assert preferences is not None
        assert preferences.timezone == "UTC"
        assert preferences.persona == ""
        assert preferences.response_preferences == {}
        assert (await upgraded.authenticate_personal_token(issued.token)).user_id == owner.user_id
    finally:
        upgraded.close()


async def test_preferences_are_validated_owner_bound_and_persisted(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        defaults = await store.preferences.get(user_id=alice.user_id)
        assert defaults is not None
        assert defaults.timezone == "UTC"
        assert defaults.response_preferences == {}

        replaced = await store.preferences.replace(
            user_id=alice.user_id,
            timezone="America/Denver",
            persona="Be concise.",
            response_preferences={"citations": "inline", "detail": 3},
        )
        assert replaced is not None
        assert replaced.timezone == "America/Denver"
        assert replaced.response_preferences == {"citations": "inline", "detail": 3}
        assert (await store.preferences.get(user_id=bob.user_id)).timezone == "UTC"
        assert await store.preferences.replace(
            user_id="usr_missing",
            timezone="UTC",
            persona="",
            response_preferences={},
        ) is None
        with pytest.raises(InvalidApplicationStateError, match="IANA"):
            await store.preferences.replace(
                user_id=alice.user_id,
                timezone="Mountain Time",
                persona="",
                response_preferences={},
            )
        with pytest.raises(InvalidApplicationStateError, match="valid JSON"):
            await store.preferences.replace(
                user_id=alice.user_id,
                timezone="UTC",
                persona="",
                response_preferences={"invalid": object()},
            )
    finally:
        store.close()

    reopened = ApplicationStore(path)
    try:
        persisted = await reopened.preferences.get(user_id=alice.user_id)
        assert persisted is not None
        assert persisted.timezone == "America/Denver"
        assert persisted.persona == "Be concise."
    finally:
        reopened.close()


async def test_conversation_and_run_ids_are_server_owned_and_transactional(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(
            user_id=owner.user_id,
            title="Canonical chat",
            default_mode="deep",
        )
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Explain the result.",
        )
        assert conversation.conversation_id.startswith("con_")
        assert started is not None
        assert started.run.run_id.startswith("run_")
        assert started.run.mode == "deep"
        assert started.run.status == "running"
        assert started.user_message.message_id.startswith("msg_")
        assert started.assistant_message.message_id.startswith("msg_")
        assert (started.user_message.sequence_no, started.assistant_message.sequence_no) == (1, 2)
        assert started.user_message.status == "completed"
        assert started.assistant_message.status == "in_progress"

        finished = await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome="succeeded",
            assistant_content="Here is the explanation.",
            finish_reason="stop",
            virtual_model="audrey_deep",
            concrete_model="qwen3.8:latest",
            prompt_tokens=12,
            completion_tokens=8,
        )
        assert finished is not None
        assert finished.run.status == "succeeded"
        assert finished.run.completed_at is not None
        assert finished.run.prompt_tokens == 12
        assert finished.assistant_message.status == "completed"
        assert finished.assistant_message.content == "Here is the explanation."
        with pytest.raises(RunAlreadyTerminalError):
            await store.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="failed",
                assistant_content="second ending",
            )
    finally:
        store.close()

    reopened = ApplicationStore(path)
    try:
        messages = await reopened.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert messages is not None
        assert [message.sequence_no for message in messages] == [1, 2]
        assert messages[1].content == "Here is the explanation."
        assert (await reopened.conversations.get_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        )).status == "succeeded"
    finally:
        reopened.close()


async def test_terminal_run_commits_search_projection_receipt_with_canonical_state(
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Project this question.",
        )
        assert started is not None
        await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome="succeeded",
            assistant_content="Project this answer.",
            finish_reason="stop",
            virtual_model="audrey_fast",
            concrete_model="qwen-test",
            prompt_tokens=11,
            completion_tokens=7,
        )

        pending = await store.chat_projections.due()
        assert len(pending) == 1
        projection = pending[0]
        assert projection.projection_id == f"native:{started.run.run_id}"
        assert projection.user_id == owner.user_id
        assert projection.storage_namespace == "alice@example.com"
        assert projection.conversation_id == conversation.conversation_id
        assert projection.user_content == "Project this question."
        assert projection.assistant_content == "Project this answer."
        assert projection.partial is False
        assert projection.virtual_model == "audrey_fast"
        assert projection.concrete_model == "qwen-test"
        assert (projection.prompt_tokens, projection.completion_tokens) == (11, 7)

        assert await store.chat_projections.mark_failed(
            projection_id=projection.projection_id,
            error="private transport detail",
            retry_interval_s=60,
        )
        assert await store.chat_projections.due() == ()
        stats = await store.chat_projections.stats(user_id=owner.user_id)
        assert stats["pending"] == 1
        assert stats["attempts"] == 1
        assert stats["with_error"] == 1

        assert await store.chat_projections.retry_now() == 1
        assert len(await store.chat_projections.due()) == 1
        assert await store.chat_projections.mark_enqueued(
            projection_id=projection.projection_id
        )
        stats = await store.chat_projections.stats(user_id=owner.user_id)
        assert stats["pending"] == 0
        assert stats["completed"] == 1
        assert await store.chat_projections.reset_all() == 1
        assert len(await store.chat_projections.due()) == 1
    finally:
        store.close()


async def test_projection_receipt_failure_rolls_back_terminal_transition(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Keep the terminal write atomic.",
        )
        assert started is not None
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TRIGGER fail_chat_projection_insert
                BEFORE INSERT ON app_chat_projections
                BEGIN
                  SELECT RAISE(ABORT, 'projection receipt blocked');
                END;
                """
            )

        with pytest.raises(sqlite3.IntegrityError, match="projection receipt blocked"):
            await store.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="succeeded",
                assistant_content="This must roll back too.",
            )

        run = await store.conversations.get_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        )
        messages = await store.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert run is not None and run.status == "running"
        assert messages is not None
        assert messages[-1].status == "in_progress"
        assert messages[-1].content == ""

        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER fail_chat_projection_insert")
        assert await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome="succeeded",
            assistant_content="Atomic after repair.",
        )
        assert len(await store.chat_projections.due()) == 1
    finally:
        store.close()


async def test_schema_v3_upgrade_does_not_duplicate_legacy_archive_writes(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store)
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Existing canonical question.",
    )
    assert started is not None
    await store.conversations.finish_run(
        user_id=owner.user_id,
        run_id=started.run.run_id,
        outcome="failed",
        assistant_content="Existing partial answer.",
        finish_reason="error",
        error_code="provider_error",
    )
    store.close()

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE app_chat_projections")
        connection.execute("DELETE FROM app_schema_migrations WHERE version = 4")
        connection.commit()

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 4
        assert await upgraded.chat_projections.due() == ()
        existing = await upgraded.conversations.get_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        )
        assert existing is not None
        assert existing.status == "failed"
    finally:
        upgraded.close()


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
async def test_non_success_terminal_outcomes_retain_explicit_partial_content(tmp_path, outcome):
    store = ApplicationStore(tmp_path / f"{outcome}.sqlite")
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Start.",
        )
        assert started is not None
        finished = await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome=outcome,
            assistant_content="Partial answer",
            error_code="client_cancelled" if outcome == "cancelled" else "provider_error",
        )
        assert finished is not None
        assert finished.run.status == outcome
        assert finished.assistant_message.status == "incomplete"
        assert finished.assistant_message.content == "Partial answer"
    finally:
        store.close()


async def test_two_store_connections_allow_one_active_run_and_keep_message_order(tmp_path):
    path = tmp_path / "app.sqlite"
    left = ApplicationStore(path)
    owner = await _resolve(left)
    conversation = await left.conversations.create(user_id=owner.user_id)
    right = ApplicationStore(path)
    try:
        results = await asyncio.gather(
            left.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="left",
            ),
            right.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="right",
            ),
            return_exceptions=True,
        )
        starts = [result for result in results if not isinstance(result, BaseException)]
        failures = [result for result in results if isinstance(result, BaseException)]
        assert len(starts) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ConversationHasActiveRunError)
        start = starts[0]
        assert start is not None
        messages = await left.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert messages is not None
        assert [message.sequence_no for message in messages] == [1, 2]
        assert start.assistant_message.sequence_no == start.user_message.sequence_no + 1

        await left.conversations.finish_run(
            user_id=owner.user_id,
            run_id=start.run.run_id,
            outcome="succeeded",
            assistant_content="first answer",
        )
        next_start = await right.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="after terminal",
        )
        assert next_start is not None
        assert (next_start.user_message.sequence_no, next_start.assistant_message.sequence_no) == (
            3,
            4,
        )
    finally:
        left.close()
        right.close()


async def test_archived_conversation_rejects_new_run(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        archived = await store.conversations.update(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            archived=True,
        )
        assert archived is not None and archived.archived_at is not None

        with pytest.raises(ConversationArchivedError, match="unarchived"):
            await store.conversations.begin_run(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Do not start.",
            )
        assert await store.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        ) == ()
    finally:
        store.close()


async def test_restart_recovery_terminalizes_interrupted_runs_once(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        alice_conversation = await store.conversations.create(user_id=alice.user_id)
        bob_conversation = await store.conversations.create(user_id=bob.user_id)
        alice_run = await store.conversations.begin_run(
            user_id=alice.user_id,
            conversation_id=alice_conversation.conversation_id,
            user_content="First interrupted turn",
        )
        bob_run = await store.conversations.begin_run(
            user_id=bob.user_id,
            conversation_id=bob_conversation.conversation_id,
            user_content="Second interrupted turn",
        )
        assert alice_run is not None and bob_run is not None
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE app_messages SET content = ? WHERE message_id = ?",
                ("partial response", alice_run.assistant_message.message_id),
            )
            connection.commit()

        assert await store.conversations.recover_interrupted_runs() == 2
        assert await store.conversations.recover_interrupted_runs() == 0

        for owner, started in ((alice, alice_run), (bob, bob_run)):
            recovered = await store.conversations.get_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
            )
            assert recovered is not None
            assert recovered.status == "failed"
            assert recovered.completed_at is not None
            assert recovered.finish_reason == "error"
            assert recovered.error_code == "server_restart"
            messages = await store.conversations.list_messages(
                user_id=owner.user_id,
                conversation_id=started.conversation.conversation_id,
            )
            assert messages is not None
            assert messages[-1].status == "incomplete"
        alice_messages = await store.conversations.list_messages(
            user_id=alice.user_id,
            conversation_id=alice_conversation.conversation_id,
        )
        assert alice_messages is not None
        assert alice_messages[-1].content == "partial response"
        projections = await store.chat_projections.due()
        assert len(projections) == 2
        by_user = {projection.user_id: projection for projection in projections}
        assert by_user[alice.user_id].assistant_content == "partial response"
        assert by_user[alice.user_id].partial is True
        assert by_user[bob.user_id].assistant_content == ""
        assert by_user[bob.user_id].partial is True
    finally:
        store.close()


async def test_two_store_connections_allow_exactly_one_terminal_transition(tmp_path):
    path = tmp_path / "app.sqlite"
    left = ApplicationStore(path)
    owner = await _resolve(left)
    conversation = await left.conversations.create(user_id=owner.user_id)
    started = await left.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Race the finish.",
    )
    assert started is not None
    right = ApplicationStore(path)
    try:
        results = await asyncio.gather(
            left.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="succeeded",
                assistant_content="first candidate",
            ),
            right.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="failed",
                assistant_content="second candidate",
            ),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, BaseException) for result in results) == 1
        assert sum(isinstance(result, RunAlreadyTerminalError) for result in results) == 1
        run = await left.conversations.get_run(user_id=owner.user_id, run_id=started.run.run_id)
        messages = await left.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert run is not None and run.status in {"succeeded", "failed"}
        assert messages is not None
        assert messages[1].status == ("completed" if run.status == "succeeded" else "incomplete")
    finally:
        left.close()
        right.close()


async def test_cross_user_reads_and_mutations_are_indistinguishable_from_missing(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        conversation = await store.conversations.create(user_id=alice.user_id)
        started = await store.conversations.begin_run(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Private turn",
        )
        assert started is not None
        assert await store.conversations.get(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
        ) is None
        assert await store.conversations.list_messages(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
        ) is None
        assert await store.conversations.get_run(
            user_id=bob.user_id,
            run_id=started.run.run_id,
        ) is None
        assert await store.conversations.begin_run(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
            user_content="intrusion",
        ) is None
        assert await store.conversations.finish_run(
            user_id=bob.user_id,
            run_id=started.run.run_id,
            outcome="failed",
            assistant_content="intrusion",
        ) is None
        assert (await store.conversations.get_run(
            user_id=alice.user_id,
            run_id=started.run.run_id,
        )).status == "running"
    finally:
        store.close()


async def test_conversation_metadata_archive_and_delete_are_owner_bound(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        conversation = await store.conversations.create(
            user_id=alice.user_id,
            title="First title",
        )
        updated = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            title="Renamed",
            default_mode="research",
        )
        assert updated is not None
        assert (updated.title, updated.default_mode) == ("Renamed", "research")

        assert await store.conversations.update(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
            title="Intrusion",
        ) is None
        assert not await store.conversations.delete(
            user_id=bob.user_id,
            conversation_id=conversation.conversation_id,
        )

        archived = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            archived=True,
        )
        assert archived is not None and archived.archived_at is not None
        assert await store.conversations.list_page(
            user_id=alice.user_id,
            archived=False,
            limit=10,
        ) == ()
        assert [record.conversation_id for record in await store.conversations.list_page(
            user_id=alice.user_id,
            archived=True,
            limit=10,
        )] == [conversation.conversation_id]

        restored = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            archived=False,
        )
        assert restored is not None and restored.archived_at is None
        assert await store.conversations.delete(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert await store.conversations.get(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
        ) is None
        deletions = await store.chat_projections.due_deletions()
        assert len(deletions) == 1
        assert deletions[0].deletion_id == (
            f"conversation:{conversation.conversation_id}"
        )
        assert deletions[0].user_id == alice.user_id
        assert deletions[0].storage_namespace == "alice@example.com"
    finally:
        store.close()


async def test_conversation_title_search_is_literal_and_owner_scoped(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        matching = await store.conversations.create(
            user_id=alice.user_id,
            title="Release 100%_Ready",
        )
        await store.conversations.create(user_id=alice.user_id, title="Other notes")
        await store.conversations.create(
            user_id=bob.user_id,
            title="Release 100%_Ready for Bob",
        )

        result = await store.conversations.list_page(
            user_id=alice.user_id,
            archived=False,
            limit=10,
            search="  100%_ready  ",
        )

        assert [record.conversation_id for record in result] == [
            matching.conversation_id
        ]
    finally:
        store.close()


async def test_projection_tombstone_failure_rolls_back_conversation_delete(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TRIGGER fail_projection_deletion_insert
                BEFORE INSERT ON app_chat_projection_deletions
                BEGIN
                  SELECT RAISE(ABORT, 'projection deletion blocked');
                END;
                """
            )

        with pytest.raises(sqlite3.IntegrityError, match="projection deletion blocked"):
            await store.conversations.delete(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
            )
        assert await store.conversations.get(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        ) is not None
        assert await store.chat_projections.due_deletions() == ()
    finally:
        store.close()


async def test_active_run_blocks_archive_and_delete_until_terminal(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    try:
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Keep this conversation alive.",
        )
        assert started is not None

        with pytest.raises(ConversationHasActiveRunError, match="archived"):
            await store.conversations.update(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
                archived=True,
            )
        with pytest.raises(ConversationHasActiveRunError, match="deleted"):
            await store.conversations.delete(
                user_id=owner.user_id,
                conversation_id=conversation.conversation_id,
            )

        await store.conversations.finish_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            outcome="cancelled",
            assistant_content="Partial response",
        )
        assert await store.conversations.delete(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert await store.conversations.get_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        ) is None
        assert await store.chat_projections.due() == ()
        assert len(await store.chat_projections.due_deletions()) == 1
    finally:
        store.close()


async def test_conversation_and_message_pages_use_stable_owner_bound_cursors(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        conversations = [
            await store.conversations.create(user_id=alice.user_id, title=f"Chat {index}")
            for index in range(5)
        ]
        expected = await store.conversations.list_page(
            user_id=alice.user_id,
            archived=False,
            limit=100,
        )
        pages = []
        activity = None
        conversation_id = None
        for _ in range(3):
            page = await store.conversations.list_page(
                user_id=alice.user_id,
                archived=False,
                limit=2,
                before_activity_at=activity,
                before_conversation_id=conversation_id,
            )
            pages.extend(page)
            if not page:
                break
            last = page[-1]
            activity = last.last_message_at or last.created_at
            conversation_id = last.conversation_id
        assert [record.conversation_id for record in pages] == [
            record.conversation_id for record in expected
        ]
        assert len({record.conversation_id for record in pages}) == len(conversations)

        target = conversations[0]
        for prompt, answer in (("one", "first"), ("two", "second")):
            started = await store.conversations.begin_run(
                user_id=alice.user_id,
                conversation_id=target.conversation_id,
                user_content=prompt,
            )
            assert started is not None
            await store.conversations.finish_run(
                user_id=alice.user_id,
                run_id=started.run.run_id,
                outcome="succeeded",
                assistant_content=answer,
            )
        first = await store.conversations.list_message_page(
            user_id=alice.user_id,
            conversation_id=target.conversation_id,
            after_sequence=0,
            limit=3,
        )
        assert first is not None
        second = await store.conversations.list_message_page(
            user_id=alice.user_id,
            conversation_id=target.conversation_id,
            after_sequence=first[-1].sequence_no,
            limit=3,
        )
        assert second is not None
        assert [message.sequence_no for message in (*first, *second)] == [1, 2, 3, 4]
        assert await store.conversations.list_message_page(
            user_id=bob.user_id,
            conversation_id=target.conversation_id,
            after_sequence=0,
            limit=3,
        ) is None
    finally:
        store.close()


async def test_schema_rejects_cross_user_run_linkage(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    conversation = await store.conversations.create(user_id=alice.user_id)
    store.close()

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO app_runs "
                "(run_id, conversation_id, user_id, mode, status, started_at) "
                "VALUES ('run_cross_user', ?, ?, 'auto', 'running', 'now')",
                (conversation.conversation_id, bob.user_id),
            )


async def test_schema_rejects_incomplete_or_rewritten_terminal_outcomes(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store)
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Finish once.",
    )
    assert started is not None
    store.close()

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                "UPDATE app_runs SET status = 'failed' WHERE run_id = ?",
                (started.run.run_id,),
            )
        conn.execute(
            "UPDATE app_runs SET status = 'succeeded', completed_at = 'now' WHERE run_id = ?",
            (started.run.run_id,),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE app_runs SET status = 'failed', completed_at = 'later' WHERE run_id = ?",
                (started.run.run_id,),
            )


async def test_local_purge_is_atomic_owner_bound_and_idempotent(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store)
    bob = await _resolve(store, subject="owui-bob", email="bob@example.com")
    try:
        await store.preferences.replace(
            user_id=alice.user_id,
            timezone="America/Denver",
            persona="Private persona",
            response_preferences={"detail": 9},
        )
        deleted_conversation = await store.conversations.create(
            user_id=alice.user_id
        )
        assert await store.conversations.delete(
            user_id=alice.user_id,
            conversation_id=deleted_conversation.conversation_id,
        )
        assert (
            await store.chat_projections.deletion_stats(user_id=alice.user_id)
        )["pending"] == 1
        alice_conversation = await store.conversations.create(user_id=alice.user_id)
        alice_run = await store.conversations.begin_run(
            user_id=alice.user_id,
            conversation_id=alice_conversation.conversation_id,
            user_content="Erase me",
        )
        bob_conversation = await store.conversations.create(user_id=bob.user_id)
        bob_run = await store.conversations.begin_run(
            user_id=bob.user_id,
            conversation_id=bob_conversation.conversation_id,
            user_content="Keep me",
        )
        token = await store.create_personal_token(
            user_id=alice.user_id,
            name="Erase me",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        assert alice_run is not None and bob_run is not None

        result = await store.purge_local_user_data(user_id=alice.user_id)
        assert result.tokens_deleted == 1
        assert result.conversations_deleted == 1
        assert result.messages_deleted == 2
        assert result.runs_deleted == 1
        assert result.preferences_reset
        with pytest.raises(PersonalTokenAuthenticationError):
            await store.authenticate_personal_token(token.token)
        assert await store.conversations.get(
            user_id=alice.user_id,
            conversation_id=alice_conversation.conversation_id,
        ) is None
        assert await store.chat_projections.deletion_stats(
            user_id=alice.user_id
        ) == {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
            "oldest_created_at": "",
        }
        preferences = await store.preferences.get(user_id=alice.user_id)
        assert preferences is not None
        assert (preferences.timezone, preferences.persona, preferences.response_preferences) == (
            "UTC",
            "",
            {},
        )
        bob_after = await store.conversations.get(
            user_id=bob.user_id,
            conversation_id=bob_conversation.conversation_id,
        )
        assert bob_after is not None
        assert bob_after.conversation_id == bob_conversation.conversation_id
        assert bob_after.last_message_at is not None
        assert (await store.conversations.get_run(
            user_id=bob.user_id,
            run_id=bob_run.run.run_id,
        )).status == "running"

        repeated = await store.purge_local_user_data(user_id=alice.user_id)
        assert repeated.tokens_deleted == 0
        assert repeated.conversations_deleted == 0
        assert repeated.messages_deleted == 0
        assert repeated.runs_deleted == 0
        assert repeated.preferences_reset
    finally:
        store.close()


async def test_repository_rejects_invalid_modes_and_terminal_metadata(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    try:
        with pytest.raises(InvalidApplicationStateError, match="mode"):
            await store.conversations.create(user_id=owner.user_id, default_mode="video")
        conversation = await store.conversations.create(user_id=owner.user_id)
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Hello",
        )
        assert started is not None
        with pytest.raises(InvalidApplicationStateError, match="outcome"):
            await store.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="stopped",
                assistant_content="",
            )
        with pytest.raises(InvalidApplicationStateError, match="negative"):
            await store.conversations.finish_run(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                outcome="succeeded",
                assistant_content="",
                completion_tokens=-1,
            )
        assert (await store.conversations.get_run(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        )).status == "running"
    finally:
        store.close()
