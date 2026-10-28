"""SQLite write failures must not disable later canonical application writes."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass

import pytest

from audrey.app_state import ApplicationStore


@dataclass(frozen=True)
class _ApplicationCase:
    store: ApplicationStore
    user_id: str
    conversation_id: str
    projection_id: str
    deletion_id: str
    token_id: str
    token: str


async def _terminal_turn(store: ApplicationStore, user_id: str):
    conversation = await store.conversations.create(user_id=user_id)
    started = await store.conversations.begin_run(
        user_id=user_id,
        conversation_id=conversation.conversation_id,
        user_content="Canonical question",
    )
    assert started is not None
    await store.conversations.finish_run(
        user_id=user_id,
        run_id=started.run.run_id,
        outcome="succeeded",
        assistant_content="Canonical answer",
    )
    return conversation


@pytest.fixture
async def application_case(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await store.resolve_external_identity(
            provider="owui",
            subject="owui-alice",
            email="alice@example.com",
            display_name="Alice",
            role="user",
            auth_method="owui_bearer",
            legacy_storage_namespace="alice@example.com",
        )
        conversation = await _terminal_turn(store, owner.user_id)
        (projection,) = await store.chat_projections.due()
        deleted = await store.conversations.create(user_id=owner.user_id)
        assert await store.conversations.delete(
            user_id=owner.user_id,
            conversation_id=deleted.conversation_id,
        )
        (deletion,) = await store.chat_projections.due_deletions()
        issued = await store.create_personal_token(
            user_id=owner.user_id,
            name="Automation",
            scopes=["compat:full"],
            expires_at=None,
        )
        # These tests need real contention without a five-second timeout per case.
        store._conn.execute("PRAGMA busy_timeout = 1")
        yield _ApplicationCase(
            store=store,
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            projection_id=projection.projection_id,
            deletion_id=deletion.deletion_id,
            token_id=issued.record.token_id,
            token=issued.token,
        )
    finally:
        store.close()


_WRITERS = [
    pytest.param("mark_enqueued", "app_chat_projections", "UPDATE", id="mark-enqueued"),
    pytest.param("mark_failed", "app_chat_projections", "UPDATE", id="mark-failed"),
    pytest.param("retry_now", "app_chat_projections", "UPDATE", id="retry-projections"),
    pytest.param("reset_all", "app_chat_projections", "UPDATE", id="reset-projections"),
    pytest.param(
        "mark_deletion_completed",
        "app_chat_projection_deletions",
        "UPDATE",
        id="complete-deletion",
    ),
    pytest.param(
        "mark_deletion_failed",
        "app_chat_projection_deletions",
        "UPDATE",
        id="fail-deletion",
    ),
    pytest.param(
        "retry_deletions_now",
        "app_chat_projection_deletions",
        "UPDATE",
        id="retry-deletions",
    ),
    pytest.param("revoke_token", "personal_access_tokens", "UPDATE", id="revoke-token"),
    pytest.param("authenticate_token", "personal_access_tokens", "UPDATE", id="record-token-use"),
    pytest.param("delete_tokens", "personal_access_tokens", "DELETE", id="delete-tokens"),
    pytest.param("update_display_name", "app_users", "UPDATE", id="update-display-name"),
]


async def _write(case: _ApplicationCase, writer: str):
    store = case.store
    projections = store.chat_projections
    operations = {
        "mark_enqueued": lambda: projections.mark_enqueued(projection_id=case.projection_id),
        "mark_failed": lambda: projections.mark_failed(
            projection_id=case.projection_id,
            error="handoff failed",
            retry_interval_s=60,
        ),
        "retry_now": projections.retry_now,
        "reset_all": projections.reset_all,
        "mark_deletion_completed": lambda: projections.mark_deletion_completed(
            deletion_id=case.deletion_id,
        ),
        "mark_deletion_failed": lambda: projections.mark_deletion_failed(
            deletion_id=case.deletion_id,
            error="handoff failed",
            retry_interval_s=60,
        ),
        "retry_deletions_now": projections.retry_deletions_now,
        "revoke_token": lambda: store.revoke_personal_token(
            user_id=case.user_id,
            token_id=case.token_id,
        ),
        "authenticate_token": lambda: store.authenticate_personal_token(case.token),
        "delete_tokens": lambda: store.delete_personal_tokens(user_id=case.user_id),
        "update_display_name": lambda: store.update_display_name(
            user_id=case.user_id,
            display_name="Alice Builder",
        ),
    }
    return await operations[writer]()


def _rows(connection: sqlite3.Connection, table: str) -> list[tuple]:
    # Table names come only from the fixed test parameters above.
    query = f"SELECT * FROM {table} ORDER BY 1"  # noqa: S608 — fixed test table names
    return [tuple(row) for row in connection.execute(query)]


async def _assert_canonical_write_succeeds(case: _ApplicationCase) -> None:
    assert not case.store._conn.in_transaction
    started = await case.store.conversations.begin_run(
        user_id=case.user_id,
        conversation_id=case.conversation_id,
        user_content="A later canonical write still works",
    )
    assert started is not None
    assert not case.store._conn.in_transaction


@pytest.mark.parametrize(("writer", "table", "operation"), _WRITERS)
async def test_writer_busy_timeout_preserves_connection_and_rows(
    application_case,
    writer,
    table,
    operation,
):
    case = application_case
    with closing(sqlite3.connect(case.store.path)) as competing_writer:
        before = _rows(competing_writer, table)
        competing_writer.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                await _write(case, writer)
        finally:
            competing_writer.rollback()
        assert not case.store._conn.in_transaction
        assert _rows(competing_writer, table) == before
    await _assert_canonical_write_succeeds(case)


@pytest.mark.parametrize(("writer", "table", "operation"), _WRITERS)
async def test_writer_statement_failure_rolls_back_and_allows_canonical_write(
    application_case,
    writer,
    table,
    operation,
):
    case = application_case
    with closing(sqlite3.connect(case.store.path)) as observer:
        before = _rows(observer, table)
        observer.execute(
            f"CREATE TRIGGER reject_test_write BEFORE {operation} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected statement failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected statement failure"):
            await _write(case, writer)
        assert not case.store._conn.in_transaction
        assert _rows(observer, table) == before
        observer.execute("DROP TRIGGER reject_test_write")
    await _assert_canonical_write_succeeds(case)


@pytest.mark.parametrize(
    ("writer", "table", "id_column"),
    [
        ("retry_now", "app_chat_projections", "projection_id"),
        ("retry_deletions_now", "app_chat_projection_deletions", "deletion_id"),
    ],
)
async def test_retry_batch_rolls_back_earlier_rows_when_later_row_fails(
    application_case,
    writer,
    table,
    id_column,
):
    case = application_case
    if writer == "retry_now":
        await _terminal_turn(case.store, case.user_id)
    else:
        deleted = await case.store.conversations.create(user_id=case.user_id)
        assert await case.store.conversations.delete(
            user_id=case.user_id,
            conversation_id=deleted.conversation_id,
        )

    with closing(sqlite3.connect(case.store.path)) as observer:
        query = f"SELECT {id_column} FROM {table} ORDER BY 1"  # noqa: S608 — fixed test identifiers
        ids = [row[0] for row in observer.execute(query)]
        assert len(ids) == 2
        observer.executemany(
            f"UPDATE {table} SET next_attempt_at = ? WHERE {id_column} = ?",  # noqa: S608 — fixed test identifiers
            [("2001-01-01T00:00:00+00:00", ids[0]), ("2001-01-02T00:00:00+00:00", ids[1])],
        )
        observer.commit()
        before = _rows(observer, table)
        # The first UPDATE succeeds; only the second executemany item is rejected.
        observer.execute(
            f"CREATE TRIGGER reject_second_retry BEFORE UPDATE OF next_attempt_at ON {table} "
            "WHEN OLD.next_attempt_at = '2001-01-02T00:00:00+00:00' "
            "BEGIN SELECT RAISE(ABORT, 'second retry failed'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="second retry failed"):
            await _write(case, writer)
        assert not case.store._conn.in_transaction
        assert _rows(case.store._conn, table) == before
        assert _rows(observer, table) == before
        observer.execute("DROP TRIGGER reject_second_retry")
    await _assert_canonical_write_succeeds(case)


async def test_commit_failure_rolls_back_projection_receipt_and_trigger_writes(application_case):
    case = application_case
    with closing(sqlite3.connect(case.store.path)) as observer:
        before = _rows(observer, "app_chat_projections")
        observer.executescript(
            "CREATE TABLE commit_parent (id INTEGER PRIMARY KEY);"
            "CREATE TABLE commit_child (parent_id INTEGER REFERENCES commit_parent(id) "
            "DEFERRABLE INITIALLY DEFERRED);"
            "CREATE TRIGGER fail_receipt_commit AFTER UPDATE OF enqueued_at "
            "ON app_chat_projections BEGIN INSERT INTO commit_child VALUES (1); END;"
        )
        # SQLite accepts the UPDATE, then rejects its deferred constraint at COMMIT.
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            await _write(case, "mark_enqueued")
        assert not case.store._conn.in_transaction
        assert _rows(observer, "app_chat_projections") == before
        assert _rows(observer, "commit_child") == []
        observer.execute("DROP TRIGGER fail_receipt_commit")
    await _assert_canonical_write_succeeds(case)
