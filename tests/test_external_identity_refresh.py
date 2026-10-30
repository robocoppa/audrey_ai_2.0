"""External identity polling stays read-only until canonical state needs a write."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import closing

import pytest

from audrey.app_state import ApplicationStore, IdentityConflictError
from audrey.app_state import store as store_module


@pytest.fixture
def identity_store(tmp_path, monkeypatch):
    clock = ["2026-10-30T12:00:00.000000+00:00"]
    monkeypatch.setattr(store_module, "_utc_now", lambda: clock[0])
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        yield store, clock
    finally:
        store.close()


async def _resolve(store, **overrides):
    options = {
        "provider": "cloudflare_access",
        "subject": "access-alice",
        "email": "alice@example.com",
        "display_name": "Alice",
        "role": "user",
        "auth_method": "cloudflare_access",
        "sync_role": False,
        "sync_display_name": False,
    }
    options.update(overrides)
    return await store.resolve_external_identity(**options)


def _identity_state(store):
    return dict(store._conn.execute(
        "SELECT u.display_name, u.role, u.status, u.updated_at, "
        "i.email, i.created_at, i.last_seen_at "
        "FROM app_users u JOIN external_identities i ON i.user_id = u.user_id "
        "WHERE i.provider = 'cloudflare_access' AND i.subject = 'access-alice'"
    ).fetchone())


async def test_unchanged_requests_neither_begin_nor_take_the_write_lock(identity_store):
    store, clock = identity_store
    first = await _resolve(store)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:04:59.999999+00:00"
    statements = []
    store._conn.set_trace_callback(statements.append)
    changes = store._conn.total_changes
    with closing(sqlite3.connect(store.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        try:
            # A write attempt would fail quickly while this connection holds
            # SQLite's writer. WAL readers can still authenticate immediately.
            store._conn.execute("PRAGMA busy_timeout = 1")
            for _ in range(3):
                assert await _resolve(store) == first
        finally:
            writer.rollback()
    store._conn.set_trace_callback(None)
    assert store._conn.total_changes == changes
    assert not any(statement.upper().startswith("BEGIN") for statement in statements)
    assert _identity_state(store) == before
    assert not store._conn.in_transaction


async def test_role_authoritative_unchanged_requests_are_also_read_only(identity_store):
    store, clock = identity_store
    before = await _resolve(store, role="admin", sync_role=True, sync_display_name=True)
    clock[0] = "2026-10-30T12:01:00.000000+00:00"
    statements = []
    changes = store._conn.total_changes
    store._conn.set_trace_callback(statements.append)
    try:
        after = await _resolve(store, role="admin", sync_role=True, sync_display_name=True)
    finally:
        store._conn.set_trace_callback(None)
    assert after == before
    assert after.groups == frozenset({"users", "admins"})
    assert store._conn.total_changes == changes
    assert not any(statement.upper().startswith("BEGIN") for statement in statements)


async def test_last_seen_expires_at_five_minutes_without_changing_the_account(identity_store):
    store, clock = identity_store
    await _resolve(store)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:05:00.000000+00:00"
    changes = store._conn.total_changes
    await _resolve(store)
    after = _identity_state(store)
    assert after == {**before, "last_seen_at": clock[0]}
    assert store._conn.total_changes == changes + 1
    clock[0] = "2026-10-30T12:05:01.000000+00:00"
    await _resolve(store)
    assert _identity_state(store) == after
    assert store._conn.total_changes == changes + 1


async def test_case_only_provider_metadata_change_is_immediate_but_not_an_account_edit(identity_store):
    store, clock = identity_store
    first = await _resolve(store)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:01:00.000000+00:00"
    after = await _resolve(store, email="ALICE@example.com")
    assert after == first
    assert _identity_state(store) == {**before, "email": "ALICE@example.com"}
    assert store._conn.execute("SELECT count(*) FROM external_identities").fetchone()[0] == 1


async def test_authoritative_profile_role_and_groups_change_before_last_seen_expires(identity_store):
    store, clock = identity_store
    await _resolve(store)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:00:01.000000+00:00"
    promoted = await _resolve(
        store, display_name="Alice Builder", role="admin", sync_role=True, sync_display_name=True,
    )
    assert promoted.display_name == "Alice Builder"
    assert promoted.role == "admin"
    assert promoted.groups == frozenset({"users", "admins"})
    assert _identity_state(store) == {
        **before, "display_name": "Alice Builder", "role": "admin", "updated_at": clock[0],
    }
    clock[0] = "2026-10-30T12:00:02.000000+00:00"
    demoted = await _resolve(store, sync_role=True, sync_display_name=False)
    assert demoted.role == "user"
    assert demoted.groups == frozenset({"users"})
    assert demoted.display_name == "Alice Builder"
    assert _identity_state(store)["updated_at"] == clock[0]
    assert _identity_state(store)["last_seen_at"] == before["last_seen_at"]


async def test_non_authoritative_evidence_preserves_local_profile_and_memberships(identity_store):
    store, clock = identity_store
    first = await _resolve(store, role="admin", sync_role=True)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:01:00.000000+00:00"
    after = await _resolve(store, role="user", display_name="Provider name")
    assert after == first
    assert _identity_state(store) == before


@pytest.mark.parametrize("role", ["user", "admin"])
async def test_authoritative_membership_repairs_remain_immediate(identity_store, role):
    store, clock = identity_store
    owner = await _resolve(store, role=role, sync_role=True)
    before = _identity_state(store)
    with closing(sqlite3.connect(store.path)) as writer:
        writer.execute("DELETE FROM user_group_memberships WHERE user_id = ?", (owner.user_id,))
        if role == "user":
            writer.execute(
                "INSERT INTO user_group_memberships VALUES (?, 'admins', NULL, ?)",
                (owner.user_id, clock[0]),
            )
        writer.commit()
    clock[0] = "2026-10-30T12:01:00.000000+00:00"
    repaired = await _resolve(store, role=role, sync_role=True)
    assert repaired.groups == frozenset({"users", "admins"} if role == "admin" else {"users"})
    assert _identity_state(store) == before


async def test_new_provider_link_keeps_account_timestamp_and_access(identity_store):
    store, clock = identity_store
    owner = await _resolve(store, role="admin", sync_role=True)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:01:00.000000+00:00"
    linked = await _resolve(
        store, subject="new-access-subject", role="user", display_name="Replacement",
        sync_role=True, sync_display_name=True,
    )
    assert linked.user_id == owner.user_id
    assert linked.role == "admin"
    assert linked.groups == owner.groups
    assert linked.display_name == owner.display_name
    assert _identity_state(store) == before
    row = store._conn.execute(
        "SELECT last_seen_at FROM external_identities WHERE subject = 'new-access-subject'"
    ).fetchone()
    assert row["last_seen_at"] == clock[0]


async def test_fast_path_reads_disabled_pending_role_and_group_changes_every_time(identity_store):
    store, _clock = identity_store
    owner = await _resolve(store, role="admin", sync_role=True)
    for status in ("pending", "disabled", "active"):
        with closing(sqlite3.connect(store.path)) as writer:
            writer.execute(
                "UPDATE app_users SET status = ?, role = 'user', display_name = 'Local edit' "
                "WHERE user_id = ?", (status, owner.user_id),
            )
            writer.execute(
                "DELETE FROM user_group_memberships WHERE user_id = ? AND group_id = 'admins'",
                (owner.user_id,),
            )
            writer.commit()
        changes = store._conn.total_changes
        current = await _resolve(store)
        assert current.status == status
        assert current.role == "user"
        assert current.groups == frozenset({"users"})
        assert current.display_name == "Local edit"
        assert store._conn.total_changes == changes


async def test_due_write_rechecks_disabled_and_deletion_authority_after_initial_read(
    identity_store, monkeypatch,
):
    store, clock = identity_store
    owner = await _resolve(store)
    clock[0] = "2026-10-30T12:05:00.000000+00:00"
    read = store._identity_row_locked
    reads = []

    def disable_after_first_read(provider, subject):
        row = read(provider, subject)
        reads.append(row)
        if len(reads) == 1:
            with closing(sqlite3.connect(store.path)) as writer:
                writer.execute(
                    "UPDATE app_users SET status = 'disabled', display_name = 'Disabled owner' "
                    "WHERE user_id = ?", (owner.user_id,),
                )
                writer.execute("DELETE FROM user_group_memberships WHERE user_id = ?", (owner.user_id,))
                writer.execute(
                    "INSERT INTO account_deletion_requests VALUES (?, 'purge-alice', ?, ?)",
                    (owner.user_id, owner.storage_namespace, clock[0]),
                )
                writer.commit()
        return row

    monkeypatch.setattr(store, "_identity_row_locked", disable_after_first_read)
    resolved = await _resolve(store)
    assert len(reads) >= 2
    assert resolved.status == "disabled"
    assert resolved.display_name == "Disabled owner"
    assert resolved.groups == frozenset()
    assert store._conn.execute("SELECT count(*) FROM account_deletion_requests").fetchone()[0] == 1
    with pytest.raises(IdentityConflictError, match="deletion is in progress"):
        await _resolve(store, subject="new-access-subject")
    assert store._conn.execute("SELECT count(*) FROM external_identities").fetchone()[0] == 1
    assert not store._conn.in_transaction


async def test_concurrent_first_login_and_expired_refresh_recheck_inside_transaction(
    identity_store, monkeypatch,
):
    store, clock = identity_store
    other = ApplicationStore(store.path)
    try:
        barrier = threading.Barrier(2)

        def synchronize_first_read(application):
            read = application._identity_row_locked
            first_read = True

            def synchronized(provider, subject):
                nonlocal first_read
                row = read(provider, subject)
                if first_read:
                    first_read = False
                    barrier.wait(timeout=5)
                return row

            monkeypatch.setattr(application, "_identity_row_locked", synchronized)

        synchronize_first_read(store)
        synchronize_first_read(other)
        first, second = await asyncio.gather(_resolve(store), _resolve(other))
        assert first == second
        assert store._conn.execute("SELECT count(*) FROM app_users").fetchone()[0] == 1
        assert store._conn.execute("SELECT count(*) FROM external_identities").fetchone()[0] == 1
        clock[0] = "2026-10-30T12:05:00.000000+00:00"
        synchronize_first_read(store)
        synchronize_first_read(other)
        statements = []
        store._conn.set_trace_callback(statements.append)
        other._conn.set_trace_callback(statements.append)
        first, second = await asyncio.gather(_resolve(store), _resolve(other))
        assert first == second
        updates = [statement for statement in statements if statement.startswith("UPDATE external_identities")]
        assert len(updates) == 1
        assert _identity_state(store)["last_seen_at"] == clock[0]
        assert _identity_state(store)["updated_at"] != clock[0]
        assert not store._conn.in_transaction
        assert not other._conn.in_transaction
    finally:
        store._conn.set_trace_callback(None)
        other.close()


@pytest.mark.parametrize("last_seen", ["", "not-a-date", "2026-10-30T12:00:00"])
async def test_invalid_last_seen_metadata_is_repaired_without_account_changes(identity_store, last_seen):
    store, _clock = identity_store
    await _resolve(store)
    before = _identity_state(store)
    with closing(sqlite3.connect(store.path)) as writer:
        writer.execute("UPDATE external_identities SET last_seen_at = ?", (last_seen,))
        writer.commit()
    await _resolve(store)
    assert _identity_state(store) == before


async def test_failed_refresh_rolls_back_profile_and_keeps_the_connection_usable(identity_store):
    store, clock = identity_store
    await _resolve(store)
    before = _identity_state(store)
    clock[0] = "2026-10-30T12:05:00.000000+00:00"
    with closing(sqlite3.connect(store.path)) as observer:
        observer.execute(
            "CREATE TRIGGER reject_identity_refresh BEFORE UPDATE ON external_identities "
            "BEGIN SELECT RAISE(ABORT, 'identity refresh failed'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="identity refresh failed"):
            await _resolve(store, display_name="Alice Builder", sync_display_name=True)
        assert _identity_state(store) == before
        assert not store._conn.in_transaction
        observer.execute("DROP TRIGGER reject_identity_refresh")
    refreshed = await _resolve(store, display_name="Alice Builder", sync_display_name=True)
    assert refreshed.display_name == "Alice Builder"
    conversation = await store.conversations.create(user_id=refreshed.user_id)
    assert conversation.user_id == refreshed.user_id
    assert not store._conn.in_transaction
