"""Canonical account snapshots used to revalidate server-owned Private work."""

from __future__ import annotations

import sqlite3

import pytest

from audrey.app_state import ApplicationStore


@pytest.fixture
def store(tmp_path):
    instance = ApplicationStore(tmp_path / "app.sqlite")
    try:
        yield instance
    finally:
        instance.close()


async def _owner(store, *, subject="owner", role="user"):
    return await store.resolve_external_identity(
        provider="cloudflare_access",
        subject=subject,
        email=f"{subject}@example.com",
        display_name=subject.title(),
        role=role,
        auth_method="cloudflare_access",
    )


async def test_task_principal_refreshes_current_role_groups_and_account_status(store):
    admin = await _owner(store, subject="admin", role="admin")
    owner = await _owner(store)
    initial = await store.get_principal(user_id=owner.user_id)
    assert initial is not None
    assert initial.status == "active"
    assert initial.role == "user"
    assert initial.groups == frozenset({"users"})

    await store.admin_update_user(
        actor_user_id=admin.user_id,
        target_user_id=owner.user_id,
        groups=["users", "admins", "testers"],
    )
    promoted = await store.get_principal(user_id=owner.user_id)
    assert promoted is not None
    assert promoted.role == "admin"
    assert promoted.is_admin
    assert promoted.is_tester
    assert promoted.groups == frozenset({"users", "admins", "testers"})
    assert promoted.storage_namespace == owner.storage_namespace
    assert initial.role == "user"

    await store.admin_update_user(
        actor_user_id=admin.user_id,
        target_user_id=owner.user_id,
        status="disabled",
        groups=["users", "bots"],
    )
    disabled = await store.get_principal(user_id=owner.user_id)
    assert disabled is not None
    assert disabled.status == "disabled"
    assert disabled.role == "user"
    assert disabled.groups == frozenset({"users", "bots"})
    assert not disabled.is_admin
    assert not disabled.is_tester


async def test_task_principal_reads_pending_account_before_browser_login(store):
    admin = await _owner(store, subject="admin", role="admin")
    invited = await store.admin_create_pending_user(
        actor_user_id=admin.user_id,
        email="invited@example.com",
        display_name="Invited user",
    )
    principal = await store.get_principal(user_id=invited.user_id)
    assert principal is not None
    assert principal.user_id == invited.user_id
    assert principal.email == "invited@example.com"
    assert principal.display_name == "Invited user"
    assert principal.status == "pending"
    assert principal.groups == frozenset()
    assert principal.storage_namespace.startswith("ns_")
    with sqlite3.connect(store.path) as db:
        assert db.execute(
            "SELECT provider FROM external_identities WHERE user_id=?",
            (invited.user_id,),
        ).fetchall() == [("manual",)]


async def test_task_principal_does_not_grant_request_credentials_or_touch_auth_records(store):
    owner = await _owner(store)
    token = await store.create_personal_token(
        user_id=owner.user_id,
        name="Existing bot token",
        scopes=["account:read", "compat:full"],
        expires_at=None,
    )

    def auth_snapshot():
        with sqlite3.connect(store.path) as db:
            return {
                "account": db.execute(
                    "SELECT * FROM app_users WHERE user_id=?", (owner.user_id,)
                ).fetchall(),
                "identity": db.execute(
                    "SELECT * FROM external_identities WHERE user_id=?", (owner.user_id,)
                ).fetchall(),
                "tokens": db.execute(
                    "SELECT * FROM personal_access_tokens WHERE user_id=?", (owner.user_id,)
                ).fetchall(),
            }

    before = auth_snapshot()
    changes = store._conn.total_changes
    attempted_writes = []

    def refuse_writes(action, *details):
        if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
            attempted_writes.append((action, details))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store._conn.set_authorizer(refuse_writes)
    try:
        principal = await store.get_principal(user_id=owner.user_id)
    finally:
        store._conn.set_authorizer(None)

    assert principal is not None
    assert principal.storage_namespace == owner.storage_namespace
    assert principal.provider == "internal"
    assert principal.provider_subject == ""
    assert principal.auth_method == "private_task"
    assert principal.token_id is None
    assert principal.scopes == frozenset()
    assert not attempted_writes
    assert store._conn.total_changes == changes
    assert auth_snapshot() == before
    assert (await store.list_personal_tokens(user_id=owner.user_id)) == (token.record,)
    assert token.record.last_used_at == ""


async def test_task_principal_excludes_unknown_and_pending_account_deletion(store):
    admin = await _owner(store, subject="admin", role="admin")
    owner = await _owner(store)
    assert await store.get_principal(user_id="usr_missing") is None
    assert await store.get_principal(user_id=owner.user_id) is not None

    purge_id, _ = await store.begin_admin_account_deletion(
        actor_user_id=admin.user_id, target_user_id=owner.user_id
    )
    with sqlite3.connect(store.path) as db:
        assert db.execute(
            "SELECT status FROM app_users WHERE user_id=?", (owner.user_id,)
        ).fetchone() == ("disabled",)
    assert await store.get_principal(user_id=owner.user_id) is None

    assert await store.finalize_account_deletion(user_id=owner.user_id, purge_id=purge_id)
    assert await store.get_principal(user_id=owner.user_id) is None
