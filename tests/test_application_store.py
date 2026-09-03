"""Contract tests for Audrey-owned users and external identity bindings."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3

import pytest

from audrey.app_state import (
    ApplicationStore,
    IdentityConflictError,
    InvalidIdentityError,
    PersonalTokenAuthenticationError,
)


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str = "owui-1",
    email: str = "alice@example.com",
    namespace: str | None = "alice@example.com",
):
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name="Alice",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=namespace,
    )


async def test_first_owui_login_keeps_exact_legacy_namespace(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        principal = await _resolve(store)
        assert principal.user_id.startswith("usr_")
        assert principal.storage_namespace == "alice@example.com"
        assert principal.provider == "owui"
        assert principal.provider_subject == "owui-1"
        assert principal.email == "alice@example.com"
        assert principal.auth_method == "owui_bearer"
    finally:
        store.close()


async def test_provider_subject_survives_email_change_without_namespace_rename(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        before = await _resolve(store)
        after = await _resolve(
            store,
            email="alice-renamed@example.com",
            namespace="alice-renamed@example.com",
        )
        assert after.user_id == before.user_id
        assert after.storage_namespace == "alice@example.com"
        assert after.email == "alice-renamed@example.com"
    finally:
        store.close()


async def test_similar_looking_emails_are_distinct_accounts(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        dotted = await _resolve(
            store,
            subject="owui-dot",
            email="audrey.test@example.com",
            namespace="audrey.test@example.com",
        )
        dashed = await _resolve(
            store,
            subject="owui-dash",
            email="audrey-test@example.com",
            namespace="audrey-test@example.com",
        )
        assert dotted.user_id != dashed.user_id
        assert dotted.storage_namespace == "audrey.test@example.com"
        assert dashed.storage_namespace == "audrey-test@example.com"
    finally:
        store.close()


async def test_different_subject_cannot_claim_existing_storage_namespace(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        await _resolve(store, subject="owui-a")
        with pytest.raises(IdentityConflictError, match="storage namespace"):
            await _resolve(store, subject="owui-b")
    finally:
        store.close()


async def test_new_nonlegacy_identity_receives_opaque_namespace(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        principal = await _resolve(store, namespace=None)
        assert principal.storage_namespace.startswith("ns_")
        assert "alice" not in principal.storage_namespace
    finally:
        store.close()


async def test_concurrent_first_login_creates_one_account(tmp_path):
    path = tmp_path / "app.sqlite"
    left = ApplicationStore(path)
    right = ApplicationStore(path)
    try:
        principals = await asyncio.gather(_resolve(left), _resolve(right))
        assert {p.user_id for p in principals} == {principals[0].user_id}
        assert {p.storage_namespace for p in principals} == {"alice@example.com"}
    finally:
        left.close()
        right.close()


async def test_binding_persists_across_reopen(tmp_path):
    path = tmp_path / "app.sqlite"
    first = ApplicationStore(path)
    before = await _resolve(first)
    first.close()

    reopened = ApplicationStore(path)
    try:
        after = await _resolve(reopened)
        assert after.user_id == before.user_id
        assert reopened.schema_version == 2
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        reopened.close()


async def test_invalid_identity_evidence_fails_before_writing(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        with pytest.raises(InvalidIdentityError, match="provider subject"):
            await _resolve(store, subject="")
        with pytest.raises(InvalidIdentityError, match="role"):
            await store.resolve_external_identity(
                provider="owui",
                subject="subject",
                email="alice@example.com",
                display_name="Alice",
                role="owner",
                auth_method="owui_bearer",
                legacy_storage_namespace="alice@example.com",
            )
    finally:
        store.close()


async def test_personal_token_secret_is_returned_once_and_hashed_at_rest(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    try:
        owner = await _resolve(store)
        issued = await store.create_personal_token(
            user_id=owner.user_id,
            name="Laptop eval",
            scopes=["compat:full", "account:read"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        assert issued.token.startswith(f"aud_{issued.record.token_id}.")
        assert issued.record.scopes == ("account:read", "compat:full")
        listed = await store.list_personal_tokens(user_id=owner.user_id)
        assert listed == (issued.record,)
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT secret_hash FROM personal_access_tokens WHERE token_id = ?",
                (issued.record.token_id,),
            ).fetchone()
        assert row is not None
        assert len(row[0]) == 64
        assert issued.token not in row[0]
        with sqlite3.connect(path) as conn:
            columns = {
                column[1] for column in conn.execute("PRAGMA table_info(personal_access_tokens)")
            }
        assert "token" not in columns
        assert "secret" not in columns
    finally:
        store.close()


async def test_personal_token_resolves_stable_principal_and_tracks_use(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _resolve(store)
        issued = await store.create_personal_token(
            user_id=owner.user_id,
            name="Automation",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        principal = await store.authenticate_personal_token(issued.token)
        assert principal.user_id == owner.user_id
        assert principal.storage_namespace == owner.storage_namespace
        assert principal.auth_method == "personal_token"
        assert principal.token_id == issued.record.token_id
        assert principal.scopes == frozenset({"compat:full"})
        listed = await store.list_personal_tokens(user_id=owner.user_id)
        assert listed[0].last_used_at
    finally:
        store.close()


async def test_wrong_secret_revoked_and_expired_tokens_are_rejected(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    try:
        owner = await _resolve(store)
        wrong = await store.create_personal_token(
            user_id=owner.user_id,
            name="Wrong secret",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        with pytest.raises(PersonalTokenAuthenticationError):
            await store.authenticate_personal_token(wrong.token + "x")

        revoked = await store.create_personal_token(
            user_id=owner.user_id,
            name="Revoked",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        assert await store.revoke_personal_token(
            user_id=owner.user_id,
            token_id=revoked.record.token_id,
        )
        with pytest.raises(PersonalTokenAuthenticationError):
            await store.authenticate_personal_token(revoked.token)

        expired = await store.create_personal_token(
            user_id=owner.user_id,
            name="Expired",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=1)).isoformat(),
        )
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE personal_access_tokens SET expires_at = ? WHERE token_id = ?",
                (
                    (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)).isoformat(),
                    expired.record.token_id,
                ),
            )
            conn.commit()
        with pytest.raises(PersonalTokenAuthenticationError):
            await store.authenticate_personal_token(expired.token)
    finally:
        store.close()


async def test_personal_token_owner_boundaries_and_scope_validation(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        alice = await _resolve(store, subject="owui-alice")
        bob = await _resolve(
            store,
            subject="owui-bob",
            email="bob@example.com",
            namespace="bob@example.com",
        )
        issued = await store.create_personal_token(
            user_id=alice.user_id,
            name="Alice token",
            scopes=["account:read"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        assert await store.list_personal_tokens(user_id=bob.user_id) == ()
        assert not await store.revoke_personal_token(
            user_id=bob.user_id,
            token_id=issued.record.token_id,
        )
        assert await store.authenticate_personal_token(issued.token)

        with pytest.raises(InvalidIdentityError, match="scope"):
            await store.create_personal_token(
                user_id=alice.user_id,
                name="Bad scope",
                scopes=["admin"],
                expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
            )
        with pytest.raises(InvalidIdentityError, match="at least one"):
            await store.create_personal_token(
                user_id=alice.user_id,
                name="No scope",
                scopes=[],
                expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
            )
        with pytest.raises(InvalidIdentityError, match="expiry"):
            await store.create_personal_token(
                user_id=alice.user_id,
                name="No expiry",
                scopes=["account:read"],
                expires_at="",
            )
    finally:
        store.close()


async def test_personal_token_bulk_delete_is_owner_bound_and_invalidates_secrets(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        alice = await _resolve(store, subject="owui-alice")
        bob = await _resolve(
            store,
            subject="owui-bob",
            email="bob@example.com",
            namespace="bob@example.com",
        )
        alice_token = await store.create_personal_token(
            user_id=alice.user_id,
            name="Alice token",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
        bob_token = await store.create_personal_token(
            user_id=bob.user_id,
            name="Bob token",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )

        assert await store.delete_personal_tokens(user_id=alice.user_id) == 1
        assert await store.list_personal_tokens(user_id=alice.user_id) == ()
        assert len(await store.list_personal_tokens(user_id=bob.user_id)) == 1
        with pytest.raises(PersonalTokenAuthenticationError):
            await store.authenticate_personal_token(alice_token.token)
        assert await store.authenticate_personal_token(bob_token.token)
        assert await store.delete_personal_tokens(user_id=alice.user_id) == 0
    finally:
        store.close()


async def test_v1_database_migrates_additively_without_changing_user_id(tmp_path):
    path = tmp_path / "app.sqlite"
    first = ApplicationStore(path)
    before = await _resolve(first)
    first.close()

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE personal_access_tokens")
        conn.execute("DELETE FROM app_schema_migrations WHERE version = 2")
        conn.commit()

    upgraded = ApplicationStore(path)
    try:
        after = await _resolve(upgraded)
        assert upgraded.schema_version == 2
        assert after.user_id == before.user_id
        assert await upgraded.list_personal_tokens(user_id=after.user_id) == ()
    finally:
        upgraded.close()
