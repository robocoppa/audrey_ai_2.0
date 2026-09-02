"""Contract tests for Audrey-owned users and external identity bindings."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from audrey.app_state import (
    ApplicationStore,
    IdentityConflictError,
    InvalidIdentityError,
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
        assert reopened.schema_version == 1
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
