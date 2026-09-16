"""Native account approval, access-group, and model-catalog contracts."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey import admin_cli
from audrey.app_state import AccountAdministrationError, ApplicationStore
from audrey.auth import require_admin_principal, require_principal
from audrey.identity import Principal
from audrey.model_catalog import catalog_for_principal, configured_models, discover_models
from audrey.models.ollama import OllamaError
from audrey.routes.app import router

_DIRECT_MODEL = "qwen-test:latest"


class _OllamaInventory:
    def __init__(self, *models: str) -> None:
        self.models = models

    async def tags(self):
        return [
            {"name": model, "model": model}
            for model in self.models
        ]


class _UnavailableOllama:
    async def tags(self):
        raise OllamaError("offline")


def _cfg():
    return SimpleNamespace(
        raw={
            "passthrough": {
                "enabled": True,
                "allowed_models": [_DIRECT_MODEL],
                "think": None,
            },
            "native_models": {
                "direct_defaults": {
                    "enabled": True,
                    "audience": "admins",
                    "num_ctx": 32768,
                    "max_tokens": 4096,
                },
                "entries": {
                    _DIRECT_MODEL: {
                        "label": "Qwen Test",
                        "description": "A direct test model.",
                        "audience": "testers",
                        "capabilities": ["text", "thinking"],
                    }
                },
            },
        },
        model_registry={
            "general": [
                {
                    "name": _DIRECT_MODEL,
                    "priority": 100,
                    "location": "local",
                }
            ]
        },
        timeouts={"medium": 180},
    )


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str,
    email: str,
    role: str = "user",
    provider: str = "owui",
    initial_status: str = "active",
) -> Principal:
    return await store.resolve_external_identity(
        provider=provider,
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role=role,
        auth_method=f"{provider}_bearer",
        legacy_storage_namespace=email if provider == "owui" else None,
        sync_role=provider == "owui",
        sync_display_name=provider == "owui",
        initial_status=initial_status,
    )


async def test_account_approval_groups_and_audit_are_atomic(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    admin = await _resolve(
        store,
        subject="owui-admin",
        email="admin@example.com",
        role="admin",
    )
    pending = await _resolve(
        store,
        subject="cf-pending",
        email="pending@example.com",
        provider="cloudflare_access",
        initial_status="pending",
    )
    try:
        assert pending.status == "pending"
        assert pending.groups == frozenset()

        approved = await store.admin_update_user(
            actor_user_id=admin.user_id,
            target_user_id=pending.user_id,
            status="active",
            groups=["users", "testers"],
            action="approve_user",
        )
        assert approved.status == "active"
        assert approved.groups == ("testers", "users")
        assert approved.role == "user"

        filtered = await store.list_admin_users(status="active", group="testers")
        assert [record.user_id for record in filtered] == [pending.user_id]
        searched = await store.list_admin_users(search="PENDING")
        assert [record.user_id for record in searched] == [pending.user_id]

        with pytest.raises(AccountAdministrationError, match="own admin access"):
            await store.admin_update_user(
                actor_user_id=admin.user_id,
                target_user_id=admin.user_id,
                groups=["users"],
            )
        unchanged = await store.get_admin_user(user_id=admin.user_id)
        assert unchanged is not None
        assert unchanged.groups == ("admins", "users")

        with pytest.raises(AccountAdministrationError, match="active accounts"):
            await store.admin_update_user(
                actor_user_id=admin.user_id,
                target_user_id=pending.user_id,
                groups=[],
            )
        still_approved = await store.get_admin_user(user_id=pending.user_id)
        assert still_approved == approved
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        audits = connection.execute(
            "SELECT action, actor_user_id, target_id, before_json, after_json "
            "FROM admin_audit_events ORDER BY created_at, event_id"
        ).fetchall()
    assert len(audits) == 1
    assert audits[0][0:3] == ("approve_user", admin.user_id, pending.user_id)
    assert '"status":"pending"' in audits[0][3]
    assert '"groups":["testers","users"]' in audits[0][4]


async def test_bootstrap_admin_activates_exact_account_and_is_audited(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    pending = await _resolve(
        store,
        subject="cf-first-admin",
        email="first-admin@example.com",
        provider="cloudflare_access",
        initial_status="pending",
    )
    try:
        bootstrapped = await store.bootstrap_admin(user_id=pending.user_id)
        assert bootstrapped.status == "active"
        assert bootstrapped.role == "admin"
        assert bootstrapped.groups == ("admins", "users")
        with pytest.raises(AccountAdministrationError, match="does not exist"):
            await store.bootstrap_admin(user_id="usr_missing")
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        event = connection.execute(
            "SELECT actor_user_id, action FROM admin_audit_events"
        ).fetchone()
    assert event == (None, "bootstrap_admin")


async def test_operator_cli_bootstraps_the_exact_pending_email(
    monkeypatch,
    tmp_path,
    capsys,
):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    pending = await _resolve(
        store,
        subject="cf-cli-admin",
        email="cli-admin@example.com",
        provider="cloudflare_access",
        initial_status="pending",
    )
    store.close()
    monkeypatch.setattr(
        admin_cli,
        "get_config",
        lambda: SimpleNamespace(raw={"application": {"sqlite_path": str(path)}}),
    )

    parsed = admin_cli._parser().parse_args(
        ["grant-admin", "--email", "cli-admin@example.com"]
    )
    assert parsed.user_id is None
    assert parsed.email == "cli-admin@example.com"

    assert await admin_cli._grant_admin(email="CLI-ADMIN@example.com") == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert f'"user_id": "{pending.user_id}"' in output.out

    reopened = ApplicationStore(path)
    try:
        record = await reopened.get_admin_user(user_id=pending.user_id)
        assert record is not None
        assert record.status == "active"
        assert record.groups == ("admins", "users")
    finally:
        reopened.close()


async def test_operator_email_bootstrap_refuses_ambiguous_accounts(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        await _resolve(
            store,
            subject="cf-shared-email",
            email="shared@example.com",
            provider="cloudflare_access",
            initial_status="pending",
        )
        await _resolve(
            store,
            subject="owui-shared-email",
            email="shared@example.com",
            provider="owui",
            initial_status="pending",
        )

        with pytest.raises(AccountAdministrationError, match="multiple accounts"):
            await store.bootstrap_admin_by_email(email="shared@example.com")
    finally:
        store.close()


async def test_catalog_filters_groups_and_applies_database_policy(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = await _resolve(
        store,
        subject="owui-admin",
        email="admin@example.com",
        role="admin",
    )
    user = await _resolve(
        store,
        subject="owui-user",
        email="user@example.com",
    )
    tester = await store.admin_update_user(
        actor_user_id=admin.user_id,
        target_user_id=user.user_id,
        groups=["users", "testers"],
    )
    tester_principal = await _resolve(
        store,
        subject="owui-user",
        email="user@example.com",
    )
    assert tester.groups == ("testers", "users")
    try:
        configured = configured_models(_cfg())
        direct = next(model for model in configured if model.kind == "direct")
        assert direct.id == f"direct/{_DIRECT_MODEL}"
        assert direct.num_ctx == 32768
        assert direct.max_tokens == 4096

        ordinary_ids = {
            model.id for model in await catalog_for_principal(_cfg(), store, user)
        }
        tester_ids = {
            model.id
            for model in await catalog_for_principal(_cfg(), store, tester_principal)
        }
        assert "auto" in ordinary_ids
        assert direct.id not in ordinary_ids
        assert direct.id in tester_ids

        await store.set_model_access_policy(
            actor_user_id=admin.user_id,
            model_id=direct.id,
            enabled=True,
            audience="users",
        )
        assert direct.id in {
            model.id for model in await catalog_for_principal(_cfg(), store, user)
        }
        await store.set_model_access_policy(
            actor_user_id=admin.user_id,
            model_id="research",
            enabled=False,
            audience="users",
        )
        assert "research" not in {
            model.id for model in await catalog_for_principal(_cfg(), store, admin)
        }
        assert "research" in {
            model.id
            for model in await catalog_for_principal(
                _cfg(), store, admin, include_hidden=True
            )
        }
    finally:
        store.close()


async def test_native_inventory_discovers_every_ollama_tag_as_admin_only():
    inventory = await discover_models(
        _cfg(),
        _OllamaInventory("unlisted:latest", "cloud-model:cloud", "unlisted:latest"),
    )

    assert inventory.source == "ollama"
    assert inventory.warning == ""
    direct = [model for model in inventory.models if model.kind == "direct"]
    assert [model.concrete_model for model in direct] == [
        "cloud-model:cloud",
        "unlisted:latest",
    ]
    assert all(model.enabled for model in direct)
    assert all(model.audience == "admins" for model in direct)
    assert [model.presentation for model in direct] == ["cloud", "local"]


async def test_native_inventory_falls_back_to_configuration_when_ollama_is_down():
    inventory = await discover_models(_cfg(), _UnavailableOllama())

    assert inventory.source == "configuration"
    assert "unavailable" in inventory.warning
    assert f"direct/{_DIRECT_MODEL}" in {model.id for model in inventory.models}


def test_admin_can_manage_a_dynamically_discovered_model(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(
        _resolve(
            store,
            subject="owui-admin",
            email="admin@example.com",
            role="admin",
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.state.ollama = _OllamaInventory("newly-pulled:latest")
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    try:
        with TestClient(app) as client:
            listed = client.get("/api/admin/models")
            assert listed.status_code == 200
            body = listed.json()
            assert body["source"] == "ollama"
            assert body["warning"] == ""
            dynamic = next(
                item for item in body["items"]
                if item["id"] == "direct/newly-pulled:latest"
            )
            assert dynamic["enabled"] is True
            assert dynamic["audience"] == "admins"
            assert dynamic["policy_overridden"] is False

            changed = client.patch(
                "/api/admin/models/direct/newly-pulled:latest",
                json={"enabled": True, "audience": "testers"},
            )
            assert changed.status_code == 200
            assert changed.json()["audience"] == "testers"
            assert changed.json()["policy_overridden"] is True
    finally:
        store.close()


def test_admin_routes_mutate_exact_accounts_and_model_policies(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    admin = asyncio.run(
        _resolve(
            store,
            subject="owui-admin",
            email="admin@example.com",
            role="admin",
        )
    )
    pending = asyncio.run(
        _resolve(
            store,
            subject="cf-pending",
            email="pending@example.com",
            provider="cloudflare_access",
            initial_status="pending",
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.state.ollama = _OllamaInventory(_DIRECT_MODEL, "admin-only:latest")
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    try:
        with TestClient(app) as client:
            listed = client.get("/api/admin/users?status=pending")
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()["items"]] == [pending.user_id]

            approved = client.post(
                f"/api/admin/users/{pending.user_id}/approve",
                json={"tester": True},
            )
            assert approved.status_code == 200
            assert approved.json()["groups"] == ["testers", "users"]

            model_id = f"direct/{_DIRECT_MODEL}"
            changed = client.patch(
                f"/api/admin/models/{model_id}",
                json={"enabled": True, "audience": "users"},
            )
            assert changed.status_code == 200
            assert changed.json()["id"] == model_id
            assert changed.json()["audience"] == "users"
            assert changed.json()["policy_overridden"] is True

            reset = client.delete(f"/api/admin/model-policies/{model_id}")
            assert reset.status_code == 200
            assert reset.json()["id"] == model_id
            assert reset.json()["audience"] == "testers"
            assert reset.json()["policy_overridden"] is False
            assert asyncio.run(store.list_model_access_policies()) == ()

            missing = client.patch(
                "/api/admin/users/usr_missing",
                json={"status": "disabled"},
            )
            assert missing.status_code == 404
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        model_actions = connection.execute(
            "SELECT action, target_id FROM admin_audit_events "
            "WHERE target_type = 'model' ORDER BY created_at, event_id"
        ).fetchall()
    assert model_actions == [
        ("set_model_policy", model_id),
        ("delete_model_policy", model_id),
    ]


def test_native_model_route_exposes_only_the_callers_catalog(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(
        _resolve(
            store,
            subject="owui-admin",
            email="admin@example.com",
            role="admin",
        )
    )
    ordinary = asyncio.run(
        _resolve(
            store,
            subject="owui-user",
            email="user@example.com",
        )
    )
    asyncio.run(
        store.admin_update_user(
            actor_user_id=admin.user_id,
            target_user_id=ordinary.user_id,
            groups=["users", "testers"],
        )
    )
    tester = asyncio.run(
        _resolve(
            store,
            subject="owui-user",
            email="user@example.com",
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.state.ollama = _OllamaInventory(_DIRECT_MODEL, "admin-only:latest")
    app.include_router(router)
    try:
        app.dependency_overrides[require_principal] = lambda: ordinary
        ordinary_response = TestClient(app).get("/api/models")
        assert ordinary_response.status_code == 200
        ordinary_ids = {item["id"] for item in ordinary_response.json()["items"]}
        assert f"direct/{_DIRECT_MODEL}" not in ordinary_ids

        app.dependency_overrides[require_principal] = lambda: tester
        tester_response = TestClient(app).get("/api/models")
        assert tester_response.status_code == 200
        tester_items = tester_response.json()["items"]
        direct = next(item for item in tester_items if item["kind"] == "direct")
        assert direct == {
            "id": f"direct/{_DIRECT_MODEL}",
            "label": "Qwen Test",
            "description": "A direct test model.",
            "kind": "direct",
            "mode": "direct",
            "presentation": "local",
            "capabilities": ["text", "thinking"],
            "enabled": True,
            "audience": "testers",
        }
        assert "concrete_model" not in tester_response.text

        app.dependency_overrides[require_principal] = lambda: admin
        admin_response = TestClient(app).get("/api/models")
        assert admin_response.status_code == 200
        admin_ids = {item["id"] for item in admin_response.json()["items"]}
        assert f"direct/{_DIRECT_MODEL}" in admin_ids
        assert "direct/admin-only:latest" in admin_ids
    finally:
        store.close()


def test_admin_routes_reject_personal_tokens_even_for_admin_account(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(
        _resolve(
            store,
            subject="owui-admin",
            email="admin@example.com",
            role="admin",
        )
    )
    issued = asyncio.run(
        store.create_personal_token(
            user_id=admin.user_id,
            name="No administration",
            scopes=["compat:full"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=1)).isoformat(),
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.include_router(router)
    try:
        response = TestClient(app).get(
            "/api/admin/users",
            headers={"Authorization": f"Bearer {issued.token}"},
        )
    finally:
        store.close()

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "External provider authentication is required for this operation."
    )


async def test_require_admin_principal_rejects_non_admin():
    user = Principal(
        user_id="usr_user",
        storage_namespace="user@example.com",
        provider="owui",
        provider_subject="owui-user",
        email="user@example.com",
        display_name="User",
        role="user",
        status="active",
        auth_method="owui_bearer",
        groups=frozenset({"users"}),
    )
    with pytest.raises(HTTPException) as exc:
        await require_admin_principal(principal=user)
    assert exc.value.status_code == 403
