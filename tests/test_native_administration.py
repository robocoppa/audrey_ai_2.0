"""Native account approval, access-group, and model-catalog contracts."""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from audrey import admin_cli
from audrey.app_state import (
    AccountAdministrationError,
    ApplicationStore,
    InvalidApplicationStateError,
)
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


async def test_admin_account_deletion_is_durable_and_refuses_self_or_edits(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = await _resolve(
        store, subject="delete-admin", email="admin@example.com", role="admin"
    )
    target = await _resolve(
        store, subject="delete-target", email="target@example.com",
        provider="cloudflare_access", initial_status="active",
    )
    try:
        conversation = await store.conversations.create(user_id=target.user_id)
        started = await store.conversations.begin_run(
            user_id=target.user_id,
            conversation_id=conversation.conversation_id,
            user_content="A running request",
        )
        assert started is not None
        with pytest.raises(AccountAdministrationError, match="active run"):
            await store.begin_admin_account_deletion(
                actor_user_id=admin.user_id, target_user_id=target.user_id,
            )
        await store.conversations.finish_run(
            user_id=target.user_id,
            run_id=started.run.run_id,
            outcome="cancelled",
            assistant_content="",
            finish_reason="cancelled",
            error_code="cancelled_by_user",
        )
        with pytest.raises(AccountAdministrationError, match="own account"):
            await store.begin_admin_account_deletion(
                actor_user_id=admin.user_id, target_user_id=admin.user_id,
            )
        purge_id, namespace = await store.begin_admin_account_deletion(
            actor_user_id=admin.user_id, target_user_id=target.user_id,
        )
        assert namespace == target.storage_namespace
        assert await store.begin_admin_account_deletion(
            actor_user_id=admin.user_id, target_user_id=target.user_id,
        ) == (purge_id, namespace)
        deleting = await store.get_admin_user(user_id=target.user_id)
        assert deleting is not None
        assert deleting.status == "disabled"
        assert deleting.deletion_pending is True
        assert deleting.groups == ()
        with pytest.raises(InvalidApplicationStateError, match="account is not active"):
            await store.conversations.begin_run(
                user_id=target.user_id,
                conversation_id=conversation.conversation_id,
                user_content="Too late",
            )
        with pytest.raises(AccountAdministrationError, match="deletion is in progress"):
            await store.admin_update_user(
                actor_user_id=admin.user_id, target_user_id=target.user_id,
                status="active", groups=["users"],
            )
        assert await store.finalize_account_deletion(
            user_id=target.user_id, purge_id="wrong",
        ) is False
        assert await store.finalize_account_deletion(
            user_id=target.user_id, purge_id=purge_id,
        ) is True
        assert await store.get_admin_user(user_id=target.user_id) is None
        assert await store.finalize_account_deletion(
            user_id=target.user_id, purge_id=purge_id,
        ) is False
    finally:
        store.close()


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
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    try:
        await _resolve(
            store,
            subject="cf-shared-email",
            email="shared@example.com",
            provider="cloudflare_access",
            initial_status="pending",
        )
        other = await _resolve(
            store,
            subject="owui-other-email",
            email="other@example.com",
            provider="owui",
            initial_status="pending",
        )
        # Simulate duplicate email rows created by the older subject-only binder.
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE app_users SET current_email = ? WHERE user_id = ?",
                ("shared@example.com", other.user_id),
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


def test_admin_delete_route_disables_account_and_queues_purge(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(_resolve(
        store, subject="route-admin", email="admin@example.com", role="admin",
    ))
    target = asyncio.run(_resolve(
        store, subject="route-target", email="target@example.com",
        provider="cloudflare_access", initial_status="pending",
    ))
    wakes: list[bool] = []
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    try:
        with TestClient(app) as client:
            unavailable = client.delete(f"/api/admin/users/{target.user_id}")
            assert unavailable.status_code == 503
            assert asyncio.run(store.get_admin_user(user_id=target.user_id)).status == "pending"

            app.state.user_data_purges = SimpleNamespace(wake=lambda: wakes.append(True))
            deleted = client.delete(f"/api/admin/users/{target.user_id}")
            assert deleted.status_code == 202
            assert deleted.json()["status"] == "deleting"
            assert wakes == [True]
            listed = client.get("/api/admin/users")
            target_row = next(
                row for row in listed.json()["items"] if row["id"] == target.user_id
            )
            assert target_row["status"] == "disabled"
            assert target_row["deletion_pending"] is True
            assert client.patch(
                f"/api/admin/users/{target.user_id}",
                json={"status": "active", "groups": ["users"]},
            ).status_code == 409
            assert client.delete(f"/api/admin/users/{admin.user_id}").status_code == 409
    finally:
        store.close()


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
            "visibility": "public",
            "roles": ["testers"],
            "portrait_url": "",
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


def test_admin_model_order_persists_and_filters_for_each_user(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    admin = asyncio.run(_resolve(
        store, subject="admin", email="admin@example.com", role="admin"
    ))
    member = asyncio.run(_resolve(
        store, subject="member", email="member@example.com"
    ))
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.state.ollama = _OllamaInventory("alpha:latest", "beta:cloud")
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    app.dependency_overrides[require_principal] = lambda: member
    try:
        with TestClient(app) as client:
            initial = client.get("/api/admin/models").json()["items"]
            workflows = [item["id"] for item in initial if item["kind"] == "workflow"]
            directs = [item["id"] for item in initial if item["kind"] == "direct"]
            assert directs == ["direct/alpha:latest", "direct/beta:cloud"]

            invalid = client.put("/api/admin/model-order", json={
                "kind": "direct", "model_ids": ["direct/alpha:latest"]
            })
            assert invalid.status_code == 409
            assert asyncio.run(store.list_model_display_order()) == {}

            reordered_workflows = [workflows[1], workflows[0], *workflows[2:]]
            assert client.put("/api/admin/model-order", json={
                "kind": "workflow", "model_ids": reordered_workflows
            }).status_code == 200
            assert client.put("/api/admin/model-order", json={
                "kind": "direct", "model_ids": list(reversed(directs))
            }).status_code == 200
            assert [item["id"] for item in client.get("/api/admin/models").json()["items"]] == [
                *reordered_workflows, *reversed(directs)
            ]
            assert [item["id"] for item in client.get("/api/models").json()["items"]] == (
                reordered_workflows
            )

            # Publishing one direct model preserves its relative place in the user catalog.
            assert client.patch("/api/admin/model-profiles/direct/beta:cloud", json={
                "visibility": "public", "roles": ["users"], "display_name": ""
            }).status_code == 200
            assert [item["id"] for item in client.get("/api/models").json()["items"]] == [
                *reordered_workflows, "direct/beta:cloud"
            ]
    finally:
        store.close()

    reopened = ApplicationStore(path)
    try:
        assert (asyncio.run(reopened.list_model_display_order()))["direct/beta:cloud"] == (
            "direct", 0
        )
    finally:
        reopened.close()


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


def test_managed_roles_and_model_publication_enforce_private_admin_only(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(_resolve(
        store, subject="admin", email="admin@example.com", role="admin"
    ))
    member = asyncio.run(_resolve(
        store, subject="member", email="member@example.com"
    ))
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.state.ollama = _OllamaInventory(_DIRECT_MODEL)
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    model_id = f"direct/{_DIRECT_MODEL}"
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/admin/roles",
                json={"id": "researchers", "name": "Researchers", "description": "Lab"},
            )
            assert created.status_code == 201
            assert created.json()["system"] is False
            assert client.get("/api/admin/roles").json()["items"][-1]["id"] == "researchers"
            assigned = client.patch(
                f"/api/admin/users/{member.user_id}",
                json={"groups": ["users", "researchers"]},
            )
            assert assigned.status_code == 200
            assert assigned.json()["groups"] == ["researchers", "users"]
            member = asyncio.run(_resolve(
                store, subject="member", email="member@example.com"
            ))
            app.dependency_overrides[require_principal] = lambda: member

            published = client.patch(
                f"/api/admin/model-profiles/{model_id}",
                json={
                    "visibility": "public",
                    "roles": ["researchers"],
                    "display_name": "Research Qwen",
                },
            )
            assert published.status_code == 200
            assert published.json()["label"] == "Research Qwen"
            assert published.json()["roles"] == ["researchers"]
            user_models = client.get("/api/models").json()["items"]
            assert model_id in {item["id"] for item in user_models}

            source = io.BytesIO()
            Image.new("RGB", (12, 12), (20, 100, 180)).save(source, "PNG")
            upload = client.put(
                f"/api/admin/model-portraits/{model_id}",
                files={"portrait": ("model.png", source.getvalue(), "image/png")},
            )
            assert upload.status_code == 200
            assert upload.json()["portrait_url"]
            portrait = client.get(f"/api/model-portraits/{model_id}")
            assert portrait.status_code == 200
            assert portrait.headers["content-type"].startswith("image/webp")

            private = client.patch(
                f"/api/admin/model-profiles/{model_id}",
                json={
                    "visibility": "private",
                    "roles": ["researchers"],
                    "display_name": "Research Qwen",
                },
            )
            assert private.status_code == 200
            assert model_id not in {item["id"] for item in client.get("/api/models").json()["items"]}
            assert client.get(f"/api/model-portraits/{model_id}").status_code == 404
            assert client.delete("/api/admin/roles/researchers").status_code == 409
            app.dependency_overrides[require_principal] = lambda: admin
            assert model_id in {item["id"] for item in client.get("/api/models").json()["items"]}
            assert client.get(f"/api/model-portraits/{model_id}").status_code == 200
            assert client.delete("/api/admin/model-policies/" + model_id).status_code == 200
            assert client.patch(
                f"/api/admin/users/{member.user_id}", json={"groups": ["users"]}
            ).status_code == 200
            assert client.delete("/api/admin/roles/researchers").status_code == 204
    finally:
        store.close()


def test_manual_pending_account_route_rejects_duplicate_email(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    admin = asyncio.run(_resolve(
        store, subject="admin", email="admin@example.com", role="admin"
    ))
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = _cfg()
    app.include_router(router)
    app.dependency_overrides[require_admin_principal] = lambda: admin
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/admin/users",
                json={"email": "invited@example.com", "display_name": "Invited"},
            )
            assert created.status_code == 201
            assert created.json()["status"] == "pending"
            assert created.json()["groups"] == []
            duplicate = client.post(
                "/api/admin/users", json={"email": "INVITED@example.com"}
            )
            assert duplicate.status_code == 409
            assert len(client.get("/api/admin/users?status=pending").json()["items"]) == 1
    finally:
        store.close()


def test_owui_import_command_previews_then_creates_pending_accounts(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "app.sqlite"
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/users/all"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={
            "users": [
                {"id": "owui-1", "email": "alice@example.com", "name": "Alice"},
                {"id": "owui-2", "email": "bob@example.com", "name": "Bob"},
            ],
            "total": 2,
        })

    monkeypatch.setattr(admin_cli, "get_config", lambda: SimpleNamespace(
        env=SimpleNamespace(owui_url="http://open-webui:8080"),
        raw={"application": {"sqlite_path": str(path)}},
    ))
    monkeypatch.setattr(admin_cli, "getpass", lambda _prompt: "test-token")
    monkeypatch.setattr(
        admin_cli.httpx, "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert asyncio.run(admin_cli._import_owui_users()) == 0
    assert '"would_create": 2' in capsys.readouterr().out
    store = ApplicationStore(path)
    try:
        assert awaitable_users(store) == ()
        admin = asyncio.run(_resolve(store, subject="owui-1", email="alice@example.com"))
        asyncio.run(store.bootstrap_admin(user_id=admin.user_id))
    finally:
        store.close()
    assert asyncio.run(admin_cli._import_owui_users(apply=True)) == 0
    output = capsys.readouterr().out
    assert '"created_pending": 1' in output
    assert "test-token" not in output
    store = ApplicationStore(path)
    try:
        pending = asyncio.run(store.list_admin_users(status="pending"))
        assert [user.email for user in pending] == ["bob@example.com"]
    finally:
        store.close()


def awaitable_users(store: ApplicationStore):
    return asyncio.run(store.list_admin_users(status="pending"))


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
