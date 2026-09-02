"""Native application resource contracts."""

import asyncio
import datetime as dt

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal, require_provider_principal
from audrey.identity import Principal
from audrey.routes.app import router


def _principal() -> Principal:
    return Principal(
        user_id="usr_123",
        storage_namespace="alice@example.com",
        provider="owui",
        provider_subject="provider-secret-subject",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )


def _persist_principal(
    store: ApplicationStore,
    template: Principal,
) -> Principal:
    return asyncio.run(
        store.resolve_external_identity(
            provider="owui",
            subject=template.provider_subject,
            email=template.email,
            display_name=template.display_name,
            role=template.role,
            auth_method=template.auth_method,
            legacy_storage_namespace=template.storage_namespace,
        )
    )


def test_get_me_returns_audrey_id_without_internal_identity_fields():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_principal] = _principal

    response = TestClient(app).get("/api/me")

    assert response.status_code == 200
    assert response.json() == {
        "id": "usr_123",
        "email": "alice@example.com",
        "display_name": "Alice",
        "role": "user",
        "status": "active",
        "auth_provider": "owui",
    }
    assert "storage_namespace" not in response.text
    assert "provider-secret-subject" not in response.text


def test_get_me_requires_authentication():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).get("/api/me")

    assert response.status_code == 401


def test_token_lifecycle_returns_secret_only_on_create(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/tokens",
                json={
                    "name": "Laptop eval",
                    "scopes": ["compat:full", "account:read"],
                    "expires_in_days": 30,
                },
            )
            assert created.status_code == 201
            body = created.json()
            token_id = body["id"]
            assert body["token"].startswith(f"aud_{token_id}.")
            assert body["scopes"] == ["account:read", "compat:full"]
            assert body["revoked_at"] is None

            listed = client.get("/api/tokens")
            assert listed.status_code == 200
            assert listed.json()["items"] == [
                {key: value for key, value in body.items() if key != "token"}
            ]
            assert "token" not in listed.text

            revoked = client.delete(f"/api/tokens/{token_id}")
            assert revoked.status_code == 200
            assert revoked.json() == {"id": body["id"], "revoked": True}

            after = client.get("/api/tokens").json()["items"][0]
            assert after["revoked_at"]
            assert "token" not in after
    finally:
        store.close()


def test_token_route_rejects_non_expiring_request(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        response = TestClient(app).post(
            "/api/tokens",
            json={
                "name": "Permanent token",
                "scopes": ["compat:full"],
                "expires_in_days": None,
            },
        )
    finally:
        store.close()

    assert response.status_code == 422


def test_token_routes_do_not_cross_owner_boundary(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)

    first = _principal()
    second = Principal(
        user_id="usr_456",
        storage_namespace="bob@example.com",
        provider="owui",
        provider_subject="owui-bob",
        email="bob@example.com",
        display_name="Bob",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )

    actual_first = _persist_principal(store, first)
    actual_second = _persist_principal(store, second)
    app.dependency_overrides[require_provider_principal] = lambda: actual_first
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/tokens",
                json={"name": "First", "scopes": ["compat:full"]},
            )
            token_id = created.json()["id"]

            app.dependency_overrides[require_provider_principal] = lambda: actual_second
            assert client.get("/api/tokens").json() == {"items": []}
            hidden = client.delete(f"/api/tokens/{token_id}")
            assert hidden.status_code == 404
    finally:
        store.close()


def test_personal_token_needs_account_read_scope_for_me():
    principal = Principal(
        user_id="usr_123",
        storage_namespace="alice@example.com",
        provider="audrey",
        provider_subject="pat_123",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        status="active",
        auth_method="personal_token",
        scopes=frozenset({"compat:full"}),
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: principal

    response = TestClient(app).get("/api/me")

    assert response.status_code == 403
    assert response.json()["detail"] == ("Personal access token lacks account:read scope.")


def test_personal_token_authenticates_through_native_me_route(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = _persist_principal(store, _principal())
    issued = asyncio.run(
        store.create_personal_token(
            user_id=owner.user_id,
            name="Native account",
            scopes=["account:read"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)

    try:
        response = TestClient(app).get(
            "/api/me",
            headers={"Authorization": f"Bearer {issued.token}"},
        )
    finally:
        store.close()

    assert response.status_code == 200
    assert response.json()["id"] == owner.user_id
    assert response.json()["auth_provider"] == "audrey"
