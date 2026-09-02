"""Native application resource contracts."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.auth import require_principal
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
