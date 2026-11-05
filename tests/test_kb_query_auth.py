"""Service-only KB queries preserve explicit trusted owner selection.

Hermetic: real token storage, fake embedders/search and no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import audrey.routes.kb as kb_module
from audrey.app_state import ApplicationStore
from audrey.auth import require_service, verify_service_token
from audrey.routes.kb import router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)




# ── verify_service_token ───────────────────────────────────────────────

def test_verify_service_token_matches():
    assert verify_service_token(SECRET, SECRET) is True


def test_verify_service_token_mismatch():
    assert verify_service_token("wrong", SECRET) is False


def test_verify_service_token_empty_expected_never_matches():
    # Fail closed: a blank configured secret can never authenticate a caller.
    assert verify_service_token("", "") is False
    assert verify_service_token("anything", "") is False


def test_verify_service_token_missing_presented():
    assert verify_service_token(None, SECRET) is False


def _fake_request(service_token: str = SECRET):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            cfg=SimpleNamespace(env=SimpleNamespace(kb_service_token=service_token)),
        )),
    )


async def test_valid_service_token_is_accepted():
    assert await require_service(_fake_request(), x_audrey_service_token=SECRET) is None


@pytest.mark.parametrize("expected,presented", [(SECRET, None), (SECRET, "wrong"), ("", "")])
async def test_service_token_gate_fails_closed(expected, presented):
    with pytest.raises(HTTPException) as exc:
        await require_service(_fake_request(expected), x_audrey_service_token=presented)
    assert exc.value.status_code == 401


# ── route-level effective-user policy ──────────────────────────────────

class _FakeTextEmbedder:
    async def embed_one(self, q):
        return [0.0]


class _FakeImageEmbedder:
    async def embed_text(self, q):
        return [0.0]

    async def embed_url(self, u):
        return [0.0]

    async def embed_b64(self, b):
        return [0.0]


def _build_app(monkeypatch, service_token: str = SECRET):
    """Mount the kb router with fakes; capture the user passed into the merge."""
    captured: dict[str, str | None] = {}

    async def _fake_text_merged(qdrant, vec, *, top_k, user, min_score=0.0, scope=None):
        captured["user"] = user
        return ([], False)

    async def _fake_image_merged(qdrant, vec, *, top_k, user, scope=None):
        captured["user"] = user
        return ([], False)

    monkeypatch.setattr(kb_module, "_search_text_merged", _fake_text_merged)
    monkeypatch.setattr(kb_module, "_search_images_merged", _fake_image_merged)

    app = FastAPI()
    app.include_router(router)
    app.state.qdrant = object()
    app.state.text_embedder = _FakeTextEmbedder()
    app.state.image_embedder = _FakeImageEmbedder()
    app.state.uploads_db = SimpleNamespace(
        private_file_ids=AsyncMock(return_value=set()),
        file_deletion_ids=AsyncMock(return_value=set()),
    )
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=service_token),
        raw={},
    )
    return app, captured


def test_route_service_token_acts_as_body_user(monkeypatch):
    app, captured = _build_app(monkeypatch)
    r = TestClient(app).post(
        "/v1/kb/query", json={"query": "hi", "user": "alice@example.com"},
        headers={"X-Audrey-Service-Token": SECRET},
    )
    assert r.status_code == 200
    assert captured["user"] == "alice@example.com"


def test_route_no_creds_401_never_searches(monkeypatch):
    app, captured = _build_app(monkeypatch)
    r = TestClient(app).post("/v1/kb/query", json={"query": "hi", "user": "alice@example.com"})
    assert r.status_code == 401
    assert "user" not in captured  # dependency 401'd before the search ran


def test_route_wrong_service_token_401(monkeypatch):
    app, _ = _build_app(monkeypatch)
    r = TestClient(app).post(
        "/v1/kb/query", json={"query": "hi"},
        headers={"X-Audrey-Service-Token": "nope"},
    )
    assert r.status_code == 401


@pytest.mark.parametrize("path", ["/v1/kb/query", "/v1/kb/query/image"])
@pytest.mark.parametrize("bad_service", [None, "wrong"])
def test_valid_personal_token_cannot_query_kb(monkeypatch, tmp_path, path, bad_service):
    app, captured = _build_app(monkeypatch)
    store = ApplicationStore(tmp_path / "app.sqlite")
    import asyncio

    async def issue():
        principal = await store.resolve_external_identity(
            provider="test", subject="kb-owner", email="alice@example.com",
            display_name="Alice", role="admin", auth_method="test",
        )
        return await store.create_personal_token(
            user_id=principal.user_id, name="KB refusal", scopes=["compat:full"],
            expires_at="2099-01-01T00:00:00+00:00",
        )

    issued = asyncio.run(issue())
    app.state.application_store = store
    headers = {"Authorization": f"Bearer {issued.token}"}
    if bad_service is not None:
        headers["X-Audrey-Service-Token"] = bad_service
    try:
        response = TestClient(app).post(
            path, json={"query": "invoice", "user": "other-owner"}, headers=headers,
        )
    finally:
        store.close()
    assert response.status_code == 401
    assert response.json()["detail"] == "Service token required."
    assert captured == {}


@pytest.mark.parametrize("path", ["/v1/kb/query", "/v1/kb/query/image"])
def test_access_assertion_cannot_query_kb(monkeypatch, path):
    app, captured = _build_app(monkeypatch)
    response = TestClient(app).post(
        path, json={"query": "invoice"},
        headers={"Cf-Access-Jwt-Assertion": "access-assertion"},
    )
    assert response.status_code == 401
    assert captured == {}


def test_route_image_service_token_acts_as_body_user(monkeypatch):
    app, captured = _build_app(monkeypatch)
    r = TestClient(app).post(
        "/v1/kb/query/image", json={"query": "a cat", "user": "alice@example.com"},
        headers={"X-Audrey-Service-Token": SECRET},
    )
    assert r.status_code == 200
    assert captured["user"] == "alice@example.com"


def test_kb_stats_route_is_removed(monkeypatch):
    app, _ = _build_app(monkeypatch)
    assert TestClient(app).get("/v1/kb/stats").status_code == 404
