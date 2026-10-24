"""Phase 31 — authentication on the KB query routes (`/v1/kb/query[/image]`).

The routes used to trust a caller-supplied `user` and merge that user's private
collection with no auth, so publishing 8000 to the LAN would let any device read
another user's uploads. These tests pin the fix:

  - `verify_service_token` fails closed on a blank/absent secret.
  - `resolve_kb_caller` accepts a valid service token (act-as) OR a user bearer,
    and 401s otherwise.
  - The routes honor the body `user` only for a service caller; an authenticated
    end user is pinned to their own email, so `user` can't widen the search.

Hermetic: qdrant/embedder are fakes, the merge is patched to capture the effective
user, and authentication is stubbed where the bearer arm is exercised. No network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import audrey.routes.kb as kb_module
from audrey import auth as auth_module
from audrey.auth import (
    KBCaller,
    resolve_kb_caller,
    verify_service_token,
)
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


# ── resolve_kb_caller (unit) ───────────────────────────────────────────







def _fake_request(
    service_token: str = SECRET,
):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            cfg=SimpleNamespace(env=SimpleNamespace(
                kb_service_token=service_token,
            )),
        )),
    )


async def test_resolve_service_token_yields_service_caller():
    caller = await resolve_kb_caller(
        _fake_request(), x_audrey_service_token=SECRET, authorization=None,
    )
    assert caller == KBCaller(email=None, is_service=True)


async def test_resolve_no_creds_raises_401():
    with pytest.raises(HTTPException) as exc:
        await resolve_kb_caller(_fake_request(), x_audrey_service_token=None, authorization=None)
    assert exc.value.status_code == 401


async def test_resolve_bad_service_token_falls_through_to_bearer_401():
    # A wrong service token is not a free pass — it falls to the bearer arm,
    # which 401s when no bearer is present.
    with pytest.raises(HTTPException) as exc:
        await resolve_kb_caller(_fake_request(), x_audrey_service_token="nope", authorization=None)  # noqa: S106
    assert exc.value.status_code == 401


async def test_resolve_user_bearer_yields_user_caller(monkeypatch):
    monkeypatch.setattr(auth_module, "require_user", AsyncMock(return_value=SimpleNamespace(email="alice@example.com")))
    caller = await resolve_kb_caller(
        _fake_request(),
        x_audrey_service_token=None,
        authorization="Bearer tok",
    )
    assert caller == KBCaller(email="alice@example.com", is_service=False)


async def test_resolve_blank_secret_disables_service_arm(monkeypatch):
    # With no configured secret, even a matching-looking header can't act-as;
    # the caller must present a bearer (here none → 401).
    with pytest.raises(HTTPException) as exc:
        await resolve_kb_caller(
            _fake_request(service_token=""), x_audrey_service_token="", authorization=None,
        )
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


def test_route_user_bearer_is_pinned_to_own_email(monkeypatch):
    # THE security assertion: an authenticated user cannot read someone else's
    # KB by naming them in `user` — the search is forced to the caller's email.
    app, captured = _build_app(monkeypatch)
    app.dependency_overrides[resolve_kb_caller] = lambda: KBCaller(
        email="bob@example.com", is_service=False,
    )
    r = TestClient(app).post(
        "/v1/kb/query", json={"query": "hi", "user": "alice@example.com"},
        headers={"Authorization": "Bearer tok"},
    )
    assert r.status_code == 200
    assert captured["user"] == "bob@example.com"  # body `user` ignored


def test_route_image_user_bearer_is_pinned_to_own_email(monkeypatch):
    app, captured = _build_app(monkeypatch)
    app.dependency_overrides[resolve_kb_caller] = lambda: KBCaller(
        email="bob@example.com", is_service=False,
    )
    r = TestClient(app).post(
        "/v1/kb/query/image", json={"query": "a cat", "user": "alice@example.com"},
        headers={"Authorization": "Bearer tok"},
    )
    assert r.status_code == 200
    assert captured["user"] == "bob@example.com"


def test_route_image_service_token_acts_as_body_user(monkeypatch):
    app, captured = _build_app(monkeypatch)
    r = TestClient(app).post(
        "/v1/kb/query/image", json={"query": "a cat", "user": "alice@example.com"},
        headers={"X-Audrey-Service-Token": SECRET},
    )
    assert r.status_code == 200
    assert captured["user"] == "alice@example.com"
