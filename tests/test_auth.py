"""Cloudflare identity, local personal-token revocation and fail-closed access."""

import datetime as dt
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import (
    AuthedUser,
    require_account_user,
    require_admin,
    require_provider_principal,
    require_user,
)
from audrey.identity import (
    CloudflareAccessClaims,
    CloudflareAccessTokenError,
    CloudflareAccessUnavailableError,
)


def _fake_request(*, application_store=None, cloudflare_access_verifier=None,
                  path="/v1/chat/completions"):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            application_store=application_store,
            cloudflare_access_verifier=cloudflare_access_verifier,
        )),
        url=SimpleNamespace(path=path),
    )


@pytest.mark.parametrize("authorization", ["Bearer unknown-session", "Bearer signed-jwt"])
async def test_unknown_bearers_are_rejected_locally(monkeypatch, authorization):
    def unexpected_network(*args, **kwargs):
        raise AssertionError("unsupported credentials must never leave Audrey")
    monkeypatch.setattr(httpx, "AsyncClient", unexpected_network)
    with pytest.raises(HTTPException) as exc:
        await require_user(_fake_request(), authorization=authorization)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Unsupported bearer token."


def test_authed_user_has_no_id_field():
    # The Phase 26 deploy hit `AttributeError: 'AuthedUser' object has
    # no attribute 'id'` in production. This test exists so any future
    # refactor that re-introduces an `.id` field is a deliberate, visible
    # change rather than a silent one.
    me = AuthedUser(email="x@y.z", role="user")
    with pytest.raises(AttributeError):
        _ = me.id


async def test_require_user_missing_header_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await require_user(_fake_request(), authorization=None)
    assert exc.value.status_code == 401


async def test_require_user_non_bearer_scheme_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await require_user(_fake_request(), authorization="Basic dXNlcjpwYXNz")
    assert exc.value.status_code == 401


async def test_require_user_empty_bearer_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        await require_user(_fake_request(), authorization="Bearer ")
    assert exc.value.status_code == 401


class _CloudflareVerifier:
    def __init__(self, result):
        self.result = result
        self.tokens: list[str] = []

    async def verify(self, token):
        self.tokens.append(token)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_cloudflare_header_creates_provider_neutral_principal(
    monkeypatch,
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    verifier = _CloudflareVerifier(
        CloudflareAccessClaims(subject="cf-subject", email="alice@example.com")
    )

    def _unexpected_network(*args, **kwargs):
        raise AssertionError("Cloudflare verification must not use an unconfigured HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", _unexpected_network)
    try:
        me = await require_account_user(
            _fake_request(
                application_store=store,
                cloudflare_access_verifier=verifier,
            ),
            authorization=None,
            cloudflare_access_jwt="signed-access-token",
        )
    finally:
        store.close()

    assert verifier.tokens == ["signed-access-token"]
    assert me.principal is not None
    assert me.principal.provider == "cloudflare_access"
    assert me.principal.provider_subject == "cf-subject"
    assert me.principal.auth_method == "cloudflare_access"
    assert me.principal.role == "user"
    assert me.principal.status == "pending"
    assert me.principal.groups == frozenset()
    assert me.email.startswith("ns_")
    assert me.email != "alice@example.com"


def test_fastapi_reads_cloudflare_access_assertion_header(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    verifier = _CloudflareVerifier(
        CloudflareAccessClaims(subject="cf-http-subject", email="alice@example.com")
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cloudflare_access_verifier = verifier

    @app.get("/probe")
    async def probe(me: AuthedUser = Depends(require_account_user)):
        return {"user_id": me.principal.user_id}

    try:
        with TestClient(app) as client:
            response = client.get(
                "/probe",
                headers={"Cf-Access-Jwt-Assertion": "http-access-token"},
            )
    finally:
        store.close()

    assert response.status_code == 200
    assert response.json()["user_id"].startswith("usr_")
    assert verifier.tokens == ["http-access-token"]


async def test_cloudflare_email_match_preserves_imported_account_and_namespace(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    imported = await store.resolve_external_identity(
        provider="owui",
        subject="historical-subject",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="test",
        legacy_storage_namespace="alice@example.com",
    )
    verifier = _CloudflareVerifier(
        CloudflareAccessClaims(subject="cf-subject", email="alice@example.com")
    )
    try:
        me = await require_account_user(
            _fake_request(
                application_store=store,
                cloudflare_access_verifier=verifier,
            ),
            authorization=None,
            cloudflare_access_jwt="signed-access-token",
        )
    finally:
        store.close()

    assert me.principal is not None
    assert me.principal.user_id == imported.user_id
    assert me.principal.storage_namespace == imported.storage_namespace
    assert me.principal.status == "active"


async def test_invalid_cloudflare_header_never_falls_back_to_bearer(
    monkeypatch,
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    verifier = _CloudflareVerifier(CloudflareAccessTokenError("bad signature"))

    def _unexpected_network(*args, **kwargs):
        raise AssertionError("invalid Access evidence must fail closed")

    monkeypatch.setattr(httpx, "AsyncClient", _unexpected_network)
    try:
        with pytest.raises(HTTPException) as exc:
            await require_user(
                _fake_request(
                    application_store=store,
                    cloudflare_access_verifier=verifier,
                ),
                authorization="Bearer otherwise-valid-token",
                cloudflare_access_jwt="invalid-access-token",
            )
    finally:
        store.close()

    assert exc.value.status_code == 401
    assert exc.value.detail == "Cloudflare Access token is invalid."


async def test_cloudflare_provider_outage_returns_distinct_gateway_error(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    verifier = _CloudflareVerifier(CloudflareAccessUnavailableError("offline"))
    try:
        with pytest.raises(HTTPException) as exc:
            await require_user(
                _fake_request(
                    application_store=store,
                    cloudflare_access_verifier=verifier,
                ),
                authorization=None,
                cloudflare_access_jwt="signed-access-token",
            )
    finally:
        store.close()

    assert exc.value.status_code == 502
    assert exc.value.detail == "Auth provider verification unavailable."


async def _issue_token(
    store: ApplicationStore,
    *,
    scopes: list[str],
) -> tuple[AuthedUser, str]:
    principal = await store.resolve_external_identity(
        provider="test",
        subject="token-owner",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="test",
        legacy_storage_namespace="alice@example.com",
    )
    issued = await store.create_personal_token(
        user_id=principal.user_id,
        name="Auth test",
        scopes=scopes,
        expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
    )
    return (
        AuthedUser(
            email=principal.email,
            role=principal.role,
            principal=principal,
        ),
        issued.token,
    )


async def test_personal_token_auth_is_local_and_uses_storage_namespace(
    monkeypatch,
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    _, token = await _issue_token(store, scopes=["compat:full"])

    def _unexpected_network(*args, **kwargs):
        raise AssertionError("Personal token verification must stay local")

    monkeypatch.setattr(httpx, "AsyncClient", _unexpected_network)
    try:
        me = await require_user(
            _fake_request(
                application_store=store,
            ),
            authorization=f"Bearer {token}",
        )
    finally:
        store.close()

    assert me.email == "alice@example.com"
    assert me.principal is not None
    assert me.principal.auth_method == "personal_token"


async def test_personal_token_without_compat_scope_cannot_call_v1(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    _, token = await _issue_token(store, scopes=["account:read"])
    try:
        with pytest.raises(HTTPException) as exc:
            await require_user(
                _fake_request(application_store=store),
                authorization=f"Bearer {token}",
            )
    finally:
        store.close()

    assert exc.value.status_code == 403
    assert exc.value.detail == ("Personal access token lacks compat:full scope.")


async def test_personal_token_revocation_takes_effect_on_next_request(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, token = await _issue_token(store, scopes=["compat:full"])
    token_id = (
        await store.list_personal_tokens(
            user_id=owner.principal.user_id,
        )
    )[0].token_id
    request = _fake_request(application_store=store)
    try:
        await require_user(request, authorization=f"Bearer {token}")
        assert await store.revoke_personal_token(
            user_id=owner.principal.user_id,
            token_id=token_id,
        )
        with pytest.raises(HTTPException) as exc:
            await require_user(request, authorization=f"Bearer {token}")
    finally:
        store.close()

    assert exc.value.status_code == 401


async def test_personal_token_cannot_manage_credentials(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    _, token = await _issue_token(store, scopes=["account:read", "compat:full"])
    try:
        me = await require_user(
            _fake_request(application_store=store, path="/api/tokens"),
            authorization=f"Bearer {token}",
        )
        with pytest.raises(HTTPException) as exc:
            await require_provider_principal(principal=me.principal)
    finally:
        store.close()

    assert exc.value.status_code == 403
    assert exc.value.detail == (
        "External provider authentication is required for this operation."
    )


async def test_personal_token_never_inherits_compatibility_admin_role(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="test",
        subject="admin-owner",
        email="admin@example.com",
        display_name="Admin",
        role="admin",
        auth_method="test",
        legacy_storage_namespace="admin@example.com",
    )
    issued = await store.create_personal_token(
        user_id=owner.user_id,
        name="Admin account automation",
        scopes=["compat:full"],
        expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
    )
    try:
        me = await require_user(
            _fake_request(application_store=store),
            authorization=f"Bearer {issued.token}",
        )
        assert me.principal is not None
        assert me.principal.role == "admin"
        assert me.role == "user"
        with pytest.raises(HTTPException) as exc:
            await require_admin(me=me)
    finally:
        store.close()

    assert exc.value.status_code == 403


async def test_require_admin_passes_admin_role():
    me = AuthedUser(email="a@b.c", role="admin")
    out = await require_admin(me=me)
    assert out is me


async def test_require_admin_rejects_user_role_with_403():
    me = AuthedUser(email="a@b.c", role="user")
    with pytest.raises(HTTPException) as exc:
        await require_admin(me=me)
    assert exc.value.status_code == 403
