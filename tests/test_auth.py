"""Cloudflare identity, local personal-token revocation and fail-closed access."""

import datetime as dt
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import (
    require_account_user,
    require_admin_principal,
    require_principal,
    require_provider_principal,
    require_user,
)
from audrey.identity import (
    CloudflareAccessClaims,
    CloudflareAccessTokenError,
    CloudflareAccessUnavailableError,
    Principal,
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
    assert me is not None
    assert me.provider == "cloudflare_access"
    assert me.provider_subject == "cf-subject"
    assert me.auth_method == "cloudflare_access"
    assert me.role == "user"
    assert me.status == "pending"
    assert me.groups == frozenset()
    assert me.storage_namespace.startswith("ns_")
    assert me.storage_namespace != "alice@example.com"


def test_fastapi_reads_cloudflare_access_assertion_header(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    verifier = _CloudflareVerifier(
        CloudflareAccessClaims(subject="cf-http-subject", email="alice@example.com")
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cloudflare_access_verifier = verifier

    @app.get("/probe")
    async def probe(me: Principal = Depends(require_account_user)):
        return {"user_id": me.user_id}

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

    assert me is not None
    assert me.user_id == imported.user_id
    assert me.storage_namespace == imported.storage_namespace
    assert me.status == "active"


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
) -> tuple[Principal, str]:
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
    return principal, issued.token


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

    assert me.storage_namespace == "alice@example.com"
    assert me is not None
    assert me.auth_method == "personal_token"


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
            user_id=owner.user_id,
        )
    )[0].token_id
    request = _fake_request(application_store=store)
    try:
        await require_user(request, authorization=f"Bearer {token}")
        assert await store.revoke_personal_token(
            user_id=owner.user_id,
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
            await require_provider_principal(principal=me)
    finally:
        store.close()

    assert exc.value.status_code == 403
    assert exc.value.detail == (
        "External provider authentication is required for this operation."
    )


async def test_admin_personal_token_keeps_account_identity_but_cannot_manage_credentials(tmp_path):
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
        assert me is not None
        assert me.role == "admin"
        with pytest.raises(HTTPException) as exc:
            await require_provider_principal(principal=me)
    finally:
        store.close()

    assert exc.value.status_code == 403


@pytest.mark.parametrize("role, expected", [("admin", None), ("user", 403)])
async def test_require_admin_principal_enforces_account_authority(tmp_path, role, expected):
    store = ApplicationStore(tmp_path / "admin.sqlite")
    try:
        principal = await store.resolve_external_identity(
            provider="cloudflare_access", subject="admin-gate", email="alice@example.com",
            display_name="Alice", role=role, auth_method="cloudflare_access",
        )
        if expected is None:
            assert await require_admin_principal(principal=principal) is principal
        else:
            with pytest.raises(HTTPException) as exc:
                await require_admin_principal(principal=principal)
            assert exc.value.status_code == expected
    finally:
        store.close()


@pytest.mark.parametrize("path", ["/api/me/memories", "/v1/chat/completions"])
async def test_compatibility_scope_is_explicit_and_independent_of_route_prefix(tmp_path, path):
    store = ApplicationStore(tmp_path / "scope.sqlite")
    _, token = await _issue_token(store, scopes=["account:read"])
    request = _fake_request(application_store=store, path=path)
    try:
        principal = await require_principal(request, authorization=f"Bearer {token}")
        assert principal.scopes == frozenset({"account:read"})
        with pytest.raises(HTTPException) as exc:
            await require_user(request, authorization=f"Bearer {token}")
        assert exc.value.status_code == 403
    finally:
        store.close()


@pytest.mark.parametrize("status", ["pending", "disabled"])
async def test_account_status_remains_visible_while_resource_auth_refuses_inactive_access(tmp_path, status):
    store = ApplicationStore(tmp_path / "status.sqlite")
    verifier = _CloudflareVerifier(
        CloudflareAccessClaims(subject="cf-status", email="alice@example.com")
    )
    request = _fake_request(application_store=store, cloudflare_access_verifier=verifier)
    try:
        principal = await require_account_user(request, cloudflare_access_jwt="signed-access")
        if status == "disabled":
            with sqlite3.connect(store.path) as connection:
                connection.execute("UPDATE app_users SET status = 'disabled'")
        principal = await require_account_user(request, cloudflare_access_jwt="signed-access")
        assert principal.status == status
        with pytest.raises(HTTPException) as exc:
            await require_principal(request, cloudflare_access_jwt="signed-access")
        assert exc.value.status_code == 403
    finally:
        store.close()


async def test_admin_personal_token_is_refused_by_provider_admin_route(tmp_path):
    store = ApplicationStore(tmp_path / "admin-token.sqlite")
    try:
        owner = await store.resolve_external_identity(
            provider="cloudflare_access", subject="admin-token", email="admin@example.com",
            display_name="Admin", role="admin", auth_method="cloudflare_access",
        )
        issued = await store.create_personal_token(
            user_id=owner.user_id, name="Admin automation", scopes=["compat:full"], expires_at=None,
        )
        app = FastAPI()
        app.state.application_store = store

        @app.get("/api/admin/probe")
        async def probe(principal: Principal = Depends(require_admin_principal)):
            return {"user_id": principal.user_id}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://audrey") as client:
            response = await client.get("/api/admin/probe", headers={"Authorization": f"Bearer {issued.token}"})
        assert response.status_code == 403
        assert response.json()["detail"] == "External provider authentication is required for this operation."
    finally:
        store.close()
