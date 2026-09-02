"""Tests for require_user + _probe_owui (Phase 14, 22, 26).

Stubs `httpx.AsyncClient` so tests run offline. Avoids `respx` to keep
the dev-dep surface tight — the auth module only ever calls
`AsyncClient().get(...)` once, so a hand-rolled fake is fewer LOC than
the mock library would be.

Critical regression guards:
  - **Phase 26 `me.id` slip:** `AuthedUser.id` does not exist; only
    `email`, `role`, `owui_id`. Tested below — accessing `.id` must
    raise `AttributeError`.
  - **OWUI `pending` role:** OWUI v0.9.2 returns 200 for pending users.
    The `_ALLOWED_ROLES` allowlist must reject them at audrey's gate
    even though OWUI itself returned 2xx.
"""

import datetime as dt
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from audrey import auth as auth_module
from audrey.app_state import ApplicationStore
from audrey.auth import (
    AuthedUser,
    _probe_owui,
    clear_auth_cache,
    clear_auth_cache_for_email,
    require_admin,
    require_provider_principal,
    require_user,
)

# ─── Test helpers ──────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal stand-in for `httpx.Response` — only what `_probe_owui` reads."""

    def __init__(self, status_code: int, body: dict | None = None, raw: str | None = None):
        self.status_code = status_code
        self._body = body
        self._raw = raw if raw is not None else ""
        self.text = raw if raw is not None else (str(body) if body else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _FakeAsyncClient:
    """Stand-in for `httpx.AsyncClient`. Returns a pre-canned response."""

    def __init__(self, response: _FakeResponse | Exception):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get(self, url: str, headers=None):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _patch_async_client(monkeypatch, response):
    """Make `httpx.AsyncClient(...)` inside `auth.py` return our fake."""

    def _factory(*args, **kwargs):
        return _FakeAsyncClient(response)

    monkeypatch.setattr(auth_module.httpx, "AsyncClient", _factory)


@pytest.fixture(autouse=True)
def _isolate_cache():
    # Auth uses a module-level dict cache; clear it before AND after each
    # test so cross-test contamination can't sneak in.
    clear_auth_cache()
    yield
    clear_auth_cache()


def _fake_request(
    owui_url: str = "http://open-webui:8080",
    *,
    application_store=None,
    path: str = "/v1/chat/completions",
):
    # FastAPI passes a `Request` whose `.app.state.cfg.env.owui_url`
    # is the OWUI base URL. Build the smallest object graph that exposes
    # that attribute path.
    state = SimpleNamespace(cfg=SimpleNamespace(env=SimpleNamespace(owui_url=owui_url)))
    if application_store is not None:
        state.application_store = application_store
    return SimpleNamespace(
        app=SimpleNamespace(state=state),
        url=SimpleNamespace(path=path),
    )


# ─── AuthedUser shape (Phase 26 regression) ────────────────────────────


def test_authed_user_exposes_email_role_owui_id():
    me = AuthedUser(email="bart@proton.me", role="admin", owui_id="uuid-x")
    assert me.email == "bart@proton.me"
    assert me.role == "admin"
    assert me.owui_id == "uuid-x"


def test_authed_user_has_no_id_field():
    # The Phase 26 deploy hit `AttributeError: 'AuthedUser' object has
    # no attribute 'id'` in production. This test exists so any future
    # refactor that re-introduces an `.id` field is a deliberate, visible
    # change rather than a silent one.
    me = AuthedUser(email="x@y.z", role="user", owui_id="uuid")
    with pytest.raises(AttributeError):
        _ = me.id  # type: ignore[attr-defined]


# ─── _probe_owui happy path ────────────────────────────────────────────


async def test_probe_owui_returns_authed_user_on_200(monkeypatch):
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid-bart",
                "email": "bart@proton.me",
                "role": "admin",
            },
        ),
    )
    me = await _probe_owui("http://open-webui:8080", "tok123")
    assert me == AuthedUser(email="bart@proton.me", role="admin", owui_id="uuid-bart")


async def test_probe_owui_lowercases_role(monkeypatch):
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid",
                "email": "x@y.z",
                "role": "ADMIN",
            },
        ),
    )
    me = await _probe_owui("http://owui", "t")
    assert me.role == "admin"


# ─── _probe_owui rejection paths ───────────────────────────────────────


async def test_probe_owui_401_raises_401(monkeypatch):
    _patch_async_client(monkeypatch, _FakeResponse(401, raw="invalid token"))
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "bad")
    assert exc.value.status_code == 401


async def test_probe_owui_5xx_raises_502(monkeypatch):
    # Distinguish "you're not logged in" (401) from "auth backend broken"
    # (502) — clients react differently.
    _patch_async_client(monkeypatch, _FakeResponse(500, raw="oops"))
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "t")
    assert exc.value.status_code == 502


async def test_probe_owui_network_error_raises_502(monkeypatch):
    _patch_async_client(monkeypatch, httpx.ConnectError("connection refused"))
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "t")
    assert exc.value.status_code == 502


async def test_probe_owui_pending_role_rejected_with_401(monkeypatch):
    # OWUI returns 200 with role=pending for unactivated users — JWT is
    # technically valid but they shouldn't have chat access. Audrey must
    # fail closed at the role allowlist.
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid",
                "email": "newuser@example.com",
                "role": "pending",
            },
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "t")
    assert exc.value.status_code == 401


async def test_probe_owui_unknown_role_rejected(monkeypatch):
    # Future-proofing: a future OWUI version might add `disabled` or
    # similar. Anything not in `_ALLOWED_ROLES` must fail closed.
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid",
                "email": "x@y.z",
                "role": "disabled",
            },
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "t")
    assert exc.value.status_code == 401


async def test_probe_owui_missing_email_raises_502(monkeypatch):
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid",
                "role": "user",
            },
        ),
    )
    with pytest.raises(HTTPException) as exc:
        await _probe_owui("http://owui", "t")
    assert exc.value.status_code == 502


# ─── require_user header parsing ───────────────────────────────────────


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


# ─── require_user happy path + cache ───────────────────────────────────


async def test_require_user_happy_path_returns_authed_user(monkeypatch):
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "uuid-x",
                "email": "ok@user.com",
                "role": "user",
            },
        ),
    )
    me = await require_user(_fake_request(), authorization="Bearer goodtoken")
    assert me.email == "ok@user.com"
    assert me.role == "user"


async def test_require_user_caches_subsequent_calls(monkeypatch):
    # First call hits OWUI; second call with the same token must hit the
    # cache (verified by counting calls to the fake client factory).
    call_count = 0
    response = _FakeResponse(
        200,
        body={
            "id": "uuid",
            "email": "x@y.z",
            "role": "user",
        },
    )

    def _factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeAsyncClient(response)

    monkeypatch.setattr(auth_module.httpx, "AsyncClient", _factory)

    req = _fake_request()
    await require_user(req, authorization="Bearer same-token")
    await require_user(req, authorization="Bearer same-token")
    assert call_count == 1, "second call should be served from cache"


async def test_require_user_binds_stable_audrey_principal(monkeypatch, tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "owui-stable-subject",
                "email": "alice@example.com",
                "name": "Alice",
                "role": "user",
            },
        ),
    )
    try:
        me = await require_user(
            _fake_request(application_store=store),
            authorization="Bearer identity-token",
        )
    finally:
        store.close()

    assert me.principal is not None
    assert me.principal.user_id.startswith("usr_")
    assert me.principal.storage_namespace == "alice@example.com"
    assert me.principal.provider_subject == "owui-stable-subject"


async def test_require_user_rejects_missing_stable_subject_when_store_is_active(
    monkeypatch,
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "email": "alice@example.com",
                "role": "user",
            },
        ),
    )
    try:
        with pytest.raises(HTTPException) as exc:
            await require_user(
                _fake_request(application_store=store),
                authorization="Bearer missing-subject-token",
            )
    finally:
        store.close()

    assert exc.value.status_code == 502
    assert exc.value.detail == "Auth provider response missing stable subject."


async def _issue_token(
    store: ApplicationStore,
    *,
    scopes: list[str],
) -> tuple[AuthedUser, str]:
    principal = await store.resolve_external_identity(
        provider="owui",
        subject="owui-token-owner",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="owui_bearer",
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
            owui_id=principal.provider_subject,
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

    def _unexpected_owui(*args, **kwargs):
        raise AssertionError("Audrey token must never be forwarded to OWUI")

    monkeypatch.setattr(auth_module.httpx, "AsyncClient", _unexpected_owui)
    try:
        me = await require_user(
            _fake_request(application_store=store),
            authorization=f"Bearer {token}",
        )
    finally:
        store.close()

    assert me.email == "alice@example.com"
    assert me.owui_id == ""
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


async def test_personal_token_revocation_is_not_hidden_by_auth_cache(tmp_path):
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
        assert auth_module.cache_size() == 0
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
        "External provider authentication is required to manage access tokens."
    )


async def test_personal_token_never_inherits_compatibility_admin_role(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="owui",
        subject="owui-admin",
        email="admin@example.com",
        display_name="Admin",
        role="admin",
        auth_method="owui_bearer",
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


# ─── require_admin ─────────────────────────────────────────────────────


async def test_require_admin_passes_admin_role():
    me = AuthedUser(email="a@b.c", role="admin", owui_id="u")
    out = await require_admin(me=me)
    assert out is me


async def test_require_admin_rejects_user_role_with_403():
    me = AuthedUser(email="a@b.c", role="user", owui_id="u")
    with pytest.raises(HTTPException) as exc:
        await require_admin(me=me)
    assert exc.value.status_code == 403


# ─── clear_auth_cache_for_email (Phase 22) ─────────────────────────────


async def test_targeted_eviction_removes_only_matching_email(monkeypatch):
    # Seed two users into the cache via real require_user calls.
    user1 = _FakeResponse(200, body={"id": "u1", "email": "alice@x.y", "role": "user"})
    user2 = _FakeResponse(200, body={"id": "u2", "email": "bob@x.y", "role": "user"})

    seq = iter([user1, user2])

    def _factory(*args, **kwargs):
        return _FakeAsyncClient(next(seq))

    monkeypatch.setattr(auth_module.httpx, "AsyncClient", _factory)
    req = _fake_request()
    await require_user(req, authorization="Bearer token-alice")
    await require_user(req, authorization="Bearer token-bob")
    assert auth_module.cache_size() == 2

    n = clear_auth_cache_for_email("alice@x.y")
    assert n == 1
    assert auth_module.cache_size() == 1


async def test_targeted_eviction_is_case_insensitive(monkeypatch):
    _patch_async_client(
        monkeypatch,
        _FakeResponse(
            200,
            body={
                "id": "u",
                "email": "Bart@Proton.ME",
                "role": "admin",
            },
        ),
    )
    await require_user(_fake_request(), authorization="Bearer t")
    n = clear_auth_cache_for_email("bart@proton.me")
    assert n == 1


def test_targeted_eviction_empty_email_returns_zero():
    assert clear_auth_cache_for_email("") == 0
    assert clear_auth_cache_for_email("   ") == 0
