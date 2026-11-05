"""Route moves preserve provider authority, personal scopes and service gates."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from audrey import main
from audrey.app_state import ApplicationStore
from audrey.identity import CloudflareAccessClaims
from audrey.routes.admin import router as admin_router
from audrey.routes.kb import admin_router as kb_admin_router
from audrey.routes.references import router as references_router
from audrey.routes.user_data import router as user_router
from audrey.tools.discovery import ToolRegistry


@pytest.fixture
async def identity_app(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    identities = {}
    for role in ("user", "admin"):
        identities[role] = await store.resolve_external_identity(
            provider="cloudflare_access", subject=role,
            email=f"{role}@example.com", display_name=role, role=role,
            auth_method="cloudflare_access", legacy_storage_namespace=f"ns_{role}",
        )

    class Verifier:
        async def verify(self, token):
            return CloudflareAccessClaims(subject=token, email=f"{token}@example.com")

    app = FastAPI()
    app.state.application_store = store
    app.state.cloudflare_access_verifier = Verifier()
    app.include_router(admin_router)
    app.include_router(kb_admin_router)
    app.include_router(user_router)
    app.include_router(references_router)
    app.add_api_route("/api/admin/tools/rediscover", main.rediscover_tools, methods=["POST"])
    monkeypatch.setattr(main.app.state, "cfg", SimpleNamespace(tools={}), raising=False)
    monkeypatch.setattr(main.app.state, "tools", ToolRegistry(), raising=False)
    monkeypatch.setattr(main, "_refresh_tool_discovery", AsyncMock(return_value=True))
    import audrey.routes.user_data as user_data

    memory_reads = AsyncMock(return_value={"items": [], "next_cursor": None})
    monkeypatch.setattr(user_data, "_request_page", memory_reads)
    app.state.reference_libraries = SimpleNamespace(list_libraries=lambda: {"items": []})
    try:
        yield app, store, identities, memory_reads
    finally:
        store.close()


@pytest.mark.parametrize("method,path,payload,allowed", [
    ("GET", "/api/admin/repair-status", None, 200),
    ("POST", "/api/admin/kb/ingest", {"paths": []}, 503),
    ("POST", "/api/admin/tools/rediscover", None, 200),
])
async def test_moved_admin_routes_require_provider_admin(identity_app, method, path, payload, allowed):
    app, store, identities, _ = identity_app
    issued = await store.create_personal_token(
        user_id=identities["admin"].user_id, name="Admin automation",
        scopes=["compat:full"], expires_at=None,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for headers, expected in [
            ({}, 401),
            ({"Cf-Access-Jwt-Assertion": "user"}, 403),
            ({"Authorization": f"Bearer {issued.token}"}, 403),
            ({"Cf-Access-Jwt-Assertion": "admin"}, allowed),
        ]:
            response = await client.request(method, path, headers=headers, json=payload)
            assert response.status_code == expected, response.text


@pytest.mark.parametrize("scope,expected", [("account:read", 403), ("compat:full", 200)])
@pytest.mark.parametrize("path", ["/api/me/memories", "/v1/references/libraries"])
async def test_data_and_bot_metadata_preserve_compat_scope(identity_app, scope, expected, path):
    app, store, identities, memory_reads = identity_app
    issued = await store.create_personal_token(
        user_id=identities["user"].user_id, name="Scoped token", scopes=[scope], expires_at=None,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(path, headers={"Authorization": f"Bearer {issued.token}"})
    assert response.status_code == expected, response.text
    if path == "/api/me/memories":
        if expected == 200:
            assert memory_reads.await_args.kwargs["user"] == "ns_user"
        else:
            memory_reads.assert_not_awaited()


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/admin/readiness"),
    ("POST", "/v1/admin/repair"),
    ("GET", "/v1/me/memories"),
    ("POST", "/v1/kb/ingest"),
    ("GET", "/v1/kb/stats"),
    ("POST", "/v1/tools/rediscover"),
    ("GET", "/v1/files"),
    ("POST", "/v1/files"),
    ("DELETE", "/v1/files/old-file"),
    ("POST", "/v1/files/from-url"),
    ("POST", "/v1/files/upload-sessions"),
    ("PUT", "/v1/files/upload-sessions/old-upload/parts/0"),
    ("POST", "/v1/files/upload-sessions/old-upload/complete"),
])
async def test_retired_http_routes_are_absent(method, path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.request(method, path)
    assert response.status_code == 404, response.text
