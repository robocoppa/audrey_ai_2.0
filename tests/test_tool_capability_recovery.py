"""Authenticated tool inspection and safe in-place discovery recovery."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from fastapi import HTTPException

from audrey import main as main_module
from audrey.app_state import ApplicationStore
from audrey.auth import AuthedUser
from audrey.identity import CloudflareAccessClaims
from audrey.tools.discovery import ToolRegistry, ToolSpec


def _spec(name: str, *, available: bool) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        server_url="http://custom-tools:8001",
        path=f"/{name}",
        available=available,
        unavailable_reason=None if available else "dependency_unavailable:qdrant",
    )


async def test_bounded_retry_waits_for_partial_capability_recovery(monkeypatch):
    live = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=False),
    })
    partial = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=False),
    })
    recovered = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=True),
    })
    discoveries = [partial, recovered]
    skills = SimpleNamespace(refresh_availability=Mock())

    async def fake_discover_all(_servers):
        return discoveries.pop(0)

    monkeypatch.setattr(main_module, "discover_all", fake_discover_all)

    await main_module._retry_tool_discovery(
        live,
        ["http://custom-tools:8001"],
        skills=skills,
        attempts=2,
        interval_s=0,
    )

    assert discoveries == []
    assert live.names() == ["kb_search", "web_search"]
    assert live.get("kb_search") is not None
    assert skills.refresh_availability.call_args_list == [
        call(frozenset({"web_search"})),
        call(frozenset({"kb_search", "web_search"})),
    ]


class _Verifier:
    async def verify(self, _token: str) -> CloudflareAccessClaims:
        return CloudflareAccessClaims(subject="tool-reader", email="reader@example.com")


@pytest.fixture
def tool_state(monkeypatch):
    live = ToolRegistry(by_name={"web_search": _spec("web_search", available=True)})
    skills = SimpleNamespace(refresh_availability=Mock())
    cfg = SimpleNamespace(
        tools={"servers": ["http://custom-tools:8001"]},
        env=SimpleNamespace(owui_auth_enabled=False),
    )
    monkeypatch.setattr(main_module.app.state, "cfg", cfg, raising=False)
    monkeypatch.setattr(main_module.app.state, "tools", live, raising=False)
    monkeypatch.setattr(main_module.app.state, "skills", skills, raising=False)
    monkeypatch.setattr(main_module.app.state, "cloudflare_access_verifier", None, raising=False)
    return live, skills, cfg


async def test_tool_catalog_requires_authentication(tool_state):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main_module.app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/tools")

    assert response.status_code == 401
    assert "tools" not in response.json()


@pytest.mark.parametrize("status, expected_http", [("active", 200), ("pending", 403), ("disabled", 403)])
async def test_tool_catalog_enforces_active_account(
    tool_state, monkeypatch, tmp_path, status, expected_http
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    await store.resolve_external_identity(
        provider="cloudflare_access",
        subject="tool-reader",
        email="reader@example.com",
        display_name="Reader",
        role="user",
        auth_method="cloudflare_access",
        initial_status="pending" if status == "pending" else "active",
    )
    if status == "disabled":
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE app_users SET status = 'disabled'")
    monkeypatch.setattr(main_module.app.state, "application_store", store, raising=False)
    monkeypatch.setattr(main_module.app.state, "cloudflare_access_verifier", _Verifier())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_module.app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/tools", headers={"Cf-Access-Jwt-Assertion": "verified-reader"}
            )
    finally:
        store.close()

    assert response.status_code == expected_http
    if status == "active":
        assert [tool["name"] for tool in response.json()["tools"]] == ["web_search"]
    else:
        assert "tools" not in response.json()


@pytest.mark.parametrize("scopes, expected_http", [(["compat:full"], 200), (["account:read"], 403)])
async def test_tool_catalog_preserves_personal_token_scope_gate(
    tool_state, monkeypatch, tmp_path, scopes, expected_http
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="owui",
        subject="token-reader",
        email="token-reader@example.com",
        display_name="Reader",
        role="user",
        auth_method="owui_bearer",
    )
    issued = await store.create_personal_token(
        user_id=owner.user_id, name="Tool catalogue", scopes=scopes, expires_at=None
    )
    monkeypatch.setattr(main_module.app.state, "application_store", store, raising=False)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_module.app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/tools", headers={"Authorization": f"Bearer {issued.token}"})
    finally:
        store.close()

    assert response.status_code == expected_http


async def test_empty_rediscovery_preserves_live_registry_and_skills(tool_state, monkeypatch):
    live, skills, cfg = tool_state
    original_records = live.by_name.copy()
    discover = AsyncMock(return_value=ToolRegistry())
    monkeypatch.setattr(main_module, "discover_all", discover)

    with pytest.raises(HTTPException) as error:
        await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin", owui_id="admin-subject"))

    assert error.value.status_code == 503
    discover.assert_awaited_once_with(cfg.tools["servers"])
    assert main_module.app.state.tools is live
    assert live.by_name == original_records
    assert live.get("web_search") is original_records["web_search"]
    assert main_module.app.state.skills is skills
    skills.refresh_availability.assert_not_called()


@pytest.mark.parametrize("available", [True, False])
async def test_successful_rediscovery_replaces_records_in_place(
    tool_state, monkeypatch, available
):
    live, skills, _cfg = tool_state
    spec = _spec("web_fetch", available=available)
    monkeypatch.setattr(main_module, "discover_all", AsyncMock(return_value=ToolRegistry(by_name={spec.name: spec})))

    result = await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin", owui_id="admin-subject"))

    names = ["web_fetch"] if available else []
    assert result == {"tools": names, "count": len(names), "declared_count": 1}
    assert main_module.app.state.tools is live
    assert live.by_name == {"web_fetch": spec}
    skills.refresh_availability.assert_called_once_with(frozenset(names))


async def test_empty_rediscovery_is_valid_without_configured_servers(tool_state, monkeypatch):
    live, skills, cfg = tool_state
    cfg.tools["servers"] = []
    discover = AsyncMock(return_value=ToolRegistry())
    monkeypatch.setattr(main_module, "discover_all", discover)

    result = await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin", owui_id="admin-subject"))

    assert result == {"tools": [], "count": 0, "declared_count": 0}
    discover.assert_awaited_once_with([])
    assert main_module.app.state.tools is live
    assert live.by_name == {}
    skills.refresh_availability.assert_called_once_with(frozenset())


async def test_legacy_upload_page_is_retired_without_removing_file_api():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main_module.app), base_url="http://test"
    ) as client:
        assert (await client.get("/upload")).status_code == 404
        assert (await client.head("/upload")).status_code == 404

    paths = {route.path for route in main_module.app.routes}
    assert "/v1/files" in paths
    assert "/v1/files/{file_id}" in paths
