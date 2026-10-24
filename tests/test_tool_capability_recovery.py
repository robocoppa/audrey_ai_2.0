"""Authenticated tool inspection and safe in-place discovery recovery."""

from __future__ import annotations

import asyncio
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
from audrey.pipeline import graph as graph_module
from audrey.skills import SkillLimits, SkillRegistry
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
        env=SimpleNamespace(),
    )
    monkeypatch.setattr(main_module.app.state, "cfg", cfg, raising=False)
    monkeypatch.setattr(main_module.app.state, "tools", live, raising=False)
    monkeypatch.setattr(main_module.app.state, "skills", skills, raising=False)
    monkeypatch.setattr(main_module.app.state, "tools_discovery_lock", asyncio.Lock(), raising=False)
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
        await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin"))

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

    result = await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin"))

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

    result = await main_module.rediscover_tools(_admin=AuthedUser(email="admin@example.com", role="admin"))

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


async def test_live_monitor_tracks_loss_and_recovery_in_existing_graph_and_skills(
    monkeypatch, tmp_path,
):
    ready = ToolRegistry(by_name={"kb_search": _spec("kb_search", available=True)})
    degraded = ToolRegistry(by_name={"kb_search": _spec("kb_search", available=False)})
    live = ToolRegistry(by_name=ready.by_name.copy())
    bundle = tmp_path / "document-analysis"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nid: document-analysis\nname: Document analysis\n"
        "description: Read document evidence.\nversion: 1\n"
        "allowed_tools: [kb_search]\nsupported_modes: [auto]\n"
        "resources: []\n---\nRead the evidence.\n",
    )
    skills = SkillRegistry(
        enabled=True, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=frozenset({"kb_search"}), available_tools=frozenset(live.names()),
    )
    snapshots = asyncio.Queue()
    publications = asyncio.Queue()
    update = skills.refresh_availability

    def record_publication(names):
        update(names)
        publications.put_nowait(names)

    monkeypatch.setattr(skills, "refresh_availability", record_publication)

    async def discover(_servers):
        return await snapshots.get()

    calls = []

    async def generate(*_args, tools, **_kwargs):
        calls.append(tools.names())
        assert tools is live
        return "example-model", {"message": {"content": "answer"}}

    monkeypatch.setattr(main_module, "discover_all", discover)
    monkeypatch.setattr(graph_module, "run_fast_path", generate)
    graph = graph_module.build_graph(
        SimpleNamespace(raw={}, router={}, timeouts={}),
        Mock(), Mock(), Mock(), Mock(), live, Mock(),
    )
    fast_node = graph.nodes["fast_path"].bound
    state = {"task_type": "general", "messages": [{"role": "user", "content": "Hello"}]}
    await fast_node.ainvoke(state)
    assert skills.snapshot().status == "ready"
    monitor = asyncio.create_task(main_module._maintain_tool_discovery(
        live, ["http://custom-tools:8001"], skills=skills,
        lock=asyncio.Lock(), interval_s=0,
    ))
    try:
        await snapshots.put(degraded)
        assert await asyncio.wait_for(publications.get(), timeout=1) == frozenset()
        assert skills.snapshot().status == "degraded"
        await fast_node.ainvoke(state)
        await snapshots.put(ready)
        assert await asyncio.wait_for(publications.get(), timeout=1) == frozenset({"kb_search"})
        assert skills.snapshot().status == "ready"
        await fast_node.ainvoke(state)
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor

    assert calls == [["kb_search"], [], ["kb_search"]]


async def test_live_monitor_recovers_after_empty_snapshot_and_fetch_exception(monkeypatch):
    original = _spec("web_search", available=True)
    live = ToolRegistry(by_name={original.name: original})
    recovery = ToolRegistry(by_name={"web_fetch": _spec("web_fetch", available=True)})
    snapshots = asyncio.Queue()
    fetched = asyncio.Queue()
    published = asyncio.Event()
    skills = SimpleNamespace(refresh_availability=Mock(side_effect=lambda _names: published.set()))

    async def discover(_servers):
        value = await snapshots.get()
        fetched.put_nowait(value)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(main_module, "discover_all", discover)
    monitor = asyncio.create_task(main_module._maintain_tool_discovery(
        live, ["http://custom-tools:8001"], skills=skills, interval_s=0,
    ))
    try:
        await snapshots.put(ToolRegistry())
        await asyncio.wait_for(fetched.get(), timeout=1)
        assert live.by_name == {original.name: original}
        await snapshots.put(httpx.ConnectError("offline"))
        await asyncio.wait_for(fetched.get(), timeout=1)
        assert live.by_name == {original.name: original}
        skills.refresh_availability.assert_not_called()
        await snapshots.put(recovery)
        await asyncio.wait_for(published.wait(), timeout=1)
        assert live.names() == ["web_fetch"]
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor


async def test_slow_monitor_continues_after_bounded_startup_retries(monkeypatch):
    live = ToolRegistry()
    retry = AsyncMock()
    monkeypatch.setattr(main_module, "_retry_tool_discovery", retry)
    discovered = asyncio.Event()
    skills = SimpleNamespace(refresh_availability=Mock(side_effect=lambda _names: discovered.set()))
    monkeypatch.setattr(main_module, "discover_all", AsyncMock(return_value=ToolRegistry(
        by_name={"calculate": _spec("calculate", available=True)},
    )))
    lock = asyncio.Lock()
    monitor = asyncio.create_task(main_module._maintain_tool_discovery(
        live, ["http://custom-tools:8001"], skills=skills, lock=lock, interval_s=0,
    ))
    try:
        await asyncio.wait_for(discovered.wait(), timeout=1)
        retry.assert_awaited_once_with(
            live, ["http://custom-tools:8001"], skills=skills, lock=lock,
        )
        assert live.names() == ["calculate"]
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor


@pytest.mark.parametrize("startup_retry", [False, True])
async def test_discovery_cancellation_does_not_publish_pending_snapshot(monkeypatch, startup_retry):
    original = _spec("web_search", available=True)
    live = ToolRegistry(by_name={original.name: original})
    started = asyncio.Event()
    finish = asyncio.Event()
    skills = SimpleNamespace(refresh_availability=Mock())
    lock = asyncio.Lock()

    async def discover(_servers):
        started.set()
        await finish.wait()
        return ToolRegistry(by_name={"kb_search": _spec("kb_search", available=False)})

    monkeypatch.setattr(main_module, "discover_all", discover)
    worker = main_module._retry_tool_discovery if startup_retry else main_module._maintain_tool_discovery
    task = asyncio.create_task(worker(
        live, ["http://custom-tools:8001"], skills=skills, lock=lock, interval_s=0,
    ))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finish.set()

    assert live.by_name == {original.name: original}
    skills.refresh_availability.assert_not_called()
    assert not lock.locked()


async def test_manual_rediscovery_waits_for_background_fetch_before_publishing_newer_view(
    tool_state, monkeypatch,
):
    live, skills, cfg = tool_state
    lock = main_module.app.state.tools_discovery_lock
    old_fetch_started = asyncio.Event()
    old_fetch_finish = asyncio.Event()
    old = ToolRegistry(by_name={"kb_search": _spec("kb_search", available=True)})
    current = ToolRegistry(by_name={"kb_search": _spec("kb_search", available=False)})
    fetch_count = 0

    async def discover(_servers):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 1:
            old_fetch_started.set()
            await old_fetch_finish.wait()
            return old
        return current

    monkeypatch.setattr(main_module, "discover_all", discover)
    background = asyncio.create_task(main_module._refresh_tool_discovery(
        live, cfg.tools["servers"], skills=skills, lock=lock,
    ))
    await asyncio.wait_for(old_fetch_started.wait(), timeout=1)
    manual = asyncio.create_task(main_module.rediscover_tools(
        _admin=AuthedUser(email="admin@example.com", role="admin"),
    ))
    try:
        await asyncio.sleep(0)
        assert fetch_count == 1
        assert not manual.done()
    finally:
        old_fetch_finish.set()
        await background
        await manual

    assert fetch_count == 2
    assert live.by_name == current.by_name
    assert live.names() == []
    assert skills.refresh_availability.call_args_list == [
        call(frozenset({"kb_search"})), call(frozenset()),
    ]
