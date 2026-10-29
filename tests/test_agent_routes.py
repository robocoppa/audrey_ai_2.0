"""Saved agents use native owner identity, revision guards and real persistence."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.routes.app import router
from audrey.tools.discovery import ToolRegistry, ToolSpec


@pytest.fixture
async def agents(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    current = [owners[0]]
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    app.state.tools = ToolRegistry(by_name={name: ToolSpec(
        name=name, description=name, parameters={"type": "object"},
        server_url="http://tools", path=f"/{name}",
    ) for name in ("web_search", "web_fetch")})
    wakes = []
    app.state.agent_runtime = SimpleNamespace(wake=lambda: wakes.append(True))
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owners=owners, current=current, app=app, client=client, wakes=wakes)
    store.close()


async def _create(state, **changes):
    return await state.client.post("/api/agents", json={
        "objective": "Compare the public release notes.", "request_id": "create-1", **changes,
    })


async def test_native_agents_create_idempotently_and_hide_worker_authority(agents):
    first = await _create(agents)
    assert first.status_code == 201, first.text
    task = first.json()
    assert task["status"] == "queued" and task["attempts"] == 0
    assert [step["id"] for step in task["steps"]] == ["plan", "research", "review"]
    assert not {"lease_token", "lease_expires_at", "fence", "user_id", "limits"} & task.keys()
    assert (await _create(agents)).json()["id"] == task["id"]
    conflict = await _create(agents, objective="A different objective.")
    assert conflict.status_code == 409
    listing = await agents.client.get("/api/agents")
    assert listing.json() == {"items": [{**task, "summary": True}]}
    assert listing.headers["cache-control"] == "private, no-store"
    assert len(agents.wakes) == 2


async def test_foreign_agent_and_missing_agent_are_indistinguishable(agents):
    task = (await _create(agents)).json()
    agents.current[0] = agents.owners[1]
    assert (await agents.client.get("/api/agents")).json() == {"items": []}
    for target in (task["id"], "missing"):
        assert (await agents.client.get(f"/api/agents/{target}")).status_code == 404
        action = await agents.client.post(f"/api/agents/{target}/actions", json={
            "action": "cancel", "expected_revision": task["revision"],
        })
        assert action.status_code == 404
        assert (await agents.client.delete(
            f"/api/agents/{target}", params={"expected_revision": task["revision"]},
        )).status_code == 404


async def test_controls_require_current_revision_and_allow_terminal_history_removal(agents):
    task = (await _create(agents)).json()
    url = f"/api/agents/{task['id']}"
    paused = await agents.client.post(f"{url}/actions", json={
        "action": "pause", "expected_revision": task["revision"],
    })
    assert paused.status_code == 200 and paused.json()["status"] == "paused"
    stale = await agents.client.post(f"{url}/actions", json={
        "action": "resume", "expected_revision": task["revision"],
    })
    assert stale.status_code == 409
    current = paused.json()
    assert (await agents.client.delete(url, params={"expected_revision": current["revision"]})).status_code == 409
    cancelled = await agents.client.post(f"{url}/actions", json={
        "action": "cancel", "expected_revision": current["revision"],
    })
    assert cancelled.status_code == 200
    removed = await agents.client.delete(url, params={"expected_revision": cancelled.json()["revision"]})
    assert removed.status_code == 200 and removed.json() == {"deleted": True}
    assert (await agents.client.get(url)).status_code == 404


@pytest.mark.parametrize("extra", [
    {"user_id": "bob"}, {"limits": {"max_operations": 999}},
    {"private_required": True}, {"project_id": "foreign"}, {"file_ids": ["foreign"]},
    {"model": "kimi-k3:cloud"},
])
async def test_first_slice_rejects_unimplemented_scope_and_client_resource_overrides(agents, extra):
    assert (await _create(agents, **extra)).status_code == 422
    assert (await agents.client.get("/api/agents")).json() == {"items": []}


async def test_disabled_tools_block_new_work_but_preserve_saved_access(agents):
    task = (await _create(agents)).json()
    agents.app.state.tools.by_name.pop("web_fetch")
    assert (await _create(agents, request_id="tools-down")).status_code == 503
    assert (await agents.client.get(f"/api/agents/{task['id']}")).status_code == 200
    assert (await agents.client.get("/api/agents")).json()["items"][0]["id"] == task["id"]


async def test_http_creation_connects_to_durable_runtime_and_saved_result(agents):
    task = (await _create(agents)).json()

    async def execute(_task, step_id):
        return f"Completed {step_id} with a saved report."

    runtime = AgentRuntime(agents.store.agents, execute)
    assert await runtime.run_once()
    detail = await agents.client.get(f"/api/agents/{task['id']}")
    assert detail.json()["status"] == "completed"
    assert detail.json()["result"] == "Completed review with a saved report."
    assert all(step["status"] == "completed" for step in detail.json()["steps"])
    summary = (await agents.client.get("/api/agents")).json()["items"][0]
    assert summary["summary"] is True and summary["result"] == ""
    assert all(step["result"] == "" for step in summary["steps"])
    assert not await runtime.run_once()


async def test_native_agents_require_authentication():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/agents")).status_code == 401
        assert (await client.post("/api/agents", json={"objective": "Test", "request_id": "x"})).status_code == 401
