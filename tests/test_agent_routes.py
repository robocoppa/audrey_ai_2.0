"""Saved agents use native owner identity, revision guards and real persistence."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.agent_dispatch import AgentDispatchStoppedError
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


async def _rerun(state, task, **changes):
    return await state.client.post(f"/api/agents/{task['id']}/actions", json={
        "action": "rerun", "expected_revision": task["revision"], **changes,
    })


async def _complete_task(state):
    task = (await _create(state)).json()

    async def execute(_task, step_id):
        return f"Saved {step_id}"

    assert await AgentRuntime(state.store.agents, execute).run_once()
    return (await state.client.get(f"/api/agents/{task['id']}")).json()


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


async def test_api_returns_persisted_title_and_short_fallback_without_exposing_original_paragraph(agents):
    objective = "Compare the public release notes and explain all meaningful changes. " * 10
    created = (await _create(agents, objective=objective)).json()
    assert created["objective"] == objective.strip()
    assert len(created["title"]) <= 72 and created["title"] != created["objective"]
    claimed = await agents.store.agents.claim()
    lease = {"task_id": claimed["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    operation = await agents.store.agents.reserve_title_operation(**lease, step_id="plan")
    await agents.store.agents.save_title(**lease, step_id="plan", operation_id=operation, title="Public Release Notes Comparison")
    current = await agents.client.get(f"/api/agents/{created['id']}")
    listing = await agents.client.get("/api/agents")
    assert current.json()["title"] == listing.json()["items"][0]["title"] == "Public Release Notes Comparison"
    assert current.json()["objective"] == created["objective"]
    assert not {"title_attempted", "title_generated", "operation_id"} & current.json().keys()


async def test_foreign_agent_and_missing_agent_are_indistinguishable(agents):
    task = (await _create(agents)).json()
    agents.current[0] = agents.owners[1]
    assert (await agents.client.get("/api/agents")).json() == {"items": []}
    for target in (task["id"], "missing"):
        assert (await agents.client.get(f"/api/agents/{target}")).status_code == 404
        for action_name in ("cancel", "rerun"):
            action = await agents.client.post(f"/api/agents/{target}/actions", json={
                "action": action_name, "expected_revision": task["revision"],
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


@pytest.mark.parametrize("failure_step", ["research", "review"])
async def test_http_rerun_continues_the_same_failed_agent_from_saved_steps(agents, failure_step):
    task = (await _create(agents)).json()
    seen = []
    fail = True

    async def execute(current, step_id):
        seen.append((current["id"], step_id))
        if fail and step_id == failure_step:
            raise AgentDispatchStoppedError(f"{step_id.capitalize()} returned no report text.")
        return f"Saved {step_id}"

    runtime = AgentRuntime(agents.store.agents, execute)
    assert await runtime.run_once()
    failed = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    assert failed["status"] == "failed"
    assert failed["steps"][0]["result"] == "Saved plan"

    response = await _rerun(agents, failed)
    assert response.status_code == 200, response.text
    queued = response.json()
    assert queued["id"] == task["id"] and queued["status"] == "queued"
    assert queued["created_at"] == task["created_at"]
    assert queued["steps"] == failed["steps"]
    assert queued["error"] == "" and queued["attempts"] == failed["attempts"] == 1
    assert response.headers["cache-control"] == "private, no-store"

    fail = False
    assert await runtime.run_once()
    saved = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    assert saved["id"] == task["id"] and saved["status"] == "completed"
    assert saved["result"] == "Saved review" and saved["attempts"] == 2
    expected = ["plan", "research", "research", "review"] if failure_step == "research" else [
        "plan", "research", "review", "review",
    ]
    assert seen == [(task["id"], step_id) for step_id in expected]
    listing = (await agents.client.get("/api/agents")).json()["items"]
    assert [item["id"] for item in listing] == [task["id"]]


async def test_http_completed_rerun_keeps_plan_and_rejects_duplicate_revision(agents):
    completed = await _complete_task(agents)
    wakes = len(agents.wakes)
    response = await _rerun(agents, completed)
    assert response.status_code == 200, response.text
    queued = response.json()
    assert queued["id"] == completed["id"] and queued["status"] == "queued"
    assert queued["steps"][0] == completed["steps"][0]
    assert all(step["status"] == "pending" and step["result"] == "" for step in queued["steps"][1:])
    assert queued["result"] == "" and queued["attempts"] == completed["attempts"]
    assert len(agents.wakes) == wakes + 1
    stale = await _rerun(agents, completed)
    assert stale.status_code == 409
    assert (await agents.client.get(f"/api/agents/{completed['id']}")).json() == queued
    assert len(agents.wakes) == wakes + 1


@pytest.mark.parametrize("revocation", ["tool", "model", "disabled"])
async def test_http_rerun_rechecks_authority_and_keeps_the_saved_report(agents, revocation):
    completed = await _complete_task(agents)
    if revocation == "tool":
        agents.app.state.tools.by_name.pop("web_fetch")
    elif revocation == "model":
        administrator = await agents.store.resolve_external_identity(
            provider="test", subject="operator", email="operator@example.com", display_name="Operator",
            role="admin", auth_method="test",
        )
        await agents.store.set_model_access_policy(
            actor_user_id=administrator.user_id, model_id="local", enabled=False, audience="users",
        )
    else:
        agents.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
    wakes = len(agents.wakes)
    response = await _rerun(agents, completed)
    assert response.status_code == 503, response.text
    assert (await agents.client.get(f"/api/agents/{completed['id']}")).json() == completed
    assert len(agents.wakes) == wakes


@pytest.mark.parametrize("cap", ["max_queued_per_user", "max_queued_global", "max_attempts"])
async def test_http_rerun_uses_current_server_admission_and_attempt_limits(agents, cap):
    completed = await _complete_task(agents)
    if cap != "max_attempts":
        if cap == "max_queued_global":
            agents.current[0] = agents.owners[1]
        assert (await _create(agents, request_id="waiting")).status_code == 201
        agents.current[0] = agents.owners[0]
    agents.app.state.cfg.raw["agent_tasks"] = {cap: 1}
    wakes = len(agents.wakes)
    response = await _rerun(agents, completed)
    assert response.status_code == 422, response.text
    expected_error = "allowance exhausted" if cap == "max_attempts" else "capacity reached"
    assert expected_error in response.text
    assert (await agents.client.get(f"/api/agents/{completed['id']}")).json() == completed
    assert len(agents.wakes) == wakes


async def test_native_agents_require_authentication():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/agents")).status_code == 401
        assert (await client.post("/api/agents", json={"objective": "Test", "request_id": "x"})).status_code == 401
