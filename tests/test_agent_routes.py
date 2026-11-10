"""Saved agents use native owner identity, revision guards and real persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.agent_dispatch import AgentDispatchStoppedError, AgentInputRequest
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


async def _add_owner_to_bots(state):
    administrator = await state.store.resolve_external_identity(
        provider="test", subject="operator", email="operator@example.com", display_name="Operator",
        role="admin", auth_method="test",
    )
    await state.store.admin_update_user(
        actor_user_id=administrator.user_id,
        target_user_id=state.owners[0].user_id,
        groups=["users", "bots"],
    )
    principal = await state.store.get_principal(user_id=state.owners[0].user_id)
    assert principal.role == "user" and principal.groups == frozenset({"users", "bots"})
    state.current[0] = principal


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


@pytest.mark.parametrize("files,project", [
    (["file"], None), (["a", "b", "c", "d"], "project"),
    (["a", "a"], "project"), ([123], "project"), ([""], "project"),
])
async def test_file_input_request_shape_rejected_before_task_creation(agents, files, project):
    response = await _create(agents, file_ids=files, project_id=project)
    assert response.status_code == 422
    assert not await agents.store.agents.list(user_id=agents.owners[0].user_id)
    assert not agents.wakes


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
    {"private_required": True}, {"file_ids": ["foreign"]},
    {"model": "kimi-k3:cloud"},
])
async def test_agents_reject_unsupported_scope_and_client_resource_overrides(agents, extra):
    assert (await _create(agents, **extra)).status_code == 422
    assert (await agents.client.get("/api/agents")).json() == {"items": []}


async def test_missing_web_tools_allow_general_tasks_and_preserve_saved_access(agents):
    task = (await _create(agents)).json()
    agents.app.state.tools.by_name.clear()
    response = await _create(agents, request_id="tools-down", objective="Draft a concise project update.")
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["status"] == "queued" and created["objective"] == "Draft a concise project update."
    assert (await agents.client.get(f"/api/agents/{task['id']}")).status_code == 200
    assert {item["id"] for item in (await agents.client.get("/api/agents")).json()["items"]} == {task["id"], created["id"]}


async def test_bots_group_blocks_new_and_resumed_work_but_preserves_owned_controls(agents):
    task = (await _create(agents)).json()
    url = f"/api/agents/{task['id']}"
    await _add_owner_to_bots(agents)
    wakes = len(agents.wakes)
    assert (await _create(agents, request_id="bot-work")).status_code == 503
    assert (await agents.client.get(url)).json() == task
    assert (await agents.client.get("/api/agents")).json()["items"][0]["id"] == task["id"]
    assert len(agents.wakes) == wakes
    paused = await agents.client.post(f"{url}/actions", json={
        "action": "pause", "expected_revision": task["revision"],
    })
    assert paused.status_code == 200 and paused.json()["status"] == "paused"
    resumed = await agents.client.post(f"{url}/actions", json={
        "action": "resume", "expected_revision": paused.json()["revision"],
    })
    assert resumed.status_code == 503
    assert (await agents.client.get(url)).json() == paused.json()
    assert len(agents.wakes) == wakes + 1
    cancelled = await agents.client.post(f"{url}/actions", json={
        "action": "cancel", "expected_revision": paused.json()["revision"],
    })
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    removed = await agents.client.delete(url, params={"expected_revision": cancelled.json()["revision"]})
    assert removed.status_code == 200 and removed.json() == {"deleted": True}


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


@pytest.mark.parametrize("revocation", ["model", "disabled", "bots"])
async def test_http_rerun_rechecks_authority_and_keeps_the_saved_report(agents, revocation):
    completed = await _complete_task(agents)
    if revocation == "model":
        administrator = await agents.store.resolve_external_identity(
            provider="test", subject="operator", email="operator@example.com", display_name="Operator",
            role="admin", auth_method="test",
        )
        await agents.store.set_model_access_policy(
            actor_user_id=administrator.user_id, model_id="local", enabled=False, audience="users",
        )
    elif revocation == "bots":
        await _add_owner_to_bots(agents)
    else:
        agents.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
    wakes = len(agents.wakes)
    response = await _rerun(agents, completed)
    assert response.status_code == 503, response.text
    assert (await agents.client.get(f"/api/agents/{completed['id']}")).json() == completed
    assert len(agents.wakes) == wakes


async def test_missing_web_tools_allow_rerun_of_same_agent_with_saved_plan(agents):
    completed = await _complete_task(agents)
    agents.app.state.tools.by_name.clear()
    response = await _rerun(agents, completed)
    assert response.status_code == 200, response.text
    queued = response.json()
    assert queued["id"] == completed["id"] and queued["status"] == "queued"
    assert queued["steps"][0] == completed["steps"][0]
    assert queued["attempts"] == completed["attempts"]
    assert queued["steps"][1]["label"] == "Work"


@pytest.mark.parametrize("cap", ["max_queued_per_user", "max_queued_global"])
async def test_http_rerun_uses_current_server_admission_limits(agents, cap):
    completed = await _complete_task(agents)
    if cap == "max_queued_global":
        agents.current[0] = agents.owners[1]
    assert (await _create(agents, request_id="waiting")).status_code == 201
    agents.current[0] = agents.owners[0]
    agents.app.state.cfg.raw["agent_tasks"] = {cap: 1}
    wakes = len(agents.wakes)
    response = await _rerun(agents, completed)
    assert response.status_code == 422, response.text
    assert "capacity reached" in response.text
    assert (await agents.client.get(f"/api/agents/{completed['id']}")).json() == completed
    assert len(agents.wakes) == wakes


async def test_http_rerun_recovers_attempt_exhaustion_on_same_agent_with_saved_plan(agents):
    agents.app.state.cfg.raw["agent_tasks"] = {"max_attempts": 1}
    task = (await _create(agents)).json()
    fail = True
    seen = []

    async def execute(current, step_id):
        seen.append((current["id"], step_id, current["attempts"]))
        if fail and step_id == "research":
            raise RuntimeError("Transient model transport failure")
        return f"Saved {step_id}"

    runtime = AgentRuntime(agents.store.agents, execute, max_attempts=1)
    assert await runtime.run_once()
    assert not await runtime.run_once()
    failed = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    assert failed["status"] == "failed" and failed["attempts"] == 1
    assert failed["steps"][0]["result"] == "Saved plan"
    wakes = len(agents.wakes)
    response = await _rerun(agents, failed)
    assert response.status_code == 200, response.text
    queued = response.json()
    assert queued["id"] == task["id"] and queued["steps"][0] == failed["steps"][0]
    assert queued["attempts"] == 1 and "attempt_base" not in queued
    assert len(agents.wakes) == wakes + 1
    assert (await _rerun(agents, failed)).status_code == 409
    fail = False
    assert await runtime.run_once()
    completed = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    assert completed["status"] == "completed" and completed["attempts"] == 2
    assert completed["steps"][0] == failed["steps"][0]
    assert seen == [(task["id"], stage, attempt) for stage, attempt in (
        ("plan", 1), ("research", 1), ("research", 2), ("review", 2),
    )]
    listing = (await agents.client.get("/api/agents")).json()["items"]
    assert [item["id"] for item in listing] == [task["id"]]
    assert all("attempt_base" not in item for item in listing)


async def test_native_agents_require_authentication():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/agents")).status_code == 401
        assert (await client.post("/api/agents", json={"objective": "Test", "request_id": "x"})).status_code == 401
        assert (await client.post("/api/agents/task/inputs/question/answer", json={"expected_revision": 1, "answer": "Example"})).status_code == 401


async def _waiting_task(state):
    created = (await _create(state)).json()

    async def execute(_task, _step):
        return AgentInputRequest("Which public product should I research?")

    assert await AgentRuntime(state.store.agents, execute).run_once()
    return (await state.client.get(f"/api/agents/{created['id']}")).json()


def _answer_path(task, request_id=None):
    return f"/api/agents/{task['id']}/inputs/{request_id or task['input_requests'][0]['id']}/answer"


async def test_native_input_wait_and_answer_resume_same_agent_and_invalidate_activity(agents):
    waiting = await _waiting_task(agents)
    assert waiting["status"] == "waiting_for_input" and waiting["steps"][0]["status"] == "pending"
    listing = (await agents.client.get("/api/agents")).json()["items"]
    assert listing[0]["status"] == "waiting_for_input" and listing[0]["input_requests"] == []
    notice = (await agents.client.get("/api/activity")).json()["items"][0]
    assert notice["kind"] == "input_required" and notice["target"]["revision"] == waiting["revision"]
    assert (await agents.client.get(f"/api/activity/agent/{notice['id']}/task")).json() == waiting
    body = {"expected_revision": waiting["revision"], "answer": "Stable public Example releases"}
    wakes = len(agents.wakes)
    response = await agents.client.post(_answer_path(waiting), json=body)
    assert response.status_code == 200, response.text
    queued = response.json()
    assert response.headers["cache-control"] == "private, no-store"
    assert queued["id"] == waiting["id"] and queued["status"] == "queued"
    assert queued["input_requests"][0]["answer"] == body["answer"]
    assert queued["steps"] == waiting["steps"] and queued["attempts"] == waiting["attempts"]
    assert len(agents.wakes) == wakes + 1
    assert (await agents.client.get(f"/api/activity/agent/{notice['id']}/task")).status_code == 404
    assert (await agents.client.get("/api/activity/unread")).json() == {"unread_count": 0}
    assert (await agents.client.post(_answer_path(waiting), json=body)).json() == queued
    different = {**body, "answer": "A different product"}
    assert (await agents.client.post(_answer_path(waiting), json=different)).status_code == 409


@pytest.mark.parametrize("kind", ["foreign", "stale", "wrong_request", "revoked", "scope"])
async def test_input_answers_reject_foreign_stale_or_revoked_work(agents, kind):
    waiting = await _waiting_task(agents)
    body = {"expected_revision": waiting["revision"], "answer": "Example"}
    path = _answer_path(waiting)
    if kind == "foreign":
        agents.current[0] = agents.owners[1]
    elif kind == "stale":
        body["expected_revision"] -= 1
    elif kind == "wrong_request":
        path = _answer_path(waiting, "missing")
    elif kind == "scope":
        agents.current[0] = replace(agents.owners[0], auth_method="personal_token", scopes=frozenset({"models:read"}))
    else:
        agents.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
    response = await agents.client.post(path, json=body)
    assert response.status_code == {"foreign": 404, "stale": 409, "wrong_request": 409, "revoked": 503, "scope": 403}[kind], response.text
    assert (await agents.store.agents.get(user_id=agents.owners[0].user_id, task_id=waiting["id"])) == waiting


@pytest.mark.parametrize("answer", ["", " ", "x" * 8001, "é" * 4001, {"text": "Example"}])
async def test_input_answer_bounds_are_enforced_without_losing_question(agents, answer):
    waiting = await _waiting_task(agents)
    response = await agents.client.post(_answer_path(waiting), json={"expected_revision": waiting["revision"], "answer": answer})
    assert response.status_code == 422
    assert (await agents.client.get(f"/api/agents/{waiting['id']}")).json() == waiting


@pytest.mark.parametrize("operation", ["get", "list", "answer_input"])
async def test_input_bodies_are_fenced_when_task_is_erased_after_await(agents, monkeypatch, operation):
    waiting = await _waiting_task(agents)
    original = getattr(agents.store.agents, operation)

    async def erased_after_read(**kwargs):
        result = await original(**kwargs)
        with agents.store._lock, agents.store._conn:
            agents.store._conn.execute("DELETE FROM app_agent_tasks WHERE task_id=?", (waiting["id"],))
        return result

    monkeypatch.setattr(agents.store.agents, operation, erased_after_read)
    response = await agents.client.post(_answer_path(waiting), json={"expected_revision": waiting["revision"], "answer": "Example"}) if operation == "answer_input" else await agents.client.get("/api/agents" if operation == "list" else f"/api/agents/{waiting['id']}")
    assert response.status_code == 404 and "Which public product" not in response.text and "Example" not in response.text


async def test_input_storage_failure_is_safe_and_leaves_question_available(agents, monkeypatch):
    waiting = await _waiting_task(agents)

    async def unavailable(**_):
        raise sqlite3.OperationalError("private storage location")

    monkeypatch.setattr(agents.store.agents, "answer_input", unavailable)
    response = await agents.client.post(_answer_path(waiting), json={"expected_revision": waiting["revision"], "answer": "Example"})
    assert response.status_code == 503 and "private storage location" not in response.text


async def test_project_agents_save_canonical_context_and_filter_without_exposing_input_bodies(agents):
    project = await agents.store.projects.create(user_id=agents.owners[0].user_id, name="Release comparison", instructions="Use stable releases only.")
    note = await agents.store.knowledge.create(user_id=agents.owners[0].user_id, project_id=project.project_id,
        kind="decision", text="Only releases from the public mainline count.", pinned=True)
    response = await _create(agents, project_id=project.project_id)
    assert response.status_code == 201, response.text
    task = response.json()
    assert task["project_context"] == {
        "project_id": project.project_id, "project_name": project.name, "available": True,
        "current": True, "instructions_included": True, "knowledge": [{"id": note["id"], "revision": note["revision"]}],
    }
    assert note["text"] not in response.text and project.instructions not in response.text
    listing = await agents.client.get("/api/agents", params={"project_id": project.project_id})
    assert listing.status_code == 200 and listing.json()["items"][0]["project_context"] == task["project_context"]
    assert (await _create(agents, request_id="unscoped")).status_code == 201
    assert len((await agents.client.get("/api/agents", params={"project_id": project.project_id})).json()["items"]) == 1
    assert (await _create(agents, project_id=project.project_id)).json()["id"] == task["id"]
    assert (await _create(agents, project_id=None)).status_code == 409


@pytest.mark.parametrize("kind", ["missing", "foreign", "private"])
async def test_project_agent_creation_and_filters_reject_unavailable_projects(agents, kind):
    project_id = "missing"
    if kind != "missing":
        project = await agents.store.projects.create(user_id=agents.owners[kind == "foreign"].user_id,
            name="Unavailable", private_required=kind == "private")
        project_id = project.project_id
    for response in (
        await _create(agents, project_id=project_id),
        await agents.client.get("/api/agents", params={"project_id": project_id}),
    ):
        assert response.status_code == 404, response.text
    assert (await agents.client.get("/api/agents")).json() == {"items": []}


async def test_project_change_stops_queued_work_but_keeps_its_saved_report_and_plan(agents):
    project = await agents.store.projects.create(user_id=agents.owners[0].user_id, name="Versions", instructions="Compare 2026.")
    created = (await _create(agents, project_id=project.project_id)).json()

    async def execute(_task, step):
        return f"Saved {step} from the selected Project."

    assert await AgentRuntime(agents.store.agents, execute).run_once()
    saved = (await agents.client.get(f"/api/agents/{created['id']}")).json()
    assert saved["status"] == "completed"
    await agents.store.projects.update(user_id=agents.owners[0].user_id, project_id=project.project_id, instructions="Compare 2025 instead.")
    current = (await agents.client.get(f"/api/agents/{created['id']}")).json()
    assert not current["project_context"]["current"] and current["result"] == saved["result"]
    assert (await _rerun(agents, current)).status_code == 409
    assert (await agents.client.get(f"/api/agents/{created['id']}")).json()["steps"] == saved["steps"]
    await agents.store.projects.delete(user_id=agents.owners[0].user_id, project_id=project.project_id)
    retained = await agents.client.get(f"/api/agents/{created['id']}")
    assert retained.status_code == 200 and retained.json()["result"] == saved["result"]
    assert retained.json()["project_context"]["project_name"] is None and not retained.json()["project_context"]["available"]


async def test_answer_cannot_continue_an_agent_after_its_project_context_changes(agents):
    project = await agents.store.projects.create(user_id=agents.owners[0].user_id, name="Release scope")
    task = (await _create(agents, project_id=project.project_id)).json()

    async def execute(_task, _step):
        return AgentInputRequest("Which public product?")

    assert await AgentRuntime(agents.store.agents, execute).run_once()
    waiting = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    await agents.store.projects.update(user_id=agents.owners[0].user_id, project_id=project.project_id, instructions="A changed scope.")
    response = await agents.client.post(_answer_path(waiting), json={"expected_revision": waiting["revision"], "answer": "Example"})
    assert response.status_code == 409 and "Project context changed" in response.text
    current = (await agents.client.get(f"/api/agents/{task['id']}")).json()
    assert current["status"] == "waiting_for_input" and current["input_requests"][0]["answer"] is None


async def test_project_metadata_is_fenced_if_project_is_deleted_after_task_read(agents, monkeypatch):
    project = await agents.store.projects.create(user_id=agents.owners[0].user_id, name="Project metadata")
    task = (await _create(agents, project_id=project.project_id)).json()
    original = agents.store.agents.get

    async def removed_after_read(**kwargs):
        saved = await original(**kwargs)
        await agents.store.projects.delete(user_id=agents.owners[0].user_id, project_id=project.project_id)
        return saved

    monkeypatch.setattr(agents.store.agents, "get", removed_after_read)
    response = await agents.client.get(f"/api/agents/{task['id']}")
    assert response.status_code == 409 and project.name not in response.text


async def _knowledge_work(state, *, owner=None, private=False, kind="action"):
    owner = owner or state.owners[0]
    project = await state.store.projects.create(user_id=owner.user_id, name="Release work", private_required=private)
    entry = await state.store.knowledge.create(
        user_id=owner.user_id, project_id=project.project_id, kind=kind,
        text="Determine the supported public stable release.",
    )
    return project, entry


@pytest.mark.parametrize("kind", ["action", "question"])
async def test_knowledge_work_creates_explicit_project_origin_and_replays_same_agent(agents, kind):
    project, entry = await _knowledge_work(agents, kind=kind)
    body = {"project_id": project.project_id, "knowledge_entry_id": entry["id"], "knowledge_entry_revision": entry["revision"]}
    response = await _create(agents, **body)
    assert response.status_code == 201, response.text
    task = response.json()
    assert task["project_context"]["origin_entry"] == {"id": entry["id"], "revision": 1}
    assert task["project_context"]["knowledge"][0] == task["project_context"]["origin_entry"]
    assert entry["text"] not in response.text and response.headers["cache-control"] == "private, no-store"
    listing = (await agents.client.get("/api/agents", params={"project_id": project.project_id})).json()["items"]
    assert listing[0]["project_context"]["origin_entry"] == task["project_context"]["origin_entry"]
    updated = await agents.store.knowledge.update(
        user_id=agents.owners[0].user_id, project_id=project.project_id, entry_id=entry["id"], expected_revision=1,
        work_status="completed" if kind == "action" else "resolved",
    )
    replay = await _create(agents, **body)
    assert replay.status_code == 201 and replay.json()["id"] == task["id"]
    assert not replay.json()["project_context"]["current"]
    assert (await _create(agents, **{**body, "knowledge_entry_revision": updated["revision"]})).status_code == 409
    assert len(listing) == 1


@pytest.mark.parametrize("extra", [
    {"knowledge_entry_id": "entry"}, {"knowledge_entry_revision": 1},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": None},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": True},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": "1"},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": 1.0},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": 0},
    {"knowledge_entry_id": "entry", "knowledge_entry_revision": -1},
    {"knowledge_entry_id": 42, "knowledge_entry_revision": 1},
    {"knowledge_entry_id": " ", "knowledge_entry_revision": 1},
    {"knowledge_entry_id": "x" * 301, "knowledge_entry_revision": 1},
])
async def test_knowledge_work_api_pair_is_strict_before_creating_or_waking(agents, extra):
    project, _ = await _knowledge_work(agents)
    response = await _create(agents, project_id=project.project_id, **extra)
    assert response.status_code == 422, response.text
    assert (await agents.client.get("/api/agents")).json() == {"items": []}
    assert agents.wakes == []


async def test_knowledge_work_api_requires_project_and_preserves_plain_creation_contract(agents):
    project, entry = await _knowledge_work(agents)
    pair = {"knowledge_entry_id": entry["id"], "knowledge_entry_revision": 1}
    assert (await _create(agents, **pair)).status_code == 422
    ordinary = await _create(agents, project_id=project.project_id)
    assert ordinary.status_code == 201 and "origin_entry" not in ordinary.json()["project_context"]
    assert (await _create(agents, project_id=project.project_id, **pair)).status_code == 409


@pytest.mark.parametrize("invalid", ["missing", "foreign", "private", "closed", "fact", "stale_revision", "file"])
async def test_knowledge_work_api_rejects_unavailable_or_changed_origins(agents, invalid):
    project, entry = await _knowledge_work(agents, private=invalid == "private", kind="fact" if invalid == "fact" else "action")
    body = {"project_id": project.project_id, "knowledge_entry_id": entry["id"], "knowledge_entry_revision": 1}
    if invalid == "missing":
        body["knowledge_entry_id"] = "missing"
    elif invalid == "foreign":
        _, foreign_entry = await _knowledge_work(agents, owner=agents.owners[1])
        body["knowledge_entry_id"] = foreign_entry["id"]
    elif invalid in {"closed", "stale_revision", "file"}:
        changes = {
            "closed": {"work_status": "completed"},
            "stale_revision": {"text": "A changed research question."},
            "file": {"sources": [{"source_id": "file:missing:text", "revision": "old"}]},
        }[invalid]
        current = await agents.store.knowledge.update(
            user_id=agents.owners[0].user_id, project_id=project.project_id, entry_id=entry["id"], expected_revision=1,
            **changes,
        )
        if invalid != "stale_revision":
            body["knowledge_entry_revision"] = current["revision"]
    response = await _create(agents, **body)
    assert response.status_code == (409 if invalid == "stale_revision" else 404), response.text
    assert entry["text"] not in response.text
    assert (await agents.client.get("/api/agents")).json() == {"items": []}
    assert agents.wakes == []


async def test_knowledge_edit_before_create_commit_rejects_stale_draft_without_waking(agents, monkeypatch):
    project, entry = await _knowledge_work(agents)
    create = agents.store.agents.create

    async def changed_before_create(**kwargs):
        await agents.store.knowledge.update(
            user_id=agents.owners[0].user_id, project_id=project.project_id, entry_id=entry["id"], expected_revision=1,
            text="A changed research question before task creation.",
        )
        return await create(**kwargs)

    monkeypatch.setattr(agents.store.agents, "create", changed_before_create)
    response = await _create(agents, project_id=project.project_id, knowledge_entry_id=entry["id"], knowledge_entry_revision=1)
    assert response.status_code == 409 and "Project knowledge changed" in response.text
    assert (await agents.client.get("/api/agents")).json() == {"items": []} and agents.wakes == []


async def test_origin_deletion_after_create_is_fenced_before_http_response(agents, monkeypatch):
    project, entry = await _knowledge_work(agents)
    create = agents.store.agents.create

    async def deleted_after_create(**kwargs):
        result = await create(**kwargs)
        await agents.store.knowledge.delete(
            user_id=agents.owners[0].user_id, project_id=project.project_id, entry_id=entry["id"], expected_revision=1,
        )
        return result

    monkeypatch.setattr(agents.store.agents, "create", deleted_after_create)
    response = await _create(agents, project_id=project.project_id, knowledge_entry_id=entry["id"], knowledge_entry_revision=1)
    assert response.status_code == 409 and entry["text"] not in response.text and agents.wakes == []


async def test_create_returns_current_owned_task_if_worker_starts_before_response(agents, monkeypatch):
    original = agents.store.agents.create
    claimed = []

    async def started_after_create(**kwargs):
        created = await original(**kwargs)
        active = await agents.store.agents.claim()
        claimed.append(active)
        lease = {key: active[key] for key in ("lease_token", "fence")}
        await agents.store.agents.reserve_operation(task_id=active["id"], step_id="plan", **lease)
        return created

    monkeypatch.setattr(agents.store.agents, "create", started_after_create)
    response = await _create(agents)
    assert response.status_code == 201, response.text
    current = response.json()
    assert current["id"] == claimed[0]["id"] and current["status"] == "running"
    assert current["revision"] > 1 and current["steps"][0]["status"] == "running"
    assert not {"lease_token", "lease_expires_at", "fence", "user_id", "limits"} & current.keys()
    monkeypatch.setattr(agents.store.agents, "create", original)
    repeated = await _create(agents)
    assert repeated.status_code == 201 and repeated.json()["id"] == current["id"]
    assert len(await agents.store.agents.list(user_id=agents.owners[0].user_id)) == 1


@pytest.mark.parametrize("operation", ["get", "list"])
async def test_task_reads_refresh_normal_worker_progress_before_response(agents, monkeypatch, operation):
    created = (await _create(agents)).json()
    original = getattr(agents.store.agents, operation)

    async def advanced_after_read(**kwargs):
        snapshot = await original(**kwargs)
        active = await agents.store.agents.claim()
        lease = {"task_id": active["id"], "lease_token": active["lease_token"], "fence": active["fence"]}
        await agents.store.agents.reserve_operation(**lease, step_id="plan")
        await agents.store.agents.checkpoint(**lease, step_id="plan", result="A real saved plan body.")
        return snapshot

    monkeypatch.setattr(agents.store.agents, operation, advanced_after_read)
    response = await agents.client.get("/api/agents" if operation == "list" else f"/api/agents/{created['id']}")
    assert response.status_code == 200, response.text
    current = response.json()["items"][0] if operation == "list" else response.json()
    assert current["id"] == created["id"] and current["status"] == "running"
    assert current["steps"][0]["status"] == "completed" and current["revision"] > created["revision"]
    if operation == "list":
        assert current["summary"] is True and current["result"] == ""
        assert all(step["result"] == "" for step in current["steps"])
        assert current["input_requests"] == [] and "A real saved plan body." not in response.text
    else:
        assert current["steps"][0]["result"] == "A real saved plan body."


@pytest.mark.parametrize("operation", ["control", "answer_input"])
async def test_successful_task_mutations_do_not_conflict_with_immediate_worker_progress(agents, monkeypatch, operation):
    task = await _waiting_task(agents) if operation == "answer_input" else await _complete_task(agents)
    original = getattr(agents.store.agents, operation)

    async def progressed_after_commit(**kwargs):
        committed = await original(**kwargs)
        active = await agents.store.agents.claim()
        assert active["id"] == committed["id"]
        return committed

    monkeypatch.setattr(agents.store.agents, operation, progressed_after_commit)
    response = await agents.client.post(_answer_path(task), json={
        "expected_revision": task["revision"], "answer": "Use the selected source text.",
    }) if operation == "answer_input" else await _rerun(agents, task)
    assert response.status_code == 200, response.text
    current = response.json()
    assert current["id"] == task["id"] and current["status"] == "running"
    assert current["revision"] > task["revision"]
    if operation == "answer_input":
        assert current["input_requests"][0]["answer"] == "Use the selected source text."
    else:
        assert current["steps"][0] == task["steps"][0]


async def test_create_still_fences_owner_revocation_after_commit(agents, monkeypatch):
    original = agents.store.agents.create

    async def revoked_after_create(**kwargs):
        result = await original(**kwargs)
        with agents.store._lock, agents.store._conn:
            agents.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (agents.owners[0].user_id,))
        return result

    monkeypatch.setattr(agents.store.agents, "create", revoked_after_create)
    response = await _create(agents)
    assert response.status_code == 404 and "Compare the public" not in response.text
    assert not agents.wakes
