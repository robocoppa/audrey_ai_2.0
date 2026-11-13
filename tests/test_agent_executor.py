"""Saved agent authority is enforced around real local-model/tool transports."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from audrey.agent_dispatch import (
    AgentDispatchStoppedError,
    AgentInputRequest,
    agent_dispatch_scope,
    agent_tool_execution,
    authorize_agent_tool,
)
from audrey.agent_executor import AgentSettings, ResearchAgentExecutor
from audrey.agent_file_inputs import get_agent_file_inputs
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskError
from audrey.auth import require_principal
from audrey.content_sources import ContentSourceReader
from audrey.kb.file_deletion import FileOperationLocks
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.models.health import HealthTracker
from audrey.models.ollama import GenerationUsageError, OllamaClient, OllamaError
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.prompts import (
    AGENT_PLAN_FILE_READS_SYSTEM,
    AGENT_PLAN_SYSTEM,
    AGENT_RESEARCH_SYSTEM,
    AGENT_REVIEW_SYSTEM,
)
from audrey.pipeline.react import ReactResult
from audrey.routes.app import router as app_router
from audrey.routes.inflight import UserInflightRegistry
from audrey.scheduling import current_local_workload
from audrey.tools.discovery import ToolRegistry, ToolSpec

_MODEL = "qwen3.8:latest"
_PUBLIC_URL = "https://example.com/releases"


def _answer(content="Grounded report.", **changes):
    return {
        "done": True,
        "eval_count": 7,
        "done_reason": "stop",
        "message": {"content": content},
        **changes,
    }


def _search(name="web_search"):
    return _answer("", message={"tool_calls": [{"function": {"name": name, "arguments": {}}}]})


@pytest.fixture
async def state(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "application.sqlite")
    owner = await store.resolve_external_identity(
        provider="test",
        subject="alice",
        email="alice@example.com",
        display_name="Alice",
        role="admin",
        auth_method="test",
    )
    tools = ToolRegistry(
        by_name={
            name: ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                server_url="http://tools.test",
                path=f"/{name}",
            )
            for name in (
                "web_search",
                "web_fetch",
                "memory_recall",
                "memory_store",
                "get_file_text",
                "create_file",
            )
        }
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(
        raw={"agent_tasks": {"max_tokens_per_call": 128}},
        env=SimpleNamespace(ollama_host="http://ollama.test"),
    )
    app.state.tools = tools
    app.state.health = HealthTracker()
    app.state.gate = FairLocalGate()
    app.state.inflight = UserInflightRegistry(max_inflight_per_user=2)
    value = SimpleNamespace(
        app=app,
        store=store,
        owner=owner,
        tools=tools,
        provider_requests=[],
        tool_requests=[],
        clients=[],
        client_options=[],
        responses=[],
        title_responses=[],
        on_title=None,
        tag_reads=0,
        show_reads=0,
        on_tags=None,
        on_show=None,
        on_chat=None,
        on_tool=None,
        model_admissions=[],
        tool_admissions=[],
        installed=[{"name": _MODEL, "model": _MODEL, "digest": "a" * 64}],
        metadata={
            "details": {"format": "gguf"},
            "model_info": {
                "general.architecture": "qwen3",
                "general.parameter_count": 27000000000,
            },
        },
    )

    async def provider(request):
        value.provider_requests.append(request)
        if request.url.path == "/api/tags":
            value.tag_reads += 1
            if value.on_tags:
                await value.on_tags()
            return httpx.Response(200, json={"models": value.installed})
        if request.url.path == "/api/show":
            value.show_reads += 1
            if value.on_show:
                await value.on_show()
            return httpx.Response(200, json=value.metadata)
        assert request.url.path == "/api/chat"
        value.model_admissions.append((
            current_local_workload(),
            app.state.inflight.snapshot(),
            app.state.gate.pressure_snapshot(),
        ))
        if value.on_chat:
            await value.on_chat()
        payload = json.loads(request.content)
        if payload["messages"][0]["content"].startswith("You create concise agent titles."):
            if value.on_title:
                await value.on_title()
            return httpx.Response(200, json=value.title_responses.pop(0) if value.title_responses else _answer("Public Release Notes Research"))
        return httpx.Response(200, json=value.responses.pop(0) if value.responses else _answer())

    async def tool(request):
        value.tool_requests.append(request)
        value.tool_admissions.append((
            current_local_workload(),
            app.state.inflight.snapshot(),
            app.state.gate.pressure_snapshot(),
        ))
        if value.on_tool:
            await value.on_tool()
        return httpx.Response(
            200, json={"results": [{"url": _PUBLIC_URL, "text": "Release notes"}]}
        )

    # ReAct owns a separate HTTP client for public tools. Both transports remain
    # real httpx clients so dispatch order and wire payloads are exercised.
    original_client = httpx.AsyncClient

    def http_client(**kwargs):
        if "base_url" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(tool)
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", http_client)

    def client_factory(endpoint, **kwargs):
        value.client_options.append((endpoint, kwargs))
        client = OllamaClient(endpoint, transport=httpx.MockTransport(provider), **kwargs)
        value.clients.append(client)
        return client

    value.executor = ResearchAgentExecutor(app, client_factory=client_factory)
    yield value
    for client in value.clients:
        await client.aclose()
    store.close()


@pytest.fixture
async def file_state(state, tmp_path):
    db = UploadsDB(tmp_path / "agent-outputs.sqlite")
    state.app.state.uploads_db = db
    state.app.state.storage_lifecycle = StorageLifecycle(db)
    state.app.state.file_operation_locks = FileOperationLocks()
    state.app.state.cfg.raw["kb"] = {
        "upload_root": str(tmp_path / "files"), "max_user_bytes": 1_000_000,
    }
    state.app.include_router(app_router)
    state.app.dependency_overrides[require_principal] = lambda: state.owner
    state.files_db = db
    try:
        yield state
    finally:
        db.close()


def _file_call(**changes):
    return _answer("", message={"tool_calls": [{"function": {
        "name": "create_file", "arguments": {
            "filename": "task-notes.md", "format": "md", "content": "# Useful task output", **changes,
        },
    }}]})


async def _step(state, step_id="plan", *, max_operations=96, generated_title=True, project_id=None, knowledge_entry=None, file_inputs=None, objective="Compare the public release notes."):
    task = await state.store.agents.create(
        user_id=state.owner.user_id,
        objective=objective,
        request_id=f"request-{step_id}",
        limits={"max_operations": max_operations},
        project_id=project_id,
        file_inputs=file_inputs,
        **({"knowledge_entry_id": knowledge_entry["id"], "knowledge_entry_revision": knowledge_entry["revision"]} if knowledge_entry else {}),
    )
    claimed = await state.store.agents.claim()
    assert claimed["id"] == task["id"]
    if generated_title:
        # Existing step-boundary tests exercise tasks that already have a title;
        # the title path has separate transport/budget/revocation coverage below.
        with state.store._lock, state.store._conn:
            state.store._conn.execute(
                "UPDATE app_agent_tasks SET title='Public Release Notes Research',title_attempted=1 WHERE task_id=?",
                (task["id"],),
            )
        claimed.update(title="Public Release Notes Research", title_generated=True, title_attempted=1)
    lease = {key: claimed[key] for key in ("lease_token", "fence")}
    lease["task_id"] = claimed["id"]
    for previous in ("plan", "research", "review"):
        if previous == step_id:
            break
        await state.store.agents.reserve_operation(**lease, step_id=previous)
        claimed = await state.store.agents.checkpoint(
            **lease, step_id=previous, result=f"Saved {previous}"
        )
    await state.store.agents.reserve_operation(**lease, step_id=step_id)
    return claimed


def _content_requests(state, *, include_titles=False):
    return [
        json.loads(request.content)
        for request in state.provider_requests
        if request.url.path == "/api/chat"
        and (include_titles or not json.loads(request.content)["messages"][0]["content"].startswith("You create concise agent titles."))
    ]


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
async def test_titles_and_every_research_round_enter_background_admission_at_transport(state, step_id):
    task = await _step(state, step_id, generated_title=False)
    if step_id == "research":
        state.responses.extend([_search(), _search("web_fetch"), _answer()])

    await state.executor(task, step_id)

    # The optional title and every real model dispatch use the same admission
    # policy. Inspect the HTTP boundary rather than a caller-supplied argument.
    assert len(state.model_admissions) == (4 if step_id == "research" else 2)
    for workload, inflight, gpu in state.model_admissions:
        assert workload == "background"
        assert inflight.get(state.owner.storage_namespace) == 1
        assert gpu["in_use"] == 1
    assert len(state.tool_admissions) == (2 if step_id == "research" else 0)
    for workload, inflight, gpu in state.tool_admissions:
        assert workload == "interactive"
        assert not inflight.get(state.owner.storage_namespace)
        assert gpu["in_use"] == 0
    assert current_local_workload() == "interactive"
    assert not state.app.state.inflight.pressure_snapshot()["in_use"]
    assert not state.app.state.gate.pressure_snapshot()["in_use"]


async def test_blocked_public_research_releases_same_user_admission_for_foreground_chat(state):
    state.app.state.inflight = UserInflightRegistry(max_inflight_per_user=1)
    task = await _step(state, "research")
    state.responses.extend([_search(), _answer()])
    tool_entered, release_tool = asyncio.Event(), asyncio.Event()

    async def blocked_public_tool():
        tool_entered.set()
        await release_tool.wait()

    state.on_tool = blocked_public_tool
    executing = asyncio.create_task(state.executor(task, "research"))
    foreground_calls = []

    def foreground_transport(request):
        foreground_calls.append(json.loads(request.content))
        assert current_local_workload() == "interactive"
        assert state.app.state.inflight.snapshot()[state.owner.storage_namespace] == 1
        assert state.app.state.gate.pressure_snapshot()["in_use"] == 1
        return httpx.Response(200, json=_answer("Foreground answer."))

    foreground = OllamaClient("http://ollama.test", transport=httpx.MockTransport(foreground_transport))
    try:
        await asyncio.wait_for(tool_entered.wait(), 2)
        assert not executing.done()
        assert state.tool_admissions[0][1].get(state.owner.storage_namespace, 0) == 0
        assert state.tool_admissions[0][2]["in_use"] == 0
        async with asyncio.timeout(2):
            async with state.app.state.inflight.slot(state.owner.storage_namespace):
                async with state.app.state.gate.acquire(
                    _MODEL, location="local", user_id=state.owner.storage_namespace,
                ):
                    response = await foreground.chat(
                        model=_MODEL, messages=[{"role": "user", "content": "A foreground question."}],
                    )
        assert response["message"]["content"] == "Foreground answer."
        assert len(foreground_calls) == 1 and not executing.done()
        release_tool.set()
        assert _PUBLIC_URL in await asyncio.wait_for(executing, 2)
    finally:
        release_tool.set()
        if not executing.done():
            executing.cancel()
        await asyncio.gather(executing, return_exceptions=True)
        await foreground.aclose()
    assert len(_content_requests(state)) == 2
    assert not state.app.state.inflight.pressure_snapshot()["in_use"]
    assert not state.app.state.gate.pressure_snapshot()["in_use"]


@pytest.mark.parametrize("queued_on", ["inflight", "gpu"])
@pytest.mark.parametrize("generated_title", [True, False])
@pytest.mark.parametrize("revocation", ["cancel", "model", "project"])
async def test_queued_background_work_rechecks_authority_without_charging_model_work(
    state, monkeypatch, queued_on, generated_title, revocation,
):
    project = await state.store.projects.create(
        user_id=state.owner.user_id, name="Fixed research scope", instructions="Original scope.",
    )
    task = await _step(state, generated_title=generated_title, project_id=project.project_id)
    operations = state.store._conn.execute(
        "SELECT operations FROM app_agent_tasks WHERE task_id=?", (task["id"],),
    ).fetchone()[0]
    state.app.state.inflight = UserInflightRegistry(max_inflight_per_user=1)
    if queued_on == "inflight":
        registry, method_name = state.app.state.inflight, "slot"
        hold = registry.slot(state.owner.storage_namespace)
    else:
        registry, method_name = state.app.state.gate, "acquire"
        hold = registry.acquire(_MODEL, location="local", user_id="foreground-owner")
    queued = asyncio.Event()
    original_admit = getattr(registry, method_name)

    @asynccontextmanager
    async def observe_queue(*args, **kwargs):
        queued.set()
        async with original_admit(*args, **kwargs) as permit:
            yield permit

    monkeypatch.setattr(registry, method_name, observe_queue)
    executing = None
    try:
        async with hold:
            executing = asyncio.create_task(state.executor(task, "plan"))
            await asyncio.wait_for(queued.wait(), 2)
            assert registry.pressure_snapshot()["waiting"] == 1
            assert not _content_requests(state, include_titles=True)
            assert all(row["kind"] == "stage" for row in await _activity(state, task))
            if revocation == "cancel":
                current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
                await state.store.agents.control(
                    user_id=state.owner.user_id, task_id=task["id"],
                    expected_revision=current["revision"], action="cancel",
                )
            elif revocation == "model":
                await _revoke(state, task, "model")
            else:
                await state.store.projects.update(
                    user_id=state.owner.user_id, project_id=project.project_id,
                    instructions="Changed scope requires new consent.",
                )
        with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
            await asyncio.wait_for(executing, 2)
    finally:
        if executing is not None:
            if not executing.done():
                executing.cancel()
            await asyncio.gather(executing, return_exceptions=True)
    assert not _content_requests(state, include_titles=True) and not state.tool_requests
    assert all(row["kind"] == "stage" for row in await _activity(state, task))
    assert state.store._conn.execute(
        "SELECT operations FROM app_agent_tasks WHERE task_id=?", (task["id"],),
    ).fetchone()[0] == operations
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["steps"][0]["result"]
    if not generated_title:
        title = state.store._conn.execute(
            "SELECT title,title_attempted FROM app_agent_tasks WHERE task_id=?", (task["id"],),
        ).fetchone()
        assert tuple(title) == ("", 0)
    assert not state.app.state.inflight.pressure_snapshot()["in_use"]
    assert not state.app.state.gate.pressure_snapshot()["in_use"]


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
async def test_every_step_uses_only_local_qwen_and_only_research_has_public_tools(state, step_id):
    task = await _step(state, step_id)
    if step_id == "research":
        state.responses.extend([_search(), _answer()])
    result = await state.executor(task, step_id)
    assert result == "Grounded report." + (
        f"\n\nSelected public sources:\n- {_PUBLIC_URL}" if step_id == "research" else ""
    )
    requests = _content_requests(state)
    assert len(requests) == (2 if step_id == "research" else 1)
    for payload in requests:
        assert payload["model"] == _MODEL and payload["options"]["num_predict"] == 128
        assert not any(AGENT_PLAN_FILE_READS_SYSTEM in message["content"]
                       for message in payload["messages"])
        offered = {tool["function"]["name"] for tool in payload.get("tools", [])}
        assert offered == ({"web_search", "web_fetch"} if step_id == "research" else set())
    supplied = json.loads(
        next(row["content"] for row in requests[0]["messages"] if row["role"] == "user")
    )
    if step_id == "review":
        assert supplied.pop("current_output_files") == []
    assert supplied == {
        "objective": task["objective"],
        "saved_work": {
            step["id"]: step["result"] for step in task["steps"] if step["status"] == "completed"
        },
    }
    assert state.client_options == [("http://ollama.test", {"trust_env": False})]
    assert state.tag_reads == 1 + len(requests) and state.show_reads == state.tag_reads
    assert all(client._client.is_closed for client in state.clients)
    assert "memory_recall" in state.tools.names()
    assert len(state.tool_requests) == (1 if step_id == "research" else 0)


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
@pytest.mark.parametrize(
    "configuration,expected",
    [
        ({}, {"plan": 2048, "research": 8192, "review": 8192}),
        (
            {"max_tokens_per_call": 4096, "plan_max_tokens_per_call": 1024},
            {"plan": 1024, "research": 4096, "review": 4096},
        ),
        (
            {"max_tokens_per_call": 128, "plan_max_tokens_per_call": 4096},
            {"plan": 128, "research": 128, "review": 128},
        ),
    ],
)
async def test_step_output_limits_are_serialized_on_every_local_model_call(
    state, step_id, configuration, expected
):
    state.app.state.cfg.raw["agent_tasks"] = configuration
    task = await _step(state, step_id)
    if step_id == "research":
        state.responses.extend([_search(), _answer()])

    await state.executor(task, step_id)

    requests = _content_requests(state)
    assert len(requests) == (2 if step_id == "research" else 1)
    assert all(request["options"]["num_predict"] == expected[step_id] for request in requests)
    assert all(request["model"] == _MODEL for request in requests)
    if step_id == "plan":
        assert all(request["think"] is False for request in requests)
    else:
        # The narrow planner fix must not disable reasoning for task execution
        # or Review, including later Work calls after a real tool result.
        assert all("think" not in request for request in requests)


@pytest.mark.parametrize("step_id,expected", [("plan", 1024), ("research", 4096), ("review", 4096)])
async def test_generation_budget_bounds_an_oversized_request_to_the_stage_limit(
    state, monkeypatch, step_id, expected
):
    state.app.state.cfg.raw["agent_tasks"] = {
        "max_tokens_per_call": 4096,
        "plan_max_tokens_per_call": 1024,
    }
    task = await _step(state, step_id)
    if step_id == "research":
        state.responses.extend([_search(), _answer()])
    original_chat = OllamaClient.chat

    async def oversized_request(client, **kwargs):
        # Exercise the real transport guard even if a caller overrequests tokens.
        kwargs["options"] = {"num_predict": 8192}
        return await original_chat(client, **kwargs)

    monkeypatch.setattr(OllamaClient, "chat", oversized_request)
    await state.executor(task, step_id)
    assert all(
        request["options"]["num_predict"] == expected for request in _content_requests(state)
    )


@pytest.mark.parametrize("invalid", [127, 8193, 2048.0, "2048", True, None])
def test_plan_output_limit_requires_an_integer_in_the_supported_range(invalid):
    with pytest.raises(ValidationError):
        AgentSettings(plan_max_tokens_per_call=invalid)


@pytest.mark.parametrize("model", ["kimi-k3:cloud", "qwen3.8:cloud", "ornith-1.5:35b"])
def test_configuration_cannot_select_a_cloud_or_other_local_model(model):
    with pytest.raises(ValidationError):
        AgentSettings(model=model)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "remote_tag",
        "remote_show",
        "digest",
        "duplicate",
        "details",
        "architecture",
        "row",
    ],
)
async def test_unverified_local_inventory_stops_before_content_and_has_no_fallback(state, invalid):
    task = await _step(state)
    if invalid == "missing":
        state.installed.clear()
    elif invalid == "remote_tag":
        state.installed[0]["remote_model"] = "qwen3.8"
    elif invalid == "remote_show":
        state.metadata["remote_host"] = "https://cloud.test"
    elif invalid == "digest":
        state.installed[0]["digest"] = "unknown"
    elif invalid == "duplicate":
        state.installed.append(dict(state.installed[0]))
    elif invalid == "details":
        state.metadata["details"] = {}
    elif invalid == "architecture":
        state.metadata["model_info"] = {}
    else:
        state.installed.append(None)
    with pytest.raises(AgentDispatchStoppedError, match=r"Local qwen3\.8"):
        await state.executor(task, "plan")
    assert not _content_requests(state)
    assert all(client._client.is_closed for client in state.clients)


async def test_digest_change_between_verification_and_generation_stops_content(state):
    task = await _step(state)

    async def change_digest():
        if state.tag_reads == 2:
            state.installed[0]["digest"] = "b" * 64

    state.on_tags = change_digest
    with pytest.raises(AgentDispatchStoppedError, match="Cloud fallback is disabled"):
        await state.executor(task, "plan")
    assert not _content_requests(state)


async def _revoke(state, task, kind):
    if kind == "account":
        with state.store._lock:
            state.store._conn.execute(
                "UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,)
            )
            state.store._conn.commit()
    elif kind == "model":
        await state.store.set_model_access_policy(
            actor_user_id=state.owner.user_id, model_id="local", enabled=False, audience="users"
        )
    elif kind == "tool":
        state.tools.by_name["web_search"].available = False
    elif kind == "bots":
        await state.store.admin_update_user(
            actor_user_id=state.owner.user_id,
            target_user_id=state.owner.user_id,
            groups=[*state.owner.groups, "bots"],
        )
    else:
        current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
        await state.store.agents.control(
            user_id=state.owner.user_id,
            task_id=task["id"],
            expected_revision=current["revision"],
            action="pause",
        )


@pytest.mark.parametrize("generated_title", [True, False])
async def test_canonical_bots_group_stops_before_any_local_model_request(state, generated_title):
    task = await _step(state, generated_title=generated_title)
    await _revoke(state, task, "bots")
    principal = await state.store.get_principal(user_id=state.owner.user_id)
    assert principal.role == "admin" and "bots" in principal.groups
    with pytest.raises(AgentDispatchStoppedError, match="Agent account access is unavailable"):
        await state.executor(task, "plan")
    assert not state.provider_requests and not state.tool_requests and not state.clients


@pytest.mark.parametrize("metadata_read", [1, 2])
@pytest.mark.parametrize("kind", ["account", "model", "pause", "bots"])
async def test_live_authority_revocation_before_generation_prevents_content(
    state, kind, metadata_read
):
    task = await _step(state)

    async def revoke():
        if state.show_reads == metadata_read:
            await _revoke(state, task, kind)

    state.on_show = revoke
    with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
        await state.executor(task, "plan")
    assert not _content_requests(state)


@pytest.mark.parametrize("kind", ["account", "model", "tool", "pause", "endpoint", "bots"])
async def test_live_authority_checked_again_before_public_tool_dispatch(state, kind):
    task = await _step(state, "research")
    state.responses.append(_search())

    async def revoke():
        if kind == "endpoint":
            state.tools.by_name["web_search"].path = "/changed"
        else:
            await _revoke(state, task, kind)

    state.on_chat = revoke
    with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
        await state.executor(task, "research")
    assert len(_content_requests(state)) == 1
    assert not state.tool_requests


async def test_personal_tool_request_is_blocked_at_dispatch(state):
    task = await _step(state, "research")
    state.responses.append(_search("memory_recall"))
    with pytest.raises(AgentDispatchStoppedError, match="tool is unavailable"):
        await state.executor(task, "research")
    assert not state.tool_requests


async def test_work_creates_owned_download_without_http_and_review_keeps_canonical_links(file_state):
    s = file_state
    project = await s.store.projects.create(user_id=s.owner.user_id, name="Task deliverables")
    task = await _step(s, "research", project_id=project.project_id)
    original_registry = dict(s.tools.by_name)
    s.responses.extend([_file_call(), _answer("The task output is complete.")])
    result = await s.executor(task, "research")
    assert result == "The task output is complete."
    assert not s.tool_requests and s.tools.by_name == original_registry
    rows = await s.files_db.list_user(s.owner.storage_namespace)
    assert len(rows) == 1 and rows[0]["origin"] == "generated"
    outputs = (await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"]))["outputs"]
    assert len(outputs) == 1 and outputs[0]["filename"] == "task-notes.md"
    assert outputs[0]["file_id"] == rows[0]["file_id"]
    payloads = _content_requests(s)
    native_spec = next(item["function"] for item in payloads[0]["tools"] if item["function"]["name"] == "create_file")
    assert "user" not in native_spec["parameters"]["properties"]
    assert "project_id" not in native_spec["parameters"]["properties"]
    assert json.loads(payloads[1]["messages"][-1]["content"])["file_id"] == outputs[0]["file_id"]
    lease = {key: task[key] for key in ("lease_token", "fence")}
    lease["task_id"] = task["id"]
    saved = await s.store.agents.checkpoint(**lease, step_id="research", result=result)
    assert outputs[0]["download_url"] in saved["steps"][1]["result"]
    s.responses.append(_answer("Reviewed task output without a model-supplied link."))
    reviewed = await s.executor(saved, "review")
    await s.store.agents.checkpoint(**lease, step_id="review", result=reviewed)
    completed = await s.store.agents.complete(**lease, result=reviewed)
    assert outputs[0]["download_url"] in completed["result"]
    assert {tool["function"]["name"] for tool in _content_requests(s)[-1]["tools"]} == {"create_file"}
    # Use the ASGI transport rather than the fixture's tool transport.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        downloaded = await client.get(outputs[0]["download_url"])
        assert downloaded.status_code == 200 and downloaded.text == "# Useful task output"
        current = (await client.get(f"/api/agents/{task['id']}")).json()
        assert current["outputs"] == outputs
        summary = (await client.get("/api/agents")).json()["items"][0]
        assert "outputs" not in summary
    await s.store.agents.control(
        user_id=s.owner.user_id, task_id=task["id"], expected_revision=completed["revision"], action="rerun",
    )
    s.responses.extend([_file_call(), _answer("Reused the task output."), _answer("Reviewed the reused task output.")])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    repeated = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert repeated["status"] == "completed" and repeated["outputs"] == outputs
    assert len(await s.files_db.list_user(s.owner.storage_namespace)) == 1
    assert repeated["steps"][0] == completed["steps"][0]


async def test_planning_cannot_create_native_outputs(file_state):
    s = file_state
    task = await _step(s, "plan")
    s.responses.append(_file_call())
    with pytest.raises(AgentDispatchStoppedError, match="returned no report text"):
        await s.executor(task, "plan")
    assert not _content_requests(s)[0].get("tools")
    assert not await s.files_db.list_user(s.owner.storage_namespace)
    assert not s.tool_requests


async def test_native_creation_consumes_the_saved_operation_allowance(file_state):
    s = file_state
    task = await _step(s, "research", max_operations=3)
    s.responses.append(_file_call())
    with pytest.raises(AgentTaskError, match="operation allowance"):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1
    assert not await s.files_db.list_user(s.owner.storage_namespace)
    assert not s.tool_requests


@pytest.mark.parametrize("kind", ["account", "model", "pause", "bots", "digest"])
async def test_native_output_rechecks_live_authority_at_dispatch(file_state, kind):
    s = file_state
    task = await _step(s, "research")
    s.responses.append(_file_call())

    async def revoke():
        if s.show_reads == 3:
            if kind == "digest":
                s.installed[0]["digest"] = "b" * 64
            else:
                await _revoke(s, task, kind)

    # Digest mutation must precede tags parsing; other authority changes can
    # happen during the renderer dispatch's final metadata await.
    if kind == "digest":
        async def change_digest():
            if s.tag_reads == 3:
                s.installed[0]["digest"] = "b" * 64
        s.on_tags = change_digest
    else:
        s.on_show = revoke
    with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1
    assert not await s.files_db.list_user(s.owner.storage_namespace)
    assert not s.tool_requests


@pytest.mark.parametrize(
    ("step_id", "tool_name"),
    [
        ("research", "memory_recall"),
        ("review", "memory_recall"),
        ("review", "get_file_text"),
        ("review", "web_search"),
        ("review", "web_fetch"),
    ],
)
async def test_native_output_capability_does_not_grant_other_tools(file_state, step_id, tool_name):
    s = file_state
    task = await _step(s, step_id)
    s.responses.append(_search(tool_name))
    with pytest.raises(AgentDispatchStoppedError, match="tool is unavailable"):
        await s.executor(task, step_id)
    assert not await s.files_db.list_user(s.owner.storage_namespace)
    assert not s.tool_requests


async def test_native_output_callbacks_expire_with_the_work_scope(file_state):
    from audrey.agent_files import agent_file_handler

    s = file_state
    task = await _step(s, "research")
    release = asyncio.Event()
    captured = {}
    arguments = {"filename": "late.md", "format": "md", "content": "Too late."}

    async def later():
        await release.wait()
        return await captured["handler"](arguments)

    async def runner(*_args, **_kwargs):
        captured["handler"] = agent_file_handler(user_id=s.owner.storage_namespace)
        captured["child"] = asyncio.create_task(later())
        return ReactResult(content="Work finished without creating a file.", tool_rounds=0, done_reason="stop")

    executor = ResearchAgentExecutor(s.app, client_factory=s.executor.client_factory, react_runner=runner)
    assert await executor(task, "research") == "Work finished without creating a file."
    release.set()
    with pytest.raises(AgentDispatchStoppedError, match="attempt has stopped"):
        await captured["child"]
    with pytest.raises(AgentDispatchStoppedError, match="attempt has stopped"):
        await captured["handler"](arguments)
    assert not await s.files_db.list_user(s.owner.storage_namespace)
    assert not s.tool_requests


async def test_every_provider_and_tool_dispatch_consumes_the_saved_operation_budget(state):
    task = await _step(state, "research", max_operations=3)
    state.responses.append(_search())
    with pytest.raises(AgentTaskError):
        await state.executor(task, "research")
    assert len(_content_requests(state)) == 1
    assert not state.tool_requests


async def test_missing_provider_usage_stops_the_step(state):
    task = await _step(state)
    state.responses.append({"done": True, "message": {"content": "Unaccounted answer"}})
    with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
        await state.executor(task, "plan")
    assert len(_content_requests(state)) == 1


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
@pytest.mark.parametrize(
    "content,reason,detail",
    [
        ("partial", "length", "reached the local model's output limit"),
        ("", "stop", "returned no report text"),
        ("   ", "stop", "returned no report text"),
        (42, "stop", "returned invalid report text"),
    ],
)
async def test_truncated_or_empty_outputs_identify_the_actual_stage(
    state, step_id, content, reason, detail
):
    task = await _step(state, step_id)
    state.responses.append(_answer(content, done_reason=reason))
    label = "Work" if step_id == "research" else step_id.capitalize()
    with pytest.raises(AgentDispatchStoppedError, match=f"{label} {detail}"):
        await state.executor(task, step_id)
    assert not state.tool_requests


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
async def test_reasoning_only_output_has_a_specific_error_and_safe_diagnostics(
    state, step_id, caplog
):
    task = await _step(state, step_id)
    state.responses.append(_answer("", message={"content": "", "thinking": "Secret reasoning"}))
    with caplog.at_level("INFO", logger="audrey.agent_executor"):
        with pytest.raises(
            AgentDispatchStoppedError, match="returned no report text after reasoning"
        ):
            await state.executor(task, step_id)
    assert f"task={task['id']} step={step_id} call=1 stop=stop" in caplog.text
    assert "output_tokens=7 content_chars=0 thinking_chars=16 tool_calls=0" in caplog.text
    assert "Secret reasoning" not in caplog.text and task["objective"] not in caplog.text


async def test_truncated_tool_call_does_not_dispatch_partial_arguments(state):
    task = await _step(state, "research")
    state.responses.append({**_search(), "done_reason": "length"})
    with pytest.raises(
        AgentDispatchStoppedError, match="Work reached the local model's output limit"
    ):
        await state.executor(task, "research")
    assert not state.tool_requests


async def test_diagnostics_do_not_log_report_or_untrusted_stop_reason(state, caplog):
    task = await _step(state)
    state.responses.append(_answer("Private report body", done_reason="\nUntrusted stop data"))
    with caplog.at_level("INFO", logger="audrey.agent_executor"):
        assert await state.executor(task, "plan") == "Private report body"
    assert "stop=unknown" in caplog.text
    assert "content_chars=19" in caplog.text
    assert "Private report body" not in caplog.text and "Untrusted stop data" not in caplog.text


async def test_larger_research_limit_reruns_the_same_agent_with_its_saved_plan_and_usage(state):
    state.app.state.cfg.raw["agent_tasks"] = {"max_tokens_per_call": 2048}
    task = await state.store.agents.create(
        user_id=state.owner.user_id, objective="Research a timeline.", request_id="truncated"
    )
    state.responses.extend(
        [_answer("Saved work plan"), _answer("Partial report", done_reason="length")]
    )
    runtime = AgentRuntime(state.store.agents, state.executor)
    assert await runtime.run_once()
    failed = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert failed["status"] == "failed" and failed["attempts"] == 1
    assert failed["steps"][0]["result"] == "Saved work plan"
    assert failed["steps"][1]["result"] == "" and failed["result"] == ""
    assert failed["error"].startswith("Work reached the local model's output limit")
    failed_usage = dict(
        state.store._conn.execute(
            "SELECT attempts,operations,active_seconds FROM app_agent_tasks WHERE task_id=?",
            (task["id"],),
        ).fetchone()
    )
    assert failed_usage["operations"] == 5
    receipt_ids = {
        row[0]
        for row in state.store._conn.execute(
            "SELECT operation_id FROM app_agent_operations WHERE task_id=?", (task["id"],)
        ).fetchall()
    }

    state.app.state.cfg.raw["agent_tasks"]["max_tokens_per_call"] = 8192
    queued = await state.store.agents.control(
        user_id=state.owner.user_id,
        task_id=task["id"],
        expected_revision=failed["revision"],
        action="rerun",
    )
    assert queued["id"] == task["id"] and queued["steps"] == failed["steps"]
    assert (
        dict(
            state.store._conn.execute(
                "SELECT attempts,operations,active_seconds FROM app_agent_tasks WHERE task_id=?",
                (task["id"],),
            ).fetchone()
        )
        == failed_usage
    )
    state.responses.extend([_search(), _answer("Complete research"), _answer("Final report")])

    assert await runtime.run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["id"] == task["id"] and saved["status"] == "completed"
    assert saved["steps"][0] == failed["steps"][0]
    assert saved["result"] == "Final report" and saved["error"] == ""
    assert len(await state.store.agents.list(user_id=state.owner.user_id)) == 1
    requests = _content_requests(state)
    assert [request["options"]["num_predict"] for request in requests] == [
        2048,
        2048,
        8192,
        8192,
        8192,
    ]
    assert all(request["model"] == _MODEL for request in requests)
    rerun_input = json.loads(
        next(row["content"] for row in requests[2]["messages"] if row["role"] == "user")
    )
    assert rerun_input["saved_work"] == {"plan": "Saved work plan"}
    final_usage = state.store._conn.execute(
        "SELECT attempts,operations,active_seconds FROM app_agent_tasks WHERE task_id=?",
        (task["id"],),
    ).fetchone()
    assert final_usage["attempts"] == 2 and final_usage["operations"] == 11
    assert final_usage["active_seconds"] >= failed_usage["active_seconds"]
    final_receipt_ids = {
        row[0]
        for row in state.store._conn.execute(
            "SELECT operation_id FROM app_agent_operations WHERE task_id=?", (task["id"],)
        ).fetchall()
    }
    assert receipt_ids < final_receipt_ids and len(final_receipt_ids) == 11
    assert len(state.tool_requests) == 1 and not state.responses
    assert not await runtime.run_once()


async def test_utf8_output_limit_is_checked_in_bytes(state):
    task = await _step(state)
    state.responses.append(_answer("é" * 30001))
    with pytest.raises(AgentDispatchStoppedError, match="saved-output limit"):
        await state.executor(task, "plan")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///private/report",
        "javascript:alert(1)",
        "https://alice:password@example.com/data",
        "https://[invalid",
        "https://example.com/" + "x" * 2049,
    ],
)
async def test_work_discards_unusable_source_links_without_requiring_sources(state, url):
    task = await _step(state, "research")

    async def report(client, _health, _tools, **kwargs):
        answer = await client.chat(model=kwargs["model"], messages=kwargs["messages"])
        return ReactResult(
            content=answer["message"]["content"], tool_rounds=0, retrieved=[{"url": url}]
        )

    state.executor.react_runner = report
    assert await state.executor(task, "research") == "Grounded report."
    assert not state.tool_requests


async def test_runner_cannot_change_the_pinned_model(state):
    task = await _step(state, "research")

    async def wrong_model(client, *_args, **_kwargs):
        await client.chat(model="kimi-k3:cloud", messages=[])

    state.executor.react_runner = wrong_model
    with pytest.raises(AgentDispatchStoppedError, match=r"only use local qwen3\.8"):
        await state.executor(task, "research")
    assert not _content_requests(state)


@pytest.mark.parametrize("local_available", [True, False])
async def test_durable_runtime_completes_locally_or_stops_without_fallback(state, local_available):
    task = await state.store.agents.create(
        user_id=state.owner.user_id,
        objective="Compare the public release notes.",
        request_id="runtime",
    )
    if not local_available:
        state.installed[0]["remote_host"] = "https://cloud.test"
    state.responses.extend(
        [_answer("Research plan"), _search(), _answer("Research report"), _answer("Final report")]
    )
    runtime = AgentRuntime(state.store.agents, state.executor)
    assert await runtime.run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == ("completed" if local_available else "failed")
    assert saved["attempts"] == 1
    if local_available:
        assert saved["result"] == "Final report"
        assert _PUBLIC_URL in saved["steps"][1]["result"]
        assert all(row["model"] == _MODEL for row in _content_requests(state))
        assert len(_content_requests(state)) == 4
    else:
        assert "Cloud fallback is disabled" in saved["error"]
        assert not _content_requests(state) and not state.tool_requests
    assert not await runtime.run_once()


async def test_first_local_title_is_budgeted_saved_before_plan_and_never_replaces_objective(state):
    task = await _step(state, generated_title=False)
    reads = []

    async def observe_title_before_plan():
        current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
        reads.append(current)

    state.on_chat = observe_title_before_plan
    assert await state.executor(task, "plan") == "Grounded report."
    requests = _content_requests(state, include_titles=True)
    assert len(requests) == 2
    title, plan = requests
    assert title["model"] == _MODEL and title["think"] is False
    assert title["options"] == {"num_predict": 64, "temperature": 0}
    title_request = next(request for request in state.provider_requests if request.url.path == "/api/chat")
    assert title_request.extensions["timeout"]["read"] == 60.0
    assert not title.get("tools") and title["messages"][1]["content"] == task["objective"]
    assert json.loads(plan["messages"][1]["content"])["objective"] == task["objective"]
    assert reads[1]["title"] == "Public Release Notes Research"
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["title"] == reads[1]["title"] and saved["objective"] == task["objective"]
    assert state.store._conn.execute("SELECT operations FROM app_agent_tasks").fetchone()[0] == 3
    assert state.tag_reads == 3 and state.show_reads == 3
    assert state.client_options == [("http://ollama.test", {"trust_env": False})]


@pytest.mark.parametrize("title_response", [
    _answer("Unfinished Public Research", done_reason="length"),
    _answer("Two words"), _answer("", message={"content": "", "thinking": "Hidden reasoning"}),
    _answer("Public Release Notes Research", eval_count=None),
])
async def test_optional_title_failure_keeps_fallback_and_useful_step_work(state, title_response):
    task = await _step(state, generated_title=False)
    state.title_responses.append(title_response)
    assert await state.executor(task, "plan") == "Grounded report."
    row = state.store._conn.execute("SELECT title,title_attempted,operations FROM app_agent_tasks").fetchone()
    assert row["title"] == "" and row["title_attempted"] == 1 and row["operations"] == 3
    assert len(_content_requests(state, include_titles=True)) == 2
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    next_task = await state.store.agents.checkpoint(**lease, step_id="plan", result="Saved plan.")
    state.responses.extend([_search(), _answer()])
    assert _PUBLIC_URL in await state.executor(next_task, "research")
    title_calls = len(_content_requests(state, include_titles=True)) - len(_content_requests(state))
    assert title_calls == 1


async def test_title_does_not_consume_the_last_operation_needed_by_the_real_step(state):
    task = await _step(state, max_operations=2, generated_title=False)
    assert await state.executor(task, "plan") == "Grounded report."
    assert len(_content_requests(state, include_titles=True)) == 1
    row = state.store._conn.execute("SELECT title,title_attempted,operations FROM app_agent_tasks").fetchone()
    assert tuple(row) == ("", 1, 2)


@pytest.mark.parametrize("invalid", [0, -1, 121, float("inf"), float("nan"), "60", True])
def test_title_timeout_requires_a_bounded_numeric_duration(invalid):
    with pytest.raises(ValueError):
        AgentSettings(title_timeout_seconds=invalid)


async def test_configured_title_timeout_allows_a_slow_title_to_complete(state):
    state.app.state.cfg.raw["agent_tasks"]["title_timeout_seconds"] = 0.5
    task = await _step(state, generated_title=False)

    async def slow_title():
        await asyncio.sleep(0.03)

    state.on_title = slow_title
    assert await state.executor(task, "plan") == "Grounded report."
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["title"] == "Public Release Notes Research"
    title_request = next(request for request in state.provider_requests if request.url.path == "/api/chat")
    assert title_request.extensions["timeout"]["read"] == 0.5
    events = await state.store.agents.events(user_id=state.owner.user_id, task_id=task["id"])
    assert [(event["stage"], event["state"]) for event in events["items"] if event["kind"] == "model"] == [
        ("plan", "completed"), ("title", "completed"),
    ]


async def test_title_timeout_is_optional_and_keeps_its_uncertain_operation_charge(state, caplog):
    task = await _step(state, generated_title=False)
    state.app.state.cfg.raw["agent_tasks"]["title_timeout_seconds"] = 0.01

    async def slow_title():
        await asyncio.sleep(1)

    state.on_title = slow_title
    assert await state.executor(task, "plan") == "Grounded report."
    row = state.store._conn.execute("SELECT title,title_attempted,operations FROM app_agent_tasks").fetchone()
    assert tuple(row) == ("", 1, 3)
    assert len(_content_requests(state, include_titles=True)) == 2
    assert "error_type=TimeoutError" in caplog.text
    assert not state.app.state.inflight.pressure_snapshot()["in_use"]
    assert not state.app.state.gate.pressure_snapshot()["in_use"]
    events = await state.store.agents.events(user_id=state.owner.user_id, task_id=task["id"])
    assert [(event["stage"], event["state"]) for event in events["items"] if event["kind"] == "model"] == [
        ("plan", "completed"), ("title", "failed"),
    ]


@pytest.mark.parametrize("kind", ["account", "bots"])
async def test_revocation_during_title_generation_stops_plan_and_prevents_title_delivery(state, kind):
    task = await _step(state, generated_title=False)

    async def revoke():
        await _revoke(state, task, kind)

    state.on_title = revoke
    with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
        await state.executor(task, "plan")
    assert len(_content_requests(state, include_titles=True)) == 1 and not _content_requests(state)
    assert state.store._conn.execute("SELECT title FROM app_agent_tasks").fetchone()[0] == ""
    assert not state.tool_requests and all(client._client.is_closed for client in state.clients)


async def test_legacy_completed_rerun_generates_title_before_research_and_reuses_saved_plan(state):
    task = await state.store.agents.create(user_id=state.owner.user_id, objective="Research all public orchard gates.", request_id="legacy")
    claimed = await state.store.agents.claim()
    lease = {"task_id": claimed["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    for step_id in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=step_id)
        claimed = await state.store.agents.checkpoint(**lease, step_id=step_id, result=f"Legacy saved {step_id}")
    completed = await state.store.agents.complete(**lease, result="Legacy report.")
    await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=completed["revision"], action="rerun")
    state.responses.extend([_search(), _answer("Fresh research."), _answer("Fresh reviewed report.")])
    runtime = AgentRuntime(state.store.agents, state.executor)
    assert await runtime.run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["id"] == task["id"] and saved["status"] == "completed"
    assert saved["steps"][0] == completed["steps"][0]
    assert saved["title"] == "Public Release Notes Research" and saved["objective"] == task["objective"]
    requests = _content_requests(state, include_titles=True)
    assert len(requests) == 4 and requests[0]["options"]["num_predict"] == 64
    assert json.loads(requests[1]["messages"][1]["content"])["saved_work"] == {"plan": "Legacy saved plan"}


async def test_escaped_child_tool_dispatch_loses_authority_after_scope_exit():
    ready = asyncio.Event()
    calls = []

    async def check(name):
        calls.append(name)

    async def child():
        await ready.wait()
        await authorize_agent_tool("web_search")

    with agent_dispatch_scope(check):
        escaped = asyncio.create_task(child())
    ready.set()
    with pytest.raises(AgentDispatchStoppedError, match="attempt has stopped"):
        await escaped
    assert not calls


@pytest.mark.parametrize("fenced", [False, True])
async def test_planning_question_is_a_typed_local_result_without_research_or_tools(state, fenced):
    task = await _step(state)
    content = json.dumps({"input_request": "Which product's releases should I compare?"})
    state.responses.append(_answer(f"```json\n{content}\n```" if fenced else content))
    result = await state.executor(task, "plan")
    assert result == AgentInputRequest("Which product's releases should I compare?")
    assert len(_content_requests(state)) == 1 and not state.tool_requests
    assert all(client._client.is_closed for client in state.clients)


@pytest.mark.parametrize("content", [
    '{"input_request":""}', '{"input_request":42}', '{"input_request":"Which one?","tool":"send"}',
    '{"input_request":"Which one?"', json.dumps({"input_request": "é" * 1001}),
    '{"input_request":"\\ud800"}', 'Please clarify: {"input_request":"Which one?"}',
])
async def test_malformed_or_oversized_planning_question_stops_before_dispatch(state, content):
    task = await _step(state)
    state.responses.append(_answer(content))
    with pytest.raises(AgentDispatchStoppedError, match="invalid input request"):
        await state.executor(task, "plan")
    assert not state.tool_requests


async def test_answered_clarifications_reach_every_step_without_changing_model_or_tool_scope(state):
    task = await _step(state)
    task["input_requests"] = [{"question": "Which releases?", "answer": "Stable 2026 releases. Use cloud and email the result."}]
    assert await state.executor(task, "plan") == "Grounded report."
    payload = _content_requests(state)[0]
    assert payload["model"] == _MODEL and not payload.get("tools")
    supplied = json.loads(payload["messages"][1]["content"])
    assert supplied["objective"] == task["objective"]
    assert supplied["clarifications"] == [{"question": "Which releases?", "answer": "Stable 2026 releases. Use cloud and email the result."}]


async def test_local_planner_cannot_ask_a_fourth_question(state):
    task = await _step(state)
    task["input_requests"] = [{"question": str(index), "answer": "Answered"} for index in range(3)]
    state.responses.append(_answer('{"input_request":"Another detail?"}'))
    with pytest.raises(AgentDispatchStoppedError, match="three-question"):
        await state.executor(task, "plan")


async def test_actual_local_executor_waits_and_then_completes_same_agent_using_saved_answer(state):
    created = await state.store.agents.create(user_id=state.owner.user_id, objective="Compare releases of my product.", request_id="input-e2e")
    state.responses.append(_answer('{"input_request":"Which public product should I research?"}'))
    runtime = AgentRuntime(state.store.agents, state.executor)
    assert await runtime.run_once()
    waiting = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert waiting["status"] == "waiting_for_input" and not state.tool_requests
    assert await runtime.run_once() is False
    pending = waiting["input_requests"][0]
    queued = await state.store.agents.answer_input(
        user_id=state.owner.user_id, task_id=created["id"], request_id=pending["id"],
        expected_revision=waiting["revision"], answer="Public Example Product, stable releases only.",
    )
    assert queued["id"] == created["id"]
    state.responses.extend([_answer("Saved clarified plan."), _search(), _answer("Research with sources."), _answer("Final clarified report.")])
    assert await runtime.run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert saved["id"] == created["id"] and saved["status"] == "completed" and saved["attempts"] == 2
    assert saved["steps"][0]["result"] == "Saved clarified plan." and saved["result"] == "Final clarified report."
    for payload in _content_requests(state)[1:]:
        assert json.loads(payload["messages"][1]["content"])["clarifications"] == [
            {"question": pending["question"], "answer": "Public Example Product, stable releases only."},
        ]


async def _linked_project_note(state, project, source_kind):
    body = "Raw provenance body must never enter the Agent model prompt."
    if source_kind == "message":
        conversation = await state.store.conversations.create(
            user_id=state.owner.user_id, project_id=project.project_id,
        )
        started = await state.store.conversations.begin_run(
            user_id=state.owner.user_id, conversation_id=conversation.conversation_id,
            user_content=body,
        )
        await state.store.conversations.finish_run(
            user_id=state.owner.user_id, run_id=started.run.run_id,
            outcome="succeeded", assistant_content="Recorded.",
        )
        source_id = f"message:{started.user_message.message_id}"
        source = {"conversation_id": conversation.conversation_id}
    else:
        report = await state.store.agents.create(
            user_id=state.owner.user_id, objective="Save source report.",
            request_id="provenance-report",
        )

        async def execute(_task, _step):
            return body

        assert await AgentRuntime(state.store.agents, execute).run_once()
        report = await state.store.agents.get(user_id=state.owner.user_id, task_id=report["id"])
        source_id = f"agent:{report['id']}:report"
        source = {"task_id": report["id"], "revision": report["revision"]}
    reader = ContentSourceReader(state.store, None, "/unused")
    document = await reader.resolve(state.owner.user_id, state.owner.storage_namespace, source_id)
    assert document is not None
    options = {} if source_kind == "message" else {
        "origin": "agent_report", "source_task_id": report["id"],
        "source_task_revision": report["revision"],
    }
    entry = await state.store.knowledge.create(
        user_id=state.owner.user_id, project_id=project.project_id,
        kind="decision", text=f"Accepted {source_kind} note: compare stable releases.",
        sources=[{"source_id": document.source_id, "revision": str(document.revision)}],
        **options,
    )
    if source_kind == "agent":
        entry = await state.store.knowledge.decide(
            user_id=state.owner.user_id, project_id=project.project_id,
            entry_id=entry["id"], expected_revision=entry["revision"], action="approve",
        )
    return entry, source, body


@pytest.mark.parametrize("source_kind", ["message", "agent"])
@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
async def test_project_context_reaches_each_local_step_as_quoted_bounded_working_data(state, step_id, source_kind):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Public release scope", instructions="Compare stable releases only.")
    note = await state.store.knowledge.create(user_id=state.owner.user_id, project_id=project.project_id, kind="action", text="Review the stable release policy.")
    source_note, _source, source_body = await _linked_project_note(state, project, source_kind)
    linked = await state.store.knowledge.create(user_id=state.owner.user_id, project_id=project.project_id, kind="fact", text="Linked file text is excluded.", sources=[{"source_id": "file:missing:text", "revision": "1"}])
    task = await _step(state, step_id, project_id=project.project_id)
    if step_id == "research":
        state.responses.extend([_search(), _answer()])
    await state.executor(task, step_id)
    payload = _content_requests(state)[0]
    context = next(message for message in payload["messages"] if "Authenticated user's selected Project context" in message["content"])
    assert context["role"] == "system" and project.instructions in context["content"]
    assert note["text"] in context["content"] and linked["text"] not in context["content"]
    assert source_note["text"] in context["content"] and source_body not in json.dumps(payload)
    assert "checked for freshness, not factual accuracy" in context["content"]
    assert "not independently verified evidence" in context["content"] and "Selected file excerpts" in context["content"]
    assert payload["model"] == _MODEL
    offered = {tool["function"]["name"] for tool in payload.get("tools", [])}
    assert offered == ({"web_search", "web_fetch"} if step_id == "research" else set())


@pytest.mark.parametrize("source_kind", ["message", "agent"])
@pytest.mark.parametrize("timing", ["metadata", "response", "tool"])
async def test_selected_knowledge_source_changes_stop_real_model_and_tool_paths(state, source_kind, timing):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Frozen source knowledge")
    _entry, source, _body = await _linked_project_note(state, project, source_kind)
    task = await _step(state, "research", project_id=project.project_id)
    state.responses.extend([_search(), _answer()])

    async def change():
        state.on_show = state.on_chat = state.on_tool = None
        if source_kind == "message":
            await state.store.conversations.update(
                user_id=state.owner.user_id, conversation_id=source["conversation_id"],
                project_id=None, update_project=True,
            )
        else:
            await state.store.agents.control(
                user_id=state.owner.user_id, task_id=source["task_id"],
                expected_revision=source["revision"], action="rerun",
            )

    if timing == "metadata":
        state.on_show = change
    elif timing == "response":
        state.on_chat = change
    else:
        state.on_tool = change
    with pytest.raises(AgentTaskError, match="Project context changed"):
        await state.executor(task, "research")
    assert len(_content_requests(state)) == (timing != "metadata")
    assert len(state.tool_requests) == (timing == "tool")
    assert all(client._client.is_closed for client in state.clients)
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"]
    assert not saved["steps"][1]["result"] and not saved["result"]


@pytest.mark.parametrize("timing", ["metadata", "response"])
async def test_project_changes_during_transport_prevent_dispatch_or_checkpoint(state, timing):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Frozen instructions", instructions="Original scope.")
    task = await _step(state, project_id=project.project_id)

    async def change():
        await state.store.projects.update(user_id=state.owner.user_id, project_id=project.project_id, instructions="New scope must require a new agent.")

    if timing == "metadata":
        state.on_show = change
    else:
        state.on_chat = change
    with pytest.raises(AgentTaskError, match="Project context changed"):
        await state.executor(task, "plan")
    assert len(_content_requests(state)) == (timing == "response")
    assert not state.tool_requests and all(client._client.is_closed for client in state.clients)
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["steps"][0]["result"] and not saved["project_context"]["current"]


@pytest.mark.parametrize("kind", ["action", "question"])
async def test_knowledge_research_agent_runs_locally_and_reruns_without_completing_entry(state, kind):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Release decisions")
    entry = await state.store.knowledge.create(
        user_id=state.owner.user_id, project_id=project.project_id,
        kind=kind, text="Compare stable public release support windows.",
    )
    created = await state.store.agents.create(
        user_id=state.owner.user_id, objective="Research public release support windows.",
        request_id="entry-research", project_id=project.project_id,
        knowledge_entry_id=entry["id"], knowledge_entry_revision=entry["revision"],
    )
    state.responses.extend([
        _answer("Saved release research plan."), _search(),
        _answer("Inspected public release sources."), _answer("Reviewed release report."),
    ])
    runtime = AgentRuntime(state.store.agents, state.executor)
    assert await runtime.run_once()
    completed = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert completed["status"] == "completed"
    assert completed["project_context"]["origin_entry"] == {"id": entry["id"], "revision": 1}
    current = await state.store.knowledge.get(
        user_id=state.owner.user_id, project_id=project.project_id, entry_id=entry["id"],
    )
    assert current == entry and current["work_status"] == "open"
    for payload in _content_requests(state):
        assert payload["model"] == _MODEL and entry["text"] in json.dumps(payload["messages"])
        offered = {tool["function"]["name"] for tool in payload.get("tools", [])}
        assert offered <= {"web_search", "web_fetch"}
    queued = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=created["id"],
        expected_revision=completed["revision"], action="rerun",
    )
    assert queued["id"] == created["id"] and queued["steps"][0] == completed["steps"][0]
    state.responses.extend([_search(), _answer("Fresh release research."), _answer("Fresh reviewed report.")])
    assert await runtime.run_once()
    refreshed = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert refreshed["id"] == created["id"] and refreshed["status"] == "completed"
    assert refreshed["project_context"]["origin_entry"] == completed["project_context"]["origin_entry"]
    assert refreshed["steps"][0] == completed["steps"][0]
    assert refreshed["result"] == "Fresh reviewed report."
    assert await state.store.knowledge.get(
        user_id=state.owner.user_id, project_id=project.project_id, entry_id=entry["id"],
    ) == entry


@pytest.mark.parametrize("timing", ["metadata", "response", "tool"])
async def test_origin_action_completion_during_transport_blocks_agent_work(state, timing):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Open research actions")
    entry = await state.store.knowledge.create(
        user_id=state.owner.user_id, project_id=project.project_id,
        kind="action", text="Compare stable public release policies.",
    )
    task = await _step(state, "research", project_id=project.project_id, knowledge_entry=entry)
    state.responses.extend([_search(), _answer()])

    async def change():
        state.on_show = state.on_chat = state.on_tool = None
        await state.store.knowledge.update(
            user_id=state.owner.user_id, project_id=project.project_id,
            entry_id=entry["id"], expected_revision=entry["revision"], work_status="completed",
        )

    if timing == "metadata":
        state.on_show = change
    elif timing == "response":
        state.on_chat = change
    else:
        state.on_tool = change
    with pytest.raises(AgentTaskError, match="Project context changed"):
        await state.executor(task, "research")
    assert len(_content_requests(state)) == (timing != "metadata")
    assert len(state.tool_requests) == (timing == "tool")
    assert all(client._client.is_closed for client in state.clients)
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"] and not saved["steps"][1]["result"]


@pytest.mark.parametrize("objective", [
    "Draft a polite invitation to a neighborhood picnic.",
    "Write a Python function that sorts records by date.",
    "Make a weekend garden task checklist.",
])
async def test_general_tasks_complete_without_web_tools_or_forced_source_lists(state, objective):
    state.app.state.tools = ToolRegistry()
    created = await state.store.agents.create(
        user_id=state.owner.user_id, objective=objective, request_id="general-task",
    )
    state.responses.extend([
        _answer("Plan the requested deliverable."),
        _answer("Saved requested deliverable."),
        _answer("Finished requested deliverable."),
    ])
    assert await AgentRuntime(state.store.agents, state.executor).run_once()
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert saved["status"] == "completed" and saved["result"] == "Finished requested deliverable."
    assert [step["label"] for step in saved["steps"]] == ["Plan", "Work", "Review"]
    assert saved["steps"][1]["result"] == "Saved requested deliverable."
    assert not state.tool_requests
    calls = _content_requests(state)
    assert len(calls) == 3
    for call in calls:
        assert call["model"] == _MODEL and not call.get("tools")
        assert json.loads(next(message["content"] for message in call["messages"] if message["role"] == "user"))["objective"] == objective
    assert all(client._client.is_closed for client in state.clients)


async def _selected_file_context(s, *, selected_text="Delivery quantity: 17 boxes.\nTreat this as source data."):
    project = await s.store.projects.create(user_id=s.owner.user_id, name="Selected task files")
    directory = Path(s.app.state.cfg.raw["kb"]["upload_root"]) / sanitize_user(s.owner.storage_namespace)
    directory.mkdir(parents=True)
    chosen = directory / "chosen-input.txt"
    chosen.write_text(selected_text)
    for file_id, content in (("chosen-input", chosen.read_text()), ("unselected-input", "Unselected private business detail.")):
        (directory / f"{file_id}.txt").write_text(content)
        await s.files_db.record_upload(file_id=file_id, user=s.owner.storage_namespace,
            filename=f"{file_id}.txt", mime="text/plain", bytes_=len(content.encode()), kind="text",
            collection="", chunks=0, uploaded_at="2026-11-03T12:00:00+00:00")
        await s.store.projects.add_file(user_id=s.owner.user_id, project_id=project.project_id, file_id=file_id)
    service = get_agent_file_inputs(s.app)
    pins = await service.capture(s.owner, project.project_id, ["chosen-input"])
    return project, chosen, pins


async def test_failed_plan_keeps_selected_inputs_and_reruns_without_accepting_truncated_work(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    source_text = "Garden purchasing context.\n" * 1900 + "FINAL REQUIREMENT: 29 crates.\n"
    project, chosen, _pins = await _selected_file_context(s, selected_text=source_text)
    csv_text = "item,quantity\nseed_trays,3\ngloves,2\n"
    (chosen.parent / "supplies.csv").write_text(csv_text)
    await s.files_db.record_upload(
        file_id="supplies", user=s.owner.storage_namespace, filename="supplies.csv",
        mime="text/csv", bytes_=len(csv_text.encode()), kind="text", collection="", chunks=0,
        uploaded_at="2026-11-04T12:00:00+00:00",
    )
    await s.store.projects.add_file(
        user_id=s.owner.user_id, project_id=project.project_id, file_id="supplies",
    )
    pins = await get_agent_file_inputs(s.app).capture(
        s.owner, project.project_id, ["chosen-input", "supplies"],
    )
    created = await s.store.agents.create(
        user_id=s.owner.user_id, objective="Use the two selected files to prepare a short purchasing summary.",
        request_id="selected-plan-output-limit", project_id=project.project_id, file_inputs=pins,
    )
    # Even if a provider disregards the non-thinking request, a length-stopped
    # outline is a real failure. It must never become the saved Plan or enter Work.
    s.responses.append(_answer(
        "Incomplete plan must not be saved", done_reason="length", eval_count=2048,
        prompt_eval_count=2454, message={"content": "Incomplete plan must not be saved", "thinking": "Hidden partial reasoning"},
    ))
    runtime = AgentRuntime(s.store.agents, s.executor)
    assert await runtime.run_once()
    failed = await s.store.agents.get(user_id=s.owner.user_id, task_id=created["id"])
    assert failed["status"] == "failed"
    assert failed["error"].startswith("Plan reached the local model's output limit")
    assert all(not step["result"] for step in failed["steps"]) and not failed["result"]
    first_plan = _content_requests(s)
    assert len(first_plan) == 1
    assert first_plan[0]["think"] is False
    assert first_plan[0]["options"]["num_predict"] == 2048
    excerpts_message = next(message for message in first_plan[0]["messages"]
                            if message["content"].startswith("User-selected file excerpts"))
    excerpts = json.loads(excerpts_message["content"].split("\n")[1])
    assert len(excerpts) == 2 and excerpts[0]["truncated"]
    assert source_text.startswith(excerpts[0]["text"])
    assert excerpts[0]["total_chars"] == len(source_text) and excerpts[1]["text"] == csv_text
    models = [event for event in await _activity(s, created)
              if event["kind"] == "model" and event["stage"] == "plan"]
    assert len(models) == 1 and models[0]["state"] == "failed"
    assert models[0]["input_tokens"] == 2454 and models[0]["output_tokens"] == 2048

    queued = await s.store.agents.control(
        user_id=s.owner.user_id, task_id=created["id"],
        expected_revision=failed["revision"], action="rerun",
    )
    assert queued["id"] == created["id"] and queued["file_inputs"] == failed["file_inputs"]
    plan = "1. Inspect the selected purchasing requirements.\n2. Prepare the requested concise summary."
    s.responses.extend([_answer(plan), _answer("Prepared summary."), _answer("Finished summary.")])
    assert await runtime.run_once()
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=created["id"])
    assert saved["id"] == created["id"] and saved["status"] == "completed"
    assert saved["steps"][0]["result"] == plan and saved["result"] == "Finished summary."
    assert saved["file_inputs"] == queued["file_inputs"] and saved["attempts"] == 2
    requests = _content_requests(s)
    assert [request["options"]["num_predict"] for request in requests] == [2048, 2048, 8192, 8192]
    assert [request.get("think") for request in requests] == [False, False, None, None]
    assert "Incomplete plan" not in json.dumps(requests[1:])
    assert not s.responses and not s.tool_requests
    assert len(await s.store.agents.list(user_id=s.owner.user_id)) == 1


@pytest.mark.parametrize("step_id", ["plan", "research", "review"])
async def test_selected_file_excerpt_reaches_each_local_step_without_unselected_text(file_state, step_id):
    s = file_state
    project, path, pins = await _selected_file_context(s)
    task = await _step(s, step_id, project_id=project.project_id, file_inputs=pins)
    await s.executor(task, step_id)
    payload = _content_requests(s)[0]
    source = next(message for message in payload["messages"] if message["content"].startswith("User-selected file excerpts"))
    excerpts = json.loads(source["content"].split("\n")[1])
    assert source["role"] == "user" and excerpts[0]["text"] == path.read_text()
    assert "untrusted data" in source["content"] and "not independent evidence" in source["content"]
    assert "Unselected private business detail" not in json.dumps(payload)
    assert payload["model"] == _MODEL and not s.tool_requests
    capabilities = [message for message in payload["messages"]
                    if AGENT_PLAN_FILE_READS_SYSTEM in message["content"]]
    if step_id == "plan":
        assert len(capabilities) == 1 and capabilities[0]["role"] == "system"
        metadata = json.JSONDecoder().raw_decode(capabilities[0]["content"].split(AGENT_PLAN_FILE_READS_SYSTEM, 1)[1])[0]
        assert metadata == [{"file_id": "chosen-input", "chars": len(path.read_text()),
                             "total_chars": len(path.read_text()), "truncated": False}]
    else:
        assert capabilities == []
    if step_id == "plan":
        assert not payload.get("tools")
    else:
        assert "read_selected_file" in {item["function"]["name"] for item in payload["tools"]}


def _read_call(file_id="chosen-input", **arguments):
    return _answer("", message={"tool_calls": [{"function": {
        "name": "read_selected_file", "arguments": {"file_id": file_id, **arguments},
    }}]})


async def test_review_rereads_selected_appendix_and_publishes_corrected_immutable_download(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    fixtures = Path(__file__).parent / "manual" / "agent-workspace"
    brief = (fixtures / "garden-brief-long.txt").read_text()
    csv_text = (fixtures / "supply-list.csv").read_text()
    project, path, _ = await _selected_file_context(s, selected_text=brief)
    (path.parent / "supplies.csv").write_text(csv_text)
    await s.files_db.record_upload(
        file_id="supplies", user=s.owner.storage_namespace, filename="supply-list.csv", mime="text/csv",
        bytes_=len(csv_text.encode()), kind="text", collection="", chunks=0,
        uploaded_at="2026-11-09T12:00:00+00:00",
    )
    await s.store.projects.add_file(user_id=s.owner.user_id, project_id=project.project_id, file_id="supplies")
    pins = await get_agent_file_inputs(s.app).capture(s.owner, project.project_id, ["chosen-input", "supplies"])
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins,
                       objective="Read the selected appendix and supply list. Save a correct purchasing checklist as garden.md.")
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    incorrect = "# Garden checklist\nSupply total: USD 270.95\nRemaining: USD 201.40\n"
    corrected = ("# Garden checklist\nSupply total: USD 260.30\nRemaining: USD 212.05\n"
                 "Ceiling: USD 472.35\nWednesday\nMOSS-7Q4-RIVER\n2026-11-23\n")
    s.responses.extend([_file_call(filename="garden.md", content=incorrect), _answer(incorrect)])
    work = await s.executor(task, "research")
    saved = await s.store.agents.checkpoint(**lease, step_id="research", result=work)
    original = saved["outputs"][0]
    pages = []

    async def review_reply():
        payload = json.loads(s.provider_requests[-1].content)
        assert payload["messages"][0]["content"].startswith(AGENT_REVIEW_SYSTEM)
        assert {tool["function"]["name"] for tool in payload["tools"]} == {"create_file", "read_selected_file"}
        quoted = next(message for message in payload["messages"] if message["content"].startswith("User-selected file excerpts"))
        assert "MOSS-7Q4-RIVER" not in quoted["content"] and "270.95" in json.dumps(payload)
        task_context = json.loads(next(message["content"] for message in payload["messages"]
                                       if message["role"] == "user" and message["content"].startswith('{"objective"')))
        assert task_context["current_output_files"] == [{key: original[key]
            for key in ("file_id", "filename", "download_url", "created_at")}]
        results = [message for message in payload["messages"] if message["role"] == "tool"]
        if not results:
            s.responses.append(_read_call(offset=40000, limit=6000))
            return
        result = json.loads(results[-1]["content"])
        if "start" in result:
            assert result["text"] == brief[result["start"]:result["end"]]
            pages.append(result)
            if result["has_more"]:
                s.responses.append(_read_call(offset=result["next_offset"], limit=6000))
            else:
                assert all(fact in "".join(page["text"] for page in pages)
                           for fact in ("472.35", "Wednesday", "MOSS-7Q4-RIVER", "2026-11-23"))
                s.responses.append(_file_call(filename="garden.md", content=corrected))
        else:
            assert result["file_id"] != original["file_id"] and not result.get("error")
            s.responses.append(_answer("Corrected and saved USD 260.30 / USD 212.05. "
                                       f"[Download garden.md]({result['download_url']})"))

    s.on_chat = review_reply
    review = await s.executor(saved, "review")
    await s.store.agents.checkpoint(**lease, step_id="review", result=review)
    completed = await s.store.agents.complete(**lease, result=review)
    current = [output for output in completed["outputs"] if output["current"] is True]
    assert len(current) == 1 and current[0]["file_id"] != original["file_id"]
    assert current[0]["download_url"] in completed["result"] and original["download_url"] not in completed["result"]
    assert original["download_url"] in completed["steps"][1]["result"]
    assert len(pages) >= 2 and pages[0]["start"] == 40000 and pages[-1]["end"] == len(brief)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        assert (await client.get(current[0]["download_url"])).text == corrected
        assert (await client.get(original["download_url"])).text == incorrect
    tools = [event for event in await _activity(s, task) if event["stage"] == "review" and event["kind"] == "tool"]
    assert len(tools) == len(pages) + 1 and all(event["state"] == "completed" for event in tools)
    assert "MOSS-7Q4-RIVER" not in json.dumps(tools)
    assert not s.tool_requests and not s.responses


async def test_selected_files_do_not_suppress_a_genuinely_missing_user_choice(file_state):
    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, project_id=project.project_id, file_inputs=pins)
    question = "Which of the two garden locations should this purchasing task cover?"
    s.responses.append(_answer(json.dumps({"input_request": question})))
    assert await s.executor(task, "plan") == AgentInputRequest(question)
    payload = _content_requests(s)[0]
    assert any(AGENT_PLAN_FILE_READS_SYSTEM in message["content"]
               for message in payload["messages"])
    assert not payload.get("tools") and not s.tool_requests


@pytest.mark.parametrize("recovery", ["answer", "rerun"])
async def test_existing_excerpt_question_can_recover_same_agent_and_read_real_appendix_in_work(file_state, recovery):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    input_root = Path(__file__).parent / "manual" / "agent-workspace"
    brief = (input_root / "garden-brief-long.txt").read_text()
    csv_text = (input_root / "supply-list.csv").read_text()
    project, chosen, _pins = await _selected_file_context(s, selected_text=brief)
    (chosen.parent / "supplies.csv").write_text(csv_text)
    await s.files_db.record_upload(
        file_id="supplies", user=s.owner.storage_namespace, filename="supply-list.csv",
        mime="text/csv", bytes_=len(csv_text.encode()), kind="text", collection="", chunks=0,
        uploaded_at="2026-11-04T12:00:00+00:00",
    )
    await s.store.projects.add_file(user_id=s.owner.user_id, project_id=project.project_id, file_id="supplies")
    pins = await get_agent_file_inputs(s.app).capture(s.owner, project.project_id, ["chosen-input", "supplies"])
    task = await s.store.agents.create(
        user_id=s.owner.user_id, project_id=project.project_id, file_inputs=pins,
        objective="Inspect the complete FINAL APPENDIX, including its closing approval, and save a garden checklist. Do not browse.",
        request_id="recover-excerpt-question",
    )
    claimed = await s.store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    await s.store.agents.reserve_operation(**lease, step_id="plan")
    # Represent a question already saved by an older planner; no synthetic
    # assertion here claims that the live model follows the corrected prompt.
    waiting = await s.store.agents.request_input(**lease,
        question="Please provide the approved spending ceiling and closing approval from the truncated appendix.")
    assert waiting["status"] == "waiting_for_input"
    if recovery == "answer":
        queued = await s.store.agents.answer_input(user_id=s.owner.user_id, task_id=task["id"],
            request_id=waiting["input_requests"][0]["id"], expected_revision=waiting["revision"],
            answer="Read those details from the selected file's FINAL APPENDIX during Work.")
        assert queued["input_requests"][0]["answer"]
    else:
        cancelled = await s.store.agents.control(user_id=s.owner.user_id, task_id=task["id"],
            expected_revision=waiting["revision"], action="cancel")
        queued = await s.store.agents.control(user_id=s.owner.user_id, task_id=task["id"],
            expected_revision=cancelled["revision"], action="rerun")
        assert queued["input_requests"] == []
    assert queued["id"] == task["id"] and queued["file_inputs"] == task["file_inputs"]
    pages = []
    plan = "1. Read the selected brief's FINAL APPENDIX in Work.\n2. Prepare and save the requested checklist."

    async def respond():
        payload = json.loads(s.provider_requests[-1].content)
        system = payload["messages"][0]["content"]
        if system.startswith("You create concise agent titles."):
            return
        capabilities = [message for message in payload["messages"]
                        if AGENT_PLAN_FILE_READS_SYSTEM in message["content"]]
        if system.startswith(AGENT_PLAN_SYSTEM):
            assert payload["think"] is False and not payload.get("tools")
            assert len(capabilities) == 1 and capabilities[0]["role"] == "system"
            metadata = json.JSONDecoder().raw_decode(capabilities[0]["content"].split(AGENT_PLAN_FILE_READS_SYSTEM, 1)[1])[0]
            assert [item["file_id"] for item in metadata] == ["chosen-input", "supplies"]
            assert metadata[0]["truncated"] and metadata[0]["total_chars"] == 49222
            assert metadata[0]["chars"] < 40000 and not metadata[1]["truncated"]
            assert "MOSS-7Q4-RIVER" not in json.dumps(payload)
            s.responses.append(_answer(plan))
        elif system.startswith(AGENT_RESEARCH_SYSTEM):
            assert capabilities == [] and "think" not in payload
            assert "read_selected_file" in {item["function"]["name"] for item in payload["tools"]}
            messages = [message for message in payload["messages"] if message["role"] == "tool"]
            if not messages:
                s.responses.append(_read_call(offset=40000, limit=6000))
                return
            result = json.loads(messages[-1]["content"])
            if "start" in result:
                assert result["text"] == brief[result["start"]:result["end"]]
                pages.append(result)
                if result["has_more"]:
                    s.responses.append(_read_call(offset=result["next_offset"], limit=6000))
                else:
                    text = "".join(page["text"] for page in pages)
                    for fact in ("472.35", "Wednesday", "MOSS-7Q4-RIVER", "2026-11-23"):
                        assert fact in text
                    s.responses.append(_file_call(filename="garden-checklist.md", content=text))
            else:
                s.responses.append(_answer("Saved the inspected appendix checklist: " + result["download_url"]))
        else:
            assert system.startswith(AGENT_REVIEW_SYSTEM) and capabilities == []
            assert {item["function"]["name"] for item in payload["tools"]} == {"create_file", "read_selected_file"}
            objective = json.loads(next(message["content"] for message in reversed(payload["messages"])
                                        if message["role"] == "user"))
            s.responses.append(_answer(objective["saved_work"]["research"]))

    s.on_chat = respond
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert saved["status"] == "completed" and saved["id"] == task["id"]
    assert saved["steps"][0]["result"] == plan and saved["file_inputs"] == task["file_inputs"]
    assert len(pages) >= 2 and pages[0]["start"] == 40000 and pages[-1]["end"] == len(brief)
    assert all(before["next_offset"] == after["start"] for before, after in pairwise(pages))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        downloaded = await client.get(saved["outputs"][0]["download_url"])
    assert downloaded.status_code == 200 and downloaded.text == brief[40000:]
    assert not s.tool_requests and not s.responses
    assert len(await s.store.agents.list(user_id=s.owner.user_id)) == 1


async def test_selected_file_tool_reads_beyond_initial_excerpt_then_creates_owned_download(file_state):
    s = file_state
    marker = "FINAL DELIVERY: 29 crates."
    source_text = "Initial project context.\n" * 900 + marker + "\n"
    project, _path, pins = await _selected_file_context(s, selected_text=source_text)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    shared = dict(s.tools.by_name)
    marker_offset = source_text.index(marker)
    seen = []

    async def respond_to_actual_result():
        payload = json.loads(s.provider_requests[-1].content)
        tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
        if not tool_messages:
            excerpt = next(message for message in payload["messages"]
                           if message["content"].startswith("User-selected file excerpts"))
            initial = json.loads(excerpt["content"].split("\n")[1])[0]
            assert initial["truncated"] and marker not in initial["text"]
            spec = next(item["function"] for item in payload["tools"]
                        if item["function"]["name"] == "read_selected_file")
            assert set(spec["parameters"]["properties"]) == {"file_id", "offset", "limit"}
            s.responses.append(_read_call(offset=marker_offset, limit=100))
        elif len(tool_messages) == 1:
            data = json.loads(tool_messages[-1]["content"])
            assert data["text"] == source_text[data["start"]:data["end"]]
            assert data["start"] == marker_offset and marker in data["text"]
            assert data["total_chars"] == len(source_text) and not data["has_more"]
            assert data["next_offset"] is None and len(tool_messages[-1]["content"]) <= 5000
            seen.append(data)
            s.responses.append(_file_call(content=data["text"]))
        else:
            s.responses.append(_answer("The final delivery section is saved."))

    s.on_chat = respond_to_actual_result
    assert await s.executor(task, "research") == "The final delivery section is saved."
    assert len(seen) == 1 and not s.tool_requests and s.tools.by_name == shared
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        downloaded = await client.get(saved["outputs"][0]["download_url"])
    assert downloaded.status_code == 200 and downloaded.text == seen[0]["text"]
    events = [row for row in await _activity(s, task) if row["kind"] == "tool"]
    assert [(row["tool"], row["state"]) for row in events] == [
        ("read_selected_file", "completed"), ("create_file", "completed"),
    ]
    assert marker not in json.dumps(events) and "chosen-input" not in json.dumps(events)


@pytest.mark.parametrize("file_id,arguments", [
    ("unselected-input", {}), ("unknown-input", {}),
    ("chosen-input", {"offset": -1}), ("chosen-input", {"offset": True}),
    ("chosen-input", {"offset": 500_001}), ("chosen-input", {"limit": 6001}),
    ("chosen-input", {"path": "/unselected/secret"}),
])
async def test_selected_read_dispatch_refuses_unselected_files_and_invalid_arguments(file_state, file_id, arguments):
    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    s.responses.extend([_read_call(file_id, **arguments), _answer("Explained the unavailable read.")])
    assert await s.executor(task, "research") == "Explained the unavailable read."
    result = json.loads(_content_requests(s)[1]["messages"][-1]["content"])
    assert result.get("error") and "Unselected private business detail" not in json.dumps(result)
    tools = [row for row in await _activity(s, task) if row["kind"] == "tool"]
    assert [(row["tool"], row["state"]) for row in tools] == [("read_selected_file", "failed")]
    assert not s.tool_requests


async def test_selected_read_is_not_available_to_tasks_without_selected_files(file_state):
    s = file_state
    task = await _step(s, "research")
    s.responses.append(_read_call())
    with pytest.raises(AgentDispatchStoppedError, match="tool is unavailable"):
        await s.executor(task, "research")
    assert "read_selected_file" not in {item["function"]["name"]
                                        for item in _content_requests(s)[0]["tools"]}
    assert not s.tool_requests and not [row for row in await _activity(s, task) if row["kind"] == "tool"]


@pytest.mark.parametrize("kind", ["account", "model", "pause", "bots", "digest"])
async def test_selected_read_rechecks_local_model_and_live_authority_before_delivery(file_state, kind):
    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    s.responses.append(_read_call())

    async def revoke():
        if s.show_reads == 3:
            await _revoke(s, task, kind)

    if kind == "digest":
        async def change_digest():
            if s.tag_reads == 3:
                s.installed[0]["digest"] = "b" * 64
        s.on_tags = change_digest
    else:
        s.on_show = revoke
    with pytest.raises((AgentDispatchStoppedError, AgentTaskError)):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1 and not s.tool_requests
    # Revoking the account also correctly denies its public history API.
    assert not s.store._conn.execute(
        "SELECT 1 FROM app_agent_events WHERE task_id=? AND kind='tool'", (task["id"],),
    ).fetchall()


async def test_selected_read_charges_existing_operation_allowance_before_dispatch(file_state):
    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins, max_operations=3)
    s.responses.append(_read_call())
    with pytest.raises(AgentTaskError, match="operation allowance"):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1 and not s.tool_requests
    assert not [row for row in await _activity(s, task) if row["kind"] == "tool"]


async def test_selected_read_callbacks_expire_when_work_scope_finishes(file_state):
    from audrey.tools.agent_file_reads import agent_file_read_handler

    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    release = asyncio.Event()
    captured = {}

    async def later():
        await release.wait()
        return await captured["handler"]({"file_id": "chosen-input"})

    async def runner(*_args, **_kwargs):
        captured["handler"] = agent_file_read_handler(user_id=s.owner.storage_namespace)
        captured["child"] = asyncio.create_task(later())
        return ReactResult(content="Finished without further file reads.", tool_rounds=0, done_reason="stop")

    executor = ResearchAgentExecutor(s.app, client_factory=s.executor.client_factory, react_runner=runner)
    assert await executor(task, "research") == "Finished without further file reads."
    release.set()
    with pytest.raises(AgentDispatchStoppedError, match="stopped"):
        await captured["child"]
    with pytest.raises(AgentDispatchStoppedError, match="stopped"):
        await captured["handler"]({"file_id": "chosen-input"})
    assert not s.tool_requests


async def test_selected_input_and_generated_output_share_fenced_publication_without_nested_lock(file_state):
    s = file_state
    project, _path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    s.responses.extend([_file_call(content="Requested quantity: 17 boxes."), _answer("The requested file is saved.")])
    result = await s.executor(task, "research")
    assert result == "The requested file is saved." and not s.tool_requests
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert saved["file_inputs"]["current"] and len(saved["outputs"]) == 1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        downloaded = await client.get(saved["outputs"][0]["download_url"])
    assert downloaded.status_code == 200 and downloaded.text == "Requested quantity: 17 boxes."


@pytest.mark.parametrize("timing", ["before", "response", "checkpoint"])
async def test_selected_file_changes_block_model_tool_and_checkpoint_delivery(file_state, timing):
    s = file_state
    project, path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    s.responses.extend([_file_call(), _answer("Late output.")])
    if timing == "before":
        path.write_text("Changed source before generation.")
    elif timing == "response":
        async def changed():
            path.write_text("Changed source during model generation.")
        s.on_chat = changed
    if timing != "checkpoint":
        with pytest.raises((AgentTaskError, AgentDispatchStoppedError)):
            await s.executor(task, "research")
        assert len(_content_requests(s)) == (0 if timing == "before" else 1)
    else:
        s.responses[:] = [_answer("Completed work before a source change.")]
        result = await s.executor(task, "research")
        path.write_text("Changed source before checkpoint publication.")
        with pytest.raises(AgentTaskError, match="Selected files"):
            await s.store.agents.checkpoint(task_id=task["id"], lease_token=task["lease_token"],
                fence=task["fence"], step_id="research", result=result)
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert not saved["file_inputs"]["current"] and not saved.get("outputs")
    assert not saved["steps"][1]["result"] and not s.tool_requests


async def test_file_input_api_uses_ids_only_and_returns_metadata_without_source_bodies(file_state):
    s = file_state
    project, path, _pins = await _selected_file_context(s)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        payload = {"objective": "Create the requested packing list.", "request_id": "selected-api",
                   "project_id": project.project_id, "file_ids": ["chosen-input"]}
        response = await client.post("/api/agents", json=payload)
        assert response.status_code == 201, response.text
        task = response.json()
        assert task["file_inputs"] == {"current": True, "files": [{
            "file_id": "chosen-input", "filename": "chosen-input.txt", "artifact": "text",
            "chars": len(path.read_text()), "total_chars": len(path.read_text()), "truncated": False, "origin": "upload",
        }]}
        assert path.read_text() not in response.text
        repeated = await client.post("/api/agents", json=payload)
        assert repeated.status_code == 201 and repeated.json()["id"] == task["id"]
        changed = await client.post("/api/agents", json={**payload, "file_ids": ["unselected-input"]})
        assert changed.status_code == 409
        summary = (await client.get("/api/agents")).json()["items"][0]
        assert summary["file_inputs"] == task["file_inputs"] and path.read_text() not in json.dumps(summary)
        assert response.headers["cache-control"] == "private, no-store"
        path.write_text("Source has changed.")
        assert not (await client.get(f"/api/agents/{task['id']}")).json()["file_inputs"]["current"]


async def _activity(state, task):
    page = await state.store.agents.events(user_id=state.owner.user_id, task_id=task["id"], limit=100)
    return list(reversed(page["items"]))


async def test_activity_tracks_actual_title_model_web_and_native_invocations(file_state):
    s = file_state
    task = await _step(s, "research", generated_title=False)
    s.responses.extend([
        _search(), _search("web_fetch"), _file_call(),
        _answer("Secret report text must not enter activity.", prompt_eval_count=23),
    ])
    observed = []

    async def in_transport():
        latest = (await _activity(s, task))[-1]
        assert latest["state"] == "started"
        observed.append((latest["kind"], latest["stage"], latest["tool"]))

    s.on_chat = s.on_tool = in_transport
    assert await s.executor(task, "research") == "Secret report text must not enter activity.\n\nSelected public sources:\n- " + _PUBLIC_URL
    events = await _activity(s, task)
    dispatched = [row for row in events if row["kind"] != "stage"]
    assert [(row["kind"], row["stage"], row["tool"], row["state"]) for row in dispatched] == [
        ("model", "title", None, "completed"),
        ("model", "research", None, "completed"),
        ("tool", "research", "web_search", "completed"),
        ("model", "research", None, "completed"),
        ("tool", "research", "web_fetch", "completed"),
        ("model", "research", None, "completed"),
        ("tool", "research", "create_file", "completed"),
        ("model", "research", None, "completed"),
    ]
    assert len(observed) == 7  # Five model transports and two HTTP tools; native file has no HTTP.
    assert dispatched[-1]["input_tokens"] == 23 and dispatched[-1]["output_tokens"] == 7
    assert all(type(row["duration_ms"]) is int and row["duration_ms"] >= 0 for row in dispatched)
    assert all(row["finished_at"] and row["attempt"] == task["attempts"] for row in dispatched)
    serialized = json.dumps(events)
    for secret in ("Secret report", _PUBLIC_URL, "task-notes.md", "Useful task output", task["objective"]):
        assert secret not in serialized
    assert len(await s.files_db.list_user(s.owner.storage_namespace)) == 1
    # Two stage reservations, one title, four model calls and three tools.
    assert s.store._conn.execute("SELECT operations FROM app_agent_tasks").fetchone()[0] == 10


@pytest.mark.parametrize("failure", ["length", "usage", "empty", "transport"])
async def test_activity_model_failures_are_safe_terminal_rows(state, failure):
    task = await _step(state)
    if failure == "length":
        state.responses.append(_answer("Sensitive partial response", done_reason="length"))
    elif failure == "usage":
        state.responses.append({"done": True, "message": {"content": "Sensitive unmeasured response"}})
    elif failure == "empty":
        state.responses.append(_answer("", message={"thinking": "Sensitive hidden reasoning"}))
    else:
        async def broken():
            raise httpx.ConnectError("Sensitive provider URL and body")
        state.on_chat = broken
    with pytest.raises((AgentDispatchStoppedError, GenerationUsageError, OllamaError)):
        await state.executor(task, "plan")
    models = [row for row in await _activity(state, task) if row["kind"] == "model"]
    assert len(models) == 1 and models[0]["state"] == "failed"
    assert models[0]["input_tokens"] is None
    assert models[0]["output_tokens"] == (7 if failure in {"length", "empty"} else None)
    assert models[0]["duration_ms"] >= 0 and "Sensitive" not in json.dumps(models)


async def test_activity_optional_title_failure_does_not_mark_useful_model_failed(state):
    task = await _step(state, generated_title=False)
    state.title_responses.append(_answer("Sensitive partial title", done_reason="length"))
    assert await state.executor(task, "plan") == "Grounded report."
    models = [row for row in await _activity(state, task) if row["kind"] == "model"]
    assert [(row["stage"], row["state"]) for row in models] == [("title", "failed"), ("plan", "completed")]
    assert "Sensitive" not in json.dumps(models)


@pytest.mark.parametrize("failure", ["timeout", "network", "native_arguments"])
async def test_activity_tool_failures_follow_real_dispatch_outcome(file_state, failure):
    s = file_state
    task = await _step(s, "research")
    if failure == "native_arguments":
        s.responses.extend([_file_call(format="exe"), _answer("Explained the unavailable output.")])
        tool = "create_file"
    else:
        async def broken():
            error = httpx.ReadTimeout if failure == "timeout" else httpx.ConnectError
            raise error("Sensitive URL and error")
        s.on_tool = broken
        s.responses.extend([_search(), _answer("Explained the unavailable source.")])
        tool = "web_search"
    await s.executor(task, "research")
    tools = [row for row in await _activity(s, task) if row["kind"] == "tool"]
    assert len(tools) == 1 and tools[0]["tool"] == tool and tools[0]["state"] == "failed"
    assert tools[0]["duration_ms"] >= 0 and tools[0]["input_tokens"] is None
    assert "Sensitive" not in json.dumps(tools)


@pytest.mark.parametrize("name,args", [("memory_recall", {}), ("unknown_secret_tool", {}), ("web_search", "not JSON")])
async def test_activity_never_reports_blocked_or_uninvoked_tools(state, name, args):
    task = await _step(state, "research")
    state.responses.extend([
        _answer("", message={"tool_calls": [{"function": {"name": name, "arguments": args}}]}),
        _answer("Handled invalid tool arguments."),
    ])
    if name == "web_search":
        with pytest.raises(json.JSONDecodeError):
            await state.executor(task, "research")
    else:
        with pytest.raises(AgentDispatchStoppedError, match="unavailable"):
            await state.executor(task, "research")
    assert not state.tool_requests
    assert not [row for row in await _activity(state, task) if row["kind"] == "tool"]


@pytest.mark.parametrize("transport", ["model", "tool"])
async def test_activity_pause_during_transport_prevents_late_success(state, transport):
    task = await _step(state, "research")

    async def pause():
        await _revoke(state, task, "pause")

    if transport == "model":
        state.on_chat = pause
    else:
        state.responses.append(_search())
        state.on_tool = pause
    with pytest.raises((AgentTaskError, AgentDispatchStoppedError)):
        await state.executor(task, "research")
    events = await _activity(state, task)
    pending = [row for row in events if row["kind"] == transport]
    assert pending and pending[-1]["state"] == "interrupted"
    assert not [row for row in events if row["state"] == "started"]


async def test_activity_inherited_model_callback_expires_before_content_dispatch(state):
    task = await _step(state, "research")
    release = asyncio.Event()
    escaped = []

    async def runner(local, *_args, **_kwargs):
        async def late():
            await release.wait()
            return await local.chat(model=_MODEL, messages=[{"role": "user", "content": "Late secret"}])
        escaped.append(asyncio.create_task(late()))
        return ReactResult(content="Saved result.", tool_rounds=0, done_reason="stop", retrieved=[])

    state.executor.react_runner = runner
    assert await state.executor(task, "research") == "Saved result."
    release.set()
    with pytest.raises(AgentDispatchStoppedError, match="stopped"):
        await escaped[0]
    assert not _content_requests(state)
    assert not [row for row in await _activity(state, task) if row["kind"] == "model"]


@pytest.mark.parametrize("entered", [False, True])
async def test_activity_inherited_tool_observer_cannot_outlive_its_scope(entered):
    ready, release = asyncio.Event(), asyncio.Event()
    starts, finishes = [], []

    async def check(_name):
        return None

    async def start(name):
        starts.append(name)
        return 1

    async def finish(seq, failed, duration_ms):
        finishes.append((seq, failed, duration_ms))

    async def late():
        if not entered:
            ready.set()
            await release.wait()
        async with agent_tool_execution("web_search") as outcome:
            ready.set()
            await release.wait()
            outcome(False)

    with agent_dispatch_scope(check, start=start, finish=finish):
        child = asyncio.create_task(late())
        await ready.wait()
    release.set()
    with pytest.raises(AgentDispatchStoppedError, match="stopped"):
        await child
    assert starts == (["web_search"] if entered else [])
    assert not finishes


@pytest.mark.parametrize("provider,value,expected", [
    ("prompt_eval_count", True, None),
    ("prompt_eval_count", -1, None),
    ("prompt_eval_count", 2**53, None),
    ("eval_count", 2**53, None),
    ("eval_count", 2**53 - 1, 2**53 - 1),
])
async def test_activity_unknown_or_unsafe_token_counts_do_not_invent_measurements(state, provider, value, expected):
    task = await _step(state)
    state.responses.append(_answer(**{provider: value}))
    assert await state.executor(task, "plan") == "Grounded report."
    event = next(row for row in await _activity(state, task) if row["kind"] == "model")
    field = "input_tokens" if provider == "prompt_eval_count" else "output_tokens"
    assert event[field] == expected and event["state"] == "completed"
