"""Saved agent authority is enforced around real local-model/tool transports."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from audrey.agent_dispatch import (
    AgentDispatchStoppedError,
    agent_dispatch_scope,
    authorize_agent_tool,
)
from audrey.agent_executor import AgentSettings, ResearchAgentExecutor
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskError
from audrey.models.health import HealthTracker
from audrey.models.ollama import GenerationUsageError, OllamaClient
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.react import ReactResult
from audrey.routes.inflight import UserInflightRegistry
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
        if value.on_chat:
            await value.on_chat()
        payload = json.loads(request.content)
        if payload["messages"][0]["content"].startswith("You create concise agent titles."):
            if value.on_title:
                await value.on_title()
            return httpx.Response(200, json=value.title_responses.pop(0) if value.title_responses else _answer("Public Release Notes Research"))
        return httpx.Response(200, json=value.responses.pop(0) if value.responses else _answer())

    def tool(request):
        value.tool_requests.append(request)
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


async def _step(state, step_id="plan", *, max_operations=96, generated_title=True):
    task = await state.store.agents.create(
        user_id=state.owner.user_id,
        objective="Compare the public release notes.",
        request_id=f"request-{step_id}",
        limits={"max_operations": max_operations},
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
        offered = {tool["function"]["name"] for tool in payload.get("tools", [])}
        assert offered == ({"web_search", "web_fetch"} if step_id == "research" else set())
    supplied = json.loads(
        next(row["content"] for row in requests[0]["messages"] if row["role"] == "user")
    )
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
        state.tools.by_name["web_fetch"].available = False
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
@pytest.mark.parametrize("kind", ["account", "model", "tool", "pause", "bots"])
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
    with pytest.raises(AgentDispatchStoppedError, match=f"{step_id.capitalize()} {detail}"):
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
        AgentDispatchStoppedError, match="Research reached the local model's output limit"
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
    assert failed["error"].startswith("Research reached the local model's output limit")
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
async def test_research_requires_usable_public_source_evidence(state, url):
    task = await _step(state, "research")

    async def report(client, _health, _tools, **kwargs):
        answer = await client.chat(model=kwargs["model"], messages=kwargs["messages"])
        return ReactResult(
            content=answer["message"]["content"], tool_rounds=0, retrieved=[{"url": url}]
        )

    state.executor.react_runner = report
    with pytest.raises(AgentDispatchStoppedError, match="no public source evidence"):
        await state.executor(task, "research")


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


async def test_title_timeout_is_optional_and_keeps_its_uncertain_operation_charge(state, monkeypatch):
    task = await _step(state, generated_title=False)
    monkeypatch.setattr("audrey.agent_executor._TITLE_TIMEOUT_SECONDS", 0.01)

    async def slow_title():
        await asyncio.sleep(1)

    state.on_title = slow_title
    assert await state.executor(task, "plan") == "Grounded report."
    row = state.store._conn.execute("SELECT title,title_attempted,operations FROM app_agent_tasks").fetchone()
    assert tuple(row) == ("", 1, 3)
    assert len(_content_requests(state, include_titles=True)) == 2


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
