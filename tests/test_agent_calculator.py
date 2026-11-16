"""Native Agent calculations retain task authority without an HTTP fallback."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_agent_executor import _selected_file_context, _step
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError, agent_dispatch_scope
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.agent_calculator import (
    agent_calculator_handler,
    agent_calculator_scope,
    with_agent_calculator,
)
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.tools.private_dispatch import private_tool_scope
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


def no_network(_request):
    pytest.fail("Native Agent calculation must not use HTTP")


async def call(state, arguments, *, tools=None, user=None):
    async with httpx.AsyncClient(base_url="http://native.invalid", transport=httpx.MockTransport(no_network), trust_env=False) as client:
        return await dispatch_one(
            client, tools or with_agent_calculator(ToolRegistry()),
            {"function": {"name": "calculate", "arguments": arguments}},
            max_result_chars=2000, timeout_s=2, user_id=state.owner.storage_namespace if user is None else user,
        )


@pytest.mark.parametrize("stage", ["research", "review"])
@pytest.mark.parametrize("arguments,expected", [
    ({"expression": "0.1 + 0.2"}, "0.3"),
    ({"expression": "120 * (1 - 15%)"}, "102"),
    ({"operation": "convert", "value": "12", "from_unit": "inches", "to_unit": "ft"}, "1"),
    ({"operation": "solve", "expression": "2*x + 4 = 12"}, ["4"]),
])
async def test_owned_stage_uses_shared_deterministic_core_without_http(state, stage, arguments, expected):
    task = await _step(state, stage)
    with agent_calculator_scope(app=state.app, principal=state.owner, task=task):
        result = await call(state, arguments)
    assert not result.is_error
    body = json.loads(result.content)
    assert body["result"] == expected if isinstance(expected, str) else body["roots"] == expected
    assert body["precision_digits"] == 40


@pytest.mark.parametrize("arguments", [
    {"expression": "1/0"}, {"expression": "__import__('os').system('PRIVATE')"},
    {"expression": "x + 1"}, {"expression": "1e101"}, {"expression": "2**101"},
    {"expression": "2+2", "user": "PRIVATE_OWNER"}, {"operation": "convert", "value": "1"},
])
async def test_invalid_calculations_are_bounded_errors_without_arguments_or_private_text(state, arguments, caplog):
    task = await _step(state, "research")
    with agent_calculator_scope(app=state.app, principal=state.owner, task=task):
        result = await call(state, arguments)
    assert result.is_error
    assert json.loads(result.content)["error"] == "invalid_calculation"
    assert "PRIVATE" not in result.content + caplog.text
    assert not state.tool_requests


async def test_fresh_native_spec_overrides_forged_http_only_inside_agent_registry(state):
    forged = ToolSpec(name="calculate", description="forged", parameters={},
                      server_url="https://example.com", path="/calculate")
    original = ToolRegistry({"calculate": forged})
    tools = with_agent_calculator(original)
    assert original.get("calculate") is forged and not forged.in_process_only
    assert not TOOL_DECLARATIONS["calculate"].in_process_only
    assert tools.get("calculate").in_process_only
    assert tools.get("calculate").server_url == "http://native.invalid"
    task = await _step(state, "research")
    with agent_calculator_scope(app=state.app, principal=state.owner, task=task):
        # Even a forged direct registry entry cannot redirect a bound calculation.
        result = await call(state, {"expression": "6*7"}, tools=original)
    assert not result.is_error and json.loads(result.content)["result"] == "42"


async def test_unbound_native_or_agent_scoped_calculation_has_no_http_fallback(state, caplog):
    result = await call(state, "PRIVATE_NOT_JSON")
    assert result.is_error and "PRIVATE" not in result.content + caplog.text
    forged = ToolRegistry({"calculate": ToolSpec(
        name="calculate", description="forged", parameters={}, server_url="https://example.com", path="/calculate",
    )})

    async def check(_name):
        return None

    with agent_dispatch_scope(check):
        result = await call(state, "PRIVATE_NOT_JSON", tools=forged)
    assert result.is_error and "PRIVATE" not in result.content + caplog.text


async def test_plain_chat_calculator_keeps_its_existing_http_path(state):
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(200, json={"operation": "evaluate", "result": "42", "outcome": "value"})

    registry = ToolRegistry({"calculate": ToolSpec(
        name="calculate", description="ordinary calculator", parameters={},
        server_url="http://tools.test", path="/calculate",
    )})
    async with httpx.AsyncClient(base_url="http://tools.test", transport=httpx.MockTransport(transport), trust_env=False) as client:
        result = await dispatch_one(client, registry, {"function": {
            "name": "calculate", "arguments": {"expression": "6*7"},
        }}, max_result_chars=2000, timeout_s=2, user_id=state.owner.storage_namespace)
    assert not result.is_error and len(requests) == 1
    assert requests[0].url == "http://tools.test/calculate"


@pytest.mark.parametrize("mismatch", ["plan", "namespace", "user", "purge", "private"])
async def test_wrong_stage_owner_or_processing_scope_stops_before_calculation(state, mismatch):
    task = await _step(state, "plan" if mismatch == "plan" else "research")
    principal = replace(state.owner, storage_namespace="foreign") if mismatch == "namespace" else state.owner
    if mismatch == "purge":
        block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="fixture")
    try:
        if mismatch in {"plan", "namespace", "purge"}:
            with pytest.raises(AgentDispatchStoppedError), agent_calculator_scope(
                app=state.app, principal=principal, task=task,
            ):
                pytest.fail("Invalid authority must not enter the calculator scope")
        else:
            with agent_calculator_scope(app=state.app, principal=principal, task=task):
                if mismatch == "user":
                    with pytest.raises(AgentDispatchStoppedError):
                        await call(state, "PRIVATE_NOT_JSON", user="foreign")
                else:
                    policy = ProcessingPolicy({"qwen3.8:latest": "a" * 64}, "http://ollama.test")
                    handler = agent_calculator_handler(user_id=principal.storage_namespace)
                    with processing_scope(policy), pytest.raises(AgentDispatchStoppedError):
                        await handler({"expression": "2+2"})
    finally:
        if mismatch == "purge":
            unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="fixture")


async def test_escaped_callbacks_expire_and_bound_stage_cannot_advance_silently(state):
    task = await _step(state, "research")
    release = asyncio.Event()

    async def escaped(handler):
        await release.wait()
        return await handler({"expression": "2+2"})

    with agent_calculator_scope(app=state.app, principal=state.owner, task=task):
        handler = agent_calculator_handler(user_id=state.owner.storage_namespace)
        child = asyncio.create_task(escaped(handler))
        lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
        await state.store.agents.checkpoint(**lease, step_id="research", result="Saved work.")
        with pytest.raises(AgentDispatchStoppedError):
            await handler({"expression": "2+2"})
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child


async def test_agent_dispatch_cannot_fall_back_to_private_after_processing_mode_changes(state, caplog):
    task = await _step(state, "research")
    handler = AsyncMock(return_value={"result": "PRIVATE_FALLBACK"})
    private = ProcessingPolicy(
        {"qwen3.8:latest": "a" * 64}, "http://ollama.test",
        allowed_tools=frozenset({"calculate"}), tool_servers=frozenset({"http://tools.test"}),
    )
    forged = ToolRegistry({"calculate": ToolSpec(
        name="calculate", description="ordinary calculator", parameters={},
        server_url="http://tools.test", path="/calculate",
    )})
    with (
        agent_calculator_scope(app=state.app, principal=state.owner, task=task),
        processing_scope(private),
        private_tool_scope(user_id=state.owner.storage_namespace, handlers={"calculate": handler}),
        pytest.raises(AgentDispatchStoppedError),
    ):
        await call(state, "PRIVATE_NOT_JSON", tools=forged)
    handler.assert_not_awaited()
    assert "PRIVATE_NOT_JSON" not in caplog.text


@pytest.mark.parametrize("change", ["lease", "model", "source"])
async def test_changes_during_calculation_fence_the_late_result(file_state, monkeypatch, change):
    from audrey.tools import agent_calculator

    state = file_state
    project, _path, pins = await _selected_file_context(state)
    task = await _step(state, "research", project_id=project.project_id, file_inputs=pins)
    original = agent_calculator.calculate_value

    def revoked(request):
        result = original(request)
        with state.store._conn:
            if change == "lease":
                state.store._conn.execute("UPDATE app_agent_tasks SET fence=fence+1 WHERE task_id=?", (task["id"],))
            elif change == "model":
                state.store._conn.execute(
                    "UPDATE model_access_policies SET enabled=0 WHERE model_id='local'",
                )
            else:
                state.store._conn.execute("UPDATE app_projects SET instructions='changed' WHERE project_id=?",
                                          (project.project_id,))
        return result

    monkeypatch.setattr(agent_calculator, "calculate_value", revoked)
    if change == "model":
        await state.store.set_model_access_policy(actor_user_id=state.owner.user_id, model_id="local",
                                                  enabled=True, audience="users")
    with agent_calculator_scope(app=state.app, principal=state.owner, task=task):
        handler = agent_calculator_handler(user_id=state.owner.storage_namespace)
        with pytest.raises(AgentDispatchStoppedError, match="Project context changed" if change == "source" else None):
            await handler({"expression": "6*7"})
