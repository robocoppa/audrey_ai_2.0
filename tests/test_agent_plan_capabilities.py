"""Plan knows later stage tools without gaining execution authority itself."""

from __future__ import annotations

import json

import pytest
from test_agent_executor import _answer, _content_requests, _selected_file_context, _step
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_runtime import AgentRuntime
from audrey.pipeline.prompts import AGENT_PLAN_CAPABILITIES_SYSTEM
from audrey.tools.discovery import ToolRegistry


def catalogue(payload):
    messages = [message for message in payload["messages"]
                if AGENT_PLAN_CAPABILITIES_SYSTEM in message["content"]]
    assert len(messages) == 1 and messages[0]["role"] == "system"
    stages = json.JSONDecoder().raw_decode(messages[0]["content"].split(AGENT_PLAN_CAPABILITIES_SYSTEM, 1)[1])[0]
    assert [stage["stage"] for stage in stages] == ["Work", "Review"]
    for stage in stages:
        assert set(stage) == {"stage", "tools"}
        assert all(isinstance(name, str) for name in stage["tools"])
        assert stage["tools"] == sorted(set(stage["tools"]))
    return {stage["stage"]: set(stage["tools"]) for stage in stages}


async def run_stages(s, task):
    payloads = []
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    for stage in ("plan", "research", "review"):
        result = await s.executor(task, stage)
        payloads.append(_content_requests(s)[-1])
        if stage != "review":
            task = await s.store.agents.checkpoint(**lease, step_id=stage, result=result)
    advertised = catalogue(payloads[0])
    assert not payloads[0].get("tools")
    for stage, payload in zip(("Work", "Review"), payloads[1:], strict=True):
        assert advertised[stage] == {tool["function"]["name"] for tool in payload.get("tools", [])}
        assert not any(AGENT_PLAN_CAPABILITIES_SYSTEM in message["content"]
                       for message in payload["messages"])
    assert not s.tool_requests
    return advertised, payloads


@pytest.mark.parametrize("context", ["none", "project", "selected"])
async def test_plan_catalogue_matches_actual_later_tools_without_executing_them(file_state, context):
    s = file_state
    arguments = {}
    if context == "project":
        project = await s.store.projects.create(user_id=s.owner.user_id, name="No implicit files")
        arguments["project_id"] = project.project_id
    elif context == "selected":
        project, _path, pins = await _selected_file_context(s)
        arguments.update(project_id=project.project_id, file_inputs=pins)
    task = await _step(s, objective="Save a budget, calculate its total and inspect its saved file.", **arguments)
    advertised, payloads = await run_stages(s, task)
    common = {"calculate", "create_file", "read_output_file"}
    selected = {"read_selected_file", "search_selected_file"} if context == "selected" else set()
    assert advertised == {
        "Work": common | selected | {"web_search", "web_fetch", "save_progress"},
        "Review": common | selected,
    }
    # The shared registry also contains personal tools; no stage catalogue or
    # actual Agent offer may widen to those unrelated capabilities.
    assert not {"memory_recall", "memory_store", "get_file_text"} & set.union(*advertised.values())
    plan_catalogue = next(message["content"] for message in payloads[0]["messages"]
                          if AGENT_PLAN_CAPABILITIES_SYSTEM in message["content"])
    assert "http://tools.test" not in plan_catalogue


async def test_missing_output_storage_is_not_advertised_to_plan(file_state):
    s = file_state
    del s.app.state.storage_lifecycle
    task = await _step(s)
    advertised, _payloads = await run_stages(s, task)
    assert advertised == {
        "Work": {"web_search", "web_fetch", "calculate", "save_progress"},
        "Review": {"calculate"},
    }


@pytest.mark.parametrize("web_state", ["unavailable", "absent"])
async def test_plan_does_not_promise_unavailable_web_or_discovered_personal_tools(state, web_state):
    if web_state == "unavailable":
        state.tools.by_name["web_search"].available = False
    else:
        state.app.state.tools = ToolRegistry()
    task = await _step(state)
    advertised, _payloads = await run_stages(state, task)
    assert advertised["Work"] == ({"web_fetch"} if web_state == "unavailable" else set()) | {
        "calculate", "save_progress",
    }
    assert advertised["Review"] == {"calculate"}


async def test_plan_can_still_wait_for_a_material_user_detail_with_stage_tools_described(state):
    created = await state.store.agents.create(user_id=state.owner.user_id,
        objective="Prepare the invitation for the recipient I will identify.", request_id="material-choice")
    question = "Who should receive the invitation?"
    state.responses.append(_answer(json.dumps({"input_request": question})))
    assert await AgentRuntime(state.store.agents, state.executor).run_once()
    waiting = await state.store.agents.get(user_id=state.owner.user_id, task_id=created["id"])
    assert waiting["status"] == "waiting_for_input"
    assert waiting["input_requests"][0]["step_id"] == "plan"
    assert waiting["input_requests"][0]["question"] == question
    assert all(step["status"] == "pending" for step in waiting["steps"])
    plan = _content_requests(state)[0]
    assert catalogue(plan)["Review"] == {"calculate"}
    assert not plan.get("tools") and not state.tool_requests


async def test_describing_execution_tools_does_not_allow_plan_to_call_them(file_state):
    s = file_state
    task = await _step(s)
    s.responses.append(_answer("", message={"tool_calls": [{"function": {
        "name": "read_output_file", "arguments": {"file_id": "invented-file"},
    }}]}))
    with pytest.raises(AgentDispatchStoppedError, match="returned no report text"):
        await s.executor(task, "plan")
    payload = _content_requests(s)[0]
    assert "read_output_file" in catalogue(payload)["Review"]
    assert not payload.get("tools") and not s.tool_requests
    assert not await s.files_db.list_user(s.owner.storage_namespace)
