"""Scripted local-model calls inspect actual task outputs before replacing them."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
from test_agent_executor import (
    _activity,
    _answer,
    _calculate_call,
    _content_requests,
    _file_call,
    _selected_file_context,
    _step,
)
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state
from test_agent_progress_execution import lease, task_context

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_files import create_agent_file
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.pipeline.prompts import AGENT_PLAN_SYSTEM, AGENT_RESEARCH_SYSTEM, AGENT_REVIEW_SYSTEM


def read_output(file_id, **arguments):
    return _answer("", message={"tool_calls": [{"function": {
        "name": "read_output_file", "arguments": {"file_id": file_id, **arguments},
    }}]})


def tool_replies(payload):
    return [json.loads(message["content"]) for message in payload["messages"] if message["role"] == "tool"]


async def test_work_and_review_read_real_outputs_and_replace_the_incorrect_download(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    task = await s.store.agents.create(
        user_id=s.owner.user_id, request_id="inspect-saved-garden-output",
        objective=("Save garden.md during Work. During Review, inspect the saved file, calculate "
                   "the supply total and remainder from472.35, and save a corrected garden.md. Do not browse."),
    )
    wrong = "# Garden purchasing\n\nTotal: USD 270.95\nRemaining: USD 201.40\n"
    corrected = "# Garden purchasing\n\nTotal: USD 260.30\nRemaining: USD 212.05\n"
    expression = "8*12.50+6*3.25+4*7.75+2*18.40+5*9.60+4*6.25"
    observed = {}

    async def reply():
        payload = json.loads(s.provider_requests[-1].content)
        system = payload["messages"][0]["content"]
        if system.startswith("You create concise agent titles."):
            return
        if system.startswith(AGENT_PLAN_SYSTEM):
            assert not payload.get("tools")
            s.responses.append(_answer("1. Save the purchasing file.\n2. Inspect and correct it during Review."))
            return
        offered = {item["function"]["name"] for item in payload["tools"]}
        assert {"create_file", "read_output_file", "calculate"} <= offered
        assert not {"memory_recall", "get_file_text", "read_selected_file", "search_selected_file"} & offered
        results = tool_replies(payload)
        if system.startswith(AGENT_RESEARCH_SYSTEM):
            if not results:
                assert task_context(payload)["current_output_files"] == []
                s.responses.append(_file_call(filename="garden.md", content=wrong))
            elif len(results) == 1:
                observed["original"] = results[-1]
                s.responses.append(read_output(results[-1]["file_id"]))
            else:
                assert len(results) == 2 and results[-1]["text"] == wrong
                assert results[-1]["revision"] == hashlib.sha256(wrong.encode()).hexdigest()
                observed["work_read"] = results[-1]
                # The claimed amounts in prose deliberately disagree with the
                # actual file. Review must use native output reading to see it.
                s.responses.append(_answer("Saved the total260.30 and remaining212.05."))
            return
        assert system.startswith(AGENT_REVIEW_SYSTEM)
        assert not {"web_search", "web_fetch"} & offered
        if not results:
            context_outputs = task_context(payload)["current_output_files"]
            assert len(context_outputs) == 1 and context_outputs[0]["file_id"] == observed["original"]["file_id"]
            s.responses.append(read_output(context_outputs[0]["file_id"]))
        elif len(results) == 1:
            assert results[-1]["text"] == wrong
            observed["review_read"] = results[-1]
            s.responses.append(_calculate_call(expression=expression))
        elif len(results) == 2:
            assert results[-1]["result"] == "260.3"
            s.responses.append(_calculate_call(expression=f"472.35-{results[-1]['result']}"))
        elif len(results) == 3:
            assert results[-1]["result"] == "212.05"
            s.responses.append(_file_call(filename="garden.md", content=corrected))
        elif len(results) == 4:
            assert results[-1]["file_id"] != observed["original"]["file_id"]
            observed["corrected"] = results[-1]
            s.responses.append(read_output(results[-1]["file_id"]))
        else:
            assert len(results) == 5 and results[-1]["text"] == corrected
            assert results[-1]["revision"] == hashlib.sha256(corrected.encode()).hexdigest()
            observed["corrected_read"] = results[-1]
            s.responses.append(_answer("Supply total: USD260.30. Remaining budget: USD212.05."))

    s.on_chat = reply
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    completed = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert completed["status"] == "completed"
    current = [item for item in completed["outputs"] if item["current"] is True]
    previous = [item for item in completed["outputs"] if item["current"] is False]
    assert len(current) == len(previous) == 1
    assert current[0]["file_id"] == observed["corrected"]["file_id"]
    assert previous[0]["file_id"] == observed["original"]["file_id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        current_download = await client.get(current[0]["download_url"])
        previous_download = await client.get(previous[0]["download_url"])
        detail = await client.get(f"/api/agents/{task['id']}")
    assert current_download.status_code == previous_download.status_code == detail.status_code == 200
    assert current_download.text == corrected and previous_download.text == wrong
    assert detail.json()["outputs"] == completed["outputs"]
    assert current[0]["download_url"] in completed["result"]
    assert previous[0]["download_url"] not in completed["result"]
    tools = [event for event in await _activity(s, task) if event["kind"] == "tool"]
    assert [(event["stage"], event["tool"], event["state"]) for event in tools] == [
        ("research", "create_file", "completed"), ("research", "read_output_file", "completed"),
        ("review", "read_output_file", "completed"), ("review", "calculate", "completed"),
        ("review", "calculate", "completed"), ("review", "create_file", "completed"),
        ("review", "read_output_file", "completed"),
    ]
    for private in (wrong, corrected, expression, observed["corrected_read"]["revision"],
                    observed["corrected"]["file_id"], "garden.md"):
        assert private not in json.dumps(tools)
    assert not s.tool_requests and not s.responses
    assert not s.app.state.inflight.pressure_snapshot()["in_use"]
    assert not s.app.state.gate.pressure_snapshot()["in_use"]


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_output_metadata_comes_from_canonical_receipts_not_incoming_snapshot(file_state, stage):
    s = file_state
    task = await _step(s, "research")
    output = await create_agent_file(s.app, s.owner, task, {
        "filename": "canonical.md", "format": "md", "content": "Actual owned output.",
    })
    if stage == "review":
        task = await s.store.agents.checkpoint(**lease(task), step_id="research", result="Prepared the output.")
        await s.store.agents.reserve_operation(**lease(task), step_id="review")
    task["outputs"] = [{"file_id": "forged-id", "filename": "forged-name", "download_url": "https://bad.example", "current": True}]
    inspected = []

    async def reply():
        payload = json.loads(s.provider_requests[-1].content)
        outputs = task_context(payload)["current_output_files"]
        assert len(outputs) == 1 and outputs[0]["file_id"] == output["file_id"]
        assert "forged" not in json.dumps(outputs) and "bad.example" not in json.dumps(outputs)
        results = tool_replies(payload)
        if not results:
            s.responses.append(read_output(outputs[0]["file_id"]))
        else:
            assert results[-1]["text"] == "Actual owned output."
            inspected.append(results[-1])
            s.responses.append(_answer("Inspected the actual output."))

    s.on_chat = reply
    assert await s.executor(task, stage) == "Inspected the actual output."
    assert len(inspected) == 1 and not s.tool_requests


@pytest.mark.parametrize("stage", ["research", "review"])
@pytest.mark.parametrize("candidate", ["other_task", "ordinary_upload"])
async def test_output_read_rejects_other_tasks_and_owned_project_files(file_state, stage, candidate):
    s = file_state
    if candidate == "other_task":
        task = await _step(s, "research")
        output = await create_agent_file(s.app, s.owner, task, {
            "filename": "other.md", "format": "md", "content": "OTHER TASK SECRET",
        })
        current = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
        await s.store.agents.control(user_id=s.owner.user_id, task_id=task["id"],
                                    expected_revision=current["revision"], action="cancel")
        await s.store.agents.create(user_id=s.owner.user_id, objective="Inspect only this Agent's output.", request_id="second")
        task = await s.store.agents.claim()
        await s.store.agents.reserve_operation(**lease(task), step_id="plan")
        task = await s.store.agents.checkpoint(**lease(task), step_id="plan", result="Inspect outputs.")
        if stage == "review":
            await s.store.agents.reserve_operation(**lease(task), step_id="research")
            task = await s.store.agents.checkpoint(**lease(task), step_id="research", result="Inspect outputs.")
        await s.store.agents.reserve_operation(**lease(task), step_id=stage)
        file_id = output["file_id"]
    else:
        project, _path, pins = await _selected_file_context(s, selected_text="ORDINARY SELECTED SECRET")
        task = await _step(s, stage, project_id=project.project_id, file_inputs=pins)
        file_id = "chosen-input"
    s.responses.extend([read_output(file_id), _answer("The file is not an output of this Agent.")])
    assert await s.executor(task, stage) == "The file is not an output of this Agent."
    response = tool_replies(_content_requests(s)[-1])[-1]
    assert response["error"] == "output_unavailable"
    assert "text" not in response and "SECRET" not in json.dumps(response)
    events = [event for event in await _activity(s, task) if event["kind"] == "tool"]
    assert [(event["tool"], event["state"]) for event in events] == [("read_output_file", "failed")]
    assert not s.tool_requests


async def test_plan_cannot_invoke_native_output_reader(file_state):
    s = file_state
    task = await _step(s, "plan")
    s.responses.append(read_output("invented-output"))
    with pytest.raises(AgentDispatchStoppedError, match="returned no report text"):
        await s.executor(task, "plan")
    assert not _content_requests(s)[0].get("tools")
    assert not s.tool_requests and not [event for event in await _activity(s, task) if event["kind"] == "tool"]


async def test_work_output_reader_survives_saved_question_and_database_reopen(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    task = await s.store.agents.create(user_id=s.owner.user_id, request_id="output-question-reopen",
        objective="Save a draft, ask for my approval during Work, then inspect that same saved file after my answer.")
    text = "# Saved invitation\n\nJoin the Autumn Social.\n"
    plan = "1. Save the invitation.\n2. Ask for approval.\n3. Inspect the saved invitation after the answer."
    s.responses.extend([_answer(plan), _file_call(filename="invitation.md", content=text),
                        _answer('{"input_request":"Should I keep the saved invitation?"}')])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    waiting = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert waiting["status"] == "waiting_for_input" and waiting["steps"][0]["result"] == plan
    assert len(waiting["outputs"]) == 1 and waiting["outputs"][0]["current"] is True
    original_output = waiting["outputs"][0]
    path = s.store.path
    s.store.close()
    reopened = ApplicationStore(path)
    s.store = s.app.state.application_store = reopened
    stages = []

    async def reply():
        payload = json.loads(s.provider_requests[-1].content)
        context = task_context(payload)
        assert context["current_stage"] in {"Work", "Review"}
        assert context["current_output_files"][0]["file_id"] == original_output["file_id"]
        results = tool_replies(payload)
        if not results:
            s.responses.append(read_output(original_output["file_id"]))
        else:
            assert results[-1]["text"] == text
            stages.append(context["current_stage"])
            s.responses.append(_answer("The saved invitation is ready."))

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
            question = waiting["input_requests"][0]
            response = await client.post(f"/api/agents/{task['id']}/inputs/{question['id']}/answer",
                json={"expected_revision": waiting["revision"], "answer": "Keep it."})
            assert response.status_code == 200 and response.json()["status"] == "queued"
            s.on_chat = reply
            assert await AgentRuntime(reopened.agents, s.executor).run_once()
            completed = (await client.get(f"/api/agents/{task['id']}")).json()
            download = await client.get(original_output["download_url"])
        assert completed["status"] == "completed" and completed["attempts"] == 2
        assert completed["steps"][0] == waiting["steps"][0]
        assert completed["outputs"] == waiting["outputs"]
        assert download.status_code == 200 and download.text == text
        assert stages == ["Work", "Review"]
        assert sum(payload["messages"][0]["content"].startswith(AGENT_PLAN_SYSTEM)
                   for payload in _content_requests(s)) == 1
        assert not s.tool_requests and not s.responses
    finally:
        reopened.close()
