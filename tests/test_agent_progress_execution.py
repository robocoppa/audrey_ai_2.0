"""Real native progress publication and restart with scripted local-model replies."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_agent_executor import (
    _activity,
    _answer,
    _content_requests,
    _file_call,
    _progress_call,
    _selected_file_context,
    _step,
)
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_file_inputs import get_agent_file_inputs
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskError
from audrey.pipeline.prompts import AGENT_PLAN_SYSTEM, AGENT_RESEARCH_SYSTEM, AGENT_REVIEW_SYSTEM


def task_context(payload):
    return json.loads(next(message["content"] for message in payload["messages"]
        if message["role"] == "user" and message["content"].startswith('{"objective"')))


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def test_work_progress_survives_question_restart_and_answer_without_old_tool_transcript(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    task = await s.store.agents.create(
        user_id=s.owner.user_id,
        objective=("During Work generate a draft label, save useful intermediate progress, then ask me "
                   "whether the invitation should be formal or casual without including the label. "
                   "After my answer save invitation.md using that label and its saved-at timestamp. Do not browse."),
        request_id="progress-question-restart",
    )
    label = "Copper Orchard"
    note = f"Draft label: {label}. Invitation opener: Join us for a garden picnic. Tone is pending."
    question = "Would you like the invitation to be formal or casual?"
    plan = "1. Save a draft label during Work.\n2. Ask for the tone.\n3. Save the invitation after the answer."
    s.responses.extend([_answer(plan), _progress_call(note), _answer(json.dumps({"input_request": question}))])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    waiting = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert waiting["status"] == "waiting_for_input" and waiting["attempts"] == 1
    assert waiting["steps"][0]["status"] == "completed" and waiting["steps"][0]["result"] == plan
    work = waiting["steps"][1]
    assert work["status"] == "pending" and work["progress"]["note"] == note and not work["result"]
    saved_progress = work["progress"]
    assert saved_progress["saved_at"] not in task["objective"] and label not in task["objective"]
    assert label not in question and saved_progress["saved_at"] not in question
    first_work = [payload for payload in _content_requests(s)
                  if payload["messages"][0]["content"].startswith(AGENT_RESEARCH_SYSTEM)]
    assert "saved_progress" not in task_context(first_work[0])
    tool_result = json.loads(first_work[1]["messages"][-1]["content"])
    assert tool_result["saved"] is True and tool_result["saved_at"] == saved_progress["saved_at"]
    assert "note" not in tool_result
    before = dict(s.store._conn.execute(
        "SELECT operations,active_seconds FROM app_agent_tasks WHERE task_id=?", (task["id"],),
    ).fetchone())
    path = s.store.path
    s.store.close()
    reopened = ApplicationStore(path)
    s.store = s.app.state.application_store = reopened
    try:
        assert get_agent_file_inputs(s.app) is not None
        assert await reopened.agents.get(user_id=s.owner.user_id, task_id=task["id"]) == waiting
        pending = waiting["input_requests"][0]
        assert pending["step_id"] == "research"
        queued = await reopened.agents.answer_input(
            user_id=s.owner.user_id, task_id=task["id"], request_id=pending["id"],
            expected_revision=waiting["revision"], answer="Casual.",
        )
        assert queued["id"] == task["id"] and queued["steps"][0] == waiting["steps"][0]
        assert queued["steps"][1]["progress"] == saved_progress
        assert dict(reopened._conn.execute(
            "SELECT operations,active_seconds FROM app_agent_tasks WHERE task_id=?", (task["id"],),
        ).fetchone()) == before
        output = f"# {label}\n\nHey, join us for a garden picnic!\n\nSaved progress: {saved_progress['saved_at']}\n"
        s.responses.extend([
            _file_call(filename="invitation.md", content=output), _answer("Saved the casual invitation."),
            _answer("Your casual garden invitation is ready."),
        ])
        previous_requests = len(_content_requests(s))
        assert await AgentRuntime(reopened.agents, s.executor).run_once()
        saved = await reopened.agents.get(user_id=s.owner.user_id, task_id=task["id"])
        assert saved["status"] == "completed" and saved["attempts"] == 2
        assert saved["id"] == task["id"] and saved["steps"][0] == waiting["steps"][0]
        assert saved["steps"][1]["progress"] == saved_progress
        resumed = _content_requests(s)[previous_requests:]
        assert resumed[0]["messages"][0]["content"].startswith(AGENT_RESEARCH_SYSTEM)
        context = task_context(resumed[0])
        assert context["saved_progress"] == saved_progress
        assert context["clarifications"] == [{"step_id": "research", "question": question, "answer": "Casual."}]
        assert not any(message["role"] == "tool" for message in resumed[0]["messages"])
        for payload in resumed:
            if payload["messages"][0]["content"].startswith(AGENT_REVIEW_SYSTEM):
                assert "saved_progress" not in task_context(payload)
        assert sum(payload["messages"][0]["content"].startswith(AGENT_PLAN_SYSTEM)
                   for payload in _content_requests(s)) == 1
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
            downloaded = await client.get(saved["outputs"][0]["download_url"])
        assert downloaded.status_code == 200 and downloaded.text == output
        events = [event for event in await _activity(s, task)
                  if event["kind"] == "tool" and event["tool"] == "save_progress"]
        assert len(events) == 1 and events[0]["state"] == "completed" and events[0]["stage"] == "research"
        assert note not in json.dumps(events) and label not in json.dumps(events)
        assert not s.tool_requests and not s.responses
        assert not s.app.state.inflight.pressure_snapshot()["in_use"]
        assert not s.app.state.gate.pressure_snapshot()["in_use"]
    finally:
        reopened.close()


async def test_work_loads_canonical_progress_instead_of_a_forged_task_copy(state):
    task = await _step(state, "research")
    canonical = await state.store.agents.save_work_progress(**lease(task), note="The canonical draft.")
    task["steps"][1]["progress"] = {"note": "FORGED_TASK_NOTE", "saved_at": "FORGED_TIME"}
    assert await state.executor(task, "research") == "Grounded report."
    context = task_context(_content_requests(state)[0])
    assert context["saved_progress"] == canonical
    assert "FORGED_" not in json.dumps(_content_requests(state))


async def test_saved_progress_is_not_supplied_to_review(state):
    task = await _step(state, "research")
    note = "PRIVATE_INTERMEDIATE_DRAFT"
    await state.store.agents.save_work_progress(**lease(task), note=note)
    task = await state.store.agents.checkpoint(**lease(task), step_id="research", result="The finished deliverable.")
    assert await state.executor(task, "review") == "Grounded report."
    context = task_context(_content_requests(state)[0])
    assert "saved_progress" not in context and note not in json.dumps(_content_requests(state))
    assert context["saved_work"]["research"] == "The finished deliverable."


async def test_changed_selected_source_blocks_progress_context_before_model_dispatch(file_state):
    s = file_state
    project, path, pins = await _selected_file_context(s)
    task = await _step(s, "research", project_id=project.project_id, file_inputs=pins)
    await s.store.agents.save_work_progress(**lease(task), note="PRIVATE_SOURCE_DRAFT")
    path.write_text("A changed selected source")
    with pytest.raises((AgentTaskError, AgentDispatchStoppedError)):
        await s.executor(task, "research")
    assert not _content_requests(s)
    assert "PRIVATE_SOURCE_DRAFT" not in json.dumps(s.provider_requests, default=str)


async def test_saving_progress_cannot_bypass_the_operation_allowance(state, monkeypatch):
    task = await _step(state, "research", max_operations=3)
    state.responses.append(_progress_call("A draft that must not be published"))
    save = AsyncMock(side_effect=AssertionError("Exhausted task reached progress publication"))
    monkeypatch.setattr(state.store.agents, "save_work_progress", save)
    with pytest.raises(AgentTaskError, match="operation allowance"):
        await state.executor(task, "research")
    save.assert_not_awaited()
    assert "progress" not in (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["steps"][1]
