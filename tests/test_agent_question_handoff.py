"""Observed prose-wrapped questions must wait instead of completing Agent Work."""

from __future__ import annotations

import json

import httpx
from test_agent_executor import _activity, _answer, _content_requests, _file_call, _progress_call
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state
from test_agent_progress_execution import task_context

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.pipeline.prompts import AGENT_PLAN_SYSTEM, AGENT_RESEARCH_SYSTEM, AGENT_REVIEW_SYSTEM


async def test_prose_wrapped_work_question_waits_then_resumes_after_restart_with_saved_progress(file_state):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    task = await s.store.agents.create(
        user_id=s.owner.user_id,
        objective=("Prepare an event invitation. During Work, invent and save an event label, "
                   "then ask whether the tone should be formal or casual. Wait for my answer. "
                   "Afterwards write invitation.md with the saved label and server timestamp. Do not browse."),
        request_id="prose-question-handoff",
    )
    label = "Autumn Social"
    note = f"Event label: {label}"
    question = "Should the invitation's tone be formal or casual?"
    # Preserve the actual reply shape that completed Work without displaying
    # an answer form: prose, a human-readable question, then a final envelope.
    observed_reply = (
        "I've saved the event label to my working progress. Before I write the invitation, one quick question:\n\n"
        "**Should the tone be formal or casual?**\n\n"
        '{"input_request": "Should the invitation\'s tone be formal or casual?"}'
    )
    plan = "1. Save an event label.\n2. Ask for the tone during Work.\n3. Write and review the invitation."
    s.responses.extend([
        _answer(plan), _progress_call(note), _progress_call(note), _answer(observed_reply),
    ])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        response = await client.get(f"/api/agents/{task['id']}")
    assert response.status_code == 200
    waiting = response.json()
    assert waiting["status"] == "waiting_for_input" and waiting["result"] == ""
    assert waiting["steps"][0]["status"] == "completed" and waiting["steps"][0]["result"] == plan
    assert waiting["steps"][1]["status"] == "pending" and waiting["steps"][1]["result"] == ""
    assert waiting["steps"][2]["status"] == "pending" and waiting["steps"][2]["result"] == ""
    assert len(waiting["input_requests"]) == 1
    pending = waiting["input_requests"][0]
    assert pending["step_id"] == "research" and pending["question"] == question and pending["answer"] is None
    assert pending["id"] and pending["created_at"] and pending["answered_at"] is None
    saved_progress = waiting["steps"][1]["progress"]
    assert saved_progress["note"] == note
    assert label not in task["objective"] and saved_progress["saved_at"] not in task["objective"]
    work_requests = [payload for payload in _content_requests(s)
                     if payload["messages"][0]["content"].startswith(AGENT_RESEARCH_SYSTEM)]
    receipts = [json.loads(message["content"]) for message in work_requests[-1]["messages"] if message["role"] == "tool"]
    assert len(receipts) == 2
    assert all(receipt["saved"] and receipt["saved_at"] == saved_progress["saved_at"] for receipt in receipts)
    assert all("note" not in receipt for receipt in receipts)
    assert not any(payload["messages"][0]["content"].startswith(AGENT_REVIEW_SYSTEM)
                   for payload in _content_requests(s))
    assert not waiting.get("outputs")
    events = await _activity(s, task)
    progress_events = [event for event in events if event["kind"] == "tool" and event["tool"] == "save_progress"]
    assert len(progress_events) == 2 and all(event["state"] == "completed" for event in progress_events)
    assert not any(event["stage"] == "review" for event in events)

    path = s.store.path
    s.store.close()
    reopened = ApplicationStore(path)
    s.store = s.app.state.application_store = reopened
    documents = []
    resumed_work = []

    async def resumed_reply():
        payload = json.loads(s.provider_requests[-1].content)
        system = payload["messages"][0]["content"]
        context = task_context(payload)
        if system.startswith(AGENT_RESEARCH_SYSTEM):
            if not resumed_work:
                assert not any(message["role"] == "tool" for message in payload["messages"])
                assert context["saved_progress"] == saved_progress
                assert context["clarifications"] == [{"step_id": "research", "question": question, "answer": "Casual"}]
                actual_progress = context["saved_progress"]
                actual_label = actual_progress["note"].removeprefix("Event label: ")
                documents.append(f"# {actual_label}\n\nHey, come join us!\n\nSaved progress: {actual_progress['saved_at']}\n")
                s.responses.append(_file_call(filename="invitation.md", content=documents[0]))
            else:
                s.responses.append(_answer("Saved your casual invitation."))
            resumed_work.append(payload)
        else:
            assert system.startswith(AGENT_REVIEW_SYSTEM)
            assert "saved_progress" not in context
            s.responses.append(_answer("Your casual invitation is ready."))

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
            response = await client.get(f"/api/agents/{task['id']}")
            assert response.status_code == 200 and response.json() == waiting
            response = await client.post(
                f"/api/agents/{task['id']}/inputs/{pending['id']}/answer",
                json={"expected_revision": waiting["revision"], "answer": "Casual"},
            )
            assert response.status_code == 200
            queued = response.json()
            assert queued["id"] == task["id"] and queued["status"] == "queued"
            assert queued["steps"][0] == waiting["steps"][0]
            assert queued["steps"][1]["progress"] == saved_progress

            s.on_chat = resumed_reply
            assert await AgentRuntime(reopened.agents, s.executor).run_once()
            response = await client.get(f"/api/agents/{task['id']}")
            assert response.status_code == 200
            finished = response.json()
            assert finished["status"] == "completed" and finished["attempts"] == 2
            assert finished["steps"][0] == waiting["steps"][0]
            assert finished["steps"][1]["progress"] == saved_progress
            assert finished["input_requests"][0]["question"] == question
            assert finished["input_requests"][0]["answer"] == "Casual"
            assert len(finished["outputs"]) == 1 and finished["outputs"][0]["filename"] == "invitation.md"
            downloaded = await client.get(finished["outputs"][0]["download_url"])
        assert downloaded.status_code == 200 and downloaded.text == documents[0]
        assert label in downloaded.text and saved_progress["saved_at"] in downloaded.text
        assert len(resumed_work) == 2
        assert sum(payload["messages"][0]["content"].startswith(AGENT_PLAN_SYSTEM)
                   for payload in _content_requests(s)) == 1
        assert not s.tool_requests and not s.responses
        assert not s.app.state.inflight.pressure_snapshot()["in_use"]
        assert not s.app.state.gate.pressure_snapshot()["in_use"]
    finally:
        reopened.close()


async def test_malformed_terminal_work_question_stops_without_completing_work_or_review(file_state, caplog):
    s = file_state
    s.app.state.cfg.raw["agent_tasks"] = {}
    task = await s.store.agents.create(
        user_id=s.owner.user_id, objective="Save a draft, ask for my choice, then wait.",
        request_id="malformed-question-handoff",
    )
    note = "PRIVATE_INTERMEDIATE_LABEL"
    s.responses.extend([
        _answer("Save the draft, ask the question, then finish after the answer."),
        _progress_call(note),
        _answer('PRIVATE_QUESTION_PREAMBLE\n\n{"input_request":42}'),
    ])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert saved["status"] == "failed" and saved["error"] == "Work returned an invalid input request."
    assert saved["steps"][0]["status"] == "completed"
    assert saved["steps"][1]["status"] == "pending" and saved["steps"][1]["result"] == ""
    assert saved["steps"][1]["progress"]["note"] == note
    assert saved["steps"][2]["status"] == "pending" and saved["steps"][2]["result"] == ""
    assert saved["input_requests"] == [] and saved["result"] == ""
    assert not any(payload["messages"][0]["content"].startswith(AGENT_REVIEW_SYSTEM)
                   for payload in _content_requests(s))
    assert not any(event["stage"] == "review" for event in await _activity(s, task))
    assert "PRIVATE_INTERMEDIATE_LABEL" not in caplog.text and "PRIVATE_QUESTION_PREAMBLE" not in caplog.text
    assert not s.tool_requests and not s.responses
