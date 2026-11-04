"""Approved file-backed notes reach Agents without granting implicit file reads."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from test_agent_executor import (
    _MODEL,
    _activity,
    _answer,
    _content_requests,
    _file_call,
    _read_call,
)
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_runtime import AgentRuntime
from audrey.app_state.agents import AgentTaskError
from audrey.content_sources import ContentSourceReader
from audrey.kb.user_store import sanitize_user

SOURCE_BODY = "RAW LINKED FILE BODY: secret delivery evidence outside the approved note."
NOTE_TEXT = "Accepted delivery decision: prepare a packing checklist for 17 boxes."


async def _source(s, project, *, file_id="knowledge-source", text=SOURCE_BODY):
    directory = Path(s.app.state.cfg.raw["kb"]["upload_root"]) / sanitize_user(s.owner.storage_namespace)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{file_id}.txt"
    path.write_text(text)
    await s.files_db.record_upload(
        file_id=file_id, user=s.owner.storage_namespace, filename=f"{file_id}.txt", mime="text/plain",
        bytes_=len(text.encode()), kind="text", collection="", chunks=0,
        uploaded_at="2026-11-03T12:00:00+00:00",
    )
    await s.store.projects.add_file(user_id=s.owner.user_id, project_id=project.project_id, file_id=file_id)
    reader = ContentSourceReader(s.store, s.files_db, s.app.state.cfg.raw["kb"]["upload_root"])
    document = await reader.resolve(s.owner.user_id, s.owner.storage_namespace, f"file:{file_id}:text")
    assert document is not None and document.text == text
    return path, {"source_id": document.source_id, "revision": str(document.revision)}


async def _knowledge(s, *, kind="decision"):
    project = await s.store.projects.create(
        user_id=s.owner.user_id, name="Packing checklist", instructions="Keep the requested work concise.",
    )
    path, source = await _source(s, project)
    note = await s.store.knowledge.create(
        user_id=s.owner.user_id, project_id=project.project_id, kind=kind, text=NOTE_TEXT,
        sources=[source], pinned=True,
    )
    return project, path, note


async def _create(s, project, **changes):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        return await client.post("/api/agents", json={
            "objective": "Prepare the agreed delivery packing checklist.", "request_id": "file-knowledge-agent",
            "project_id": project.project_id, **changes,
        })


async def _step(s, project, stage="research", **changes):
    response = await _create(s, project, **changes)
    assert response.status_code == 201, response.text
    created = response.json()
    task = await s.store.agents.claim()
    assert task["id"] == created["id"]
    # Keep the optional title generation separate from the source/model checks.
    with s.store._lock, s.store._conn:
        s.store._conn.execute("UPDATE app_agent_tasks SET title='Delivery checklist',title_attempted=1 WHERE task_id=?", (task["id"],))
    task.update(title="Delivery checklist", title_generated=True, title_attempted=1)
    lease = {key: task[key] for key in ("lease_token", "fence")}
    lease["task_id"] = task["id"]
    for previous in ("plan", "research", "review"):
        if previous == stage:
            break
        await s.store.agents.reserve_operation(**lease, step_id=previous)
        task = await s.store.agents.checkpoint(**lease, step_id=previous, result=f"Saved {previous}.")
    await s.store.agents.reserve_operation(**lease, step_id=stage)
    return task


@pytest.mark.parametrize("stage", ["plan", "research", "review"])
async def test_file_backed_approved_note_reaches_actual_local_step_without_linked_body_or_read_grant(file_state, stage):
    s = file_state
    project, _path, note = await _knowledge(s)
    task = await _step(s, project, stage)
    assert task["project_context"]["knowledge"] == [{"id": note["id"], "revision": note["revision"]}]
    assert "file_inputs" not in task
    assert await s.executor(task, stage) == "Grounded report."
    payload = _content_requests(s)[0]
    context = next(message for message in payload["messages"]
                   if "Authenticated user's selected Project context" in message["content"])
    assert payload["model"] == _MODEL and NOTE_TEXT in context["content"]
    assert "not independently verified evidence" in context["content"]
    assert SOURCE_BODY not in json.dumps(payload) and "knowledge-source" not in json.dumps(payload)
    offered = {item["function"]["name"] for item in payload.get("tools", [])}
    assert "read_selected_file" not in offered
    if stage != "research":
        assert not offered
    assert not s.tool_requests
    manifest = s.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts").fetchone()[0]
    assert SOURCE_BODY not in manifest and NOTE_TEXT not in manifest


async def test_file_backed_note_cannot_authorize_a_model_requested_linked_file_read(file_state):
    s = file_state
    project, _path, _note = await _knowledge(s)
    task = await _step(s, project)
    s.responses.append(_read_call("knowledge-source"))
    with pytest.raises(AgentDispatchStoppedError, match="tool is unavailable"):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1 and SOURCE_BODY not in json.dumps(_content_requests(s))
    assert not s.tool_requests and not [row for row in await _activity(s, task) if row["kind"] == "tool"]


async def test_file_backed_note_can_publish_owned_output_without_reentering_upload_catalog_lock(file_state):
    s = file_state
    project, _path, _note = await _knowledge(s)
    task = await _step(s, project)
    s.responses.extend([_file_call(content="Approved packing checklist: 17 boxes."),
                        _answer("The approved packing checklist is saved.")])
    assert await s.executor(task, "research") == "The approved packing checklist is saved."
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert len(saved["outputs"]) == 1 and saved["project_context"]["current"]
    assert "file_inputs" not in saved and SOURCE_BODY not in json.dumps(_content_requests(s))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        download = await client.get(saved["outputs"][0]["download_url"])
    assert download.status_code == 200 and download.text == "Approved packing checklist: 17 boxes."
    assert not s.tool_requests


async def test_approved_note_and_explicit_file_selection_coexist_without_granting_its_other_source(file_state):
    s = file_state
    project, _path, note = await _knowledge(s)
    explicit_text = "EXPLICIT INPUT: use reusable crates and label each box."
    await _source(s, project, file_id="explicit-input", text=explicit_text)
    task = await _step(s, project, file_ids=["explicit-input"])
    assert task["project_context"]["knowledge"] == [{"id": note["id"], "revision": note["revision"]}]
    assert [item["file_id"] for item in task["file_inputs"]["files"]] == ["explicit-input"]
    s.responses.extend([
        _read_call("knowledge-source"), _read_call("explicit-input"),
        _file_call(content=explicit_text), _answer("The packing checklist is saved."),
    ])
    assert await s.executor(task, "research") == "The packing checklist is saved."
    payloads = _content_requests(s)
    assert len(payloads) == 4 and all(NOTE_TEXT in json.dumps(payload) for payload in payloads)
    assert SOURCE_BODY not in json.dumps(payloads)
    assert json.loads(payloads[1]["messages"][-1]["content"])["error"] == "file_not_selected"
    assert json.loads(payloads[2]["messages"][-1]["content"])["text"] == explicit_text
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://test") as client:
        download = await client.get(saved["outputs"][0]["download_url"])
    assert download.status_code == 200 and download.text == explicit_text
    events = [row for row in await _activity(s, task) if row["kind"] == "tool"]
    assert [(row["tool"], row["state"]) for row in events] == [
        ("read_selected_file", "failed"), ("read_selected_file", "completed"), ("create_file", "completed"),
    ]
    assert SOURCE_BODY not in json.dumps(events) and explicit_text not in json.dumps(events)
    assert not s.tool_requests


@pytest.mark.parametrize("change", ["body", "detach", "delete"])
async def test_file_knowledge_source_changes_during_actual_model_response_stop_dispatch_and_publication(file_state, change):
    s = file_state
    project, path, _note = await _knowledge(s)
    task = await _step(s, project)
    s.responses.append(_file_call(content="A stale packing checklist."))

    async def changed():
        s.on_chat = None
        if change == "body":
            path.write_text("Changed supporting evidence.")
        elif change == "detach":
            await s.store.projects.remove_file(user_id=s.owner.user_id, project_id=project.project_id, file_id="knowledge-source")
        else:
            await s.files_db.request_file_deletion("knowledge-source", user=s.owner.storage_namespace,
                                                 requested_at="2026-11-03T12:00:01+00:00")

    s.on_chat = changed
    with pytest.raises(AgentTaskError, match="Project context changed"):
        await s.executor(task, "research")
    assert len(_content_requests(s)) == 1 and not s.tool_requests
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=task["id"])
    assert not saved["project_context"]["current"] and not saved.get("outputs")
    assert not saved["steps"][1]["result"] and not saved["result"]
    assert not [row for row in await _activity(s, task) if row["kind"] == "tool"]


@pytest.mark.parametrize("kind", ["action", "question"])
async def test_assign_file_backed_origin_completes_locally_and_leaves_approved_entry_open(file_state, kind):
    s = file_state
    project, _path, note = await _knowledge(s, kind=kind)
    response = await _create(s, project, knowledge_entry_id=note["id"], knowledge_entry_revision=note["revision"])
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["project_context"]["origin_entry"] == {"id": note["id"], "revision": note["revision"]}
    assert "file_inputs" not in created
    s.responses.extend([_answer("Use the approved packing requirements."),
                        _answer("A packing checklist for 17 boxes."), _answer("The packing checklist is ready.")])
    assert await AgentRuntime(s.store.agents, s.executor).run_once()
    saved = await s.store.agents.get(user_id=s.owner.user_id, task_id=created["id"])
    assert saved["status"] == "completed" and saved["project_context"]["current"]
    assert saved["project_context"]["origin_entry"] == created["project_context"]["origin_entry"]
    assert await s.store.knowledge.get(user_id=s.owner.user_id, project_id=project.project_id, entry_id=note["id"]) == note
    for payload in _content_requests(s):
        assert NOTE_TEXT in json.dumps(payload) and SOURCE_BODY not in json.dumps(payload)
        assert payload["model"] == _MODEL
        assert "read_selected_file" not in {item["function"]["name"] for item in payload.get("tools", [])}
    assert not s.tool_requests


@pytest.mark.parametrize("origin", [False, True])
async def test_stale_file_source_is_excluded_or_rejects_explicit_origin_at_api_start(file_state, origin):
    s = file_state
    project, path, note = await _knowledge(s, kind="action")
    path.write_text("A new source version must not silently update an approved note.")
    changes = {"knowledge_entry_id": note["id"], "knowledge_entry_revision": note["revision"]} if origin else {}
    response = await _create(s, project, **changes)
    if origin:
        assert response.status_code == 404, response.text
        assert await s.store.agents.list(user_id=s.owner.user_id) == []
    else:
        assert response.status_code == 201, response.text
        assert response.json()["project_context"]["knowledge"] == []
        assert "file_inputs" not in response.json()
    assert await s.store.knowledge.get(user_id=s.owner.user_id, project_id=project.project_id, entry_id=note["id"]) == note
    assert not s.provider_requests and not s.tool_requests
