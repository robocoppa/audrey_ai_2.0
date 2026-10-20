"""Private Project evidence reaches native answers only after scoped admission."""
from __future__ import annotations

import json

from test_native_private_workflow import _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native

from audrey.kb.qdrant import KBHit
from audrey.processing_policy import current_processing_policy
from audrey.routes.app import runs

EVIDENCE = "SYNTHETIC_PROJECT_FACT_5723: the invoice total is 425 dollars."


async def _ready_project(state):
    response = await state.client.post("/api/projects", json={
        "name": "Synthetic Private review", "instructions": "Use the owned invoice evidence.",
        "model_id": "private",
    })
    assert response.status_code == 201, response.text
    project_id = response.json()["id"]
    upload = await state.client.post(f"/api/projects/{project_id}/uploads", files={
        "file": ("invoice.txt", EVIDENCE, "text/plain"),
    })
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    row = await state.env.db.get_upload(file_id)
    hit = KBHit(score=0.99, source="invoice.txt", kind="text", chunk_idx=0, text=EVIDENCE,
                payload={"file_id": file_id, "user": state.owner.storage_namespace,
                         "filename": "invoice.txt", "private_required": True,
                         "artifact": "document"})
    state.env.qdrant.search_hybrid.return_value = ([hit], [])
    conversation = await state.client.post(f"/api/projects/{project_id}/conversations", json={})
    assert conversation.status_code == 201, conversation.text
    assert conversation.json()["private_required"] is True
    assert row["private_required"] and row["collection"].startswith("kb_private_text_")
    return project_id, conversation.json()["id"], file_id


async def test_native_private_project_answer_uses_protected_context_and_sources(native):
    state = native
    _, conversation_id, file_id = await _ready_project(state)
    response = await state.client.post(f"/api/conversations/{conversation_id}/runs", json={
        "content": "What is the invoice total?",
    })
    assert response.status_code == 202, response.text
    await _settle(state, response.json()["id"])
    context = next(message for message in state.captured[-1]
                   if message.get("name") == "audrey_project_context")
    assert EVIDENCE in context["content"] and "invoice.txt" in context["content"]
    scope = state.env.qdrant.search_hybrid.call_args.kwargs["scope"]
    assert scope.user == state.owner.storage_namespace and scope.file_ids == [file_id]
    saved = await state.client.get(f"/api/conversations/{conversation_id}/messages")
    answer = saved.json()["items"][-1]
    assert answer["status"] == "completed" and answer["sources"][0]["title"] == "invoice.txt"
    assert await state.store.chat_projections.due() == ()
    assert all(request.url.host == "ollama" for request in state.env.seen)


async def test_project_edit_during_private_preparation_requires_retry_before_message_save(native, monkeypatch):
    state = native
    project_id, conversation_id, _ = await _ready_project(state)
    original = runs.resolve_project_context

    async def edit_after_snapshot(request, **kwargs):
        assert current_processing_policy() is not None
        snapshot = await original(request, **kwargs)
        await state.store.projects.update(user_id=state.owner.user_id, project_id=project_id,
                                          instructions="Updated instructions after the snapshot")
        return snapshot

    monkeypatch.setattr(runs, "resolve_project_context", edit_after_snapshot)
    response = await state.client.post(f"/api/conversations/{conversation_id}/runs", json={
        "content": "Read the current invoice",
    })
    assert response.status_code == 409, response.text
    assert state.captured == []
    messages = await state.client.get(f"/api/conversations/{conversation_id}/messages")
    assert messages.json()["items"] == []


async def test_private_project_retrieval_failure_is_safe_and_saves_no_question(native):
    state = native
    _, conversation_id, _ = await _ready_project(state)
    state.env.qdrant.search_hybrid.side_effect = RuntimeError("SYNTHETIC_PRIVATE_BACKEND_CONTENT")
    response = await state.client.post(f"/api/conversations/{conversation_id}/runs", json={
        "content": "Read the invoice",
    })
    assert response.status_code == 503, response.text
    assert "SYNTHETIC_PRIVATE_BACKEND_CONTENT" not in json.dumps(response.json())
    assert state.captured == []
    messages = await state.client.get(f"/api/conversations/{conversation_id}/messages")
    assert messages.json()["items"] == []
