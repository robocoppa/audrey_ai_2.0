"""Tax project grouping preserves owned sources and saved work before chat."""
from __future__ import annotations

import hashlib
import json

import pytest
from test_native_private_workflow import _conversation, _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native
from test_private_document_ingestion import PRIVATE_TEXT
from test_tax_review_runtime import _enable

from audrey import tax_documents


@pytest.fixture(autouse=True)
async def tax_runtime(native, monkeypatch):
    _enable(native)
    native.env.qdrant.search_hybrid.return_value = ([], [])

    async def prepare(_app):
        return native.env.runtime

    monkeypatch.setattr(tax_documents, "prepare_native_private_runtime", prepare)


async def create_case(state, conversation_id):
    return await state.client.post(f"/api/conversations/{conversation_id}/tax-review",
        json={"tax_year": 2025, "goal": "Review only the synthetic income statement"})


async def upload(state, conversation_id):
    response = await state.client.post(f"/api/conversations/{conversation_id}/files",
        files={"file": ("statement.txt", PRIVATE_TEXT, "text/plain")})
    assert response.status_code == 200, response.text
    return response.json()["id"]


async def test_case_setup_survives_empty_draft_cleanup_and_has_a_private_project(native):
    state = native
    conversation = await _conversation(state)
    created = await create_case(state, conversation["id"])
    assert created.status_code == 201, created.text
    case = created.json()
    project_id = case["project_id"]
    project = await state.client.get(f"/api/projects/{project_id}")
    assert project.status_code == 200 and project.json()["private_required"]
    assert (await state.client.delete(f"/api/conversations/{conversation['id']}/empty")).status_code == 409
    saved = await state.client.get(f"/api/conversations/{conversation['id']}/tax-review")
    assert saved.json()["case"]["id"] == case["id"]
    overview = await state.client.get(f"/api/projects/{project_id}/tax-reviews")
    assert overview.status_code == 200
    assert [item["id"] for item in overview.json()["items"]] == [case["id"]]
    # Untouched ordinary drafts still disappear through the established flow.
    blank = await _conversation(state)
    assert (await state.client.delete(f"/api/conversations/{blank['id']}/empty")).status_code == 204


async def test_grouping_existing_chat_carries_its_ready_private_documents_into_project(native):
    state = native
    conversation = await _conversation(state)
    file_id = await upload(state, conversation["id"])
    sent = await state.client.post(f"/api/conversations/{conversation['id']}/runs",
        json={"content": "Read the synthetic statement", "attachment_ids": [file_id]})
    assert sent.status_code == 202, sent.text
    await _settle(state, sent.json()["id"])
    created = await create_case(state, conversation["id"])
    assert created.status_code == 201, created.text
    project_id = created.json()["project_id"]
    links = await state.store.projects.list_files(user_id=state.owner.user_id, project_id=project_id)
    assert [item.file_id for item in links] == [file_id]
    assert (await state.env.db.get_upload(file_id))["private_required"]


async def test_new_tax_review_chat_document_links_before_project_context_is_pinned(native, monkeypatch):
    state = native
    conversation = await _conversation(state)
    created = await create_case(state, conversation["id"])
    assert created.status_code == 201, created.text
    project_id = created.json()["project_id"]
    file_id = await upload(state, conversation["id"])
    assert await state.store.projects.list_files(user_id=state.owner.user_id, project_id=project_id) == ()
    manifests = []
    admission_type = type(state.manager.private_reviews.admission)
    original_manifest = admission_type._manifest

    async def manifest(self, *args, **kwargs):
        value = await original_manifest(self, *args, **kwargs)
        manifests.append(value)
        return value

    monkeypatch.setattr(admission_type, "_manifest", manifest)
    sent = await state.client.post(f"/api/conversations/{conversation['id']}/runs", json={
        "content": "Review the synthetic statement", "skill": "tax-review",
        "model_id": "private", "attachment_ids": [file_id],
    })
    assert sent.status_code == 202, sent.text
    await _settle(state, sent.json()["id"])
    links = await state.store.projects.list_files(user_id=state.owner.user_id, project_id=project_id)
    assert [item.file_id for item in links] == [file_id]
    assert PRIVATE_TEXT in json.dumps(state.captured[-1])
    task = await state.store.private_tasks.latest_for_conversation(user_id=state.owner.user_id,
                                                 conversation_id=conversation["id"])
    assert task["project_id"] == project_id
    assert manifests and all(file_id in value["project_files"] for value in manifests)
    expected = hashlib.sha256(json.dumps(manifests[0], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert task["pins"]["input_digest"] == expected


async def test_document_deleted_during_grouping_does_not_leave_a_project_reference(native, monkeypatch):
    state = native
    conversation = await _conversation(state)
    file_id = await upload(state, conversation["id"])
    sent = await state.client.post(f"/api/conversations/{conversation['id']}/runs",
        json={"content": "Read the synthetic statement", "attachment_ids": [file_id]})
    assert sent.status_code == 202, sent.text
    await _settle(state, sent.json()["id"])
    original = state.store.tax_cases.create

    async def delete_after_commit(**kwargs):
        result = await original(**kwargs)
        assert await state.env.db.delete_upload(file_id, user=state.owner.storage_namespace)
        return result

    monkeypatch.setattr(state.store.tax_cases, "create", delete_after_commit)
    response = await create_case(state, conversation["id"])
    assert response.status_code in {409, 503}, response.text
    # The case/project are durable, but the concurrently deleted document is
    # never advertised as a retained source. A retry can recover the saved case.
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=conversation["id"])
    assert saved is not None and saved["project_id"] is not None
    assert await state.store.projects.list_files(user_id=state.owner.user_id,
                                               project_id=saved["project_id"]) == ()
