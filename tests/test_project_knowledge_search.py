"""Approved Project knowledge uses the same canonical local search fences."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_search import ContentSearchIndex
from audrey.content_search_service import ContentSearchService
from audrey.content_sources import ContentSourceReader
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.routes.app import router


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    index = ContentSearchIndex(tmp_path / "search.sqlite")
    root = tmp_path / "uploads"
    owners = [
        await store.resolve_external_identity(
            provider="test",
            subject=name,
            email=f"{name}@example.com",
            display_name=name,
            role="user",
            auth_method="test",
        )
        for name in ("alice", "bob")
    ]
    project = await store.projects.create(user_id=owners[0].user_id, name="Home")
    reader = ContentSourceReader(store, uploads, root)
    service = ContentSearchService(store, reader, index)
    app = FastAPI()
    app.state.application_store = store
    app.state.uploads_db = uploads
    app.state.content_search = service
    app.state.cfg = SimpleNamespace(raw={"kb": {"upload_root": str(root)}})
    current = [owners[0]]
    app.dependency_overrides[require_principal] = lambda: current[0]
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield SimpleNamespace(
            store=store,
            uploads=uploads,
            reader=reader,
            index=index,
            service=service,
            root=root,
            owners=owners,
            current=current,
            project=project,
            client=client,
        )
    await service.stop()
    index.close()
    uploads.close()
    store.close()


async def test_approved_knowledge_body_is_searchable_and_deleted_project_is_fenced(state):
    owner = state.owners[0]
    entry = await state.store.knowledge.create(
        user_id=owner.user_id,
        project_id=state.project.project_id,
        kind="decision",
        text="Use heliotrope on the north border.",
    )
    result = await state.client.get("/api/search", params={"q": "heliotrope", "kind": "knowledge"})
    assert result.status_code == 200, result.text
    [item] = result.json()["items"]
    assert (
        item["target"]["entry_id"] == entry["id"] and item["project_id"] == state.project.project_id
    )
    assert item["title"] == "Home: Decision" and not result.json()["indexing"]
    source_params = {"source_id": item["source_id"], "revision": item["revision"]}
    opened = await state.client.get("/api/search/source", params=source_params)
    assert opened.json()["text"] == entry["text"]
    state.current[0] = state.owners[1]
    assert (await state.client.get("/api/search/source", params=source_params)).status_code == 404
    state.current[0] = owner
    deleted = await state.client.delete(f"/api/projects/{state.project.project_id}")
    assert deleted.status_code == 204
    assert await state.index.list_sources(owner.user_id) == []
    assert (await state.client.get("/api/search/source", params=source_params)).status_code == 404
    assert (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"] == []


async def test_note_retains_user_text_with_missing_source_notice_without_exposing_deleted_file(
    state,
):
    owner = state.owners[0]
    directory = state.root / sanitize_user(owner.storage_namespace)
    directory.mkdir(parents=True)
    (directory / "evidence.txt").write_text("Original evidence body.", encoding="utf-8")
    await state.uploads.record_upload(
        file_id="evidence",
        user=owner.storage_namespace,
        filename="evidence.txt",
        mime="text/plain",
        bytes_=23,
        kind="text",
        collection="unused",
        chunks=0,
        uploaded_at="2026-10-29T12:00:00Z",
    )
    await state.store.projects.add_file(
        user_id=owner.user_id, project_id=state.project.project_id, file_id="evidence"
    )
    document = await state.reader.resolve(
        owner.user_id, owner.storage_namespace, "file:evidence:text"
    )
    entry = await state.store.knowledge.create(
        user_id=owner.user_id,
        project_id=state.project.project_id,
        kind="fact",
        text="My heliotrope note.",
        sources=[{"source_id": document.source_id, "revision": str(document.revision)}],
    )
    assert (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"]
    await state.uploads.request_file_deletion(
        "evidence", user=owner.storage_namespace, requested_at="2026-10-29T13:00:00Z"
    )
    # First request repairs the changed source metadata; next reads only the
    # explicitly saved user note, marked for source review.
    await state.client.get("/api/search", params={"q": "heliotrope"})
    result = await state.client.get("/api/search", params={"q": "heliotrope"})
    [item] = result.json()["items"]
    assert item["target"]["stale"] and item["excerpt"] == "My heliotrope note."
    unavailable = await state.client.get(
        f"/api/projects/{state.project.project_id}/knowledge/{entry['id']}/sources/0",
        params={"expected_revision": entry["revision"]},
    )
    assert unavailable.status_code == 409 and "Original evidence body" not in unavailable.text


async def test_pending_report_proposal_is_excluded_until_explicit_approval(state):
    from audrey.agent_runtime import AgentRuntime

    owner = state.owners[0]
    task = await state.store.agents.create(
        user_id=owner.user_id, objective="Public research", request_id="one"
    )

    async def execute(_task, step):
        return f"Saved {step} report."

    assert await AgentRuntime(state.store.agents, execute).run_once()
    task = await state.store.agents.get(user_id=owner.user_id, task_id=task["id"])
    created = await state.client.post(
        f"/api/projects/{state.project.project_id}/knowledge/proposals",
        json={
            "task_id": task["id"],
            "task_revision": task["revision"],
            "kind": "decision",
            "text": "A heliotrope proposal to review.",
            "request_id": "proposal-one",
        },
    )
    assert created.status_code == 201, created.text
    entry = created.json()
    assert entry["approval_state"] == "proposed"
    assert (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"] == []
    response = await state.client.post(
        f"/api/projects/{state.project.project_id}/knowledge/{entry['id']}/decision",
        json={"decision": "approve", "expected_revision": entry["revision"]},
    )
    assert response.status_code == 200, response.text
    result = await state.client.get("/api/search", params={"q": "heliotrope"})
    assert len(result.json()["items"]) == 1


async def test_stale_note_edit_during_source_resolution_cannot_return_previous_body(state, monkeypatch):
    owner = state.owners[0]
    entry = await state.store.knowledge.create(
        user_id=owner.user_id, project_id=state.project.project_id, kind="fact",
        text="Old heliotrope note.", sources=[{"source_id": "message:missing", "revision": "old"}],
    )
    source_id = f"knowledge:{entry['id']}"
    old = await state.reader.resolve(owner.user_id, owner.storage_namespace, source_id)
    assert old.target["stale"]
    resolve = state.reader._knowledge_resolver.resolve

    async def changing(*args):
        proof = await resolve(*args)
        await state.store.knowledge.update(
            user_id=owner.user_id, project_id=state.project.project_id,
            entry_id=entry["id"], expected_revision=1, text="New owner-authored note.",
        )
        return proof

    monkeypatch.setattr(state.reader._knowledge_resolver, "resolve", changing)
    assert await state.reader.resolve(owner.user_id, owner.storage_namespace, source_id) is None
    assert not state.reader.is_current(old)


async def test_corrupt_projection_cannot_turn_an_owned_note_into_foreign_text(state):
    from dataclasses import replace

    owner = state.owners[0]
    entry = await state.store.knowledge.create(
        user_id=owner.user_id, project_id=state.project.project_id,
        kind="fact", text="Actual heliotrope note.",
    )
    await state.client.get("/api/search", params={"q": "heliotrope"})
    document = await state.reader.resolve(owner.user_id, owner.storage_namespace, f"knowledge:{entry['id']}")
    await state.index.replace_source(replace(document, text="Forged orchid body.", target={"kind": "knowledge", "entry_id": "foreign"}))
    result = await state.client.get("/api/search", params={"q": "orchid"})
    assert result.status_code == 200 and result.json()["items"] == []
    assert "Forged orchid body" not in result.text
    repaired = await state.client.get("/api/search", params={"q": "heliotrope"})
    assert repaired.json()["items"][0]["target"]["entry_id"] == entry["id"]


async def test_private_project_knowledge_is_local_searchable_and_owner_scoped(state):
    owner = state.owners[0]
    project = await state.store.projects.create(user_id=owner.user_id, name="Private home", private_required=True)
    entry = await state.store.knowledge.create(
        user_id=owner.user_id, project_id=project.project_id,
        kind="decision", text="Private heliotrope decision.",
    )
    result = await state.client.get("/api/search", params={"q": "heliotrope", "kind": "knowledge"})
    [item] = result.json()["items"]
    assert item["private_required"] and item["target"]["entry_id"] == entry["id"]
    assert item["project_id"] == project.project_id
    state.current[0] = state.owners[1]
    opened = await state.client.get("/api/search/source", params={"source_id": item["source_id"], "revision": item["revision"]})
    assert opened.status_code == 404 and "Private heliotrope decision" not in opened.text


async def test_create_edit_and_delete_refresh_recent_completed_search_immediately(state):
    owner = state.owners[0]
    assert (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"] == []
    scan = state.service._owners[owner.user_id]
    completed_at = scan.completed_at
    assert completed_at > 0
    created = await state.client.post(
        f"/api/projects/{state.project.project_id}/knowledge",
        json={"kind": "fact", "text": "A heliotrope note.", "request_id": "create-refresh"},
    )
    assert created.status_code == 201, created.text
    entry = created.json()
    assert scan.completed_at == completed_at
    found = (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"]
    assert found[0]["target"]["entry_id"] == entry["id"]
    updated = await state.client.patch(
        f"/api/projects/{state.project.project_id}/knowledge/{entry['id']}",
        json={"expected_revision": entry["revision"], "text": "An orchid note."},
    )
    assert updated.status_code == 200, updated.text
    assert (await state.client.get("/api/search", params={"q": "heliotrope"})).json()["items"] == []
    assert (await state.client.get("/api/search", params={"q": "orchid"})).json()["items"]
    deleted = await state.client.delete(
        f"/api/projects/{state.project.project_id}/knowledge/{entry['id']}",
        params={"expected_revision": updated.json()["revision"]},
    )
    assert deleted.status_code == 200
    assert await state.index.list_sources(owner.user_id) == []
    assert (await state.client.get("/api/search", params={"q": "orchid"})).json()["items"] == []


async def test_async_refresh_started_before_purge_cannot_restore_note_body(state, monkeypatch):
    owner = state.owners[0]
    entry = await state.store.knowledge.create(
        user_id=owner.user_id, project_id=state.project.project_id,
        kind="fact", text="A heliotrope note.",
    )
    await state.client.get("/api/search", params={"q": "heliotrope"})
    read = asyncio.Event()
    release = asyncio.Event()
    resolve = state.reader.resolve

    async def delayed(*args):
        document = await resolve(*args)
        read.set()
        await release.wait()
        return document

    monkeypatch.setattr(state.reader, "resolve", delayed)
    refresh = asyncio.create_task(state.service.refresh_knowledge(
        owner.user_id, owner.storage_namespace, [entry["id"]],
    ))
    await asyncio.wait_for(read.wait(), 2)
    await state.service.purge_owner(owner.user_id)
    release.set()
    await asyncio.wait_for(refresh, 2)
    assert await state.index.list_sources(owner.user_id) == []
    assert owner.user_id not in state.service._owners
