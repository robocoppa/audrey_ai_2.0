"""Canonical source picking and exact previews without Search or model calls."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.content_sources import ContentSourceReader
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.knowledge_source_browser import KnowledgeSourceBrowser
from audrey.knowledge_sources import KnowledgeSourceResolver
from audrey.routes.app import router

STAMP = "2026-11-01T12:00:00+00:00"


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    projects = [await store.projects.create(user_id=owner.user_id, name=f"{owner.display_name} work")
                for owner in owners]
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    root = tmp_path / "uploads"
    root.mkdir()
    reader = ContentSourceReader(store, uploads, root)
    app = FastAPI()
    app.state.application_store = store
    app.state.knowledge_source_resolver = KnowledgeSourceResolver(store, reader)
    app.include_router(router)
    current = [owners[0]]

    async def authenticate():
        return current[0]

    app.dependency_overrides[require_principal] = authenticate
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owners=owners, project=projects[0], other_project=projects[1],
                              uploads=uploads, root=root, reader=reader, app=app, current=current,
                              client=client, path=f"/api/projects/{projects[0].project_id}/knowledge")
    uploads.close()
    store.close()


async def message(state, *, mid=None, project=None, owner=None, role="user", status="completed",
                  text="Saved source passage.", archived=False):
    owner = owner or state.owners[0]
    project = project or state.project
    conversation = await state.store.conversations.create(
        user_id=owner.user_id, title="Orchard decisions", project_id=project.project_id,
        private_required=project.private_required,
    )
    mid = mid or "msg_" + uuid.uuid4().hex
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_messages(message_id,conversation_id,user_id,sequence_no,role,status,content,"
            "created_at,updated_at) VALUES (?,?,?,1,?,?,?,?,?)",
            (mid, conversation.conversation_id, owner.user_id, role, status, text, STAMP, STAMP),
        )
        if archived:
            state.store._conn.execute("UPDATE app_conversations SET archived_at=? WHERE conversation_id=?",
                                      (STAMP, conversation.conversation_id))
    return f"message:{mid}", conversation.conversation_id


async def file(state, *, fid=None, owner=None, project=None, private=False, kind="text",
               status="ready", artifact=None, text="Saved file source.", attach=True):
    owner = owner or state.owners[0]
    project = project or state.project
    fid = fid or "file_" + uuid.uuid4().hex
    name = {"text": "orchard.txt", "video": "orchard.mp4", "image": "orchard.png"}[kind]
    directory = state.root / sanitize_user(owner.storage_namespace)
    directory.mkdir(exist_ok=True)
    path = directory / f"{fid}{Path(name).suffix}"
    path.write_text(text, encoding="utf-8")
    if artifact:
        path.with_suffix(f".{artifact}.txt").write_text(text, encoding="utf-8")
    with state.uploads._lock:
        state.uploads._record_locked(fid, owner.storage_namespace, name, "text/plain", len(text.encode()),
                                    kind, "", 0, STAMP, status, private_required=private)
    if attach:
        # Deliberately permits corrupt mixed-privacy memberships for boundary tests.
        with state.store._lock, state.store._conn:
            state.store._conn.execute(
                "INSERT INTO app_project_files(project_id,file_id,added_at) VALUES (?,?,?)",
                (project.project_id, fid, STAMP),
            )
    return fid, path


async def report(state, owner=None):
    owner = owner or state.owners[0]
    task = await state.store.agents.create(user_id=owner.user_id, objective="Study public orchard guidance.",
                                           request_id=uuid.uuid4().hex)
    claimed = await state.store.agents.claim()
    lease = {"task_id": claimed["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    for step in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=step)
        await state.store.agents.checkpoint(**lease, step_id=step, result="Saved checkpoint.")
    await state.store.agents.complete(**lease, result="Saved generated report.")
    return f"agent:{task['id']}:report"


async def candidates(state, **params):
    return await state.client.get(f"{state.path}/source-candidates", params=params)


async def preview(state, source_id, **params):
    return await state.client.get(f"{state.path}/source-preview", params={"source_id": source_id, **params})


async def test_message_browse_is_project_scoped_metadata_and_includes_archived_incomplete(state, monkeypatch):
    selected, _ = await message(state, archived=True, status="incomplete")
    await message(state, role="tool")
    await message(state, status="in_progress")
    await message(state, text=" ")
    await message(state, owner=state.owners[1], project=state.other_project)
    unrelated = await state.store.projects.create(user_id=state.owners[0].user_id, name="Other work")
    await message(state, project=unrelated)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("Browsing metadata must not resolve source bodies or depend on Search")

    monkeypatch.setattr(state.reader, "resolve", forbidden)
    response = await candidates(state)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"items": [{"source_id": selected, "kind": "message",
                                         "title": "Orchard decisions · User message 1", "private_required": False}],
                               "next_cursor": None}
    assert "Saved source passage" not in response.text


async def test_browse_cursor_is_keyset_and_bound_to_project_kind_owner_and_version(state):
    for suffix in ("a", "b", "c"):
        await message(state, mid=f"msg_{suffix}")
    first = (await candidates(state, limit=1)).json()
    assert first["items"][0]["source_id"] == "message:msg_a"
    cursor = first["next_cursor"]
    second = (await candidates(state, limit=1, cursor=cursor)).json()
    assert second["items"][0]["source_id"] == "message:msg_b"
    assert second["next_cursor"] != cursor
    assert (await candidates(state, kind="agent", cursor=cursor)).status_code == 422
    assert (await candidates(state, cursor=cursor[:-3] + "bad")).status_code == 422
    other = await state.store.projects.create(user_id=state.owners[0].user_id, name="Another Project")
    other_path = f"/api/projects/{other.project_id}/knowledge/source-candidates"
    assert (await state.client.get(other_path, params={"cursor": cursor})).status_code == 422
    state.current[0] = state.owners[1]
    assert (await candidates(state, cursor=cursor)).status_code == 404
    state.current[0] = state.owners[0]
    await state.store.projects.update(user_id=state.owners[0].user_id, project_id=state.project.project_id,
                                      name="Renamed work")
    assert (await candidates(state, cursor=cursor)).status_code == 422


async def test_files_show_only_attached_ready_owned_safe_existing_artifacts_without_reading(state, monkeypatch):
    fid, _ = await file(state)
    video, _ = await file(state, kind="video", artifact="transcript")
    await file(state, attach=False)
    await file(state, status="processing")
    await file(state, private=True, artifact="ocr")
    await file(state, owner=state.owners[1])
    await file(state, kind="image")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Metadata browsing must not open or extract file bodies")

    monkeypatch.setattr(state.reader, "_read_file", forbidden)
    monkeypatch.setattr("audrey.content_sources._read_text", forbidden)
    monkeypatch.setattr("audrey.content_sources._extract_search_text", forbidden)
    response = await candidates(state, kind="file")
    assert response.status_code == 200, response.text
    assert {item["source_id"] for item in response.json()["items"]} == {
        f"file:{fid}:text", f"file:{video}:transcript",
    }
    assert all(item["kind"] == "file" and not item["private_required"] for item in response.json()["items"])
    assert "Saved file source" not in response.text and str(state.root) not in response.text


async def test_all_owned_reports_are_candidates_and_provenance_only(state):
    own = await report(state)
    await report(state, state.owners[1])
    response = await candidates(state, kind="agent")
    assert [item["source_id"] for item in response.json()["items"]] == [own]
    opened = await preview(state, own)
    assert opened.status_code == 200 and opened.json()["target"]["origin"] == "generated"
    assert opened.json()["text"] == "Saved generated report."


async def test_preview_pages_exact_revision_and_references_are_usable_for_creation(state):
    source, cid = await message(state, text="A" * 4000 + "B" * 1000)
    first = await preview(state, source)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["text"] == "A" * 4000 and body["next_offset"] == 4000 and body["total_chars"] == 5000
    second = await preview(state, source, revision=body["revision"], offset=4000)
    assert second.json()["text"] == "B" * 1000 and second.json()["next_offset"] is None
    assert (await preview(state, source, offset=5001)).status_code == 422
    assert (await preview(state, source, revision="old")).status_code == 409
    created = await state.client.post(state.path, json={"kind": "decision", "text": "Use the gate.",
        "sources": [{"source_id": source, "revision": body["revision"]}], "request_id": uuid.uuid4().hex})
    assert created.status_code == 201 and not created.json()["stale"]
    await state.store.conversations.delete(user_id=state.owners[0].user_id, conversation_id=cid)
    assert (await preview(state, source, revision=body["revision"])).status_code == 404
    entry = (await state.client.get(f"{state.path}/{created.json()['id']}")).json()
    assert entry["text"] == "Use the gate." and entry["stale"]


async def test_retained_file_text_previews_without_original_and_disappears_on_tombstone(state):
    fid, path = await file(state, artifact="ocr")
    path.unlink()
    source = f"file:{fid}:text"
    assert [item["source_id"] for item in (await candidates(state, kind="file")).json()["items"]] == [source]
    opened = await preview(state, source)
    assert opened.status_code == 200 and opened.json()["availability"] == "text_only"
    await state.uploads.request_file_deletion(fid, user=state.owners[0].storage_namespace, requested_at=STAMP)
    assert (await candidates(state, kind="file")).json()["items"] == []
    assert (await preview(state, source)).status_code == 404


async def test_private_text_is_visible_only_inside_private_project_and_never_extracted(state, monkeypatch):
    private = await state.store.projects.create(user_id=state.owners[0].user_id, name="Protected", private_required=True)
    fid, _ = await file(state, project=private, private=True, artifact="ocr")
    source = f"file:{fid}:text"
    assert (await preview(state, source)).status_code == 409
    state.path = f"/api/projects/{private.project_id}/knowledge"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Protected sources may only use existing local saved text")

    monkeypatch.setattr("audrey.content_sources._extract_search_text", forbidden)
    assert (await candidates(state, kind="file")).json()["items"][0]["private_required"]
    opened = await preview(state, source)
    assert opened.status_code == 200 and opened.json()["private_required"]


@pytest.mark.parametrize("kind", ["message", "file", "agent"])
async def test_foreign_sources_and_projects_never_return_body(state, kind):
    if kind == "message":
        source, _ = await message(state, owner=state.owners[1], project=state.other_project)
    elif kind == "file":
        fid, _ = await file(state, owner=state.owners[1], project=state.other_project)
        source = f"file:{fid}:text"
    else:
        source = await report(state, state.owners[1])
    assert (await preview(state, source)).status_code == 404
    state.path = f"/api/projects/{state.other_project.project_id}/knowledge"
    assert (await candidates(state, kind=kind)).status_code == 404
    assert (await preview(state, source)).status_code == 404


@pytest.mark.parametrize("change", ["delete", "detach", "project", "owner", "privacy", "text"])
async def test_preview_final_fence_catches_changes_after_awaited_read(state, monkeypatch, change):
    source, cid = await message(state)
    original = state.reader.resolve

    async def racing(*args):
        document = await original(*args)
        with state.store._lock, state.store._conn:
            if change == "delete":
                state.store._conn.execute("DELETE FROM app_messages WHERE conversation_id=?", (cid,))
            elif change == "detach":
                state.store._conn.execute("UPDATE app_conversations SET project_id=NULL WHERE conversation_id=?", (cid,))
            elif change == "project":
                state.store._conn.execute("DELETE FROM app_projects WHERE project_id=?", (state.project.project_id,))
            elif change == "owner":
                state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owners[0].user_id,))
            elif change == "privacy":
                state.store._conn.execute("UPDATE app_projects SET private_required=1 WHERE project_id=?", (state.project.project_id,))
            else:
                state.store._conn.execute("UPDATE app_messages SET content='Changed',updated_at='later' WHERE conversation_id=?", (cid,))
        return document

    monkeypatch.setattr(state.reader, "resolve", racing)
    response = await preview(state, source)
    assert response.status_code == 409 and "Saved source passage" not in response.text


@pytest.mark.parametrize("kind", ["message", "file", "agent"])
async def test_candidate_final_fence_catches_deletion_before_response(state, monkeypatch, kind):
    if kind == "message":
        await message(state)
    elif kind == "file":
        await file(state)
    else:
        await report(state)
    original = KnowledgeSourceBrowser.page

    async def racing(browser, *args):
        result = await original(browser, *args)
        with state.store._lock, state.store._conn:
            state.store._conn.execute("DELETE FROM app_projects WHERE project_id=?", (state.project.project_id,))
        return result

    monkeypatch.setattr(KnowledgeSourceBrowser, "page", racing)
    response = await candidates(state, kind=kind)
    assert response.status_code == 409 and "items" not in response.json()


@pytest.mark.parametrize("kind", ["message", "file", "agent"])
async def test_candidate_final_fence_rejects_source_deleted_after_metadata_read(state, monkeypatch, kind):
    if kind == "message":
        source, cid = await message(state)
    elif kind == "file":
        fid, _ = await file(state)
    else:
        source = await report(state)
    original = KnowledgeSourceBrowser.page

    async def racing(browser, *args):
        result = await original(browser, *args)
        if kind == "file":
            await state.uploads.request_file_deletion(fid, user=state.owners[0].storage_namespace,
                                                     requested_at=STAMP)
        else:
            with state.store._lock, state.store._conn:
                if kind == "message":
                    state.store._conn.execute("DELETE FROM app_messages WHERE conversation_id=?", (cid,))
                else:
                    state.store._conn.execute("DELETE FROM app_agent_tasks WHERE task_id=?", (source.split(":")[1],))
        return result

    monkeypatch.setattr(KnowledgeSourceBrowser, "page", racing)
    response = await candidates(state, kind=kind)
    assert response.status_code == 409 and "items" not in response.json()


async def test_file_candidate_keyset_can_page_multiple_artifacts_without_repeats(state):
    fid, path = await file(state, fid="file_a")
    path.with_suffix(".summary.txt").write_text("Saved generated summary.", encoding="utf-8")
    video, _ = await file(state, fid="file_b", kind="video", artifact="transcript")
    expected = [f"file:{fid}:summary", f"file:{fid}:text", f"file:{video}:transcript"]
    cursor, seen = None, []
    for _ in range(3):
        response = await candidates(state, kind="file", limit=1, **({"cursor": cursor} if cursor else {}))
        assert response.status_code == 200, response.text
        page = response.json()
        seen.extend(item["source_id"] for item in page["items"])
        next_cursor = page["next_cursor"]
        assert next_cursor != cursor
        cursor = next_cursor
    assert seen == expected and cursor is None


async def test_file_detachment_fences_candidate_and_preview(state, monkeypatch):
    fid, _ = await file(state)
    source = f"file:{fid}:text"
    original = state.reader.resolve

    async def racing(*args):
        document = await original(*args)
        await state.store.projects.remove_file(user_id=state.owners[0].user_id,
                                               project_id=state.project.project_id, file_id=fid)
        return document

    monkeypatch.setattr(state.reader, "resolve", racing)
    assert (await preview(state, source)).status_code == 409
    assert (await candidates(state, kind="file")).json()["items"] == []


@pytest.mark.parametrize("source", ["knowledge:any", "file:../secret:text", "file:x:binary", "message:x:extra",
                                    "agent:x:text", "file:x/secret:text", "file:x\\secret:text"])
async def test_preview_rejects_unsupported_ids_before_reader(state, monkeypatch, source):
    async def forbidden(*_args):
        pytest.fail("Unsupported IDs must not reach the canonical reader")

    monkeypatch.setattr(state.reader, "resolve", forbidden)
    assert (await preview(state, source)).status_code == 422


async def test_reader_failure_never_exposes_paths_or_private_details(state, monkeypatch):
    source, _ = await message(state)

    async def broken(*_args):
        raise OSError("/private/account/path secret text")

    monkeypatch.setattr(state.reader, "resolve", broken)
    response = await preview(state, source)
    assert response.status_code == 503 and "secret" not in response.text and "/private" not in response.text
