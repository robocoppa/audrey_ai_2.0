"""Private Project API ownership and protected upload/link boundaries."""
from __future__ import annotations

import asyncio

import pytest
from test_private_document_ingestion import PRIVATE_TEXT, USER

from audrey.app_state import ProjectNotFoundError
from audrey.processing_policy import ProcessingPolicyError
from audrey.routes.app import private_files, projects

pytest_plugins = ("test_native_private_workflow",)


@pytest.fixture(autouse=True)
async def private_project_runtime(native, monkeypatch):
    async def prepare(_app):
        return native.env.runtime
    monkeypatch.setattr(projects, "prepare_native_private_runtime", prepare)


async def _project(s, *, private=True):
    response = await s.client.post("/api/projects", json={
        "name": "Private review" if private else "Ordinary review",
        "model_id": "private" if private else "auto",
    })
    assert response.status_code == 201, response.text
    return response.json()


async def _upload(s, project):
    return await s.client.post(f"/api/projects/{project['id']}/uploads", files={
        "file": ("review.txt", PRIVATE_TEXT, "text/plain"),
    })


async def test_private_project_creation_selection_and_immutable_flag(native):
    s = native
    private = await _project(s)
    ordinary = await _project(s, private=False)
    assert private["private_required"] is True
    assert ordinary["private_required"] is False
    for payload in ({"private_required": False}, {"model_id": "auto"}):
        denied = await s.client.patch(f"/api/projects/{private['id']}", json=payload)
        assert denied.status_code == 422
    stored = await s.client.get(f"/api/projects/{private['id']}")
    assert stored.json()["private_required"] is True
    omitted = await s.client.post(f"/api/projects/{private['id']}/conversations", json={})
    assert omitted.status_code == 201, omitted.text
    assert omitted.json()["private_required"] is True
    assert omitted.json()["default_model_id"] == "private"
    for model in ("auto", "cloud", "direct/not-local"):
        denied = await s.client.post(f"/api/projects/{private['id']}/conversations", json={"model_id": model})
        assert denied.status_code == 409
    denied = await s.client.post(f"/api/projects/{ordinary['id']}/conversations", json={"model_id": "private"})
    assert denied.status_code == 409
    assert not s.env.seen  # Creation/reads never generate or embed.


@pytest.mark.parametrize("case", ["disabled", "role"])
async def test_private_project_creation_requires_enabled_role_permitted_model(native, case):
    s = native
    if case == "disabled":
        s.app.state.cfg.raw["private_processing"]["enabled"] = False
    else:
        admin = await s.store.resolve_external_identity(provider="owui", subject="project-admin", email="admin@example.com", display_name="Admin", role="admin", auth_method="owui_bearer")
        await s.store.set_model_access_policy(actor_user_id=admin.user_id, model_id="private", enabled=True, audience="testers")
    denied = await s.client.post("/api/projects", json={"name": "Not admitted", "model_id": "private"})
    assert denied.status_code == 404
    assert await s.store.projects.list_page(user_id=s.owner.user_id, limit=10) == ()
    assert not s.env.seen


async def test_private_project_upload_links_private_owner_document_and_lists_it(native):
    s = native
    project = await _project(s)
    response = await _upload(s, project)
    assert response.status_code == 200, response.text
    fid = response.json()["id"]
    row = await s.env.db.get_upload(fid)
    assert row["user"] == USER and row["private_required"]
    assert row["status"] == "ready" and row["collection"].startswith("kb_private_text_")
    links = await s.store.projects.list_files(user_id=s.owner.user_id, project_id=project["id"])
    assert [item.file_id for item in links] == [fid]
    listed = await s.client.get(f"/api/projects/{project['id']}/files")
    assert listed.status_code == 200 and [item["id"] for item in listed.json()["items"]] == [fid]
    assert {request.url.host for request in s.env.seen} == {"ollama"}
    ordinary = await _project(s, private=False)
    denied = await s.client.post(f"/api/projects/{ordinary['id']}/files", json={"file_id": fid})
    assert denied.status_code == 409
    second = await _project(s)
    added = await s.client.post(f"/api/projects/{second['id']}/files", json={"file_id": fid})
    assert added.status_code == 201, added.text


@pytest.mark.parametrize("case", ["foreign", "ordinary", "media", "outage"])
async def test_private_project_upload_denies_before_storage(native, monkeypatch, case):
    s = native
    project = await _project(s, private=case != "ordinary")
    if case == "foreign":
        other = await s.store.resolve_external_identity(provider="owui", subject="other-project-owner", email="other@example.com", display_name="Other", role="user", auth_method="owui_bearer")
        project = {"id": (await s.store.projects.create(user_id=other.user_id, name="Other private", private_required=True)).project_id}
    if case == "outage":
        async def fail(_app):
            raise ProcessingPolicyError("private_model_unavailable")
        monkeypatch.setattr(private_files, "prepare_native_private_runtime", fail)
    response = await s.client.post(f"/api/projects/{project['id']}/uploads", files={"file": ("x.mp4" if case == "media" else "x.txt", PRIVATE_TEXT)})
    assert response.status_code == {"foreign": 404, "ordinary": 409, "media": 422, "outage": 503}[case]
    assert await s.env.db.list_user(USER) == []
    assert not s.env.seen
    s.env.qdrant.upsert_text.assert_not_awaited()


async def test_committed_private_upload_is_retained_if_project_disappears(native, monkeypatch):
    s = native
    project = await _project(s)
    async def gone(**_kwargs):
        raise ProjectNotFoundError("gone")
    monkeypatch.setattr(s.store.projects, "add_file", gone)
    response = await _upload(s, project)
    assert response.status_code == 409, response.text
    fid = response.json()["detail"]["file_id"]
    row = await s.env.db.get_upload(fid)
    assert row and row["private_required"] and row["status"] == "ready"
    assert "remains in My Files" in response.json()["detail"]["message"]


async def test_cancelled_project_upload_waits_for_started_link(native, monkeypatch):
    s = native
    project = await _project(s)
    entered, release = asyncio.Event(), asyncio.Event()
    original = s.store.projects.add_file
    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)
    monkeypatch.setattr(s.store.projects, "add_file", blocked)
    pending = asyncio.create_task(_upload(s, project))
    await asyncio.wait_for(entered.wait(), 2)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    links = await s.store.projects.list_files(user_id=s.owner.user_id, project_id=project["id"])
    assert len(links) == 1
    row = await s.env.db.get_upload(links[0].file_id)
    assert row and row["status"] == "ready" and row["private_required"]


async def test_private_project_links_refuse_ordinary_or_foreign_uploads(native):
    s = native
    project = await _project(s)
    await s.env.db.record_upload(file_id="ordinary", user=USER, filename="ordinary.txt", mime="text/plain", bytes_=12, kind="text", collection="ordinary", chunks=1, uploaded_at="2026-10-20T00:00:00+00:00")
    await s.env.db.record_upload(file_id="foreign", user="another-owner", filename="foreign.txt", mime="text/plain", bytes_=12, kind="text", collection="kb_private_text_another-owner", chunks=1, uploaded_at="2026-10-20T00:00:00+00:00")
    ordinary = await s.client.post(f"/api/projects/{project['id']}/files", json={"file_id": "ordinary"})
    foreign = await s.client.post(f"/api/projects/{project['id']}/files", json={"file_id": "foreign"})
    assert ordinary.status_code == 409
    assert foreign.status_code == 404
    assert await s.store.projects.list_files(user_id=s.owner.user_id, project_id=project["id"]) == ()
    assert not s.env.seen


async def test_private_project_conversation_upload_remains_a_chat_attachment(native):
    s = native
    project = await _project(s)
    conversation = await s.client.post(f"/api/projects/{project['id']}/conversations", json={})
    cid = conversation.json()["id"]
    response = await s.client.post(f"/api/conversations/{cid}/files", files={"file": ("review.txt", PRIVATE_TEXT, "text/plain")})
    assert response.status_code == 200, response.text
    links = await s.store.projects.list_files(user_id=s.owner.user_id, project_id=project["id"])
    assert links == ()  # Chat attachments do not silently enter shared Project context.
    row = await s.env.db.get_upload(response.json()["id"])
    assert row and row["private_required"] and row["status"] == "ready"
