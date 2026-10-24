"""Changing an unused Private selection replaces its draft without relabelling data."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from test_native_private_workflow import _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native
from test_private_document_ingestion import PRIVATE_TEXT

from audrey.processing_policy import processing_scope
from audrey.routes.app import private_files


async def _draft(s, *, project_id=None):
    payload = {"model_id": "private", "title": "Unsent draft"}
    if project_id is not None:
        payload["project_id"] = project_id
    response = await s.client.post("/api/conversations", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


async def _switch(s, draft, **extra):
    return await s.client.patch(
        f"/api/conversations/{draft['id']}", json={"model_id": "cloud", **extra},
    )


async def test_unused_private_model_can_change_repeatedly_with_new_draft_identity(native):
    s = native
    first = await _draft(s)
    response = await _switch(s, first)
    assert response.status_code == 200, response.text
    ordinary = response.json()
    assert ordinary["id"] != first["id"]
    assert ordinary["private_required"] is False
    assert ordinary["default_model_id"] == "cloud" and ordinary["title"] == first["title"]
    assert (await s.client.get(f"/api/conversations/{first['id']}")).status_code == 404
    again = await s.client.patch(
        f"/api/conversations/{ordinary['id']}", json={"model_id": "private"},
    )
    assert again.status_code == 200 and again.json()["id"] == ordinary["id"]
    final = await _switch(s, again.json())
    assert final.status_code == 200 and final.json()["id"] != ordinary["id"]
    rows = await s.store.conversations.list_for_user(user_id=s.owner.user_id)
    assert len(rows) == 1 and rows[0].conversation_id == final.json()["id"]
    assert await s.store.chat_projections.due_deletions() == ()
    assert not s.env.seen  # Switching unused selections does not call any model.


async def test_replacing_draft_keeps_previously_uploaded_document_protected(native):
    s = native
    draft = await _draft(s)
    uploaded = await s.client.post(
        f"/api/conversations/{draft['id']}/files",
        files={"file": ("private.txt", PRIVATE_TEXT, "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    switched = await _switch(s, draft)
    assert switched.status_code == 200, switched.text
    retained = await s.env.db.get_upload(file_id)
    assert retained["private_required"] and retained["status"] == "ready"
    denied = await s.client.post(
        f"/api/conversations/{switched.json()['id']}/runs",
        json={"content": "Use this file", "attachment_ids": [file_id]},
    )
    assert denied.status_code == 409 and s.captured == []


async def test_upload_already_in_progress_cannot_land_as_ordinary_after_draft_switch(native, monkeypatch):
    s = native
    draft = await _draft(s)
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(_app):
        entered.set()
        await release.wait()
        return s.env.runtime

    monkeypatch.setattr(private_files, "prepare_native_private_runtime", prepare)
    pending = asyncio.create_task(s.client.post(
        f"/api/conversations/{draft['id']}/files",
        files={"file": ("private.txt", PRIVATE_TEXT, "text/plain")},
    ))
    await asyncio.wait_for(entered.wait(), 2)
    switched = await _switch(s, draft)
    assert switched.status_code == 200, switched.text
    release.set()
    uploaded = await asyncio.wait_for(pending, 3)
    assert uploaded.status_code == 200, uploaded.text
    row = await s.env.db.get_upload(uploaded.json()["id"])
    assert row["private_required"] and row["collection"].startswith("kb_private_text_")
    assert not switched.json()["private_required"]


@pytest.mark.parametrize("outcome", [None, "succeeded", "failed"])
async def test_first_canonical_message_locks_private_even_when_run_fails(native, outcome):
    s = native
    draft = await _draft(s)
    with processing_scope(s.env.runtime.policy):
        started = await s.store.conversations.begin_run(
            user_id=s.owner.user_id, conversation_id=draft["id"],
            user_content="Accepted first message", expected_private_required=True,
        )
    if outcome:
        await s.store.conversations.finish_run(
            user_id=s.owner.user_id, run_id=started.run.run_id,
            outcome=outcome, assistant_content="" if outcome == "failed" else "Saved answer",
        )
    response = await _switch(s, draft)
    assert response.status_code == 409, response.text
    retained = (await s.client.get(f"/api/conversations/{draft['id']}")).json()
    assert retained["private_required"] and retained["default_model_id"] == "private"
    assert len(await s.store.conversations.list_for_user(user_id=s.owner.user_id)) == 1


async def test_private_run_record_without_messages_still_prevents_replacement(native):
    s = native
    draft = await _draft(s)
    with processing_scope(s.env.runtime.policy):
        started = await s.store.conversations.begin_run(
            user_id=s.owner.user_id, conversation_id=draft["id"],
            user_content="Accepted first message", expected_private_required=True,
        )
    await s.store.conversations.finish_run(
        user_id=s.owner.user_id, run_id=started.run.run_id, outcome="failed", assistant_content="",
    )
    # A retained run is enough to forbid relabelling even if messages were repaired away.
    with sqlite3.connect(s.store.path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("DELETE FROM app_messages WHERE conversation_id=?", (draft["id"],))
    response = await _switch(s, draft)
    assert response.status_code == 409
    assert (await s.client.get(f"/api/conversations/{draft['id']}")).json()["private_required"]


async def test_private_project_requires_explicit_detachment_before_ordinary_draft(native):
    s = native
    project = await s.store.projects.create(user_id=s.owner.user_id, name="Protected", private_required=True)
    draft = await _draft(s, project_id=project.project_id)
    denied = await _switch(s, draft)
    assert denied.status_code == 409 and "Project" in denied.json()["detail"]
    allowed = await _switch(s, draft, project_id=None)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["project_id"] is None and not allowed.json()["private_required"]
    retained = await s.store.projects.get(user_id=s.owner.user_id, project_id=project.project_id)
    assert retained.private_required


async def test_run_start_racing_model_change_is_checked_inside_replacement_transaction(native, monkeypatch):
    s = native
    draft = await _draft(s)
    original_update = s.store.conversations.update

    async def start_before_update(**kwargs):
        with processing_scope(s.env.runtime.policy):
            started = await s.store.conversations.begin_run(
                user_id=s.owner.user_id, conversation_id=draft["id"],
                user_content="Saved while the selection request was preparing", expected_private_required=True,
            )
        await s.store.conversations.finish_run(
            user_id=s.owner.user_id, run_id=started.run.run_id,
            outcome="failed", assistant_content="",
        )
        return await original_update(**kwargs)

    monkeypatch.setattr(s.store.conversations, "update", start_before_update)
    denied = await _switch(s, draft)
    assert denied.status_code == 409
    messages = (await s.client.get(f"/api/conversations/{draft['id']}/messages")).json()["items"]
    assert messages[0]["content"].startswith("Saved while")
    assert (await s.client.get(f"/api/conversations/{draft['id']}")).json()["private_required"]


async def test_delete_populated_private_conversation_preserves_protected_upload(native):
    s = native
    draft = await _draft(s)
    uploaded = await s.client.post(
        f"/api/conversations/{draft['id']}/files",
        files={"file": ("private.txt", PRIVATE_TEXT, "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    started = await s.client.post(f"/api/conversations/{draft['id']}/runs", json={
        "content": "Read my document", "attachment_ids": [file_id],
    })
    assert started.status_code == 202, started.text
    await _settle(s, started.json()["id"])
    deleted = await s.client.delete(f"/api/conversations/{draft['id']}")
    assert deleted.status_code == 204, deleted.text
    assert (await s.client.get(f"/api/conversations/{draft['id']}")).status_code == 404
    assert (await s.client.get(f"/api/conversations/{draft['id']}/messages")).status_code == 404
    assert await s.store.conversations.get_run(user_id=s.owner.user_id, run_id=started.json()["id"]) is None
    row = await s.env.db.get_upload(file_id)
    assert row["private_required"] and row["status"] == "ready"
