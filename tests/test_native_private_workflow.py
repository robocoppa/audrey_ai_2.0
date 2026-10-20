"""Native Private selection, document upload and future-reuse boundaries."""

from __future__ import annotations

import asyncio
import json
import stat
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from test_private_document_ingestion import CHAT, EMBED, PRIVATE_TEXT, USER
from test_private_document_ingestion import setup as document_setup  # noqa: F401

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.model_catalog import configured_models
from audrey.private_processing import prepare_native_private_runtime
from audrey.processing_policy import ProcessingPolicyError
from audrey.routes.app import private_files, router, runs
from audrey.routes.app.runs import NativeRunManager


@pytest.fixture
async def native(document_setup, tmp_path, monkeypatch):  # noqa: F811 - imported shared pytest fixture
    env = document_setup
    cfg = env.app.state.cfg
    cfg.raw["private_processing"] = {"enabled": True}
    cfg.raw["deep_panel_local"] = {
        "general": {"workers": [CHAT], "synthesizer": CHAT, "fallback_synth": CHAT}
    }
    cfg.raw["router"] = {"model": CHAT}
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="owui",
        subject="private-native-owner",
        email="owner@example.com",
        display_name="Owner",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=USER,
    )
    app = FastAPI()
    for name, value in vars(env.app.state).items():
        setattr(app.state, name, value)
    app.state.application_store = store
    app.state.storage_lifecycle = env.storage
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    captured = []

    async def prepare(_app):
        return env.runtime

    monkeypatch.setattr(runs, "prepare_native_private_runtime", prepare)
    monkeypatch.setattr(private_files, "prepare_native_private_runtime", prepare)

    async def stream(execution_app, _payload, messages, _options, *, event_context, **_kw):
        captured.append(messages)
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        result = await execution_app.state.ollama.chat(model=CHAT, messages=messages)
        emitter.text_delta(result["message"]["content"])
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "terminal"

    manager = NativeRunManager(app=app, store=store, stream_factory=stream)
    app.state.native_runs = manager
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            owner=owner,
            store=store,
            manager=manager,
            captured=captured,
            env=env,
        )
    await manager.stop()
    store.close()


async def _conversation(state, model="private"):
    response = await state.client.post("/api/conversations", json={"model_id": model})
    assert response.status_code == 201, response.text
    return response.json()


async def _settle(state, run_id):
    live = state.manager._runs[run_id]
    await asyncio.wait_for(live.task, 3)
    titles = tuple(state.manager._title_tasks.values())
    if titles:
        await asyncio.wait_for(asyncio.gather(*titles), 3)


async def test_private_document_chat_roundtrip_stays_local_and_out_of_ordinary_reuse(native):
    s = native
    conversation = await _conversation(s)
    assert (
        conversation["private_required"] is True and conversation["default_model_id"] == "private"
    )
    upload = await s.client.post(
        f"/api/conversations/{conversation['id']}/files",
        files={"file": ("private-test.txt", PRIVATE_TEXT, "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    row = await s.env.db.get_upload(file_id)
    assert row["private_required"] and row["collection"].startswith("kb_private_text_")
    source = s.env.app.state.cfg.raw["kb"]["upload_root"]
    from pathlib import Path

    from audrey.kb.user_store import sanitize_user

    path = Path(source) / sanitize_user(USER) / f"{file_id}.txt"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    listed = await s.client.get("/api/files")
    assert listed.status_code == 200, listed.text
    assert (
        next(item for item in listed.json()["items"] if item["id"] == file_id)["private_required"]
        is True
    )
    run = await s.client.post(
        f"/api/conversations/{conversation['id']}/runs",
        json={"content": "Summarize my document", "attachment_ids": [file_id]},
    )
    assert run.status_code == 202, run.text
    await _settle(s, run.json()["id"])
    assert PRIVATE_TEXT in json.dumps(s.captured[-1])
    saved = await s.client.get(f"/api/conversations/{conversation['id']}/messages")
    assert [message["role"] for message in saved.json()["items"]] == ["user", "assistant"]
    assert saved.json()["items"][-1]["content"]
    assert await s.store.chat_projections.due() == ()
    assert all(request.url.host == "ollama" for request in s.env.seen)
    ordinary = await _conversation(s, "auto")
    before = len(s.captured)
    denied = await s.client.post(
        f"/api/conversations/{ordinary['id']}/runs",
        json={"content": "Use this file", "attachment_ids": [file_id]},
    )
    assert denied.status_code == 409, denied.text
    assert len(s.captured) == before
    switched = await s.client.patch(
        f"/api/conversations/{conversation['id']}", json={"model_id": "cloud"}
    )
    assert switched.status_code == 409
    reloaded = await s.client.get(f"/api/conversations/{conversation['id']}")
    assert reloaded.json()["private_required"] and reloaded.json()["default_model_id"] == "private"


async def test_private_selection_upgrade_is_atomic_and_refuses_old_conversations(native):
    s = native
    empty = await _conversation(s, "auto")
    selected = await s.client.patch(
        f"/api/conversations/{empty['id']}", json={"model_id": "private"}
    )
    assert selected.status_code == 200 and selected.json()["private_required"]
    old = await _conversation(s, "auto")
    started = await s.store.conversations.begin_run(
        user_id=s.owner.user_id, conversation_id=old["id"], user_content="Already processed"
    )
    await s.store.conversations.finish_run(
        user_id=s.owner.user_id,
        run_id=started.run.run_id,
        outcome="succeeded",
        assistant_content="Ordinary answer",
    )
    selected = await s.client.patch(f"/api/conversations/{old['id']}", json={"model_id": "private"})
    assert selected.status_code == 422 and "new conversation" in selected.json()["detail"]
    row = await s.store.conversations.get(user_id=s.owner.user_id, conversation_id=old["id"])
    assert not row.private_required and row.default_model_id == "auto"


@pytest.mark.parametrize("case", ["foreign", "ordinary", "media", "project", "outage"])
async def test_private_upload_refuses_before_saving_or_processing_for_invalid_context(
    native, monkeypatch, case
):
    s = native
    c = await _conversation(s, "auto" if case == "ordinary" else "private")
    cid = "con_unknown" if case == "foreign" else c["id"]
    if case == "project":
        project = await s.store.projects.create(user_id=s.owner.user_id, name="Existing")
        response = await s.client.patch(
            f"/api/conversations/{cid}", json={"project_id": project.project_id}
        )
        assert response.status_code == 409
        return
    if case == "outage":

        async def fail(_app):
            raise ProcessingPolicyError("private_model_unavailable")

        monkeypatch.setattr(private_files, "prepare_native_private_runtime", fail)
    filename = "video.mp4" if case == "media" else "doc.txt"
    response = await s.client.post(
        f"/api/conversations/{cid}/files", files={"file": (filename, PRIVATE_TEXT)}
    )
    assert (
        response.status_code == {"foreign": 404, "ordinary": 409, "media": 422, "outage": 503}[case]
    ), response.text
    assert await s.env.db.list_user(USER) == []
    s.env.qdrant.upsert_text.assert_not_awaited()
    assert s.env.seen == []


async def test_private_run_outage_does_not_save_message_or_try_cloud(native, monkeypatch):
    s = native
    c = await _conversation(s)

    async def fail(_app):
        raise ProcessingPolicyError("private_model_unavailable")

    monkeypatch.setattr(runs, "prepare_native_private_runtime", fail)
    response = await s.client.post(
        f"/api/conversations/{c['id']}/runs", json={"content": PRIVATE_TEXT}
    )
    assert response.status_code == 503
    assert (await s.client.get(f"/api/conversations/{c['id']}/messages")).json()["items"] == []
    assert s.captured == [] and s.env.seen == []


async def test_native_private_roster_comes_from_local_deployment_pools_and_profile(
    native, monkeypatch
):
    s = native
    captured = {}

    async def prepare(**kwargs):
        captured.update(kwargs)
        return s.env.runtime.policy

    monkeypatch.setattr("audrey.private_processing.prepare_processing_policy", prepare)
    await prepare_native_private_runtime(s.app)
    assert set(captured["model_names"]) == {CHAT, EMBED}
    assert captured["expected_digests"] == {EMBED: "b" * 64}
    s.app.state.cfg.raw["private_processing"]["enabled"] = False
    assert not any(model.id == "private" for model in configured_models(s.app.state.cfg))
    with pytest.raises(ProcessingPolicyError):
        await prepare_native_private_runtime(s.app)


@pytest.mark.parametrize("transport", ["native", "agui"])
@pytest.mark.parametrize("stored_model,requested_model", [
    ("auto", "private"), ("auto", " private "), ("private", None), ("private", "cloud"),
])
async def test_private_model_requires_durable_conversation_binding_before_any_work(
    native, monkeypatch, transport, stored_model, requested_model,
):
    s = native
    conversation = await s.store.conversations.create(
        user_id=s.owner.user_id, default_model_id=stored_model,
        default_mode="local" if stored_model == "private" else "auto",
    )
    assert not conversation.private_required

    async def forbidden(*args, **kwargs):
        pytest.fail("Private model without durable binding must refuse before preparing content")

    for name in (
        "resolve_model", "resolve_owned_attachments", "resolve_project_context",
        "resolve_native_auto_skill", "_image_parts_for_run", "prepare_native_private_runtime",
    ):
        monkeypatch.setattr(runs, name, forbidden)
    if transport == "native":
        payload = {"content": PRIVATE_TEXT}
        if requested_model is not None:
            payload["model_id"] = requested_model
        response = await s.client.post(
            f"/api/conversations/{conversation.conversation_id}/runs", json=payload,
        )
    else:
        response = await s.client.post(
            "/api/agent", params={"model": requested_model} if requested_model else None,
            json={
                "threadId": conversation.conversation_id, "runId": "client-private-unbound",
                "messages": [{"id": "client-user", "role": "user", "content": PRIVATE_TEXT}],
            },
        )
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Select Private in a new or empty conversation before sending a message."
    )
    assert PRIVATE_TEXT not in response.text
    assert await s.store.conversations.list_messages(
        user_id=s.owner.user_id, conversation_id=conversation.conversation_id,
    ) == ()
    saved = await s.store.conversations.get(
        user_id=s.owner.user_id, conversation_id=conversation.conversation_id,
    )
    assert saved.default_model_id == stored_model and not saved.private_required
    assert s.captured == [] and s.env.seen == []
