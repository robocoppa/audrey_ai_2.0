"""Owner isolation and lifecycle contracts for Audrey-native runs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.pipeline.run_events import RunEventContext, RunFinishedEvent
from audrey.routes.app import router
from audrey.routes.app.runs import (
    NativeRunCursorExpiredError,
    NativeRunManager,
)
from audrey.routes.openai.schemas import ChatCompletionRequest


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str = "owui-alice",
    email: str = "alice@example.com",
) -> Principal:
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


async def _successful_stream(
    _app: Any,
    payload: ChatCompletionRequest,
    _messages: list[dict[str, Any]],
    _options: dict[str, Any],
    *,
    event_context: RunEventContext,
    **_kwargs: Any,
):
    emitter = event_context.emitter
    assert emitter is not None
    emitter.run_started()
    emitter.message_started()
    emitter.stage_started("thinking", label="Thinking")
    emitter.stage_progress("working", stage="thinking")
    emitter.text_delta("native answer")
    emitter.stage_finished("thinking", status="succeeded")
    emitter.usage_reported(prompt_tokens=7, completion_tokens=3)
    emitter.message_finished(status="completed")
    emitter.run_finished(
        status="succeeded",
        finish_reason="stop",
        concrete_model="qwen-test",
    )
    yield f"ignored:{payload.model}"


def _principal_sync(store: ApplicationStore, **kwargs: str) -> Principal:
    return asyncio.run(_resolve(store, **kwargs))


def _native_app(
    tmp_path,
    *,
    stream_factory=_successful_stream,
) -> tuple[FastAPI, ApplicationStore, Principal, NativeRunManager]:
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = _principal_sync(store)
    app = FastAPI()
    app.state.application_store = store
    manager = NativeRunManager(
        app=app,
        store=store,
        stream_factory=stream_factory,
    )
    app.state.native_runs = manager
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    return app, store, owner, manager


def _sse_events(body: str) -> list[dict[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        if not block or block.startswith(":"):
            continue
        lines = block.splitlines()
        event_type = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        assert data["type"] == event_type
        assert int(next(line[4:] for line in lines if line.startswith("id: "))) == data[
            "sequence"
        ]
        events.append(data)
    return events


def test_native_run_create_stream_persist_and_resume_are_canonical(tmp_path):
    app, store, owner, _manager = _native_app(tmp_path)
    conversation = asyncio.run(
        store.conversations.create(
            user_id=owner.user_id,
            title="Native run",
            default_mode="fast",
        )
    )
    try:
        with TestClient(app) as client:
            created = client.post(
                f"/api/conversations/{conversation.conversation_id}/runs",
                json={"content": "Answer natively."},
            )
            assert created.status_code == 202
            run = created.json()
            assert run["status"] == "running"
            assert run["events_url"] == f"/api/runs/{run['id']}/events"
            assert "user_id" not in created.text

            streamed = client.get(run["events_url"])
            assert streamed.status_code == 200
            assert streamed.headers["content-type"].startswith("text/event-stream")
            events = _sse_events(streamed.text)
            assert [event["type"] for event in events] == [
                "run.started",
                "message.started",
                "stage.started",
                "stage.progress",
                "text.delta",
                "stage.finished",
                "usage.reported",
                "message.finished",
                "run.finished",
            ]
            assert [event["sequence"] for event in events] == list(range(1, 10))
            assert owner.user_id not in streamed.text
            assert owner.storage_namespace not in streamed.text

            resumed = client.get(run["events_url"], headers={"Last-Event-ID": "5"})
            assert [event["sequence"] for event in _sse_events(resumed.text)] == [
                6,
                7,
                8,
                9,
            ]
            persisted = client.get(f"/api/runs/{run['id']}")
            assert persisted.status_code == 200
            assert persisted.json()["status"] == "succeeded"
            assert persisted.json()["concrete_model"] == "qwen-test"
            assert persisted.json()["prompt_tokens"] == 7
            assert persisted.json()["completion_tokens"] == 3
            messages = client.get(
                f"/api/conversations/{conversation.conversation_id}/messages"
            ).json()["items"]
            assert [message["content"] for message in messages] == [
                "Answer natively.",
                "native answer",
            ]
            assert messages[-1]["status"] == "completed"
    finally:
        store.close()


def test_native_run_routes_hide_cross_owner_and_reject_archived_or_active(tmp_path):
    async def blocking_stream(
        _app,
        _payload,
        _messages,
        _options,
        *,
        event_context,
        **_kwargs,
    ):
        emitter = event_context.emitter
        assert emitter is not None
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("partial")
        await asyncio.Event().wait()
        yield "unreachable"

    app, store, alice, _manager = _native_app(
        tmp_path,
        stream_factory=blocking_stream,
    )
    bob = _principal_sync(
        store,
        subject="owui-bob",
        email="bob@example.com",
    )
    active = asyncio.run(store.conversations.create(user_id=alice.user_id))
    archived = asyncio.run(store.conversations.create(user_id=alice.user_id))
    asyncio.run(
        store.conversations.update(
            user_id=alice.user_id,
            conversation_id=archived.conversation_id,
            archived=True,
        )
    )
    try:
        with TestClient(app) as client:
            created = client.post(
                f"/api/conversations/{active.conversation_id}/runs",
                json={"content": "Keep running."},
            )
            assert created.status_code == 202
            run_id = created.json()["id"]
            assert client.post(
                f"/api/conversations/{active.conversation_id}/runs",
                json={"content": "Second active run."},
            ).status_code == 409
            assert client.post(
                f"/api/conversations/{archived.conversation_id}/runs",
                json={"content": "Archived run."},
            ).status_code == 409

            app.dependency_overrides[require_principal] = lambda: bob
            assert client.get(f"/api/runs/{run_id}").status_code == 404
            assert client.get(f"/api/runs/{run_id}/events").status_code == 404
            assert client.post(f"/api/runs/{run_id}/cancel").status_code == 404

            app.dependency_overrides[require_principal] = lambda: alice
            cancelled = client.post(f"/api/runs/{run_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert cancelled.json()["error_code"] == "cancelled_by_user"
            events = _sse_events(client.get(f"/api/runs/{run_id}/events").text)
            assert isinstance(events[-1], dict)
            assert events[-1]["type"] == "run.finished"
            assert events[-1]["status"] == "cancelled"
    finally:
        store.close()


async def test_native_manager_keeps_full_answer_beyond_reconnect_window(tmp_path):
    async def long_stream(
        _app,
        _payload,
        _messages,
        _options,
        *,
        event_context,
        **_kwargs,
    ):
        emitter = event_context.emitter
        assert emitter is not None
        emitter.run_started()
        emitter.message_started()
        for _ in range(110):
            emitter.text_delta("x")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "ignored"

    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Long answer",
    )
    assert started is not None
    manager = NativeRunManager(
        app=SimpleNamespace(),
        store=store,
        stream_factory=long_stream,
        max_events_per_run=100,
    )
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=[{"role": "user", "content": "Long answer"}],
        stream=True,
    )
    try:
        await manager.launch(
            principal=owner,
            started=started,
            payload=payload,
            messages=payload.model_dump()["messages"],
            options={},
        )
        await asyncio.sleep(0)
        persisted = await manager.cancel(user_id=owner.user_id, run_id=started.run.run_id)
        assert persisted is not None and persisted.status == "succeeded"
        messages = await store.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert messages is not None
        assert messages[-1].content == "x" * 110
        with pytest.raises(NativeRunCursorExpiredError):
            await manager.open_events(
                user_id=owner.user_id,
                run_id=started.run.run_id,
                after_sequence=0,
            )
    finally:
        await manager.stop()
        store.close()


async def test_native_manager_cancel_persists_partial_answer(tmp_path):
    ready = asyncio.Event()

    async def blocking_stream(
        _app,
        _payload,
        _messages,
        _options,
        *,
        event_context,
        **_kwargs,
    ):
        emitter = event_context.emitter
        assert emitter is not None
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("partial answer")
        ready.set()
        await asyncio.Event().wait()
        yield "unreachable"

    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Cancel this",
    )
    assert started is not None
    manager = NativeRunManager(
        app=SimpleNamespace(),
        store=store,
        stream_factory=blocking_stream,
    )
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=[{"role": "user", "content": "Cancel this"}],
        stream=True,
    )
    try:
        await manager.launch(
            principal=owner,
            started=started,
            payload=payload,
            messages=payload.model_dump()["messages"],
            options={},
        )
        await ready.wait()
        cancelled = await manager.cancel(
            user_id=owner.user_id,
            run_id=started.run.run_id,
        )
        assert cancelled is not None
        assert cancelled.status == "cancelled"
        assert cancelled.error_code == "cancelled_by_user"
        messages = await store.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert messages is not None
        assert messages[-1].status == "incomplete"
        assert messages[-1].content == "partial answer"
        live = await manager.open_events(
            user_id=owner.user_id,
            run_id=started.run.run_id,
            after_sequence=0,
        )
        events = [
            event
            async for event in manager.iter_events(
                live,
                after_sequence=0,
                heartbeat_seconds=0.01,
            )
            if event is not None
        ]
        assert isinstance(events[-1], RunFinishedEvent)
        assert events[-1].status == "cancelled"
    finally:
        await manager.stop()
        store.close()


async def test_native_manager_terminalizes_pipeline_failure(tmp_path):
    async def failing_stream(
        _app,
        _payload,
        _messages,
        _options,
        *,
        event_context,
        **_kwargs,
    ):
        emitter = event_context.emitter
        assert emitter is not None
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("partial before failure")
        yield "ignored"
        raise RuntimeError("private provider detail")

    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store)
    conversation = await store.conversations.create(user_id=owner.user_id)
    started = await store.conversations.begin_run(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        user_content="Trigger failure",
    )
    assert started is not None
    manager = NativeRunManager(
        app=SimpleNamespace(),
        store=store,
        stream_factory=failing_stream,
    )
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=[{"role": "user", "content": "Trigger failure"}],
        stream=True,
    )
    try:
        await manager.launch(
            principal=owner,
            started=started,
            payload=payload,
            messages=payload.model_dump()["messages"],
            options={},
        )
        await asyncio.sleep(0)
        failed = await manager.cancel(user_id=owner.user_id, run_id=started.run.run_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.finish_reason == "error"
        assert failed.error_code == "pipeline_error"
        messages = await store.conversations.list_messages(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert messages is not None
        assert messages[-1].status == "incomplete"
        assert messages[-1].content == "partial before failure"
    finally:
        await manager.stop()
        store.close()
