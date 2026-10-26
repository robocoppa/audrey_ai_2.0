"""Native generated-file callbacks never escape their owner, run or policy."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from test_native_private_reviews import saved_case, start_review, task_status
from test_native_private_workflow import _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native

from audrey.identity import Principal
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.discovery import ToolRegistry
from audrey.tools.dispatch import dispatch_one
from audrey.tools.native_files import (
    NativeFileToolUnavailableError,
    append_generated_file_links,
    native_file_handler,
    native_file_scope,
    with_create_file,
)
from audrey.tools.private_dispatch import private_tool_scope
from audrey.tools.private_handlers import make_private_tool_handlers

OWNER = Principal("usr_owner", "owned-storage", "cloudflare_access", "subject", "owner@example.com",
                  "Owner", "user", "active", "cloudflare_access")
SERVER = "http://custom-tools:8001"
OUTPUT = {"file_id": "file_123", "filename": "notes.md", "created": True,
          "download_url": "/api/files/file_123/download", "project_id": None}


def private_policy():
    return ProcessingPolicy({"local:latest": "a" * 64}, "http://ollama:11434",
                            allowed_tools=frozenset({"create_file"}), tool_servers=frozenset({SERVER}))


def no_network(request):
    pytest.fail("Native file content reached an HTTP transport")


async def dispatch(client, *, user=OWNER.storage_namespace, arguments=None, server="http://native.invalid"):
    return await dispatch_one(client, with_create_file(ToolRegistry(), server_url=server), {
        "function": {"name": "create_file", "arguments": arguments or {"filename": "notes.md", "format": "md", "content": "Hello"}},
    }, max_result_chars=2000, timeout_s=2, user_id=user)


async def test_missing_native_scope_refuses_before_arguments_are_logged(caplog):
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        result = await dispatch(client, arguments="PRIVATE_CONTENT_NOT_JSON")
    assert result.is_error and "PRIVATE_CONTENT" not in result.content + caplog.text


async def test_creation_uses_server_bound_owner_run_and_collects_result(monkeypatch):
    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    app, files = object(), []
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with native_file_scope(app=app, principal=OWNER, conversation_id="con_owned", run_id="run_owned", files=files):
            result = await dispatch(client)
            retry = await dispatch(client)
    assert not result.is_error and not retry.is_error
    assert files == [OUTPUT]
    assert service.await_args.args[:4] == (app, OWNER, "con_owned", "run_owned")
    assert service.await_args.args[4]["filename"] == "notes.md"


@pytest.mark.parametrize("user", [None, "", "someone-else"])
async def test_another_owner_cannot_use_the_binding(user, monkeypatch):
    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with native_file_scope(app=object(), principal=OWNER, conversation_id="con_owned", run_id="run_owned", files=[]):
            result = await dispatch(client, user=user, arguments="HIDDEN_INVALID_JSON")
    assert result.is_error and "HIDDEN" not in result.content
    service.assert_not_awaited()


async def test_inherited_callback_is_revoked_when_owning_scope_exits(monkeypatch):
    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    release = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        async def later():
            await release.wait()
            return await dispatch(client)
        with native_file_scope(app=object(), principal=OWNER, conversation_id="con_owned", run_id="run_owned", files=[]):
            task = asyncio.create_task(later())
            captured = native_file_handler(user_id=OWNER.storage_namespace)
        release.set()
        assert (await task).is_error
        with pytest.raises(NativeFileToolUnavailableError):
            await captured({})
    service.assert_not_awaited()


async def test_private_creation_retains_existing_callback_gate(monkeypatch):
    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    files, app = [], object()
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with processing_scope(private_policy()), native_file_scope(
            app=app, principal=OWNER, conversation_id="con_private", run_id="run_private", files=files,
        ):
            # No bypass: Private still needs its independently bound handler.
            assert (await dispatch(client, server=SERVER)).is_error
            handlers = make_private_tool_handlers(app, OWNER)
            with private_tool_scope(user_id=OWNER.storage_namespace, handlers=handlers):
                result = await dispatch(client, server=SERVER)
    assert not result.is_error and files == [OUTPUT]


async def test_creation_failure_does_not_leak_content_or_create_download(monkeypatch, caplog):
    service = AsyncMock(side_effect=RuntimeError("PRIVATE_DOCUMENT_CONTENT"))
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    files = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with native_file_scope(app=object(), principal=OWNER, conversation_id="con_owned", run_id="run_owned", files=files):
            result = await dispatch(client)
    assert result.is_error and files == []
    assert "PRIVATE_DOCUMENT" not in result.content + caplog.text


async def test_concurrent_calls_obey_per_run_file_cap(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def service(*args):
        nonlocal count
        count += 1
        if count == 8:
            entered.set()
        number = count
        await release.wait()
        return {**OUTPUT, "file_id": f"file_{number}", "download_url": f"/api/files/file_{number}/download"}

    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)
    files = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with native_file_scope(app=object(), principal=OWNER, conversation_id="con_owned", run_id="run_owned", files=files):
            tasks = [asyncio.create_task(dispatch(client)) for _ in range(8)]
            await asyncio.wait_for(entered.wait(), 2)
            ninth = await dispatch(client)
            assert json.loads(ninth.content)["error"] == "generated_file_limit"
            release.set()
            assert all(not item.is_error for item in await asyncio.gather(*tasks))
    assert count == 8 and len(files) == 8


def test_approved_links_are_canonical_bounded_and_retry_safe():
    text = append_generated_file_links("Result", [OUTPUT, OUTPUT])
    assert text.count(OUTPUT["download_url"]) == 1
    assert append_generated_file_links(text, [OUTPUT]) == text
    long = append_generated_file_links("x" * 64000, [OUTPUT])
    assert len(long) <= 64000 and long.endswith(f"]({OUTPUT['download_url']})")
    assert append_generated_file_links("No output", [{**OUTPUT, "download_url": "https://bad.example"}]) == "No output"


def test_scoped_registry_does_not_mutate_shared_discovery():
    shared = ToolRegistry()
    scoped = with_create_file(shared)
    assert shared.names() == [] and scoped.names() == ["create_file"]
    assert "$ref" not in json.dumps(scoped.get("create_file").parameters)


def test_ordinary_chat_always_saves_a_server_file_link_even_if_answer_omits_it(tmp_path, monkeypatch):
    from test_native_runs import _native_app

    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)

    async def stream(app, payload, messages, options, *, model_tools, event_context, user_id, **kwargs):
        assert "create_file" in model_tools.names()
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
            result = await dispatch_one(client, model_tools, {"function": {"name": "create_file", "arguments": {
                "filename": "notes.md", "format": "md", "content": "Hello",
            }}}, max_result_chars=2000, timeout_s=2, user_id=user_id)
        assert not result.is_error, result.content
        emitter.text_delta("Created your notes.")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "terminal"

    app, store, owner, manager = _native_app(tmp_path, stream_factory=stream)
    conversation = asyncio.run(store.conversations.create(user_id=owner.user_id))
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/conversations/{conversation.conversation_id}/runs", json={"content": "Create notes."})
            assert response.status_code == 202, response.text
            run = response.json()
            assert client.get(run["events_url"]).status_code == 200
            messages = client.get(f"/api/conversations/{conversation.conversation_id}/messages").json()["items"]
            assert messages[-1]["content"].startswith("Created your notes.")
            assert OUTPUT["download_url"] in messages[-1]["content"]
    finally:
        asyncio.run(manager.stop())
        store.close()


async def test_private_task_creation_is_charged_and_link_commits_with_delivery(native, monkeypatch):
    state = native
    conversation, _case = await saved_case(state)
    state.env.runtime = replace(state.env.runtime, policy=replace(
        state.env.runtime.policy, allowed_tools=frozenset({"tax_review", "create_file"}),
    ))
    service = AsyncMock(return_value=OUTPUT)
    monkeypatch.setattr("audrey.generated_files.create_generated_file", service)

    async def stream(app, payload, messages, options, *, model_tools, event_context, user_id, **kwargs):
        assert "create_file" in model_tools.names()
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        result = await dispatch_one(app.state.archive_http, model_tools, {
            "function": {"name": "create_file", "arguments": {
                "filename": "notes.md", "format": "md", "content": "Working notes, not a checked tax case.",
            }},
        }, max_result_chars=2000, timeout_s=2, user_id=user_id)
        assert not result.is_error, result.content
        emitter.text_delta("Created working notes.")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "terminal"

    state.manager._stream_factory = stream
    run = await start_review(state, conversation["id"], "Create working notes as a Markdown file.")
    await _settle(state, run["id"])
    task = await task_status(state, conversation["id"])
    assert task["status"] == "paused" and task["usage"]["tool_calls"] == 1
    messages = await state.store.conversations.list_messages(user_id=state.owner.user_id, conversation_id=conversation["id"])
    assert OUTPUT["download_url"] in messages[-1].content
    assert "Worksheet not saved" in messages[-1].content
    assert service.await_args.args[3] == run["id"]


def test_plain_or_off_origin_mentions_do_not_suppress_canonical_download():
    for content in ("Saved at " + OUTPUT["download_url"],
                    "[Download](https://other.example" + OUTPUT["download_url"] + ")"):
        assert "[Download notes.md](/api/files/file_123/download)" in append_generated_file_links(content, [OUTPUT])
