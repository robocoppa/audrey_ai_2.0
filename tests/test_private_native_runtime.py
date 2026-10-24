"""Native task lifetime, safe transports and durable restriction integration."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.config import Config, EnvOverrides
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.private_processing import PrivateProcessingRuntime
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    current_processing_policy,
    processing_scope,
)
from audrey.routes.app.runs import NativeRunManager
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.tools.calculator import CalculateRequest, calculate_value
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.private_dispatch import private_tool_scope

MODEL = "local-test:latest"
PRIVATE_TEXT = "PRIVATE_NATIVE_CONTENT_6385"
POLICY = ProcessingPolicy(
    local_models={MODEL: "a" * 64},
    ollama_base_url="http://ollama:11434",
    allowed_tools=frozenset({"calculate"}),
    tool_servers=frozenset({"http://custom-tools:8001"}),
)


def _app():
    cfg = Config(
        {
            "router": {"model": MODEL},
            "title_generation": {"model": MODEL, "no_thinking": False},
            "model_registry": {"general": [{"name": MODEL, "location": "local"}]},
        },
        EnvOverrides(_env_file=None),
    )
    app = FastAPI()
    app.state.cfg = cfg
    app.state.registry = ModelRegistry(cfg)
    app.state.health = HealthTracker()
    app.state.gate = FairLocalGate(concurrency=1)
    app.state.inflight = UserInflightRegistry(max_inflight_per_user=3)
    app.state.ollama = object()
    app.state.graph = object()
    app.state.archive_http = object()
    app.state.archive_client = object()
    app.state.conversation_titles = object()
    app.state.tools = ToolRegistry(
        by_name={
            name: ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                server_url="http://custom-tools:8001",
                path=f"/{name}",
            )
            for name in ("calculate", "web_search", "memory_search")
        }
    )
    return app


async def _started(store, *, private=True):
    owner = await store.resolve_external_identity(
        provider="owui",
        subject="native-private-test",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace="alice@example.com",
    )
    conversation = await store.conversations.create(
        user_id=owner.user_id,
        private_required=private,
    )
    with processing_scope(POLICY if private else None):
        started = await store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content=PRIVATE_TEXT,
            expected_private_required=private,
        )
    assert started is not None
    return owner, started


def _handler(seen, *, digest=None):
    def handle(request):
        seen.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": MODEL,
                            "model": MODEL,
                            "digest": digest() if digest else "a" * 64,
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "details": {"format": "gguf"},
                    "capabilities": [],
                    "model_info": {"general.architecture": "qwen", "general.parameter_count": 100},
                },
            )
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        response = {"done": True, "message": {"content": "Local Private Summary"}}
        return (
            httpx.Response(
                200,
                content=json.dumps(response) + "\n",
                headers={"content-type": "application/json"},
            )
            if body.get("stream")
            else httpx.Response(200, json=response)
        )

    return handle


def _payload():
    return ChatCompletionRequest(
        model="audrey_fast",
        messages=[{"role": "user", "content": PRIVATE_TEXT}],
        stream=True,
    )


async def _launch(manager, owner, started, **kwargs):
    payload = _payload()
    await manager.launch(
        principal=owner,
        started=started,
        payload=payload,
        messages=payload.model_dump()["messages"],
        options={},
        **kwargs,
    )
    return manager._runs[started.run.run_id]


async def test_private_run_and_later_title_keep_scope_without_mutating_ordinary_app(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, started = await _started(store)
    app, requests, observed = _app(), [], []
    ordinary_ollama, ordinary_graph = app.state.ollama, app.state.graph
    runtime = PrivateProcessingRuntime(
        POLICY, ollama_transport=httpx.MockTransport(_handler(requests))
    )

    async def stream(
        execution_app, _payload, messages, _options, *, event_context, model_tools, **_kw
    ):
        assert execution_app.state.ollama is not ordinary_ollama
        assert execution_app.state.archive_client is None
        assert model_tools.names() == ["calculate"]
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        try:
            for _ in range(2):
                await asyncio.sleep(0)
                child_policy = await asyncio.create_task(
                    asyncio.to_thread(current_processing_policy)
                )
                assert child_policy is POLICY
                async for chunk in execution_app.state.ollama.chat_stream(
                    model=MODEL, messages=messages
                ):
                    emitter.text_delta(chunk["message"]["content"])
                    observed.append(current_processing_policy())
                    yield "frame"
            emitter.message_finished(status="completed")
            emitter.run_finished(status="succeeded", finish_reason="stop")
            yield "terminal"
        finally:
            observed.append(current_processing_policy())
            assert not execution_app.state.ollama._client.is_closed

    manager = NativeRunManager(app=app, store=store, stream_factory=stream)
    try:
        live = await _launch(
            manager,
            owner,
            started,
            processing_runtime=runtime,
            generate_title=True,
            model_tools=app.state.tools,
        )
        assert current_processing_policy() is None
        await asyncio.wait_for(live.task, 3)
        await asyncio.wait_for(asyncio.gather(*tuple(manager._title_tasks.values())), 3)
        current = await store.conversations.get(
            user_id=owner.user_id, conversation_id=started.conversation.conversation_id
        )
        assert current.title == "Local Private Summary"
        assert current.private_required
        assert len([r for r in requests if r.url.path == "/api/chat"]) == 3
        assert all(
            PRIVATE_TEXT.encode() not in r.content for r in requests if r.url.path != "/api/chat"
        )
        assert observed and all(p is POLICY for p in observed)
        assert app.state.ollama is ordinary_ollama and app.state.graph is ordinary_graph
        assert app.state.tools.names() == ["calculate", "memory_search", "web_search"]
        assert await store.chat_projections.due() == ()
        saved = await store.conversations.get_run(user_id=owner.user_id, run_id=started.run.run_id)
        assert saved.status == "succeeded" and saved.private_required
    finally:
        await manager.stop()
        store.close()
    assert current_processing_policy() is None


async def test_private_generator_cleanup_stays_bound_when_terminal_settlement_is_cancelled(
    tmp_path,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, started = await _started(store)
    app, requests = _app(), []
    cleanup, settling = asyncio.Event(), asyncio.Event()
    runtime = PrivateProcessingRuntime(
        POLICY, ollama_transport=httpx.MockTransport(_handler(requests))
    )

    async def stream(execution_app, *_args, event_context, **_kwargs):
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("finished")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        try:
            yield "terminal"
        finally:
            assert current_processing_policy() is POLICY
            assert not execution_app.state.ollama._client.is_closed
            cleanup.set()

    manager = NativeRunManager(app=app, store=store, stream_factory=stream)
    original_settle = manager._settle_terminal

    async def blocked_settle(live):
        if not settling.is_set():
            settling.set()
            await asyncio.Event().wait()
        await original_settle(live)

    manager._settle_terminal = blocked_settle
    try:
        live = await _launch(manager, owner, started, processing_runtime=runtime)
        await asyncio.wait_for(settling.wait(), 3)
        live.task.cancel()
        await asyncio.wait_for(live.task, 3)
        assert cleanup.is_set()
        saved = await store.conversations.get_run(user_id=owner.user_id, run_id=started.run.run_id)
        assert saved.status == "succeeded"
    finally:
        await manager.stop()
        store.close()
    assert current_processing_policy() is None


@pytest.mark.parametrize("failure", ["preflight", "stream"])
async def test_private_failures_terminalize_without_content_logs_or_ordinary_history(
    tmp_path, caplog, failure
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, started = await _started(store)
    requests, invoked = [], []
    runtime = PrivateProcessingRuntime(
        POLICY,
        ollama_transport=httpx.MockTransport(
            _handler(requests, digest=lambda: "b" * 64 if failure == "preflight" else "a" * 64),
        ),
    )

    async def stream(*_args, event_context, **_kw):
        invoked.append(True)
        event_context.emitter.run_started()
        yield "frame"
        raise RuntimeError(PRIVATE_TEXT)

    manager = NativeRunManager(app=_app(), store=store, stream_factory=stream)
    try:
        with caplog.at_level(logging.WARNING):
            live = await _launch(manager, owner, started, processing_runtime=runtime)
            await asyncio.wait_for(live.task, 3)
        assert bool(invoked) == (failure == "stream")
        saved = await store.conversations.get_run(user_id=owner.user_id, run_id=started.run.run_id)
        assert saved.status == "failed"
        assert saved.error_code == (
            "local_model_digest_mismatch"
            if failure == "preflight"
            else "private_processing_unavailable"
        )
        assert PRIVATE_TEXT not in caplog.text
        assert await store.chat_projections.due() == ()
        assert all(r.url.path in {"/api/tags", "/api/show"} for r in requests)
    finally:
        await manager.stop()
        store.close()


@pytest.mark.parametrize(
    "case",
    ["private_without_runtime", "ordinary_with_runtime", "ordinary_in_scope", "snapshot_mismatch"],
)
async def test_launch_refuses_missing_or_inconsistent_private_authority_before_creating_task(
    tmp_path, case
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    private = case in {"private_without_runtime", "snapshot_mismatch"}
    owner, started = await _started(store, private=private)
    manager = NativeRunManager(app=SimpleNamespace(), store=store)
    runtime = PrivateProcessingRuntime(POLICY) if case == "ordinary_with_runtime" else None
    if case == "snapshot_mismatch":
        started = replace(
            started, conversation=replace(started.conversation, private_required=False)
        )
    try:
        with processing_scope(POLICY if case == "ordinary_in_scope" else None):
            with pytest.raises(ProcessingPolicyError):
                await _launch(manager, owner, started, processing_runtime=runtime)
        assert not manager._runs and not manager._title_tasks
    finally:
        await manager.stop()
        store.close()


async def test_protected_title_write_needs_scope_and_exact_stored_restriction(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, started = await _started(store)
    c = started.conversation
    args = dict(
        user_id=owner.user_id,
        conversation_id=c.conversation_id,
        expected_title=c.title,
        expected_updated_at=c.updated_at,
        title="Protected title",
    )
    try:
        assert not await store.conversations.apply_generated_title(**args)
        with pytest.raises(ProcessingPolicyError):
            await store.conversations.apply_generated_title(**args, expected_private_required=True)
        with processing_scope(POLICY):
            with pytest.raises(ProcessingPolicyError):
                await store.conversations.apply_generated_title(**args)
            assert await store.conversations.apply_generated_title(
                **args, expected_private_required=True
            )
            assert not await store.conversations.apply_generated_title(
                **args, expected_private_required=True
            )
        current = await store.conversations.get(
            user_id=owner.user_id, conversation_id=c.conversation_id
        )
        assert current.title == "Protected title" and current.private_required
    finally:
        store.close()


async def test_private_graph_dispatches_calculator_without_environment_proxy_or_memory(monkeypatch):
    app = _app()
    app.state.cfg.raw["router"]["skip_llm_under_tokens"] = 100
    app.state.cfg.raw["fast_path"] = {"tool_capable_models": [MODEL], "no_thinking": False}
    app.state.cfg.raw["agentic"] = {
        "escalation": {"enabled": False},
        "reflection": {"enabled": False},
    }
    requests, clients = [], []
    metadata = _handler(requests)
    generations = 0

    def handle(request):
        nonlocal generations
        if request.url.path == "/calculate":
            requests.append(request)
            assert json.loads(request.content)["expression"] == "1250 * 1.08"
            return httpx.Response(200, json={"result": "1350"})
        if request.url.path != "/api/chat":
            return metadata(request)
        requests.append(request)
        generations += 1
        payload = json.loads(request.content)
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["calculate"]
        if generations == 1:
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": {"expression": "1250 * 1.08", "operation": "evaluate"},
                        }
                    }
                ],
            }
        else:
            assert any(
                message.get("role") == "tool" and "1350" in message.get("content", "")
                for message in payload["messages"]
            )
            message = {"content": "1350"}
        return httpx.Response(200, json={"done": True, "message": message})

    original_client = httpx.AsyncClient

    def client_factory(**kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = httpx.MockTransport(handle)
        client = original_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setenv("HTTP_PROXY", "http://untrusted.invalid:8080")
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    runtime = PrivateProcessingRuntime(POLICY)
    async def calculate(args):
        return calculate_value(CalculateRequest.model_validate(args)).model_dump()

    async with runtime.scope(app) as execution_app:
        with private_tool_scope(user_id="owner", handlers={"calculate": calculate}):
            final = await execution_app.state.graph.ainvoke(
                {
                    "virtual_model": "audrey_fast",
                    "messages": [{"role": "user", "content": "Calculate 1250 times 1.08"}],
                    "user_id": "owner",
                    "compatibility_request": False,
                }
            )
            assert final["content"] == "1350"
            assert final["tool_rounds"] == 1
    assert generations == 2
    assert not any(r.url.path == "/calculate" for r in requests)
    assert not any("memory" in r.url.path or "web" in r.url.path for r in requests)
    assert all(not client.trust_env and client.is_closed for client in clients)


async def test_private_execution_view_allows_only_declared_shared_services():
    app, requests = _app(), []
    shared = object()
    app.state.application_store = shared
    app.state.future_cloud_service = object()
    original_ollama = app.state.ollama
    runtime = PrivateProcessingRuntime(POLICY, ollama_transport=httpx.MockTransport(_handler(requests)))
    async with runtime.scope(app) as execution:
        assert execution.state.application_store is shared
        assert execution.state.health is app.state.health
        assert execution.state.gate is app.state.gate
        assert execution.state.inflight is app.state.inflight
        assert execution.state.ollama is not original_ollama
        with pytest.raises(ProcessingPolicyError):
            _ = execution.state.future_cloud_service
        with pytest.raises(ProcessingPolicyError):
            _ = getattr(execution.state, "another_ordinary_service", None)
    assert app.state.ollama is original_ollama
    assert app.state.future_cloud_service is not None
