"""Verified local runtime ownership, isolation and transport boundaries."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import httpx
import pytest

from audrey.config import Config, EnvOverrides
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.private_processing import PrivateProcessingRuntime, prepare_processing_policy
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    current_processing_policy,
    processing_scope,
)
from audrey.routes.inflight import UserInflightRegistry
from audrey.tools.discovery import ToolRegistry, ToolSpec

_MODEL = "local-chat:latest"
_DIGEST = "a" * 64
_PRIVATE = "PRIVATE_RUNTIME_EXAMPLE_7381"
_SERVER = "http://custom-tools:8001"


class TrackingTransport(httpx.MockTransport):
    def __init__(self, handler, *, policy_on_close=...):
        super().__init__(handler)
        self.closes = 0
        self.policy_on_close = policy_on_close

    async def aclose(self):
        if self.policy_on_close is not ...:
            assert current_processing_policy() is self.policy_on_close
        self.closes += 1


def _handler(seen, *, digest=_DIGEST, issue=None):
    def handle(request):
        seen.append(request)
        if request.url.path == "/api/tags":
            if issue == "redirect":
                return httpx.Response(307, headers={"Location": "https://outside.invalid/api/tags"})
            if issue == "outage":
                return httpx.Response(503, text=_PRIVATE)
            if issue == "bad_json":
                return httpx.Response(200, text=_PRIVATE)
            models = [{"name": _MODEL, "model": _MODEL, "digest": digest}]
            if issue == "missing":
                models = []
            if issue == "duplicate":
                models *= 2
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/show":
            metadata = {
                "details": {"format": "gguf"},
                "model_info": {"general.architecture": "qwen", "general.parameter_count": 100},
                "capabilities": [],
            }
            if issue == "remote":
                metadata["remote_model"] = "remote"
            if issue == "missing_weights":
                metadata["model_info"] = {}
            return httpx.Response(200, json=metadata)
        assert request.url.path == "/api/chat"
        body = {"message": {"content": "complete"}, "done": True}
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=json.dumps(body) + "\n")
        return httpx.Response(200, json=body)
    return handle


def _policy(**kwargs):
    return ProcessingPolicy(
        local_models={_MODEL: _DIGEST}, ollama_base_url="http://ollama:11434", **kwargs,
    )


def _app(*, tools=None, title=True):
    cfg = Config({
        "router": {"model": _MODEL, "skip_llm_under_tokens": 100},
        "model_registry": {"general": [{"name": _MODEL, "location": "local"}]},
        "fast_path": {"no_thinking": False},
        "agentic": {"escalation": {"enabled": False}, "reflection": {"enabled": False}},
        "title_generation": {"enabled": title, "model": _MODEL},
    }, EnvOverrides(_env_file=None))
    return SimpleNamespace(state=SimpleNamespace(
        cfg=cfg, registry=ModelRegistry(cfg), health=HealthTracker(),
        gate=FairLocalGate(concurrency=1), inflight=UserInflightRegistry(max_inflight_per_user=3),
        tools=tools or ToolRegistry(), ollama=object(), graph=object(),
        archive_http=object(), archive_client=object(), conversation_titles=object(),
        ordinary_marker=object(),
    ))


async def test_policy_factory_snapshots_local_weights_without_sending_content():
    seen = []
    transport = TrackingTransport(_handler(seen))
    policy = await prepare_processing_policy(
        ollama_base_url="http://ollama:11434/", model_names=["local-chat"],
        expected_digests={_MODEL: "sha256:" + _DIGEST}, transport=transport,
    )
    assert dict(policy.local_models) == {_MODEL: _DIGEST}
    assert policy.ollama_base_url == "http://ollama:11434"
    assert [r.url.path for r in seen] == ["/api/tags", "/api/show"]
    assert all(_PRIVATE.encode() not in r.content for r in seen)
    assert transport.closes == 1
    assert current_processing_policy() is None


@pytest.mark.parametrize("issue", ["missing", "duplicate", "remote", "missing_weights", "outage", "bad_json", "redirect"])
async def test_policy_factory_fails_closed_and_closes_metadata_transport(issue, caplog):
    seen = []
    transport = TrackingTransport(_handler(seen, issue=issue))
    with pytest.raises(ProcessingPolicyError) as error:
        await prepare_processing_policy(
            ollama_base_url="http://ollama:11434", model_names=[_MODEL], transport=transport,
        )
    assert _PRIVATE not in str(error.value) + caplog.text
    assert all(r.url.path in {"/api/tags", "/api/show"} for r in seen)
    assert transport.closes == 1
    assert current_processing_policy() is None


@pytest.mark.parametrize("names", [[], [_MODEL, _MODEL], ["local-chat", _MODEL], [123], ["kimi-k3:cloud"], "local-chat"])
async def test_invalid_server_roster_is_refused_before_metadata(names):
    seen = []
    with pytest.raises(ProcessingPolicyError):
        await prepare_processing_policy(
            ollama_base_url="http://ollama:11434", model_names=names,
            transport=httpx.MockTransport(_handler(seen)),
        )
    assert seen == []


async def test_configured_digest_mismatch_never_opens_a_content_request():
    seen = []
    with pytest.raises(ProcessingPolicyError) as error:
        await prepare_processing_policy(
            ollama_base_url="http://ollama:11434", model_names=[_MODEL],
            expected_digests={_MODEL: "b" * 64},
            transport=httpx.MockTransport(_handler(seen)),
        )
    assert error.value.code == "local_model_digest_mismatch"
    assert [r.url.path for r in seen] == ["/api/tags"]


async def test_parent_policy_cannot_be_weakened_during_preparation():
    seen = []
    parent = ProcessingPolicy(local_models={_MODEL: "b" * 64}, ollama_base_url="http://ollama:11434")
    with processing_scope(parent), pytest.raises(ProcessingPolicyError) as error:
        await prepare_processing_policy(
            ollama_base_url="http://ollama:11434", model_names=[_MODEL],
            transport=httpx.MockTransport(_handler(seen)),
        )
    assert error.value.code == "processing_policy_downgrade"
    assert all(r.url.path == "/api/tags" for r in seen)


async def test_runtime_owns_safe_clients_runs_real_graph_and_keeps_shared_state(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8123")
    seen = []
    policy = _policy()
    transport = TrackingTransport(_handler(seen), policy_on_close=policy)
    tool_transport = TrackingTransport(
        lambda request: pytest.fail("Unexpected sidecar request"), policy_on_close=policy,
    )
    runtime = PrivateProcessingRuntime(policy, transport, tool_transport)
    app = _app()
    original = dict(vars(app.state))
    async with runtime.scope(app) as view:
        assert current_processing_policy() is policy
        with pytest.raises(ProcessingPolicyError):
            _ = view.state.ordinary_marker
        assert view.state.health is app.state.health
        assert view.state.gate is app.state.gate
        assert view.state.inflight is app.state.inflight
        assert view.state.ollama is not app.state.ollama
        assert view.state.graph is not app.state.graph
        assert view.state.archive_client is None
        assert view.state.archive_http.trust_env is False
        assert view.state.archive_http.follow_redirects is False
        final = await view.state.graph.ainvoke({
            "messages": [{"role": "user", "content": "Say hello"}],
            "virtual_model": "audrey_fast", "user_id": "owner",
            "compatibility_request": False,
        })
        assert final["content"] == "complete"
        title = await view.state.conversation_titles.generate(user_id="owner", user_content=_PRIVATE)
        assert title == "complete"
        assert dict(vars(app.state)) == original
    assert current_processing_policy() is None
    assert transport.closes == tool_transport.closes == 1
    assert view.state.archive_http.is_closed
    assert dict(vars(app.state)) == original
    assert any(r.url.path == "/api/chat" for r in seen)


async def test_scope_covers_stream_iteration_and_inherited_tasks_without_affecting_ordinary_task():
    seen = []
    policy = _policy()
    runtime = PrivateProcessingRuntime(policy, httpx.MockTransport(_handler(seen)))
    app = _app()
    enter = asyncio.Event()
    observed = []

    async def ordinary():
        await enter.wait()
        observed.append(current_processing_policy())
        return app.state.ollama

    task = asyncio.create_task(ordinary())
    async with runtime.scope(app) as view:
        enter.set()
        assert await task is app.state.ollama

        async def child():
            assert current_processing_policy() is policy
            return [chunk async for chunk in view.state.ollama.chat_stream(
                model=_MODEL, messages=[{"role": "user", "content": _PRIVATE}],
            )]

        chunks = await asyncio.create_task(child())
        assert chunks[-1]["message"]["content"] == "complete"
        assert current_processing_policy() is policy
    assert observed == [None]
    assert current_processing_policy() is None


async def test_scope_rechecks_every_pin_before_yield_or_graph_execution():
    seen = []
    transport = TrackingTransport(_handler(seen, digest="b" * 64))
    runtime = PrivateProcessingRuntime(_policy(), transport)
    with pytest.raises(ProcessingPolicyError) as error:
        async with runtime.scope(_app()):
            pytest.fail("Changed model identity must refuse before work starts")
    assert error.value.code == "local_model_digest_mismatch"
    assert all(r.url.path != "/api/chat" for r in seen)
    assert transport.closes == 1
    assert current_processing_policy() is None


@pytest.mark.parametrize("issue", ["endpoint", "path"])
async def test_tool_endpoint_and_operation_are_checked_before_model_sees_tools(issue):
    tool = ToolSpec(
        name="calculate", description="Calculate", parameters={},
        server_url="https://outside.invalid" if issue == "endpoint" else _SERVER,
        path="/other" if issue == "path" else "/calculate",
    )
    runtime = PrivateProcessingRuntime(
        _policy(allowed_tools=frozenset({"calculate"}), tool_servers=frozenset({_SERVER})),
        httpx.MockTransport(_handler([])),
    )
    with pytest.raises(ProcessingPolicyError):
        async with runtime.scope(_app(tools=ToolRegistry({"calculate": tool}))):
            pytest.fail("Invalid tool dispatch identity must not reach a model")


async def test_scope_filters_model_tools_and_never_expands_roster_for_optional_title():
    tools = ToolRegistry({
        name: ToolSpec(name=name, description=name, parameters={}, server_url=_SERVER, path=f"/{name}")
        for name in ("calculate", "memory_search", "web_search")
    })
    runtime = PrivateProcessingRuntime(
        _policy(allowed_tools=frozenset({"calculate"}), tool_servers=frozenset({_SERVER})),
        httpx.MockTransport(_handler([])),
    )
    app = _app(tools=tools)
    app.state.cfg.raw["title_generation"]["model"] = "unapproved:latest"
    async with runtime.scope(app) as view:
        assert view.state.tools.names() == ["calculate"]
        assert view.state.conversation_titles is None
    assert app.state.tools.names() == ["calculate", "memory_search", "web_search"]


async def test_cancelled_scope_closes_transports_and_restores_context():
    policy = _policy()
    model_transport = TrackingTransport(_handler([]), policy_on_close=policy)
    tool_transport = TrackingTransport(
        lambda request: pytest.fail("No sidecar call expected"), policy_on_close=policy,
    )
    runtime = PrivateProcessingRuntime(policy, model_transport, tool_transport)
    entered = asyncio.Event()
    captured = []

    async def work():
        try:
            async with runtime.scope(_app()) as view:
                entered.set()
                await asyncio.Event().wait()
                return view
        finally:
            captured.append(current_processing_policy())

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model_transport.closes == tool_transport.closes == 1
    assert captured == [None]
    assert current_processing_policy() is None


def test_runtime_policy_cannot_be_replaced():
    runtime = PrivateProcessingRuntime(_policy())
    with pytest.raises(FrozenInstanceError):
        runtime.policy = _policy()
