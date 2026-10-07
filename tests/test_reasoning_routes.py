"""Reasoning admission and wire forwarding through real mocked Ollama clients."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import AuthedUser, require_user
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import routes

MODEL = "reasoning-model"
DESCRIPTOR = {"values": [False, True, "low", "medium", "high"], "default": "medium"}
FUNCTIONS = [{
    "type": "function", "name": "get_marker", "strict": True,
    "parameters": {"type": "object", "properties": {"marker": {"type": "string"}},
                   "required": ["marker"], "additionalProperties": False},
}]


@pytest.fixture
async def provider():
    clients = []

    def build(*, descriptor=DESCRIPTOR, caps=None, show_status=200,
              config_think=None, allowed=True, require_role=None):
        captured, paths = [], []

        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/api/show":
                body = {"capabilities": ["completion", "tools", "thinking"] if caps is None else caps}
                if descriptor is not None:
                    body["thinking"] = copy.deepcopy(descriptor)
                return httpx.Response(show_status, json=body)
            assert request.url.path == "/api/chat"
            native = json.loads(request.content)
            captured.append(native)
            message = {"role": "assistant", "content": "answer"}
            if native.get("tools"):
                message = {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "get_marker", "arguments": {"marker": "ok"}}},
                ]}
            terminal = {"message": message, "done": True, "done_reason": "stop",
                        "prompt_eval_count": 12, "eval_count": 5}
            return (httpx.Response(200, content=(json.dumps(terminal) + "\n").encode())
                    if native.get("stream") else httpx.Response(200, json=terminal))

        ollama = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
        cfg = SimpleNamespace(
            raw={"passthrough": {"enabled": True, "require_role": require_role,
                                 "allowed_models": [MODEL] if allowed else ["other-model"],
                                 "think": config_think},
                 "vision": {"describe_for_text_models": False}},
            timeouts={"medium": 3}, model_registry={},
        )
        registry = ModelRegistry(cfg)
        registry.location_of = lambda _model: "local"
        user = AuthedUser(email="alice@example.com", role="user", owui_id="alice")
        app = FastAPI()
        app.include_router(routes.router)
        app.state.cfg = cfg
        app.state.ollama = ollama
        app.state.registry = registry
        app.state.gate = FairLocalGate(concurrency=1)
        app.state.inflight = UserInflightRegistry(max_inflight_per_user=1)
        app.state.health = HealthTracker()
        app.dependency_overrides[require_user] = lambda: user
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://audrey")
        clients.append((client, ollama))
        return SimpleNamespace(app=app, client=client, captured=captured, paths=paths, cfg=cfg)

    yield build
    for client, ollama in clients:
        await client.aclose()
        await ollama.aclose()


def body(protocol, *, stream=False, effort=None, **updates):
    common = {"model": f"audrey_passthrough/{MODEL}", "stream": stream}
    if protocol == "chat":
        value = {**common, "messages": [{"role": "user", "content": "Answer briefly."}]}
        if effort is not None:
            value["reasoning_effort"] = effort
    else:
        value = {**common, "input": "Answer briefly."}
        if protocol == "functions":
            value["tools"] = copy.deepcopy(FUNCTIONS)
        if effort is not None:
            value["reasoning"] = {"effort": effort}
    return {**value, **updates}


def endpoint(protocol):
    return "/v1/chat/completions" if protocol == "chat" else "/v1/responses"


def terminal(response, protocol, stream):
    assert response.status_code == 200, response.text
    if not stream:
        result = response.json()
    elif protocol == "chat":
        assert response.text.rstrip().endswith("data: [DONE]")
        return None
    else:
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert frames[-1]["type"] == "response.completed"
        result = frames[-1]["response"]
    if protocol != "chat":
        assert result["status"] == "completed"
    return result


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("effort,expected", [("low", "low"), ("medium", "medium"), ("high", "high"), ("none", False)])
async def test_control_reaches_real_provider_in_every_generation_path(provider, protocol, stream, effort, expected):
    service = provider(config_think=True)
    response = await service.client.post(endpoint(protocol), json=body(protocol, stream=stream, effort=effort))
    result = terminal(response, protocol, stream)
    assert len(service.captured) == 1
    assert service.captured[0]["think"] == expected
    assert type(service.captured[0]["think"]) is type(expected)
    assert service.captured[0]["stream"] is stream
    if protocol == "functions":
        assert service.captured[0]["tools"][0]["function"]["name"] == "get_marker"
        assert result["output"][0]["type"] == "function_call"
    if result is not None and protocol != "chat":
        assert result["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configured", [None, False, True])
async def test_omitted_effort_preserves_existing_configured_boolean(provider, protocol, stream, configured):
    service = provider(config_think=configured)
    result = terminal(await service.client.post(endpoint(protocol), json=body(protocol, stream=stream)), protocol, stream)
    if configured is None:
        assert "think" not in service.captured[0]
    else:
        assert service.captured[0]["think"] is configured
    if result is not None and protocol != "chat":
        assert "reasoning" not in result


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_boolean_request_still_overrides_config(provider, stream):
    service = provider(config_think=True)
    terminal(await service.client.post(endpoint("chat"), json=body("chat", stream=stream, think=False)), "chat", stream)
    assert service.captured[0]["think"] is False


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [False, True])
async def test_conflicting_chat_controls_reject_before_provider_probe(provider, stream, think):
    service = provider()
    response = await service.client.post(endpoint("chat"), json=body("chat", stream=stream, effort="high", think=think))
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert service.paths == []


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["model", "role"])
async def test_model_authorization_precedes_reasoning_metadata(provider, protocol, stream, policy):
    service = provider(allowed=policy != "model", require_role="admin" if policy == "role" else None)
    response = await service.client.post(endpoint(protocol), json=body(protocol, stream=stream, effort="high"))
    assert response.status_code == 403
    assert service.paths == []


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
async def test_authentication_precedes_reasoning_metadata(provider, protocol):
    service = provider()
    service.app.dependency_overrides.clear()
    response = await service.client.post(endpoint(protocol), json=body(protocol, effort="high"))
    assert response.status_code == 401
    assert service.paths == []


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_virtual_model_rejects_effort_before_any_generation(provider, protocol, stream):
    service = provider()
    response = await service.client.post(endpoint(protocol), json=body(protocol, stream=stream, effort="high", model="audrey_fast"))
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert service.paths == []


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_unadvertised_level_is_not_coerced_to_boolean(provider, protocol, stream):
    service = provider()
    response = await service.client.post(endpoint(protocol), json=body(protocol, stream=stream, effort="minimal"))
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert service.captured == []


@pytest.mark.parametrize("protocol", ["responses", "functions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode,status", [("unsupported", 400), ("off", 400), ("unavailable", 503)])
async def test_control_failure_precedes_file_hydration_and_sse(provider, monkeypatch, protocol, stream, mode, status):
    service = provider(descriptor={"values": ["low", True], "default": "low"} if mode == "off" else DESCRIPTOR,
                       show_status=503 if mode == "unavailable" else 200)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("Rejected reasoning controls reached file hydration")

    monkeypatch.setattr(routes, "response_input_messages", forbidden)
    effort = "none" if mode == "off" else "minimal" if mode == "unsupported" else "high"
    value = body(protocol, stream=stream, effort=effort, input=[{
        "role": "user", "content": [{"type": "input_file", "file_url": "https://example.org/document.pdf"}],
    }])
    response = await service.client.post(endpoint(protocol), json=value)
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert service.captured == []


@pytest.mark.parametrize("protocol", ["chat", "responses", "functions"])
async def test_boolean_only_descriptor_does_not_promise_graded_effort(provider, protocol):
    service = provider(descriptor={"values": [False, True], "default": True})
    result = await service.client.post(endpoint(protocol), json=body(protocol, effort="high"))
    assert result.status_code == 400 and service.captured == []


@pytest.mark.parametrize("stream", [False, True])
async def test_saved_response_keeps_echo_but_continuation_does_not_inherit_control(provider, tmp_path, stream):
    service = provider()
    store = ApplicationStore(tmp_path / "reasoning.sqlite")
    account = await store.resolve_external_identity(provider="owui", subject="reasoning-alice", email="alice@example.com",
                                                    display_name="Alice", role="user", auth_method="owui_bearer")
    service.app.state.application_store = store
    service.app.dependency_overrides[require_user] = lambda: AuthedUser(
        email=account.email, role=account.role, owui_id="alice", principal=account,
    )
    try:
        first = terminal(await service.client.post(endpoint("responses"), json=body(
            "responses", effort="high", store=True, stream=stream,
        )), "responses", stream)
        assert (await service.client.get("/v1/responses/" + first["id"])).json() == first
        second = terminal(await service.client.post(endpoint("responses"), json=body(
            "responses", previous_response_id=first["id"], input="A follow-up question.",
        )), "responses", False)
        assert "reasoning" not in second
        assert service.captured[0]["think"] == "high" and "think" not in service.captured[1]
        assert [item["role"] for item in service.captured[1]["messages"]] == ["user", "assistant", "user"]
    finally:
        store.close()
