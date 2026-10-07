"""Provider-advertised thinking admission with the actual Ollama HTTP client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.routes.openai.reasoning import resolve_reasoning_effort, validate_reasoning_request

_MODEL = "reasoner:latest"
_API_MODEL = f"audrey_passthrough/{_MODEL}"
_GOOD = {"capabilities": ["completion", "thinking"],
         "thinking": {"values": [False, True, "low", "high", "max"], "default": "high"}}


@asynccontextmanager
async def _client(handler: Callable[..., Any]) -> AsyncIterator[OllamaClient]:
    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.parametrize("body, expected", [
    (_GOOD, (False, True, "low", "high", "max")),
    ({"capabilities": ["thinking"], "thinking": {"values": [False, True], "default": True}}, (False, True)),
    ({"capabilities": ["thinking"], "thinking": {"values": ["economy", "careful"], "default": "careful"}}, ("economy", "careful")),
    ({"capabilities": ["completion"], "thinking": {"values": [False], "default": False}}, (False,)),
    ({"capabilities": ["completion"], "thinking": {"values": [False, "high"], "default": "high"}}, (False,)),
    ({"thinking": {"values": ["high"], "default": "high"}}, ()),
    ({"capabilities": ["thinking"]}, ()),
    ({"capabilities": ["thinking"], "thinking": None}, ()),
])
async def test_verified_and_absent_metadata_are_cached(body: dict, expected: tuple):
    requests = []

    def handle(request: httpx.Request):
        requests.append(request)
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": _MODEL}
        assert request.extensions["timeout"]["read"] == 5.0
        return httpx.Response(200, json=body)

    async with _client(handle) as client:
        assert await client.thinking_values(_MODEL) == expected
        assert await client.thinking_values(_MODEL) == expected
    assert len(requests) == 1


@pytest.mark.parametrize("descriptor", [
    {}, [], "thinking", True,
    {"values": None, "default": False},
    {"values": False, "default": False},
    {"values": "high", "default": "high"},
    {"values": [], "default": False},
    {"values": [0], "default": 0},
    {"values": [1], "default": 1},
    {"values": [1.0], "default": 1.0},
    {"values": [None], "default": None},
    {"values": [{}], "default": {}},
    {"values": [""], "default": ""},
    {"values": [False, False], "default": False},
    {"values": ["high", "high"], "default": "high"},
    {"values": [False], "default": 0},
    {"values": [True], "default": 1},
    {"values": [False], "default": "false"},
    {"values": ["high"], "default": "medium"},
    {"values": ["high"]},
])
async def test_malformed_descriptor_is_not_cached(descriptor: Any):
    requests = []

    def handle(request: httpx.Request):
        requests.append(request)
        body = {**_GOOD, "thinking": descriptor} if len(requests) == 1 else _GOOD
        return httpx.Response(200, json=body)

    async with _client(handle) as client:
        with pytest.raises(OllamaError, match="/api/show"):
            await client.thinking_values(_MODEL)
        assert await client.thinking_values(_MODEL) == (False, True, "low", "high", "max")
    assert len(requests) == 2


@pytest.mark.parametrize("capabilities", [None, "thinking", {"thinking": True}, [1], [False]])
async def test_malformed_capabilities_do_not_admit_a_named_level(capabilities: Any):
    async with _client(lambda _: httpx.Response(200, json={**_GOOD, "capabilities": capabilities})) as client:
        with pytest.raises(HTTPException) as error:
            await resolve_reasoning_effort(client, _MODEL, "high")
    assert error.value.status_code == 503
    assert error.value.detail["error"] == "reasoning_metadata_unavailable"


@pytest.mark.parametrize("failure", ["transport", "http", "json"])
async def test_lookup_failure_is_retryable_and_uncached(failure: str):
    requests = []

    def handle(request: httpx.Request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "transport":
                raise httpx.ConnectError("unreachable", request=request)
            if failure == "http":
                return httpx.Response(500, json={"error": "unavailable"})
            return httpx.Response(200, content=b"not JSON")
        return httpx.Response(200, json=_GOOD)

    async with _client(handle) as client:
        with pytest.raises(HTTPException) as error:
            await resolve_reasoning_effort(client, _MODEL, "high")
        assert error.value.status_code == 503
        assert error.value.detail["error"] == "reasoning_metadata_unavailable"
        assert await resolve_reasoning_effort(client, _MODEL, "high") == "high"
    assert len(requests) == 2


@pytest.mark.parametrize("effort", ["low", "high", "max"])
async def test_named_effort_is_preserved_exactly(effort: str):
    async with _client(lambda _: httpx.Response(200, json=_GOOD)) as client:
        assert await resolve_reasoning_effort(client, _MODEL, effort) == effort


async def test_model_defined_level_requires_no_model_name_mapping():
    body = {"capabilities": ["thinking"],
            "thinking": {"values": ["economy", "careful"], "default": "careful"}}
    async with _client(lambda _: httpx.Response(200, json=body)) as client:
        assert await resolve_reasoning_effort(client, "new-model:tag", "economy") == "economy"


@pytest.mark.parametrize("body, effort, expected", [
    (_GOOD, "medium", [False, True, "low", "high", "max"]),
    (_GOOD, "HIGH", [False, True, "low", "high", "max"]),
    (_GOOD, " high ", [False, True, "low", "high", "max"]),
    (_GOOD, "xhigh", [False, True, "low", "high", "max"]),
    ({"capabilities": ["thinking"], "thinking": {"values": [True], "default": True}}, "none", [True]),
    ({"capabilities": ["thinking"], "thinking": {"values": [False, True], "default": True}}, "low", [False, True]),
    ({"capabilities": ["thinking"]}, "high", []),
    ({"capabilities": ["completion"], "thinking": {"values": ["high"], "default": "high"}}, "high", []),
    ({"capabilities": ["thinking"], "thinking": {"values": ["none"], "default": "none"}}, "none", ["none"]),
])
async def test_unadvertised_controls_reject_without_generation(body: dict, effort: str, expected: list):
    requests = []

    def handle(request: httpx.Request):
        requests.append(request.url.path)
        return httpx.Response(200, json=body)

    async with _client(handle) as client:
        with pytest.raises(HTTPException) as error:
            await resolve_reasoning_effort(client, _MODEL, effort)
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "reasoning_unsupported"
    assert error.value.detail["supported_values"] == expected
    assert requests == ["/api/show"]


@pytest.mark.parametrize("capabilities", [["thinking"], ["completion"]])
async def test_none_requires_explicit_boolean_false(capabilities: list):
    body = {"capabilities": capabilities,
            "thinking": {"values": [False], "default": False}}
    async with _client(lambda _: httpx.Response(200, json=body)) as client:
        assert await resolve_reasoning_effort(client, _MODEL, "none") is False


async def test_legacy_boolean_behavior_does_not_require_new_descriptor():
    async with _client(lambda _: httpx.Response(200, json={"capabilities": ["thinking"]})) as client:
        assert await client.thinking_values(_MODEL) == ()
        assert await client.thinking_flag(_MODEL, False) is False
        assert await client.thinking_flag(_MODEL, True) is True


@pytest.mark.parametrize("think", [False, True])
def test_request_rejects_both_controls_even_when_think_is_false(think: bool):
    with pytest.raises(HTTPException) as error:
        validate_reasoning_request(_API_MODEL, effort="high", think=think)
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "reasoning_controls_conflict"


@pytest.mark.parametrize("model", ["audrey_fast", "audrey_deep", "auto", "direct/reasoner:latest", "audrey_passthrough"])
def test_pipeline_and_native_models_reject_effort(model: str):
    with pytest.raises(HTTPException) as error:
        validate_reasoning_request(model, effort="high")
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "reasoning_unsupported"


@pytest.mark.parametrize("model, effort, think", [
    (_API_MODEL, "high", None), (_API_MODEL, None, False), (_API_MODEL, None, True),
    (_API_MODEL, None, None), ("audrey_fast", None, False), ("audrey_fast", None, None),
])
def test_request_preserves_legacy_and_unset_control_admission(model: str, effort: str | None, think: bool | None):
    validate_reasoning_request(model, effort=effort, think=think)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", ["low", "max", False, True, None])
async def test_chat_wire_preserves_named_boolean_and_omitted_values(stream: bool, think: bool | str | None):
    sent = []

    def handle(request: httpx.Request):
        body = json.loads(request.content)
        sent.append(body)
        terminal = {"done": True, "message": {"role": "assistant", "content": "OK"}}
        if body["stream"]:
            return httpx.Response(200, content=(json.dumps(terminal) + "\n").encode())
        return httpx.Response(200, json=terminal)

    async with _client(handle) as client:
        args = {"model": _MODEL, "messages": [{"role": "user", "content": "Reply OK."}], "think": think}
        if stream:
            chunks = [chunk async for chunk in client.chat_stream(**args)]
            assert chunks[-1]["done"] is True
        else:
            assert (await client.chat(**args))["done"] is True
    assert len(sent) == 1
    if think is None:
        assert "think" not in sent[0]
    else:
        assert type(sent[0]["think"]) is type(think)
        assert sent[0]["think"] == think
