"""Streaming passthrough end to end against the REAL OllamaClient (2026-08-12).

`tests/test_passthrough_dispatch.py` stubs the client, and on 2026-08-12 that
stub was edited to accept a `think` kwarg the real client did not have. Every
passthrough turn then failed with `RemoteProtocolError: peer closed connection
without sending complete message body`, and the whole suite stayed green.

So this file uses a real `OllamaClient` over `httpx.MockTransport`: real
signatures, real payload construction, real streaming generator — only the
socket is fake. A kwarg the client cannot accept fails HERE, at the layer the
stub cannot vouch for.

⚠️ The failure mode is why this matters. The exception fires INSIDE the
StreamingResponse generator, after the headers are out, so it can never
surface as a 500 in any test that only checks status codes. It has to be
caught by draining the stream.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import (
    PASSTHROUGH_PREFIX,
    ChatCompletionRequest,
    _handle_passthrough,
)

_MODEL = "qwen3.6:35b-64k"


def _ollama_with_capture(captured: list[dict]) -> OllamaClient:
    """A real client whose socket is a mock. Records each /api/chat payload."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "thinking"]})
        payload = json.loads(request.content)
        captured.append(payload)
        # Ollama answers /api/chat differently by mode: one JSON object when
        # `stream` is false, newline-delimited objects when it is true. The
        # mock has to honour that or the non-streaming client rightly rejects
        # the body as trailing garbage.
        if not payload.get("stream"):
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": "hi there"},
                "prompt_eval_count": 12, "eval_count": 34, "done": True,
            })
        body = "".join(
            json.dumps(c) + "\n" for c in (
                {"message": {"role": "assistant", "content": "hi "}, "done": False},
                {"message": {"role": "assistant", "content": "there"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            )
        )
        return httpx.Response(200, content=body.encode())

    return OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))


def _app(ollama: OllamaClient, think):
    raw = {"passthrough": {
        "enabled": True, "allowed_models": [_MODEL],
        "require_role": None, "think": think,
    }}
    cfg = SimpleNamespace(
        raw=raw, timeouts={"medium": 180},
        model_registry={"vl": [
            {"name": "qwen3-vl:32b", "priority": 100, "location": "local"},
        ]},
    )
    registry = ModelRegistry(cfg)  # type: ignore[arg-type]
    registry.location_of = lambda _n: "local"  # type: ignore[method-assign]
    return SimpleNamespace(state=SimpleNamespace(
        cfg=cfg, registry=registry, ollama=ollama,
        gate=FairLocalGate(concurrency=1),
        inflight=UserInflightRegistry(max_inflight_per_user=3),
        health=HealthTracker(),
    ))


async def _drain(app) -> list[str]:
    """Run a streaming passthrough turn and collect every SSE frame.

    ⚠️ Draining is the point. A generator that raises on its first `__anext__`
    would leave a test that merely *builds* the response perfectly happy.
    """
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=ChatCompletionRequest(
            model=f"{PASSTHROUGH_PREFIX}{_MODEL}",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        ),
        me=SimpleNamespace(email="alice@example.com", role="user", owui_id="abc"),
    )
    return [chunk.decode() if isinstance(chunk, bytes) else chunk
            async for chunk in resp.body_iterator]


@pytest.mark.parametrize("think", [None, True, False])
async def test_streaming_passthrough_completes_in_every_think_state(think):
    """The regression, stated directly: with `think` threaded through, the
    stream must still run to completion. It did not for any state."""
    captured: list[dict] = []
    frames = await _drain(_app(_ollama_with_capture(captured), think))
    assert frames, "no frames — the generator raised before yielding"
    assert "".join(frames).count("data:") >= 2
    assert frames[-1].strip() == "data: [DONE]"
    assert len(captured) == 1


@pytest.mark.parametrize("think,expected", [(None, None), (True, True), (False, False)])
async def test_the_think_field_reaches_ollamas_payload(think, expected):
    """Absent when None — Ollama hard-errors on `think` for a model that does
    not declare the capability, so "omit" must stay genuinely absent."""
    captured: list[dict] = []
    await _drain(_app(_ollama_with_capture(captured), think))
    if expected is None:
        assert "think" not in captured[0]
    else:
        assert captured[0]["think"] is expected


async def test_non_streaming_passthrough_also_completes():
    captured: list[dict] = []
    app = _app(_ollama_with_capture(captured), False)
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=ChatCompletionRequest(
            model=f"{PASSTHROUGH_PREFIX}{_MODEL}",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        ),
        me=SimpleNamespace(email="alice@example.com", role="user", owui_id="abc"),
    )
    assert resp["object"] == "chat.completion"
    assert captured[0]["think"] is False


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_temperature",
        "description": "Get the temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _tool_loop_ollama(captured: list[dict]) -> OllamaClient:
    """Real client over a fake socket for a complete two-request tool loop."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        payload = json.loads(request.content)
        captured.append(payload)
        has_tool_result = any(
            message.get("role") == "tool" for message in payload["messages"]
        )
        if has_tool_result:
            message = {"role": "assistant", "content": "It is 22 C in New York."}
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "index": 0,
                        "name": "get_temperature",
                        "arguments": {"city": "New York"},
                    },
                }],
            }
        if not payload["stream"]:
            return httpx.Response(200, json={
                "message": message,
                "prompt_eval_count": 12,
                "eval_count": 8,
                "done": True,
            })
        body = "".join(json.dumps(chunk) + "\n" for chunk in (
            {"message": message, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ))
        return httpx.Response(200, content=body.encode())

    return OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))


async def _run_tool_loop_request(app, *, messages, stream: bool):
    response = await _handle_passthrough(
        app,
        request=SimpleNamespace(app=app),
        payload=ChatCompletionRequest(
            model=f"{PASSTHROUGH_PREFIX}{_MODEL}",
            messages=messages,
            tools=[_WEATHER_TOOL],
            stream=stream,
        ),
        me=SimpleNamespace(email="alice@example.com", role="user", owui_id="abc"),
    )
    if not stream:
        return response
    frames = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]
    return [
        json.loads(line.removeprefix("data: "))
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_two_request_tool_loop_preserves_linkage_to_ollama(stream):
    captured: list[dict] = []
    ollama = _tool_loop_ollama(captured)
    app = _app(ollama, None)
    first = await _run_tool_loop_request(
        app,
        messages=[{"role": "user", "content": "Temperature in New York?"}],
        stream=stream,
    )
    if stream:
        tool_delta = next(
            frame["choices"][0]["delta"]
            for frame in first
            if frame["choices"][0]["delta"].get("tool_calls")
        )
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_delta["tool_calls"],
        }
    else:
        assistant = first["choices"][0]["message"]
    call_id = assistant["tool_calls"][0]["id"]

    second = await _run_tool_loop_request(
        app,
        messages=[
            {"role": "developer", "content": "Answer concisely."},
            {"role": "user", "content": "Temperature in New York?"},
            assistant,
            {"role": "tool", "content": "22 C", "tool_call_id": call_id},
        ],
        stream=stream,
    )

    assert len(captured) == 2
    forwarded = captured[1]["messages"]
    assert forwarded[:2] == [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Temperature in New York?"},
    ]
    assert forwarded[2].get("content", "") == ""
    assert forwarded[2]["tool_calls"] == [{
        "type": "function",
        "function": {
            "index": 0,
            "name": "get_temperature",
            "arguments": {"city": "New York"},
        },
    }]
    assert forwarded[3] == {
        "role": "tool",
        "content": "22 C",
        "tool_name": "get_temperature",
    }
    if stream:
        assert any(
            frame["choices"][0]["delta"].get("content") == "It is 22 C in New York."
            for frame in second
        )
    else:
        assert second["choices"][0]["message"]["content"] == "It is 22 C in New York."
