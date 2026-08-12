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
