"""Hermetic tests for inline-image (multimodal) chat support.

Covers the two ends of the Phase 15 change that don't already live in
`test_messages.py` (the `has_image_part` detector) or `test_complexity.py`:

  1. The request schema accepts the OpenAI multimodal `content` list
     instead of 422-ing it (the original bug).
  2. The `vl:` registry pool resolves to a local vision model, so an
     image turn forced to `task="vl"` reaches a model that can see it.

The gate decision itself (image turn → `task="vl"`, fast path) is driven
entirely by `has_image_part`, which is unit-tested in `test_messages.py`;
the branch wiring lives inline in both `routes/openai.py` (streaming) and
`pipeline/graph.py` `node_complexity` (non-streaming). That gate now has
one exception — an explicit deep/research pick keeps its model and gets
the image transcribed instead (see `pipeline/vision.py`) — so the two
sides of the fork are pinned here too.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline import graph as gmod
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.openai import ChatCompletionRequest

# ─── Schema accepts multimodal content ─────────────────────────────────

_IMAGE_CONTENT = [
    {"type": "text", "text": "describe this image"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
]


def test_chat_message_accepts_multimodal_list_content():
    """The original bug: a list `content` 422'd against `content: str`."""
    m = ChatCompletionRequest(
        model="audrey_auto",
        messages=[{"role": "user", "content": _IMAGE_CONTENT}],
    ).messages[0]
    assert isinstance(m.content, list)
    assert m.content[1]["type"] == "image_url"


def test_chat_message_still_accepts_plain_string_content():
    m = ChatCompletionRequest(
        model="audrey_auto",
        messages=[{"role": "user", "content": "just words"}],
    ).messages[0]
    assert m.content == "just words"


def test_chat_message_rejects_non_str_non_list_content():
    # Loosening to `str | list` must not start accepting scalars like ints.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="audrey_auto",
            messages=[{"role": "user", "content": 42}],
        )


def test_completion_request_round_trips_image_message():
    """model_dump (how the route feeds the pipeline) preserves the list."""
    req = ChatCompletionRequest(
        model="audrey_auto",
        messages=[{"role": "user", "content": _IMAGE_CONTENT}],
    )
    dumped = [m.model_dump(exclude_none=True) for m in req.messages]
    assert dumped[0]["content"] == _IMAGE_CONTENT


_TOOL_CALL = {
    "id": "call_weather_1",
    "type": "function",
    "function": {
        "name": "get_temperature",
        "arguments": """{"city":"New York"}""",
    },
}


def test_completion_request_accepts_developer_message():
    req = ChatCompletionRequest(
        model="audrey_auto",
        messages=[
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert req.messages[0].role == "developer"


def test_assistant_tool_call_allows_null_content_and_round_trips():
    req = ChatCompletionRequest(
        model="audrey_passthrough/qwen3.8:latest",
        messages=[
            {"role": "user", "content": "What is the temperature?"},
            {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
            {"role": "tool", "content": "22 C", "tool_call_id": "call_weather_1"},
        ],
    )
    dumped = [message.model_dump(exclude_none=True) for message in req.messages]
    assert dumped[1]["tool_calls"] == [_TOOL_CALL]
    assert "content" not in dumped[1]
    assert dumped[2]["tool_call_id"] == "call_weather_1"


def test_owui_metadata_extensions_survive_validation_but_can_be_stripped():
    req = ChatCompletionRequest(
        model="audrey_auto",
        messages=[{
            "role": "user",
            "content": "Hello",
            "metadata": {"chat_id": "owui-chat-1"},
        }],
        chat_id="owui-chat-1",
    )
    assert req.messages[0].metadata == {"chat_id": "owui-chat-1"}
    assert req.chat_id == "owui-chat-1"
    assert req.messages[0].model_dump(exclude_none=True) == {
        "role": "user",
        "content": "Hello",
        "metadata": {"chat_id": "owui-chat-1"},
    }
    assert req.messages[0].model_dump(
        exclude_none=True,
        exclude={"metadata"},
    ) == {
        "role": "user",
        "content": "Hello",
    }


def test_tool_message_requires_tool_call_id():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="audrey_auto",
            messages=[{"role": "tool", "content": "22 C"}],
        )


def test_assistant_message_requires_content_or_tool_calls():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="audrey_auto",
            messages=[{"role": "assistant", "content": None}],
        )


def test_message_fields_are_role_specific():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="audrey_auto",
            messages=[{
                "role": "user",
                "content": "This is not an assistant turn.",
                "tool_calls": [_TOOL_CALL],
            }],
        )


# ─── vl: pool resolves to a local vision model ─────────────────────────

class _Cfg:
    """Minimal Config stand-in exposing just `.model_registry`."""

    def __init__(self, registry: dict) -> None:
        self._registry = registry

    @property
    def model_registry(self) -> dict:
        return self._registry


def test_vl_pool_first_healthy_picks_local_vision_model():
    """An image turn forces task='vl'; first_healthy('vl') must return a
    real local vision model, not a text model."""
    registry = ModelRegistry(_Cfg({
        "vl": [
            {"name": "qwen3-vl:32b", "priority": 100, "speed": 70,
             "quality": 90, "location": "local"},
            {"name": "llava:34b", "priority": 90, "speed": 65,
             "quality": 85, "location": "local"},
        ],
    }))
    spec = registry.first_healthy("vl", lambda name: True)
    assert spec is not None
    assert spec.name == "qwen3-vl:32b"
    assert spec.location == "local"


def test_vl_pool_falls_back_to_llava_when_primary_unhealthy():
    registry = ModelRegistry(_Cfg({
        "vl": [
            {"name": "qwen3-vl:32b", "priority": 100, "speed": 70,
             "quality": 90, "location": "local"},
            {"name": "llava:34b", "priority": 90, "speed": 65,
             "quality": 85, "location": "local"},
        ],
    }))
    spec = registry.first_healthy("vl", lambda name: name != "qwen3-vl:32b")
    assert spec is not None
    assert spec.name == "llava:34b"


# ─── The routing fork: pin to vl, or transcribe and keep the pick ──────


class _NoTools:
    def all_specs(self) -> list:
        return []

    def __iter__(self):
        return iter([])


async def _complexity_node(monkeypatch, *, describe_returns=None):
    """Compile the graph and hand back its `complexity` node.

    The transcription itself is stubbed: this asserts the *routing* fork,
    and a real vision call would need a live Ollama. `vision.py`'s own
    tests cover the rewrite.
    """
    calls: list[dict] = []

    async def _fake_describe(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return describe_returns if describe_returns is not None else messages, 1

    monkeypatch.setattr(gmod, "describe_for_text_model", _fake_describe)
    cfg = get_config()
    ollama = OllamaClient(base_url="http://unused")
    compiled = gmod.build_graph(
        cfg, ollama, ModelRegistry(cfg), HealthTracker(),
        FairLocalGate(concurrency=1), _NoTools(),
    )
    return compiled.nodes["complexity"].bound, calls, ollama


async def test_image_on_auto_is_still_pinned_to_the_vl_pool(monkeypatch):
    """No explicit model pick → the vision model answers directly, which
    beats reasoning over a description."""
    node, calls, ollama = await _complexity_node(monkeypatch)
    out = await node.ainvoke({
        "virtual_model": "audrey_auto",
        "messages": [{"role": "user", "content": _IMAGE_CONTENT}],
    })
    await ollama.aclose()

    assert out["mode"] == "fast"
    assert out["task_type"] == "vl"
    assert calls == []  # nothing transcribed — the vl model gets the pixels


async def test_image_on_explicit_deep_pick_keeps_deep_and_transcribes(monkeypatch):
    """The reported complaint: picking a model and having it overridden."""
    described = [{"role": "user", "content": "what is this?\n\n[Attached image 1 of 1 …]"}]
    node, calls, ollama = await _complexity_node(monkeypatch, describe_returns=described)
    out = await node.ainvoke({
        "virtual_model": "audrey_cloud",
        "messages": [{"role": "user", "content": _IMAGE_CONTENT}],
        "user_id": "alice@example.com",
    })
    await ollama.aclose()

    assert out["mode"] == "deep"
    assert "task_type" not in out  # not hijacked to the vl pool
    assert out["messages"] == described  # panel sees text, not an image part
    assert len(calls) == 1
    assert calls[0]["user_id"] == "alice@example.com"


async def test_sidecar_off_restores_the_vl_pin_for_deep_picks(monkeypatch):
    node, calls, ollama = await _complexity_node(monkeypatch)
    monkeypatch.setattr(gmod, "describe_enabled", lambda _cfg: False)
    out = await node.ainvoke({
        "virtual_model": "audrey_cloud",
        "messages": [{"role": "user", "content": _IMAGE_CONTENT}],
    })
    await ollama.aclose()

    assert out["mode"] == "fast"
    assert out["task_type"] == "vl"
    assert calls == []
