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
`pipeline/graph.py` `node_complexity` (non-streaming).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from audrey.models.registry import ModelRegistry
from audrey.routes.openai import ChatCompletionRequest, ChatMessage

# ─── Schema accepts multimodal content ─────────────────────────────────

_IMAGE_CONTENT = [
    {"type": "text", "text": "describe this image"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
]


def test_chat_message_accepts_multimodal_list_content():
    """The original bug: a list `content` 422'd against `content: str`."""
    m = ChatMessage(role="user", content=_IMAGE_CONTENT)
    assert isinstance(m.content, list)
    assert m.content[1]["type"] == "image_url"


def test_chat_message_still_accepts_plain_string_content():
    m = ChatMessage(role="user", content="just words")
    assert m.content == "just words"


def test_chat_message_rejects_non_str_non_list_content():
    # Loosening to `str | list` must not start accepting scalars like ints.
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=42)


def test_completion_request_round_trips_image_message():
    """model_dump (how the route feeds the pipeline) preserves the list."""
    req = ChatCompletionRequest(
        model="audrey_auto",
        messages=[{"role": "user", "content": _IMAGE_CONTENT}],
    )
    dumped = [m.model_dump(exclude_none=True) for m in req.messages]
    assert dumped[0]["content"] == _IMAGE_CONTENT


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
