"""Tests for the vision sidecar — images for models that can't see.

`glm-5.2:cloud` and every other text pool member has no vision encoder, so
an attached image either errors at Ollama or gets answered blind. The
sidecar transcribes the image with a local `vl` model first and splices the
description in where the `image_url` part was, so the model the caller
picked still writes the answer.

What's worth pinning here: the no-op guards (this must cost nothing on the
overwhelmingly common text-only turn), the rewrite shape (image part out,
labelled text in, original list untouched), the per-image cache (OWUI
resends the data URI on every turn), and the fail-soft branches — a dead
vision model has to degrade to a note, never a 502.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from audrey.config import _load_yaml
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline import vision
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.vision import (
    describe_for_text_model,
    describe_images,
    has_any_image_part,
    is_vision_capable,
    vision_capable_models,
)

_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
_PNG2 = "data:image/png;base64,ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"


class _Cfg:
    """Config stand-in exposing the two attributes the sidecar reads."""

    def __init__(self, raw: dict[str, Any] | None = None, vl: list | None = None) -> None:
        self.raw = raw if raw is not None else {}
        self.model_registry = {
            "vl": vl if vl is not None else [
                {"name": "qwen3-vl:32b", "priority": 100, "location": "local"},
            ],
            "general": [{"name": "glm-5.2:cloud", "priority": 100, "location": "cloud"}],
        }


class _ScriptedOllama:
    """Fake Ollama recording every vision call it is handed."""

    def __init__(self, outcome: Any = None) -> None:
        self._outcome = outcome or {"message": {"content": "a red bicycle on a lawn"}}
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, model: str, messages, options=None, tools=None, timeout_s=None, think=None):
        self.calls.append({
            "model": model, "messages": messages, "timeout_s": timeout_s,
            "options": options, "think": think,
        })
        if isinstance(self._outcome, OllamaError):
            raise self._outcome
        return self._outcome


def _registry(cfg: _Cfg) -> ModelRegistry:
    return ModelRegistry(cfg)  # type: ignore[arg-type]


def _image_msgs(*, text: str = "what is this?", url: str = _PNG) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "you are helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]


async def _describe(messages, cfg, ollama, *, target_model="glm-5.2:cloud", health=None):
    return await describe_for_text_model(
        messages,
        ollama=ollama, registry=_registry(cfg), health=health or HealthTracker(),
        gate=FairLocalGate(concurrency=1), cfg=cfg, target_model=target_model,
        user_question="what is this?", user_id="alice@example.com",
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    """The description cache is process-global — isolate every test."""
    vision._cache.clear()
    yield
    vision._cache.clear()


# ─── Capability lookup ─────────────────────────────────────────────────


def test_vl_pool_is_the_capability_source_of_truth():
    cfg = _Cfg()
    assert vision_capable_models(cfg) == {"qwen3-vl:32b"}
    assert is_vision_capable("qwen3-vl:32b", cfg)
    assert not is_vision_capable("glm-5.2:cloud", cfg)


def test_also_capable_extends_the_vl_pool():
    cfg = _Cfg(raw={"vision": {"also_capable": ["some-vlm:cloud"]}})
    assert vision_capable_models(cfg) == {"qwen3-vl:32b", "some-vlm:cloud"}


def test_has_any_image_part_sees_earlier_history():
    """Broader than `messages.has_image_part`, which only reads the last turn."""
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _PNG}}]},
        {"role": "assistant", "content": "that's a bicycle"},
        {"role": "user", "content": "what colour was it?"},
    ]
    assert has_any_image_part(msgs)
    assert not has_any_image_part([{"role": "user", "content": "no image here"}])


# ─── No-op guards ──────────────────────────────────────────────────────


async def test_text_only_turn_is_untouched_and_costs_nothing():
    cfg = _Cfg()
    ollama = _ScriptedOllama()
    msgs = [{"role": "user", "content": "plain words"}]

    out, n = await _describe(msgs, cfg, ollama)

    assert out is msgs  # same object — no rebuild on the common path
    assert n == 0
    assert ollama.calls == []


async def test_vision_capable_target_is_left_alone():
    """qwen3-vl can read the image itself — transcribing would be pure loss."""
    cfg = _Cfg()
    ollama = _ScriptedOllama()
    msgs = _image_msgs()

    out, n = await _describe(msgs, cfg, ollama, target_model="qwen3-vl:32b")

    assert out is msgs
    assert n == 0
    assert ollama.calls == []


async def test_disabled_by_config_is_a_no_op():
    cfg = _Cfg(raw={"vision": {"describe_for_text_models": False}})
    ollama = _ScriptedOllama()
    msgs = _image_msgs()

    out, n = await _describe(msgs, cfg, ollama)

    assert out is msgs
    assert n == 0
    assert ollama.calls == []


# ─── The rewrite ───────────────────────────────────────────────────────


async def test_image_part_becomes_labelled_text():
    cfg = _Cfg()
    ollama = _ScriptedOllama()
    msgs = _image_msgs()

    out, n = await _describe(msgs, cfg, ollama)

    assert n == 1
    assert len(ollama.calls) == 1
    assert ollama.calls[0]["model"] == "qwen3-vl:32b"

    user = out[1]
    assert isinstance(user["content"], str)  # no list content left for Ollama to reject
    assert "what is this?" in user["content"]  # the user's own words survive
    assert "a red bicycle on a lawn" in user["content"]
    assert "qwen3-vl:32b" in user["content"]  # provenance is stated
    assert "do not ask them to" in user["content"]  # don't re-request the attachment
    assert out[0] == msgs[0]  # untouched messages pass through by identity


async def test_input_messages_are_never_mutated():
    cfg = _Cfg()
    msgs = _image_msgs()

    out, _ = await _describe(msgs, cfg, _ScriptedOllama())

    assert msgs[1]["content"][1]["type"] == "image_url"  # caller's copy intact
    assert out is not msgs


async def test_user_question_is_passed_to_the_vision_model():
    """The transcriber gets the question so it details the relevant parts."""
    cfg = _Cfg()
    ollama = _ScriptedOllama()

    await _describe(_image_msgs(), cfg, ollama)

    sent = ollama.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "cannot see images" in sent[0]["content"]
    assert "what is this?" in sent[1]["content"][0]["text"]
    assert sent[1]["content"][1]["image_url"]["url"] == _PNG


async def test_images_across_history_are_all_described_and_numbered():
    cfg = _Cfg()
    ollama = _ScriptedOllama()
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _PNG}}]},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "and this one?"},
                {"type": "image_url", "image_url": {"url": _PNG2}},
            ],
        },
    ]

    out, n = await _describe(msgs, cfg, ollama)

    assert n == 2
    assert len(ollama.calls) == 2
    assert "image 1 of 2" in out[0]["content"]
    assert "image 2 of 2" in out[2]["content"]


# ─── Cache ─────────────────────────────────────────────────────────────


async def test_same_image_is_transcribed_once_across_turns():
    """OWUI resends the data URI every turn; the cache stops re-paying."""
    cfg = _Cfg()
    ollama = _ScriptedOllama()

    first, _ = await _describe(_image_msgs(), cfg, ollama)
    second, n = await _describe(_image_msgs(text="and now?"), cfg, ollama)

    assert len(ollama.calls) == 1  # second turn served from cache
    assert n == 1
    assert "a red bicycle on a lawn" in second[1]["content"]
    assert "and now?" in second[1]["content"]
    assert "a red bicycle on a lawn" in first[1]["content"]


async def test_cache_size_is_bounded():
    cfg = _Cfg(raw={"vision": {"cache_size": 1}})
    ollama = _ScriptedOllama()

    await _describe(_image_msgs(url=_PNG), cfg, ollama)
    await _describe(_image_msgs(url=_PNG2), cfg, ollama)

    assert len(vision._cache) == 1  # first entry evicted


# ─── Fail-soft ─────────────────────────────────────────────────────────


async def test_vision_model_error_degrades_to_a_note():
    cfg = _Cfg()
    ollama = _ScriptedOllama(OllamaError("connection refused"))
    health = HealthTracker()

    out, n = await _describe(_image_msgs(), cfg, ollama, health=health)

    assert n == 0
    assert "could not be read" in out[1]["content"]
    assert "what is this?" in out[1]["content"]  # question still reaches the model
    assert not health.is_healthy("qwen3-vl:32b")  # failure cooled the model down


async def test_empty_description_degrades_to_a_note():
    cfg = _Cfg()
    ollama = _ScriptedOllama({"message": {"content": "   "}})

    out, n = await _describe(_image_msgs(), cfg, ollama)

    assert n == 0
    assert "could not be read" in out[1]["content"]


async def test_no_healthy_vision_model_degrades_to_a_note():
    cfg = _Cfg()
    ollama = _ScriptedOllama()
    health = HealthTracker()
    health.record_failure("qwen3-vl:32b", "down")

    out, n = await _describe(_image_msgs(), cfg, ollama, health=health)

    assert n == 0
    assert ollama.calls == []
    assert "no vision model available" in out[1]["content"]


async def test_remote_url_is_skipped_rather_than_described_blind():
    """`_to_ollama_messages` drops non-inline URLs — the vision model would
    receive no image at all and describe one it never saw."""
    cfg = _Cfg()
    ollama = _ScriptedOllama()

    out, n = await _describe(_image_msgs(url="https://example.com/cat.png"), cfg, ollama)

    assert n == 0
    assert ollama.calls == []
    assert "not an inline attachment" in out[1]["content"]


async def test_images_past_the_per_turn_cap_are_noted_not_described():
    cfg = _Cfg(raw={"vision": {"max_images_per_turn": 1}})
    ollama = _ScriptedOllama()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these"},
                {"type": "image_url", "image_url": {"url": _PNG}},
                {"type": "image_url", "image_url": {"url": _PNG2}},
            ],
        },
    ]

    out, n = await _describe(msgs, cfg, ollama)

    assert n == 1
    assert len(ollama.calls) == 1
    assert "over the 1-image limit" in out[0]["content"]


async def test_pinned_model_overrides_the_pool_pick():
    cfg = _Cfg(raw={"vision": {"model": "llava:34b"}})
    ollama = _ScriptedOllama()

    await _describe(_image_msgs(), cfg, ollama)

    assert ollama.calls[0]["model"] == "llava:34b"


async def test_describe_images_ignores_capability_when_called_directly():
    """The deep path calls it with no single target — a panel of many
    models — so the vl-target shortcut must not apply."""
    cfg = _Cfg()
    ollama = _ScriptedOllama()

    out, n = await describe_images(
        _image_msgs(),
        ollama=ollama, registry=_registry(cfg), health=HealthTracker(),
        gate=FairLocalGate(concurrency=1), cfg=cfg, target_model="qwen3-vl:32b",
        user_question="what is this?", user_id="alice@example.com",
    )

    assert n == 1
    assert isinstance(out[1]["content"], str)


# ─── Sampling: thinking, num_predict, temperature (Phase 38) ───────────

class TestSamplingSettings:
    """Measured on the box 2026-08-04: six keyframes generated 9,486 tokens to
    produce 12,490 characters — 1.3 chars/token where prose runs at ~4. Roughly
    two thirds of the generation never reached `message.content`, and
    `ollama show qwen3-vl:32b` lists `thinking` among its capabilities.

    Generation was 87% of the visual pass, so this is the only lever that
    matters — and it is one the phase-38 plan never listed.
    """

    async def test_nothing_configured_sends_no_sampling_fields(self):
        """The pre-phase-38 request, byte for byte. A deployment that never
        opts in must not have its vision calls quietly change shape."""
        ollama = _ScriptedOllama()
        cfg = _Cfg({"vision": {}})

        await describe_images(
            _image_msgs(), ollama=ollama, registry=_registry(cfg),
            health=HealthTracker(), gate=FairLocalGate(concurrency=1), cfg=cfg,
        )

        assert ollama.calls[0]["think"] is None
        assert ollama.calls[0]["options"] is None

    async def test_think_false_is_forwarded(self):
        ollama = _ScriptedOllama()
        cfg = _Cfg({"vision": {"think": False}})

        await describe_images(
            _image_msgs(), ollama=ollama, registry=_registry(cfg),
            health=HealthTracker(), gate=FairLocalGate(concurrency=1), cfg=cfg,
        )

        assert ollama.calls[0]["think"] is False

    async def test_think_null_is_not_the_same_as_think_false(self):
        """The distinction the whole tri-state exists for. Ollama REJECTS the
        `think` field for a model without the capability rather than ignoring
        it, so `null` has to mean "send no field" — otherwise adding a
        non-thinking model to the `vl` pool breaks its calls outright."""
        ollama = _ScriptedOllama()
        cfg = _Cfg({"vision": {"think": None}})

        await describe_images(
            _image_msgs(), ollama=ollama, registry=_registry(cfg),
            health=HealthTracker(), gate=FairLocalGate(concurrency=1), cfg=cfg,
        )

        assert ollama.calls[0]["think"] is None

    async def test_num_predict_and_temperature_reach_the_options(self):
        ollama = _ScriptedOllama()
        cfg = _Cfg({"vision": {"num_predict": 1024, "temperature": 0.3}})

        await describe_images(
            _image_msgs(), ollama=ollama, registry=_registry(cfg),
            health=HealthTracker(), gate=FairLocalGate(concurrency=1), cfg=cfg,
        )

        assert ollama.calls[0]["options"] == {"num_predict": 1024, "temperature": 0.3}

    async def test_a_zero_temperature_is_not_dropped_as_falsy(self):
        """`if conf.get("temperature")` would silently discard 0.0 — the most
        deterministic setting, and the one someone reaches for first when
        chasing the non-deterministic description lengths."""
        ollama = _ScriptedOllama()
        cfg = _Cfg({"vision": {"temperature": 0.0}})

        await describe_images(
            _image_msgs(), ollama=ollama, registry=_registry(cfg),
            health=HealthTracker(), gate=FairLocalGate(concurrency=1), cfg=cfg,
        )

        assert ollama.calls[0]["options"] == {"temperature": 0.0}

    def test_the_committed_config_turns_thinking_off(self):
        """The measurement that justified this is in the config comment; this
        pins that the value still matches it. Sync deliberately — reading a
        file from an async test trips the blocking-`Path` lint."""
        raw = _load_yaml(Path(__file__).resolve().parent.parent / "config.yaml")
        vision_block = raw["vision"]
        assert vision_block["think"] is False
        assert vision_block["num_predict"] == 1024
