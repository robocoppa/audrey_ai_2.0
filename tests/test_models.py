from __future__ import annotations

from typing import Any

import httpx
import pytest

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry


class _Cfg:
    def __init__(self, model_registry: dict[str, list[dict[str, Any]]]) -> None:
        self.model_registry = model_registry


def _ollama_client(handler) -> OllamaClient:
    return OllamaClient(
        "http://ollama.test",
        default_timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )


# ─── ModelRegistry ─────────────────────────────────────────────────────

def test_model_registry_sorts_candidates_by_priority_and_returns_copy():
    registry = ModelRegistry(_Cfg({
        "general": [
            {"name": "slow", "priority": 10, "location": "local"},
            {"name": "fast", "priority": 90, "location": "cloud"},
        ],
    }))

    candidates = registry.candidates("general")

    assert [spec.name for spec in candidates] == ["fast", "slow"]
    assert candidates[0].location == "cloud"

    candidates.pop()
    assert [spec.name for spec in registry.candidates("general")] == ["fast", "slow"]


def test_model_registry_first_healthy_skips_unhealthy_models():
    registry = ModelRegistry(_Cfg({
        "general": [
            {"name": "primary", "priority": 100, "location": "local"},
            {"name": "fallback", "priority": 50, "location": "cloud"},
        ],
    }))

    spec = registry.first_healthy("general", lambda name: name != "primary")

    assert spec is not None
    assert spec.name == "fallback"


def test_model_registry_rejects_unknown_location():
    with pytest.raises(ValueError, match=r"Invalid model location.*clodu"):
        ModelRegistry(_Cfg({
            "general": [
                {"name": "oops", "priority": 100, "location": "clodu"},
            ],
        }))


def test_model_registry_location_of_returns_declared_location():
    registry = ModelRegistry(_Cfg({
        "general": [
            {"name": "local-model", "priority": 100, "location": "local"},
            {"name": "cloud-model", "priority": 50, "location": "cloud"},
        ],
    }))

    assert registry.location_of("local-model") == "local"
    assert registry.location_of("cloud-model") == "cloud"


def test_model_registry_location_of_defaults_to_local_for_unknown():
    # The deep panel and synthesizer call this with worker names pulled
    # from the pool config. An unknown model — typo, or a model removed
    # from the registry but not from the pool — must default to "local"
    # so it goes through the GPU gate instead of bypassing it as "cloud".
    registry = ModelRegistry(_Cfg({"general": []}))

    assert registry.location_of("never-registered") == "local"


def test_an_unregistered_cloud_tagged_model_is_not_gated_as_local():
    """Ollama's `:cloud` tag is authoritative about where the weights are.

    The motivating defect (2026-08-29): a `:cloud` model reachable through
    `passthrough.allowed_models` but holding no registry slot was gated as
    local, so every request reserved the box's only GPU slot — held for the
    whole stream — for a call that runs entirely off-box. Nothing errored;
    local work just queued behind it.
    """
    registry = ModelRegistry(_Cfg({"general": []}))

    assert registry.location_of("glm-5.3-flash:cloud") == "cloud"
    # The qualified form Ollama also uses.
    assert registry.location_of("qwen3.5:397b-cloud") == "cloud"
    assert registry.location_of("deepseek-v4-flash:0731-cloud") == "cloud"


def test_only_the_tag_decides_not_the_model_name():
    """A local model whose NAME contains "cloud" must stay gated.

    Inspecting the whole string would turn this inference into the kind of
    guess the "unknown -> local" default exists to avoid.
    """
    registry = ModelRegistry(_Cfg({"general": []}))

    assert registry.location_of("cloudy-vision:34b") == "local"
    assert registry.location_of("mycloud:latest") == "local"
    assert registry.location_of("bare-name-no-tag") == "local"


def test_a_declared_location_still_beats_the_tag():
    """Explicit config wins. The tag is a fallback, not an override.

    Without this, a deliberate `location: local` on a cloud-tagged entry
    would be silently ignored.
    """
    registry = ModelRegistry(_Cfg({
        "general": [
            {"name": "odd:cloud", "priority": 100, "location": "local"},
        ],
    }))

    assert registry.location_of("odd:cloud") == "local"


def test_model_registry_location_of_finds_model_across_task_types():
    # A worker can appear in multiple task lists; the lookup walks every
    # task type, so finding the model under any one is enough.
    registry = ModelRegistry(_Cfg({
        "code": [{"name": "shared-model", "priority": 50, "location": "cloud"}],
        "general": [{"name": "other-model", "priority": 50, "location": "local"}],
    }))

    assert registry.location_of("shared-model") == "cloud"


# ─── HealthTracker ─────────────────────────────────────────────────────

def test_health_tracker_cools_down_after_failure_and_success_resets():
    health = HealthTracker(base_cooldown_s=30.0, max_cooldown_s=60.0)

    assert health.is_healthy("model-a")

    health.record_failure("model-a", "timeout")

    assert not health.is_healthy("model-a")
    snapshot = health.snapshot()
    assert snapshot["model-a"]["consecutive_failures"] == 1
    assert 0 < snapshot["model-a"]["cooling_down_for_s"] <= 30.0
    assert snapshot["model-a"]["last_error"] == "timeout"

    health.record_success("model-a")

    assert health.is_healthy("model-a")
    assert "model-a" not in health.snapshot()


def test_health_tracker_exponential_backoff_is_capped():
    health = HealthTracker(base_cooldown_s=5.0, max_cooldown_s=8.0)

    health.record_failure("model-a", "first")
    health.record_failure("model-a", "second")
    health.record_failure("model-a", "third")

    snapshot = health.snapshot()["model-a"]
    assert snapshot["consecutive_failures"] == 3
    assert 0 < snapshot["cooling_down_for_s"] <= 8.0
    assert snapshot["last_error"] == "third"


# ─── OllamaClient ──────────────────────────────────────────────────────

async def test_ollama_chat_wraps_transport_errors_as_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError, match="transport error"):
            await client.chat(model="m", messages=[])
    finally:
        await client.aclose()


async def test_ollama_chat_wraps_http_status_as_ollama_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError, match="/api/chat -> 503"):
            await client.chat(model="m", messages=[])
    finally:
        await client.aclose()


async def test_ollama_chat_wraps_malformed_json_as_ollama_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError, match="invalid JSON response"):
            await client.chat(model="m", messages=[])
    finally:
        await client.aclose()


async def test_ollama_tags_rejects_unexpected_response_shape():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError, match="expected JSON object"):
            await client.tags()
    finally:
        await client.aclose()


async def test_ollama_embed_rejects_vector_count_mismatch():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError, match="expected 2 vectors, got 1"):
            await client.embed(model="embedder", texts=["one", "two"])
    finally:
        await client.aclose()


# ─── /api/show and capabilities ────────────────────────────────────────
#
# `think` cannot be sent blind: Ollama REJECTS the field for a model that does
# not declare `thinking`, rather than ignoring it, so anything choosing the
# flag per model has to ask first. These pin the asking.


async def test_ollama_show_posts_the_model_name():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json={"capabilities": ["completion", "thinking"]})

    client = _ollama_client(handler)
    try:
        body = await client.show("qwen3.6:35b")
    finally:
        await client.aclose()
    assert seen["path"] == "/api/show"
    assert seen["body"] == {"model": "qwen3.6:35b"}
    assert body["capabilities"] == ["completion", "thinking"]


async def test_ollama_capabilities_returns_the_declared_list():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": ["completion", "tools", "thinking"]})

    client = _ollama_client(handler)
    try:
        assert await client.capabilities("m") == ["completion", "tools", "thinking"]
    finally:
        await client.aclose()


async def test_ollama_capabilities_is_empty_when_the_field_is_absent():
    """Absent must read as "do not send the flag", which is the safe
    direction: omitting `think` works on every model, sending it to one that
    cannot think is a hard error."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model_info": {}})

    client = _ollama_client(handler)
    try:
        assert await client.capabilities("m") == []
    finally:
        await client.aclose()


async def test_ollama_capabilities_is_empty_when_the_field_is_not_a_list():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": "thinking"})

    client = _ollama_client(handler)
    try:
        assert await client.capabilities("m") == []
    finally:
        await client.aclose()


async def test_ollama_show_raises_on_an_error_status():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = _ollama_client(handler)
    try:
        with pytest.raises(OllamaError):
            await client.show("no-such-model")
    finally:
        await client.aclose()
