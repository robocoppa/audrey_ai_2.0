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
