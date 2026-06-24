"""Tests for the fast path's model selection + bounded fallback.

`run_fast_path` is integration-shaped (Ollama client, gate, health), but the
non-tools branch's fallback logic is the load-bearing new behavior (Phase 16),
so it's worth pinning with light fakes:

  - Happy path: top healthy model answers in one attempt (byte-identical to
    the pre-fallback behavior — one chat call, one dispatch).
  - Fallback: top model raises `OllamaError`, the next healthy model answers;
    the failed model is cooled down, the winner recorded healthy.
  - Exhaustion: every healthy candidate fails → `OllamaError` propagates (the
    graph node's 502 contract holds).
  - The fallback is capped at `_FAST_FALLBACK_LIMIT` — a third healthy model
    is never tried.
  - The tools branch is single-shot: a tool-capable model failing does NOT
    fall back to another model (tool side effects make a blind retry unsafe).

We fake the Ollama client at the `chat`/`chat_stream` method level rather than
the transport level so a per-model success/failure script is easy to express.
"""

from __future__ import annotations

from typing import Any

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.fast_path import (
    _FAST_FALLBACK_LIMIT,
    _healthy_fast_candidates,
    run_fast_path,
)
from audrey.tools.discovery import ToolRegistry, ToolSpec


class _Cfg:
    def __init__(self, model_registry: dict[str, list[dict[str, Any]]]) -> None:
        self.model_registry = model_registry


class _ScriptedOllama:
    """Fake Ollama whose `chat` succeeds or fails per model name.

    `outcomes` maps model name → either a response dict (success) or an
    `OllamaError` instance (raised). Records the call order in `.calls`.
    """

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def chat(self, *, model: str, messages, options=None, tools=None, timeout_s=None):
        self.calls.append(model)
        outcome = self._outcomes[model]
        if isinstance(outcome, OllamaError):
            raise outcome
        return outcome


def _resp(content: str) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 1,
        "eval_count": 1,
    }


def _registry(*models: tuple[str, int, str]) -> ModelRegistry:
    """Build a one-task ('general') registry from (name, priority, location)."""
    return ModelRegistry(_Cfg({
        "general": [
            {"name": n, "priority": p, "location": loc} for n, p, loc in models
        ],
    }))


def _gate() -> FairLocalGate:
    return FairLocalGate(concurrency=1)


# ─── _healthy_fast_candidates ──────────────────────────────────────────


def test_healthy_candidates_orders_by_priority_and_caps():
    registry = _registry(("a", 100, "local"), ("b", 50, "local"), ("c", 10, "local"))
    health = HealthTracker()
    out = _healthy_fast_candidates(registry, health, task="general", limit=2)
    assert [s.name for s in out] == ["a", "b"]  # top 2 by priority, c dropped


def test_healthy_candidates_skips_cooled_down_models():
    registry = _registry(("a", 100, "local"), ("b", 50, "local"))
    health = HealthTracker()
    health.record_failure("a", "boom")  # a is now cooling down
    out = _healthy_fast_candidates(registry, health, task="general", limit=2)
    assert [s.name for s in out] == ["b"]


# ─── run_fast_path: non-tools branch ───────────────────────────────────


async def test_fast_path_happy_path_single_attempt():
    registry = _registry(("a", 100, "local"), ("b", 50, "local"))
    health = HealthTracker()
    ollama = _ScriptedOllama({"a": _resp("hi from a")})

    concrete, resp = await run_fast_path(
        ollama, registry, health, _gate(),  # type: ignore[arg-type]
        task="general", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5.0,
    )

    assert concrete == "a"
    assert resp["message"]["content"] == "hi from a"
    assert ollama.calls == ["a"]  # fallback never engaged
    assert health.is_healthy("a")


async def test_fast_path_falls_back_to_next_healthy_model():
    registry = _registry(("a", 100, "local"), ("b", 50, "local"))
    health = HealthTracker()
    ollama = _ScriptedOllama({
        "a": OllamaError("a is down"),
        "b": _resp("hi from b"),
    })

    concrete, resp = await run_fast_path(
        ollama, registry, health, _gate(),  # type: ignore[arg-type]
        task="general", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5.0,
    )

    assert concrete == "b"
    assert resp["message"]["content"] == "hi from b"
    assert ollama.calls == ["a", "b"]  # tried a first, then fell back to b
    assert not health.is_healthy("a")  # a cooled down
    assert health.is_healthy("b")      # b recorded healthy


async def test_fast_path_raises_when_all_candidates_fail():
    registry = _registry(("a", 100, "local"), ("b", 50, "local"))
    health = HealthTracker()
    ollama = _ScriptedOllama({
        "a": OllamaError("a down"),
        "b": OllamaError("b down"),
    })

    try:
        await run_fast_path(
            ollama, registry, health, _gate(),  # type: ignore[arg-type]
            task="general", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0,
        )
    except OllamaError as e:
        assert "b down" in str(e)  # last error propagates
    else:
        raise AssertionError("expected OllamaError when all candidates fail")

    assert ollama.calls == ["a", "b"]


async def test_fast_path_fallback_capped_at_limit():
    # Three healthy models; only the first two should ever be tried.
    assert _FAST_FALLBACK_LIMIT == 2
    registry = _registry(("a", 100, "local"), ("b", 50, "local"), ("c", 10, "local"))
    health = HealthTracker()
    ollama = _ScriptedOllama({
        "a": OllamaError("a down"),
        "b": OllamaError("b down"),
        "c": _resp("hi from c"),  # would succeed, but must never be reached
    })

    try:
        await run_fast_path(
            ollama, registry, health, _gate(),  # type: ignore[arg-type]
            task="general", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0,
        )
    except OllamaError:
        pass
    else:
        raise AssertionError("expected OllamaError — c is past the cap")

    assert ollama.calls == ["a", "b"]  # c never tried
    assert health.is_healthy("c")      # untouched


async def test_fast_path_raises_when_no_healthy_candidates():
    registry = _registry(("a", 100, "local"))
    health = HealthTracker()
    health.record_failure("a", "cooling")  # nothing healthy
    ollama = _ScriptedOllama({})

    try:
        await run_fast_path(
            ollama, registry, health, _gate(),  # type: ignore[arg-type]
            task="general", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0,
        )
    except OllamaError as e:
        assert "No healthy model" in str(e)
    else:
        raise AssertionError("expected OllamaError when nothing is healthy")

    assert ollama.calls == []


# ─── run_fast_path: tools branch stays single-shot ─────────────────────


async def test_fast_path_tools_branch_does_not_fall_back():
    """A tool-capable model failing must NOT swap to another model — tool
    side effects make a blind cross-model retry unsafe. The error propagates
    after a single attempt (the model is still cooled down for next time)."""
    registry = _registry(("a", 100, "local"), ("b", 50, "local"))
    health = HealthTracker()
    ollama = _ScriptedOllama({
        "a": OllamaError("a down"),
        "b": _resp("would have worked"),  # must never be reached
    })
    tools = ToolRegistry(by_name={
        "web_search": ToolSpec(
            name="web_search", description="d", parameters={"type": "object", "properties": {}},
            server_url="http://t", path="/web_search",
        ),
    })

    try:
        await run_fast_path(
            ollama, registry, health, _gate(),  # type: ignore[arg-type]
            task="general", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0,
            tools=tools, tool_capable_models={"a"},
        )
    except OllamaError as e:
        assert "a down" in str(e)
    else:
        raise AssertionError("expected OllamaError from the single tool attempt")

    assert ollama.calls == ["a"]   # b (non-tool-capable) never tried
    assert not health.is_healthy("a")
