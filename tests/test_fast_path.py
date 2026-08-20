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

    async def chat(self, *, model: str, messages, options=None, tools=None, timeout_s=None, think=None):
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


class TestThinkingOnTheFastPath:
    """2026-08-07. Measured with TOOLS=1 on `qwen3.6:35b`: tool selection was
    3/3 with `think=false`, matching `think=true` and beating the current
    default (2/3 — with reasoning on it sometimes reached for `list_my_files`
    on a question about a named file's contents). 45 eval tokens against 277.
    """

    class _Caps:
        """An ollama fake that records `think` and answers capability probes."""

        def __init__(self, thinking=True, boom=None):
            self.thinking = thinking
            self.boom = boom
            self.thinks: list[object] = []
            self.probes: list[str] = []

        async def thinking_flag(self, model, want):
            self.probes.append(model)
            if self.boom:
                return None
            return want if self.thinking else None

        async def chat(self, *, model, messages, options=None, tools=None,
                       timeout_s=None, think=None):
            self.thinks.append(think)
            return {"message": {"role": "assistant", "content": "hi"}}


    async def test_a_thinking_model_is_told_not_to(self):
        from audrey.pipeline.fast_path import _think
        o = self._Caps(thinking=True)
        assert await _think(o, "qwen3.6:35b", True) is False


    async def test_a_model_that_cannot_think_is_not_sent_the_field(self):
        """⚠️ 3 of the 14 tool_capable_models do not declare `thinking`, and
        sending it to them is a hard error — so a flat False would break every
        chat turn that landed on granite4.1:30b, qwen2.5-coder:32b or
        qwen3-coder-next:latest."""
        from audrey.pipeline.fast_path import _think
        o = self._Caps(thinking=False)
        assert await _think(o, "granite4.1:30b", True) is None


    async def test_the_setting_off_never_probes_at_all(self):
        """Default is off, so an untouched config costs no /api/show."""
        from audrey.pipeline.fast_path import _think
        o = self._Caps(thinking=True)
        assert await _think(o, "qwen3.6:35b", False) is None
        assert o.probes == []


# ─── the two branches carry SEPARATE thinking decisions ────────────────
# Added 2026-08-19. `no_thinking` was one flat flag applied to both branches;
# measurement showed the branches want opposite answers. Tool selection is 5/5
# in all three thinking states (thinking buys nothing, costs ~0.7s per ReAct
# round); prose is qwen3.8 116/125 thinking-on vs 112/125 off (thinking buys
# +4/125 for ~0.7s once). These tests exist so a later "simplification" back to
# one flag fails loudly.


class _ThinkRecordingOllama:
    """Records the `think` value each chat call received."""

    def __init__(self) -> None:
        self.thinks: list[bool | None] = []
        self.probes: list[str] = []

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        self.probes.append(model)
        return want

    async def chat(self, *, model, messages, options=None, tools=None,
                   timeout_s=None, think=None):
        self.thinks.append(think)
        return _resp("hi")


async def _run_prose(**kw):
    """Drive the non-tools branch and return the `think` the model was sent."""
    o = _ThinkRecordingOllama()
    await run_fast_path(
        o, _registry(("a", 100, "local")), HealthTracker(), _gate(),  # type: ignore[arg-type]
        task="general", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5.0, **kw,
    )
    return o.thinks[0]


async def test_prose_branch_can_think_while_the_tool_branch_does_not():
    """The whole point of the split: opposite answers, one call."""
    assert await _run_prose(no_thinking=True, no_thinking_prose=False) is None


async def test_prose_branch_still_honours_an_explicit_no_thinking_prose():
    assert await _run_prose(no_thinking=False, no_thinking_prose=True) is False


async def test_prose_branch_falls_back_to_no_thinking_when_unset():
    """A config predating `no_thinking_prose` keeps exactly its old behaviour."""
    assert await _run_prose(no_thinking=True) is False
    assert await _run_prose(no_thinking=False) is None


def test_the_two_flags_are_not_collapsed_back_into_one():
    """⛔ Guard: both call sites must read a DIFFERENT name.

    A prose-only probe justified the wrong thing once already (see `_think`).
    If someone re-unifies these, the measurement that separated them is lost
    silently — the tool branch would start paying ~0.7s per round for a
    selection accuracy it already had at 5/5.
    """
    import inspect
    from audrey.pipeline import fast_path as fp
    src = inspect.getsource(fp.run_fast_path)
    assert "prose_no_thinking" in src, (
        "the plain-chat branch must use its own resolved flag"
    )
    # The ReAct branch keeps the original name; the prose branch must not.
    prose_call = "_think(ollama, cand.name, prose_no_thinking)"
    react_call = "_think(ollama, spec.name, no_thinking)"
    assert prose_call in src, f"prose branch changed shape: expected {prose_call!r}"
    assert react_call in src, f"ReAct branch changed shape: expected {react_call!r}"
