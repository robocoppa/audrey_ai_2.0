"""Tests for deep-panel worker preparation (Phase 17 de-dup).

`run_panel` and `run_panel_streaming` now share `_prepare_panel`, which does
the load-bearing setup both paths depend on: healthy-worker selection, the
registry fallback when no pool worker is healthy, subtask→worker assignment,
and coroutine construction. Before Phase 17 this logic was duplicated and had
**no** direct test coverage — the only deep_panel tests were for config
validation / pool-key mapping / timeout selection. This file fills that gap so
the shared helper is pinned through both entry points.

`_prepare_panel` only *builds* the worker coroutines; it doesn't await them.
That's the seam we test — selection + fallback + assignment — without running a
real model. The unawaited coroutines are closed so pytest doesn't warn.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline import deep_panel as dpmod
from audrey.pipeline import graph as gmod
from audrey.pipeline.deep_panel import (
    _log_draft_shape,
    _merge_ledgers,
    _prefix_ledger_ids,
    _prepare_panel,
    pick_panel_timeout,
    pool_key_for,
    run_panel,
    run_panel_streaming,
    select_workers,
)
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.ledger import Claim, ResearchResult, Source
from audrey.pipeline.prompts import DEEP_WORKER_SYSTEM


class _Cfg:
    """Minimal Config stand-in: `_prepare_panel` reads `model_registry`
    (via the registry) and `cfg.raw[pool_key]` (via `select_workers`)."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.model_registry = raw.get("model_registry", {})


def _registry(*models: tuple[str, int, str]) -> ModelRegistry:
    """Build a registry from (name, priority, location) under task 'reasoning'."""
    return ModelRegistry(_Cfg({
        "model_registry": {
            "reasoning": [
                {"name": n, "priority": p, "location": loc} for n, p, loc in models
            ],
        },
    }))


def _cfg_with_pool(workers: list[str], registry_models: tuple, synth: str = "s") -> _Cfg:
    return _Cfg({
        "model_registry": {
            "reasoning": [
                {"name": n, "priority": p, "location": loc} for n, p, loc in registry_models
            ],
        },
        "deep_panel": {
            "reasoning": {"workers": workers, "synthesizer": synth},
        },
    })


def _prep(cfg, registry, health, *, subtasks=None, tools=None, capable=None, messages=None):
    """Call _prepare_panel with sane defaults; return (workers, coros)."""
    return _prepare_panel(
        cfg, object(), registry, health, FairLocalGate(concurrency=1),
        pool_key="deep_panel", task="reasoning",
        messages=messages if messages is not None else [{"role": "user", "content": "q"}],
        subtasks=subtasks or [],
        options={}, timeout_s=5.0, max_workers_cloud=3,
        tools=tools, tool_capable_models=capable,
        react_max_rounds=2, react_compress_after=2, react_max_tool_chars=2000,
        react_dispatch_timeout_s=30.0, react_compress_keep_last=1,
        react_max_web_searches=0, user_id=None,
    )


def _close(coros: list) -> None:
    """Close unawaited coroutines so pytest doesn't warn."""
    for c in coros:
        c.close()


# ─── _prepare_panel: selection ─────────────────────────────────────────


def test_prepare_panel_selects_healthy_pool_workers():
    reg = _registry(("a", 100, "local"), ("b", 50, "local"))
    cfg = _cfg_with_pool(["a", "b"], (("a", 100, "local"), ("b", 50, "local")))
    health = HealthTracker()

    workers, coros = _prep(cfg, reg, health)
    try:
        assert [name for name, _ in workers] == ["a", "b"]
        assert len(coros) == 2
    finally:
        _close(coros)


def test_prepare_panel_skips_unhealthy_pool_worker():
    reg = _registry(("a", 100, "local"), ("b", 50, "local"))
    cfg = _cfg_with_pool(["a", "b"], (("a", 100, "local"), ("b", 50, "local")))
    health = HealthTracker()
    health.record_failure("a", "down")  # a cools down

    workers, coros = _prep(cfg, reg, health)
    try:
        assert [name for name, _ in workers] == ["b"]
    finally:
        _close(coros)


# ─── _prepare_panel: registry fallback ─────────────────────────────────


def test_prepare_panel_falls_back_to_registry_when_no_pool_worker_healthy():
    # Pool names only `a`; `a` is unhealthy. Fallback walks the registry's
    # healthy candidates (top-2): here `b` and `c` (priority order).
    reg = _registry(("a", 100, "local"), ("b", 90, "local"), ("c", 80, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"), ("b", 90, "local"), ("c", 80, "local")))
    health = HealthTracker()
    health.record_failure("a", "down")

    workers, coros = _prep(cfg, reg, health)
    try:
        # Fallback is capped at 2, highest priority first.
        assert [name for name, _ in workers] == ["b", "c"]
    finally:
        _close(coros)


def test_prepare_panel_returns_empty_when_nothing_healthy():
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))
    health = HealthTracker()
    health.record_failure("a", "down")

    workers, coros = _prep(cfg, reg, health)
    assert workers == []
    assert coros == []


# ─── _prepare_panel: subtask assignment ────────────────────────────────


def test_prepare_panel_builds_one_coro_per_worker():
    reg = _registry(("a", 100, "local"), ("b", 50, "local"))
    cfg = _cfg_with_pool(["a", "b"], (("a", 100, "local"), ("b", 50, "local")))
    health = HealthTracker()

    # With subtasks, each worker still gets exactly one coroutine (round-robin
    # over subtasks); coro count tracks worker count, not subtask count.
    workers, coros = _prep(cfg, reg, health, subtasks=["sub1", "sub2", "sub3"])
    try:
        assert len(coros) == len(workers) == 2
    finally:
        _close(coros)


# ─── _prepare_panel: worker role prompt ────────────────────────────────
# Deep workers ran with NO role prompt until 2026-07-21, which is why a cloud
# worker could decline to answer while holding the same evidence its panel-mates
# wrote from — there was nowhere to tell it its output is one draft of several.
# These pin who gets the prompt and where it lands.


def _captured_worker_messages(cfg, registry, health, **kw) -> dict[str, list]:
    """Map model name → the `messages` `_prepare_panel` built for it.

    `_run_one_worker` is stubbed with a *sync* function that records its kwargs
    and hands back a coroutine. It has to be sync: calling an `async def`
    only builds a coroutine object without running the body, so an async stub
    would capture nothing until awaited — and `_prepare_panel` never awaits.
    """
    seen: dict[str, list] = {}

    async def _noop() -> dict:
        return {}

    def _fake_worker(*_args, **kwargs):
        seen[kwargs["model"]] = kwargs["messages"]
        return _noop()

    with patch.object(dpmod, "_run_one_worker", _fake_worker):
        _workers, coros = _prep(cfg, registry, health, **kw)
        _close(coros)
    return seen


def test_worker_role_prompt_only_reaches_tool_capable_workers():
    reg = _registry(("a", 100, "local"), ("b", 50, "local"))
    cfg = _cfg_with_pool(["a", "b"], (("a", 100, "local"), ("b", 50, "local")))

    seen = _captured_worker_messages(
        cfg, reg, HealthTracker(),
        tools=_one_tool_registry(), capable={"a"},
    )

    assert any(DEEP_WORKER_SYSTEM in m.get("content", "") for m in seen["a"])
    # `b` runs a plain one-shot chat: no evidence to reason about, so no prompt.
    assert not any(DEEP_WORKER_SYSTEM in m.get("content", "") for m in seen["b"])


def test_worker_role_prompt_lands_after_the_users_own_system_messages():
    # compose_system_messages fixes the order: incoming system messages first so
    # the user's persona wins on tone, THEN the task role. Prepending would
    # invert that silently.
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))

    seen = _captured_worker_messages(
        cfg, reg, HealthTracker(),
        tools=_one_tool_registry(), capable={"a"},
        messages=[
            {"role": "system", "content": "PERSONA"},
            {"role": "system", "content": "DATETIME"},
            {"role": "user", "content": "q"},
        ],
    )

    roles = [m["role"] for m in seen["a"]]
    contents = [m["content"] for m in seen["a"]]
    assert roles == ["system", "system", "system", "user"]
    assert contents[0] == "PERSONA"
    assert contents[1] == "DATETIME"
    assert contents[2] == DEEP_WORKER_SYSTEM
    assert contents[3] == "q"


def test_no_tool_registry_leaves_messages_untouched():
    # The tool-free path must stay byte-identical: a capable name means nothing
    # when there are no tools to call.
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))
    original = [{"role": "user", "content": "q"}]

    seen = _captured_worker_messages(
        cfg, reg, HealthTracker(), tools=None, capable={"a"}, messages=original,
    )
    assert seen["a"] == original


def test_worker_role_prompt_composes_with_subtask_substitution():
    # Subtask assignment replaces the focal user turn; the role prompt rides
    # along rather than being dropped by whichever branch runs.
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))

    seen = _captured_worker_messages(
        cfg, reg, HealthTracker(),
        tools=_one_tool_registry(), capable={"a"}, subtasks=["sub1"],
    )

    contents = [m["content"] for m in seen["a"]]
    assert DEEP_WORKER_SYSTEM in contents
    assert "sub1" in contents
    assert "q" not in contents  # the original user turn was replaced


# ─── entry-point short-circuit (no workers) ────────────────────────────


async def test_run_panel_returns_empty_when_no_workers():
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))
    health = HealthTracker()
    health.record_failure("a", "down")

    drafts, attempted = await run_panel(
        cfg, object(), reg, health, FairLocalGate(concurrency=1),
        pool_key="deep_panel", task="reasoning",
        messages=[{"role": "user", "content": "q"}], subtasks=[],
        options={}, timeout_s=5.0, max_workers_cloud=3,
    )
    assert drafts == []
    assert attempted == []


async def test_run_panel_streaming_emits_only_final_when_no_workers():
    reg = _registry(("a", 100, "local"))
    cfg = _cfg_with_pool(["a"], (("a", 100, "local"),))
    health = HealthTracker()
    health.record_failure("a", "down")

    events = [
        evt async for evt in run_panel_streaming(
            cfg, object(), reg, health, FairLocalGate(concurrency=1),
            pool_key="deep_panel", task="reasoning",
            messages=[{"role": "user", "content": "q"}], subtasks=[],
            options={}, timeout_s=5.0, max_workers_cloud=3,
        )
    ]
    # Exactly one event — the final sentinel — with empty drafts/attempted.
    assert events == [{"type": "final", "drafts": [], "attempted": []}]


# ─── synth timeout is pool-aware (Phase 23a) ───────────────────────────
#
# Regression guard: the synthesizer must use `pick_panel_timeout(cfg, pool_key)`
# — the same source the panel uses — not the raw `deep_worker` timeout. Before
# Phase 23a both deep paths passed `cfg.timeouts.deep_worker` to synthesis, so a
# cloud-only (`deep_panel_cloud`) request synthesized on 360s while its panel ran
# on `cfg.timeouts.cloud` (240s). These pin the call-site contract so the two
# can't drift again. `pick_panel_timeout` itself is unit-tested in
# `test_config_validation.py`; here we prove the synth node forwards its value.


_HTTP = object()


class _NoTools:
    def all_specs(self) -> list:
        return []

    def __iter__(self):
        return iter([])


async def _captured_synth_timeout_for(virtual_model: str) -> float | None:
    """Build the graph with `synthesize_fn` stubbed, invoke just the synthesize
    node for `virtual_model`, and return the `timeout_s` it forwarded."""
    cfg = get_config()
    ollama = OllamaClient(base_url="http://unused")
    registry = ModelRegistry(cfg)
    health = HealthTracker()
    gate = FairLocalGate(concurrency=1)

    captured: dict[str, Any] = {}

    async def _fake_synth(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["timeout_s"] = kwargs.get("timeout_s")
        return {"final_answer": "ok", "synthesizer_model": "m"}

    with patch.object(gmod, "synthesize_fn", _fake_synth):
        compiled = gmod.build_graph(cfg, ollama, registry, health, gate, _NoTools(), _HTTP)
        node = compiled.nodes["synthesize"].bound
        await node.ainvoke({
            "virtual_model": virtual_model,
            "task_type": "reasoning",
            "messages": [{"role": "user", "content": "q"}],
            "drafts": [{"model": "a", "content": "d"}],
            "subtasks": [],
            "user_id": None,
        })
    await ollama.aclose()
    return captured.get("timeout_s")


async def test_synth_uses_cloud_timeout_for_cloud_pool():
    cfg = get_config()
    expected = pick_panel_timeout(cfg, pool_key_for("audrey_cloud"))
    assert await _captured_synth_timeout_for("audrey_cloud") == expected


async def test_synth_uses_deep_worker_timeout_for_local_pools():
    cfg = get_config()
    # Mixed (`deep_panel`) and local-only (`deep_panel_local`) both hold the GPU
    # gate, so both keep the `deep_worker` budget — unchanged by Phase 23a.
    for vm in ("audrey_deep", "audrey_local"):
        expected = pick_panel_timeout(cfg, pool_key_for(vm))
        assert await _captured_synth_timeout_for(vm) == expected


# ─── reflect routing ends on deterministic no_drafts ───────────────────
#
# Regression guard: `route_after_reflect` must NOT retry when the failure is
# `reflect_reason == "no_drafts"`. That reason means synthesis found zero usable
# drafts — a deterministic dead-panel outcome. Retrying re-runs the same dead
# panel and produces no_drafts again, wasting a deep-panel pass. A retryable
# failure like `too_short` (under the retry budget) must still retry. `reflect()`
# itself is covered in `test_reflect.py`; this pins the *router* decision.


def _route_after_reflect():
    """Pull the compiled graph's `route_after_reflect` conditional-edge fn.

    It's a closure inside `build_graph`, reachable via the LangGraph branch
    registry. Kept in one helper so a LangGraph internals change is a one-line
    fix here, not scattered across tests.
    """
    cfg = get_config()
    ollama = OllamaClient(base_url="http://unused")
    registry = ModelRegistry(cfg)
    compiled = gmod.build_graph(
        cfg, ollama, registry, HealthTracker(), FairLocalGate(concurrency=1),
        _NoTools(), _HTTP,
    )
    branch = compiled.builder.branches["reflect"]["route_after_reflect"]
    return branch.path.func


def test_route_after_reflect_ends_on_no_drafts():
    route = _route_after_reflect()
    # Deterministic failure, retry budget untouched → must end, not retry.
    assert route({
        "reflect_passed": False,
        "reflect_reason": "no_drafts",
        "reflect_attempts": 1,
    }) == "end"


def test_route_after_reflect_still_retries_too_short_under_budget():
    route = _route_after_reflect()
    # Retryable failure under the retry budget → still retries (no regression).
    assert route({
        "reflect_passed": False,
        "reflect_reason": "too_short",
        "reflect_attempts": 1,
    }) == "retry"


def test_route_after_reflect_ends_when_passed():
    route = _route_after_reflect()
    assert route({"reflect_passed": True}) == "end"


def test_route_after_reflect_research_retries_into_research_branch():
    # A retryable failure on audrey_research must re-enter the staged research
    # branch ("retry_research"), not the standard panel ("retry").
    route = _route_after_reflect()
    assert route({
        "reflect_passed": False,
        "reflect_reason": "too_short",
        "reflect_attempts": 1,
        "virtual_model": "audrey_research",
    }) == "retry_research"
    # Non-research deep modes still take the standard retry edge.
    assert route({
        "reflect_passed": False,
        "reflect_reason": "too_short",
        "reflect_attempts": 1,
        "virtual_model": "audrey_deep",
    }) == "retry"


def _route_deep_kind():
    """Pull the compiled graph's `route_deep_kind` planner-fork conditional."""
    cfg = get_config()
    compiled = gmod.build_graph(
        cfg, OllamaClient(base_url="http://unused"), ModelRegistry(cfg),
        HealthTracker(), FairLocalGate(concurrency=1), _NoTools(), _HTTP,
    )
    branch = compiled.builder.branches["planner"]["route_deep_kind"]
    return branch.path.func


def test_route_deep_kind_forks_research_vs_panel():
    route = _route_deep_kind()
    assert route({"virtual_model": "audrey_research"}) == "research"
    assert route({"virtual_model": "audrey_deep"}) == "panel"
    assert route({"virtual_model": "audrey_cloud"}) == "panel"


async def test_node_complexity_forces_deep_for_research():
    # audrey_research must force the deep path in the non-streaming gate, even
    # on a short prompt that would otherwise route fast.
    cfg = get_config()
    ollama = OllamaClient(base_url="http://unused")
    compiled = gmod.build_graph(
        cfg, ollama, ModelRegistry(cfg), HealthTracker(),
        FairLocalGate(concurrency=1), _NoTools(), _HTTP,
    )
    node = compiled.nodes["complexity"].bound
    out = await node.ainvoke({
        "virtual_model": "audrey_research",
        "messages": [{"role": "user", "content": "hi"}],  # short → would be fast
    })
    await ollama.aclose()
    assert out["mode"] == "deep"


# ─── audrey_research staged pipeline (Phase 24) ────────────────────────
#
# The staged executor: research fan-out → verify → write. We stub a fake
# OllamaClient so no real model runs. Researchers are NOT tool-capable here
# (tool_capable_models=None), so `_run_one_worker` takes the single-`chat`
# branch; the writer streams via `chat_stream`.

from audrey.pipeline.deep_panel import (  # noqa: E402
    run_research_pipeline,
    run_research_pipeline_streaming,
    select_researchers,
)


class _FakeOllama:
    """Records calls; returns canned content per model.

    `chat` (researchers + verifier) returns `responses[model]`; `chat_stream`
    (writer) yields it in two chunks. A model in `fail` raises OllamaError.
    """

    def __init__(self, responses: dict[str, str], fail: set[str] | None = None):
        self.responses = responses
        self.fail = fail or set()
        self.chat_models: list[str] = []
        self.stream_models: list[str] = []

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None, think=None):
        # `tools=` is accepted so the fact-checker's run_react loop can call us;
        # we never return tool_calls, so run_react treats the content as final.
        # `format=` is accepted because the ledger structuring passes pin a schema.
        self.chat_models.append(model)
        if model in self.fail:
            from audrey.models.ollama import OllamaError
            raise OllamaError(f"{model} down")
        return {"message": {"content": self.responses.get(model, "")}, "prompt_eval_count": 1, "eval_count": 1}

    async def chat_stream(self, *, model, messages, options=None, timeout_s=0):
        self.stream_models.append(model)
        if model in self.fail:
            from audrey.models.ollama import OllamaError
            raise OllamaError(f"{model} down")
        text = self.responses.get(model, "")
        mid = len(text) // 2
        yield {"message": {"content": text[:mid]}, "done": False}
        yield {"message": {"content": text[mid:]}, "done": True, "prompt_eval_count": 1, "eval_count": 1}


def _research_cfg(
    researchers: list[str], verifier: str, writer: str,
    registry_models: tuple, fallback: str | None = None,
) -> _Cfg:
    body: dict[str, Any] = {"researchers": researchers, "verifier": verifier, "writer": writer}
    if fallback:
        body["fallback_synth"] = fallback
    return _Cfg({
        "model_registry": {
            "reasoning": [
                {"name": n, "priority": p, "location": loc} for n, p, loc in registry_models
            ],
        },
        "deep_panel_research": {"reasoning": body},
    })


async def _run_research(cfg, reg, health, ollama, *, max_cloud=2):
    return await run_research_pipeline(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "tell me about euclid"}],
        options={}, timeout_s=5.0, max_researchers_cloud=max_cloud,
        tools=None, tool_capable_models=None, user_id=None,
    )


async def test_research_pipeline_happy_path():
    reg = _registry(("r1", 100, "cloud"), ("r2", 90, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    cfg = _research_cfg(["r1", "r2"], "v", "w",
                        (("r1", 100, "cloud"), ("r2", 90, "cloud"), ("v", 80, "cloud"), ("w", 70, "local")))
    health = HealthTracker()
    ollama = _FakeOllama({"r1": "fact A", "r2": "fact B", "v": "looks fine", "w": "Euclid was a mathematician."})

    out = await _run_research(cfg, reg, health, ollama)

    assert out["content"] == "Euclid was a mathematician."
    assert out["writer_model"] == "w"
    assert out["error"] == ""
    assert out["research_critique"] == "looks fine"
    assert "fact A" in out["research_findings"] and "fact B" in out["research_findings"]
    assert len(out["drafts"]) == 2
    # Verifier ran via chat; writer streamed.
    assert "v" in ollama.chat_models
    assert ollama.stream_models == ["w"]


async def test_research_pipeline_empty_research_skips_verify_and_flags_writer():
    # All researchers fail → no findings → verify skipped, writer still runs.
    reg = _registry(("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    cfg = _research_cfg(["r1"], "v", "w",
                        (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local")))
    health = HealthTracker()
    ollama = _FakeOllama({"w": "Caveat: unverified. Euclid..."}, fail={"r1"})

    out = await _run_research(cfg, reg, health, ollama)

    assert out["research_findings"] == ""       # no grounding
    assert out["research_critique"] == ""        # verify skipped
    assert "v" not in ollama.chat_models         # verifier never called
    assert out["content"] == "Caveat: unverified. Euclid..."
    assert out["writer_model"] == "w"
    assert out["error"] == ""


async def test_research_pipeline_writer_falls_back():
    reg = _registry(("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"), ("fb", 60, "cloud"))
    cfg = _research_cfg(["r1"], "v", "w",
                        (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"), ("fb", 60, "cloud")),
                        fallback="fb")
    health = HealthTracker()
    # Primary writer fails before any token → fallback writes.
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "fb": "fallback answer"}, fail={"w"})

    out = await _run_research(cfg, reg, health, ollama)

    assert out["content"] == "fallback answer"
    assert out["writer_model"] == "fb"
    assert out["error"] == ""


async def test_research_pipeline_no_writer_configured_errors_gracefully():
    reg = _registry(("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    cfg = _research_cfg(["r1"], "v", "w",
                        (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local")))
    health = HealthTracker()
    # Writer unhealthy, no fallback → no usable candidate.
    health.record_failure("w", "down")
    ollama = _FakeOllama({"r1": "fact", "v": "ok"})

    out = await _run_research(cfg, reg, health, ollama)

    assert out["content"] == ""
    assert out["writer_model"] == "none"
    assert out["error"] == "write_failed"


async def test_select_researchers_caps_cloud():
    reg = _registry(("r1", 100, "cloud"), ("r2", 90, "cloud"), ("r3", 80, "cloud"), ("r4", 70, "local"))
    cfg = _research_cfg(["r1", "r2", "r3", "r4"], "v", "w",
                        (("r1", 100, "cloud"), ("r2", 90, "cloud"), ("r3", 80, "cloud"), ("r4", 70, "local")))
    health = HealthTracker()
    chosen = select_researchers(cfg, reg, health, task="reasoning", max_researchers_cloud=2)
    locs = [loc for _, loc in chosen]
    # 2 cloud (capped) + 1 local = 3 total; the 3rd cloud is dropped.
    assert locs.count("cloud") == 2
    assert locs.count("local") == 1


async def test_research_pipeline_streaming_event_order():
    reg = _registry(("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    cfg = _research_cfg(["r1"], "v", "w",
                        (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local")))
    health = HealthTracker()
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "w": "answer text"})

    types = [
        evt["type"] async for evt in run_research_pipeline_streaming(
            cfg, ollama, reg, health, FairLocalGate(concurrency=1),
            task="reasoning", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0, max_researchers_cloud=2,
            tools=None, tool_capable_models=None, user_id=None,
        )
    ]
    # Stage order: researcher_done → findings_ready → verify_done → write_delta(s) → done.
    assert types[0] == "researcher_done"
    assert "findings_ready" in types
    assert types.index("verify_done") < types.index("write_delta")
    assert types[-1] == "done"


async def test_research_done_event_carries_trace_intermediates():
    # The routes render the opt-in research trace (agentic.debug_research_trace)
    # from the done event — pin that the intermediate keys are present even when
    # the ledger/fact-check stages didn't run (None/"" then, never missing).
    reg = _registry(("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    cfg = _research_cfg(["r1"], "v", "w",
                        (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local")))
    health = HealthTracker()
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "w": "answer text"})

    final = {}
    async for evt in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=None, tool_capable_models=None, user_id=None,
    ):
        if evt["type"] == "done":
            final = evt

    assert final["ledger"] is None          # ledger flag off in this cfg
    assert final["factcheck"] is None       # fact-check stage didn't run
    assert final["dispositions"] == ""      # hedge policy off in this cfg
    assert final["critique"] == "ok"


# ─── fact-check stage (Phase 25) ───────────────────────────────────────
# The optional Stage-3 fact-checker runs via run_react (tool-capable). It only
# fires when a `factchecker` is configured, healthy, tool-capable, and tools
# exist. Its corrections thread into the writer's prompt.

from audrey.tools.discovery import ToolRegistry, ToolSpec  # noqa: E402


def _one_tool_registry() -> ToolRegistry:
    """A registry with a single web_search tool so run_react has something to offer."""
    spec = ToolSpec(
        name="web_search", description="search", parameters={"type": "object", "properties": {}},
        server_url="http://unused", path="/web_search",
    )
    return ToolRegistry(by_name={"web_search": spec})


def _research_cfg_fc(researchers, verifier, factchecker, writer, registry_models):
    return _Cfg({
        "model_registry": {
            "reasoning": [
                {"name": n, "priority": p, "location": loc} for n, p, loc in registry_models
            ],
        },
        "deep_panel_research": {"reasoning": {
            "researchers": researchers, "verifier": verifier,
            "factchecker": factchecker, "writer": writer,
        }},
    })


async def test_factcheck_stage_runs_and_threads_corrections():
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc(["r1"], "v", "fc", "w", models)
    health = HealthTracker()
    # Factchecker returns an actionable correction; writer echoes what it's told.
    ollama = _FakeOllama({
        "r1": "DeepSeek-R1 released Jan 26 2025.",
        "v": "looks ok",
        "fc": "CORRECT: findings say Jan 26, but the official blog shows Jan 20 — use Jan 20 (url)",
        "w": "DeepSeek-R1 was released on January 20, 2025.",
    })

    final = {}
    async for evt in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "when did R1 release"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
    ):
        if evt["type"] == "done":
            final = evt

    # The factchecker ran (via run_react → chat) and its corrections are surfaced.
    assert "fc" in ollama.chat_models
    assert "CORRECT:" in final["corrections"]
    assert final["content"] == "DeepSeek-R1 was released on January 20, 2025."


async def test_factcheck_stage_skipped_when_not_configured():
    # No `factchecker` in the pool → stage skipped entirely; pipeline unchanged.
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg(["r1"], "v", "w", models)
    health = HealthTracker()
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "w": "answer"})

    types = [
        evt["type"] async for evt in run_research_pipeline_streaming(
            cfg, ollama, reg, health, FairLocalGate(concurrency=1),
            task="reasoning", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0, max_researchers_cloud=2,
            tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
        )
    ]
    assert "factcheck_done" not in types  # no factchecker configured → no event


async def test_factcheck_stage_order_when_present():
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc(["r1"], "v", "fc", "w", models)
    health = HealthTracker()
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "fc": "CONFIRMED: fact (src)", "w": "answer"})

    types = [
        evt["type"] async for evt in run_research_pipeline_streaming(
            cfg, ollama, reg, health, FairLocalGate(concurrency=1),
            task="reasoning", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0, max_researchers_cloud=2,
            tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
        )
    ]
    # research → verify → fact-check → write, in that order.
    assert types.index("verify_done") < types.index("factcheck_done") < types.index("write_delta")


# ── `fallback_factcheck`: the role-doubling escape hatch ───────────────
# The factchecker is also a researcher in every shipped pool, so a Stage-1
# failure used to put it in health cooldown and delete Stage 3 outright — with
# no error logged, because the gate skips before any batch dispatches. These
# pin that the gate now SELECTS a model rather than being vetoed by one.

def _research_cfg_fc2(researchers, verifier, factchecker, fallback, writer,
                      registry_models):
    cfg = _research_cfg_fc(researchers, verifier, factchecker, writer, registry_models)
    cfg.raw["deep_panel_research"]["reasoning"]["fallback_factcheck"] = fallback
    return cfg


async def _factcheck_run(cfg, reg, health, ollama, capable):
    types, final = [], {}
    async for evt in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=_one_tool_registry(), tool_capable_models=capable, user_id=None,
    ):
        types.append(evt["type"])
        if evt["type"] == "done":
            final = evt
    return types, final


async def test_factcheck_falls_back_when_the_primary_is_in_health_cooldown():
    # The exact 2026-08-18 shape: `fc` is a researcher AND the factchecker, and
    # it fails during Stage 1. Before the fallback existed, the whole stage
    # vanished; now `fb` — which is NOT a researcher — carries it.
    models = (("fc", 100, "cloud"), ("r2", 95, "cloud"), ("v", 80, "cloud"),
              ("fb", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc2(["fc", "r2"], "v", "fc", "fb", "w", models)
    health = HealthTracker()
    health.record_failure("fc", "model 'fc' is temporarily overloaded")
    ollama = _FakeOllama({"r2": "fact", "v": "ok",
                          "fb": "CONFIRMED: fact (src)", "w": "answer"})

    types, final = await _factcheck_run(cfg, reg, health, ollama, {"fc", "fb"})

    assert "factcheck_done" in types, "the stage was skipped instead of falling back"
    assert "fb" in ollama.chat_models   # the fallback did the work
    assert "fc" not in ollama.chat_models  # and the cooled-down primary was untouched
    assert "CONFIRMED:" in final["corrections"]


async def test_factcheck_falls_back_when_the_primary_is_not_tool_capable():
    # The same selection path, different failed precondition. Worth its own
    # test because a name missing from `fast_path.tool_capable_models` fails
    # the gate exactly like an unhealthy model — silently, and permanently.
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 78, "cloud"),
              ("fb", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc2(["r1"], "v", "fc", "fb", "w", models)
    ollama = _FakeOllama({"r1": "fact", "v": "ok",
                          "fb": "CONFIRMED: fact (src)", "w": "answer"})

    types, _ = await _factcheck_run(cfg, reg, HealthTracker(), ollama, {"fb"})

    assert "factcheck_done" in types
    assert "fb" in ollama.chat_models


async def test_factcheck_still_skips_when_neither_candidate_is_available():
    # The fallback widens the gate; it must not remove it. With both models
    # cooling down there is nothing to dispatch, and the stage stays silent
    # rather than emitting an empty Fact-checking banner.
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 78, "cloud"),
              ("fb", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc2(["r1"], "v", "fc", "fb", "w", models)
    health = HealthTracker()
    health.record_failure("fc", "down")
    health.record_failure("fb", "down")
    ollama = _FakeOllama({"r1": "fact", "v": "ok", "w": "answer"})

    types, final = await _factcheck_run(cfg, reg, health, ollama, {"fc", "fb"})

    assert "factcheck_done" not in types
    assert final["corrections"] == ""
    assert final["content"] == "answer"  # the answer still lands


async def test_the_primary_is_preferred_while_it_is_healthy():
    # The fallback is a degrade path, not a load balancer.
    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 78, "cloud"),
              ("fb", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc2(["r1"], "v", "fc", "fb", "w", models)
    ollama = _FakeOllama({"r1": "fact", "v": "ok",
                          "fc": "CONFIRMED: fact (src)", "w": "answer"})

    _types, _final = await _factcheck_run(cfg, reg, HealthTracker(), ollama,
                                          {"fc", "fb"})

    assert "fc" in ollama.chat_models
    assert "fb" not in ollama.chat_models


# ── Phase 26 Stage 1: ledger merge/prefix helpers ──────────────────────

def test_prefix_ledger_ids_namespaces_claims_and_sources():
    r = ResearchResult(
        claims=[Claim(id="c1", text="x", source_ids=["s1"])],
        sources=[Source(id="s1", title="T", url="https://e.com",
                        source_type="official", supports=["c1"])],
    )
    out = _prefix_ledger_ids(r, "w0_")
    assert out.claims[0].id == "w0_c1"
    assert out.claims[0].source_ids == ["w0_s1"]
    assert out.sources[0].id == "w0_s1"
    assert out.sources[0].supports == ["w0_c1"]


def test_merge_ledgers_keeps_all_claims_dedups_sources_by_url():
    r1 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="a")],
        sources=[Source(id="s1", title="T", url="https://e.com/",
                        source_type="official")],
    ), "w0_")
    r2 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="b")],  # same local id, different worker
        sources=[Source(id="s1", title="T2", url="https://e.com",  # dup URL
                        source_type="news")],
    ), "w1_")
    m = _merge_ledgers([r1, r2])
    # Both workers' claims survive (conflicting claims surface for the checker).
    assert {c.id for c in m.claims} == {"w0_c1", "w1_c1"}
    # Source deduped by normalized URL → only the first kept.
    assert [s.id for s in m.sources] == ["w0_s1"]


def test_merge_ledgers_empty_is_safe():
    m = _merge_ledgers([])
    assert m.claims == []
    assert m.sources == []


def test_merge_ledgers_remaps_claim_refs_of_deduped_sources():
    # The 2026-07-07 transformer failure shape: worker 2's arXiv source
    # matched worker 0's by URL and was dropped, orphaning every w2 claim
    # that cited it — hedge_policy then hedged textbook facts. The dedup must
    # remap the dropped id to the canonical one, fold `supports` across, and
    # let the typed duplicate upgrade a canonical typed `unknown`.
    r0 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="a", source_ids=["s1"])],
        sources=[Source(id="s1", title="Attention Is All You Need",
                        url="https://arxiv.org/abs/1706.03762",
                        source_type="unknown")],
    ), "w0_")
    r2 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="b", source_ids=["vaswani2017"])],
        sources=[Source(id="vaswani2017", title="Vaswani et al. (2017)",
                        url="https://arxiv.org/abs/1706.03762",
                        source_type="reference", supports=["c1"])],
    ), "w2_")
    m = _merge_ledgers([r0, r2])
    assert [s.id for s in m.sources] == ["w0_s1"]
    # The w2 claim now cites the canonical source instead of the dropped id.
    w2_claim = next(c for c in m.claims if c.id == "w2_c1")
    assert w2_claim.source_ids == ["w0_s1"]
    # Supports folded; the typed duplicate upgraded the `unknown` canonical.
    assert "w2_c1" in m.sources[0].supports
    assert m.sources[0].source_type == "reference"


def test_merge_ledgers_leaves_refs_to_kept_sources_alone():
    # Distinct URLs → nothing dropped, nothing remapped; a known canonical
    # type is NOT overwritten by a duplicate's type.
    r0 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="a", source_ids=["s1"])],
        sources=[Source(id="s1", title="T", url="https://e.com",
                        source_type="official")],
    ), "w0_")
    r1 = _prefix_ledger_ids(ResearchResult(
        claims=[Claim(id="c1", text="b", source_ids=["s1"])],
        sources=[Source(id="s1", title="T2", url="https://e.com",  # dup URL
                        source_type="news")],
    ), "w1_")
    m = _merge_ledgers([r0, r1])
    assert next(c for c in m.claims if c.id == "w0_c1").source_ids == ["w0_s1"]
    assert next(c for c in m.claims if c.id == "w1_c1").source_ids == ["w0_s1"]
    assert m.sources[0].source_type == "official"  # not downgraded to news


# ── Phase 26 Stage 2: factcheck-result → corrections rendering ──────────

def _ledger_with(*claims):
    from audrey.pipeline.ledger import ResearchResult
    return ResearchResult(claims=list(claims))


def test_factcheck_corrections_unsupported_becomes_drop():
    from audrey.pipeline.deep_panel import _factcheck_result_to_corrections
    from audrey.pipeline.ledger import Claim, ClaimCheck, FactCheckResult
    led = _ledger_with(Claim(id="c1", text="The Conics survives intact"))
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="unsupported",
                                            notes="Conics is lost")])
    out = _factcheck_result_to_corrections(fc, led)
    assert "DROP:" in out
    assert "Conics" in out


def test_factcheck_corrections_needs_hedge_uses_corrected_text():
    from audrey.pipeline.deep_panel import _factcheck_result_to_corrections
    from audrey.pipeline.ledger import Claim, ClaimCheck, FactCheckResult
    led = _ledger_with(Claim(id="c1", text="released Jan 26"))
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="needs_hedge",
                                            corrected_text="reportedly released in late January")])
    out = _factcheck_result_to_corrections(fc, led)
    assert "CORRECT:" in out
    assert "reportedly released in late January" in out


def test_factcheck_corrections_conflicting_becomes_unverified():
    from audrey.pipeline.deep_panel import _factcheck_result_to_corrections
    from audrey.pipeline.ledger import Claim, ClaimCheck, FactCheckResult
    led = _ledger_with(Claim(id="c1", text="X beats Y"))
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="conflicting",
                                            notes="sources disagree")])
    out = _factcheck_result_to_corrections(fc, led)
    assert "UNVERIFIED:" in out


def test_factcheck_corrections_conflicting_discards_its_corrected_text():
    """`conflicting` is the ONE verdict whose `corrected_text` is thrown away.

    `needs_hedge` renders `CORRECT: use "<corrected_text>"`; `conflicting`
    renders only `UNVERIFIED: … HEDGE it` and drops the rewrite on the floor.
    That asymmetry is deliberate — when sources genuinely disagree, a model
    picking the winner is inventing a resolution it cannot have researched —
    but nothing pinned it, and the old test passed no `corrected_text` at all,
    so the discard was never exercised.

    ⚠️ Run `181202` is the evidence on both sides, and it does not point one
    way. Three `conflicting` verdicts arrived carrying rewrites: two were
    right and useful (Llama's previous release, and "first MoE since Mixtral"
    narrowing a false "first open-weight MoE"), one asserted what a Wikipedia
    page says — which the checker cannot see and did not retrieve. The cost of
    discarding shows up in the answer: the writer hedged a claim the checker
    had already corrected, and shipped "the exact release history is somewhat
    unclear" over a fact that was not unclear. ⚠️ Do NOT start adopting
    `corrected_text` here on that sample — the failure mode it buys is the
    writer stating a fabricated correction plainly, which is strictly worse
    than an over-hedge. Surfacing it as a non-binding suggestion is the option
    worth testing if this recurs.
    """
    from audrey.pipeline.deep_panel import _factcheck_result_to_corrections
    from audrey.pipeline.ledger import Claim, ClaimCheck, FactCheckResult
    led = _ledger_with(Claim(id="c1", text="X shipped in July"))
    fc = FactCheckResult(checks=[ClaimCheck(
        claim_id="c1", verdict="conflicting", notes="sources disagree",
        corrected_text="X shipped in December",
    )])
    out = _factcheck_result_to_corrections(fc, led)
    assert "UNVERIFIED:" in out
    assert "X shipped in July" in out          # the original still reaches the writer
    assert "X shipped in December" not in out  # the rewrite does not
    assert "CORRECT:" not in out


def test_factcheck_corrections_all_supported_is_no_corrections():
    from audrey.pipeline.deep_panel import (
        _NO_CORRECTIONS,
        _factcheck_result_to_corrections,
        _has_corrections,
    )
    from audrey.pipeline.ledger import Claim, ClaimCheck, FactCheckResult
    led = _ledger_with(Claim(id="c1", text="well-known fact"))
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="supported")])
    out = _factcheck_result_to_corrections(fc, led)
    assert out == _NO_CORRECTIONS
    assert _has_corrections(out) is False


class TestConfirmedAndHedgeCanNeverContradict:
    """⚠️ The two blocks the writer receives are separate renderings of one
    `FactCheckResult`. Run `132152` still paired `CONFIRMED: <claim>` with
    `HEDGE: <same claim>` three times after the rule-3 fix, through the paths a
    verdict deliberately does not override. Deferring CONFIRMED to `hedge_policy`
    makes the pairing structurally impossible instead of merely rarer."""

    @staticmethod
    def _both_blocks(led, fc):
        from audrey.pipeline.deep_panel import (
            _factcheck_result_to_corrections,
            _render_dispositions_block,
        )
        return (_factcheck_result_to_corrections(fc, led),
                _render_dispositions_block(led, fc))

    def test_a_supported_claim_the_researcher_flagged_gets_no_confirmed(self):
        # `w0_c9` in run `132152`: the researcher said "from stored memory, not
        # re-verified", the structurer set `needs_hedge`, and the fact-checker
        # returned `supported` anyway. HEDGE is right; CONFIRMED was the bug.
        from audrey.pipeline.ledger import (
            Claim,
            ClaimCheck,
            FactCheckResult,
            ResearchResult,
            Source,
        )
        led = ResearchResult(
            claims=[Claim(id="c1", text="V3.1 shipped in Aug or Sep", risk="high",
                          needs_hedge=True),
                    Claim(id="c2", text="a plain fact", source_ids=["s1"], risk="low")],
            sources=[Source(id="s1", title="docs", url="https://e.com",
                            source_type="official")],
        )
        fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="supported")])
        corrections, dispositions = self._both_blocks(led, fc)
        assert "HEDGE: V3.1 shipped in Aug or Sep" in dispositions
        assert "CONFIRMED" not in corrections

    def test_a_supported_claim_backed_only_by_a_blog_gets_no_confirmed(self):
        # `w1_c18`: risk medium, one `blog` source, verdict `supported`. Rule 5
        # hedges it, so the corrections block must not call it confirmed.
        from audrey.pipeline.ledger import (
            Claim,
            ClaimCheck,
            FactCheckResult,
            ResearchResult,
            Source,
        )
        led = ResearchResult(
            claims=[Claim(id="c1", text="unveiled on July 28", source_ids=["s1"],
                          risk="medium")],
            sources=[Source(id="s1", title="b", url="https://b.com",
                            source_type="blog")],
        )
        fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="supported")])
        corrections, _ = self._both_blocks(led, fc)
        assert "CONFIRMED" not in corrections

    def test_a_supported_and_authoritative_claim_still_gets_its_confirmed(self):
        # The fix must not swallow the case CONFIRMED exists for.
        from audrey.pipeline.ledger import (
            Claim,
            ClaimCheck,
            FactCheckResult,
            ResearchResult,
            Source,
        )
        led = ResearchResult(
            claims=[Claim(id="c1", text="the release is v1.53.1", source_ids=["s1"],
                          risk="high"),
                    # c2 only exists to make the block actionable — a body of
                    # nothing but CONFIRMED collapses to _NO_CORRECTIONS.
                    Claim(id="c2", text="a shakier fact", source_ids=["s1"], risk="high")],
            sources=[Source(id="s1", title="releases", url="https://gh.com/releases",
                            source_type="official")],
        )
        fc = FactCheckResult(checks=[
            ClaimCheck(claim_id="c1", verdict="supported"),
            ClaimCheck(claim_id="c2", verdict="needs_hedge", corrected_text="allegedly"),
        ])
        corrections, dispositions = self._both_blocks(led, fc)
        assert "CONFIRMED: the release is v1.53.1" in corrections
        assert "the release is v1.53.1" not in dispositions

    def test_an_unresolvable_claim_id_does_not_crash_or_confirm(self):
        from audrey.pipeline.ledger import ClaimCheck, FactCheckResult, ResearchResult
        led = ResearchResult(claims=[])
        fc = FactCheckResult(checks=[ClaimCheck(claim_id="ghost", verdict="supported")])
        corrections, _ = self._both_blocks(led, fc)
        assert "CONFIRMED" not in corrections


def test_has_corrections_recognizes_drop():
    from audrey.pipeline.deep_panel import _has_corrections
    assert _has_corrections("DROP: some claim — unsupported") is True


def test_render_claims_includes_ids_and_no_source_marker():
    from audrey.pipeline.deep_panel import _render_claims_for_factcheck
    from audrey.pipeline.ledger import Claim, Source
    claims = [Claim(id="c1", text="grounded", source_ids=["s1"], risk="high"),
              Claim(id="c2", text="ungrounded")]
    sources = [Source(id="s1", title="T", url="https://e.com", source_type="official")]
    out = _render_claims_for_factcheck(claims, sources)
    assert "c1" in out and "c2" in out
    assert "[sources: none]" in out  # c2 has none


def test_render_claims_never_calls_an_unlinked_claim_unsourced():
    # ⚠️ The marker for "we attached no source id" must not read as a finding
    # about the world. It used to say `[no source]`, which is almost verbatim
    # the structuring prompt's definition of `unsupported` ("no source actually
    # supports it") — and `unsupported` DELETES the claim. On run `103331`, 12 of
    # 13 DROPs landed on unlinked claims, with the same fact CONFIRMED elsewhere
    # in the pass wherever its link had survived.
    from audrey.pipeline.deep_panel import _render_claims_for_factcheck
    from audrey.pipeline.ledger import Claim
    out = _render_claims_for_factcheck([Claim(id="c1", text="unlinked")], [])
    assert "[no source]" not in out
    # And the batch carries the rule, not just the neutral wording: the marker
    # is about our ledger, and the claim is judged like any other.
    assert "NOT a finding" in out
    assert "linkage" in out.lower()


def test_render_claims_shows_all_sources_for_a_batch():
    # A batch is judged against the FULL source set, not just the sources its
    # own claims cite — so both sources must render even though the single claim
    # in this batch cites neither.
    from audrey.pipeline.deep_panel import _render_claims_for_factcheck
    from audrey.pipeline.ledger import Claim, Source
    claims = [Claim(id="c9", text="a claim", source_ids=[])]
    sources = [Source(id="s1", title="One", url="https://a.com", source_type="official"),
               Source(id="s2", title="Two", url="https://b.com", source_type="reference")]
    out = _render_claims_for_factcheck(claims, sources)
    assert "s1" in out and "s2" in out


async def test_structure_factcheck_chunks_and_merges():
    # 2026-07-08 eval: a single structuring call over a 75+-claim ancient-bio
    # ledger collapsed to an empty checks array, losing every fact-check verdict.
    # Batching must (a) split the ledger, (b) merge all batches' checks, and
    # (c) survive one batch collapsing to empty — the other batches' verdicts
    # must still reach the caller.
    import json

    from audrey.pipeline.deep_panel import (
        _FACTCHECK_STRUCTURE_BATCH,
        _structure_factcheck,
    )
    from audrey.pipeline.ledger import Claim

    n = _FACTCHECK_STRUCTURE_BATCH + 3  # forces a second batch
    claims = [Claim(id=f"c{i}", text=f"claim {i}", risk="low") for i in range(n)]
    led = _ledger_with(*claims)

    class _BatchFake:
        """Returns a `supported` verdict for every claim_id it is ASKED about,
        except the second batch, which returns an empty checks array (the
        collapse we're guarding against)."""
        def __init__(self):
            self.batch_calls = 0

        async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None, think=None):
            self.batch_calls += 1
            user = messages[-1]["content"]
            asked = [c.id for c in claims if f"- {c.id} (" in user]
            if self.batch_calls == 2:  # simulate a collapsed batch
                asked = []
            body = {"checks": [{"claim_id": cid, "verdict": "supported"} for cid in asked],
                    "fatal_errors": []}
            return {"message": {"content": json.dumps(body)}, "prompt_eval_count": 1, "eval_count": 1}

    fake = _BatchFake()
    fc = await _structure_factcheck(
        fake, HealthTracker(), FairLocalGate(concurrency=1), _Cfg({}),
        model="fc", location="cloud", ledger=led,
        fc_notes="notes", timeout_s=1, user_id=None,
    )
    assert fc is not None
    assert fake.batch_calls == 2  # split into two batches
    checked = {c.claim_id for c in fc.checks}
    # First batch's 15 claims survived even though the second batch collapsed.
    assert len(checked) == _FACTCHECK_STRUCTURE_BATCH
    assert "c0" in checked and f"c{_FACTCHECK_STRUCTURE_BATCH - 1}" in checked


async def test_structure_factcheck_logs_which_batches_answered(caplog):
    # ⚠️ Batching stopped one collapsed call from zeroing the rest — and thereby
    # hid how often a call collapses. Run `171922` reported "30 checks" on one
    # case and "0 checks" on the other; the batch boundaries showed the truth was
    # 2 of 5 batches answering and 0 of 6. A FAILED batch already logs its reason;
    # a batch that parses fine and returns an empty `checks` array logged nothing.
    import json
    import logging

    from audrey.pipeline.deep_panel import (
        _FACTCHECK_STRUCTURE_BATCH,
        _structure_factcheck,
    )
    from audrey.pipeline.ledger import Claim

    n = _FACTCHECK_STRUCTURE_BATCH + 3
    claims = [Claim(id=f"c{i}", text=f"claim {i}", risk="low") for i in range(n)]
    led = _ledger_with(*claims)

    class _EmptySecondBatch:
        def __init__(self):
            self.batch_calls = 0

        async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None, think=None):
            self.batch_calls += 1
            user = messages[-1]["content"]
            asked = [c.id for c in claims if f"- {c.id} (" in user]
            if self.batch_calls == 2:
                # Parses cleanly, answers nothing — the silent case.
                body = {"checks": [], "fatal_errors": ["c1 and c2 disagree"]}
            else:
                body = {"checks": [{"claim_id": cid, "verdict": "supported"} for cid in asked],
                        "fatal_errors": []}
            return {"message": {"content": json.dumps(body)}, "prompt_eval_count": 1, "eval_count": 1}

    with caplog.at_level(logging.INFO, logger="audrey.pipeline.deep_panel"):
        await _structure_factcheck(
            _EmptySecondBatch(), HealthTracker(), FairLocalGate(concurrency=1), _Cfg({}),
            model="fc", location="cloud", ledger=led,
            fc_notes="notes", timeout_s=1, user_id=None,
        )
    line = next(m for m in caplog.messages if "factcheck batches" in m)
    assert "1/2 answered" in line          # one batch answered, one did not
    assert "EMPTY fatal=1" in line         # and the silent one is named EMPTY
    assert f"c0..c{_FACTCHECK_STRUCTURE_BATCH - 1} ok:" in line  # id span present


async def test_structure_factcheck_failure_lines_name_their_batch(caplog):
    # ⚠️ The caller's summary says a batch FAILED but not why; these lines say why
    # but used to omit which. Localising `[w1_c34..w1_c48 FAILED]` on run `173826`
    # meant correlating three log lines by timestamp. Both greps must stand alone.
    import logging

    from audrey.models.ollama import OllamaError
    from audrey.pipeline.deep_panel import _structure_factcheck
    from audrey.pipeline.ledger import Claim

    claims = [Claim(id=f"c{i}", text=f"claim {i}", risk="low") for i in range(3)]
    led = _ledger_with(*claims)

    class _Boom:
        async def chat(self, **kw):
            raise OllamaError("upstream timed out")

    class _Garbage:
        async def chat(self, **kw):
            return {"message": {"content": "not json at all"},
                    "prompt_eval_count": 1, "eval_count": 1}

    for fake, needle in ((_Boom(), "FAILED (transport)"),
                         (_Garbage(), "FAILED (unusable JSON)")):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="audrey.pipeline.deep_panel"):
            await _structure_factcheck(
                fake, HealthTracker(), FairLocalGate(concurrency=1), _Cfg({}),
                model="fc", location="cloud", ledger=led,
                fc_notes="notes", timeout_s=1, user_id=None,
            )
        line = next(m for m in caplog.messages if needle in m)
        assert "c0..c2" in line, f"{needle} line does not name its batch: {line}"


async def test_empty_content_is_logged_with_enough_to_tell_thinking_apart(caplog):
    # ⚠️ Run `173826` lost a batch to `len=0, head='', tail=''` — a 200 OK with no
    # content. That reads identically whether the model thought its whole budget
    # away (Ollama keeps reasoning in `message.thinking`, which never reaches
    # `content`) or genuinely returned nothing, and the two need opposite fixes.
    import logging

    from audrey.pipeline.deep_panel import _structure_factcheck
    from audrey.pipeline.ledger import Claim

    led = _ledger_with(Claim(id="c1", text="a claim", risk="low"))

    class _AllThinkingNoContent:
        async def chat(self, **kw):
            return {"message": {"content": "", "thinking": "x" * 8213},
                    "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 4096}

    with caplog.at_level(logging.INFO, logger="audrey.pipeline.deep_panel"):
        await _structure_factcheck(
            _AllThinkingNoContent(), HealthTracker(), FairLocalGate(concurrency=1),
            _Cfg({}), model="fc", location="cloud", ledger=led,
            fc_notes="notes", timeout_s=1, user_id=None,
        )
    line = next(m for m in caplog.messages if "unusable JSON" in m)
    assert "len=0" in line
    assert "thinking=8213" in line     # the discriminator
    assert "done_reason='stop'" in line
    assert "eval_count=4096" in line


def test_empty_content_diag_survives_a_response_missing_every_field():
    # Fail-soft: a diagnostic that raises turns a logged failure into a crash.
    from audrey.pipeline.deep_panel import _empty_content_diag
    assert "thinking=0" in _empty_content_diag({})
    assert "thinking=0" in _empty_content_diag({"message": None})
    assert "thinking=0" in _empty_content_diag({"message": {"content": ""}})


def test_factcheck_structure_prompt_demands_a_verdict_for_every_claim():
    # ⚠️ "Put any claim that contradicts another claim in `fatal_errors`" read as
    # an instruction to put the claim ID there INSTEAD of giving it a verdict —
    # run `171922` returned `fatal_errors: ["w1_c19", "w1_c46"]` with zero checks,
    # and the writer got `UNVERIFIED: contradiction — w1_c19`, which names a
    # conflict without saying what it is.
    from audrey.pipeline.prompts import FACTCHECK_STRUCTURE_SYSTEM
    assert "EVERY claim in the list gets its own entry in `checks`" in FACTCHECK_STRUCTURE_SYSTEM
    assert "never bare claim ids" in FACTCHECK_STRUCTURE_SYSTEM
    assert "never a substitute for a verdict" in FACTCHECK_STRUCTURE_SYSTEM


def test_factcheck_structure_prompt_bars_deleting_a_claim_for_a_lost_link():
    # `unsupported` is the only verdict the writer acts on by DELETING, so its
    # bar is contradiction — not absence, and above all not `[sources: none]`,
    # which describes our own ledger. The prose `FACTCHECK_SYSTEM` already got
    # this right; the policy was being lost at the structuring pass, which is
    # the one that emits the machine-readable verdict.
    from audrey.pipeline.prompts import FACTCHECK_STRUCTURE_SYSTEM
    assert "CONTRADICT" in FACTCHECK_STRUCTURE_SYSTEM
    assert "[sources: none]" in FACTCHECK_STRUCTURE_SYSTEM
    assert "could not confirm is NOT unsupported" in FACTCHECK_STRUCTURE_SYSTEM


def test_factcheck_structure_prompt_requires_a_real_difference_for_a_conflict():
    # Run `103331` reported "w0_c16 dates Tokio 1.53.0 to July 17, 2026 but w1_c2
    # ALSO dates it to July 17, 2026 …" — agreement filed as a conflict, and
    # across two different releases (1.53.0 vs 1.53.1). Every `fatal_errors`
    # sentence becomes an UNVERIFIED line, so a false one hedges a fact two
    # researchers independently confirmed.
    from audrey.pipeline.prompts import FACTCHECK_STRUCTURE_SYSTEM
    assert "SAME entity" in FACTCHECK_STRUCTURE_SYSTEM
    assert "actually DIFFER" in FACTCHECK_STRUCTURE_SYSTEM
    assert "AGREE" in FACTCHECK_STRUCTURE_SYSTEM


# ── Worker-reply think-stripping ───────────────────────────────────────
def test_strip_think_dangling_close_keeps_final():
    # The observed leak shape: opening tag missing, reasoning (a full draft)
    # before a dangling </think>, the real draft after — keep only the after.
    from audrey.pipeline.deep_panel import _strip_think
    assert _strip_think("draft v1 …</think>final draft") == "final draft"


def test_strip_think_wellformed_block_removed():
    from audrey.pipeline.deep_panel import _strip_think
    assert _strip_think("<think>hmm</think>the answer").strip() == "the answer"


def test_strip_think_all_think_falls_back_to_original():
    # A reply wrapped entirely in think tags: an empty draft reads as a
    # dropped worker, so return the original rather than "".
    from audrey.pipeline.deep_panel import _strip_think
    assert _strip_think("<think>only reasoning</think>") == "<think>only reasoning</think>"


def test_strip_think_plain_content_untouched():
    from audrey.pipeline.deep_panel import _strip_think
    assert _strip_think("no tags here") == "no tags here"


# ── Stage 3: deterministic Sources block ───────────────────────────────
def _grounded_ledger():
    """A ledger with two claims each linked to a distinct usable-URL source."""
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    return ResearchResult(
        claims=[Claim(id="c1", text="Euclid ~300 BCE", source_ids=["s1"]),
                Claim(id="c2", text="Elements has 13 books", source_ids=["s2"])],
        sources=[Source(id="s1", title="MacTutor",
                        url="https://mathshistory.st-andrews.ac.uk/Euclid/", source_type="reference"),
                 Source(id="s2", title="Britannica",
                        url="https://www.britannica.com/biography/Euclid", source_type="reference")],
    )


def test_sources_block_lists_surviving_sources():
    from audrey.pipeline.deep_panel import _render_sources_block
    out = _render_sources_block(_grounded_ledger(), None)
    assert out.startswith("\n\n## Sources\n")
    assert "[MacTutor](https://mathshistory.st-andrews.ac.uk/Euclid/)" in out
    assert "Britannica" in out


def test_sources_block_drops_unsupported_claim_source():
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import ClaimCheck, FactCheckResult
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="unsupported"),
                                 ClaimCheck(claim_id="c2", verdict="supported")])
    out = _render_sources_block(_grounded_ledger(), fc)
    # s1 backed only the dropped c1 → gone; s2 (Britannica) survives.
    assert "MacTutor" not in out
    assert "Britannica" in out


def test_sources_block_omitted_when_no_ledger():
    from audrey.pipeline.deep_panel import _render_sources_block
    assert _render_sources_block(None, None) == ""


def test_sources_block_omitted_when_no_usable_url():
    # A source with a bare title / empty URL must not produce an empty header.
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="x", source_ids=["s1"])],
        sources=[Source(id="s1", title="Some Book", url="", source_type="unknown")])
    assert _render_sources_block(led, None) == ""


def test_sources_block_falls_back_to_all_when_no_linkage():
    # Models often skip source_ids/supports; rather than render nothing, list
    # the sources that were consulted.
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="x")],  # no source_ids
        sources=[Source(id="s1", title="T", url="https://e.com", source_type="news")])
    out = _render_sources_block(led, None)
    assert "[T](https://e.com)" in out


def test_sources_block_garbage_linkage_still_falls_back():
    # source_ids that resolve to NO real source (unrepairable title fragments)
    # must not count as linkage — a non-empty-but-useless keep-set was silently
    # defeating the render-all fallback (2026-07-06 `current-rust-async`).
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="x", source_ids=["Glommio repository"])],
        sources=[Source(id="s1", title="T", url="https://e.com", source_type="news")])
    out = _render_sources_block(led, None)
    assert "[T](https://e.com)" in out


def test_sources_block_falls_back_when_kept_source_has_no_usable_url():
    # tech-transformer-attention (2026-07-15 trace run): a claim linked the one
    # URL-less "Search result" source, so `keep` was non-empty and the render-all
    # fallback DIDN'T fire — even though the ledger also held a real arXiv URL that
    # simply wasn't linked to a surviving claim. The answer rendered `sources:0`
    # despite having a citable source. The fallback must fire when NO kept source
    # has a usable URL, not only when `keep` is empty.
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="x", source_ids=["s_search"])],  # links the URL-less one
        sources=[
            Source(id="s_search", title="Search result", url="", source_type="reference"),
            Source(id="s1", title="Attention Is All You Need",
                   url="https://arxiv.org/abs/1706.03762", source_type="primary_paper"),
        ])
    out = _render_sources_block(led, None)
    assert "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" in out


def test_sources_block_dedups_by_url_and_caps():
    from audrey.pipeline.deep_panel import _MAX_SOURCES_RENDERED, _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    # 12 sources, two of them the same URL with a trailing slash → 11 unique,
    # capped to _MAX_SOURCES_RENDERED.
    sources = [Source(id=f"s{i}", title=f"T{i}", url=f"https://e{i}.com", source_type="news")
               for i in range(12)]
    sources.append(Source(id="dup", title="Dup", url="https://e0.com/", source_type="news"))
    led = ResearchResult(claims=[Claim(id="c1", text="x")], sources=sources)
    out = _render_sources_block(led, None)
    assert out.count("\n- ") == _MAX_SOURCES_RENDERED
    assert out.count("https://e0.com") == 1  # the trailing-slash dup folded in


def test_sources_block_ranks_authoritative_over_weak_at_cap():
    from audrey.pipeline.deep_panel import _MAX_SOURCES_RENDERED, _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    # 8 weak sources listed FIRST, then 1 authoritative one last. Without ranking
    # the cap would drop the authoritative source; with ranking it must survive.
    weak = [Source(id=f"w{i}", title=f"Blog {i}", url=f"https://blog{i}.com",
                   source_type="blog")
            for i in range(_MAX_SOURCES_RENDERED)]
    auth = Source(id="auth", title="Stanford", url="https://plato.stanford.edu/x",
                  source_type="reference")
    led = ResearchResult(claims=[Claim(id="c1", text="x")], sources=[*weak, auth])
    out = _render_sources_block(led, None)
    assert out.count("\n- ") == _MAX_SOURCES_RENDERED
    assert "Stanford" in out  # authoritative source kept despite being listed last
    assert "Blog 7" not in out  # the lowest-priority weak one got dropped instead


def test_sources_block_stable_within_same_tier():
    from audrey.pipeline.deep_panel import _render_sources_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    # Same source_type → ledger order preserved (stable sort, no reshuffle).
    led = ResearchResult(
        claims=[Claim(id="c1", text="x")],
        sources=[Source(id="s1", title="First", url="https://a.com", source_type="reference"),
                 Source(id="s2", title="Second", url="https://b.com", source_type="reference")],
    )
    out = _render_sources_block(led, None)
    assert out.index("First") < out.index("Second")


def test_source_rank_tiers():
    from audrey.pipeline.deep_panel import _source_rank
    # authoritative tier
    for st in ("official", "primary_paper", "scholarly", "reference"):
        assert _source_rank(st) == 3
    assert _source_rank("news") == 2
    assert _source_rank("blog") == _source_rank("company_claim") == 1
    assert _source_rank("unknown") == 0
    assert _source_rank("garbage_offenum") == 0  # off-enum sorts to bottom


def test_usable_url_rejects_non_http():
    from audrey.pipeline.deep_panel import _usable_url
    assert _usable_url("https://e.com") is True
    assert _usable_url("http://e.com") is True
    assert _usable_url("") is False
    assert _usable_url("ftp://e.com") is False
    assert _usable_url("just a title") is False
    assert _usable_url("https://") is False


class _LedgerOllama(_FakeOllama):
    """A _FakeOllama that returns ledger JSON for the two structuring passes
    (identified by their system prompt) so the Stage-3 Sources block has a real
    ledger to render. Everything else behaves like the base fake."""

    def __init__(self, responses, research_json: str, factcheck_json: str):
        super().__init__(responses)
        self._research_json = research_json
        self._factcheck_json = factcheck_json

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None, think=None):
        sys = (messages[0].get("content", "") if messages else "").lower()
        if "claim/source ledger" in sys:  # RESEARCH_STRUCTURE_SYSTEM
            self.chat_models.append(model)
            return {"message": {"content": self._research_json}, "prompt_eval_count": 1, "eval_count": 1}
        if "per-claim verdict" in sys:  # FACTCHECK_STRUCTURE_SYSTEM
            self.chat_models.append(model)
            return {"message": {"content": self._factcheck_json}, "prompt_eval_count": 1, "eval_count": 1}
        return await super().chat(model=model, messages=messages, options=options,
                                  timeout_s=timeout_s, tools=tools, format=format)


def _research_cfg_fc_ledger(researchers, verifier, factchecker, writer, registry_models):
    cfg = _research_cfg_fc(researchers, verifier, factchecker, writer, registry_models)
    cfg.raw["agentic"] = {"research_ledger": {"enabled": True}}
    return cfg


async def test_sources_block_reaches_stream_when_ledger_present():
    # End-to-end: a grounded ledger with usable URLs → the writer answer is
    # followed by a `## Sources` block in the final content + a write_delta.
    import json as _json

    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc_ledger(["r1"], "v", "fc", "w", models)
    health = HealthTracker()
    research_json = _json.dumps({
        "summary_notes": "n",
        "claims": [{"id": "c1", "text": "Euclid ~300 BCE", "source_ids": ["s1"], "risk": "low"}],
        "sources": [{"id": "s1", "title": "MacTutor",
                     "url": "https://mathshistory.st-andrews.ac.uk/Euclid/",
                     "source_type": "reference", "supports": ["c1"]}],
    })
    factcheck_json = _json.dumps({"checks": [{"claim_id": "c1", "verdict": "supported"}]})
    ollama = _LedgerOllama(
        {"r1": "Euclid lived around 300 BCE.", "v": "ok",
         "fc": "CONFIRMED: Euclid ~300 BCE", "w": "Euclid lived around 300 BCE."},
        research_json=research_json, factcheck_json=factcheck_json,
    )

    final = {}
    deltas = []
    async for evt in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "who was Euclid"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
    ):
        if evt["type"] == "write_delta":
            deltas.append(evt["text"])
        elif evt["type"] == "done":
            final = evt

    assert "## Sources" in final["content"]
    assert "MacTutor" in final["content"]
    assert any("## Sources" in d for d in deltas)  # streamed, not just in final


async def test_factcheck_log_separates_drops_that_landed_on_unlinked_claims(caplog):
    # The prompt rule is the fix; this counter is how we find out whether it
    # held. A DROP deletes the claim outright, and an unlinked claim is unlinked
    # because of OUR structurer — so `[N unlinked]` is the rate at which we
    # delete for the wrong reason. It must count only the intersection: a drop
    # on a linked claim is an ordinary fact-check judgement and does not belong
    # in this number, or the instrument reports the pathology as permanent.
    import json as _json
    import logging

    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc_ledger(["r1"], "v", "fc", "w", models)
    research_json = _json.dumps({
        "summary_notes": "n",
        "claims": [
            {"id": "c1", "text": "linked and dropped", "source_ids": ["s1"], "risk": "low"},
            {"id": "c2", "text": "unlinked and dropped", "source_ids": [], "risk": "low"},
            {"id": "c3", "text": "unlinked but kept", "source_ids": [], "risk": "low"},
        ],
        "sources": [{"id": "s1", "title": "MacTutor",
                     "url": "https://mathshistory.st-andrews.ac.uk/Euclid/",
                     "source_type": "reference", "supports": ["c1"]}],
    })
    # Ids are namespaced per worker (`w0_`) before the fact-check sees them.
    factcheck_json = _json.dumps({"checks": [
        {"claim_id": "w0_c1", "verdict": "unsupported"},
        {"claim_id": "w0_c2", "verdict": "unsupported"},
        {"claim_id": "w0_c3", "verdict": "needs_hedge"},
    ]})
    ollama = _LedgerOllama(
        {"r1": "notes", "v": "ok", "fc": "notes", "w": "answer"},
        research_json=research_json, factcheck_json=factcheck_json,
    )

    with caplog.at_level(logging.INFO, logger="audrey.pipeline.deep_panel"):
        async for _evt in run_research_pipeline_streaming(
            cfg, ollama, reg, HealthTracker(), FairLocalGate(concurrency=1),
            task="reasoning", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5.0, max_researchers_cloud=2,
            tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
        ):
            pass

    line = next((r.getMessage() for r in caplog.records
                 if "factcheck ledger" in r.getMessage()), None)
    assert line is not None, "the factcheck ledger summary never logged"
    # 2 drops total, of which exactly 1 sat on an unlinked claim.
    assert "2 drop [1 unlinked]" in line, line


# ── Stage 4: deterministic hedging dispositions ────────────────────────
def test_dispositions_block_lists_only_action_claims_with_framing():
    # Plain claims are NOT enumerated — the framing line covers them; only the
    # action-bearing (attribute/hedge) claims get a line.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="R1 released 2025-01-20", source_ids=["s1"], risk="low"),
                Claim(id="c2", text="Maverick beats GPT-4o", source_ids=["s2"], risk="medium")],
        sources=[Source(id="s1", title="DeepSeek docs", url="https://e.com",
                        source_type="official"),
                 Source(id="s2", title="Meta blog", url="https://m.com",
                        source_type="company_claim")],
    )
    out = _render_dispositions_block(led, None)
    assert out.startswith("State every claim plainly EXCEPT")
    assert "ATTRIBUTE TO SOURCE: Maverick beats GPT-4o" in out
    # the plain claim is folded into the framing line, not enumerated
    assert "R1 released 2025-01-20" not in out


def test_dispositions_block_skips_dropped_claims():
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import (
        Claim,
        ClaimCheck,
        FactCheckResult,
        ResearchResult,
        Source,
    )
    # c1 plain (kept), c2 a company_claim that would attribute — but it's dropped,
    # so the only action line is gone → all-plain → suppressed.
    led = ResearchResult(
        claims=[Claim(id="c1", text="kept", source_ids=["s1"], risk="low"),
                Claim(id="c2", text="dropped vendor claim", source_ids=["s2"], risk="low")],
        sources=[Source(id="s1", title="T", url="https://e.com", source_type="official"),
                 Source(id="s2", title="M", url="https://m.com", source_type="company_claim")],
    )
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c2", verdict="unsupported")])
    out = _render_dispositions_block(led, fc)
    assert "dropped vendor claim" not in out  # dropped claim never appears
    assert out == ""  # its attribution was the only action line → all-plain → omit


def test_dispositions_block_suppressed_when_all_hedge():
    # An ungrounded answer: every claim hedges (no authoritative source) → the
    # block carries no signal beyond blanket caution, so it's omitted (this is
    # the recursion-control over-hedging fix).
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="recursion calls itself", source_ids=["s1"], risk="low"),
                Claim(id="c2", text="needs a base case", source_ids=["s1"], risk="low")],
        sources=[Source(id="s1", title="a blog", url="https://b.com", source_type="blog")],
    )
    assert _render_dispositions_block(led, None) == ""


def test_dispositions_block_suppressed_when_starved_hedge_or_cite():
    # No ledger source has a usable URL → nothing can be "cited strongly", so
    # hedge_or_cite_strongly degenerates to blanket caution and counts toward
    # the all-hedge suppression (the 44-line HEDGE wall from the 2026-07-06
    # search-starved run slipped through this gap).
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="high-risk claim", risk="high"),
                Claim(id="c2", text="unbacked claim", risk="low")],
        sources=[Source(id="s1", title="Tokio official site", url="tokio.rs",
                        source_type="official")])  # scheme-less → not usable
    assert _render_dispositions_block(led, None) == ""


def test_dispositions_block_renders_hedge_or_cite_when_citable():
    # With a usable-URL source in the ledger, hedge_or_cite_strongly keeps its
    # non-hedge signal (the writer CAN cite strongly) → block renders.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="high-risk claim", risk="high"),
                Claim(id="c2", text="plain fact", source_ids=["s1"], risk="low")],
        sources=[Source(id="s1", title="ref", url="https://e.com",
                        source_type="reference")])
    out = _render_dispositions_block(led, None)
    assert "HEDGE (unless a strong source backs it): high-risk claim" in out


def test_dispositions_block_does_not_hedge_what_the_factchecker_confirmed():
    # ⚠️ The two blocks the writer receives are built by different functions from
    # the same FactCheckResult. `_factcheck_result_to_corrections` renders a
    # `supported` verdict as "CONFIRMED: <claim>"; this block must not then render
    # "HEDGE" for the same sentence. Run `113119` shipped exactly that pairing on
    # the Tokio release claim and the writer obeyed the hedge, downgrading a date
    # verified against the official releases page to "around mid-July 2026".
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import (
        Claim,
        ClaimCheck,
        FactCheckResult,
        ResearchResult,
        Source,
    )
    led = ResearchResult(
        claims=[Claim(id="c1", text="Tokio's latest release is v1.53.1",
                      source_ids=["s1"], risk="high"),
                Claim(id="c2", text="a vendor benchmark", source_ids=["s2"], risk="low")],
        sources=[Source(id="s1", title="releases", url="https://github.com/t/releases",
                        source_type="official"),
                 Source(id="s2", title="M", url="https://m.com",
                        source_type="company_claim")],
    )
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="supported")])
    out = _render_dispositions_block(led, fc)
    # c2 keeps the block alive, so an empty string can't be what passes this test.
    assert "ATTRIBUTE TO SOURCE: a vendor benchmark" in out
    assert "Tokio's latest release is v1.53.1" not in out


def test_dispositions_block_still_hedges_an_unchecked_high_risk_claim():
    # The checker samples. A high-risk claim it never looked at must keep its
    # hedge — the exemption is for verdicts, not for optimism.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import (
        Claim,
        ClaimCheck,
        FactCheckResult,
        ResearchResult,
        Source,
    )
    led = ResearchResult(
        claims=[Claim(id="c1", text="the verified fact", source_ids=["s1"], risk="high"),
                Claim(id="c2", text="the unsampled fact", source_ids=["s1"], risk="high")],
        sources=[Source(id="s1", title="ref", url="https://e.com",
                        source_type="official")],
    )
    fc = FactCheckResult(checks=[ClaimCheck(claim_id="c1", verdict="supported")])
    out = _render_dispositions_block(led, fc)
    assert "HEDGE (unless a strong source backs it): the unsampled fact" in out
    assert "the verified fact" not in out


def test_dispositions_block_dedups_near_identical_claims():
    # Three workers each contribute the same fact; the wall gets ONE line for
    # it, not three. Normalization ignores case/punctuation differences.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="w0_c1", text="Euclid flourished c. 300 BCE.",
                      source_ids=["s1"], risk="high"),
                Claim(id="w1_c1", text="euclid flourished c 300 bce",
                      source_ids=["s1"], risk="high"),
                Claim(id="w2_c1", text="Euclid flourished C. 300 BCE",
                      source_ids=["s1"], risk="high"),
                Claim(id="w0_c2", text="plain fact", source_ids=["s1"], risk="low")],
        sources=[Source(id="s1", title="ref", url="https://e.com",
                        source_type="reference")])
    out = _render_dispositions_block(led, None)
    assert out.count("Euclid flourished") == 1
    # Distinct claims are untouched (the plain one folds into the framing line).
    assert out.startswith("State every claim plainly EXCEPT")


def test_dispositions_block_dedup_does_not_change_suppression():
    # All-hedge stays suppressed even when duplicates would have deduped —
    # the suppression counters see every claim, dedup only trims lines.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="unbacked claim", risk="low"),
                Claim(id="c2", text="Unbacked claim!", risk="low")],
        sources=[Source(id="s1", title="a blog", url="https://b.com",
                        source_type="blog")])
    assert _render_dispositions_block(led, None) == ""


def test_dispositions_block_renders_selective_hedge_against_plain():
    # Mix: some plain (folded into framing), a few specific hedges → renders the
    # framing + only the hedge lines. This is the selective-handling the stage is for.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="grounded fact", source_ids=["s1"], risk="low"),
                Claim(id="c2", text="grounded fact two", source_ids=["s1"], risk="low"),
                Claim(id="c3", text="shaky claim", source_ids=["s2"], risk="low")],
        sources=[Source(id="s1", title="ref", url="https://e.com", source_type="reference"),
                 Source(id="s2", title="blog", url="https://b.com", source_type="blog")],
    )
    out = _render_dispositions_block(led, None)
    assert out.startswith("State every claim plainly EXCEPT")
    assert "HEDGE: shaky claim" in out
    assert "grounded fact" not in out  # plain claims folded into framing


def test_dispositions_block_empty_when_all_plain():
    # Everything authoritative + low-risk → all state_plainly → nothing to instruct.
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import Claim, ResearchResult, Source
    led = ResearchResult(
        claims=[Claim(id="c1", text="fact one", source_ids=["s1"], risk="low"),
                Claim(id="c2", text="fact two", source_ids=["s1"], risk="low")],
        sources=[Source(id="s1", title="ref", url="https://e.com", source_type="reference")],
    )
    assert _render_dispositions_block(led, None) == ""


def test_dispositions_block_empty_when_no_ledger():
    from audrey.pipeline.deep_panel import _render_dispositions_block
    assert _render_dispositions_block(None, None) == ""


def test_dispositions_block_empty_when_no_claims():
    from audrey.pipeline.deep_panel import _render_dispositions_block
    from audrey.pipeline.ledger import ResearchResult
    assert _render_dispositions_block(ResearchResult(), None) == ""


def test_write_user_block_includes_dispositions_when_present():
    from audrey.pipeline.deep_panel import _write_user_block
    out = _write_user_block("q", "findings", "", dispositions="- HEDGE: x")
    assert "CLAIM DISPOSITIONS (apply these):" in out
    assert "- HEDGE: x" in out


def test_write_user_block_omits_dispositions_when_empty():
    from audrey.pipeline.deep_panel import _write_user_block
    out = _write_user_block("q", "findings", "")
    assert "CLAIM DISPOSITIONS" not in out


def test_hedge_policy_enabled_reads_flag():
    from audrey.pipeline.deep_panel import _hedge_policy_enabled
    cfg = _research_cfg_fc(["r1"], "v", "fc", "w",
                           (("r1", 100, "cloud"), ("v", 80, "cloud"),
                            ("fc", 75, "cloud"), ("w", 70, "local")))
    cfg.raw["agentic"] = {"research_ledger": {"enabled": True, "hedge_policy": True}}
    assert _hedge_policy_enabled(cfg) is True
    cfg.raw["agentic"] = {"research_ledger": {"enabled": True}}
    assert _hedge_policy_enabled(cfg) is False


class _DispoCapturingOllama(_LedgerOllama):
    """Records the writer's user-message content so a test can assert the
    dispositions block reached the writer prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer_user_msg = ""

    async def chat_stream(self, *, model, messages, options=None, timeout_s=0):
        # The writer is the only chat_stream caller in the research pipeline.
        self.writer_user_msg = messages[-1].get("content", "") if messages else ""
        async for chunk in super().chat_stream(
            model=model, messages=messages, options=options, timeout_s=timeout_s
        ):
            yield chunk


async def test_dispositions_reach_writer_when_flag_on():
    # End-to-end: hedge_policy on → the writer's user block carries a
    # CLAIM DISPOSITIONS section computed from the surviving ledger claim.
    import json as _json

    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc(["r1"], "v", "fc", "w", models)
    cfg.raw["agentic"] = {"research_ledger": {"enabled": True, "hedge_policy": True}}
    health = HealthTracker()
    # One plain (reference) claim + one vendor claim → the vendor claim produces an
    # ATTRIBUTE action line, so the block renders (a single plain claim alone would
    # be all-plain → suppressed).
    research_json = _json.dumps({
        "summary_notes": "n",
        "claims": [{"id": "c1", "text": "Euclid ~300 BCE", "source_ids": ["s1"], "risk": "low"},
                   {"id": "c2", "text": "AcmeAI beats everyone", "source_ids": ["s2"], "risk": "low"}],
        "sources": [{"id": "s1", "title": "MacTutor",
                     "url": "https://mathshistory.st-andrews.ac.uk/Euclid/",
                     "source_type": "reference", "supports": ["c1"]},
                    {"id": "s2", "title": "AcmeAI blog",
                     "url": "https://acme.example/blog",
                     "source_type": "company_claim", "supports": ["c2"]}],
    })
    factcheck_json = _json.dumps({"checks": [{"claim_id": "c1", "verdict": "supported"},
                                            {"claim_id": "c2", "verdict": "supported"}]})
    ollama = _DispoCapturingOllama(
        {"r1": "Euclid lived around 300 BCE.", "v": "ok",
         "fc": "CONFIRMED", "w": "Euclid lived around 300 BCE."},
        research_json=research_json, factcheck_json=factcheck_json,
    )

    async for _ in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "who was Euclid"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
    ):
        pass

    assert "CLAIM DISPOSITIONS" in ollama.writer_user_msg
    # the vendor claim is attributed; the plain claim is folded into the framing line
    assert "ATTRIBUTE TO SOURCE: AcmeAI beats everyone" in ollama.writer_user_msg
    assert "State every claim plainly EXCEPT" in ollama.writer_user_msg


async def test_dispositions_absent_when_flag_off():
    # Same ledger, hedge_policy off → no dispositions in the writer block
    # (write stage byte-identical to pre-Stage-4).
    import json as _json

    models = (("r1", 100, "cloud"), ("v", 80, "cloud"), ("fc", 75, "cloud"), ("w", 70, "local"))
    reg = _registry(*models)
    cfg = _research_cfg_fc_ledger(["r1"], "v", "fc", "w", models)  # enabled, no hedge_policy
    health = HealthTracker()
    research_json = _json.dumps({
        "summary_notes": "n",
        "claims": [{"id": "c1", "text": "Euclid ~300 BCE", "source_ids": ["s1"], "risk": "low"}],
        "sources": [{"id": "s1", "title": "MacTutor",
                     "url": "https://mathshistory.st-andrews.ac.uk/Euclid/",
                     "source_type": "reference", "supports": ["c1"]}],
    })
    factcheck_json = _json.dumps({"checks": [{"claim_id": "c1", "verdict": "supported"}]})
    ollama = _DispoCapturingOllama(
        {"r1": "Euclid lived around 300 BCE.", "v": "ok",
         "fc": "CONFIRMED: Euclid ~300 BCE", "w": "Euclid lived around 300 BCE."},
        research_json=research_json, factcheck_json=factcheck_json,
    )

    async for _ in run_research_pipeline_streaming(
        cfg, ollama, reg, health, FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "who was Euclid"}],
        options={}, timeout_s=5.0, max_researchers_cloud=2,
        tools=_one_tool_registry(), tool_capable_models={"fc"}, user_id=None,
    ):
        pass

    assert "CLAIM DISPOSITIONS" not in ollama.writer_user_msg


class TestLinkageLostMarker:
    """`UNLINKED-LEDGER` in the structuring log. It exists so one grep can answer
    "did a worker's citations get dropped", and the counts below are the real
    lines from the 2026-08-13 `--only current-` run."""

    def test_many_sources_none_linked_is_the_pathology(self):
        # The shape that started this: a worker cited five URLs in its notes and
        # the ledger linked none of them, so every claim read as unsourced.
        assert dpmod._linkage_lost(n_linked=0, n_sources=5)

    def test_one_stray_link_does_not_defeat_the_marker(self):
        # The first version tested `not n_linked`, so a single linked claim
        # silenced it — a strict-equality cliff, the same shape as the
        # over-hedge suppression guard.
        assert dpmod._linkage_lost(n_linked=1, n_sources=11)

    def test_thin_researcher_is_not_flagged(self):
        # `claims=16 linked=1 sources=1` (qwen3.6, `current-rust-async`). Its
        # notes carried NO `SOURCES:` block at all, so the empty `source_ids`
        # were correct. That is a researcher-prompt problem upstream, not lost
        # linkage, and flagging it here would send the next reader to the wrong
        # file.
        assert not dpmod._linkage_lost(n_linked=1, n_sources=1)

    def test_healthy_ledgers_are_not_flagged(self):
        # The other five calls from the same run.
        for n_linked, n_sources in [(20, 7), (46, 11), (40, 16), (22, 7), (53, 17)]:
            assert not dpmod._linkage_lost(n_linked=n_linked, n_sources=n_sources)

    def test_sourceless_ledger_is_not_flagged(self):
        # Nothing to lose: a researcher that found nothing is a grounding
        # problem, and the `sources=0` in the same log line already says so.
        assert not dpmod._linkage_lost(n_linked=0, n_sources=0)



class TestLinkageCounts:
    """`linked=` in the structuring log must count RESOLVABLE ids.

    Grounded in the 2026-08-14 run, where both qwen3.6 drafts cited source ids
    the ledger never defined and the old non-empty-list counter reported 21 and
    16 linked claims against zero real ones."""

    def _ledger(self, claims, sources):
        return ResearchResult(
            claims=[Claim(id=i, text="t", source_ids=s) for i, s in claims],
            sources=[Source(id=i, title="t", url="https://example.com/") for i in sources],
        )

    def test_fabricated_ids_are_dangling_not_linked(self):
        # The exact shape from `current-2025-recent`: every claim cites a
        # descriptive id the ledger never emitted.
        r = self._ledger(
            [("c1", ["src_microsoft_phi4"]), ("c2", ["src_alibaba_qwen3"])],
            ["s1"],
        )
        assert dpmod._linkage_counts(r) == (0, 2)

    def test_one_resolving_id_is_enough_to_count_linked(self):
        r = self._ledger([("c1", ["s1"]), ("c2", ["s1", "ghost"])], ["s1"])
        assert dpmod._linkage_counts(r) == (2, 0)

    def test_empty_source_ids_are_neither_linked_nor_dangling(self):
        # The honest case — the notes carried no source. It wants a different
        # fix from the fabrication above, so it must not share a number.
        r = self._ledger([("c1", [])], ["s1"])
        assert dpmod._linkage_counts(r) == (0, 0)

    def test_a_fully_dangling_ledger_trips_the_marker(self):
        # linked=0 with several sources present is exactly `_linkage_lost`.
        r = self._ledger([("c1", ["ghost"]), ("c2", ["ghost2"])], ["s1", "s2"])
        linked, dangling = dpmod._linkage_counts(r)
        assert (linked, dangling) == (0, 2)
        assert dpmod._linkage_lost(linked, len(r.sources))


class TestSourceCatalogue:
    """The structuring pass is handed the ids instead of minting them.

    This is the fix for the shape `TestLinkageCounts` measures: the model emitted
    `sources` numbered `s1`, `s2` and cited `src-corrode`, so nothing could tell
    it the two disagreed. Now `s1..sN` arrive in the request."""

    def test_rows_are_numbered_from_one_with_url_and_title(self):
        cat = dpmod._source_catalogue([
            {"title": "The State of Async Rust", "url": "https://corrode.dev/blog/async/"},
            {"title": "Tokio", "url": "https://docs.rs/crate/tokio/latest"},
        ])
        assert cat.splitlines() == [
            "s1\thttps://corrode.dev/blog/async/\tThe State of Async Rust",
            "s2\thttps://docs.rs/crate/tokio/latest\tTokio",
        ]

    def test_missing_title_still_yields_a_citable_row(self):
        # A row without a title is still a real URL the model may need to cite;
        # dropping it would silently shrink the catalogue below what was fetched.
        cat = dpmod._source_catalogue([{"url": "https://example.com/x"}])
        assert cat == "s1\thttps://example.com/x\t(untitled)"

    def test_no_retrieval_yields_an_empty_catalogue(self):
        # A tool-free worker falls back to the prose-only request — the shape
        # that shipped before this, so it must stay reachable.
        assert dpmod._source_catalogue([]) == ""


class TestReconcileWithCatalogue:
    """One id authority. The half-measure was worse than the original bug.

    With the catalogue in the prompt but the model still emitting `sources`, run
    `220557` produced 37 out-of-range cites (against 0 in 249 pre-change sources)
    AND silent wrong attachments where an out-of-range id landed inside the
    model's own shorter array. This function removes the second authority."""

    CAT: ClassVar[list[dict[str, str]]] = [
        {"title": "Qwen3", "url": "https://alibaba.example/qwen3"},
        {"title": "Nemotron", "url": "https://nvidia.example/nemotron"},
        {"title": "gpt-oss", "url": "https://openai.example/gpt-oss"},
    ]

    def _ledger(self, claim_ids, sources=()):
        return ResearchResult(
            claims=[Claim(id=f"c{i}", text="t", source_ids=list(s))
                    for i, s in enumerate(claim_ids)],
            sources=list(sources),
        )

    def test_out_of_range_ids_are_dropped_not_left_to_mislead(self):
        # The `s33`-against-a-16-row-array shape, in miniature.
        r = self._ledger([["s1", "s9"], ["s7"]])
        dropped, _ = dpmod._reconcile_with_catalogue(r, self.CAT)
        assert dropped == 2
        assert [c.source_ids for c in r.claims] == [["s1"], []]

    def test_sources_are_rebuilt_from_the_catalogue_rows_actually_cited(self):
        # Only cited rows materialise, which is also what stops raw search junk
        # (async.com, brutalist.report) landing in the ledger as sources.
        r = self._ledger([["s3"]])
        dpmod._reconcile_with_catalogue(r, self.CAT)
        assert [(s.id, s.url) for s in r.sources] == [
            ("s3", "https://openai.example/gpt-oss")]

    def test_the_models_source_type_survives_for_a_url_it_named(self):
        # The catalogue knows what was fetched; only the model read the content,
        # and official-vs-company_claim drives the hedge policy downstream.
        r = self._ledger([["s1"]], [Source(id="whatever", url="https://alibaba.example/qwen3",
                                           source_type="company_claim")])
        dpmod._reconcile_with_catalogue(r, self.CAT)
        assert (r.sources[0].id, r.sources[0].source_type) == ("s1", "company_claim")

    def test_notes_only_sources_are_counted_as_the_cost_of_the_trade(self):
        # A memory_recall hit or an unfetched snippet. Dropped — but the number is
        # logged, so the rescue path can be a measurement rather than a guess.
        r = self._ledger([["s1"]], [Source(id="x", url="https://recalled.example/from-memory")])
        _, notes_only = dpmod._reconcile_with_catalogue(r, self.CAT)
        assert notes_only == 1

    def test_the_source_to_claim_index_survives_the_rebuild(self):
        # ⚠️ Rebuilding `sources` mints FRESH objects, so the parser's
        # `backfill_supports` (which ran before reconciliation) is wiped. Run
        # `065953` shipped `supports: none` on 47 of 47 rows — the identical
        # symptom that function was written for in July. Nothing user-facing
        # broke, because both readers of this index OR it with
        # `claim.source_ids`; the redundancy is exactly what hid it.
        r = self._ledger([["s1"], ["s1", "s2"]])
        dpmod._reconcile_with_catalogue(r, self.CAT)
        by_id = {s.id: s for s in r.sources}
        assert by_id["s1"].supports == ["c0", "c1"]
        assert by_id["s2"].supports == ["c1"]

    def test_a_fully_dropped_draft_must_still_trip_the_marker(self):
        # ⚠️ The trap reconciliation introduces: rebuilding `sources` from cited
        # rows means a draft where NOTHING resolves ends with `sources=0`, and
        # `_linkage_lost` needs `n_sources >= 2` — so reading the rebuilt list
        # would mute the alarm in precisely the worst case. The denominator has
        # to be what the worker HAD, which is the catalogue.
        r = self._ledger([["s41"], ["s55"]])
        dpmod._reconcile_with_catalogue(r, self.CAT)
        linked, _ = dpmod._linkage_counts(r)
        assert (linked, len(r.sources)) == (0, 0)
        assert not dpmod._linkage_lost(linked, len(r.sources)), "the trap"
        assert dpmod._linkage_lost(linked, len(self.CAT)), "the fix"

    def test_an_empty_catalogue_leaves_the_ledger_untouched(self):
        # A tool-free worker keeps the pre-catalogue path exactly, so the change
        # cannot regress a draft it was never meant to touch.
        r = self._ledger([["anything"]], [Source(id="s1", url="https://a.example/")])
        assert dpmod._reconcile_with_catalogue(r, []) == (0, 0)
        assert r.claims[0].source_ids == ["anything"]
        assert len(r.sources) == 1


# ─── _fence_anomaly: naming a malformed draft ──────────────────────────
# `nemotron-3.5-lightning` returned bare, unfenced source for the same eval
# case three runs running while fencing every other draft it produced. The
# synthesizer repaired it each time, so no check ever failed and nothing in
# the system could say what had happened. This detector is what notices.

def test_bare_code_with_no_fence_is_named():
    from audrey.pipeline.deep_panel import _fence_anomaly
    # The exact shape observed: opens on an import, no fence anywhere.
    draft = "from collections import OrderedDict\n\n\nclass TTLCache:\n    pass\n"
    assert _fence_anomaly(draft) == "unfenced_code"


def test_an_opened_but_unclosed_fence_is_named_differently():
    # Distinct from `unfenced_code` on purpose: an odd fence count means the
    # generation STOPPED mid-block, which points at `done_reason`, while no
    # fence at all means it finished and never fenced. Opposite causes.
    from audrey.pipeline.deep_panel import _fence_anomaly
    assert _fence_anomaly("```python\nx = 1\n") == "unterminated_fence"


def test_a_well_formed_fenced_draft_is_clean():
    from audrey.pipeline.deep_panel import _fence_anomaly
    assert _fence_anomaly("Here you go:\n\n```python\nx = 1\n```\n") == ""


def test_prose_is_never_flagged_as_unfenced_code():
    # The detector is anchored and narrow by design. Ordinary prose that
    # happens to discuss imports must not read as code, or the warning fires
    # on every research draft and stops meaning anything.
    from audrey.pipeline.deep_panel import _fence_anomaly
    assert _fence_anomaly("The import of foreign grain rose sharply.") == ""
    assert _fence_anomaly("") == ""


def test_a_draft_that_opens_on_async_def_counts_as_code():
    from audrey.pipeline.deep_panel import _fence_anomaly
    assert _fence_anomaly("async def fetch_all(fetch, keys):\n    pass\n") == "unfenced_code"


# ─── think_for: the per-role thinking knob ─────────────────────────────
# ⚠️ Tri-state, and the middle state is the whole point. `None` omits the
# `think` field and every model accepts that; `False` is a DIFFERENT request
# that Ollama rejects outright on a model without the `thinking` capability.
# A knob that collapsed the two would turn "this role need not think" into a
# hard failure on the next model that cannot.

class _CapClient:
    """Fake Ollama that records whether the capability probe was consulted."""

    def __init__(self, capable: bool = True):
        self.capable = capable
        self.asked: list[tuple[str, bool]] = []

    async def thinking_flag(self, model: str, want: bool):
        self.asked.append((model, want))
        return want if self.capable else None


class _ThinkCfg:
    def __init__(self, thinking: dict | None = None):
        self.thinking = thinking or {}


async def _tf(client, cfg, *, role="deep_worker", model="m"):
    from audrey.pipeline.deep_panel import think_for
    return await think_for(client, cfg, role=role, model=model)


@pytest.mark.asyncio
async def test_no_policy_omits_the_field_and_never_probes():
    """The historical path. Absent config must not cost an `/api/show`, and
    must not send `think` at all — omitting it is the only universally safe
    request."""
    c = _CapClient()
    assert await _tf(c, _ThinkCfg()) is None
    assert c.asked == [], "capability probe ran with no policy configured"


@pytest.mark.asyncio
async def test_a_named_model_skips_thinking():
    c = _CapClient()
    cfg = _ThinkCfg({"no_thinking_models": ["nemotron-3.5-lightning:latest"]})
    assert await _tf(c, cfg, model="nemotron-3.5-lightning:latest") is False


@pytest.mark.asyncio
async def test_an_unnamed_model_is_untouched_by_the_list():
    """Blast radius equals the evidence: naming one model must not quiet the
    rest of the pool."""
    c = _CapClient()
    cfg = _ThinkCfg({"no_thinking_models": ["nemotron-3.5-lightning:latest"]})
    assert await _tf(c, cfg, model="deepseek-v4-pro:cloud") is None


@pytest.mark.asyncio
async def test_a_model_that_cannot_think_gets_no_field_even_when_named():
    """⚠️ The hard-error guard. Ollama REJECTS `think` on a model without the
    capability, so a named-but-incapable model must degrade to omission — not
    to `False`, which would break every call that lands on it."""
    c = _CapClient(capable=False)
    cfg = _ThinkCfg({"no_thinking_models": ["old-model:latest"]})
    assert await _tf(c, cfg, model="old-model:latest") is None


@pytest.mark.asyncio
async def test_a_role_switch_applies_to_every_model_in_that_role():
    c = _CapClient()
    cfg = _ThinkCfg({"deep_worker": True})
    assert await _tf(c, cfg, role="deep_worker", model="anything") is False


@pytest.mark.asyncio
async def test_a_role_switch_does_not_leak_into_another_role():
    """`deep_worker` and `deep_synth` are separate because their products are
    separate: a worker drafts, a synthesizer reasons over drafts. Turning one
    off must not quietly turn off the other."""
    c = _CapClient()
    cfg = _ThinkCfg({"deep_worker": True})
    assert await _tf(c, cfg, role="deep_synth", model="anything") is None


@pytest.mark.asyncio
async def test_a_missing_cfg_is_the_historical_path():
    c = _CapClient()
    assert await _tf(c, None) is None
    assert c.asked == []


class _ThinkCapturingOllama:
    """Records the `think` each `format=`-pinned call actually sent."""

    def __init__(self, content: str = "{}"):
        self.content = content
        self.thinks: list[bool | None] = []

    async def thinking_flag(self, model, want):
        return want  # model is thinking-capable

    async def chat(self, *, model, messages, options=None, timeout_s=0,
                   tools=None, format=None, think=None):
        self.thinks.append(think)
        return {"message": {"content": self.content}, "eval_count": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["draft", "factcheck"])
async def test_both_structuring_sites_send_the_ledger_structure_think_policy(site):
    """The two schema-pinned passes are direct `ollama.chat`, not `run_react`,
    so no other thinking rule reaches them — they were the last callers sending
    no `think` at all. That is not a cosmetic gap: thinking there spent a whole
    64k budget and returned zero bytes of JSON, which the fail-soft path then
    swallowed. Assert the policy ARRIVES, per site, because wiring one and
    forgetting the other is exactly the shape of the original bug.
    """
    from audrey.pipeline.deep_panel import (
        _structure_factcheck_batch,
        _structure_one_draft,
    )
    from audrey.pipeline.ledger import Claim
    c = _ThinkCapturingOllama()
    cfg = _ThinkCfg({"ledger_structure": True})
    common = dict(
        model="deepseek-v4-pro:cloud", location="cloud", timeout_s=5.0, user_id=None,
    )
    if site == "draft":
        await _structure_one_draft(
            c, HealthTracker(), FairLocalGate(concurrency=1), cfg,
            prose="notes", retrieved=[], worker_idx=0, **common,
        )
    else:
        await _structure_factcheck_batch(
            c, HealthTracker(), FairLocalGate(concurrency=1), cfg,
            claims=[Claim(id="w0_c1", text="a claim")], sources=[],
            fc_notes="notes", **common,
        )
    assert c.thinks == [False], (
        f"the {site} structuring call sent think={c.thinks!r}; `None` means it "
        "omits the field and the model thinks its budget away again"
    )


@pytest.mark.asyncio
async def test_a_worker_that_raises_anything_still_returns_a_draft():
    """⚠️ `_run_one_worker` documents "never raises". Until 2026-08-17 it caught
    only `OllamaError`, so anything else escaped into `asyncio.gather` — and on
    the streaming path that DEADLOCKED the response generator rather than
    failing it, with the headers already sent so the client could not be told.

    A panel that loses one worker still has others. A panel that hangs has
    nothing, and the operator gets no error to search for.
    """
    class _Exploding:
        async def thinking_flag(self, model, want):
            return None

        async def chat(self, **kw):
            raise RuntimeError("upstream did something unexpected")

    draft = await dpmod._run_one_worker(
        _Exploding(), HealthTracker(), FairLocalGate(concurrency=1),
        model="m", location="cloud", messages=[{"role": "user", "content": "hi"}],
        options={}, timeout_s=5.0, tools=None, tool_capable=False,
        react_max_rounds=1, react_compress_after=1, react_max_tool_chars=100,
        react_dispatch_timeout_s=5.0,
    )
    assert draft["content"] == ""
    assert "RuntimeError" in draft["error"], draft


class TestTheCloudWorkerCapIsAudible:
    """A worker dropped for the cap must say so — silence costs a whole run.

    `deep_panel_cloud.code` was given three cloud workers on 2026-08-17 against
    an effective cap of 2. `deepseek-v4-pro` therefore never drafted, and the
    eval artifact showed two drafts with nothing anywhere explaining the third.
    The cap is set in TWO places — `config.yaml`'s `agentic.max_deep_workers_cloud`
    and the `MAX_DEEP_WORKERS_CLOUD` env override — so the effective value has
    to be printed, not inferred from config that may not be what is running.
    """

    _MODELS = (("a:cloud", 100, "cloud"), ("b:cloud", 90, "cloud"),
               ("c:cloud", 80, "cloud"))

    def _select(self, cap):
        cfg = _cfg_with_pool(["a:cloud", "b:cloud", "c:cloud"], self._MODELS)
        return select_workers(
            cfg, _registry(*self._MODELS), HealthTracker(),
            pool_key="deep_panel", task="reasoning", max_workers_cloud=cap,
        )

    def test_the_cap_still_drops_the_extra_worker(self):
        assert [n for n, _ in self._select(2)] == ["a:cloud", "b:cloud"]
        assert [n for n, _ in self._select(3)] == ["a:cloud", "b:cloud", "c:cloud"]

    def test_a_dropped_worker_is_logged_at_warning_with_the_effective_cap(self, caplog):
        with caplog.at_level(logging.WARNING, logger="audrey.pipeline.deep_panel"):
            self._select(2)
        dropped = [r for r in caplog.records if "DROPPING cloud worker" in r.getMessage()]
        assert len(dropped) == 1, "the dropped worker must announce itself exactly once"
        msg = dropped[0].getMessage()
        assert "c:cloud" in msg, "the message must name WHICH worker never runs"
        assert "2" in msg, "the message must carry the EFFECTIVE cap, not just config"
        assert "MAX_DEEP_WORKERS_CLOUD" in msg, (
            "the env override is the likeliest reason the cap is not what "
            "config.yaml says, so the message has to point at it"
        )

    def test_nothing_is_logged_when_every_worker_fits(self, caplog):
        with caplog.at_level(logging.WARNING, logger="audrey.pipeline.deep_panel"):
            self._select(3)
        assert not [r for r in caplog.records if "DROPPING" in r.getMessage()]


class TestCharsPerTokIsMeasuredOnRawOutput:
    """The ratio must not double-count what `_strip_think` removed.

    It answers "did the tokens billed become TEXT at all". A `<think>` block
    emitted inline DID become text; the stripper removing it afterwards is a
    separate fact, already reported as `think_stripped`. Dividing the stripped
    length merges the two and reads as heavy hidden thinking when there was
    none — which is exactly how a local draft came back at 1.73 on 2026-08-17
    while its raw ratio was a perfectly healthy 3.94.
    """

    def test_a_stripped_draft_still_reports_the_raw_ratio(self):
        raw = "x" * 4000
        stripped = "x" * 1000
        _, cpt = _log_draft_shape(
            "m", raw=raw, stripped=stripped, done_reason="stop",
            eval_count=1000, elapsed=1.0, subtask="",
        )
        assert cpt == 4.0, (
            f"expected the RAW ratio (4000/1000); got {cpt}, which is the "
            "stripped ratio and would read as hidden thinking"
        )

    def test_genuine_hidden_thinking_still_reads_low(self):
        # Nothing stripped, few chars per billed token → the real signal.
        _, cpt = _log_draft_shape(
            "m", raw="x" * 1053, stripped="x" * 1053, done_reason="stop",
            eval_count=6172, elapsed=1.0, subtask="",
        )
        assert cpt < 0.5

# ─── Cancellation ownership ───────────────────────────────────────────


async def _assert_cancel_drains_workers(stream, started, settled, child_tasks):
    consumer = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=1,
        )
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in settled)),
            timeout=1,
        )
    finally:
        consumer.cancel()
        for task in child_tasks:
            task.cancel()
        await asyncio.gather(consumer, *child_tasks, return_exceptions=True)
        await stream.aclose()


async def test_streaming_panel_cancellation_drains_fanout_and_gpu_slot(monkeypatch):
    gate = FairLocalGate(concurrency=1)
    started = [asyncio.Event(), asyncio.Event()]
    settled = [asyncio.Event(), asyncio.Event()]
    child_tasks = []

    async def worker(index, user):
        child_tasks.append(asyncio.current_task())
        started[index].set()
        try:
            async with gate.acquire("m", location="local", user_id=user):
                await asyncio.Event().wait()
        finally:
            settled[index].set()

    def prepare(*args, **kwargs):
        return (
            [("one", "local"), ("two", "local")],
            [worker(0, "alice"), worker(1, "bob")],
        )

    monkeypatch.setattr(dpmod, "_prepare_panel", prepare)
    stream = run_panel_streaming(
        object(),
        object(),
        object(),
        object(),
        gate,
        pool_key="deep_panel",
        task="reasoning",
        messages=[],
        subtasks=[],
        options={},
        timeout_s=30,
        max_workers_cloud=2,
    )
    await _assert_cancel_drains_workers(stream, started, settled, child_tasks)

    async def later_request():
        async with gate.acquire("m", location="local", user_id="carol"):
            return True

    assert await asyncio.wait_for(later_request(), timeout=1) is True
    assert gate._available == gate.concurrency


async def test_research_fanout_cancellation_drains_all_researchers(monkeypatch):
    gate = FairLocalGate(concurrency=1)
    started = [asyncio.Event(), asyncio.Event()]
    settled = [asyncio.Event(), asyncio.Event()]
    child_tasks = []
    indexes = {"r1": 0, "r2": 1}

    monkeypatch.setattr(
        dpmod,
        "select_researchers",
        lambda *args, **kwargs: [("r1", "cloud"), ("r2", "cloud")],
    )

    async def blocked_worker(*args, model, **kwargs):
        index = indexes[model]
        child_tasks.append(asyncio.current_task())
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            settled[index].set()

    monkeypatch.setattr(dpmod, "_run_one_worker", blocked_worker)
    stream = dpmod.run_research_pipeline_streaming(
        get_config(),
        object(),
        object(),
        HealthTracker(),
        gate,
        task="reasoning",
        messages=[{"role": "user", "content": "question"}],
        options={},
        timeout_s=30,
        max_researchers_cloud=2,
    )
    await _assert_cancel_drains_workers(stream, started, settled, child_tasks)
