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

from typing import Any
from unittest.mock import patch

from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline import graph as gmod
from audrey.pipeline.deep_panel import (
    _merge_ledgers,
    _prefix_ledger_ids,
    _prepare_panel,
    pick_panel_timeout,
    pool_key_for,
    run_panel,
    run_panel_streaming,
)
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.ledger import Claim, ResearchResult, Source


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


def _prep(cfg, registry, health, *, subtasks=None):
    """Call _prepare_panel with sane defaults; return (workers, coros)."""
    return _prepare_panel(
        cfg, object(), registry, health, FairLocalGate(concurrency=1),
        pool_key="deep_panel", task="reasoning",
        messages=[{"role": "user", "content": "q"}],
        subtasks=subtasks or [],
        options={}, timeout_s=5.0, max_workers_cloud=3,
        tools=None, tool_capable_models=None,
        react_max_rounds=2, react_compress_after=2, react_max_tool_chars=2000,
        react_dispatch_timeout_s=30.0, react_compress_keep_last=1, user_id=None,
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
        compiled = gmod.build_graph(cfg, ollama, registry, health, gate, _NoTools())
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
        cfg, ollama, registry, HealthTracker(), FairLocalGate(concurrency=1), _NoTools()
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
        HealthTracker(), FairLocalGate(concurrency=1), _NoTools(),
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
        FairLocalGate(concurrency=1), _NoTools(),
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

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None):
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


def test_has_corrections_recognizes_drop():
    from audrey.pipeline.deep_panel import _has_corrections
    assert _has_corrections("DROP: some claim — unsupported") is True


def test_render_claims_includes_ids_and_no_source_marker():
    from audrey.pipeline.deep_panel import _render_claims_for_factcheck
    from audrey.pipeline.ledger import Claim, Source
    led = _ledger_with(Claim(id="c1", text="grounded", source_ids=["s1"], risk="high"),
                       Claim(id="c2", text="ungrounded"))
    led.sources = [Source(id="s1", title="T", url="https://e.com", source_type="official")]
    out = _render_claims_for_factcheck(led)
    assert "c1" in out and "c2" in out
    assert "[no source]" in out  # c2 has none


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

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, format=None):
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
