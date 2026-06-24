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

from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.deep_panel import (
    _prepare_panel,
    run_panel,
    run_panel_streaming,
)
from audrey.pipeline.fair_gate import FairLocalGate


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
