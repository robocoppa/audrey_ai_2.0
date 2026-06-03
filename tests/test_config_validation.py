"""Tests for the deep-panel startup validator, `pool_key_for`, and `pick_panel_timeout`.

`_validate_deep_panel_pools` runs from `get_config()` so a misconfigured
`config.yaml` crashes the process at boot instead of 500ing the first
deep request. `pool_key_for` warns when an unknown virtual model falls
back to the default pool, so a typo doesn't silently route to the wrong
worker list. `pick_panel_timeout` is shared between `routes/openai.py`
(streaming deep) and `pipeline/graph.py` (non-streaming deep) — testing
it once here keeps the two callers from drifting.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from audrey.config import _validate_deep_panel_pools
from audrey.pipeline.deep_panel import pick_panel_timeout, pool_key_for

# ─── _validate_deep_panel_pools ────────────────────────────────────────

def test_validator_accepts_well_formed_pools():
    cfg = {
        "deep_panel": {
            "general": {"workers": ["a"], "synthesizer": "s", "fallback_synth": "f"},
        },
        "deep_panel_cloud": {
            "code": {"workers": ["b"], "synthesizer": "s"},
        },
    }

    _validate_deep_panel_pools(cfg)  # must not raise


def test_validator_ignores_non_deep_panel_keys():
    # The validator should scope itself to keys starting with `deep_panel`.
    # Other top-level sections (router, tools, etc.) are not its business.
    cfg = {
        "router": {"model": "x"},
        "tools": {"servers": []},
        "deep_panel": {
            "general": {"workers": ["a"], "synthesizer": "s"},
        },
    }

    _validate_deep_panel_pools(cfg)


def test_validator_rejects_missing_synthesizer():
    cfg = {
        "deep_panel": {
            "general": {"workers": ["a"]},  # no synthesizer
        },
    }

    with pytest.raises(ValueError, match=r"deep_panel/general:.*synthesizer"):
        _validate_deep_panel_pools(cfg)


def test_validator_rejects_empty_synthesizer():
    # An empty string is just as bad as a missing key — `pick_synthesizer`
    # treats both as a misconfig.
    cfg = {
        "deep_panel": {
            "reasoning": {"workers": ["a"], "synthesizer": ""},
        },
    }

    with pytest.raises(ValueError, match=r"deep_panel/reasoning:.*synthesizer"):
        _validate_deep_panel_pools(cfg)


def test_validator_collects_every_error_into_one_message():
    # All bad pools in one exception so the operator sees the full picture
    # on a single boot attempt instead of fixing one, restarting, hitting
    # the next, etc.
    cfg = {
        "deep_panel": {
            "general": {"workers": ["a"]},  # missing synthesizer
            "code": {"workers": ["b"], "synthesizer": ""},  # empty synthesizer
        },
        "deep_panel_cloud": {
            "vl": {"workers": ["c"]},  # missing synthesizer
        },
    }

    with pytest.raises(ValueError) as exc:
        _validate_deep_panel_pools(cfg)

    message = str(exc.value)
    assert "deep_panel/general" in message
    assert "deep_panel/code" in message
    assert "deep_panel_cloud/vl" in message


def test_validator_accepts_pool_models_present_in_registry():
    # When a model_registry exists, every worker/synth/fallback must be in
    # it. This config is internally consistent and must pass.
    cfg = {
        "model_registry": {
            "general": [
                {"name": "qwen3.6:35b", "location": "local"},
                {"name": "kimi-k2.6:cloud", "location": "cloud"},
            ],
        },
        "deep_panel": {
            "general": {
                "workers": ["qwen3.6:35b", "kimi-k2.6:cloud"],
                "synthesizer": "qwen3.6:35b",
                "fallback_synth": "qwen3.6:35b",
            },
        },
    }

    _validate_deep_panel_pools(cfg)  # must not raise


def test_validator_rejects_dangling_fallback_synth_not_in_registry():
    # Regression guard for the dangling-tag bug: a fallback_synth naming a
    # model absent from the registry would silently waste a GPU-gate slot on
    # a model that can't load before degrading. Catch it at boot instead.
    cfg = {
        "model_registry": {
            "code": [{"name": "qwen3.6:35b", "location": "local"}],
        },
        "deep_panel_local": {
            "code": {
                "workers": ["qwen3.6:35b"],
                "synthesizer": "qwen3.6:35b",
                "fallback_synth": "ghost-model:31b",  # not in registry
            },
        },
    }

    with pytest.raises(ValueError, match=r"fallback_synth 'ghost-model:31b' is not in model_registry"):
        _validate_deep_panel_pools(cfg)


def test_validator_rejects_unknown_worker_name():
    cfg = {
        "model_registry": {
            "general": [{"name": "qwen3.6:35b", "location": "local"}],
        },
        "deep_panel": {
            "general": {
                "workers": ["qwen3.6:35b", "typo-worker:latest"],
                "synthesizer": "qwen3.6:35b",
            },
        },
    }

    with pytest.raises(ValueError, match=r"worker 'typo-worker:latest' is not in model_registry"):
        _validate_deep_panel_pools(cfg)


def test_validator_skips_name_check_when_no_registry():
    # Stripped-down configs with no model_registry have nothing to validate
    # names against — the name check must not invent failures.
    cfg = {
        "deep_panel": {
            "general": {"workers": ["anything"], "synthesizer": "whatever"},
        },
    }

    _validate_deep_panel_pools(cfg)  # must not raise


def test_validator_rejects_non_dict_pool():
    cfg = {"deep_panel": "not-a-dict"}

    with pytest.raises(ValueError, match=r"deep_panel: expected dict"):
        _validate_deep_panel_pools(cfg)


def test_validator_rejects_non_dict_task_body():
    cfg = {"deep_panel": {"general": ["not", "a", "dict"]}}

    with pytest.raises(ValueError, match=r"deep_panel/general: expected dict"):
        _validate_deep_panel_pools(cfg)


def test_validator_passes_when_no_deep_panel_keys_exist():
    # Edge case: a stripped-down test config with no panel pools at all.
    # The validator should not invent failures out of thin air.
    _validate_deep_panel_pools({"router": {}, "tools": {}})


# ─── pool_key_for warning ──────────────────────────────────────────────

def test_pool_key_for_known_virtual_model_returns_pool_silently(caplog):
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.deep_panel"):
        assert pool_key_for("audrey_deep") == "deep_panel"
        assert pool_key_for("audrey_cloud") == "deep_panel_cloud"
        assert pool_key_for("audrey_local") == "deep_panel_local"

    # No warnings for known virtual models — these are the normal path.
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_pool_key_for_unknown_virtual_model_warns_and_falls_back(caplog):
    # Typo in config, or a virtual model added without a pool registration.
    # Must still return a usable pool so the request answers, but log so
    # the operator notices the misconfig.
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.deep_panel"):
        pool = pool_key_for("audrey_typo")

    assert pool == "deep_panel"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "audrey_typo" in warnings[0].getMessage()
    assert "deep_panel" in warnings[0].getMessage()


# ─── pick_panel_timeout ────────────────────────────────────────────────

def _cfg(timeouts: dict[str, float]):
    """Minimal cfg stub. The helper only reads `cfg.timeouts.get(...)`."""
    return SimpleNamespace(timeouts=timeouts)


def test_pick_panel_timeout_cloud_pool_uses_cloud_timeout():
    cfg = _cfg({"cloud": 90.0, "deep_worker": 180.0})
    assert pick_panel_timeout(cfg, "deep_panel_cloud") == 90.0


def test_pick_panel_timeout_mixed_pool_uses_deep_worker_timeout():
    cfg = _cfg({"cloud": 90.0, "deep_worker": 180.0})
    assert pick_panel_timeout(cfg, "deep_panel") == 180.0


def test_pick_panel_timeout_local_pool_uses_deep_worker_timeout():
    # `deep_panel_local` workers serialize through the local GPU gate just
    # like the mixed pool's local workers, so they share the same timeout.
    cfg = _cfg({"cloud": 90.0, "deep_worker": 180.0})
    assert pick_panel_timeout(cfg, "deep_panel_local") == 180.0


def test_pick_panel_timeout_defaults_when_keys_missing():
    # Empty timeouts dict — fall back to the helper's documented defaults
    # (cloud=120, deep_worker=240).
    cfg = _cfg({})
    assert pick_panel_timeout(cfg, "deep_panel_cloud") == 120.0
    assert pick_panel_timeout(cfg, "deep_panel") == 240.0
