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
from pathlib import Path
from types import SimpleNamespace

import pytest

from audrey.config import (
    Config,
    EnvOverrides,
    _load_yaml,
    _validate_deep_panel_pools,
    _validate_upload_limits,
)
from audrey.pipeline.deep_panel import pick_panel_timeout, pool_key_for

# Repo root holds the real config.yaml the app boots with.
_REPO_ROOT = Path(__file__).resolve().parent.parent

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


# ─── Staged pool (deep_panel_research) ─────────────────────────────────
# A body with a `researchers` list is the staged audrey_research shape:
# required roles are researchers/verifier/writer (no synthesizer), and the
# model-bearing slots are researchers[*]/verifier/writer/fallback_synth.

def test_validator_accepts_well_formed_research_pool():
    cfg = {
        "deep_panel_research": {
            "reasoning": {
                "researchers": ["r1", "r2"],
                "verifier": "v",
                "writer": "w",
                "fallback_synth": "f",
            },
        },
    }
    _validate_deep_panel_pools(cfg)  # must not raise


def test_validator_research_pool_does_not_require_synthesizer():
    # The staged shape has no `synthesizer` — the writer produces the answer.
    # The flat-pool "missing synthesizer" check must not fire here.
    cfg = {
        "deep_panel_research": {
            "general": {"researchers": ["r1"], "verifier": "v", "writer": "w"},
        },
    }
    _validate_deep_panel_pools(cfg)  # must not raise


def test_validator_rejects_empty_researchers():
    cfg = {
        "deep_panel_research": {
            "reasoning": {"researchers": [], "verifier": "v", "writer": "w"},
        },
    }
    with pytest.raises(ValueError, match=r"deep_panel_research/reasoning:.*researchers.*non-empty"):
        _validate_deep_panel_pools(cfg)


def test_validator_rejects_missing_verifier_or_writer():
    cfg = {
        "deep_panel_research": {
            "reasoning": {"researchers": ["r1"], "writer": "w"},   # no verifier
            "general": {"researchers": ["r1"], "verifier": "v"},   # no writer
        },
    }
    with pytest.raises(ValueError) as exc:
        _validate_deep_panel_pools(cfg)
    message = str(exc.value)
    assert "deep_panel_research/reasoning: missing required `verifier`" in message
    assert "deep_panel_research/general: missing required `writer`" in message


def test_validator_rejects_unknown_research_model_names():
    # Every researcher/verifier/writer/fallback name must resolve to the
    # registry, same posture as the flat-pool worker/synth check.
    cfg = {
        "model_registry": {
            "reasoning": [{"name": "r1"}, {"name": "v"}, {"name": "w"}],
        },
        "deep_panel_research": {
            "reasoning": {
                "researchers": ["r1", "ghost-researcher:cloud"],  # second is unknown
                "verifier": "v",
                "writer": "w",
            },
        },
    }
    with pytest.raises(ValueError, match=r"researcher 'ghost-researcher:cloud' is not in model_registry"):
        _validate_deep_panel_pools(cfg)


def test_validator_factchecker_is_optional_but_validated_when_present():
    # `factchecker` is optional — a staged pool without it is valid.
    ok = {
        "deep_panel_research": {
            "reasoning": {"researchers": ["r1"], "verifier": "v", "writer": "w"},
        },
    }
    _validate_deep_panel_pools(ok)  # must not raise (no factchecker)

    # But when present, its model name must resolve to the registry.
    bad = {
        "model_registry": {"reasoning": [{"name": "r1"}, {"name": "v"}, {"name": "w"}]},
        "deep_panel_research": {
            "reasoning": {
                "researchers": ["r1"], "verifier": "v", "writer": "w",
                "factchecker": "ghost-checker:cloud",  # unknown
            },
        },
    }
    with pytest.raises(ValueError, match=r"factchecker 'ghost-checker:cloud' is not in model_registry"):
        _validate_deep_panel_pools(bad)


# ─── Real config.yaml boots clean ──────────────────────────────────────

def test_committed_config_yaml_passes_boot_validation():
    """The actual `config.yaml` must survive the boot path.

    Every other test in this file feeds the validator a synthetic dict, so a
    broken *committed* config (a deep-panel worker absent from the registry, a
    missing synthesizer) slips through `pytest` and only crash-loops on
    deploy — exactly how an unregistered `nemotron3:33b` vl worker took the
    process down once. This loads the real file and runs the same three steps
    `get_config()` does at startup, so that class of bug fails here instead.
    """
    cfg = Config(_load_yaml(_REPO_ROOT / "config.yaml"), EnvOverrides())
    _validate_deep_panel_pools(cfg.raw)  # must not raise


def test_committed_config_deep_panel_models_are_registered():
    """Belt-and-braces: every deep-panel worker/synth resolves to a registry
    entry. `_validate_deep_panel_pools` already enforces this, but asserting it
    independently pins the invariant even if the validator's scope ever changes.
    """
    raw = _load_yaml(_REPO_ROOT / "config.yaml")
    registry_names = {
        spec["name"]
        for specs in raw.get("model_registry", {}).values()
        for spec in specs
        if spec.get("name")
    }
    for pool_key in (k for k in raw if k.startswith("deep_panel")):
        for task, body in raw[pool_key].items():
            if "researchers" in body:  # staged audrey_research shape
                named = list(body.get("researchers") or [])
                named += [body[s] for s in ("verifier", "writer", "fallback_synth") if body.get(s)]
            else:  # flat panel shape
                named = list(body.get("workers") or [])
                named += [body[s] for s in ("synthesizer", "fallback_synth") if body.get(s)]
            for name in named:
                assert name in registry_names, f"{pool_key}/{task}: {name!r} not in registry"


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


class TestDiagnosticTraceEnvOverrides:
    """⚠️ The diagnostic traces are env-overridable for the same reason the
    video lease is, and the cost was paid on 2026-08-12.

    Both flags live in `config.yaml`, which is TRACKED and under active
    development. Turning one on for a run therefore leaves a working-tree diff
    that every later `git pull` has to be worked around. Worse, the on-box
    `sed` that flipped one of them matched the OTHER flag's line and replaced
    it — leaving two `debug_context_trace` keys under `agentic`. YAML keeps the
    last, so the flag read `false` while looking `true` in the file, and the
    diagnostic run it was turned on for would have produced no trace lines at
    all.

    `.env` is gitignored. Setting them there survives every pull untouched and
    needs no edit to a tracked file.
    """

    def test_yaml_wins_when_the_env_var_is_absent(self):
        """Unset must mean unset, or the committed `false` becomes a lie."""
        cfg = Config({"agentic": {"debug_context_trace": False,
                                  "debug_research_trace": False}},
                     EnvOverrides(DEBUG_CONTEXT_TRACE=None,
                                  DEBUG_RESEARCH_TRACE=None))
        assert cfg.raw["agentic"]["debug_context_trace"] is False
        assert cfg.raw["agentic"]["debug_research_trace"] is False

    def test_the_env_var_turns_a_trace_on_without_touching_the_file(self):
        cfg = Config({"agentic": {"debug_context_trace": False}},
                     EnvOverrides(DEBUG_CONTEXT_TRACE=True))
        assert cfg.raw["agentic"]["debug_context_trace"] is True

    def test_each_flag_is_independent(self):
        """The sed that started this conflated the two. They must not."""
        cfg = Config({"agentic": {"debug_context_trace": False,
                                  "debug_research_trace": False}},
                     EnvOverrides(DEBUG_CONTEXT_TRACE=True))
        assert cfg.raw["agentic"]["debug_context_trace"] is True
        assert cfg.raw["agentic"]["debug_research_trace"] is False

    def test_the_env_var_can_also_force_a_trace_off(self):
        """Not symmetry for its own sake: it is how a flag left `true` in a
        pulled config gets silenced without editing the file back."""
        cfg = Config({"agentic": {"debug_context_trace": True}},
                     EnvOverrides(DEBUG_CONTEXT_TRACE=False))
        assert cfg.raw["agentic"]["debug_context_trace"] is False

    def test_it_works_when_yaml_has_no_agentic_block(self):
        cfg = Config({}, EnvOverrides(DEBUG_CONTEXT_TRACE=True))
        assert cfg.raw["agentic"]["debug_context_trace"] is True

    def test_the_override_does_not_disturb_the_rest_of_agentic(self):
        cfg = Config({"agentic": {"react": {"compress_keep_last": 4}}},
                     EnvOverrides(DEBUG_CONTEXT_TRACE=True))
        assert cfg.raw["agentic"]["react"]["compress_keep_last"] == 4


class TestVideoLeaseEnvOverrides:
    """`kb.video` is env-overridable so the lease verification never needs an
    edit to `config.yaml` on the box.

    That edit dirties the deployed working tree and has to be reverted by hand.
    A forgotten revert leaves production on a one-minute lease — every worker
    that takes longer than a minute has its job swept out from under it, and
    the only symptom is jobs mysteriously running twice.
    """

    def test_yaml_wins_when_the_env_var_is_absent(self):
        """The default that matters. An override that applied when unset would
        make the committed config a lie."""
        cfg = Config({"kb": {"video": {"lease_minutes": 30, "max_attempts": 3}}},
                     EnvOverrides(VIDEO_LEASE_MINUTES=None, VIDEO_MAX_ATTEMPTS=None))
        assert cfg.raw["kb"]["video"]["lease_minutes"] == 30
        assert cfg.raw["kb"]["video"]["max_attempts"] == 3

    def test_the_env_var_overrides_yaml(self):
        cfg = Config({"kb": {"video": {"lease_minutes": 30, "max_attempts": 3}}},
                     EnvOverrides(VIDEO_LEASE_MINUTES=1, VIDEO_MAX_ATTEMPTS=9))
        assert cfg.raw["kb"]["video"]["lease_minutes"] == 1
        assert cfg.raw["kb"]["video"]["max_attempts"] == 9

    def test_it_works_when_yaml_has_no_video_block_at_all(self):
        """`setdefault` has to build both levels — `kb` may exist without
        `video`, or neither may exist."""
        cfg = Config({}, EnvOverrides(VIDEO_LEASE_MINUTES=2))
        assert cfg.raw["kb"]["video"]["lease_minutes"] == 2

    def test_the_override_does_not_disturb_the_rest_of_kb(self):
        cfg = Config({"kb": {"dataset_paths": ["/datasets/x"], "video": {"max_attempts": 3}}},
                     EnvOverrides(VIDEO_LEASE_MINUTES=1))
        assert cfg.raw["kb"]["dataset_paths"] == ["/datasets/x"]
        assert cfg.raw["kb"]["video"]["max_attempts"] == 3
        assert cfg.raw["kb"]["video"]["lease_minutes"] == 1

    def test_the_committed_config_still_ships_production_values(self):
        """Guards the exact accident the override exists to prevent: a
        `lease_minutes: 1` left behind in the committed file."""
        raw = _load_yaml(_REPO_ROOT / "config.yaml")
        video = raw["kb"]["video"]
        assert video["lease_minutes"] == 30
        assert video["max_attempts"] == 3


# ─── _validate_upload_limits ───────────────────────────────────────────

class TestUploadLimitCoherence:
    """`kb.max_user_bytes` and `kb.chunked.max_upload_mb` are two numbers in
    two blocks that have to agree, and editing either one prompts you to check
    the other exactly never.

    Shipped disagreeing: the quota was 1 GiB while the chunked transport
    advertised a 2 GiB per-file ceiling, so the largest upload the transport
    accepted could never fit the quota. The refusal is a correct-looking 413
    naming the quota, so it reads as "you are out of space" — which is why this
    is a boot failure rather than something to notice in a log.
    """

    def test_a_quota_below_the_per_file_ceiling_is_refused(self):
        with pytest.raises(ValueError, match="max_user_bytes"):
            _validate_upload_limits({
                "kb": {
                    "max_user_bytes": 1024 * 1024 * 1024,      # 1 GiB
                    "chunked": {"max_upload_mb": 2048},         # 2 GiB
                },
            })

    def test_the_error_names_both_numbers_and_says_what_to_do(self):
        """A boot failure that does not say which knob to turn just moves the
        confusion from request time to start-up."""
        with pytest.raises(ValueError) as e:
            _validate_upload_limits({
                "kb": {
                    "max_user_bytes": 1024 * 1024 * 1024,
                    "chunked": {"max_upload_mb": 2048},
                },
            })
        message = str(e.value)
        assert "max_user_bytes" in message
        assert "chunked.max_upload_mb" in message
        assert "Raise" in message

    def test_a_quota_above_the_ceiling_is_fine(self):
        _validate_upload_limits({
            "kb": {
                "max_user_bytes": 10 * 1024 * 1024 * 1024,
                "chunked": {"max_upload_mb": 2048},
            },
        })

    def test_exactly_equal_is_allowed(self):
        """One max-size upload into an empty account must fit. Anything
        stricter would make the advertised ceiling unreachable again."""
        _validate_upload_limits({
            "kb": {
                "max_user_bytes": 2048 * 1024 * 1024,
                "chunked": {"max_upload_mb": 2048},
            },
        })

    def test_a_config_missing_either_key_is_not_an_error(self):
        """Stripped-down configs are used throughout the tests and by the
        tools sidecar. Absent is not the same as wrong."""
        _validate_upload_limits({})
        _validate_upload_limits({"kb": {}})
        _validate_upload_limits({"kb": {"max_user_bytes": 1}})
        _validate_upload_limits({"kb": {"chunked": {"max_upload_mb": 2048}}})

    def test_the_committed_config_is_coherent(self):
        """The one that matters — the file the app actually boots with."""
        raw = _load_yaml(_REPO_ROOT / "config.yaml")
        _validate_upload_limits(raw)

    def test_the_committed_quota_clears_the_video_case_that_prompted_it(self):
        """Three 300 MB videos exhausted the old 1 GiB quota and the fourth was
        refused with terabytes free on the array."""
        raw = _load_yaml(_REPO_ROOT / "config.yaml")
        quota = raw["kb"]["max_user_bytes"]
        assert quota >= 10 * 300 * 1024 * 1024
