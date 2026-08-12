"""`thinking_audit._collect` sees eval candidates too (2026-08-12).

Every group in the report reads a production ROLE, so a model sitting only in
`passthrough.allowed_models` was invisible: asked on 2026-08-12 which of three
bake-off models declare thinking, the report answered for one — the other two
held no role. The capability is an input to the promotion decision, so the
model has to be visible before it is promoted, not after.

The dedup rule is the part worth pinning. Most of `allowed_models` is the
role-holding pool re-listed for `--models` sweeps, so adding it wholesale would
duplicate most of the report and cost an `/api/show` per entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import thinking_audit  # noqa: E402

_CANDIDATE = "candidate-not-in-a-role"


def _stances(uses: dict, model: str) -> set[str]:
    return {stance for stance, _where in uses.get(model, set())}


def test_a_passthrough_only_model_is_reported():
    uses = thinking_audit._collect({
        "passthrough": {"allowed_models": ["muse-glimmer:latest"]},
    })
    assert _stances(uses, "muse-glimmer:latest") == {_CANDIDATE}


def test_a_model_holding_a_role_is_not_also_listed_as_a_candidate():
    """qwen3.6:35b is in `allowed_models` AND runs the fast path. It must be
    reported under the role, which is what carries the decision."""
    uses = thinking_audit._collect({
        "fast_path": {"tool_capable_models": ["qwen3.6:35b"]},
        "passthrough": {"allowed_models": ["qwen3.6:35b", "muse-glimmer:latest"]},
    })
    assert _stances(uses, "qwen3.6:35b") == {"user-is-waiting"}
    assert _stances(uses, "muse-glimmer:latest") == {_CANDIDATE}


def test_a_deep_panel_role_also_wins_over_the_candidate_group():
    uses = thinking_audit._collect({
        "deep_panel": {"general": {"synthesizer": "glm-5.2:cloud"}},
        "passthrough": {"allowed_models": ["glm-5.2:cloud"]},
    })
    assert _stances(uses, "glm-5.2:cloud") == {"reasoning-is-the-product"}


def test_no_passthrough_block_is_not_an_error():
    uses = thinking_audit._collect({
        "fast_path": {"tool_capable_models": ["qwen3.6:35b"]},
    })
    assert _CANDIDATE not in {s for places in uses.values() for s, _ in places}


def test_the_real_config_lists_the_bakeoff_models_as_candidates():
    """⚠️ Reads the repo's config.yaml, so it fails if a bake-off model is
    promoted into a role — which is the correct moment to revisit this."""
    cfg = thinking_audit._load(thinking_audit._find_config())
    uses = thinking_audit._collect(cfg)
    for m in ("nemotron-3.5-lightning:latest", "muse-glimmer:latest"):
        assert _stances(uses, m) == {_CANDIDATE}, m
