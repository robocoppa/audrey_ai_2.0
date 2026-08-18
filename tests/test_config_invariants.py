"""Standing gotchas that became assertions.

Every test here replaces a rule that used to live only as prose in
`docs/PROJECT_STATE.md` → `Standing gotchas`. The prose survives as a one-line
pointer; the enforcement is here.

The distinction that decides whether a rule belongs in this file: **can the
repo tell, on its own, that the rule is broken?** A fact about somebody else's
software (Ollama's renderer, OWUI's model cache, `docker logs` writing to
stderr) can only ever be prose — nothing here can check it. A fact about THIS
repo's own config, on the other hand, is a rule I was asking a future reader to
hold in their head when a `assert` could have refused the edit outright.

⚠️ These assert against the REAL `config.yaml` and the REAL `pull-models.sh`.
That is the point: a hermetic fixture would pass forever while the deployed
config drifted. When one of these fails, the question is not "how do I make the
test pass" — it is "did I mean to change this?" Each test says what a legitimate
change looks like.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from audrey.pipeline.classify import classify

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "config.yaml"
_PULL_SCRIPT = _ROOT / "scripts" / "pull-models.sh"

_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_model_inventory as cmi  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(_CONFIG.read_text())


def _leader(pool: list[dict]) -> dict:
    """The entry the fast path actually reaches for: highest `priority`."""
    return max(pool, key=lambda m: m.get("priority", 0))


# ─── Cloud spend ───────────────────────────────────────────────────────
# Replaces: "THE FAST-PATH PRIMARY FOR `general` STAYS LOCAL" and
#           "`reasoning` IS THE ONLY FAST-PATH TASK THAT LANDS ON CLOUD".

def test_the_general_fast_path_primary_is_local(cfg):
    """`general` is the ordinary-chat pool, so its leader bills every turn.

    A cloud model here puts routine conversation on paid inference. This has
    been done once — `kimi-k3:cloud` was promoted to lead it and reverted the
    same day on cost. Cloud models earn deep-pool slots, not this one.
    """
    leader = _leader(cfg["model_registry"]["general"])
    assert leader["location"] == "local", (
        f"`general` now leads with {leader['name']} ({leader['location']}). "
        "Every ordinary chat turn is billed. Cloud models earn deep-pool slots "
        "only, and only against measured quality."
    )


def test_reasoning_is_the_only_fast_path_pool_that_leads_with_cloud(cfg):
    """Which pool leads with cloud decides what a router MISROUTE costs.

    `general` ↔ `code` is nearly free — both lead with the same local model, so
    the same weights answer either way. `→ reasoning` spends cloud credit on a
    turn that never entered a panel, a cost path `analyze_escalations.py` cannot
    see because it counts panels.

    If a second pool goes cloud, that arithmetic changes and the escalation
    analysis silently under-reports. Change this deliberately or not at all.
    """
    cloud_leaders = {
        task: _leader(pool)["name"]
        for task, pool in cfg["model_registry"].items()
        if _leader(pool)["location"] == "cloud"
    }
    assert set(cloud_leaders) == {"reasoning"}, (
        f"fast-path pools leading with cloud: {cloud_leaders}. Only `reasoning` "
        "is budgeted for that; weigh what a router misroute now costs."
    )


# ─── The two model authorities ─────────────────────────────────────────
# Replaces the manual chore in `pull-models.sh`'s own header: "▶
# `check_model_inventory.py` compares CONFIG to the box; it cannot see this
# file. Reconcile the two by hand whenever either changes."
#
# It had drifted in BOTH directions at once before anyone reconciled it: names
# in the script never pulled, a name in the script config never used while
# config dispatched a different one (`minimax-m2.7` vs `minimax-m3`), and two
# models on the box the script never listed — which reached config only through
# `passthrough.allowed_models`, so a rebuilt box would have come up missing
# them. None of that is visible from either file read alone.

def _pull_script_models() -> set[str]:
    text = _PULL_SCRIPT.read_text()
    names: set[str] = set()
    for array in ("LOCAL_MODELS", "CLOUD_MODELS"):
        body = re.search(rf"^{array}=\((.*?)^\)", text, re.S | re.M)
        assert body, f"{array} not found in pull-models.sh — did the shape change?"
        names |= {cmi.normalise(m) for m in re.findall(r'^\s*"([^"]+)"', body.group(1), re.M)}
    return names


def _config_model_names(cfg: dict) -> set[str]:
    # `config_models` descends generically rather than walking a hand-written
    # role list — that hand-written list is what missed
    # `deep_panel_research.*.factchecker` and `kb.video.summarise_model`.
    return {cmi.normalise(name) for name in cmi.config_models(cfg)}


def test_every_model_the_config_names_is_pulled_by_the_script(cfg):
    """A model config dispatches but the script never pulls is a phantom on a
    rebuilt box: it passes boot validation (which checks config against itself),
    passes `HealthTracker.is_healthy` (unknown → True), defaults to
    `location="local"`, takes a GPU-gate slot, and only then fails the call.
    """
    missing = sorted(_config_model_names(cfg) - _pull_script_models())
    assert not missing, (
        f"config names models `pull-models.sh` never pulls: {missing}. "
        "A rebuilt box would come up without them."
    )


def test_the_script_pulls_nothing_the_config_never_uses(cfg):
    """The other direction: a name only the script knows is dead weight at best
    and a decoy at worst — it looks like evidence a model is available.

    ⚠️ `pull-models.sh` is a statement of INTENT, never proof of what is on the
    box. `ollama list` is the only authority on that, and no test can reach it.
    """
    unused = sorted(_pull_script_models() - _config_model_names(cfg))
    assert not unused, (
        f"`pull-models.sh` pulls models the config never names: {unused}. "
        "Drop them, or wire them up — not pulled means gone, and not USED "
        "means the same."
    )


# ─── Thinking policy ───────────────────────────────────────────────────

def test_every_no_thinking_model_is_one_the_config_actually_dispatches(cfg):
    """A typo here is silent in BOTH directions.

    `think_for` matches on the exact model string, so a misspelled entry never
    fires — the model keeps thinking and nothing says the policy was ignored.
    And a correctly-spelled model the config no longer uses leaves a rule
    nobody can trace to a call site.
    """
    named = {str(m) for m in (cfg.get("thinking", {}).get("no_thinking_models") or [])}
    unknown = sorted(named - _config_model_names(cfg))
    assert not unknown, (
        f"`thinking.no_thinking_models` names models the config never "
        f"dispatches: {unknown}. The policy silently does nothing."
    )


def test_a_role_switch_is_a_bool_not_a_model_list(cfg):
    """The two knobs read differently and mixing them up fails quietly:
    `deep_worker: ["some-model"]` is truthy, so it would turn thinking off for
    the WHOLE role while looking like a narrow per-model rule."""
    thinking = cfg.get("thinking", {})
    for role in ("deep_worker", "deep_synth", "ledger_structure"):
        if role in thinking:
            assert isinstance(thinking[role], bool), (
                f"`thinking.{role}` must be a bool — a list here reads as "
                "truthy and silently widens to the entire role. Use "
                "`no_thinking_models` to name models."
            )


# ─── The YAML that reads one way and looks another ─────────────────────
# Replaces the parenthetical in "A temporary toggle belongs in `.env`, not
# `config.yaml`": an on-box `sed` once produced a duplicate key, and YAML keeps
# the LAST one — so the file read `false` while looking `true`.

class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key.

    PyYAML's default is last-wins, silently. That is the whole failure: the
    file you read and the file the process read disagreed, and nothing said so.
    """


def _no_duplicates(loader: _DuplicateKeyLoader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise AssertionError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                "— YAML keeps the LAST, so this file does not do what it looks "
                "like it does."
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def test_config_yaml_has_no_duplicate_keys():
    yaml.load(_CONFIG.read_text(), Loader=_DuplicateKeyLoader)  # noqa: S506


# ─── Escalation semantics, across two files ────────────────────────────
# Replaces the confidence arithmetic under `ESCALATION COST`: the trigger is
# `conf < ceiling` STRICTLY, and `classify()` hardcodes its confidences as
# literals. Neither file states the relationship; only the pair does.

@pytest.mark.asyncio
async def test_a_strong_keyword_route_never_escalates(cfg):
    """A keyword-routed turn is meant to be a decided turn.

    `classify()` returns 0.95 for a strong keyword signal and the escalation
    ceiling is 0.95 with a strict `<` — so keyword routes sit exactly ON the
    bar and never fire. Lower either number in isolation and every fenced-code
    question starts buying a panel; nothing in either file would say so.

    `ollama=None` is safe here: a strong signal returns before the router runs.
    """
    ceiling = float(cfg["agentic"]["escalation"]["confidence_ceiling"])
    _task, reason, conf = await classify(
        None,  # type: ignore[arg-type]
        router_model="unused",
        router_timeout_s=1.0,
        max_router_strikes=1,
        user_text="```python\nprint('hi')\n```",
    )
    assert reason.startswith("keyword:"), f"expected a keyword route, got {reason}"
    assert conf >= ceiling, (
        f"a keyword route now scores {conf} against a {ceiling} ceiling with a "
        "strict `<` trigger — every keyword-routed turn escalates into a panel."
    )
