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
from audrey.routes.openai.routes import VIRTUAL_MODELS

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "config.yaml"
_PULL_SCRIPT = _ROOT / "scripts" / "pull-models.sh"
_MEDIA_FETCHER_DOCKERFILE = _ROOT / "docker" / "media-fetcher.Dockerfile"
_ENV_EXAMPLE = _ROOT / ".env.example"
_CONFIG_SOURCE = _ROOT / "src" / "audrey" / "config.py"
_README = _ROOT / "README.md"

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


# ─── The fact-check stage's two silent preconditions ───────────────────
# Both of these fail the SAME way: `deep_panel.py`'s Stage-3 gate skips the
# whole stage and the answer renders without verdicts or a corrections block.
# Nothing errors, because the gate runs before anything is dispatched.

def _research_pools(cfg: dict) -> list[tuple[str, dict]]:
    return [
        (f"deep_panel_research/{task}", body)
        for task, body in (cfg.get("deep_panel_research") or {}).items()
        if isinstance(body, dict)
    ]


def test_every_factcheck_model_is_tool_capable(cfg):
    """The gate tests `factchecker in fast_path.tool_capable_models`.

    A fact-checker missing from that list is not a degraded fact-checker — it
    is no fact-checker at all, permanently, on every research turn. And the
    config reads perfectly: the role is filled, the model is registered, the
    model is pulled.

    ⚠️ This docstring used to cite glm as the example of a model excluded by
    THIS rule. That was stale — glm has been in `tool_capable_models` for some
    time, so what actually keeps it out of the fallback is the next test:
    it is a `researcher` in every one of these pools. Corrected 2026-08-29.
    """
    capable = {str(m) for m in (cfg.get("fast_path", {}).get("tool_capable_models") or [])}
    bad = [
        (pool, slot, body[slot])
        for pool, body in _research_pools(cfg)
        for slot in ("factchecker", "fallback_factcheck")
        if body.get(slot) and str(body[slot]) not in capable
    ]
    assert not bad, (
        f"fact-check models absent from `fast_path.tool_capable_models`: {bad}. "
        "The Stage-3 gate skips silently — no error, no verdicts, no clue."
    )


def test_the_factcheck_fallback_is_not_also_a_researcher(cfg):
    """A fallback that Stage 1 can disqualify is not a fallback.

    `HealthTracker` cools a model down on ANY failure, so a model that
    researches and then fact-checks can be knocked out of the second role by a
    transient failure in the first. That is the defect the fallback exists to
    cover — measured 2026-08-18, one upstream 503 on `deepseek-v4-pro:cloud`
    deleted a whole answer's verdicts. Naming a researcher here reintroduces it
    while looking like the fix.
    """
    bad = [
        (pool, body["fallback_factcheck"])
        for pool, body in _research_pools(cfg)
        if body.get("fallback_factcheck")
        and str(body["fallback_factcheck"]) in {str(r) for r in (body.get("researchers") or [])}
    ]
    assert not bad, (
        f"`fallback_factcheck` is also a `researcher` in the same pool: {bad}. "
        "One Stage-1 failure would cool down BOTH fact-check candidates and "
        "skip the stage exactly as before."
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


# ─── URL fetch client ownership, across config and image ──────────────

def test_url_fetch_uses_the_pinned_ytdlp_default_clients(cfg):
    """Do not pin a transient YouTube client in the slower-moving config.

    YouTube retired `android_vr` on 2026-08-17. Audrey still forced it after
    yt-dlp removed it from the defaults, turning upstream's repair into a no-op.
    An empty extractor argument delegates this volatile choice to the pinned
    downloader release while preserving Audrey's explicit override mechanism.
    """
    fetch = cfg["kb"]["fetch"]
    assert fetch["extractor_args"] == [""]
    assert fetch["max_client_attempts"] == 1


def test_media_fetcher_pin_knows_the_post_android_vr_clients():
    """The default-client repair first shipped in stable 2026.08.19."""
    text = _MEDIA_FETCHER_DOCKERFILE.read_text()
    match = re.search(
        r"^ARG YTDLP_VERSION=(\d{4})\.(\d{2})\.(\d{2})$",
        text,
        re.MULTILINE,
    )
    assert match, "media-fetcher must keep an explicit date-versioned yt-dlp pin"
    version = tuple(int(part) for part in match.groups())
    assert version >= (2026, 8, 19), (
        f"yt-dlp {'.'.join(match.groups())} predates the release that removed "
        "the retired android_vr client from YouTube defaults"
    )


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


# ── the eval harness must not launch into a stack that is still starting ────

_EVAL_ONBOX = _PULL_SCRIPT.parent / "eval-onbox.sh"


class TestEvalOnboxWaitsForTheStack:
    """A run started seconds after a recreate reported "12 cases, 0 passed".

    That is the worst possible shape for a failure: a total verdict, phrased
    exactly like a real one, on a suite that never reached a model. The gate
    that prevents it is only useful BEFORE the launch, so the order is what
    gets pinned here rather than the mere presence of the function.
    """

    def _text(self) -> str:
        return _EVAL_ONBOX.read_text()

    def test_the_readiness_gate_exists(self):
        assert "wait_ready" in self._text()

    def test_the_gate_runs_before_the_eval_container_launches(self):
        text = self._text()
        gate = text.index("for _c in ${READY_CONTAINERS}")
        launch = text.index('docker run -d --name "${CONTAINER}"')
        assert gate < launch, (
            "the readiness gate must run before `docker run` — a gate after the "
            "launch cannot stop the run it is meant to hold back"
        )

    def test_both_containers_the_harness_depends_on_are_gated(self):
        text = self._text()
        # audrey-ai is what was late; open-webui is what the harness actually
        # talks to, and it gets bounced whenever allowed_models changes.
        assert "READY_CONTAINERS:-audrey-ai open-webui" in text

    def test_there_is_an_escape_hatch(self):
        # A gate with no bypass becomes the thing people delete.
        assert "SKIP_READY" in self._text()

    def test_a_missing_container_is_skipped_not_fatal(self):
        # The script has to stay runnable against a box where one of these is
        # named differently, so an absent container warns rather than dies.
        text = self._text()
        assert "no such container, skipping" in text


class TestASetupFailureStillNotifies:
    """`die` exits before the notify block, so a gate failure was silent.

    Harmless while every `die` was an instant pre-flight check the operator
    would see in their own terminal. The readiness gate broke that: it can fire
    after a 180s wait, on a run launched with `nohup … &` and walked away from.
    2026-08-18 produced no eval, no ping, and no signal until the missing models
    were noticed by hand.
    """

    def _text(self) -> str:
        return _EVAL_ONBOX.read_text()

    def test_die_notifies_before_exiting(self):
        text = self._text()
        die = text[text.index("die() {"):]
        die = die[:die.index("\n")]
        assert "notify_setup_failure" in die, (
            "die() must notify before `exit 2` — a gate failure after a long "
            "wait is exactly the case nobody is watching the terminal for"
        )

    def test_the_notifier_is_defined_before_die_uses_it(self):
        text = self._text()
        assert text.index("notify_setup_failure() {") < text.index("die() {")

    def test_a_missing_watchdog_env_is_not_fatal(self):
        # The setup error has already happened; a failed send must not mask it
        # or change the exit code the caller sees.
        text = self._text()
        fn = text[text.index("notify_setup_failure() {"):text.index("die() {")]
        assert 'return 0' in fn and "|| true" in fn


# ─── Operational documentation must follow executable sources ────────


class TestOperationalDocumentation:
    def test_env_example_does_not_activate_yaml_overrides(self):
        text = _ENV_EXAMPLE.read_text()
        active = {
            line.split("=", 1)[0]
            for line in text.splitlines()
            if line and line[0].isupper() and "=" in line
        }
        quote = chr(34)
        override_names = set(re.findall(
            f"self[.]_set[(] *{quote}([A-Z0-9_]+){quote}",
            _CONFIG_SOURCE.read_text(),
        ))
        unexpected = sorted(active & override_names)
        assert not unexpected, (
            f".env.example actively overrides config.yaml: {unexpected}. "
            "Document supported overrides as commented opt-ins."
        )

    def test_retired_example_settings_do_not_return(self):
        retired = {
            "AUDREY_PORT",
            "AUDREY_LOG_LEVEL",
            "AUDREY_CONFIG_PATH",
            "CLIP_CACHE_DIR",
            "AUDREY_UPLOAD_ROOT",
            "AUDREY_MAX_UPLOAD_MB",
            "AUDREY_MAX_USER_BYTES",
            "FAST_PATH_CONFIDENCE",
            "CACHE_SIZE",
            "CACHE_TTL_SECONDS",
        }
        text = _ENV_EXAMPLE.read_text()
        declared = {
            line.lstrip("# ").split("=", 1)[0]
            for line in text.splitlines()
            if "=" in line
        }
        present = sorted(retired & declared)
        assert not present, f"inert or superseded example settings: {present}"

    def test_readme_virtual_model_table_matches_the_route_source(self):
        section = _README.read_text().partition("## Virtual models")[2]
        section = section.partition("## Chat Completions compatibility")[0]
        documented = set(re.findall(
            r"[|] [`](audrey_[a-z_]+)[`] [|]", section,
        ))
        assert documented == set(VIRTUAL_MODELS)
