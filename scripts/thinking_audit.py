"""Which models Audrey runs declare a thinking capability, and where they run.

Nothing in Audrey sets `think` except the vision path. Every other model — the
deep-panel workers, the synthesizers, the research verifier, the factchecker,
the fast path — runs at whatever its template defaults to, chosen by nobody.
This prints the map needed to decide that deliberately, in one command.

## The two questions, and why only one of them is here

**"Which models declare `thinking`?"** — answered here, from `/api/show`.
Capability decides whether the field can be sent at all: Ollama *rejects*
`think` for a model that does not declare it rather than ignoring it, so a
config-driven setting has to know this per model first.

**"Does setting it change anything?"** — NOT answered here, and the more
important one. `qwen3-vl:32b` declares `thinking` and produces the same
reasoning whether the flag is true or false (measured 2026-08-04, 93-101% of
baseline either way). Capability is a precondition for the setting mattering,
never evidence that it does. `scripts/thinking_probe.py` settles that per
model, and should be run on anything this report says is worth changing.

## Reading the output

Roles are grouped by what thinking is *for* in each, because the right answer
is not uniform:

  reasoning-is-the-product   deep-panel workers, synthesizers, the research
                             verifier. If these run at a default nobody chose,
                             there may be quality on the table for a config
                             line. Turning it UP is the interesting direction.
  reasoning-is-discarded     the vision sidecar, the keyframe pass, the video
                             summariser, the classifier — anything whose prompt
                             says "describe" or "label", where reasoning tokens
                             are billed and then thrown away.
  user-is-waiting            the fast path. Thinking here is latency somebody
                             watches, so it should be deliberate either way.
  structured-output          the factchecker. Thinking plus `format=` is
                             exactly where `pipeline/ledger.py` already records
                             glm-5.2 returning unusable JSON. Change with care
                             and re-read that note first.

`$` marks a cloud model. Thinking tokens are billed in wall clock locally and
in credits on cloud models, and Ollama cloud credits are a hard budget
constraint — so turning thinking up on a cloud deep pool is a spend decision,
not only a quality one.

## Running it

The static half — which model sits in which role — needs only `config.yaml`
and runs anywhere:

    # laptop
    uv run python scripts/thinking_audit.py

The capability column needs Ollama, so the full report runs on the box.
`audrey-ai` reaches it over `ollama-net`; fed on stdin, so no rebuild:

    # Unraid box, from /mnt/user/appdata/audrey_ai_2.0
    docker exec -i audrey-ai python3 - < scripts/thinking_audit.py

Environment:

    CONFIG       path to config.yaml (default: repo root next to this script)
    OLLAMA_HOST  default http://ollama:11434; unset/unreachable → static only
    TIMEOUT_S    per-model /api/show timeout (default 10)
    ONLY         comma-separated model names, to re-check a few
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("CONFIG", "")) if os.environ.get("CONFIG") else None
HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "10"))
ONLY = {m.strip() for m in os.environ.get("ONLY", "").split(",") if m.strip()}

# Where a role sits on the "is the reasoning the product?" axis. The grouping is
# the whole point of the report — a flat list of models says nothing about what
# to do with any of them.
STANCE = {
    "reasoning-is-the-product": "reasoning IS the product — turning thinking UP may buy quality",
    "reasoning-is-discarded": "reasoning is DISCARDED — thinking is billed and thrown away",
    "user-is-waiting": "a user is WAITING — thinking is latency, decide deliberately",
    "structured-output": "STRUCTURED output — thinking + format= has broken JSON here before",
}


def _find_config() -> Path:
    if CONFIG:
        return CONFIG
    return Path(__file__).resolve().parent.parent / "config.yaml"


def _load(path: Path) -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - only when run outside the venv
        sys.exit("PyYAML is not importable here; run inside audrey-ai or `uv run`.")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect(cfg: dict) -> dict[str, list[tuple[str, str]]]:
    """model -> [(stance, where)]. One row per place a model is actually used.

    Deliberately reads the real `config.yaml` rather than a hand-kept list:
    the point of the report is to catch a model running at an unexamined
    default, and a list maintained beside the config would drift into
    describing a stack nobody runs.
    """
    uses: dict[str, set[tuple[str, str]]] = {}

    def add(model: object, stance: str, where: str) -> None:
        if not isinstance(model, str) or not model:
            return
        uses.setdefault(model, set()).add((stance, where))

    # The task dimension (code / general / reasoning / vl) is collapsed to `*`.
    # It multiplies the rows by four and changes no decision — whether a
    # synthesizer should think is not a different question for the code pool
    # than the general one. Without this, glm-5.2 alone prints 24 lines.
    for pool_key in ("deep_panel", "deep_panel_cloud", "deep_panel_local", "deep_panel_research"):
        for roles in (cfg.get(pool_key) or {}).values():
            if not isinstance(roles, dict):
                continue
            for role, value in roles.items():
                # The research pool's factchecker is the constrained-decoding
                # stage; everything else in a deep pool drafts or merges prose.
                stance = "structured-output" if "factcheck" in role else "reasoning-is-the-product"
                for m in (value if isinstance(value, list) else [value]):
                    add(m, stance, f"{pool_key}.*.{role}")

    for m in (cfg.get("fast_path") or {}).get("tool_capable_models") or []:
        add(m, "user-is-waiting", "fast_path.tool_capable_models")

    # `vision.model` is normally commented out, meaning "highest-priority
    # healthy member of the vl pool" — so the pool is what actually runs, and
    # listing only an explicit override would show nothing on a default config.
    vision = cfg.get("vision") or {}
    add(vision.get("model"), "reasoning-is-discarded", "vision.model (explicit override)")
    for m in ((cfg.get("model_registry") or {}).get("vl") or []):
        if isinstance(m, dict):
            add(m.get("name"), "reasoning-is-discarded", "model_registry.vl (vision + keyframes)")

    video = ((cfg.get("kb") or {}).get("video") or {})
    add(video.get("summarise_model"), "reasoning-is-discarded", "kb.video.summarise_model")

    return uses


def _capability(model: str) -> tuple[str, str]:
    """('yes'|'no'|'?', detail). '?' means we could not ask, never 'no'."""
    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(  # noqa: S310 - fixed http(s) host from env
        f"{HOST}/api/show", data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310
            payload = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return "?", f"{type(e).__name__}"
    caps = payload.get("capabilities")
    if not isinstance(caps, list):
        return "?", "no capabilities field"
    return ("yes" if "thinking" in caps else "no"), ",".join(str(c) for c in caps)


def main() -> int:
    path = _find_config()
    if not path.is_file():
        sys.exit(f"config not found: {path}")
    uses = _collect(_load(path))
    if ONLY:
        uses = {m: v for m, v in uses.items() if m in ONLY}
    if not uses:
        sys.exit("no models found in config — wrong file?")

    print(f"config: {path}")
    print(f"ollama: {HOST}\n")

    caps: dict[str, tuple[str, str]] = {m: _capability(m) for m in sorted(uses)}
    unreachable = sum(1 for v in caps.values() if v[0] == "?")
    if unreachable == len(caps):
        print("!! Ollama unreachable — capability column is unknown, roles below are still valid.")
        print("!! Re-run on the box: docker exec -i audrey-ai python3 - < scripts/thinking_audit.py\n")

    by_stance: dict[str, dict[str, set[str]]] = {}
    for model, places in uses.items():
        for stance, where in places:
            by_stance.setdefault(stance, {}).setdefault(model, set()).add(where)

    for stance, blurb in STANCE.items():
        models = by_stance.get(stance) or {}
        if not models:
            continue
        print(f"── {stance} " + "─" * max(0, 58 - len(stance)))
        print(f"   {blurb}\n")
        for model in sorted(models):
            thinks, _detail = caps[model]
            mark = "$" if ":cloud" in model or "-cloud" in model else " "
            print(f"   {mark} {model:<32} thinking={thinks:<3}  "
                  f"{', '.join(sorted(models[model]))}")
        print()

    thinkers = sorted(m for m, (t, _) in caps.items() if t == "yes")
    unknown = sorted(m for m, (t, _) in caps.items() if t == "?")
    print("── summary " + "─" * 58)
    print(f"   {len(uses)} distinct models in use.")
    if unknown:
        # Never report an unasked question as a "no". A capability column that
        # says 0 because Ollama was unreachable would read as "nothing thinks,
        # there is nothing to do here" — the exact wrong conclusion.
        print(f"   {len(unknown)} could not be asked (Ollama unreachable) — UNKNOWN, not 'no'.")
    print(f"   {len(thinkers)} declare `thinking`"
          + (f", {len(caps) - len(thinkers) - len(unknown)} do not." if not unknown else "."))
    if thinkers:
        print()
        print("   Declaring it means the flag is ACCEPTED, not that setting it changes")
        print("   anything — qwen3-vl:32b declares it and ignores it. Probe before acting:")
        for m in thinkers:
            print(f"     MODEL={m} uv run python scripts/thinking_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
