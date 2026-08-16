#!/usr/bin/env python3
"""Count what `audrey_auto` escalation actually costs, from audrey-ai logs.

WHY

`Standing gotchas` has carried this line for weeks: "each escalation silently
buys a 3-worker panel with two cloud models". Nothing counted them. Cloud
credits are a hard budget, so the sentence is either a rounding error or the
largest uninstrumented spend in the system, and there is no way to tell by
reading code.

The awkward part is that an escalated turn LIES ABOUT ITSELF — it keeps the
fast stream's identity (`route: fast`, `Thinking` banner) while a planner,
three deep workers and a synthesis pass run underneath. The completion log is
no help either: `stream deep done` carries no `escalated=` field, so on the
streaming path an escalated panel is indistinguishable from a requested one.

What IS exact is the decision itself. `escalation_decision` logs one line
every time it says yes, and `node_complexity` logs one line for every turn
that entered the graph. Those two counts are a rate that needs no pairing and
no heuristics.

WHAT THE NUMBERS MEAN

  escalation rate   escalations / fast turns that REACHED the decision.
                    Turns whose fast model is not tool-capable never enter
                    the graph on the streaming path, so they cannot escalate
                    and are correctly absent from both sides of the ratio.

  would-escalate    the share of classifications sitting below
                    `confidence_ceiling`, ignoring every suppression but
                    `fallback:`. The gap between this and the actual rate is
                    the work the six suppression rules are doing. Read it as
                    a mechanism, not a forecast — it is an upper bound.

  cloud worker      per panel, from `config.yaml`: `<pool>.<task>.workers`
  calls             filtered to cloud models, plus the synthesizer. This is
                    the unit of spend; the escalation rate is how often an
                    unasked-for one is bought.

                    ⚠️ It is THREE, not the two the gotcha says. `general`,
                    `reasoning` and `code` each run one local worker and two
                    cloud ones — and then synthesize on `glm-5.2:cloud`. The
                    gotcha counted workers and forgot the synthesizer, so
                    every previous estimate was 33 % low. Pinned by a test
                    against the live config, so a pool edit reopens it.

⚠️ A rate is only worth reading with enough turns behind it. Every proportion
is printed with a Wilson 95 % interval, and the script says so out loud when
the denominator is too small to distinguish 5 % from 50 %.

⚠️ The box runs three clocks (host PDT, container logs MDT, `docker inspect`
UTC). `--since` compares against the string in the log line, which is the
CONTAINER clock.

USAGE

  docker logs audrey-ai 2>&1 | python3 scripts/analyze_escalations.py -

  python3 scripts/analyze_escalations.py /tmp/audrey.log
  python3 scripts/analyze_escalations.py /tmp/audrey.log --since 2026-08-15
  python3 scripts/analyze_escalations.py /tmp/audrey.log --json

Offline — no Ollama, no Qdrant, no network. Pure log parsing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ─── Log lines this script reads ──────────────────────────────────────────
#
# ⚠️ Each pattern is anchored on the FULL prefix emitted by the logging
# format, never on a fragment that could appear inside a message body. The
# repo has burned six sessions on parsers bounded loosely enough to match
# their own payload.
#
#   graph.py:372  complexity: 463 tokens -> fast (owui_task)
#   graph.py:321  classify: general (keyword:code_fence, conf=0.95)
#   graph.py:174  escalate: fast→deep (chars=42, conf=0.60, reason=too_short)
#   graph.py:477  deep_panel: pool=deep_panel task=general workers=3 ok=3 ...
_COMPLEXITY_RE = re.compile(
    r"\bcomplexity:\s+(?P<tokens>\d+)\s+tokens\s+->\s+(?P<mode>fast|deep)\s+"
    r"\((?P<reason>[^)]*)\)"
)
_CLASSIFY_RE = re.compile(
    r"\bclassify:\s+(?P<task>\S+)\s+\((?P<reason>[^,]+),\s+conf=(?P<conf>[0-9.]+)\)"
)
# The arrow is U+2192, written literally in graph.py's format string.
_ESCALATE_RE = re.compile(
    r"\bescalate:\s+fast→deep\s+\(chars=(?P<chars>\d+),\s+"
    r"conf=(?P<conf>[0-9.]+),\s+reason=(?P<reason>\w+)\)"
)
_PANEL_RE = re.compile(
    r"\bdeep_panel:\s+pool=(?P<pool>\S+)\s+task=(?P<task>\S+)\s+"
    r"workers=(?P<workers>\d+)\s+ok=(?P<ok>\d+)"
)
_TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})")

# Loose probes used ONLY to detect lines this script SHOULD have parsed and
# did not — a format change upstream must fail loudly, not silently deflate
# the numerator. See `Unparsed` in the report.
_LOOSE_PROBES = {
    "complexity": (re.compile(r"\bcomplexity:\s"), _COMPLEXITY_RE),
    "classify": (re.compile(r"\bclassify:\s"), _CLASSIFY_RE),
    "escalate": (re.compile(r"\bescalate:\s"), _ESCALATE_RE),
    "deep_panel": (re.compile(r"\bdeep_panel:\s"), _PANEL_RE),
}


@dataclass(slots=True)
class Tally:
    """Everything countable in one pass over the log."""

    fast_turns: int = 0
    deep_turns: int = 0
    fast_owui: int = 0                      # structurally suppressed subset
    complexity_reasons: Counter[str] = field(default_factory=Counter)

    classify_conf: list[float] = field(default_factory=list)
    classify_reasons: Counter[str] = field(default_factory=Counter)
    classify_below_ceiling: int = 0         # the low_conf trigger's population
    classify_fallback: int = 0              # suppressed: reason startswith "fallback:"

    escalations: int = 0
    escalate_reasons: Counter[str] = field(default_factory=Counter)
    escalate_chars: list[int] = field(default_factory=list)

    panels: int = 0
    panel_pools: Counter[str] = field(default_factory=Counter)
    panel_keys: list[tuple[str, str]] = field(default_factory=list)  # (pool, task)

    unparsed: Counter[str] = field(default_factory=Counter)
    lines_read: int = 0


def _reason_family(reason: str) -> str:
    """Collapse `keyword:code_fence` to `keyword`, leaving bare reasons alone.

    The families are what `classify.py` pins a confidence to — `keyword` is
    always 0.95, `short_skip` always 0.5, `router` whatever the model said —
    so the family is the unit that predicts escalation, not the suffix.
    """
    return reason.split(":", 1)[0] if ":" in reason else reason


def _line_after_since(line: str, since: str | None) -> bool:
    """True if the line is at or after `since`, or carries no timestamp.

    Undated lines pass: dropping signal we cannot date is worse than
    including it, and continuation lines of a wrapped record are undated.
    """
    if since is None:
        return True
    m = _TIMESTAMP_RE.match(line)
    if not m:
        return True
    return m.group("ts") >= since


def parse_log(lines: object, *, since: str | None, ceiling: float) -> Tally:
    """Single pass. Every counter here is an exact count of a logged event."""
    t = Tally()
    for raw in lines:  # type: ignore[union-attr]
        line = raw.rstrip("\n")
        if not _line_after_since(line, since):
            continue
        t.lines_read += 1

        m = _COMPLEXITY_RE.search(line)
        if m:
            reason = m.group("reason")
            t.complexity_reasons[reason] += 1
            if m.group("mode") == "fast":
                t.fast_turns += 1
                if "owui_task" in reason:
                    t.fast_owui += 1
            else:
                t.deep_turns += 1
            continue

        m = _CLASSIFY_RE.search(line)
        if m:
            conf = float(m.group("conf"))
            reason = m.group("reason")
            t.classify_conf.append(conf)
            t.classify_reasons[_reason_family(reason)] += 1
            # Mirror `escalation_decision` exactly: `conf > 0`, strictly below
            # the ceiling, and not the all-attempts-failed fallback. ⚠️ Only
            # `fallback:` is suppressed — `fallback_keyword:` is a different
            # reason one underscore away, and it DOES escalate.
            if reason.startswith("fallback:"):
                t.classify_fallback += 1
            elif 0.0 < conf < ceiling:
                t.classify_below_ceiling += 1
            continue

        m = _ESCALATE_RE.search(line)
        if m:
            t.escalations += 1
            t.escalate_reasons[m.group("reason")] += 1
            t.escalate_chars.append(int(m.group("chars")))
            continue

        m = _PANEL_RE.search(line)
        if m:
            t.panels += 1
            t.panel_pools[m.group("pool")] += 1
            t.panel_keys.append((m.group("pool"), m.group("task")))
            continue

        # Nothing matched. Did a line that LOOKS like one of ours slip past?
        for name, (loose, strict) in _LOOSE_PROBES.items():
            if loose.search(line) and not strict.search(line):
                t.unparsed[name] += 1
                break
    return t


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials.

    Wilson rather than normal-approximation because these proportions live
    near 0 and n is often small — the normal interval would report a negative
    lower bound and read as precision that is not there.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def read_config(config_path: Path) -> dict[str, Any]:
    """Load `config.yaml`, or `{}` when it is missing or unparseable.

    Read ONCE and shared by the pool map and the ceiling — two reads could
    disagree if the file changed under a long run, and a ceiling that does not
    match the pools it is reported next to is worse than no ceiling.

    Degrading to `{}` is deliberate: the turn census is still worth printing
    without cost attribution, and this script must never be the reason someone
    cannot read their own logs. ⚠️ The catch is narrow on purpose — a blind
    `except` here would also swallow a bug in this file and report zero spend.
    """
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_pools(raw: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Split a loaded config into the deep-panel pools and a model→location map."""
    pools = {k: v for k, v in raw.items()
             if k.startswith("deep_panel") and isinstance(v, dict)}
    locations: dict[str, str] = {}
    for specs in (raw.get("model_registry") or {}).values():
        if not isinstance(specs, list):
            continue
        for spec in specs:
            if isinstance(spec, dict) and spec.get("name"):
                locations[str(spec["name"])] = str(spec.get("location", "local"))
    return pools, locations


def _ceiling_from(raw: dict[str, Any]) -> float:
    """`agentic.escalation.confidence_ceiling`, defaulting the way graph.py does.

    ⚠️ The default matters more than it looks. At 0.95 the only confidence
    `classify.py` emits that clears the bar is a keyword hit, which is pinned
    at exactly 0.95 — so a wrong default here would move the would-escalate
    population from "almost none" to "almost all".
    """
    try:
        cfg = ((raw.get("agentic") or {}).get("escalation") or {})
        return float(cfg.get("confidence_ceiling", 0.95))
    except (AttributeError, TypeError, ValueError):
        return 0.95


def is_cloud(model: str, locations: dict[str, str]) -> bool:
    """Cloud by the registry when it knows the model, by name when it does not.

    ⚠️ The suffix is not uniform — `kimi-k2.6:cloud` and `qwen3.5:397b-cloud`
    are both cloud — so the heuristic checks for the token anywhere after the
    tag separator rather than an exact `:cloud` ending.
    """
    known = locations.get(model)
    if known is not None:
        return known == "cloud"
    return "cloud" in model.split(":", 1)[-1]


def panel_cost(
    pool: str, task: str, pools: dict[str, dict[str, Any]], locations: dict[str, str]
) -> tuple[int, int] | None:
    """(cloud_calls, total_calls) for one panel: workers plus the synthesizer.

    `None` when the pool/task is not in config — a pool that was renamed, or a
    log older than the config. Counting those as zero would quietly understate
    spend, which is the one direction that matters here.
    """
    body = (pools.get(pool) or {}).get(task)
    if not isinstance(body, dict):
        return None
    models = [str(m) for m in (body.get("workers") or [])]
    # Staged research pools carry `researchers` + `writer` instead.
    models += [str(m) for m in (body.get("researchers") or [])]
    for role in ("synthesizer", "writer", "verifier"):
        if body.get(role):
            models.append(str(body[role]))
    if not models:
        return None
    return sum(1 for m in models if is_cloud(m, locations)), len(models)


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "  n/a"
    return f"{100.0 * k / n:5.1f}%"


def build_report(t: Tally, pools: dict[str, dict[str, Any]],
                 locations: dict[str, str], ceiling: float) -> dict[str, Any]:
    """Everything the text and JSON renderers both need, computed once."""
    lo, hi = wilson(t.escalations, t.fast_turns)
    would_n = t.classify_below_ceiling + t.classify_fallback
    would_lo, would_hi = wilson(t.classify_below_ceiling, len(t.classify_conf))

    cloud_calls = 0
    total_calls = 0
    unknown_panels = 0
    for pool, task in t.panel_keys:
        cost = panel_cost(pool, task, pools, locations)
        if cost is None:
            unknown_panels += 1
            continue
        cloud_calls += cost[0]
        total_calls += cost[1]

    # Cost per escalation is a CONFIG constant, not something to infer from
    # pairing escalations to panels across interleaved concurrent turns. The
    # mean over observed panels is the honest stand-in when several pools ran.
    cloud_per_panel = (cloud_calls / t.panels) if t.panels else 0.0

    return {
        "lines_read": t.lines_read,
        "turns": {
            "fast": t.fast_turns,
            "deep": t.deep_turns,
            "fast_owui_task": t.fast_owui,
        },
        "escalations": {
            "count": t.escalations,
            "rate": (t.escalations / t.fast_turns) if t.fast_turns else 0.0,
            "ci95": [lo, hi],
            "by_reason": dict(t.escalate_reasons),
            "median_chars": (
                sorted(t.escalate_chars)[len(t.escalate_chars) // 2]
                if t.escalate_chars else None
            ),
        },
        "classification": {
            "observed": len(t.classify_conf),
            "below_ceiling": t.classify_below_ceiling,
            "fallback_suppressed": t.classify_fallback,
            "would_escalate_rate": (
                t.classify_below_ceiling / len(t.classify_conf)
                if t.classify_conf else 0.0
            ),
            "would_escalate_ci95": [would_lo, would_hi],
            "by_family": dict(t.classify_reasons),
            "ceiling": ceiling,
            "_n": would_n,
        },
        "panels": {
            "count": t.panels,
            "by_pool": dict(t.panel_pools),
            "cloud_calls": cloud_calls,
            "total_calls": total_calls,
            "cloud_per_panel": cloud_per_panel,
            "escalated_share": (t.escalations / t.panels) if t.panels else 0.0,
            "unattributed": unknown_panels,
        },
        "unparsed": dict(t.unparsed),
    }


def render(rep: dict[str, Any]) -> str:
    out: list[str] = []
    w = out.append
    turns = rep["turns"]
    esc = rep["escalations"]
    cls = rep["classification"]
    pan = rep["panels"]

    w("═══ audrey_auto escalation cost ═══")
    w(f"lines read: {rep['lines_read']:,}")
    w("")
    w("── Turn census (turns that entered the graph) ──")
    w(f"  fast : {turns['fast']:6,}   (of which OWUI utility turns: "
      f"{turns['fast_owui_task']:,} — structurally suppressed)")
    w(f"  deep : {turns['deep']:6,}   (requested directly, not escalated)")
    w("")
    w("── Escalations ──")
    if turns["fast"] == 0:
        w("  no fast turns in range — nothing to rate.")
    else:
        lo, hi = esc["ci95"]
        w(f"  escalated: {esc['count']:,} of {turns['fast']:,} fast turns  "
          f"= {100 * esc['rate']:.1f}%  (95% CI {100 * lo:.1f}–{100 * hi:.1f}%)")
        for reason, n in sorted(esc["by_reason"].items(), key=lambda kv: -kv[1]):
            w(f"    {reason:<14} {n:6,}  {_pct(n, esc['count'])} of escalations")
        if esc["median_chars"] is not None:
            w(f"    median answer length at escalation: {esc['median_chars']} chars")
    w("")
    w("── What the suppressions are worth ──")
    if cls["observed"] == 0:
        w("  no classify: lines in range.")
    else:
        wlo, whi = cls["would_escalate_ci95"]
        w(f"  below confidence_ceiling={cls['ceiling']}: "
          f"{cls['below_ceiling']:,} of {cls['observed']:,} classifications "
          f"= {100 * cls['would_escalate_rate']:.1f}% "
          f"(95% CI {100 * wlo:.1f}–{100 * whi:.1f}%)")
        w(f"  suppressed as fallback:  {cls['fallback_suppressed']:,}")
        w("  ↑ upper bound on the low_conf trigger alone. The gap between this")
        w("    and the actual rate above is what tool_rounds / memory_hits /")
        w("    owui_task / audrey_fast are saving.")
        w("  confidence by reason family:")
        for fam, n in sorted(cls["by_family"].items(), key=lambda kv: -kv[1]):
            w(f"    {fam:<18} {n:6,}")
    w("")
    w("── Panels bought ──")
    w(f"  panels run: {pan['count']:,}")
    for pool, n in sorted(pan["by_pool"].items(), key=lambda kv: -kv[1]):
        w(f"    {pool:<22} {n:6,}")
    if pan["count"]:
        w(f"  escalated share of panels: {100 * pan['escalated_share']:.1f}%")
        w(f"  cloud model calls: {pan['cloud_calls']:,} of {pan['total_calls']:,} "
          f"({pan['cloud_per_panel']:.1f} cloud calls per panel)")
        w(f"  ≈ cloud calls bought by escalation alone: "
          f"{esc['count'] * pan['cloud_per_panel']:.0f}")
    if pan["unattributed"]:
        w(f"  ⚠️ {pan['unattributed']:,} panels had no matching pool/task in "
          f"config.yaml — cost NOT counted, so the figure above is a floor.")
    if not pan["count"]:
        w("    (none — no deep_panel: lines in range)")

    # Guards. Both of these have cost this repo a session before: a parser
    # that silently matched nothing, and a rate quoted off a handful of turns.
    w("")
    if rep["unparsed"]:
        w("⚠️ LINES THAT LOOK LIKE OURS BUT DID NOT PARSE — a log format changed;")
        w("   every count above is understated until the regex is fixed:")
        for name, n in sorted(rep["unparsed"].items()):
            w(f"     {name}: {n:,}")
    if 0 < turns["fast"] < 30:
        w(f"⚠️ Only {turns['fast']} fast turns. The interval above spans "
          "too much to act on — this is a mechanism read, not a rate.")
    elif turns["fast"] == 0 and rep["lines_read"]:
        w("⚠️ No `complexity:` lines at all. Either the range is empty, or the "
          "logs predate the line — check the range before reading zero as a "
          "finding.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Count audrey_auto escalations and the cloud calls they buy.",
    )
    ap.add_argument("logfile", help="path to a log dump, or '-' for stdin")
    ap.add_argument("--since", default=None,
                    help="drop lines before this timestamp (CONTAINER clock, "
                         "e.g. 2026-08-15 or 2026-08-15T09:00:00)")
    ap.add_argument("--config", default="config.yaml",
                    help="config.yaml to read pool shapes from (default: ./config.yaml)")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="override agentic.escalation.confidence_ceiling")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    raw = read_config(Path(args.config))
    pools, locations = load_pools(raw)

    ceiling = args.ceiling
    if ceiling is None:
        ceiling = _ceiling_from(raw)

    if args.logfile == "-":
        tally = parse_log(sys.stdin, since=args.since, ceiling=ceiling)
    else:
        path = Path(args.logfile)
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            return 2
        with path.open(encoding="utf-8", errors="replace") as fh:
            tally = parse_log(fh, since=args.since, ceiling=ceiling)

    report = build_report(tally, pools, locations, ceiling)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
