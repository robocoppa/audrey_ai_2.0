#!/usr/bin/env python3
"""Build a case-by-model comparison table from eval harness JSON results.

WHAT THIS DOES

`scripts/eval_research.py --save-json` writes one flat JSON array of per-case
records (checks + latency, no answer text). This script reads one or more of
those files and renders a single markdown comparison: a case × model matrix
(pass/fail + total latency per cell), a per-model summary (pass rate, mean
TTFT/total, mean answer length), and a short list of every failure with its
failing checks. It is the seed for the hand-written report — the quality read
of the paired answers file stays yours.

Sweep runs (`--models`) suffix case names with ` [<model>]`; that suffix is
stripped here so the same case lands on one row across models. Feeding several
JSON files (e.g. one sweep run + one audrey_deep run) merges them into one
matrix — models are columns in first-seen order. Two records for the same
(case, model) pair: the later file wins (rerun semantics).

USAGE

    # One sweep run:
    .venv/bin/python scripts/eval_compare.py \\
        docs/testing/2026-07-10-code-sweep-results.json

    # Merge a sweep with a deep-protocol run, write the compare file:
    .venv/bin/python scripts/eval_compare.py \\
        docs/testing/2026-07-10-code-sweep-results.json \\
        docs/testing/2026-07-10-code-deep-results.json \\
        --out docs/testing/2026-07-10-code-compare.md

Exit 0 on success (regardless of pass/fail content — this is a renderer, not a
gate); 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The sweep suffix eval_research.py's _expand_sweep appends to case names.
_SWEEP_SUFFIX = re.compile(r"\s*\[[^\]]+\]\s*$")


def _case_key(name: str) -> str:
    """Case name with any trailing ' [<model>]' sweep suffix stripped."""
    return _SWEEP_SUFFIX.sub("", name)


def _fmt_cell(rec: dict) -> str:
    mark = "✅" if rec.get("ok") else "❌"
    total = rec.get("total_s")
    return f"{mark} {total:.0f}s" if isinstance(total, (int, float)) else mark


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt_mean(value: float | None, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value is not None else "—"


def build_table(records: list[dict]) -> str:
    """Render the full comparison markdown from flat result records.

    Pure string-in/string-out (no I/O) so the hermetic tests can pin it.
    Models become columns in first-seen order; cases become rows in
    first-seen order — both follow the run order, which the sweep groups
    by model, so feeding files in a deliberate order controls the layout.
    """
    models: list[str] = []
    cases: list[str] = []
    grid: dict[str, dict[str, dict]] = {}
    for rec in records:
        model = rec.get("model") or "?"
        key = _case_key(rec.get("name") or "?")
        if model not in models:
            models.append(model)
        if key not in cases:
            cases.append(key)
        grid.setdefault(key, {})[model] = rec

    lines: list[str] = ["## Case × model matrix", ""]
    lines.append("| case | " + " | ".join(f"`{m}`" for m in models) + " |")
    lines.append("|---" * (len(models) + 1) + "|")
    for key in cases:
        cells = [
            _fmt_cell(grid[key][m]) if m in grid[key] else "—"
            for m in models
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-model summary", ""]
    lines.append("| model | pass | mean ttft | mean total | mean answer chars |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        recs = [grid[key][m] for key in cases if m in grid[key]]
        passed = sum(1 for r in recs if r.get("ok"))
        ttft = _mean([r["ttft_s"] for r in recs
                      if isinstance(r.get("ttft_s"), (int, float))])
        total = _mean([r["total_s"] for r in recs
                       if isinstance(r.get("total_s"), (int, float))])
        alen = _mean([float(r["answer_len"]) for r in recs
                      if isinstance(r.get("answer_len"), (int, float))])
        lines.append(
            f"| `{m}` | {passed}/{len(recs)} | {_fmt_mean(ttft, 's')} "
            f"| {_fmt_mean(total, 's')} | {_fmt_mean(alen)} |"
        )

    failures = [
        (key, m, grid[key][m])
        for key in cases for m in models
        if m in grid[key] and not grid[key][m].get("ok")
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for key, m, rec in failures:
            failed = [c for c, v in (rec.get("checks") or {}).items() if v is False]
            bits = [", ".join(failed) or "(no check recorded)"]
            if rec.get("code_detail") and rec.get("checks", {}).get("code_runs") is False:
                bits.append(f"code: {rec['code_detail']}")
            if rec.get("error"):
                bits.append(f"error: {rec['error']}")
            lines.append(f"- **{key}** on `{m}` — {'; '.join(bits)}")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path,
                   help="one or more --save-json files from eval_research.py")
    p.add_argument("--out", type=Path, default=None,
                   help="write the markdown here instead of stdout "
                        "(convention: docs/testing/<date>-<desc>-compare.md)")
    args = p.parse_args()

    records: list[dict] = []
    for path in args.results:
        if not path.exists():
            print(f"error: results file not found: {path}", file=sys.stderr)
            return 2
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(loaded, list):
            print(f"error: {path} is not a JSON array of records", file=sys.stderr)
            return 2
        records.extend(loaded)

    if not records:
        print("error: no records in the given files", file=sys.stderr)
        return 2

    table = build_table(records)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
