#!/usr/bin/env python3
"""Analyze production synth-draft sizes from audrey-ai logs.

WHY

The Lesson 8 audit flagged `_format_drafts_for_synth` for not capping
draft sizes (`pipeline/synthesize.py`). Worst-case math: 4 workers x
~8 KB drafts = ~32 KB handed to the synthesizer. Whether that's
actually a problem in production depends on the real draft-size
distribution — which we don't have, because draft sizes aren't stored
in the chat archive (only final answers are).

This script is the analysis half of a measurement loop:

  1. Phase 12 added a `log.info` line in `_format_drafts_for_synth`
     emitting one parseable record per synth call:
     `synth_draft_sizes: drafts=N shown=M total_chars=T per_draft_chars=[c1, c2, ...]`
  2. After production has run for a while with that instrumentation,
     pull the logs and feed them to this script.
  3. The script reports per-draft and total-bundle percentiles, plus
     a histogram, so the cap decision is data-driven.

WHAT TO DO WITH THE OUTPUT

Look at the p95 and p99 of the *total* distribution. Worst-case
matters more here than mean — synthesizers fail catastrophically
when context overflows; they don't gracefully degrade.

  - p99 < 16 KB total → accept the finding. Synth budgets handle
    32 KB comfortably; current worst case isn't close.
  - p99 in 16-24 KB → judgment call. Pick a per-draft cap that
    leaves room (e.g. p95 of single-draft + 25 % headroom).
  - p99 > 24 KB → ship a per-draft cap soon. The cap value comes
    from the p95 of single-draft sizes.

USAGE

  # On Unraid, dump the logs to a file first (the instrumentation
  # uses `log.info`, so default container logging captures it).
  #
  # ⚠️ The file redirect goes FIRST. `docker logs X 2>&1 > file` writes an
  # almost-empty file: `2>&1` points stderr at the terminal (the stdout of
  # the moment), then `>` moves stdout to the file — and docker writes the
  # container log to stderr. The pipe form below is fine, because the pipe
  # is already stdout by the time `2>&1` is evaluated.
  docker logs audrey-ai > /tmp/audrey.log 2>&1

  # Then analyze:
  python3 scripts/analyze_draft_sizes.py /tmp/audrey.log

  # Read from stdin:
  docker logs audrey-ai 2>&1 | python3 scripts/analyze_draft_sizes.py -

  # JSON output:
  python3 scripts/analyze_draft_sizes.py /tmp/audrey.log --json

  # Filter to a time range (logs include ISO timestamps from
  # `logging.basicConfig`):
  python3 scripts/analyze_draft_sizes.py /tmp/audrey.log \\
      --since 2026-05-26T00:00:00

The script is OFFLINE — no Ollama, no Qdrant, no network. Pure log
parsing + percentile math.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# Log-line format we're matching:
#   synth_draft_sizes: drafts=N shown=M total_chars=T per_draft_chars=[c1, c2, ...]
# The line will be embedded in a standard logging.basicConfig line:
#   2026-05-26 12:34:56,789 INFO audrey.pipeline.synthesize: synth_draft_sizes: ...
_LINE_RE = re.compile(
    r"synth_draft_sizes:\s+"
    r"drafts=(?P<drafts>\d+)\s+"
    r"shown=(?P<shown>\d+)\s+"
    r"total_chars=(?P<total>\d+)\s+"
    r"per_draft_chars=(?P<per>\[[^\]]*\])"
)
_TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})")


@dataclass(slots=True)
class Record:
    """One synth call's draft sizes."""

    drafts: int        # total workers in the panel (including empty)
    shown: int         # non-empty drafts (the ones the synth actually merges)
    total_chars: int   # sum of all non-empty draft sizes
    per_draft: list[int]


@dataclass(slots=True)
class Stats:
    records: int = 0
    per_draft_sizes: list[int] = field(default_factory=list)  # one entry per draft
    total_sizes: list[int] = field(default_factory=list)       # one entry per synth call
    drafts_count_distribution: Counter[int] = field(default_factory=Counter)


# Total-size buckets in KB. A "21 KB" total falls into the third bucket.
_TOTAL_BUCKETS_KB = [
    ("0-4 KB",   0, 4),
    ("4-8 KB",   4, 8),
    ("8-16 KB",  8, 16),
    ("16-24 KB", 16, 24),
    ("24-32 KB", 24, 32),
    ("32+ KB",   32, 10_000),
]


def _parse_line(line: str) -> Record | None:
    """Parse one log line. Returns None for non-matching lines."""
    m = _LINE_RE.search(line)
    if not m:
        return None
    try:
        per_draft = ast.literal_eval(m.group("per"))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(per_draft, list) or not all(isinstance(c, int) for c in per_draft):
        return None
    return Record(
        drafts=int(m.group("drafts")),
        shown=int(m.group("shown")),
        total_chars=int(m.group("total")),
        per_draft=per_draft,
    )


def _line_after_since(line: str, since: str | None) -> bool:
    """True if the line's timestamp is at or after `since` (or no filter set).

    Lines without a parseable timestamp pass the filter (defensive — we'd
    rather include a record we can't date than drop signal).
    """
    if since is None:
        return True
    m = _TIMESTAMP_RE.match(line)
    if not m:
        return True
    return m.group("ts") >= since


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Linear-interpolation percentile on a pre-sorted list. Empty -> 0."""
    if not sorted_values:
        return 0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    # nearest-rank: pick index `ceil(pct/100 * n) - 1`
    k = max(0, min(len(sorted_values) - 1, int(pct / 100.0 * len(sorted_values))))
    return sorted_values[k]


def _aggregate(records: list[Record]) -> Stats:
    stats = Stats(records=len(records))
    for r in records:
        stats.total_sizes.append(r.total_chars)
        stats.per_draft_sizes.extend(c for c in r.per_draft if c > 0)
        stats.drafts_count_distribution[r.shown] += 1
    stats.total_sizes.sort()
    stats.per_draft_sizes.sort()
    return stats


def _bucket_label(total_chars: int) -> str:
    kb = total_chars / 1024.0
    for label, lo, hi in _TOTAL_BUCKETS_KB:
        if lo <= kb < hi:
            return label
    return _TOTAL_BUCKETS_KB[-1][0]


def _format_kb(n_chars: int) -> str:
    """Render character count as KB with one decimal."""
    return f"{n_chars / 1024.0:.1f} KB"


def _report_human(stats: Stats) -> None:
    if stats.records == 0:
        print("\nNo synth_draft_sizes records found.")
        print("Check that:")
        print("  - The instrumentation patch is deployed (Phase 12).")
        print("  - The log file actually contains audrey-ai's stdout/stderr.")
        print("  - You've had at least one deep request since deploy.")
        return
    print(f"\nSynth-draft size analysis - {stats.records} synth calls")
    print("-" * 72)
    print("  Per-draft sizes (one entry per non-empty draft):")
    pds = stats.per_draft_sizes
    print(f"    samples : {len(pds)}")
    print(f"    p50     : {_format_kb(_percentile(pds, 50))}")
    print(f"    p95     : {_format_kb(_percentile(pds, 95))}")
    print(f"    p99     : {_format_kb(_percentile(pds, 99))}")
    print(f"    max     : {_format_kb(max(pds))}" if pds else "    max     : -")
    print()
    print("  Total-bundle sizes (sum of per-call drafts, what synth actually sees):")
    ts = stats.total_sizes
    print(f"    samples : {len(ts)}")
    print(f"    p50     : {_format_kb(_percentile(ts, 50))}")
    print(f"    p95     : {_format_kb(_percentile(ts, 95))}")
    print(f"    p99     : {_format_kb(_percentile(ts, 99))}")
    print(f"    max     : {_format_kb(max(ts))}" if ts else "    max     : -")
    print()
    print("  Total-bundle distribution:")
    bucket_counts: Counter[str] = Counter()
    for t in ts:
        bucket_counts[_bucket_label(t)] += 1
    max_count = max(bucket_counts.values()) if bucket_counts else 1
    for label, _lo, _hi in _TOTAL_BUCKETS_KB:
        count = bucket_counts.get(label, 0)
        bar = "#" * min(40, int(40 * count / max_count))
        print(f"    {label:>9} : {count:>5}  {bar}")
    print()
    print("  Drafts-shown count distribution (calls grouped by # non-empty drafts):")
    for k in sorted(stats.drafts_count_distribution.keys()):
        n = stats.drafts_count_distribution[k]
        print(f"    shown={k}: {n} calls")
    print()


def _report_json(stats: Stats) -> None:
    pds = stats.per_draft_sizes
    ts = stats.total_sizes
    out = {
        "records": stats.records,
        "per_draft": {
            "samples": len(pds),
            "p50": _percentile(pds, 50),
            "p95": _percentile(pds, 95),
            "p99": _percentile(pds, 99),
            "max": pds[-1] if pds else 0,
            "median": int(median(pds)) if pds else 0,
        },
        "total_bundle": {
            "samples": len(ts),
            "p50": _percentile(ts, 50),
            "p95": _percentile(ts, 95),
            "p99": _percentile(ts, 99),
            "max": ts[-1] if ts else 0,
            "median": int(median(ts)) if ts else 0,
        },
        "drafts_shown_distribution": dict(stats.drafts_count_distribution),
    }
    print(json.dumps(out, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyze synth_draft_sizes log lines from audrey-ai.",
    )
    p.add_argument(
        "logfile", type=str,
        help="Path to the audrey-ai log file. Use '-' to read from stdin.",
    )
    p.add_argument(
        "--since", type=str, default=None,
        help=(
            "ISO timestamp (YYYY-MM-DD[THH:MM:SS]). Only lines at or "
            "after this time are included. Lines without a parseable "
            "timestamp pass through unfiltered."
        ),
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the human table.",
    )
    args = p.parse_args()

    records: list[Record] = []
    if args.logfile == "-":
        source = sys.stdin
    else:
        path = Path(args.logfile)
        if not path.exists():
            print(f"error: log file does not exist: {path}", file=sys.stderr)
            return 2
        source = path.open("r", encoding="utf-8", errors="replace")

    try:
        for line in source:
            if not _line_after_since(line, args.since):
                continue
            r = _parse_line(line)
            if r is not None:
                records.append(r)
    finally:
        if args.logfile != "-":
            source.close()

    stats = _aggregate(records)
    if args.json:
        _report_json(stats)
    else:
        _report_human(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
