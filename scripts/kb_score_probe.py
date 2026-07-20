#!/usr/bin/env python3
"""kb_score_probe.py — collect KB hit-score distributions to pick `kb.min_score`.

Why this exists: `kb_search` always returns its top_k NEAREST vectors regardless
of distance, so an off-domain query (this KB holds geology/botany/bushcraft/
fishing) gets the least-irrelevant junk, which pollutes a researcher's context
(2026-07-15 trace: a vaccine query returned PowerApps / ServiceNow docs). A
cosine floor (`kb.min_score`, applied in routes/kb.py `_search_text_merged`)
fixes it — but the RIGHT floor is corpus-dependent and can't be guessed. This
harness probes many labeled queries and reports the score distributions so the
floor is set from data: the max off-domain score and the min on-domain score
bound the safe window, and a candidate floor's misclassifications are counted.

It hits Audrey's `/v1/kb/query` directly (NOT OWUI): that endpoint takes no auth
and returns per-hit scores. Run it ON THE BOX (ollama-net DNS resolves there).
As of the 2026-07-18 security review, Audrey's `:8000` is NOT published to the
host, so the old `http://192.168.1.11:8000` over-the-LAN form no longer works —
run from a container on ollama-net, or tunnel `:8000` from the laptop first
(`ssh -N -L 8000:audrey-ai:8000 <unraid>` then KB_PROBE_BASE_URL=http://localhost:8000).

USAGE (on the box, on ollama-net):
    # default: http://audrey-ai:8000 (internal name) or set --base-url
    python3 scripts/kb_score_probe.py
    KB_PROBE_BASE_URL=http://localhost:8000 python3 scripts/kb_score_probe.py  # via tunnel
    python3 scripts/kb_score_probe.py --queries scripts/kb_probe_queries.json --top-k 5
    python3 scripts/kb_score_probe.py --save-json /out/kb-scores.json   # machine-readable

It does NOT change anything — read-only queries. Pair the JSON output with
config.yaml's `kb.min_score` comment when re-tuning after a corpus change.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - runtime guard, mirrors eval_research
    print("httpx not installed. Run: uv sync --extra dev", file=sys.stderr)
    raise SystemExit(2) from None

DEFAULT_QUERIES = Path(__file__).with_name("kb_probe_queries.json")
DEFAULT_BASE_URL = os.environ.get("KB_PROBE_BASE_URL", "http://audrey-ai:8000")


def _top_score(base_url: str, query: str, top_k: int, timeout_s: float) -> tuple[float | None, str]:
    """Return (top hit score, top source) for one query, or (None, err) on failure.

    None score means the query returned zero hits OR the call failed — the second
    element carries the reason so an empty KB and a transport error don't look the
    same in the report.
    """
    url = base_url.rstrip("/") + "/v1/kb/query"
    body = {"query": query, "top_k": top_k}
    try:
        r = httpx.post(url, json=body, timeout=timeout_s)
    except httpx.HTTPError as e:
        return None, f"error: {type(e).__name__}: {str(e)[:120]}"
    if r.status_code >= 300:
        return None, f"http {r.status_code}: {r.text[:120]}"
    results = (r.json() or {}).get("results") or []
    if not results:
        return None, "no hits"
    top = results[0]
    return float(top.get("score", 0.0)), str(top.get("source", ""))


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "  —   "


def _summarize(label: str, rows: list[tuple[str, float | None, str]]) -> dict:
    scores = [s for _, s, _ in rows if s is not None]
    summary = {
        "label": label,
        "n": len(rows),
        "n_with_hits": len(scores),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean": round(statistics.fmean(scores), 4) if scores else None,
        "median": round(statistics.median(scores), 4) if scores else None,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="Audrey base URL (or env KB_PROBE_BASE_URL). "
                        f"Default {DEFAULT_BASE_URL}")
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                   help="labeled probe-query JSON (on_domain / off_domain lists)")
    p.add_argument("--top-k", type=int, default=5, help="hits per query (top score is used)")
    p.add_argument("--timeout", type=float, default=30.0, help="per-query timeout (s)")
    p.add_argument("--save-json", type=Path, default=None,
                   help="write per-query scores + summary as JSON")
    args = p.parse_args()

    spec = json.loads(args.queries.read_text())
    labels = [k for k in ("on_domain", "off_domain") if isinstance(spec.get(k), list)]
    if not labels:
        print("no on_domain/off_domain query lists in the probe file", file=sys.stderr)
        return 2

    per_label: dict[str, list[tuple[str, float | None, str]]] = {}
    print(f">> probing {args.base_url}/v1/kb/query (top_k={args.top_k})\n")
    for label in labels:
        rows: list[tuple[str, float | None, str]] = []
        print(f"── {label} ──")
        for q in spec[label]:
            score, info = _top_score(args.base_url, q, args.top_k, args.timeout)
            rows.append((q, score, info))
            src = info if score is None else info.rsplit("/", 1)[-1]
            print(f"  {_fmt(score)}  {q[:52]:<52}  {src[:40]}")
        per_label[label] = rows
        print()

    summaries = {label: _summarize(label, rows) for label, rows in per_label.items()}

    # The core output: the safe-floor window. A floor must sit ABOVE every
    # off-domain top-score (else junk survives) and BELOW every on-domain
    # top-score (else real hits are cut). If off_max < on_min there's a clean
    # valley; otherwise the distributions overlap and no floor separates them
    # perfectly — report that honestly rather than pick a lossy cut.
    on = summaries.get("on_domain", {})
    off = summaries.get("off_domain", {})
    print("── floor analysis ──")
    for label in labels:
        s = summaries[label]
        print(f"  {label:<11} n={s['n']:<2} hits={s['n_with_hits']:<2} "
              f"min={_fmt(s['min'])} median={_fmt(s['median'])} max={_fmt(s['max'])}")
    on_min, off_max = on.get("min"), off.get("max")
    if on_min is not None and off_max is not None:
        if off_max < on_min:
            mid = round((off_max + on_min) / 2, 4)
            print(f"\n  CLEAN VALLEY: off_max={off_max:.4f} < on_min={on_min:.4f}")
            print(f"  → safe floor window: ({off_max:.4f}, {on_min:.4f}); midpoint {mid}")
            print(f"  → recommend kb.min_score ≈ {mid} "
                  f"(bias LOWER within the window to protect weak real hits)")
        else:
            print(f"\n  OVERLAP: off_max={off_max:.4f} ≥ on_min={on_min:.4f} — no clean cut.")
            print("  A floor here trades junk-kept against real-hits-cut. Inspect the "
                  "overlapping queries; the corpus may just lack that domain, in which "
                  "case loading relevant data beats tuning the floor.")
    else:
        print("\n  insufficient hits to compute a window (a label had no hits).")

    if args.save_json:
        out = {
            "base_url": args.base_url,
            "top_k": args.top_k,
            "summaries": summaries,
            "per_query": {
                label: [{"query": q, "top_score": s, "info": i} for q, s, i in rows]
                for label, rows in per_label.items()
            },
        }
        args.save_json.write_text(json.dumps(out, indent=2))
        print(f"\n>> wrote {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
