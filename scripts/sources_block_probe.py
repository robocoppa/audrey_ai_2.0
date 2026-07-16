#!/usr/bin/env python3
"""sources_block_probe.py — replay research ledgers through `_render_sources_block`.

Why this exists: the Sources list under a research answer is rendered
deterministically from the merged claim/source ledger (`deep_panel.
_render_sources_block`), NOT asked of the writer. A subtle bug made it emit
`sources:0` on an answer that DID have citable URLs: a claim linked the one
URL-less "Search result" source, so the surviving-source set was non-empty and
the render-all fallback didn't fire — even though the ledger also held real
arXiv/Wikipedia URLs that simply weren't linked to a surviving claim
(tech-transformer-attention, 2026-07-15 trace run). The fix falls back to all
consulted sources when NO kept source has a usable URL.

This harness feeds real (or captured) ledger dicts through the SAME render
function the pipeline uses and prints what each one renders — so a fix or a
regression is visible on real data, not just one unit test. Ledgers come from
the `### Ledger` trace blocks / `ledger` dicts on the pipeline's `done` event
(the eval results JSON does NOT carry them, so pass a captured dump).

USAGE (laptop, hermetic — no box, no network):
    # built-in regression fixtures (incl. the attention bug shape):
    python3 scripts/sources_block_probe.py
    # replay a captured ledger dump (a ResearchResult .model_dump() as JSON,
    # or {"ledger": {...}, "factcheck": {...}} — e.g. saved off a done event):
    python3 scripts/sources_block_probe.py --ledger /path/to/ledger.json
    python3 scripts/sources_block_probe.py --ledger dump.json --expect-sources
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the REAL render path + models so the harness can't drift from prod.
from audrey.pipeline.deep_panel import _render_sources_block
from audrey.pipeline.ledger import FactCheckResult, ResearchResult

# ── Built-in regression fixtures ──────────────────────────────────────────
# Each: (name, ledger_dict, factcheck_dict|None, expect_sources_bool). The
# ledger dicts mirror shapes seen in real trace runs; `expect_sources` is what a
# correct renderer SHOULD do (non-empty vs empty Sources block).
_FIXTURES: list[tuple[str, dict, dict | None, bool]] = [
    (
        # THE regression: a claim links a URL-less "Search result" source (so
        # `keep` is non-empty), while real arXiv/Wikipedia URLs sit UNLINKED.
        # Pre-fix: sources:0. Post-fix: renders the usable URLs.
        "attention-urlless-linked-blocks-real-urls",
        {
            "claims": [
                {"id": "c1", "text": "Attention exists", "source_ids": ["s_search"]},
            ],
            "sources": [
                {"id": "s_search", "title": "Search result", "url": "",
                 "source_type": "reference"},
                {"id": "s1", "title": "Attention Is All You Need",
                 "url": "https://arxiv.org/abs/1706.03762", "source_type": "primary_paper"},
                {"id": "s2", "title": "Transformer (deep learning architecture)",
                 "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
                 "source_type": "reference"},
            ],
        },
        None,
        True,  # should render the arXiv + Wikipedia URLs
    ),
    (
        # Normal grounded case: claims link real usable-URL sources → render them.
        "grounded-linked-usable-urls",
        {
            "claims": [{"id": "c1", "text": "x", "source_ids": ["s1"]}],
            "sources": [{"id": "s1", "title": "MacTutor",
                         "url": "https://mathshistory.st-andrews.ac.uk/", "source_type": "reference"}],
        },
        None,
        True,
    ),
    (
        # Genuinely ungrounded: sources exist but NONE have a usable URL →
        # correctly render nothing (must NOT sprout an empty Sources header).
        "no-usable-url-renders-nothing",
        {
            "claims": [{"id": "c1", "text": "x", "source_ids": ["s1"]}],
            "sources": [{"id": "s1", "title": "Prior knowledge", "url": "",
                         "source_type": "unknown"}],
        },
        None,
        False,
    ),
    (
        # Fact-check drops ONE of two claims: the dropped claim's source is
        # filtered out, the surviving claim's source is still rendered. `keep`
        # stays non-empty (s2 survives) so the render-all fallback does NOT fire
        # — this is the drop-unsupported path, distinct from the empty-keep
        # fallback. Correct: renders only s2. (A run where the fact-check drops
        # the ONLY claim leaves keep empty → fallback → renders all consulted;
        # that's by design, so it is NOT the assertion here.)
        "unsupported-claim-dropped-among-two",
        {
            "claims": [
                {"id": "c1", "text": "dropped", "source_ids": ["s1"]},
                {"id": "c2", "text": "kept", "source_ids": ["s2"]},
            ],
            "sources": [
                {"id": "s1", "title": "Dropped", "url": "https://dropped.com",
                 "source_type": "news"},
                {"id": "s2", "title": "Kept", "url": "https://kept.com",
                 "source_type": "news"},
            ],
        },
        {"checks": [{"claim_id": "c1", "verdict": "unsupported"},
                    {"claim_id": "c2", "verdict": "supported"}]},
        True,  # renders s2 (Kept); s1 (Dropped) filtered out
    ),
]


def _render(ledger_d: dict, fc_d: dict | None) -> str:
    ledger = ResearchResult.model_validate(ledger_d) if ledger_d else None
    fc = FactCheckResult.model_validate(fc_d) if fc_d else None
    return _render_sources_block(ledger, fc)


def _count_urls(block: str) -> int:
    return block.count("\n- [")


def _run_fixture(name: str, ledger_d: dict, fc_d: dict | None, expect: bool) -> bool:
    block = _render(ledger_d, fc_d)
    got = bool(block.strip())
    ok = got == expect
    n = _count_urls(block)
    status = "PASS" if ok else "FAIL"
    exp = "sources" if expect else "empty"
    print(f"  [{status}] {name}")
    print(f"         expected: {exp:<7}  got: {'sources' if got else 'empty':<7}  urls: {n}")
    if not ok:
        print(f"         rendered: {block!r}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=Path, default=None,
                   help="a captured ledger dump (ResearchResult .model_dump() JSON, "
                        "or {\"ledger\":{...},\"factcheck\":{...}})")
    p.add_argument("--expect-sources", action="store_true",
                   help="with --ledger: assert the block is non-empty (exit 1 if not)")
    args = p.parse_args()

    if args.ledger:
        raw = json.loads(args.ledger.read_text())
        # Accept either a bare ResearchResult dump or a done-event-shaped wrapper.
        ledger_d = raw.get("ledger", raw) if isinstance(raw, dict) else raw
        fc_d = raw.get("factcheck") if isinstance(raw, dict) else None
        block = _render(ledger_d, fc_d)
        n = _count_urls(block)
        print(f">> {args.ledger.name}: rendered {n} source URL(s), "
              f"block {'non-empty' if block.strip() else 'EMPTY'}")
        if block.strip():
            print(block.rstrip())
        if args.expect_sources and not block.strip():
            print("EXPECTED sources but got an empty block", file=sys.stderr)
            return 1
        return 0

    print(">> replaying built-in Sources-render fixtures\n")
    results = [_run_fixture(*f) for f in _FIXTURES]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} fixtures passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
