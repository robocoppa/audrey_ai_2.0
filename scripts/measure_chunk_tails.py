#!/usr/bin/env python3
"""Measure how much new content the tail chunk actually adds in real KB files.

WHY

`chunk_text` (`src/audrey/kb/chunk.py`) iterates
`range(0, len(tokens), stride)` and emits one chunk per iteration. When
the final stride lands close to the end of the document, the tail chunk
overlaps heavily with the prior chunk — most of its tokens were already
covered by the prior chunk's overlap window.

ORIGINAL AUDIT vs. CORRECTED ANALYSIS

The Lesson 11 audit finding originally proposed: "skip the final chunk
when `end - start <= overlap_tokens` and there was a previous chunk
emitted." With the production defaults (`chunk_tokens=1000`,
`overlap_tokens=100`, `stride=900`) **that condition is structurally
unreachable** — when `n - k*900 <= 100` (the tail is small enough to
match the proposed fix), the previous iteration's `end` already
reached `n`, so the tail iteration never fires. The originally-proposed
fix would be a no-op.

The real bug is still there — when `n` falls just past a stride
boundary, the tail chunk contributes very few *new* tokens beyond what
the prior chunk already covered. Example with the defaults:

  - n=1901 → tail chunk covers tokens [1800, 1901]; prior chunk covered
    [900, 1900]; tail's `new_tokens = 1`. Wasted embed + Qdrant point on
    99 % redundant content.

WHAT THIS SCRIPT DOES

Walks a root directory, loads each text file using Audrey's own
`load_text`, tokenizes with the same `cl100k_base` encoder, simulates
the chunker with production defaults, and characterizes the *tail
chunk's new content* across the corpus:

  - per-file: `tail_new_tokens` (count of tokens in the tail chunk
    that weren't already in the prior chunk) and
    `tail_new_pct` (those new tokens as a fraction of the tail
    chunk's total size).
  - distribution across the corpus, bucketed by `new_pct`:
    <=5 % (near-pure duplicate), 6-10 %, 11-25 %, 26-50 %, 51-100 %.
  - aggregate: how many files would the *corrected* fix
    (`tail_new_tokens <= waste_threshold_pct/100 * chunk_tokens`)
    drop, and what fraction of total chunks that is.

USAGE

  # Full sweep on Unraid against the production KB:
  python3 scripts/measure_chunk_tails.py /mnt/user/knowledge

  # Quick sample-based sanity check (10 % of files, deterministic):
  python3 scripts/measure_chunk_tails.py /mnt/user/knowledge \\
      --sample-fraction 0.1 --seed 42

  # Try a stricter or looser waste threshold (default 10):
  python3 scripts/measure_chunk_tails.py /mnt/user/knowledge \\
      --waste-threshold-pct 5

  # Non-default chunk sizes (must match config.yaml's kb.chunk_tokens /
  # kb.chunk_overlap if you've changed them from the 1000/100 default):
  python3 scripts/measure_chunk_tails.py /mnt/user/knowledge \\
      --chunk-tokens 1000 --overlap-tokens 100

  # JSON output for machine-readable downstream tooling:
  python3 scripts/measure_chunk_tails.py /mnt/user/knowledge --json

DECISION CRITERIA

Look at the new-content distribution and the "wasted tails the fix
would drop" count under the default threshold:

  - <1 % of multi-chunk files affected → accept the finding. The
    chunk-tail bug exists but is too rare to justify code change.
  - 1-5 % affected → judgment call. If the wasted chunks cluster on
    large files where every chunk costs more, fix is worth it.
  - >5 % affected → ship the fix with the validated threshold.

The script is OFFLINE — no Ollama, no Qdrant, no network. Safe to run
on any host with the audrey venv available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Audrey's own loader + tokenizer — same code path as ingest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from audrey.kb.chunk import (
    IMAGE_SUFFIXES,
    TEXT_SUFFIXES,
    _encoder,
    load_text,
)

# Doc suffixes that go through extraction. Matches the union the watcher
# allows through (`kb/watcher.py:_DOC_SUFFIXES`).
_DOC_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".html", ".htm"}


# New-content buckets for the distribution report. Each entry is
# `(label, lower_pct_inclusive, upper_pct_inclusive)`. A file with tail
# `new_pct = 0.04` (4 %) falls into the first bucket; `new_pct = 0.06`
# falls into the second.
_NEW_PCT_BUCKETS = [
    ("<=5%",     0.0,  0.05),
    ("6-10%",    0.05, 0.10),
    ("11-25%",   0.10, 0.25),
    ("26-50%",   0.25, 0.50),
    ("51-100%",  0.50, 1.01),  # upper bound 1.01 so 1.00 lands here
]


@dataclass(slots=True)
class FileResult:
    path: Path
    tokens: int = 0
    chunks_current: int = 0
    chunks_fixed: int = 0
    tail_new_tokens: int = 0         # tokens in the tail chunk not in the prior chunk
    tail_new_pct: float = 0.0        # those new tokens as a fraction of the tail's size
    tail_dropped_by_fix: bool = False
    error: str = ""                  # non-empty if extraction or tokenization failed


@dataclass(slots=True)
class Totals:
    files_scanned: int = 0
    files_with_no_text: int = 0
    files_with_extraction_error: int = 0
    files_below_threshold: int = 0   # single-chunk path, no tail issue possible
    files_multi_chunk: int = 0
    files_tail_dropped: int = 0      # files where the corrected fix would drop the tail
    total_chunks_current: int = 0
    total_chunks_fixed: int = 0
    new_pct_distribution: Counter[str] = field(default_factory=Counter)


def _simulate_chunker(n_tokens: int, *, chunk_tokens: int, overlap_tokens: int) -> int:
    """Mirror `chunk_text`'s loop, counting chunks without building strings.

    Token-count-only — much faster than calling the real chunker (which
    decodes/strips each chunk). The loop shape must stay in sync with
    `src/audrey/kb/chunk.py:chunk_text`.
    """
    if n_tokens == 0:
        return 0
    if n_tokens <= chunk_tokens:
        return 1
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = chunk_tokens // 5
    stride = chunk_tokens - overlap_tokens
    count = 0
    for start in range(0, n_tokens, stride):
        end = min(start + chunk_tokens, n_tokens)
        # `chunk_text` skips empty pieces; we can't tell from token
        # counts alone, but in practice tokenized text never produces
        # whitespace-only pieces from arbitrary stride positions.
        count += 1
        if end >= n_tokens:
            break
    return count


def _last_chunk_geometry(
    n_tokens: int, *, chunk_tokens: int, overlap_tokens: int,
) -> tuple[int, int, int] | None:
    """Return `(start, end, prev_end)` for the last chunk, or None if single-chunk.

    `start` and `end` are the last chunk's bounds.
    `prev_end` is the end of the chunk *before* the last one. The wasted
    region is `[start : prev_end]`; the new content the last chunk
    contributes is `[prev_end : end]`.
    """
    if n_tokens <= chunk_tokens:
        return None
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = chunk_tokens // 5
    stride = chunk_tokens - overlap_tokens

    last_start = 0
    last_end = 0
    prev_end = 0
    for start in range(0, n_tokens, stride):
        end = min(start + chunk_tokens, n_tokens)
        prev_end = last_end if last_end > 0 else end
        last_start = start
        last_end = end
        if end >= n_tokens:
            break
    return (last_start, last_end, prev_end)


def _classify_tail(
    n_tokens: int, *, chunk_tokens: int, overlap_tokens: int,
    waste_threshold_pct: float,
) -> tuple[int, float, bool]:
    """Return `(new_tokens, new_pct, dropped_by_fix)` for the last chunk.

    `new_tokens` is the count of tokens in the tail chunk that the prior
    chunk's content didn't already cover (`end - prev_end`). Zero would
    mean the tail is a pure duplicate; tail chunks in practice always
    have at least 1 new token (otherwise the loop would have terminated
    on the previous iteration).

    `new_pct` is `new_tokens / chunk_size` — the fraction of the tail
    that's actually new content. A `new_pct` of 0.05 means 95 % of the
    tail's tokens were already in the prior chunk.

    `dropped_by_fix` is True when the corrected fix condition fires:
    the tail's new content is at or below `waste_threshold_pct` of the
    chunk-tokens budget. The fix would skip emitting the tail.
    """
    geom = _last_chunk_geometry(
        n_tokens, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
    )
    if geom is None:
        return (0, 0.0, False)
    last_start, last_end, prev_end = geom
    chunk_size = last_end - last_start
    new_tokens = max(0, last_end - prev_end)
    new_pct = new_tokens / chunk_size if chunk_size > 0 else 0.0
    threshold_tokens = int(chunk_tokens * waste_threshold_pct / 100.0)
    dropped_by_fix = new_tokens <= threshold_tokens
    return (new_tokens, new_pct, dropped_by_fix)


def _bucket_label(new_pct: float) -> str:
    """Return the `_NEW_PCT_BUCKETS` label that `new_pct` falls into."""
    for label, lo, hi in _NEW_PCT_BUCKETS:
        if lo <= new_pct < hi:
            return label
    return _NEW_PCT_BUCKETS[-1][0]  # defensive: pin to last bucket


def _iter_supported_files(root: Path) -> list[Path]:
    """Walk `root` for files in `_DOC_SUFFIXES`, skipping dot-segments.

    Mirrors `kb/ingest.py:_iter_files`'s dot-segment skip rule so this
    measurement matches what ingest actually sees.
    """
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip anything under a dot-prefixed directory (.git/, .cache/, etc.).
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        suffix = path.suffix.lower()
        if suffix in _DOC_SUFFIXES:
            out.append(path)
        elif suffix in IMAGE_SUFFIXES:
            continue  # images skip the chunker entirely
    return out


def _process_file(
    path: Path, *, chunk_tokens: int, overlap_tokens: int,
    waste_threshold_pct: float,
) -> FileResult:
    result = FileResult(path=path)
    try:
        text = load_text(path)
    except Exception as e:  # noqa: BLE001 — defensive; load_text already swallows most
        result.error = f"load: {e}"
        return result
    if text is None:
        result.error = "no_loader"
        return result
    cleaned = text.strip()
    if not cleaned:
        result.error = "empty"
        return result
    try:
        n_tokens = len(_encoder().encode(cleaned))
    except Exception as e:  # noqa: BLE001
        result.error = f"tokenize: {e}"
        return result
    result.tokens = n_tokens
    result.chunks_current = _simulate_chunker(
        n_tokens, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
    )
    new_tokens, new_pct, dropped = _classify_tail(
        n_tokens, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        waste_threshold_pct=waste_threshold_pct,
    )
    result.tail_new_tokens = new_tokens
    result.tail_new_pct = new_pct
    result.tail_dropped_by_fix = dropped
    # The corrected fix drops the tail iff `dropped` AND there was a
    # previous chunk. `_simulate_chunker` returning >=2 satisfies that.
    result.chunks_fixed = (
        result.chunks_current - 1 if dropped and result.chunks_current >= 2
        else result.chunks_current
    )
    return result


def _aggregate(results: list[FileResult]) -> Totals:
    totals = Totals()
    for r in results:
        totals.files_scanned += 1
        if r.error:
            if r.error in ("empty", "no_loader"):
                totals.files_with_no_text += 1
            else:
                totals.files_with_extraction_error += 1
            continue
        totals.total_chunks_current += r.chunks_current
        totals.total_chunks_fixed += r.chunks_fixed
        if r.chunks_current <= 1:
            totals.files_below_threshold += 1
            continue
        totals.files_multi_chunk += 1
        if r.tail_dropped_by_fix:
            totals.files_tail_dropped += 1
        # Bucket every multi-chunk file's tail into the new-content
        # distribution, not just the ones the fix would drop. That way
        # you can see the full shape of "how much is the tail actually
        # adding?" across the corpus.
        totals.new_pct_distribution[_bucket_label(r.tail_new_pct)] += 1
    return totals


def _report_human(
    totals: Totals, *,
    chunk_tokens: int, overlap_tokens: int, waste_threshold_pct: float,
) -> None:
    pct = lambda num, denom: (100.0 * num / denom) if denom else 0.0  # noqa: E731
    threshold_tokens = int(chunk_tokens * waste_threshold_pct / 100.0)
    print(
        f"\nChunk-tail measurement - chunk_tokens={chunk_tokens}, "
        f"overlap_tokens={overlap_tokens}, "
        f"waste_threshold={waste_threshold_pct}% ({threshold_tokens} tokens)",
    )
    print("-" * 72)
    print(f"  Files scanned                     : {totals.files_scanned}")
    print(f"    skipped (no text / no loader)   : {totals.files_with_no_text}")
    print(f"    skipped (extraction error)      : {totals.files_with_extraction_error}")
    print(f"    single-chunk (no tail issue)    : {totals.files_below_threshold}")
    print(f"    multi-chunk (tail-eligible)     : {totals.files_multi_chunk}")
    print(f"  Files where fix would drop tail   : {totals.files_tail_dropped}")
    print(
        f"    as % of all files scanned       : {pct(totals.files_tail_dropped, totals.files_scanned):.2f} %",
    )
    print(
        f"    as % of multi-chunk files       : {pct(totals.files_tail_dropped, totals.files_multi_chunk):.2f} %",
    )
    print()
    print(f"  Total chunks (current chunker)    : {totals.total_chunks_current}")
    print(f"  Total chunks (fixed chunker)      : {totals.total_chunks_fixed}")
    saved = totals.total_chunks_current - totals.total_chunks_fixed
    print(
        f"  Wasted chunks the fix would drop  : {saved} "
        f"({pct(saved, totals.total_chunks_current):.2f} % of total)",
    )
    if totals.new_pct_distribution:
        print()
        print("  Tail-chunk new-content distribution (multi-chunk files):")
        print("  (each bucket = what fraction of the tail is NEW vs. duplicate)")
        max_count = max(totals.new_pct_distribution.values()) or 1
        for label, _lo, _hi in _NEW_PCT_BUCKETS:
            count = totals.new_pct_distribution.get(label, 0)
            bar = "#" * min(40, int(40 * count / max_count))
            print(f"    new={label:>9} : {count:>5}  {bar}")
    print()


def _report_json(
    totals: Totals, *,
    chunk_tokens: int, overlap_tokens: int, waste_threshold_pct: float,
) -> None:
    out = {
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "waste_threshold_pct": waste_threshold_pct,
        "files_scanned": totals.files_scanned,
        "files_with_no_text": totals.files_with_no_text,
        "files_with_extraction_error": totals.files_with_extraction_error,
        "files_below_threshold": totals.files_below_threshold,
        "files_multi_chunk": totals.files_multi_chunk,
        "files_tail_dropped": totals.files_tail_dropped,
        "total_chunks_current": totals.total_chunks_current,
        "total_chunks_fixed": totals.total_chunks_fixed,
        "chunks_dropped_by_fix": totals.total_chunks_current - totals.total_chunks_fixed,
        "new_pct_distribution": dict(totals.new_pct_distribution),
    }
    print(json.dumps(out, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measure wasted overlap-only tail chunks in a KB root.",
    )
    p.add_argument("root", type=Path, help="Path to the KB root (e.g. /mnt/user/knowledge)")
    p.add_argument(
        "--chunk-tokens", type=int, default=1000,
        help="Match config.yaml's kb.chunk_tokens (default 1000)",
    )
    p.add_argument(
        "--overlap-tokens", type=int, default=100,
        help="Match config.yaml's kb.chunk_overlap (default 100)",
    )
    p.add_argument(
        "--waste-threshold-pct", type=float, default=10.0,
        help=(
            "Tail is considered 'wasted' when its new tokens are at or "
            "below this %% of chunk_tokens. Default 10 (i.e. <=100 new "
            "tokens out of 1000) — a tail that adds <10%% new content "
            "is mostly a duplicate."
        ),
    )
    p.add_argument(
        "--sample-fraction", type=float, default=1.0,
        help="Sample a fraction of files for a quicker estimate (default 1.0 = all)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Deterministic seed for sampling (default 0)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the human table",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print one line per file as it's processed (for long sweeps)",
    )
    args = p.parse_args()

    if not args.root.exists():
        print(f"error: root does not exist: {args.root}", file=sys.stderr)
        return 2
    if not args.root.is_dir():
        print(f"error: root is not a directory: {args.root}", file=sys.stderr)
        return 2

    files = _iter_supported_files(args.root)
    if args.sample_fraction < 1.0:
        # Measurement sampling — reproducibility matters, not crypto strength.
        rng = random.Random(args.seed)  # noqa: S311 — not a cryptographic context
        n_keep = max(1, int(len(files) * args.sample_fraction))
        files = rng.sample(files, n_keep)
        if not args.json:
            print(
                f"sampling {n_keep} of {len(files) / args.sample_fraction:.0f} files "
                f"(fraction={args.sample_fraction}, seed={args.seed})",
                file=sys.stderr,
            )

    if not files:
        print(f"warning: no supported files found under {args.root}", file=sys.stderr)
        return 1

    results: list[FileResult] = []
    for i, path in enumerate(files, 1):
        if args.verbose:
            print(f"  [{i}/{len(files)}] {path}", file=sys.stderr)
        results.append(_process_file(
            path,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            waste_threshold_pct=args.waste_threshold_pct,
        ))

    totals = _aggregate(results)
    if args.json:
        _report_json(
            totals,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            waste_threshold_pct=args.waste_threshold_pct,
        )
    else:
        _report_human(
            totals,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            waste_threshold_pct=args.waste_threshold_pct,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
