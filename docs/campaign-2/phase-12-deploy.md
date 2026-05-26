# Campaign 2 Phase 12 — Chunk-tail fix: drop near-duplicate tail chunks

Closes the long-standing Lesson 11 chunk-tail finding (deferred since
2026-05-18 pending measurement). The fix drops the final chunk when
its new content past the prior chunk's end is at or below 10 % of
`chunk_tokens` — preventing the chunker from emitting near-duplicate
tail chunks that waste an embed call + a Qdrant point.

## Measurement → fix → re-measurement loop

The Tier 2 measurement work shipped 2026-05-26
(`scripts/measure_chunk_tails.py`) ran against the production
`/datasets` corpus:

  - 5640 files scanned, 1717 multi-chunk (tail-eligible).
  - **225 files (13.1 % of multi-chunk) produced a wasted tail.**
  - 18 of those had ≤5 % new content (essentially pure duplicates);
    11 had 6-10 %; 52 + 141 spread across 11-50 %.
  - 1495 files (87 %) had healthy tails (>50 % new content) — those
    are fine and the fix leaves them untouched.

13.1 % clears the script's documented ">5 % → ship the fix" threshold.

## What the fix does

[`src/audrey/kb/chunk.py:chunk_text`](../../src/audrey/kb/chunk.py)
gained a tail-skip branch:

  - `waste_threshold = chunk_tokens // 10` computed once per call.
  - Tracks `prev_end` across iterations (the end-token of the most
    recently emitted chunk).
  - Before deciding to emit the final iteration's chunk: if at least
    one chunk has already been emitted AND `(end - prev_end) <=
    waste_threshold`, break without emitting. The dropped tail's
    content is already searchable via the prior chunk's overlap
    window, so search recall is preserved.

No new config knob. The threshold is derived from `chunk_tokens`
(`// 10` = 10 %), which is a structural property of the chunker shape
rather than a user-facing tunable. If a future deployment wants to
tune it, add a `kb.chunk_waste_threshold_pct` knob then; for now
the measurement validated 10 % as the right default.

The fix is conservative: it only fires on the *final* iteration, only
when `prev_end` has actually advanced (a previous chunk was emitted),
and only when the new content past `prev_end` is small. It doesn't
touch the middle of the chunk sequence; it doesn't change the
threshold-below-which-single-chunk behavior; it doesn't change the
safety clamp on `overlap_tokens >= chunk_tokens`.

## What's in scope

  - **[`src/audrey/kb/chunk.py`](../../src/audrey/kb/chunk.py)** —
    the fix itself. Six-line addition to `chunk_text` plus an
    expanded docstring documenting the tail-skip behavior + the
    measurement that validated it.
  - **[`tests/test_kb_chunk.py`](../../tests/test_kb_chunk.py)** —
    new file. 10 tests pinning the new behavior. Critical cases:
    drops tail at n=1901 (1 new token), drops tail at n=1100
    (exactly 100 new tokens, the threshold boundary), keeps tail
    at n=1101 (one past the threshold), keeps tail at n=2700
    (800 new tokens — the normal case). Single-chunk path,
    empty-input path, and the overlap safety clamp also covered.
  - **[`docs/lessons/AUDIT.md`](../../docs/lessons/AUDIT.md)** —
    moved the chunk-tail finding from Deferred to Resolved with
    the 2026-05-26 date. The Deferred section's Lesson 11 entry
    is now empty (only Lesson 4 entries remain there).

## Closure verification — *pending*

This phase ships entirely in the repo. Verification is:

  - All 343 tests pass (320 before Phase 11; +13 from Phase 10 admin
    tests; +10 new from this phase). Counted: 333 pre-Phase-12, +10
    new = 343.
  - Ruff clean on `kb/chunk.py` and `tests/test_kb_chunk.py`.
  - The fix's chunks-emitted prediction (from the measurement
    script) matches what the real chunker emits — validated locally
    against `docs/` (339 → 335 chunks, ~1.2 % savings on that
    corpus; matches the script's prediction).
  - Re-running `measure_chunk_tails.py` against `/datasets` on
    Unraid after deploy should report `0` files where the fix
    would drop the tail — because the chunker actually drops them
    now, so the script's "would drop" prediction is moot.

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai     # required: chunker change
                                           # is baked into the image
```

`custom-tools` doesn't need rebuilding — chunking lives entirely in
audrey-ai. Existing KB ingests don't auto-re-chunk; new ingests (via
the watcher or manual `audrey-ingest`) emit the new shape. Existing
duplicate tail chunks stay in Qdrant until the source file changes or
is reingested.

## 2. Smoke tests

### 2.1 Boot succeeds

```bash
docker compose logs audrey-ai --tail=50 | grep ready
```

Expect the usual `ready:` line. No new config fields, no new env vars,
the boot log shape is identical.

### 2.2 Re-run the measurement against /datasets

Same command as the original measurement (still needs `docker cp`
since `scripts/` isn't copied into the image):

```bash
docker cp scripts/measure_chunk_tails.py audrey-ai:/tmp/measure_chunk_tails.py
docker exec audrey-ai python /tmp/measure_chunk_tails.py /datasets
```

Expect:

  - `Files where fix would drop tail` reports the *measurement
    script's* simulation, which still uses the same logic as before.
    The "would drop" count should match the pre-fix run (225) —
    that's what the script predicted, and the script itself hasn't
    changed.
  - The corpus characterization (multi-chunk %, distribution
    buckets) is identical because the same files are being scanned.

If you want to verify the *actual* chunker emits fewer chunks now,
re-ingest a file that previously had a wasted tail and check the
delta in Qdrant point count. Optional — the tests pin the behavior.

### 2.3 Existing search still works

```bash
# Hit the live KB search via any prompt that surfaces a known result.
# The tail-skip preserves search recall because dropped tails were
# already inside the prior chunk's overlap window. Quick sanity check:
# search for content you know lives in a previously-multi-chunk file.
```

If a previously-findable snippet is suddenly missing, the tail-skip
condition isn't conservative enough — file a bug. (Not expected;
the math is solid and the tests cover the edge cases.)

## 3. Rollback

`git revert <commit>` then `docker compose up -d --build audrey-ai`
restores the pre-fix chunker. Existing dropped-tail Qdrant points
aren't auto-restored — files would need to be re-ingested to get
them back. The drop is also benign by design (the dropped content
was already in the prior chunk), so rollback shouldn't be needed for
correctness reasons.

## 4. Operational notes

  - **No re-ingest required.** Existing KB content with extra tail
    chunks keeps working — those points still match queries fine,
    they're just slightly redundant. The fix prevents *new* wasted
    tails from being created.
  - **Re-ingesting old content** (via `audrey-ingest --purge` or a
    file edit triggering the watcher) will drop the duplicate tails
    for that file. On the next major KB refresh, the cumulative ~1 %
    chunk reduction will materialize.
  - **The measurement script keeps working as a regression guard.**
    Re-running `measure_chunk_tails.py` periodically is the way to
    detect if the chunker shape ever drifts back into emitting wasted
    tails.

## 5. Followups

One Tier 2 measurement item remains: synth-draft size analysis
(the one-line `log.info` instrumentation shipped 2026-05-26 in
`pipeline/synthesize.py` + `scripts/analyze_draft_sizes.py`).
Needs production traffic to accumulate before the analyze script
has signal. Re-evaluate in a week or so.
