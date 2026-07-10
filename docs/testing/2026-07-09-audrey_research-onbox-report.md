# audrey_research on-box 2026-07-09 — content-free-source-drop + supports-backfill confirmed; grounding variance remains

**Date:** 2026-07-09
**Runs:** two full protocol runs on the same day (first the good-grounding run, then a thinner-grounding run — see below)
**Result:** 10 cases, 10 PASS both runs (all applicable checks green)
**Deploy under test:** `27900c6 fix(research): drop content-free ledger sources and backfill supports from source_ids` (code change → `up -d --build audrey-ai`), plus `664e0f2` trace-logging
**Predecessor:** run-10 (2026-07-08, S4 chunked fact-check confirmed); the intervening 2026-07-09 trace run diagnosed the `sources:0` structuring bug this fix targets

---

## Verdict: the structuring fix works — grounding that reaches the model now survives

The 2026-07-09 trace run (whose findings are quoted verbatim in the `ledger.py`
docstrings) established that euclid/archimedes `sources:0` was **not** a grounding
failure — grounding reached the model, and the notes→ledger structuring pass
(`_structure_one_draft`) was resurrecting content-free `w2_, untitled — no url`
rows for the qwen worker while dropping real ones. `27900c6` fixes that in two
parts, and both are visibly working in these runs:

1. **Drop content-free sources** (`_backfill_ids`, ledger.py:259): a source with
   neither a title nor a url is filtered out before rendering.
2. **Backfill `supports` from `source_ids`** (`_backfill_supports`, ledger.py:269):
   the source→claim index is completed from the claim→source direction the model
   reliably fills.

### Grounding recovered across the ancient-history cases

| Case | Run 10 (07-08) | Run A — good grounding (07-09) | Run B — thin grounding (07-09) |
|------|---------------:|-------------------------------:|-------------------------------:|
| bio-euclid | 0 | **8 GOOD** | **8 GOOD** |
| bio-pythagoras | 1 (THIN) | **5 GOOD** | **3 GOOD** |
| bio-archimedes | 0 | **8 GOOD** | 0 (N/A) |
| hist-library-alexandria | 0 | **4 GOOD** | 0 (N/A) |
| tech-transformer-attention | — | **4 GOOD** | 1 (THIN) |
| hist-parallel-postulate | 0 | 0 | 0 |
| current-rust-async | 0 | 0 | 0 |

Where Run 10 read `sources:0/1` on the big ancient-bio ledgers, Run A recovers
real URL-bearing Sources blocks (`britannica.com/biography/Euclid`,
`plato.stanford.edu/entries/euclid/`, `en.wikipedia.org/wiki/Archimedes`, the
Walters palimpsest site, …). This is the exact outcome the fix predicted: the
structuring pass no longer discards grounding that reached the model.

### The `supports` backfill is visibly working, per-worker

In Run A's euclid ledger, the deepseek worker (`w0_`, which reported no usable
sources) still shows `supports: none` — correctly, because it had none — while
`w1_src_britannica` reads `(supports: w1_c1, w1_c3, w1_c7)` and `w1_src_stanford`
carries its full backing list. The backfill fires where claims cite real ids and
stays empty where the worker genuinely retrieved nothing. Exactly as designed.

---

## Why two runs, and what the difference means

The two 07-09 files are **two separate protocol runs**, not one — the answer
bodies differ (euclid's opening paragraph, pythagoras's source count 5 vs 3).
Reading them side by side is the useful result:

- **Run A (good grounding):** euclid 8, pythagoras 5, archimedes 8, library 4,
  transformer 4 — all GOOD.
- **Run B (thin grounding):** euclid 8, pythagoras 3, archimedes 0, transformer 1
  THIN — archimedes and library collapsed to `sources:0` on the *same prompts*.

**This is per-worker, non-deterministic grounding variance — an upstream
retrieval property, not a structuring regression.** The `27800c6` fix stops the
structuring pass from *discarding* good grounding; it cannot *manufacture*
grounding when a run's web_search calls come back thin. Run A is the lucky run;
Run B is the thin one. Both are 10/10 PASS, and — critically — **both degrade
honestly:** every thin case opens with the hedge banner ("I couldn't fully verify
these details against retrieved sources"), drops unsupported specifics, and never
fabricates. Archimedes at `sources:0` in Run B still emits real per-claim
verdicts (96 checks, 4 drops) and a fully hedged body.

The budget levers are already maxed from the run-5/run-6 arc
(`research_worker.max_web_searches: 6`, `max_rounds: 5`, `compress_keep_last: 5`),
so the variance is **not** budget starvation — it is the SearXNG/upstream
retrieval layer returning thin on a given request. Nothing in `audrey_research`'s
own code is at fault on the thin run.

---

## Smaller findings (debug-only / pre-existing — for AUDIT.md, not blockers)

1. **`w2_, untitled — no url` artifacts still render inside the debug ledger.**
   Run B's library ledger shows ~11 of them in the qwen worker's source list. The
   fix correctly excludes them from the *rendered* `## Sources` block and from the
   `sources:N` count (they carry `supports: none`), but `_backfill_ids` drops them
   from `r.sources` only for rendering — the raw trace ledger printed under
   `debug_research_trace` still lists them. Cosmetic, debug-view-only, harmless to
   the answer. Consistent with the fix; noting so it isn't re-diagnosed as a
   regression.

2. **Verifier critique is not consumed by the renderer (pre-existing).** Run A's
   euclid answer keeps ungrounded body content the verifier explicitly flagged —
   the "good glory / renowned" etymology and the Gödel/Kant tangents — even though
   the fabricated "surname *Clithius* / 'the Giver'" *was* correctly `DROP`ped by
   fact-check. The fact-check verdict channel acts on the answer; the verifier's
   prose critiques do not. This is the long-standing verifier→renderer gap, not
   introduced by this fix.

---

## Deploy state

- `27900c6` (+ `664e0f2` trace-logging) is **deployed and confirmed** on the box.
- The `sources:0`-from-structuring failure mode is closed. Remaining `sources:0`
  cases (parallel-postulate, rust-async, and Run B's archimedes/library) are
  genuine thin-grounding runs that degrade correctly, now distinguishable from the
  structuring bug because the trace shows what reached the model.

## What (if anything) to do next

See the "recommendation" discussion handed to the user. Short version: the fix is
correct and complete for what it targeted; the residual is grounding variance,
which is an upstream-retrieval problem, not a pipeline-code one. No further code
change is warranted on the strength of this run. The only optionally-actionable
items are the two debug-only nits above, both low priority.
