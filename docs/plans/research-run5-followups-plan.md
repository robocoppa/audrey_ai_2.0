# Plan — research-mode followups from the 2026-07-07 run-5 assessment

> **STATUS (2026-07-08):** Stages 0–2 BUILT on laptop, awaiting deploy.
> **S0** search-budget relax shipped in config (research 4→6, factcheck 3→4,
> deep 3→4, fast unchanged; real config resolves to a 22-call ceiling, verified
> via the budget probe). **S1** `src{N}`/`source{N}` alias in
> `_repair_source_links` + 2 pins (the run-5 shapes, plus a real-`src2`-id
> no-shadow guard). **S2** compaction-proof-notes sentence in
> `RESEARCHER_SYSTEM` + byte pin updated. Verification: 707 pytests pass, ruff
> at the 9-ASYNC240 baseline, cite checker 0 confident drift / 0 broken (2
> `hedge_policy` cites re-anchored 350→375 for the ledger.py additions).
> **S3** = the gate run (not yet run). **S4** = user-run box greps (pending).
> Deploy: config picks up via `up -d --force-recreate audrey-ai`
> (bind-mount stale-handle); S1+S2 need `up -d --build audrey-ai`.

Source: [`docs/testing/2026-07-07-audrey_research-onbox-run2-report.md`](../testing/2026-07-07-audrey_research-onbox-run2-report.md)
(and PROJECT_STATE +62). Run 5 verified the A+B linkage fixes and produced the
best fact-check run yet; what remains are three issues with known mechanics,
plus a user policy decision: **relax the per-worker search caps now that Brave
is renewed and SearXNG absorbs overflow.**

The issues, ranked by answer-quality impact:

| # | Issue | Mechanics (traced, not theorized) |
|---|---|---|
| 1 | Cap→tool-pivot→compaction: workers spend the web budget by round 2–3, pivot to kb/memory/chat for 2–3 more rounds, and `compress_keep_last: 3` evicts their own web evidence (euclid ×2 workers: "compacted out … before I could read them") | Budget too tight for 5-round workers + notes aren't compaction-proof |
| 2 | New orphan-ref shape: claims cite `SRC-1`/`src_1`-style refs against backfilled `s{N}` ids (euclid w2 ×4, transformer w2 ×2, pythagoras w1 ×2) → sourced claims hedge | `"src-2".lower()` ≠ `"s2"`; repair has no alias for it |
| 3 | Grounding variance run-to-run (euclid/rust THIN on run 5) muddies every hedge-density comparison | SearXNG upstream throttle + Brave quota caution |

## Stage 0 — Relax the search budgets (config-only; user-decided policy)

Brave is renewed (primary) and the Brave→SearXNG fallback means an exhausted
quota degrades to self-hosted search instead of dying — so the caps can carry
more headroom than the renewal-era minimum. Run-5 evidence supports it:
workers fully consume 4 and then pivot to kb/memory/chat calls that returned
junk (geology KB hits, empty memory) while inflating round counts.

Proposed numbers (all in [config.yaml](../../config.yaml)):

| profile | now | proposed |
|---|---|---|
| `agentic.react.research_worker.max_web_searches` ([config.yaml:309](../../config.yaml#L309)) | 4 | **6** |
| `agentic.react.factcheck_worker.max_web_searches` ([config.yaml:319](../../config.yaml#L319)) | 3 | **4** |
| `agentic.react.deep_worker.max_web_searches` ([config.yaml:286](../../config.yaml#L286)) | 3 | **4** |
| root `agentic.react.max_web_searches` (fast path, [config.yaml:276](../../config.yaml#L276)) | 3 | 3 (unchanged) |

A research request then tops out ≈ **22** Brave-able calls (3×6 + 4) — up from
15, still half the pre-cap 30–45. Fast path stays tight (no starvation
evidence there). `max_rounds`/`compress_keep_last` unchanged — models batch
searches, so 6 fits in the same 5 rounds, and the extra web headroom *reduces*
the junk-tool pivot that drives compaction pressure.

Deploy: config edit on the box + `docker compose up -d --force-recreate
audrey-ai` (bind-mount stale-handle — a plain restart does NOT pick up config
edits). No rebuild.

Watch-items: SearXNG `200 + 0 results` pattern on Brave-exhausted days (the
extra budget buys nothing if the fallback is throttled), and per-case latency.

## Stage 1 — `src{N}`/`source{N}` alias repair (code, deterministic)

Extend `_repair_source_links` ([ledger.py:229](../../src/audrey/pipeline/ledger.py#L229)):
alongside the existing id/title/URL aliases, register normalized aliases so a
ref whose lowercased, punctuation-stripped form is `src{N}` or `source{N}`
resolves to the source with id `s{N}` (the `_backfill_ids` shape). Exact-id
match keeps priority; a genuine `src2` id can't be shadowed (`setdefault`,
real ids registered first). No fuzzy matching — observed shapes only.

Files: `ledger.py` + pins with the three captured run-5 shapes
(`SRC-2`→`s2`, `src_1`→`s1`, `src_3`→`s3`). Hermetic; ships with any rebuild.

Explicitly out of scope: author-year mnemonics (`w0_bahdanau2014` against a
source titled "Neural Machine Translation…") — no deterministic handle; stays
parked.

## Stage 2 — Compaction-proof researcher notes (prompt, one sentence)

The plan-designated S5 fallback lever, trigger met by run 5. Add one sentence
to `RESEARCHER_SYSTEM` ([prompts.py:125](../../src/audrey/pipeline/prompts.py#L125)):
instruct the researcher to restate the key facts it just retrieved — with
their URLs — in its own reply each round, because its own words are what
persists. Wording constraint: frame it positively ("your notes are your
memory"), do NOT teach the model new failure-narration vocabulary ("results
were lost/compacted") — same wording discipline as the budget stub.

Live-tunable via `agentic.prompts.researcher`
([deep_panel.py:1248](../../src/audrey/pipeline/deep_panel.py#L1248)) if a
rollback is needed without rebuild. Files: `prompts.py` + substring test.

Cost: slightly longer researcher replies (they already write notes; this
front-loads URLs into them). Benefit: a 5-round worker's evidence survives
`keep_last: 3` regardless of tool mix.

## Stage 3 — Gate: one protocol run for Stages 0+1+2

Three changes, one run — acceptable because the observables are distinct:

- **Stage 0** → footers (≤6/worker research, ≤4 factcheck), `sources:N`
  quality lines (expect fewer THIN cases), latency delta.
- **Stage 1** → ledger refs: zero `src{N}`-shaped orphans; sourced claims out
  of the walls.
- **Stage 2 (+0 jointly)** → zero "compacted out / not retained" narrations in
  researcher notes. Attribution between the nudge and the bigger budget is
  accepted as joint — the target is zero occurrences, not causal credit.

Also carry the standing checks: no under-hedging on ancient-bio anecdotes,
controls clean, cap arithmetic in footers.

## Stage 4 — Fact-check no-verdicts classification (user-run, unchanged)

Still outstanding from the run-3 plan; run 5 adds pythagoras + library to the
sample (verdicts block absent, bare "NO CORRECTIONS"). On the box, covering
the run-5 window:

```bash
docker logs audrey-ai --since 48h 2>&1 | grep -E "parse_factcheck|factcheck ledger|fact-check stage" | tail -30
docker logs audrey-ai --since 48h 2>&1 | grep -B2 -A2 "unusable model output" | tail -40
```

Outcome A (tolerable JSON shape): extend the parser + pin. Outcome B (model
garbage): fail-soft already handles it; no action.

## Explicitly not doing (so it isn't re-litigated)

- **No author-year mnemonic repair** — no deterministic mapping exists; parked
  until a shape appears that one does.
- **No Sources-block URL validation** — the wrong-domain MacTutor URL (run 5,
  euclid) would need live fetching to catch; verifier already flags it in the
  trace. Revisit only if hallucinated URLs recur on grounded runs.
- **No paraphrase-disposition conflict resolution** (postulate's hedge-soak) —
  needs fuzzy cross-worker text matching; re-measure after Stages 0–2, since
  better grounding shrinks the ungrounded-worker claim sets that cause it.
- **No hedge_policy reorder / S4** — still deferred; runs 4–5 baselines were
  both muddied by grounding variance, so the grounded baseline S4 wants
  doesn't exist yet. Stage 0 is also the best lever for producing one.

## Suggested order

**0 (config, immediate) → 1+2 (one build) → 3 (gate) → 4 whenever.** Stage 0
can ship today without waiting for the code; 1+2 are small and ride one
rebuild.
