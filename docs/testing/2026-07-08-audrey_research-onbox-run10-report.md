# audrey_research on-box run 10 — S4 chunking fix confirmed in production

**Date:** 2026-07-08
**Run:** 10 (first run after the chunked fact-check-structuring deploy — `up -d --build audrey-ai`)
**Result:** 10 cases, 10 PASS (all applicable checks green)
**Predecessors:** run-7 (S4 coercer classified + fixed), run-8/9 (coercer verified on box; the prompt-wording attempt failed and was reverted)

---

## Verdict: the wasteful verdict loss is gone

The three largest-ledger cases — the ancient biographies whose single-shot
structuring pass was collapsing to an empty `checks` array in runs 8–9 — now
emit real per-claim fact-check verdicts:

| Case | Ledger claims | Sources | Fact-check verdicts | Drops | Hedges |
|------|--------------:|--------:|--------------------:|------:|-------:|
| bio-euclid | 102 | 0 | **33** | 1 | 4 |
| bio-pythagoras | 92 | 1 (THIN) | **30** | 1 | 5 |
| bio-archimedes | 84 | 0 | **27** | 2 | 8 |

In runs 8–9 these three returned **NO CORRECTIONS** to the writer despite
carrying real verifier flags — the fact-checker's active judgments were being
discarded. Run 10 recovers them, including the **high-value active verdicts**,
not just hedges:

- **euclid** `w2_c42 → unsupported → DROP`: "Ratdolt's 1482 edition was the
  first printed book to use diagrams from movable type" (the diagrams were
  woodcuts).
- **archimedes** `w2_c5 → unsupported → DROP`: sphere/cylinder relationship
  "carved on his tomb **by Marcellus**" (Marcellus was the Roman general; the
  tomb was later found by Cicero).
- **archimedes** `w2_c7 → unsupported → DROP`: Sand-Reckoner "proposing a
  heliocentric model" (Archimedes **reports** Aristarchus's; he doesn't propose
  it).

These are exactly the CORRECT/DROP signals the deterministic hedge-disposition
path (`hedge_policy`) could never reconstruct — it re-derives HEDGE from
`risk`/`needs_hedge` but has no knowledge of the fact-checker's active
corrections. Chunking is what put them back in front of the writer.

## The batch-independence property is visibly working

The recovered verdicts' claim-IDs are **scattered across both worker prefixes
and non-contiguous** (euclid checks span `w0_claim_16..40` and `w2_c12..42`;
archimedes spans `w0`, `w1`, `w2`). A collapsed batch would show a contiguous
ID gap. There is no such gap in any of the three — every 15-claim batch
structured cleanly, and no batch's failure could have zeroed its neighbours.

No `ValidationError` crash, no empty pass, no `fatal_errors`-object rejection
(the run-7 coercer + the run-10 chunking together close both failure modes).

## The prompt revert did not reintroduce the contradiction problem

The run-9 prompt edit (redirecting contradictions to a `conflicting` verdict)
had caused the model to skip the structured pass entirely. Reverted. In run 10,
contradictions surface as **verifier critiques + drops** (e.g. the
pythagoras historicity contradiction between researcher 1 and researcher 2 is
flagged in the verifier critique and softened via `needs_hedge` corrections),
never wiping the whole pass.

## The 7 non-ancient-bio cases held

- **hist-library-alexandria**, **hist-parallel-postulate**, **current-rust-async**,
  **current-2025-recent**, **tech-transformer-attention** — all PASS, verdicts
  present where claims exist; transformer's aggressive drop set (9 drops on
  unsupported no-url claims) is the fact-checker correctly pruning
  training-data claims that never got a retrieved source.
- **ctrl-birthday-toast**, **ctrl-explain-recursion** — both correctly emit
  `NO CORRECTIONS` for rule-free creative/explanatory content (verifier: "no
  factual claims… nothing to flag"). This is the intended path, not a failure.

---

## Why euclid and archimedes show `sources:0` (grounding, not the fix)

This is upstream of the structuring pass and is **not** a regression — it's the
known intermittent-grounding story. The chain:

1. **web_search returned empty / irrelevant, not errored.** Every researcher's
   footer shows `web_search ✅6` (euclid) / `✅6` / `✅5 ❌1` (archimedes) — the
   calls succeeded at the HTTP level. But the researchers' own notes say the
   payloads were empty: euclid/deepseek "web searches… returned no usable
   results"; euclid/glm "my web-search quota for this turn was hit after a few
   calls, and the knowledge base had no Euclid material"; archimedes/glm "the
   web searches came back empty… the KB hits were irrelevant (PowerApps entity
   references, mineralogy textbooks)." A `200 + 0 results` is the documented
   SearXNG upstream-throttle signature, and the footer's `✅/❌` counts
   `is_error` only, so an empty 200 shows green.

2. **No retrieved pages → training-data fallback → `no url` sources.** With
   nothing retrieved, the researchers cited scholarly works by name (Heath,
   Bulmer-Thomas, Proclus, Netz…) with `— no url`, `supports: none`. Every
   source in the euclid and archimedes ledgers is a `no url` reference.

3. **The `sources:N` counter only counts URL-bearing retrieved sources.**
   Named-but-unretrieved references don't qualify, so the footer reads
   `sources:0`.

**Pythagoras is the tell:** same topic, same session, but `sources:1` because
one researcher (glm) *did* retrieve the SEP page with a real URL
(`https://plato.stanford.edu/entries/pythagoras/`) while its siblings came up
empty. Same query, different luck against the throttled upstream — the
signature of intermittent rate-limiting, not a dead SearXNG. Nothing to
restart; the cure is Brave-key renewal. See memory
`project_searxng_upstream_throttle`.

Crucially, the pipeline **degraded correctly** on the ungrounded cases: every
ancient-bio answer opens with a hedge banner ("I couldn't fully verify these
details against retrieved sources"), the fact-check pass hedged the anecdotes
and dropped the unsupported specifics, and nothing was fabricated. Thin
grounding is a quality ceiling on these three answers, not a correctness bug —
and it is orthogonal to the structuring fix this run confirms.

---

## Deploy state

- Code change (chunked structuring + `_norm_fatal_errors` coercer + prompt
  revert) is **deployed and confirmed** on the box via `up -d --build audrey-ai`.
- The S4 arc (coercer → prompt revert → chunking) is closed end-to-end.

## Known, non-blocking, pre-existing (unchanged from run-7 report)

- `tests/test_research_stream.py::test_research_stream_no_trace_block_by_default`
  fails on clean `main` — two-layered (fake's missing `format=` kwarg masks a
  trace-renders-flag-off assertion). Logged to `docs/lesson-ai/AUDIT.md` with a
  "don't just patch the assertion" note. Not in S4 scope.
- lesson-17 cite drift (worsened by the chunking insertion's `deep_panel.py`
  anchor shift; 0 broken). Logged to AUDIT.md.
