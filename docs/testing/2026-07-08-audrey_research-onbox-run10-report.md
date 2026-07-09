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

## Why euclid and archimedes show `sources:0` (CORRECTED: structuring, not grounding)

> **Correction (2026-07-09 run, with `debug_research_trace` on).** The original
> version of this section attributed the euclid/archimedes `sources:0` to
> SearXNG upstream throttle ("`200 + 0 results`", "Brave-key renewal is the
> cure"). **That diagnosis was wrong.** The research-trace debug block, which
> logs the web_search tool content actually present in each researcher's model
> context and renders the merged ledger, shows the opposite: grounding *reached
> the model*. In the 2026-07-09 run the same two topics returned `sources:8`
> each, with the researchers' notes full of real URLs
> (`en.wikipedia.org/wiki/Euclid`, `britannica.com/biography/Euclid`,
> `plato.stanford.edu/entries/euclid/`, the Walters palimpsest site, …). SearXNG
> was never the failure. The real defect is downstream, in the notes→ledger
> **structuring pass**.

The corrected chain:

1. **Grounding arrived; the structuring pass dropped it for one worker.** The
   notes→`ResearchResult` pass (`_structure_one_draft`) converts each
   researcher's prose SOURCES block into a structured `sources` array. For the
   qwen worker (`w2_`), that pass intermittently emits **content-free source
   rows** — the 2026-07-09 trace shows them as `w2_, (unknown) untitled — no
   url` (a stray token became the `id`; title and url came back empty) even
   though qwen's *prose* notes for the same query carry real URLs. The
   fail-soft `Source` schema — every field defaulted so one `url: null` can't
   discard the whole worker (the 2/3-drop guard in `ledger.py`) — is now *too*
   tolerant: it resurrects an empty row instead of dropping it.

2. **The `sources:N` counter only counts URL-bearing rows.** An empty
   `— no url` row doesn't qualify. When the *other* workers' clean sources also
   failed to carry (or in run 10 hadn't been structured at all), the footer read
   `sources:0`. In the 2026-07-09 run deepseek+glm's clean rows carried the
   count to `8`, while qwen's rows were *still* malformed — which is exactly why
   the failure looked topic-specific and intermittent.

3. **This is per-worker and non-deterministic**, so it masqueraded as "luck
   against a throttled upstream." It is not. There is nothing to restart and no
   Brave key to renew; the fix is in `ledger.py` (drop content-free sources;
   see the S5 ledger-hardening change).

**Pythagoras in run 10** (`sources:1`) fits the corrected story: one worker's
clean URL survived structuring while its siblings' rows were dropped or empty —
a structuring outcome, not a retrieval one.

A genuinely ungrounded case *does* exist and is now distinguishable from this
bug: `hist-parallel-postulate` and `current-rust-async` (2026-07-09 run) show
`sources:0` where **all three** researchers' notes explicitly report empty
web_search + irrelevant KB, and the answers hedge the whole body. That is the
pipeline degrading correctly on real grounding loss — a different failure from
the structuring drop, and only tellable apart now that the trace shows what
reached the model.

Crucially, the pipeline **degraded correctly** on the ungrounded cases: every
ancient-bio answer opens with a hedge banner ("I couldn't fully verify these
details against retrieved sources"), the fact-check pass hedged the anecdotes
and dropped the unsupported specifics, and nothing was fabricated. Thin
grounding is a quality ceiling on those answers, not a correctness bug — and it
is orthogonal to the structuring fix this run confirms.

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
