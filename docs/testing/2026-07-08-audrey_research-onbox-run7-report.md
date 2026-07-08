# Run-7 assessment — 2026-07-08 (post `compress_keep_last:5` deploy + S4 fix)

Answers: [`2026-07-08-audrey_research-onbox-answers.md`](2026-07-08-audrey_research-onbox-answers.md)
Baseline: run 6 ([`2026-07-07-audrey_research-onbox-run2-report.md`](2026-07-07-audrey_research-onbox-run2-report.md) captured run 5; run 6 assessment lives in PROJECT_STATE +65)
Plan: [`docs/plans/research-run5-followups-plan.md`](../plans/research-run5-followups-plan.md)

**Verdict: 10/10 PASS. This is the gate run for Stages 0+1+2, and all three
land. S2 — the one that shipped-but-ineffective in run 6 — is now decisively
fixed: zero "compacted out" narrations across all ten cases. The run also
supplied the S4 evidence, which overturned the earlier hypothesis: the
fact-check no-verdicts cases are a parser crash (Outcome A), not a
grounding-decline. S4 is now classified and fixed in the same session.**

---

## S2 — compaction-proof notes (the open stage) ✅ DECISIVE

The run-6 failure was that a 5-round `research_worker` at `compress_keep_last:3`
had its early rounds evicted, and `_summarize_tool_message` replaced each
evicted `web_search` with a **contentless** stub — so non-compliant models
narrated their own evidence being lost. The fix raised `compress_keep_last:3→5`
so `keep_last == max_rounds` and `_compress_history` returns before it can stub
any researcher round.

**Result: the narration is gone.** Scanning every researcher-notes block in the
debug trace (now on via `debug_research_trace:true`), not one instance of
"compacted out", "lost before I could read them", "not retained", or
"evicted" appears in run 7. The framing that remains is strictly
**upstream-retrieval** language — glm on archimedes says results were
"truncated/omitted by context-compaction" as an *Ollama-side* fact, and the
ungrounded cases say "web searches returned no results" / "hit the tool-call
budget". That is a different failure mode (the model never received results),
not a worker discarding evidence it held. Mechanism removed, not masked.

## S0 — budget relax ✅

Every research-worker footer caps at exactly `web_search ✅6`; factcheck workers
stay ≤4. No case shows 7+, so the ceiling is holding. The THIN / `sources:0`
cases are **grounding failures, not budget starvation**:

- pythagoras `sources:1 THIN`, and `sources:0 N/A` on archimedes, library,
  parallel-postulate, both `current-*`, transformer — all show SearXNG
  returning empties in the notes (the known upstream-throttle path), and every
  one degraded correctly to "drawn from training data, flagged unverified"
  rather than fabricating a grounded-looking answer.

This is the desired behavior: a starved worker that says so, out loud, beats a
worker that invents citations. No THIN-collapse attributable to the caps.

## S1 — `src{N}` alias repair ✅

Zero `src{N}` / `SRC-` / `src_`-shaped orphans survive to any ledger. The
transformer case is the clean tell: `w2_SRC-1` / `w2_SRC-2` (the exact orphan
shape from runs 5–6) now resolve — claims cite `w2_SRC-1`, the sources block
lists `w2_SRC-1`, linkage intact. Repair is holding across the run.

## S4 — fact-check no-verdicts ✅ CLASSIFIED + FIXED

**The prior hypothesis was wrong.** Last turn's read was "no-verdicts fires when
the fact-check worker gets zero grounding, and the fail-soft is correctly
returning nothing to check." The box greps refute that cleanly.

Box logs (`audrey-ai`, run window), fact-check completions ordered against the
10 cases:

| time | ledger result | case |
|---|---|---|
| 09:52 | **ValidationError (4)** → fail-soft | bio-euclid |
| 09:59 | 5 checks (1 drop, 1 hedge) | bio-pythagoras |
| 10:07 | 10 checks (0 drop, 8 hedge) | bio-archimedes |
| 10:13 | **ValidationError (7)** → fail-soft | hist-library-alexandria |
| 10:18 | 75 checks (0 drop, 0 hedge) | hist-parallel-postulate |
| 10:22 | 9 checks | current-rust-async |
| 10:26 | 7 checks (4 drop, 2 hedge) | current-2025-recent |
| 10:31 | 6 checks (0 drop, 4 hedge) | tech-transformer-attention |
| 10:35 | 18 checks (0 drop, 1 hedge) | ctrl-explain-recursion |

The two NO-CORRECTIONS-in-the-answer cases are **euclid and library**, and both
are **`ValidationError` crashes**, not grounding declines:

- **euclid 09:52**: `fatal_errors.0: Input should be a valid string
  [input_value={'claim_ids': ['w0_c3', ...], ...}]` — the model put a
  correction **object** where the schema wants a string.
- **library 10:13**: `fatal_errors.0: ... input_value={'claim_id':
  'w2_claim-10...', 'conflicting_claim_id': 'w0_c6'}` — same shape, a conflict
  object.

Grounding is not the variable: **pythagoras** got zero grounding too and emitted
5 clean checks; **parallel-postulate** ran near-zero grounding and produced 75.
The variable is **output shape**. When `fatal_errors` (typed `list[str]`)
receives a dict, Pydantic rejects the *entire* `FactCheckResult`;
`parse_factcheck_result` catches the `ValidationError`, returns `None`, and the
fail-soft hands the writer **NO CORRECTIONS** — silently discarding a fact-check
that had real content (euclid's had 10 verifier flags behind it).

This is the run-5 plan's **Outcome A** (tolerable JSON shape → extend the parser
+ pin), confirmed against the box, not theorized.

### Fix

`_norm_fatal_errors` `BeforeValidator` on `FactCheckResult.fatal_errors`
([ledger.py](../../src/audrey/pipeline/ledger.py)) — coerces each entry to a
one-line string (prefers a `message`/`text`/`error`-like key, else a compact
`key=value` join) instead of rejecting it. Same fail-soft intent as
`_norm_verdict` (unknown verdict → `irrelevant` rather than discard the result).
The valid `checks` array now survives a malformed `fatal_errors`. Two pins in
`tests/test_ledger.py` for the euclid and library shapes; the load-bearing
assertion is that `checks` is not lost.

**This is a CODE change** → box needs `up -d --build audrey-ai` (not just
force-recreate).

## Standing checks

- **Controls clean.** `ctrl-birthday-toast` and `ctrl-explain-recursion` route
  research, produce their creative/explanatory output, and the fact-check
  correctly returns NO CORRECTIONS with an explicit "no checkable factual
  claims" note (recursion still emitted 18 checks — the pins there are real).
- **No under-hedging on ancient-bio anecdotes.** euclid's inflated claims
  (`1,000 languages`, `second-most published after the Bible`) are walked back
  in the answer's Part 2 ("difficult to substantiate and likely inflated")
  despite the fact-check crash — the writer's hedge-disposition path caught what
  the crashed fact-check dropped.
- **Cap arithmetic in footers** consistent with S0.

## Two pre-existing problems surfaced (not introduced this session)

Flagged here and logged to `docs/lessons/AUDIT.md`; both fail on clean `main`:

1. `tests/test_research_stream.py::test_research_stream_no_trace_block_by_default`
   fails — the `_FakeOllama.chat()` test double does not accept a `format=`
   kwarg that production now passes. Confirmed by stashing this session's edits
   and re-running: identical failure at HEAD.
2. Lesson-17 carries ~104 stale cites (5 `DRIFT` + 99 `DRIFT?`) into
   `prompts.py` / `deep_panel.py`, from the research-trace plumbing that shifted
   those files in earlier commits. Zero drift is attributable to the S4 edit
   (the `ledger.py` cites did not move).

## Bottom line

All four research-eval stages are closed:

| stage | run 6 | run 7 |
|---|---|---|
| S0 budget relax | ✅ | ✅ (footers cap, THIN = grounding not starvation) |
| S1 `src{N}` repair | ✅ | ✅ (zero orphans) |
| S2 compaction-proof notes | ⚠️ ineffective | ✅ (zero "compacted out" at keep_last:5) |
| S4 fact-check no-verdicts | pending | ✅ (Outcome A — parser coercer + pins) |

The research-eval campaign is at a clean close once the box takes the S4 build.
Parked **Fix B** (URL-carrying compaction stub) is still not needed — keep_last:5
removed the eviction window entirely.
