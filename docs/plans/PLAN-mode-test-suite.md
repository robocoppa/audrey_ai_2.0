# Plan — full mode test suite (deep + fast, after research)

Goal: extend the research-mode eval protocol into a **full testing suite for
all Audrey modes**. We already have a working, diffable harness
(`scripts/eval_research.py` + a case JSON + paired `answers.md`/`report.md`).
This plan adds **deep-mode** and **fast-path** protocols in the same shape, plus
the small script additions the new metrics need.

This is a **plan for approval** — no case files or script edits are written yet.

---

## 1. What we already have (and reuse unchanged)

- `scripts/eval_research.py` — streams cases against the live stack over OWUI,
  runs structural checks, prints + saves answers, exits non-zero on any failure.
  It is **already mode-agnostic**: `_BANNER_SETS` wires `audrey_deep` /
  `audrey_cloud` / `audrey_local` to the deep banners
  (`Planning → Dispatching panel → Synthesizing`), and each case pins its own
  `model`. So **deep mode needs zero script changes** — only a case file.
- The run/diff workflow: one `<date>-<desc>-answers.md` (machine-written, all
  cases) + one hand-written `<date>-<desc>-report.md` (the quality read). Diff a
  new run's answers against the prior file; the report is the verdict.
- The mode surface, confirmed from source:
  - **Always-deep:** `audrey_deep` (mixed pool), `audrey_cloud` (cloud-only),
    `audrey_local` (local-only), `audrey_research` (staged).
  - **Adaptive:** `audrey_auto` — fast when last-prompt < `token_threshold`
    (500) **and** no deep-intent phrase, else deep.
  - **Always-fast:** `audrey_fast` (no escalation, ever).

---

## 2. Deep-mode protocol (`eval_prompts_deep.json`)

Deep mode is the panel-synthesis path: 2–3 workers draft in parallel, a
synthesizer merges with the `[tool-grounded]` factual-anchor rule. The thing to
test is **synthesis quality** — does the merged answer stay accurate, avoid the
uncited-flourish failure mode (the Euclid saga), and use breadth well — across a
**wide topic range**.

**Metrics (key ones):**
- Accuracy / factual discipline (eyeball, same as research).
- Synthesis quality — coherent merge, not a stapled-together list; no
  contradiction between worker contributions.
- Flourish leakage — the specific deep failure mode: a specific claim
  (name/date/coined term) that appears in only one non-grounded draft surviving
  into the answer. This is what the synth anchor (+16) was built to stop.
- Banner order (`Planning → Dispatching panel → Synthesizing`) — already checked.
- No regression vs research on shared prompts (run the same prompt on both,
  compare).

**Case taxonomy (~12 cases, breadth is the point):**
1. **Hard-science explainer** — e.g. "Explain how CRISPR-Cas9 edits a gene."
   (tests technical accuracy from parametric knowledge.)
2. **Contested history** — reuse `hist-library-alexandria` (cross-mode anchor;
   lets us diff deep vs research on an identical prompt).
3. **Multi-part synthesis** — "Compare the causes of the 1929 and 2008 financial
   crises." (forces the synthesizer to merge two strands — where stapling shows.)
4. **Biography with legend/fact split** — reuse `bio-pythagoras` (cross-mode
   anchor; the flourish-leak canary).
5. **Current/dated tech** — reuse `current-rust-async` (does deep, without the
   fact-check stage, still hedge or does it assert a stale date?).
6. **2025-recency stress** — reuse `current-2025-recent` (the headline cross-mode
   diff: research's fact-check stage vs deep's anchor-only — quantifies what the
   fact-check stage adds).
7. **Conceptual/abstract** — "What is the difference between correlation and
   causation, and why does it matter?" (no facts to ground; tests reasoning merge.)
8. **Trade-off analysis** — "SQL vs NoSQL for a new web app — how should I decide?"
   (open-ended; tests breadth without a single right answer.)
9. **Step-by-step / procedural** — "Walk me through setting up SSH key auth."
   (tests correctness + ordering survive synthesis.)
10. **Math/proof sketch** — "Explain why there are infinitely many primes."
    (a known proof; tests the panel doesn't garble a tight logical chain.)
11. **Ambiguous-scope** — "Tell me about Mercury." (planet? element? deity? Roman
    god? — tests how the panel handles genuine ambiguity.)
12. **Control — simple creative** — reuse `ctrl-birthday-toast` (deep should not
    over-engineer a 3-line toast; checks no bloat / latency-only cost).

**Cross-mode anchors** (cases 2, 4, 5, 6, 12) are deliberately shared with the
research protocol so deep-vs-research is a direct prose diff on identical prompts.

**Variants to consider** (decide at build): also run the same deep cases against
`audrey_cloud` and `audrey_local` to compare pools — but that triples runtime
(~90 min). Recommendation: **`audrey_deep` only** in the standard protocol; a
`--model audrey_cloud` re-run is a separate, on-demand pool comparison, not part
of the routine suite.

---

## 3. Fast-path protocol (`eval_prompts_fast.json`)

Fast path is the low-latency single-model path. You chose three metrics:
**answer quality + routing correctness + latency/TTFT**. Two of those need
small script additions (§4) — the harness today measures neither latency nor
which path `audrey_auto` actually took.

**Metrics (key ones):**
- **Answer quality** — fast answers should still be correct and useful for
  their (shorter, simpler) prompts. Eyeball, same rubric.
- **Routing correctness** — for `audrey_auto` cases, did it route to the path we
  expected (fast for short/simple, deep for long/deep-intent)? This is the gate
  that's been tuned (token_threshold + deep_intent_phrases) and most likely to
  regress. Needs a per-case `expect_route: fast|deep` and a way to observe the
  actual route (§4).
- **Latency / TTFT** — fast path's reason to exist. Capture **wall-clock** and
  **time-to-first-token** per case (§4). These are the numbers that make "fast"
  falsifiable run-to-run.

**Case taxonomy (~12 cases), split by intended route:**

*Always-fast (`audrey_fast`) — pure quality+latency, no routing question:*
1. Short factual — "What's the capital of Australia?"
2. Short definition — "What does idempotent mean in HTTP?"
3. Short how-to — "How do I undo the last git commit but keep the changes?"
4. Short creative — reuse `ctrl-birthday-toast` (cross-mode latency anchor).
5. Short reasoning — reuse `ctrl-explain-recursion` (cross-mode anchor).

*Adaptive (`audrey_auto`) — routing-correctness is the point:*
6. Short+simple → **expect fast** — "What time zone is Tokyo in?"
7. Long paste → **expect deep** — a >500-token wall of text + "summarize this."
   (tests the token_threshold gate.)
8. Short but deep-intent → **expect deep** — "Give me a comprehensive deep dive,
   think hard about the tradeoffs." (tests the deep_intent_phrases gate — the
   Phase 22 fix; short-but-demanding.)
9. Borderline length → **expect fast** — a prompt just under threshold (guards
   the boundary; catches an off-by-threshold regression).
10. Deep-intent phrase in a casual prompt → **expect deep** — "can you really
    dig deep on this: why is the sky blue?" (catches over/under-triggering.)

*Edge / robustness (`audrey_fast`):*
11. Empty-ish / one-word — "Hi" (does fast handle a trivial turn cleanly?)
12. Ambiguous short — "Mercury?" (fast's handling of the same ambiguity case 11
    of deep gets — cross-mode behavior diff.)

---

## 4. Script additions (the only code in this plan)

Modest, additive edits to `scripts/eval_research.py` — they don't change
existing behavior (the new fields are opt-in per case):

1. **Latency + TTFT capture.** `_post_stream` already iterates the SSE stream;
   record `t0` before the request, `t_first` on the first content delta,
   `t_end` after `[DONE]`. Return `ttft_s` and `total_s`. Print them per case
   and write them into the answers-file header. **No new dependency.** This
   benefits deep/research too (free latency numbers everywhere).

2. **Route observation for `audrey_auto`.** The fast path emits no banner, so we
   can't currently tell which path ran. Two options:
   - (a) **Banner-presence inference** — if any deep banner appeared, it routed
     deep; if none, fast. Cheap, zero server change, but indirect.
   - (b) **A response signal** — would need an Audrey change to surface the
     chosen route (e.g. a header or a debug marker). Out of scope for a
     send-and-read harness.
   **Recommendation: (a)** — add `expect_route: "fast"|"deep"` to a case; the
   check passes if observed-route (from banner presence) matches. Honest about
   being inference, not ground truth, in the report.

3. **`expect_banners: false` for fast cases** — already supported per-case; fast
   cases set it so the banner check is correctly N/A (not a failure).

4. **Optional `min_words`/`max_words`** sanity bound per case (e.g. the toast
   shouldn't balloon under deep) — cheap guardrail, opt-in. *Nice-to-have;
   drop if it adds noise.*

Everything else (auth, streaming, save-file, `--only`, exit code) is reused.

---

## 5. Suite layout & naming

```
scripts/
  eval_research.py              # the one harness (gains latency + route checks)
  eval_prompts_protocol.json    # research (exists)
  eval_prompts_deep.json        # NEW — deep cases
  eval_prompts_fast.json        # NEW — fast + auto cases
docs/testing/
  <date>-research-*.md          # exists (the runs we just did)
  <date>-deep-answers.md / -report.md      # per deep run
  <date>-fast-answers.md  / -report.md      # per fast run
  PLAN-mode-test-suite.md       # this plan
  README.md                     # NEW — short index: which protocol tests what,
                                #       how to run each, where baselines live
```

**Run commands** (after build):
```
# deep
.venv/bin/python scripts/eval_research.py --model audrey_deep \
  --cases scripts/eval_prompts_deep.json \
  --save-file docs/testing/<date>-deep-answers.md
# fast + auto
.venv/bin/python scripts/eval_research.py --model audrey_fast \
  --cases scripts/eval_prompts_fast.json \
  --save-file docs/testing/<date>-fast-answers.md
```

A future `scripts/run_all_evals.sh` could chain research+deep+fast into a dated
suite directory — offered, not in this plan's scope.

---

## 6. What this plan is NOT doing (scope guard)

- No refusals/safety case set (you scoped fast to quality+routing+latency).
- No automated quality grading — the report stays a human read; checks are
  guardrails. (Same honesty caveat as the research reports: I'm a fallible
  accuracy judge, one sample per case, non-deterministic system.)
- No server/Audrey changes — send-and-read only. Route observation is inferred
  from banners, not a new Audrey signal.
- No pool sweep (cloud/local variants) in the routine suite — on-demand only.

---

## 7. Build order (once approved)

1. Script additions (§4: latency/TTFT, `expect_route`, per-case bounds) +
   extend the hermetic unit tests for the new pure helpers.
2. `eval_prompts_deep.json` (§2 taxonomy, cross-mode anchors reused verbatim).
3. `eval_prompts_fast.json` (§3 taxonomy).
4. `docs/testing/README.md` index.
5. First real runs of each → establish the deep + fast **baselines** (the files
   every future run diffs against), one report each.

Steps 2–4 are pure content (no risk). Step 1 is the only code; it's additive and
unit-testable offline. Step 5 needs the live box (LAN/VPN), like the research runs.

---

## Open decisions for you (before build)

- **Deep case count / topics** — the ~12 in §2 are a proposal; add/cut topics?
- **Cross-mode anchors** — I reused 5 research prompts in deep and 2 controls in
  fast so we get direct diffs. Good, or keep the protocols fully independent?
- **Route observation** — OK to infer route from banner presence (§4.2a), or do
  you want an Audrey-side route signal (bigger change, truer signal)?
- **`run_all_evals.sh`** — build the one-shot suite runner now, or later?
