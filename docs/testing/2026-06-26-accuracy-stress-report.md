# audrey_research evaluation — 2026-06-26

First run of the testing protocol (`scripts/eval_research.py` +
`eval_prompts_protocol.json`). 10 prompts against the live stack via OWUI.
Paired answers file: `2026-06-26-accuracy-stress-answers.md` (all 10 outputs from this run).
Structural checks: **10/10 PASS** (all reachable, no error markers, all four
banners in order, all non-empty). This report is the *quality* read the
structural checks can't give.

**Reading this:** scores are 1–5, directional not absolute. I am a fallible
accuracy judge on niche history — treat "flagged claims" as candidates for your
expert review, not confirmed errors. Each verdict is one sample of a
non-deterministic system.

## Headline

**The +30 keyword-list verifier is holding — and clearly.** Across all three
biographies, both history prompts, and even the technical explainer, the caution
discipline is visibly active and correctly applied: "traditionally associated,"
"commonly dated," "no proof he founded," "highly suspect," "researchers widely
credit ... as the first to formulate" (not "introduced"). The exact leak classes
you flagged in prior single-prompt runs are mostly *gone* on this run. The two
controls confirm the other half: caution is **not** over-applied — the birthday
toast has zero hedging and no spurious Sources/"could not verify" caveat.

This is the strongest the mode has looked. The remaining issues are minor
precision wording, consistent with your "precision leaks, not hallucinations"
read — and a couple match leaks you named before that the keyword list doesn't
yet cover.

## Per-case scores

Dimensions: **G**rounding, **C**aution/overstatement-control, **S**tructure,
**A**pparent-accuracy (as far as I can judge).

| case | G | C | S | A | one-line read |
|---|---|---|---|---|---|
| bio-euclid | 5 | 4 | 5 | 4 | Excellent caution; 2 residual leaks ("axiomatic-deductive method," edition count framing) |
| bio-pythagoras | 5 | 5 | 5 | 5 | Best of the set — textbook separation of attested core vs. legend |
| bio-archimedes | 5 | 5 | 5 | 4 | Strong; careful anachronism + "Do not disturb my circles" note |
| hist-library-alexandria | 5 | 5 | 5 | 5 | Outstanding disputed-claims discipline; scroll-count overstatement explicitly corrected |
| hist-parallel-postulate | 4 | 5 | 5 | 4 | Opens with the no-grounding caveat yet delivers a detailed, well-attributed timeline (NOT "compressed") |
| current-rust-async | 5 | 4 | 4 | 4 | Genuinely current (async-std 2024 EOL, io_uring runtimes); grounding beats parametric here |
| current-2025-recent | 5 | 5 | 4 | 4 | Hardest test, best justification for the mode — specific dates, honest "reportedly"/negative results |
| tech-transformer-attention | 5 | 5 | 5 | 5 | Technically excellent; applies verb-caution to attribution ("not the absolute first") |
| ctrl-birthday-toast | — | 5 | 5 | — | Control PASS: zero over-hedging, no spurious caveat/Sources |
| ctrl-explain-recursion | — | 5 | 5 | 5 | Control PASS: clean, no caution where none is needed |

## Flagged claims (candidates for your review, with my confidence)

**bio-euclid** — the one case still showing named leaks:
- *"the axiomatic-deductive method he operationalized at an unprecedented scale"*
  — "operationalized ... at an unprecedented scale" is softer than the bare
  "introduced the axiomatic method" you flagged before, but still an
  origination-flavored claim. **Med confidence it's a mild overstatement.**
- *"more than a thousand editions appeared between ... 1482 and 1900"* — the
  inflated-count pattern. It's now bounded with dates (better than "well over
  1,000"), but the figure itself is the kind of number that varies by source.
  **Low-med confidence.**
- *"erased its predecessors from view"* — vivid, slightly rhetorical. Not a
  factual overclaim, but the flourish-y register the synth anchor usually trims.
  **Low confidence / stylistic.**

**bio-archimedes**:
- *"effectively anticipating integral calculus by nearly two millennia"* —
  common framing, defensible, but "anticipating calculus" is a known
  historians'-caution phrase. **Low confidence.**
- *"Cicero in 75 BC" rediscovering the tomb* — widely repeated; I believe it's
  right but it's a specific date worth your eye. **Low confidence on the year.**

**current-rust-async**:
- *"monoio (alongside glommio and compio)"* and *"kernel 5.12+"* — specific and
  plausibly current, but exactly the version-fact that drifts. **Low confidence
  — needs a domain check, which is the point of grounding.**

**current-2025-recent**:
- Many specific dates ("January 26," "March 12," "April 28," "September 11").
  The answer hedges most with "reportedly," which is correct behavior — but if
  any are wrong, they're confident-looking. **This is the case where I'm least
  able to judge accuracy** (2025 events near/after my own knowledge edge); your
  review matters most here.

## Verifier behavior read

**What it's catching well** (the keyword list is visibly firing):
- Attribution: "traditionally ascribed to Eudoxus," "traditionally credited,"
  "is attributed to" forms throughout.
- Sweeping quantifiers: notably absent — no "virtually all" / "every" leaks this
  run.
- Origination: the transformer answer explicitly demotes "the absolute first."
- Purpose claims: no "written for navigation"–style leaks; the Euclid answer
  lists *Phaenomena* without a fabricated purpose.
- Counts: the Alexandria scroll count is *actively corrected* in-text.

**Where it still leaks** (candidates, NOT yet recommending action):
- *Origination phrasing that paraphrases around the keyword* — "operationalized
  at an unprecedented scale" (Euclid) slips past a list that watches
  "introduced/invented." This is the +29 tension: a list catches words, not
  rephrasings. One sample, though.
- *Conceptual register* — "erased its predecessors from view," "move the Earth"
  framing. These are stylistic flourishes, arguably the synthesizer's job, not
  the verifier's.

The two deferred +31 items (the "innovation" word; Proclus-as-mathematician
role) **did not appear as leaks this run** — Proclus is consistently framed as a
"5th-century CE" commentator/source, correctly. So the data does *not* currently
justify adding them.

## Recommendation

**Hold the verifier as-is.** This run is strong evidence the +30 list is at a
good equilibrium — leaks are down to one case with mild, mostly-stylistic
residue, and the controls prove it isn't over-cautious. Adding more keywords now
would chase a single Euclid-specific rephrasing ("operationalized at an
unprecedented scale") and risk the over-flagging that caused the earlier
regression — exactly the loop the protocol exists to break.

If you want to push further, the *only* data-supported lever is **not another
keyword** but possibly the synthesizer/writer trimming rhetorical flourishes
("erased its predecessors," "move the Earth") — a different surface, lower
priority, and worth its own measured test rather than a reflexive edit.

**Re-run this protocol after any future verifier/prompt change** and diff the
new answers file against this baseline (`2026-06-26-accuracy-stress-answers.md`) — that's the
comparison that turns "seems better/worse" into evidence.

## Caveats on this report

- One run, one sample per prompt. A leak's absence here isn't proof it's fixed;
  a leak's presence isn't proof it's systematic.
- I cannot reliably verify 2025-era facts or niche history dates — the
  apparent-accuracy scores lean on internal consistency + hedging quality, not
  ground truth. You are the accuracy authority.
- Measures whatever build is *deployed*. The strong caution behavior is
  consistent with the +30 verifier being live, but confirm the deploy state if
  it matters.
