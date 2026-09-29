# Eval report — 2026-07-06 research protocol RUN 3 (post-fix, first clean full protocol) — 10/10

Paired with [`2026-07-06-audrey_research-onbox-run3-answers.md`](2026-07-06-audrey_research-onbox-run3-answers.md).
Third run today; first **complete** protocol on the fixed build (run 1 = pre-fix
baseline, run 2 = killed at case 3 by the 10:00 stack restart). **10/10 PASS,
~42 min total, no infrastructure interruptions.**

## Headline

**All three fixes hold across a full protocol, and the correction chain is the
star of this run** — the verifier caught real factual errors on five cases and
the corrected text landed in the answers verbatim. The run itself was heavily
search-starved (deepseek got empty results on essentially every case), so this
is the best stress test yet of degraded-mode behavior: honest everywhere, with
one confident false specific slipping through (MiniMax-M1 date) and one
over-caution in the other direction (Llama 4 demoted to "unconfirmed").

## Fix validation (all three, full-protocol)

- **sources:N matches the rendered block on all 10 cases** (3/0/0/8/0/4/0/4/0/0)
  with traces present everywhere — no hijack, no cap violations.
- **Zero `</think>`** in 406 KB of output.
- **The starve-proof disposition suppression fired correctly, twice:**
  `bio-pythagoras` and `current-2025-recent` both had all-hedge ledgers with
  zero usable-URL sources → no wall handed to the writer. On pythagoras the
  all-hedge state was itself caused by the *mnemonic-ref* linkage failure
  (below), so the suppression papered over it exactly as intended.
- Where linkage worked, the Sources blocks are tight and clean — euclid
  rendered exactly the 3 claim-backed URLs (all official-tier) instead of run-1's
  junk-flecked 8; transformer rendered 4 arXiv links (quality GOOD, academic:4).

## The correction chain earned its keep (5 cases)

Real errors caught by the verifier/fact-check and visibly fixed in the answer:

- **archimedes:** "most accurate estimate of π for centuries" → corrected for
  Zu Chongzhi's 355/113 ("a highly accurate estimate that remained influential");
  215 BC → 214 BC invasion date; Stomachion combinatorial claim softened to
  "some scholars interpret"; Euclid-as-contemporary error flagged (Euclid died
  before Archimedes was born) and absent from the answer.
- **alexandria:** al-Qifti's century corrected, the "Bettini" book
  misattribution flagged, anachronistic "Christian Crusade" replaced, the
  "800–1,000 years" lifespan corrected to "several centuries" — all verbatim
  in the answer.
- **rust-async:** the "~30 transitive crates" claim was **dropped** (fact-check
  `unsupported`) and is absent from the answer; async-std status corrected to
  the specific "latest release 1.13.0, September 2023"; qwen's false
  "async-std is single-threaded" claim caught; the "six major runtimes"
  framing (including Pollster, which isn't a runtime) never reached the answer.
- **transformer:** the false "introduced independently by Bahdanau and Luong"
  claim dropped; the origination claim softened to the alignment-mechanism form.
- **recursion (control):** even here — `StackOverflowError` corrected to
  "RecursionError in Python or StackOverflowError in Java", applied verbatim.

This is the Stage-2/3 design working end-to-end on real errors, not synthetic
ones.

## Run context — heaviest starvation yet, clustered per worker

SearXNG empties again, but with a new shape: **deepseek got all-empty results
on essentially every case** (its notes say so verbatim on 7 cases), glm on
several, while **qwen grounded most cases** (it supplied the URLs behind every
rendered Sources block). Per-worker clustering rather than per-query — worth a
box-side glance at whether the burst ordering starves whichever worker
searches first, but not an Audrey-code issue. Two degraded-mode behaviors
worth naming:

- **deepseek's refusal notes are model behavior at its best:** "Without any
  retrieved evidence, I cannot responsibly distinguish historically attested
  facts from later legends" (pythagoras). A refusing researcher costs a
  worker's contribution but poisons nothing downstream.
- **A new starvation vector — compaction ate a worker's own results
  (postulate):** deepseek ran 5 search rounds, then wrote "the results of the
  searches I performed were omitted from the conversation during context
  compaction" and refused. `compress_keep_last: 2` keeps the last two rounds
  verbatim; a worker that searches five rounds and synthesizes at the end has
  lost its early evidence. The stub reword did its job (no "failed search"
  narration), but the underlying starvation is real. Candidate lever:
  raise `compress_keep_last` for research workers (or compact less
  aggressively near the final round) — needs its own eval.

## Quality reads

- **The one factual miss: MiniMax-M1 "released February 3, 2025"**
  (2025-recent) — stated near-plainly, from a single no-URL pseudo-source, and
  wrong (that date matches neither MiniMax-01 in January nor M1 in June).
  Neither verifier nor fact-check could catch a wrong date with zero retrieved
  grounding — the known architectural limit, now with a concrete specimen.
- **The flip side: Llama 4 demoted to "unconfirmed rumor"** on the same case —
  epistemically correct given zero retrieved evidence this session, factually
  over-cautious (run 1 grounded the April 2025 release with official sources).
  Grounding variance, not a regression; the pair makes a good teaching example
  of why the lever for recent-events prompts is retrieval, not prompt tuning.
- **Hedge-density tracks grounding thinness, and the mechanics are now clear:**
  euclid's body says "reportedly" **20×** (run 1: 2×), including on textbook
  facts ("the first printed edition was reportedly produced by Erhard Ratdolt
  in Venice in 1482"). Chain: a worker's blanket "couldn't verify this
  session" caveat → structuring marks *every* claim from it `needs_hedge` or
  `risk: high` → wall of HEDGE lines (rendered walls this run: euclid 43,
  archimedes 59, alexandria 49, postulate 23, rust-async **94**, transformer
  34) → the writer obeys. Writer obedience is correct; the wall's inputs are
  the problem. Same carry-forward as run 2, sharper evidence.
- **Controls clean for the third consecutive run** (toast 66s no hedging;
  recursion confident, one legitimate correction applied).

## Fact-check verdict channel — now clearly two-faced

- **Real verdicts with bite** on archimedes (11 checks, 9 hedge), rust-async
  (23 checks, 1 drop + 9 corrected hedges), alexandria, transformer, recursion.
- **Rubber stamps** on postulate (42/42 "supported" against no-URL
  pseudo-sources) and 2025-recent (51/51 "supported" — including the training
  -data Llama-4 claims the *verifier* simultaneously flagged as unverifiable).
- **Absent entirely** on euclid and pythagoras (stage aborted/unparseable,
  fail-soft, second+ occurrence for pythagoras).

Pattern: the verdict channel means "consistent with the notes", not "verified
against sources", whenever grounding is thin — the prose corrections channel
is doing the real verification work. Carry-forward from run 2, confirmed.

## Residual (new, low-priority): mnemonic source refs

The structuring models emitted a third linkage shape: mnemonic IDs
(`w2_SUDA`, `w0_plutarch_marcellus`, `w1_src_diogenes`) that match neither the
source id (`w2_s1`) nor its title. The title/URL repair can't resolve these.
Impact this run was nil-to-benign (the affected sources were no-URL anyway,
and the pythagoras all-hedge fallout was correctly suppressed), but it starves
`hedge_policy` of type info on claims that do have authoritative backing.
Possible repair: fuzzy/substring match of the ref against titles. File it;
don't fix without a case where it visibly hurts.

## Latency

Fastest full protocol yet: euclid 349.7s, pythagoras 311.5s, archimedes
333.2s, alexandria 287.4s, postulate 230.7s, rust-async 371.2s, 2025-recent
212.8s, transformer 226.3s, toast 66.3s, recursion 125.2s (~42 min total).
Empty search results end ReAct rounds early — starvation is cheap.

## Disposition

- **The three fixes are validated at full-protocol scale.** Flip
  `debug_research_trace` back to `false` for normal operation whenever the
  eval campaign pauses.
- **Carry-forwards (design-first, not reflex fixes):** (1) disposition-wall
  inputs — structuring's wholesale `needs_hedge`/`risk: high` on caveated
  notes; (2) fact-check verdict reliability under thin grounding; (3)
  research-worker `compress_keep_last` starvation; (4) mnemonic source-ref
  repair; (5) per-worker search starvation clustering (box-side look).
- The SearXNG throttle remains the dominant quality lever for the
  current-events cases — Brave-key renewal is still the cure.
