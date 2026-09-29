# Eval report — 2026-07-06 research protocol, ON-BOX, first run with `debug_research_trace: true`

Paired with [`2026-07-06-audrey_research-onbox-answers.md`](2026-07-06-audrey_research-onbox-answers.md).
First live exercise of the research-trace debug view (commit `da2365a`), full
protocol on the box. **10/10 structural PASS.** File is ~575 KB vs ~73 KB for the
07-02 run — the trace is ~8× the answer content, fine for an opt-in debug artifact.

## Headline

**The trace view works — every soft answer in this run is explainable from the
saved file alone, which was the entire point.** It immediately paid for itself:
it pinpointed the stage behind each of the three weak answers (all SearXNG
starvation, not code), surfaced **one real product bug** (structuring-stage
claim→source linkage break) and **one eval bug** (the `sources:N` line is
corrupted by the trace when the flag is on), plus a minor `</think>` leak.
Answer quality on the grounded-successfully cases matches the 07-02 baseline.

## 1. The run itself — a mid-run SearXNG throttle window, handled honestly

Cases 1–5 (bios + history) grounded richly (6–8 authoritative sources each,
answers on par with 07-02). Then cases 6–8 — `current-rust-async`,
`current-2025-recent`, `tech-transformer-attention`, consecutive in run order —
got **zero usable search results**. The footers show almost all ✅ (the
known signature: HTTP 200 + 0 results counts as success), but the researcher
notes state it outright: "every query came back empty" (deepseek), "search
tools returned no usable results" (glm). This is the documented upstream
throttle kicking in after the ~150-search burst of the first five cases — the
single empty-retry recovers blips, not a sustained window. Environmental, not a
regression; the 07-02 run grounded all three of these cases with 8 sources each.

What matters is the pipeline's behavior under starvation, and it was correct
everywhere:

- **`current-2025-recent` is a model degrade.** qwen hallucinated a confident
  2025 narrative from training data (including Mistral NeMo as an early-2025
  release); the **verifier caught it** — flagged the anachronism (NeMo is July
  2024) and the unsupported superlatives; **fact-check dropped 6 of 7 checked
  claims**; the writer produced an honest two-part answer (grounded baseline
  plainly, 2025 specifics explicitly declined). The whole causal chain is
  readable in the trace in under a minute.
- **`tech-transformer-attention`** fell back to canonical training knowledge,
  said so, declined the bibliographic specifics. Accurate content.
- **No "elided" leak, no fabricated sources, no error markers.** qwen's
  2025-recent notes narrate the compaction stub as "results were omitted from my
  context window due to budget constraints" — the reworded stub doing its job
  (no longer reads as a failed search), though qwen still narrates it rather
  than working around it.

## 2. NEW product bug — structuring emits broken claim→source links (`current-rust-async`)

The trace's ledger for rust-async shows **79 claims, 5 sources** — but every
source has `supports: none`, and the claims that do reference sources use
**title strings as IDs** (`sources: w1_Glommio repository (Datadog)`) instead of
the source IDs (`w1_s3`). On top of that, all 5 source URLs are scheme-less
(`tokio.rs`, `github.com/DataDog/glommio`), which `_usable_url` rightly rejects.

Downstream, everything keyed on that linkage broke at once:

- `_surviving_source_ids` returns a non-empty set of garbage IDs, which
  **defeats the deliberate empty-linkage fallback** in `_render_sources_block`
  (deep_panel.py:948–950 — "if linkage is empty across the board, fall back to
  every source"). Non-empty-but-useless linkage isn't "empty", so no fallback.
- `_source_types_for_claim` finds no backing for any claim →
  `hedge_policy` hits the empty-source_types default → **hedge for essentially
  everything** → a **44-line HEDGE wall** handed to the writer. The +38
  all-hedge suppression didn't fire because `hedge_or_cite_strongly` lines count
  as non-hedge signal — but with zero usable sources nothing *can* be cited
  strongly, so in practice the block was a blanket-hedge instruction.
- Result: the rust-async answer is complete and accurate but soaked in
  "reportedly" — it hedges stable facts like "tokio-console exists". The writer
  even wrote "(Hedged per Findings)" in a heading. This is writer *obedience*,
  not writer timidity — the trace proves the instructions were the problem.

Caveat that keeps this honest: with zero retrieved grounding, those 5 "sources"
were reconstructed from prose mentions in memory-written notes, so suppressing
the Sources block was arguably the right *outcome* this time. The bug is the
mechanism, and it will bite on a properly-grounded run too.

Fix candidates (separable, all small):
1. **Repair linkage in `ledger.py`:** during `_backfill_ids`, resolve any
   `source_ids` entry that matches no source ID by title match (the failure
   shape is literally the title string).
2. **Harden the fallback:** in `_surviving_source_ids`/`_render_sources_block`,
   treat "keep ∩ actual source IDs = ∅" the same as empty linkage.
3. **Starve-proof the disposition block:** suppress (or collapse to one line)
   when no ledger source has a usable URL — nothing can be "cited strongly",
   so the wall carries no per-claim signal.

## 3. NEW eval bug — the trace corrupts the `sources:N` quality line

`_sources_block` (eval_research.py:312) locates the Sources section with
`rfind("## sources")` over the *whole* answer, relying on the +52 invariant
that no trace heading starts with "Sources". The invariant holds for
**renderer-generated** headings — but researcher-note *content* is embedded
verbatim (only `---` lines are neutralized), and researchers write their own
`## Sources` / `## SOURCES:` headings, which land *after* the real block. Three
hijack vectors, all present in this run:

- **`bio-euclid`** — glm's notes contain `## Sources` (twice, see §4); the rfind
  landed on the later one → **eval counted 6, the real block has 8 entries**.
- **`hist-parallel-postulate`** — a note has `## SOURCES:`; and because the
  trace's own section headers are `###` (the bound is the next `\n## ` H2
  only), the hijacked block swallowed the **entire ledger** → **sources:25 vs
  the real capped 8**.
- **`tech-transformer-attention`** — landed on a URL-less `## Sources` in
  deepseek's notes → sources:0 "correct" by luck.
- Latent fourth vector: the substring match also fires *inside* `### Sources…`
  (offset 1), so even an H3 note heading hijacks it.

`bio-pythagoras`, `bio-archimedes`, `hist-library-alexandria` had no note-level
Sources headings, so their counts are real. Net: **with the trace flag on, the
sources:N line is untrustworthy** — this run's per-case source-quality numbers
should be read from the answers file, not the eval output.

Fix (eval-side, 2 lines, keeps the renderer honest): cut the answer at
`\n## Research trace (debug)` before searching, and anchor the search to a
line-start H2 (`^## sources`, multiline) instead of a bare substring.
Belt-and-suspenders renderer-side option: demote embedded note headings in
`_sanitize_draft_text`.

## 4. Minor — `</think>` leak in researcher notes

glm's `bio-euclid` note carries its full thinking block *plus* the final
draft — the entire draft appears twice, separated by a dangling `</think>`
(the opening tag is gone). One occurrence in the file, but note content feeds
the structuring and verifier stages too, so when it happens it doubles that
worker's context contribution, not just the display. Worth a look at where
worker note content is captured (think-stripping on the research-worker path).

## 5. Fact-check stage variance (visible for the first time — a trace win)

| case | checks | drop / hedge | note |
|---|---|---|---|
| bio-euclid | 8 | 3 / 4 | healthy |
| bio-pythagoras | **0** | 0 / 0 | **4 fatal errors** (bare claim IDs) — stage aborted, fail-soft; answer unaffected |
| bio-archimedes | 13 | 0 / 10 | healthy |
| hist-library-alexandria | 2 | 0 / 2 | thin |
| current-rust-async | **79** | 0 / 0 | rubber-stamped 79 claims against 0 linked sources |
| current-2025-recent | 7 | 6 / 1 | the star of the run |
| tech-transformer-attention | 4 | 0 / 4 | fine |
| ctrl-explain-recursion | 24 | 0 / 0 | all "supported" with no sources — verdict here means "consistent with model knowledge" |

Before the trace, pythagoras's silent fact-check abort and rust-async's
79-check rubber stamp were invisible. Neither harmed its answer this run;
both are now observable. The fatal-errors rendering (bare claim IDs, no
reason) could usefully carry the error text.

## 6. Controls — both clean (hedge_policy canary still passing)

- **`ctrl-explain-recursion`** (114.3s): confident tutorial, zero hedge-speak.
  Trace shows 24 low-risk claims, all supported, and the **all-hedge
  disposition block correctly suppressed** — the +38 fix visibly working.
- **`ctrl-birthday-toast`** (68.2s): warm toast, no hedging, no Sources. The
  trace shows the verifier and corrections stages explicitly recognizing
  "purely creative, nothing to audit" — the short-circuit is legible now.

## Latency

| case | total | ttft |
|---|---|---|
| ctrl-birthday-toast | 68.2s | 13.7s |
| ctrl-explain-recursion | 114.3s | 15.2s |
| tech-transformer-attention | 254.0s | 0.0s |
| hist-library-alexandria | 275.1s | 11.1s |
| current-2025-recent | 276.4s | 12.7s |
| current-rust-async | 291.1s | 8.5s |
| hist-parallel-postulate | 362.8s | 0.0s |
| bio-pythagoras | 399.5s | 9.3s |
| bio-euclid | 409.7s | 11.7s |
| bio-archimedes | 439.9s | 9.2s |

In line with 07-02. Notably the three starved cases were *faster* than the
grounded bios — empty results end ReAct rounds early. All three workers
contributed on every case (no drops this run; the archimedes qwen-drop did not
recur).

## Disposition

- **`debug_research_trace`: validated.** Keep the flag available; flip back to
  `false` for normal operation (575 KB answer files and ~8× trace overhead are
  eval-only artifacts).
- **Followups (a)–(c) FIXED same day** (laptop, awaiting deploy): (a) eval
  `_sources_block` now cuts at the trace opener and anchors to a line-start H2
  — replayed against this run's answers file, parallel-postulate corrects
  25→8 and euclid 6→9 (its real block has 8 entries, one carrying two URLs);
  (b) `ledger.py` repairs title/URL-string `source_ids` at parse time,
  `_surviving_source_ids` ignores unresolvable refs (the fallback fires
  again), and the disposition block suppresses when no ledger source has a
  usable URL (nothing can be "cited strongly" → the HEDGE wall is blanket
  caution); (c) worker replies are think-stripped in `_run_one_worker`.
  (d) fact-check fatal-error text in the trace remains a nice-to-have — the
  model only emitted bare claim IDs, so there was no richer text to render.
- **SearXNG throttle:** no new action; this run adds the data point that the
  throttle can open mid-protocol and starve the back half of a run. Brave-key
  renewal remains the cure.
- **hedge_policy=true:** controls clean again — second consecutive clean canary.
