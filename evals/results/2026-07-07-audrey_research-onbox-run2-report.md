# Run-5 assessment — 2026-07-07 second run (post A+B linkage-fix deploy)

Answers: [`2026-07-07-audrey_research-onbox-run2-answers.md`](2026-07-07-audrey_research-onbox-run2-answers.md)
Baseline: run 4 ([`2026-07-07-audrey_research-onbox-report.md`](2026-07-07-audrey_research-onbox-report.md))

**Verdict: 10/10 PASS. Both linkage fixes verified working in the wild, and the
fact-check channel had its best run yet — but the hedge-density gate is
partially confounded by a grounding collapse on two cases (SearXNG thin), and
the run surfaced a NEW orphan-ref shape plus a cap↔compaction interaction that
meets the plan's S5-fallback trigger.**

## Fix verification (the reason this run exists)

**Bug B (merge remap) — VERIFIED.** Cross-worker canonical refs are visible in
three ledgers: transformer w1 claims cite `w0_s1/s3/s4` (glm's arXiv sources
URL-deduped against deepseek's, claims re-pointed instead of orphaned);
pythagoras w2 claims cite `w0_s1` (SEP dedup); library w1 claims cite
`w0_source_1` (13 claims). In run 4 all of these would have hedged; this run
they carry authoritative backing.

**Bug A (case-variant ids) — no counter-evidence.** No case-variant refs
appeared this run to exercise it directly; nothing regressed.

## Per-case metrics (run 4 → run 5)

| case | reportedly | wall lines | sources |
|---|---|---|---|
| bio-euclid | 10 → **1** | 44 → 29 | 4 GOOD → **1 THIN** ⚠ |
| bio-pythagoras | 1 → 2 | 57 → 37 | 2 → 4 GOOD |
| bio-archimedes | 2 → 1 | 59 → 43 | 8 → 4 GOOD |
| hist-library-alexandria | 1 → 1 | 27 → 24 | 4 → 8 GOOD |
| hist-parallel-postulate | 43 → 19 | 58 → 76 | 7 → 8 GOOD |
| current-rust-async | 10 → 22 | 68 → 68 | 8 PARTIAL → **1 THIN** ⚠ |
| current-2025-recent | 2 → 14 | 23 → 43 | 8 → 5 GOOD |
| tech-transformer-attention | 30 → **17** | 48 → 23 | 4 → 3 GOOD |
| controls | 0 / 0 | 0 | clean |

Reading the deltas honestly:

- **transformer 30→17, wall 48→23** — the remap rescued glm's claims (the w1
  block states plainly now), but only partially met the "single digits" target
  because deepseek and qwen orphaned their refs via NEW shapes (below).
- **euclid 10→1 is NOT a clean win** — grounding collapsed (two of three
  workers lost their web evidence, see the compaction finding), so the writer
  used an explicit two-part structure ("what is known with confidence" /
  "unconfirmed details") that reads very well but isn't the same mechanism
  under test.
- **rust-async 10→22 and 2025-recent 2→14 are correct behavior**, not
  regressions: rust had 1 THIN source (all three workers ungrounded — SearXNG
  rate-limit narrations in the notes) and 2025-recent hit five fact-check
  contradiction fatals (Llama 4 vs Llama 3.3). Hedging ungrounded and
  contradicted claims is the system working.
- **postulate 43→19** — still the parked third mechanism (deepseek emitted
  41 sourceless training-data claims → hedge-by-default), improved because its
  other workers were better grounded this run.

## Fact-check channel — best run to date

- **The euclid abort did NOT recur**: verdicts block present (8 checks,
  2 drop / 5 hedge) with real catches — the first-English-translation
  correction (Henry Billingsley 1570, Dee wrote the preface) and the wrong
  Arabic title dropped.
- **archimedes: 11 checks, 6 drops** — dropped the "Gauss solved the Cattle
  Problem" fabrication (answer now says Amthor 1880 ✓), dropped an invented
  Marcellus quote, moved the π bounds to *Measurement of a Circle*, fixed
  Heiberg's nationality/dates, fixed the Archimedean-solids definition. All
  landed in the final answer.
- **transformer: dropped both false lineage claims** (visual-attention /
  pointer-network origins) — absent from the answer ✓.
- **Contradiction fatals handled gracefully**: 2025-recent's answer explicitly
  surfaces the Llama 4 vs Llama 3.3 conflict, names which researcher found
  what, and recommends primary-source verification — exemplary behavior on
  conflicting evidence.
- **pythagoras + library: no verdicts block again** ("NO CORRECTIONS" with no
  checks) — same unclassified shape as before. **The Stage-1 box greps remain
  outstanding** and would classify these too.

## NEW findings

**1. New orphan-ref shape: `src`-style refs against backfilled `s{N}` ids.**
qwen twice emitted claims citing `w2_SRC-1…4` (euclid) / `w2_src_1,_2`
(transformer) / `w1_src_3,_5` (pythagoras) while its sources carried backfilled
`s1…sN` ids. `"src-2".lower()` ≠ `"s2"`, so the repair can't catch it — these
claims fell to hedge despite having sources. Cheap deterministic extension:
when building the alias map, also register `src{N}`/`source{N}` normalized
aliases for each backfilled `s{N}` id (observed-shape-only, pinned). deepseek's
author-year mnemonics (`w0_bahdanau2014` → source titled "Neural Machine
Translation…") remain undeterministically-repairable — that class stays parked.

**2. Cap → tool-pivot → compaction interaction (S5 fallback trigger MET).**
With the web budget spent by round 2–3, workers now pivot to kb/memory/chat
tools for 2–3 MORE rounds (euclid deepseek: ws✅4 then kb✅3 + memory✅1 +
chat✅1). Those extra rounds push the early web results past
`compress_keep_last: 3` — and both euclid deepseek ("search results were
compacted out of the conversation history before I could read them") and glm
("not retained in usable form") lost their web evidence to it. This is the
run-3 starvation pattern in a new costume, and it is exactly the trigger the
plan set for the **S5 fallback lever**: the one-sentence researcher-prompt
nudge to restate key retrieved facts + URLs in each reply, making notes
compaction-proof. (Raising keep_last again would just chase the round count.)

**3. Wrong-domain URL survived into a Sources block.** Euclid's only source is
"MacTutor" pointing at `history.math.ucdavis.edu` (correct domain:
mathshistory.st-andrews.ac.uk). The verifier flagged it in the critique, but
the Sources renderer doesn't consume the critique and nothing validates source
URLs beyond well-formedness. Low priority; noting the gap.

**4. Cap holding everywhere** — all workers ≤4 web_search dispatches across
all 10 cases (qwen even came in under budget at ✅3 twice).

## Recommended next steps

1. **`src{N}`/`source{N}` alias extension** to `_repair_source_links` — one
   small deterministic addition + pins with the three captured run-5 shapes.
2. **S5 fallback nudge** (researcher-prompt: restate key facts+URLs each
   round) — trigger met by finding 2; live-tunable prompt change. Distinct
   observable from #1 (linkage → wall composition; nudge → no "compacted out"
   narrations), so both can ride one eval gate with clean attribution.
3. **Stage-1 box greps** — still pending; now would also classify the
   pythagoras/library no-verdicts shape.
4. Author-year mnemonic repair and the paraphrase-disposition-conflict lever
   stay parked; S4 stays deferred.
