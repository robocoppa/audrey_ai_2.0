# Run-4 gate assessment — 2026-07-07 (post S0-cap / S2 / S3 / S5 / S6 deploy)

Answers: [`2026-07-07-audrey_research-onbox-answers.md`](2026-07-07-audrey_research-onbox-answers.md)
Baseline: run 3 ([`2026-07-06-audrey_research-onbox-run3-report.md`](2026-07-06-audrey_research-onbox-run3-report.md))
Plan under test: [`docs/plans/research-run3-followups-plan.md`](../plans/research-run3-followups-plan.md)

**Verdict: gate passed.** 10/10 cases green on all applicable checks, every gate
expectation met or explained. The search cap is verified at scale, the S2
prompt discipline collapsed hedge density on five of eight research cases, and
the run surfaced **two new source-linkage bugs** (verified in code) that fully
explain the two remaining hedge-soaked answers. Both are deterministic,
small-fix territory.

## Scorecard vs gate expectations

| Expectation | Result |
|---|---|
| ≤4 `web_search` per researcher / ≤15 Brave-able calls per case | ✅ **verified** — all 22 worker footer lines ≤4 dispatches; failures count against budget (2025-recent deepseek `✅2 ❌2` = 4 dispatched, then stopped) |
| Grounding recovered (run-3 had sourceless current-* cases) | ✅ all 8 research cases have Sources blocks — 7 GOOD, 1 PARTIAL |
| euclid "reportedly" density back toward single digits | ✅ 20 → **10** (trajectory 2→9→20→10) |
| Walls shrunk / no duplicate lines | ◐ **zero exact duplicates** in every wall (S3 works as built), but walls still 23–68 lines — cross-worker *paraphrases* differ textually and pass exact-dedup by design; see linkage bugs below for the other driver |
| No compaction refusals (S5, keep_last 3) | ✅ zero "omitted/compacted" notes; postulate deepseek ran 5 tool rounds and wrote full notes (run-3 refusal case) |
| Watch for UNDER-hedging | ✅ clean — royal-road / golden-thigh / Eureka anecdotes still hedge; both controls at 0 hedge phrases |
| Eval ConnectError retry (S6) | not exercised (no connection drop this run) |

## Per-case hedge metrics (answer body only, wall = rendered disposition lines)

| case | reportedly | hedge-phrases | wall lines | exact dups | sources |
|---|---|---|---|---|---|
| bio-euclid | 10 | 19 | 44 | 0 | 4 GOOD |
| bio-pythagoras | 1 | 3 | 57 | 0 | 2 GOOD |
| bio-archimedes | 2 | 2 | 59 | 0 | 8 GOOD |
| hist-library-alexandria | 1 | 3 | 27 | 0 | 4 GOOD |
| hist-parallel-postulate | **43** | 44 | 58 | 0 | 7 GOOD |
| current-rust-async | 10 | 16 | 68 | 0 | 8 PARTIAL |
| current-2025-recent | 2 | 2 | 23 | 0 | 8 GOOD |
| tech-transformer-attention | **30** | 33 | 48 | 0 | 4 GOOD |
| ctrl-birthday-toast | 0 | 0 | 0 | — | N/A |
| ctrl-explain-recursion | 0 | 0 | 0 | — | N/A |

Five research cases at 1–2 "reportedly" is the S2 payoff. The ledger-level
evidence is direct: parallel-postulate's deepseek worker wrote a session-wide
"could not verify sources" caveat and its 41 claims carry **zero** `needs_hedge`
flags (run-3 behavior: wholesale propagation); pythagoras's glm worker
("tool budget exhausted") same. Risk discipline also landed: textbook dates now
`low`/`medium` (archimedes "c. 287–212 BC" = low), `high` reserved for
superlatives/firsts/contested attributions.

## NEW: two source-linkage bugs (verified in code, observed in this run)

The two outlier cases (postulate 43, transformer 30) plus euclid's residual
density are explained by claims **losing their sources after structuring**, which
drops them into `hedge_policy`'s conservative else→hedge branch even when the
backing was authoritative (`reference` types like Wikipedia/Britannica/arXiv DO
qualify for `state_plainly` — `ledger.py` `_AUTHORITATIVE_SOURCES`).

**Bug A — case-variant source ids not repaired.** `_repair_source_links`
(`src/audrey/pipeline/ledger.py:241`) builds its alias map from source **title
and URL only** — never the id itself — so a claim citing `w1_S1` against a
source whose id is `w1_s1` stays unresolved (the exact-id check is
case-sensitive, and `"w1_s1".lower()` isn't in a map keyed by titles/URLs).
Observed: euclid qwen (19 claims cite `w1_S1…S6`, ledger ids `w1_s1…s6`),
pythagoras qwen (16 claims). Fix is one word: add `s.id` to the alias tuple.

**Bug B — merge-time URL dedup orphans claim refs.** `_merge_ledgers`
(`src/audrey/pipeline/deep_panel.py:712`) dedups sources by URL across workers
and even records the canonical id in `seen_urls` — but never **remaps the
dropping worker's claim refs** to it, so those claims cite an id that no longer
exists. Observed: transformer — glm's 27 claims cite `w2_vaswani2017` /
`w2_bahdanau2014` / `w2_luong2015`, all dropped for URL-matching deepseek's
`w0_source-2/-3/-4` arXiv entries → orphaned → hedged → the answer literally
says "QKᵀ reportedly produces a matrix of dot-product similarities" about the
textbook attention formula. Also hit euclid (`w1_s1` Britannica deduped against
`w0_source-2`) and pythagoras (qwen's two SEP links deduped against deepseek's).
Fix: remap refs `dropped→canonical`, fold `supports` into the canonical source,
and prefer a non-`unknown` `source_type` when the copies disagree (the canonical
transformer arXiv source is typed `unknown`; the dropped copy was better typed —
without the type upgrade the remap alone wouldn't rescue those claims).

The two bugs compose on euclid: Bug A repairs `w1_S2→w1_s2`; Bug B is needed for
`w1_S1→w0_source-2`.

**Mnemonic-repair trigger check:** the parked item's trigger ("a run shows it
costing a Sources block or a plain statement") is *technically* met — but the
observed shapes are case-variants and dedup-orphaning, both deterministic. No
true mnemonic refs (`w2_SUDA`-class) appeared this run. Mnemonic repair stays
parked; fix A+B instead.

**Third mechanism, no action:** postulate's 43 "reportedly" is deepseek emitting
41 claims with **no sources at all** (its searches returned nothing usable —
SearXNG-empty pattern, ✅4 footer) → all hedge by policy, correctly. But qwen/glm
covered the same facts with sourced, state-plainly claims — and the writer
followed the hedge lines for the paraphrases (e.g. Saccheri-1733 hedged despite
w1's plain-statement licence). Cross-worker *disposition conflict on paraphrases*
is a real future lever (resolve duplicate-fact dispositions toward the
best-sourced variant); parked — measure again after A+B.

## Fact-check channel

- **Aborts persisted**: euclid and library-alexandria have no verdicts block
  (freeform corrections passed through by fail-soft — content was real and
  useful). Same signature as run 3 (then: euclid + pythagoras). **Stage-1 box
  greps remain the diagnostic**, window now includes this run.
- **New shape**: 2025-recent rendered "0 checks" + `Fatal errors: w1_c5,
  w2_claim_magistral_release` → UNVERIFIED/contradiction corrections → the
  answer explicitly flags the Mistral Large 3 vs Magistral naming conflict.
  Graceful handling, worth keeping an eye on.
- **Real corrections landed**: postulate DROP verdict removed the
  Gerard-of-Cremona-translating-al-Tusi anachronism from the answer; Wallis
  1663-lecture correction integrated; archimedes π-accuracy ("until Ptolemy…
  around 150 AD") integrated verbatim; recursion control's single hedge applied
  word-for-word.
- **Rubber-stamp persists** (known no-op): rust-async 95/95 "supported"
  including claims the verifier flagged as likely fabricated (async-compute).
  Writer dropped async-compute from prose, but its GitHub URL survives in the
  Sources block — cosmetic, noted.
- **Abort-path integration miss**: euclid's freeform corrections CONFIRMED the
  Einstein "holy little geometry book" quote (Schilpp 1949), yet the answer
  hedges it as "lacks direct contemporary verification" — the disposition line
  outranked the prose correction. Minor; only occurs on the abort path.

## Odds and ends

- qwen ran **zero tools** on archimedes and transformer (no footer line, no tool
  rounds) — answered from weights. Pre-existing behavior, not cap-related (cap
  stub never fired for it; it simply didn't call).
- `ttft:0.0s` on postulate and transformer — eval measurement quirk, not a
  latency signal.
- Latency range 283–487s on research cases — no S5 blowup from keep_last 3.
- pythagoras writer still leaks pipeline anatomy ("outlined by your
  researchers") — known, unchanged.

## Recommended next steps (in order)

1. **Fix Bug A + Bug B together** (one eval gate; both are deterministic
   linkage repairs, not prompt churn): id-alias in `_repair_source_links` +
   ref-remap/supports-fold/type-preference in `_merge_ledgers`, each pinned with
   the captured run-4 shapes. Expect: transformer + euclid walls shrink to
   genuinely-uncertain content; "reportedly" on transformer → single digits.
2. **Stage-1 box greps** (still pending, unchanged commands) to classify the
   euclid/library fact-check aborts.
3. **S4 stays deferred** — after A+B, the remaining hedges on ancient-bio topics
   look legitimate; re-measure before touching hedge_policy order.
