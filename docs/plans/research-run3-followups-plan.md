# Plan — research-mode followups from the 2026-07-06 run-3 assessment

> **STATUS (2026-07-07, post-gate):** deployed and **GATE PASSED** — run 4
> ([report](../testing/2026-07-07-audrey_research-onbox-report.md)): 10/10,
> cap verified ≤4/worker on all 22 worker footers, grounding recovered (8/8
> Sources), euclid "reportedly" 20→10 with five cases at 1–2, zero exact wall
> dups, zero compaction refusals, no under-hedging. The two remaining
> hedge-soaked answers (postulate 43, transformer 30) traced to **two new
> code-verified linkage bugs**: (A) `_repair_source_links` doesn't alias
> source *ids* → case-variant refs (`w1_S1` vs `w1_s1`) unresolved;
> (B) `_merge_ledgers` URL-dedup drops duplicate sources without remapping
> the dropping worker's claim refs. **A+B are BUILT (2026-07-07, laptop,
> awaiting deploy):** id added to the repair alias map; merge now remaps
> dropped→canonical, folds `supports`, and lets a typed duplicate upgrade a
> canonical `unknown` — pinned with the captured run-4 shapes (one eval
> gate). **S1** box greps still pending (euclid + library aborted
> fact-check again). **S4** stays deferred — post-A+B walls look like
> legitimate uncertainty.
>
> _(Previous status: user approved with the S0-cap scope change — Brave-key
> renewal conditional on the per-request cap: research 4/worker, deep 3,
> fast 3, factcheck 3 → ≈15 Brave calls/request, down from 30–45. S2/S3/S5/S6
> shipped with it; mnemonic repair parked.)_

Source: [`docs/testing/2026-07-06-audrey_research-onbox-run3-report.md`](../testing/2026-07-06-audrey_research-onbox-run3-report.md)
(and the +53…+57 PROJECT_STATE entries). Run 3 validated the pipeline and the
July-6 fixes; what remains are quality issues with known mechanics. This plan
addresses them in stages, **one lever per eval run** (the +44/+45 lesson:
two levers on one eval muddies attribution; the +21→+26 lesson: no prompt
churn without a gate).

The issues, ranked by answer-quality impact:

| # | Issue | Mechanics (traced, not theorized) |
|---|---|---|
| 1 | Search starvation (deepseek all-empty; current-events cases sourceless; MiniMax-M1 false date uncatchable) | Brave 402'd → 100% of load on SearXNG → upstream throttle; per-worker clustering unexplained |
| 2 | Hedge-soaked answers ("reportedly" ×20 on euclid; walls of 43–94 lines) | Structuring marks ALL dates/authorship `risk: high` ([prompts.py:159](../../src/audrey/pipeline/prompts.py#L159)) + note-level caveats propagate `needs_hedge` to every claim → wall → writer obeys |
| 3 | Compaction ate a worker's own results (postulate: deepseek refusal) | 5 search rounds, `compress_keep_last: 2` ([config.yaml:293](../../config.yaml#L293)) → early-round evidence stubbed before the final write |
| 4 | Fact-check aborted on euclid + pythagoras (no verdicts section) | Unknown — needs box-log classification before any fix |
| 5 | Mnemonic source refs (`w2_SUDA` vs `w2_s1`) | Third linkage shape; title/URL repair can't resolve; benign so far |

## Stage 0 — Operational grounding recovery (no code; the dominant lever)

Grounding is the root of #1 and the *only* real fix for the
MiniMax-M1-class miss (a verifier can't check a date it can't retrieve).
All user-run, on the box:

1. **Renew the Brave API key** — the documented cure
   (memory `project_searxng_upstream_throttle`): Brave live → SearXNG idles
   as rare fallback → throttle dissolves, latency drops, fact-check gets
   real URLs to bite on.
2. **SearXNG engine health:** `docker logs SearXNG 2>&1 | tail -60` — look
   for `TooManyRequests` / `CAPTCHA` / `suspended_time`. If DDG is the
   repeat offender, disable it in SearXNG `settings.yml` (fix #2 from the
   memory).
3. **Per-worker clustering triage:** during the next run window,
   `docker logs custom-tools --since 1h | grep -c "SearXNG returned 0 results"`
   and eyeball whether the empties cluster in time (burst-start) or by query
   phrasing. One-line finding, decides whether anything code-side is even
   implicated (expectation: no).

**Gate:** re-run the research protocol. Expect: deepseek grounded,
`current-*` cases with Sources blocks, footer ✅-counts backed by real URLs.
Everything below is easier to evaluate on a grounded baseline — run Stage 0
first.

## Stage 1 — Investigate the fact-check aborts (no code until classified)

Euclid and pythagoras produced no verdicts section (fail-soft carried them).
Per the AGENTS.md discipline: look at the raw reply before theorizing.

- `docker logs audrey-ai --since <run window> | grep -E "parse_factcheck|factcheck ledger|unusable"` —
  the Phase-26 diagnostic logging should show whether it's parse-None (new
  JSON shape to tolerate), an empty `checks` list, or `fc_can_run` never
  true for those requests.
- Outcome A (new tolerable shape): extend the `ledger.py` parser + pin with
  the captured shape, same as the fences/bare-array fixes.
- Outcome B (model returned garbage): leave fail-soft alone; it worked.

**Note — deliberately NOT gating fact-check on grounding:** run 3 proved the
verdict channel produces real corrections even with zero usable URLs
(archimedes: 214 BC, Zu Chongzhi). The rubber-stamp runs (42/42, 51/51
"supported") are downstream no-ops — cosmetic trace noise and some latency,
not wrong answers. No action beyond the abort investigation.

## Stage 2 — Structuring risk/needs_hedge discipline (prompt-only, live-tunable)

The main answer-quality fix. Three-run evidence (walls 44/61–105/43–94;
euclid "reportedly" 2→9→20 tracking grounding thinness) — past the
one-sample bar. Two surgical edits to `RESEARCH_STRUCTURE_SYSTEM`
([prompts.py:155](../../src/audrey/pipeline/prompts.py#L155)):

1. **Scope `needs_hedge` to claim-specific doubt.** Today: "when the notes
   themselves marked the claim uncertain" — a researcher's blanket
   "I couldn't verify anything this session" note reads as marking *every*
   claim, which is what built the 94-line rust-async wall. New instruction:
   session-level sourcing caveats ("searches returned nothing", "from
   training data") must NOT set `needs_hedge`; the pipeline already knows
   the grounding state from the sources themselves. `needs_hedge` is for
   claims the notes individually flag (disputed, "possibly legendary",
   "date approximate").
2. **Reserve `risk: high` for genuinely contestable specifics.** Today every
   date/authorship/"first" is high. New: high = recent events, contested
   attributions, single-source specifics, superlatives/rankings, vendor
   claims; a widely-reproduced textbook fact (Ratdolt 1482, Heath 1908)
   is medium even though it's a date. This keeps fact-check targeting
   meaningful (it checks high-risk claims) while stopping the blanket
   hedging of encyclopedia facts.

Files: `prompts.py` (+ byte-for-byte regression test), cite checker after.
Live-tunable via `agentic.prompts.research_structure` if a rollback is
needed without rebuild.

**Gate:** full protocol vs the Stage-0 baseline. Measure: euclid-style
"reportedly" density (expect single digits), wall sizes (expect the walls to
shrink to the genuinely uncertain claims), AND the +37 watch-item in
reverse — **watch for under-hedging** (ancient-biography anecdotes must
still hedge; the controls must stay clean).

## Stage 3 — Disposition wall dedup (deterministic, small)

Independent of Stage 2 and uncontroversial: the walls repeat the same fact
2–3× because three workers each contribute a near-identical claim
("Euclid flourished c. 300 BCE" gets three HEDGE lines).
`_render_dispositions_block` gains a dedup by normalized claim text
(casefold, strip punctuation/whitespace) — first occurrence wins. No cap
yet; revisit a cap (~30 lines) only if post-Stage-2 walls are still large.

Files: `deep_panel.py` + tests (duplicate-claims-collapse,
distinct-claims-kept). Hermetic; ships with any rebuild.

**Gate:** rides along with the Stage-2 eval (render-only, can't change
which claims hedge — attribution stays clean).

## Stage 4 — DECISION POINT: authoritative backing vs `risk: high`

Today `hedge_policy` puts risk-high above authoritative backing (deliberate,
+37), so an official-URL-backed release date still gets
"HEDGE (unless a strong source backs it)" — delegating the call to the
writer. With linkage now repaired, we *could* resolve it deterministically:
`hedge_or_cite_strongly` + authoritative usable-URL backing →
`attribute`/`state_plainly`. This refines a deliberate prior decision, so
it's a user call, and it only matters once Stage 0 restores real URL-backed
linkage. Recommendation: **defer** — evaluate the Stage-0+2 baseline first;
if grounded runs still hedge well-sourced dates, take this then, alone,
with its own eval.

## Stage 5 — Compaction starvation (config-only)

Bump `agentic.react.research_worker.compress_keep_last` 2→3
([config.yaml:293](../../config.yaml#L293)). Restart, no rebuild. The +48
bump (1→2) fixed the "elided" narration; run 3 shows 2 still starves a
5-round worker of its own evidence (deepseek's postulate refusal). Cost:
bigger worker prompts on search-heavy cases — watch per-case latency and
prompt_eval counts in the gate run. Fallback lever if 3 isn't enough
(NOT simultaneously): a one-sentence researcher-prompt nudge to restate key
retrieved facts+URLs in its own reply each round, making its notes
compaction-proof.

**Gate:** postulate-class cases (5-round searchers) — zero
"results were omitted/compacted" refusals; latency delta acceptable.

## Stage 6 — Opportunistic hardening (no eval gate needed)

- **Eval resilience:** one delayed retry (~60s) on `ConnectError` in
  `eval_research.py` — would have saved run 2's cases 4–10 across the OWUI
  restart gap. Pure harness change.
- **Mnemonic source-ref repair** (`w2_SUDA` → `w2_s1` via normalized
  substring match against titles): parked behind a trigger — implement only
  when a run shows it costing a Sources block or a plain statement.

## Explicitly not doing (so it isn't re-litigated)

- **No writer degrade-path loosening** — tried at +44, made answers more
  timid, reverted at +45.
- **No wholesale hedge_policy reorder** — the +37 rule order is deliberate;
  Stage 4 is the one surgical exception, deferred and user-gated.
- **No central/shared search manager** — declined at +39 (the Phase-23
  homogenization mistake in a new shape).
- **No fact-check-on-grounding gate** — see Stage 1 note; the channel earns
  its keep ungrounded.

## Suggested order

**0 → (1 in parallel) → 2+3 (one eval) → 5 → 4 if still warranted → 6
whenever.** Stage 0 is user-ops and unblocks honest measurement of
everything else; 2+3 are the answer-quality core; nothing below 2 should be
tuned against a starved baseline.
