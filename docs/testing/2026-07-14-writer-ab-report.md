# eval report — 2026-07-14 writer route A/B (audrey_research writer: local vs cloud)

Paired with `2026-07-14-writer-local-onbox-answers.md` (arm A) and
`2026-07-14-writer-cloud-onbox-answers.md` (arm B). Both arms: `audrey_research`,
same 5 cases (`eval_prompts_writer_ab.json`), `debug_research_trace: true` so each
arm's staged intermediates are visible. Only variable: `deep_panel_research.*.writer`
= `qwen3.6:35b` (arm A) vs `glm-5.2:cloud` (arm B).

## Decision

**Switch the research writer to `glm-5.2:cloud`.** Committed: the four
`deep_panel_research` `writer:` lines are now `glm-5.2:cloud` in config.yaml.
Cloud wins on latency and prose quality; the token cost is marginal on a route
that already spends cloud on 3 researchers + verifier + factchecker.

## Why (the trace made this a clean A/B)

The `debug_research_trace` dump shows the **hedge dispositions handed to the
writer were the same aggressive list in both arms** — that list is produced by
the `hedge_policy` stage, upstream of the writer. So any difference in the final
prose is the writer alone, not different research. That is what the A/B isolates.

### 1. Latency — cloud faster on every substantive case

| case | arm A (local writer) | arm B (cloud writer) | Δ |
|---|---|---|---|
| science-attention | 248s | 184s | −64s |
| science-plate-tectonics | 290s | 230s | −60s |
| science-mrna-vaccines | 345s | 212s | **−133s** |
| reasoning-race-order | 81s | 90s | +9s |
| gk-element-w | 160s | 131s | −29s |

Cloud is faster on 4/5, by 30–133s. The one regression (race-order, +9s) is the
trivial no-research puzzle where the writer stage is negligible. This confirms
the runbook's hypothesis: the local writer pays a GPU cold-load tail (the final
stage can't co-reside with a local researcher under `GPU_CONCURRENCY=1`); moving
the writer off-box removes it.

### 2. Prose quality — cloud decisively better

Arm A produced undifferentiated wall-of-text. Arm B produced real headers,
**rendered LaTeX** (`$$\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$$`),
comparison tables, and the Q/K/V dictionary-lookup framing with an explicit
"this analogy is a pedagogical simplification" caveat. Arm B's attention answer
is the best explainer produced across any of the 2026-07-14 runs.

### 3. Appropriate confidence — cloud better, not perfect

- gk-element-w arm A: "the element is **reportedly tungsten**" (hedging a
  dead-certain fact). Arm B: "The element with the symbol **W** is **tungsten**"
  — states it plainly. Cloud has better judgment about which hedges to drop.
- But arm B still leaks some: "Pfizer-BioNTech ... **reportedly** received EUA in
  late 2020, though the exact timing details should be confirmed against official
  FDA records" — hedging a fact not in doubt. Neither writer fully rescues this,
  because the root cause is upstream (see the hedge_policy finding below).

## Separate finding: hedge_policy was over-hedging settled facts

The trace exposed a bug independent of the writer choice: the hedge-disposition
lists tagged **~all** claims `HEDGE`, including settled low-risk ones ("softmax is
applied row-wise", "the element with symbol W is tungsten").

**Root cause (verified in source, not inferred):** `hedge_policy` (ledger.py)
states a claim plainly only if its backing source has an *authoritative* type
(`official/primary_paper/scholarly/reference`); everything else falls through to
the conservative `hedge` default (rule 5). But `source_type` is emitted by the
**researcher model**, and the models mislabel authoritative sources as `unknown`
— in this trace the **NeurIPS PDF and arxiv.org/abs/1706.03762 for "Attention Is
All You Need" were tagged `unknown`**, so their claims hedged for lack of an
authoritative type. It's a source-*classification* problem, not a policy-logic
problem.

**Fix shipped (this session):** a deterministic parse-time
`_upgrade_source_types` pass in ledger.py that upgrades an `unknown`-tagged source
to its real type when its URL host is on an unambiguous authoritative domain
(arxiv.org, doi.org, *.neurips.cc, pubmed/pmc, `.gov`, `.edu`, wikipedia.org).
Conservative: only touches `unknown` (never overrides an explicit model choice),
and lookalike hosts (`notarxiv.org`, `arxiv.org.evil.com`) don't match. Sits in
the same parse-time normalizer chain as `_repair_source_links`, so the corrected
types reach `hedge_policy` downstream — the policy logic is untouched. Covered by
8 new tests in test_ledger.py including an end-to-end assert that a low-risk
arxiv-backed claim now returns `state_plainly` instead of `hedge`.

**Not yet re-run live:** the hedge fix is verified hermetically (749 pytest pass,
ruff clean). Its effect on real research answers needs a re-run of the writer_ab
cases after deploy to confirm the settled-fact over-hedging drops.

## Gate

- 749 pytest pass (+8 new ledger tests), ruff clean, lesson-links 0 drift.
- config.yaml: research writer → glm-5.2:cloud (committed default);
  `debug_research_trace` reverted to false.

## Follow-ups

- Re-run `eval_prompts_writer_ab.json` after deploying the hedge fix; compare the
  hedge density on the gk/science cases against this baseline.
- Consider whether the domain allowlist should grow (it's deliberately narrow now).
