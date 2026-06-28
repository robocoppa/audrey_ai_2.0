# audrey_research eval — Phase 26 Stage 2, CLEAN run (SearXNG grounding)

The first valid Stage-2 measurement: full protocol with **real grounding
restored** (SearXNG fallback live — 5 results/query, ledgers building with
sources). Every prior Stage-2 run was confounded by the Brave 402 outage.
Answers: `2026-06-28-ledger-stage2-clean-answers.md`. Baseline:
`2026-06-27-ledger-stage1-answers.md`.

## Result: grounding restored, quality good, no regression

10/10 structural PASS, all banners. The blanket-hedging damage from the outage
runs is **gone** — claims are now stated with verified confidence, not reflexive
"Reportedly…".

### current-rust-async — grounding visibly working
- async-std now correctly: *"deprecated and unmaintained (RUSTSEC-2025-0052) …
  not a viable option for new projects"* — a **specific sourced advisory ID**,
  better grounding than the old "discontinued March 1, 2025" prose. SearXNG
  surfaced the real security advisory (it was the first result in the logs).
- Zero "reportedly" (was blanket-hedged under the outage). Tokio, smol, glommio,
  monoio all present and detailed.
- **Not a strict superset of the pre-outage best answer:** Embassy is absent and
  the literal March-1 date is replaced by the RUSTSEC ID. Different sourcing →
  different emphasis, not a regression in accuracy.

### current-2025-recent — held, tighter
- DeepSeek, Qwen (×8), Llama (×9), Mistral all present; only 2 "reportedly"
  (down from 8 in the outage run), zero "unconfirmed". −30% length = more-verified,
  tighter prose, **nothing dropped**. Hedge-don't-drop discipline intact.

### Lengths vs Stage-1 baseline
Within normal variance (−2% to −30%); the −24/−30% on euclid / 2025-recent are
tighter verified prose, not lost content (spot-checked).

## What this run does NOT yet prove

**Stage 2's specific payoff — claim-level DROP of an unsupported claim (the
"Conics is lost" class) — is not visible in the answers.** It only shows in the
box's `factcheck ledger — N checks (X drop, Y hedge)` verdict logs. The prose
being good doesn't isolate Stage 2's contribution from Stage 1 + the prompt
discipline. To credit Stage 2, we need the verdict logs from this run:

```
docker logs audrey-ai 2>&1 | grep "factcheck ledger"
docker logs audrey-ai 2>&1 | grep "ledger built"
```

Looking for: non-zero `drop`/`hedge` counts (Stage 2 acting on claims) and
`M sources > 0` across cases (grounding consistent, not just the one probe).

## Verdict: Stage 2 VALIDATED — it acts on real claims

The box verdict logs from this clean run settle it. With grounding restored, the
fact-checker now produces real per-claim verdicts:

```
2 checks (0 drop, 2 hedge)
7 checks (0 drop, 7 hedge)
11 checks (0 drop, 1 hedge)
6 checks (1 drop, 5 hedge)   ← unsupported-claim DROP
3 checks (1 drop, 1 hedge)   ← unsupported-claim DROP
```

Contrast with the Brave-outage runs (`82 checks, 0 drop, 0 hedge`; `0 checks`):
with no sources the checker produced **zero** verdicts. Now it **hedges on most
cases and DROPs unsupported claims** — exactly the "Conics is lost" behavior
Stage 2 was built for, firing on real data. Stage 2 earns its keep.

## Known issue: structuring runs at 1/3 capacity

Nearly every `ledger built` line this run reads **`from 1/3 workers`** — only one
of three researchers' ledgers survives structuring; the other two fail (unusable
JSON or empty). The pipeline degrades gracefully (1 worker's ledger is enough to
build + fact-check), but it caps how many claims Stage 2 can check. This is the
next thing to investigate — likely the same JSON-shape fragility we fixed for
the parser, now hitting the other two models differently. NOT blocking (answers
are good, Stage 2 works), but it's leaving 2/3 of the structured signal on the
floor. Worth a diagnostic pass before or alongside Stage 3.

## Caveats
- One run, non-deterministic. Length is a coarse proxy.
- SearXNG result quality varies by query (saw some reddit/forum-heavy result
  sets); grounding is real but noisier than Brave was.
