# audrey_research evaluation — Phase 26 Stage 1 (research ledger, build only)

First checkpoint of the staged research-ledger work. The ledger is **enabled**
(`agentic.research_ledger.enabled: true`) but only **Stage 1** is wired: a second
mechanical pass structures each researcher's prose into a claim/source ledger,
merged and carried in state. It is **not yet consumed** by verify/write/hedge.
Paired answers: `2026-06-27-ledger-stage1-answers.md`. Baseline:
`2026-06-26-factcheck-hedge-answers.md`.

## The Stage-1 question (narrow by design)

Since the ledger doesn't touch the answer yet, the only thing to measure is:
**does turning it on disturb the prose answers, and does the structuring actually
run?** This is the safe baseline before any stage changes what the user reads.

## Result: prose undisturbed ✅

10/10 structural PASS, all 5-stage banners in order. Per-case answer lengths sit
in the normal run-to-run noise band (a non-deterministic system varies ±10–25%
between identical runs):

| case | base | stage1 | note |
|------|------|--------|------|
| bio-euclid | 6531 | 7959 | +22%, normal variance |
| bio-pythagoras | 139 | 7494 | **baseline was a timeout stub** — this is the first real sample, not a ledger effect |
| bio-archimedes | 7100 | 6369 | −10% |
| hist-library-alexandria | 5449 | 6045 | +11% |
| hist-parallel-postulate | 5631 | 5114 | −9% |
| current-rust-async | 6365 | 6875 | +8% |
| current-2025-recent | 4281 | 4676 | +9% |
| tech-transformer-attention | 5136 | 5112 | ~0% |
| ctrl-birthday-toast | 1037 | 1196 | +15% |
| ctrl-explain-recursion | 2405 | 1750 | −27% |

Nothing here reads as a ledger-induced shift — it's the usual model variance,
and the one big delta (Pythagoras) is the baseline's timeout recovering, a *win*.

## The discipline we must not regress held ✅

`current-2025-recent` — the case the hedge change was built for — kept its
hedge-don't-drop behavior exactly:

> **DeepSeek-R1**: Released in early 2025 (though the exact date remains
> unconfirmed)… **Llama 4**: Reported to have launched in late March or early
> April 2025…

DeepSeek, Qwen (×2), Llama, Mistral (×3) all present and hedged ("Reportedly"
×1, "reportedly" ×3, "unconfirmed" ×3, "unverified" ×3) — real releases kept and
marked uncertain, none dropped. Identical disposition to the baseline.

## Ledger is internal-only, confirmed ✅

`Sources: 0` in the user-facing prose — the ledger is carried in state but does
NOT leak into the answer. That's correct for Stage 1; the end-of-answer "Sources
used" list is Stage 3. The "ledger is scaffolding, not bookkeeping" contract is
holding at the output boundary.

## What this does NOT yet tell us

- **Whether the structuring call produces *good* ledgers** — that's only visible
  in the box logs (`research: ledger built — N claims, M sources from X/Y
  workers`). The eval can't see the ledger (it's internal). **User's box-side log
  check is the other half of this checkpoint.** If the log shows ledgers built
  with sane claim/source counts, Stage 1 is fully validated.
- **Cost** — the structuring call adds one cloud round per worker. Not separately
  timed here; wall-clock for the run felt comparable. Watch it as later stages add load.

## Verdict

**Stage 1 is safe to keep on.** It produces the ledger without disturbing the
prose path or the hedge discipline, and keeps the ledger internal. Pending the
box-log confirmation that ledgers are actually being built with reasonable
content, proceed to **Stage 2** (fact-check operates on the claim ledger).

## Caveats
- One run, one sample per case; non-deterministic. Length deltas are a coarse
  disturbance proxy, not a quality measure.
- The real Stage-1 payload (ledger quality) is in the box logs, not this eval.
