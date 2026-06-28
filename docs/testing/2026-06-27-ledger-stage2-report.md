# audrey_research evaluation — Phase 26 Stage 2 (claim-level fact-check)

Full protocol run against the Stage-2 build (fact-check operates on the claim
ledger). Paired answers: `2026-06-27-ledger-stage2-answers.md`. Baseline:
`2026-06-27-ledger-stage1-answers.md`.

## ⚠️ This run is NOT a clean Stage-2 measurement — read this first

It ran while **Brave Search was 402'd (quota exhausted) and the DuckDuckGo
fallback was not yet deployed.** So the fact-checker's `web_search` calls were
failing throughout — it could not verify anything. The effects below are
dominated by **degraded grounding**, not by Stage 2's claim-checking. A valid
Stage-2 measurement needs the DDG fallback (or a renewed Brave key) deployed
first, then a re-run. Recording this run anyway because the confound itself is
the finding.

## What the run shows

10/10 structural PASS, all 5-stage banners. Per-case answer lengths vs Stage 1:

| case | s1 | s2 | Δ |
|------|----|----|----|
| current-rust-async | 6875 | 4747 | **−31%** |
| current-2025-recent | 4676 | 3398 | **−27%** |
| hist-library-alexandria | 6045 | 5153 | −15% |
| bio-euclid | 7959 | 6312 | −21% |
| (others) | — | — | ±4–67%, normal variance |

The two **current-facts** cases shrank most — exactly the cases that lean hardest
on live web verification, and exactly where a dead search backend hurts.

## The grounding loss is visible in the prose

- **current-rust-async:** lost real content. **Embassy** (a leading runtime,
  present in every prior run) is **gone**; the **async-std discontinuation date**
  ("officially discontinued March 1, 2025" — a clean win in the +hedge and
  Stage-1 runs) is **gone**. Everything is now blanket-hedged to "Reportedly…"
  because the fact-checker could verify nothing. This is search-outage damage,
  not Stage-2 dropping unsupported claims.
- **current-2025-recent:** hedge discipline HELD (DeepSeek ×5, Qwen ×5, Llama,
  Mistral all present and hedged; "reportedly" ×8; nothing dropped). The −27% is
  tighter hedging, not omission — good — but again the fact-checker is hedging
  because it's blind, not because it checked.

## What we CAN conclude

- **No regression from Stage 2's code path** — the structured fact-check runs,
  fails soft, and the answers are coherent. Stage 2 didn't break anything.
- **The hedge-don't-drop discipline survives** even under a dead search backend
  (current-2025-recent kept every real release, hedged).
- **We cannot yet credit Stage 2 with catching unsupported claims** — with
  search down, the fact-checker had no sources to check against, so the
  "unsupported → DROP" behavior couldn't fire on real data.

## Verdict

**Inconclusive for Stage 2 — re-run required after grounding is restored.**
Deploy the DuckDuckGo fallback (`feat(tools): DuckDuckGo fallback…`, built +
live-tested this session, needs a `custom-tools` rebuild), confirm `web_search`
returns results again, then re-run this protocol. THAT run is the real Stage-2
measurement — specifically: does a bio or history case now correctly DROP/hedge a
claim its sources don't support (the "Conics is lost" class), while
current-2025-recent keeps its releases?

## Caveats
- One run, non-deterministic, degraded backend. Length is a coarse proxy.
- The single clean takeaway is the confound: research quality is gated on a
  working search provider; restore it before judging Stage 2.
