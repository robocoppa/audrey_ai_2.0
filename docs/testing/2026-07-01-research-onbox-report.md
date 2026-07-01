# Eval report — 2026-07-01 research protocol, ON-BOX, hedge_policy=true

Paired with [`2026-07-01-research-onbox-answers.md`](2026-07-01-research-onbox-answers.md).
First run of the full `audrey_research` protocol executed **inside a container on
the box** (Phase 27 — `Dockerfile.eval` + `scripts/eval-onbox.sh`), under
`hedge_policy: true`, against the post-Ollama-update stack. **10/10 structural
PASS, exit 0, uninterrupted.**

## Headline

**`hedge_policy: true` is working correctly — the Stage-4 over-hedge regression
that motivated shipping `false` does NOT reproduce.** And this whole exercise's
blocker is gone: the run completed clean end-to-end because it ran on the box,
immune to the laptop connectivity that killed **five** prior attempts (one
mid-run Ollama update, two laptop internet drops, plus two partial runs). The two
ungrounded controls — the deciding canaries we could never capture from the
laptop — finally produced data, and both are clean.

Three separate wins in one run:
1. **Phase 27 validated** — on-box execution eliminates laptop-net failure.
2. **hedge_policy=true canary passed** — no ungrounded over-hedge.
3. **All three of this session's research fixes confirmed on the box** (compaction
   reword, `compress_keep_last` bump, worker-drop warning).

## The deciding test — ungrounded controls (both CLEAN)

These are the exact cases that regressed into blanket-caution mush under the
original Stage-4 `hedge_policy: true` eval. Both are now clean:

- **`ctrl-explain-recursion`** — a clear, confident tutorial (base case, factorial
  walk-through, call-stack/overflow caveat). **Zero hedge-speak** — not one
  "reportedly" / "commonly said". No Sources block, no dispositions (correct: an
  ungrounded answer has nothing to cite). This is the all-hedge-suppression fix
  working under `true`.
- **`ctrl-birthday-toast`** — a warm 3-part toast, no hedging, no sources. Clean.

Verdict: the "hedge the uncertain, not everything" floor holds on ungrounded
turns. The regression is tamed.

## Grounded cases — selective hedging behaves as designed

Hedging discriminates per-claim rather than blanketing:
- **Firm facts stated plainly:** Euclid's 13 books / 1482 Ratdolt edition;
  Archimedes' 2:3 sphere-cylinder; the QKV scaled-dot-product math;
  DeepSeek-V3-0324 dated March 24 2025; Llama 4 dated April 5 2025.
- **Genuinely uncertain claims hedged:** "almost certainly apocryphal" (Euclid
  anecdotes), "almost certainly legendary" (Pythagoras's death / bean-field),
  "likely apocryphal" (Archimedes' Eureka). Correct targets.
- All grounded cases render a proper authority-ranked **Sources** block (7–8 real
  URLs each: SEP, MacTutor, Britannica, arXiv, official DeepSeek/Qwen/Meta docs).

## The two prior bugs — both fixed, confirmed on the box

**1. The "elided" leak is GONE (`current-2025-recent`).** Prior runs narrated
"searches returned sparse or elided results" and produced a sourceless degraded
answer — "elided" being our own ReAct history-compaction stub bleeding into the
model's narration. This run: a rich, dated, well-sourced synthesis (DeepSeek R1
Jan, V3-0324 Mar 24, Llama 4 Apr 5, Qwen3 Apr 28–29, Kimi K2 Jul, gpt-oss Aug)
with 7 sources and **46 successful searches** (`✅15 / ✅13 / ✅18`). No "elided"
appears anywhere in the file. The stub reword + `compress_keep_last: 1→2` worked.

**2. `bio-archimedes` worker-drop — partial recovery + observability confirmed.**
Still only 2 of 3 workers contributed (deepseek + glm; qwen absent), so the drop
isn't fully cured — but two things improved. First, it produced a full answer
with 8 sources this run (vs. the sourceless single-worker answer before). Second,
the new drop-diagnostic is visibly working: the footer shows
`deepseek-v4-pro:cloud — web_search ✅11, kb_search ✅0 ❌1` — a real per-tool
error now surfaced instead of vanishing. Worth a box-log check on the qwen drop
(`research: N/M researchers produced content` line), but it no longer degrades
the answer.

## Latency

| case | total | ttft |
|---|---|---|
| ctrl-birthday-toast | 64.6s | 9.1s |
| ctrl-explain-recursion | 114.2s | 11.0s |
| tech-transformer-attention | 242.8s | 0.0s |
| current-rust-async | 314.4s | 8.6s |
| current-2025-recent | 359.9s | 12.6s |
| bio-pythagoras | 422.2s | 9.3s |
| hist-parallel-postulate | 464.7s | 0.0s |
| hist-library-alexandria | 472.0s | 10.9s |
| bio-euclid | 478.1s | 9.0s |
| **bio-archimedes** | **614.3s** | 9.1s |

Normal for the research pipeline; totals track case depth. No throttle-driven
latency balloon this run (SearXNG returned content readily — see the high
✅-counts on `current-2025-recent`).

## Disposition

- **Phase 27:** validated. On-box eval is the resilient path for long protocol
  runs; the laptop `.venv` path stays for quick `--only` checks.
- **hedge_policy=true:** passed its canary. Safe to keep on the box; **could be
  promoted to the repo default** on this evidence (user's call; one more
  confirming run is the belt-and-suspenders option).
- **This session's three research fixes:** all confirmed live.
- **Open:** the `bio-archimedes` qwen-worker drop (box-log check; the warning
  makes it a one-line find now, not a degraded answer).
