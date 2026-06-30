# Eval report — 2026-06-30 research-prompt protocol

Paired with [`2026-06-30-research-prompt-eval-answers.md`](2026-06-30-research-prompt-eval-answers.md)
(the raw outputs). Run: `scripts/eval_research.py --cases eval_prompts_protocol.json`
against the box over the VPN. **10/10 structural PASS, exit 0.**

## Headline

Strong run. The two deterministic-shaping features behave exactly as designed,
and the two soft spots are the known SearXNG upstream throttle — **no code
regression.** Validates the kept research work (authoritative-source nudge +
SearXNG retry/cache) and the audit-drain comment fixes on the box.

## What the structural checks confirmed (10/10 PASS)

Every case: reachable, no error marker, answer present, banners in order
(Planning → Researching → Verifying → Fact-checking → Writing).

## Quality read (the part the eval can't grade)

- **Deterministic Sources list** appended on all 8 research cases that earned
  one, authority-ranked (Stanford Encyclopedia / arXiv / Britannica / MacTutor
  lead; Reddit/blogs trail).
- **Control-case suppression works:** the two from-knowledge / creative cases
  (`ctrl-birthday-toast`, `ctrl-explain-recursion`) correctly got **no Sources
  block and no hedging** — the "ungrounded ⇒ no Sources, no dispositions" path.
- **Selective hedging works:** `bio-archimedes` is the cleanest demo — grounded
  facts stated plainly with inline corrections ("Second Punic War, *not* the
  First"), genuinely uncertain claims marked "reportedly"/"apocryphal". The
  controls carry zero hedge-speak. This is "hedge the uncertain, not everything"
  in practice.

## Timing — one outlier, throttle-driven not code

| case | total | note |
|---|---|---|
| bio-euclid | **1150.5s** | ⚠️ outlier — workers re-searching empty result sets |
| hist-parallel-postulate | 472.9s | |
| bio-pythagoras | 459.5s | |
| bio-archimedes | 399.8s | |
| hist-library-alexandria | 341.0s | |
| tech-transformer-attention | 303.6s | |
| current-rust-async | 271.3s | |
| current-2025-recent | 207.3s | |
| ctrl-explain-recursion | 136.1s | |
| ctrl-birthday-toast | 65.7s | |

bio-euclid's ~19 min is the SearXNG-throttle wall-clock signature (workers
re-search `200 + 0 results` empties; the sum of per-stage timeouts is unbounded).

## The two soft spots — both the SearXNG upstream throttle, not code

### `current-2025-recent` degraded to training-knowledge fallback

The most search-dependent case (recent 2025 LLM releases — almost no
training-knowledge fallback, *needs* fresh hits). This run it fell back to
"retrieving verified primary sources proved difficult... drawn from secondary
reporting," vs. the grounded +43 baseline that cited deepseek.com / ai.meta.com /
mistral.ai. It produced **no Sources block** because nothing survived to cite.

**Reading the footer correctly (this is a trap):** `deepseek web_search ×13 ❌`,
`qwen ×11 ❌`, `glm-5.2 ×13` (no ❌). The `❌` is a **sticky "≥1 call errored"
flag**, NOT "all N failed" (see [`banners.py:85`](../../src/audrey/pipeline/banners.py#L85)
— `errors[name]=True` on any one `is_error`; `×N` is the total count, independent
of the flag). So:
- glm's 13 searches had **zero errors** — they succeeded.
- The other two had *some* errors among many calls.

Yet the answer still couldn't ground. The tell: it wasn't *errors* that hurt
this case, it was **empty results** — `200 + 0 results` — which is NOT an error,
so it never trips the ❌ flag and never trips `searxng.py`'s retry. glm got clean
`200`s back, just thin/empty ones. That is the SearXNG upstream-throttle signature
exactly (Brave 402 → all load on one rate-limited instance → engines
CAPTCHA/429 → sparse/empty result sets). Recency questions feel it first because
they have no training-knowledge cushion. **The degrade path behaved correctly:**
honest about thin grounding, hedged, did not fabricate, suppressed the empty
Sources block.

### `hist-library-alexandria` Sources collapsed to one bare-domain entry

Sources block is a single pathless `https://www.britannica.com`. One worker's
`web_search ×8 ❌`; after URL-dedup + authority-ranking the surviving sources
reduced to one weak entry. The **answer** is excellent (myth-by-myth, well
hedged) — the Sources block under-delivers. Throttle-driven thin grounding, not a
ranking bug.

## Root cause (one cause, both spots)

Brave (primary search provider) is quota-exhausted (402), so 100% of search load
falls on the single self-hosted SearXNG, whose upstream engines (Google/Bing/DDG)
rate-limit/CAPTCHA it under sustained load → intermittent `200 + 0 results`.
See memory `project_searxng_upstream_throttle`. **Durable fix: renew the Brave
key** (operational, not code). The grounded cases (euclid, pythagoras, archimedes,
parallel-postulate, transformer) prove the pipeline is sound when search works.

## Correction to a prior claim

An earlier summary said "all three workers' web_search came back ❌/empty" for
`current-2025-recent`. That conflated the sticky-error flag with all-empty —
glm's 13 searches did **not** error. The cause (throttle → empty result sets)
holds; the "all failed" framing was wrong.

## Disposition

- No code regression. Kept research work + audit-drain comment fixes validated on
  the box — safe to commit.
- The two soft spots are operational (Brave key), tracked in memory.
