# Campaign 2 Phase 23 — Nudge workers to ground factual answers

Makes tool-capable workers verify factual claims with `web_search` instead of
answering from memory. Before this, a worker with confident training-data
recall (e.g. on a well-known figure) would answer directly while another
searched — so one panel could mix grounded and ungrounded drafts.

**Prompt + wiring change. No behavior change when `web_search` is absent.**

## What you saw

A deep "deep dive into the life of euclid" fanned out to three tool-capable
workers; only `deepseek-v4-pro:cloud` called `web_search` (×5). `kimi-k2.6` and
`qwen3.6:35b` answered from parametric memory.

That's the model deciding — Audrey *offers* tools, it doesn't force them, and
different models have different "do I need to look this up?" thresholds. For a
well-documented topic that's fine. The real risk is a *factual* query where the
parametric answer is stale or wrong: two confident-but-unverified drafts can
outweigh one grounded draft at synthesis time. This phase tilts workers toward
grounding.

## What it does

A new tool-gated guidance system message, `WEB_SEARCH_GUIDANCE`, nudges the
model to verify facts (names, dates, events, recent/niche info) with
`web_search` before answering — while explicitly *not* searching things it
plainly knows. It's injected **only when `web_search` is in the live registry**,
exactly like the existing `chat_history_search` guidance, so it costs nothing
when the tool is absent.

This reaches every request that composes context: non-streaming fast + deep
(via the graph's `node_memory_recall`) and streaming deep (via
`_phase_thinking`). See scope note below for the one path it doesn't touch.

## Why a tool-gated nudge, not forced search

Forcing `web_search` on every worker would burn tool calls + latency on trivia
the model obviously knows, and remove the model's judgment. A guidance message
keeps the decision with the model but moves its threshold toward grounding for
claims that actually warrant it. It follows the codebase's established pattern
(`CHAT_HISTORY_SEARCH_SYSTEM`): steer toward a tool, only when the tool exists.

## What's in scope

- **[`src/audrey/pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py)** —
  new `WEB_SEARCH_GUIDANCE` constant + `web_search_guidance` / `web_search_text`
  params on `compose_system_messages` (canonical order: memory → chat-history →
  web-search guidance). New `web_search_guidance` override key in `_PROMPT_KEYS`
  so `agentic.prompts.web_search_guidance` can replace the default.
- **[`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)** —
  `node_memory_recall` gates the guidance on `web_search in tools.by_name`
  (non-streaming fast + deep).
- **[`src/audrey/routes/openai/pipeline.py`](../../src/audrey/routes/openai/pipeline.py)** —
  `_phase_thinking` does the same for the streaming deep path.
- **[`tests/test_prompts.py`](../../tests/test_prompts.py)** — +6 tests
  (flag on/off, canonical order after chat-history, custom text, blank-text
  guard).

## Scope note — the streaming fast path

The **streaming fast** path (`audrey_fast`, or `audrey_auto → fast`) does **not**
get the guidance, because it skips context composition entirely — it goes
straight to the model with the raw messages for minimum latency. That's a
pre-existing design boundary (it skips memory recall too), not a regression.
So: streaming-fast stays lean; everything else (which already pays for context
injection) gets the nudge. If grounding on streaming-fast turns out to matter,
that's a separate, larger change (it'd add a compose step to the lean path).

## Behavior invariant

When `web_search` isn't registered, behavior is identical to before (the flag
is False, nothing is injected). The guidance is one short system message; it
doesn't change routing, tool dispatch, or the synthesizer.

## Tuning

`agentic.prompts.web_search_guidance` overrides the default text if you want it
stronger/softer (same override mechanism as `synthesizer`, `classifier`, etc.).
There's no on/off config flag — it's gated purely on `web_search` being in the
registry. To disable, override the key with whitespace (the composer skips a
blank body), or remove `web_search` from the tool servers.

## Deploy on Unraid

No `config.yaml` change required (the default prompt ships in code). No
custom-tools change. From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): **507 pytests pass** (+6); ruff clean on the three touched
source files; existing prompt/compose tests unchanged and green.

Live, on the box:

1. Re-run a factual deep prompt (e.g. the Euclid one) on **audrey_deep** /
   **audrey_auto**. More workers should now call `web_search` — check the
   per-worker "Tools used" footer for `web_search` under more than one model,
   and the logs:

   ```
   docker logs audrey-ai 2>&1 | grep -E "dispatch: web_search ok|react: round.*tool_calls"
   ```

2. Regression: a clearly-known prompt ("what is 2+2", "define photosynthesis")
   should **not** trigger a flurry of searches — the guidance explicitly says
   don't search what you plainly know. If you see search-spam on trivia, soften
   `agentic.prompts.web_search_guidance`.

3. With `web_search` absent (or removed from tools), no guidance message appears
   and behavior is exactly as before.

## What this unblocks

Deep panels lean toward *grounded* drafts on factual questions instead of
leaving grounding to whichever model happens to decide it needs to search.
Reduces the "two confident-but-unverified drafts outvote one grounded one" risk
at synthesis. Follows from the Euclid-panel observation (2026-06-24).
