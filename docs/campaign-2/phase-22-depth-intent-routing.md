# Campaign 2 Phase 22 — Depth-intent routing on `audrey_auto`

Makes `audrey_auto` send **short-but-demanding** prompts to the deep panel.
Before this, the deep-vs-fast decision was purely prompt *length* (≥ 500
tokens), so a 30-token "give me a comprehensive deep dive, think hard" looked
trivial to the gate and went fast — a single-model answer to a question that
wanted multi-draft synthesis.

## What you saw

Two short prompts went to fast when they clearly wanted depth:

```
List the top 3 most influential mathematical theorems … concise but
thorough … Think hard about this                          → fast (32 tokens)

Give me a comprehensive deep dive into the lost works of philosophy …
                                                          → fast (short)
```

The complexity gate (`is_complex`) only compares token count to
`complexity.token_threshold` (500). It never looks at *what the prompt asks
for*, so "short prompt, large task" always slipped through to fast.

## What it does

`audrey_auto` now goes deep if the prompt is long **OR** its latest user
message contains an explicit depth cue. A new `has_deep_intent()` check
matches a configurable phrase list against the latest user turn
(case-insensitive substring), and both routing gates OR it into the deep
decision:

```text
audrey_auto → deep  iff  tokens ≥ token_threshold  OR  has_deep_intent(...)
```

Only `audrey_auto` is affected. `audrey_fast` still forces fast,
`audrey_deep/cloud/local` still force deep, OWUI utility tasks and image turns
still force fast — the depth check sits in the same `audrey_auto`-only branch
the token check was in.

## Why phrases, not "lower the threshold"

Length is the wrong lever here — dropping `token_threshold` would send lots of
genuinely-simple short prompts to deep too (slow, burns cloud quota). The
missing signal is *intent*, so we match intent directly. The phrase set is
kept to **unambiguous** depth cues — "analyze"/"compare" alone are too common
(they'd send casual one-liners to deep), so they're deliberately excluded.

## What's in scope

- **[`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py)** —
  new `has_deep_intent(messages, phrases)`: inspects only the latest user turn
  (mirrors `is_owui_task_request`), case-insensitive substring match, empty
  phrase list disables it.
- **[`src/audrey/routes/openai/pipeline.py`](../../src/audrey/routes/openai/pipeline.py)**
  (streaming gate) and
  **[`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)**
  (`node_complexity`, non-streaming gate) — both compute `deep_intent` and use
  `complex_ or deep_intent` for `audrey_auto`. Kept in sync (the two gates are
  a documented don't-drift pair). The deep log line gains a `deep_intent=1`
  marker when a short prompt escalated on intent alone; the graph's reason
  string is `deep_intent`.
- **[`config.yaml`](../../config.yaml)** — new `complexity.deep_intent_phrases`
  (10 conservative phrases: "think hard", "deep dive", "in depth", "in-depth",
  "comprehensive", "step by step", "step-by-step", "thorough", "exhaustive",
  "detailed analysis"). Empty the list to turn the feature off.
- **[`tests/test_complexity.py`](../../tests/test_complexity.py)** — +7 tests,
  incl. both reported prompts, the casual-prompt negative, latest-turn-only,
  multimodal text parts, and empty-phrases-disables.

## Tuning

`complexity.deep_intent_phrases` is the knob. Add phrases to catch more depth
prompts; remove them if something over-escalates; set the list empty to
disable the feature entirely (back to length-only routing). Substring match,
so "in depth" also fires inside "go in depth here." Watch for false positives —
if a common phrase keeps sending casual prompts to deep, drop it.

## Behavior invariant

When the phrase list is empty, behavior is identical to before (length-only).
A long prompt still goes deep on tokens regardless of phrases. Forced models,
OWUI tasks, and image turns are unchanged — the depth check only governs the
`audrey_auto` length branch.

## Deploy on Unraid

`config.yaml` changed (read at startup). No custom-tools change. From
`/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): **501 pytests pass** (+7); config validates; ruff clean on
the three touched source files. The two reported prompts route deep, a casual
short prompt stays fast (pinned in tests).

Live, on the box:

1. Send one of the reported prompts to **audrey_auto** — it should now show the
   deep banners (`Planning → Dispatching panel → Synthesizing`), not the fast
   `Thinking` line. Confirm the routing reason:

   ```
   docker logs audrey-ai 2>&1 | grep -E "mode=deep .*deep_intent=1|-> deep \(deep_intent\)"
   ```

2. Regression: a casual short prompt ("what's the capital of France") still
   goes fast — no deep banners, no `deep_intent` marker.

3. A long pasted prompt (≥ 500 tokens) still goes deep on tokens (reason
   `tokens>=500`, not `deep_intent`).

## What this unblocks

`audrey_auto` stops under-serving short, demanding prompts: "think hard /
comprehensive / step by step / thorough" now reach the multi-draft deep panel
without the user having to manually switch to `audrey_deep`. Follows from the
two fast-path-misroute reports (2026-06-24).
