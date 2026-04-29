# Phase 25 — synth context cleanup

**Goal:** make synthesized answers user-friendly and time-aware. Two
small changes to the synthesizer:

- **Drop the `## Approach` preamble.** Synth used to be required to
  open with a meta paragraph explaining how it reconciled drafts —
  useful for debugging the panel, useless for end users. Removed.
  `## Caveats` is now optional and only appears when drafts genuinely
  disagreed on facts (no more `- none` placeholder bullets).
- **Forward the original conversation's system messages into synth.**
  Pre-Phase-25, the synth call ran on a fresh `[synth_system,
  user_with_drafts]` stack — no datetime, no memory recall, nothing.
  That's why a "what year is it" question would get a year-from-
  training-cutoff answer even when workers (which DO see the datetime
  system message) answered correctly. Now synth sees the same
  system-message floor as workers.

What stays the same:

- Worker scheduling, gate behavior, ReAct, deep panel orchestration.
- Phase 18a's `node_datetime` and the streaming `_phase_thinking`
  datetime injection. This phase just makes synth see what they already
  emit.
- The `[tool-grounded: N rounds]` draft tagging — synth still uses it
  to prefer tool-grounded drafts on factual disagreement.
- Reflect's length-and-presence checks (`too_short`, `no_drafts`).

What changed:

- `src/audrey/pipeline/synthesize.py` — `_SYNTH_SYSTEM` rewritten to
  ask for direct prose with optional Caveats; `_build_synth_messages`
  now takes the original `prior_messages` and forwards `role=system`
  entries into the synth call (not user/assistant turns — those are
  represented in the drafts block already).
- `src/audrey/pipeline/reflect.py` — dropped `_REQUIRED_HEADERS`
  (Approach/Answer/Caveats triplet) and the `require_sections` param.
  The synth no longer outputs that fixed structure, so reflecting
  against it would falsely fail.
- `src/audrey/pipeline/graph.py` — drops the `require_sections=True`
  arg from the `reflect_fn` call.

Out of scope (deliberately):

- **Reflect-on-stream.** The streaming path still doesn't reflect
  (once tokens are on the wire we can't un-emit them). That's a
  separate future phase if too-short streamed answers ever become a
  real complaint.
- **Tweaking individual worker prompts.** Workers are unchanged —
  the synth context is the only thing that touches user-facing output.
- **Per-task synth prompt variants.** All five virtual models share
  the same `_SYNTH_SYSTEM`. If we want code-task synth to behave
  differently (e.g. "preserve the strongest code block verbatim, no
  prose at all"), that's a Phase 26+ split.

**Prereqs:** Phase 24 + 24a verified. No env vars, no migrations.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

---

## 2. Smoke tests

### 2.1 Container starts cleanly

```bash
docker compose logs --tail 30 audrey-ai | grep -E 'ready|ERROR|Traceback'
```

Expect: one `ready: ...` line, no errors. If you see a
`TypeError: reflect() got an unexpected keyword argument 'require_sections'`
the Dockerfile build cached the old reflect.py — rebuild with
`--no-cache`.

### 2.2 No more `## Approach` preamble (the headline)

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":false,"messages":[{"role":"user","content":"give me a one-paragraph history of the screwdriver"}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content' | head -20
```

Expect:
- The answer starts with prose directly (e.g. "The screwdriver
  emerged in...").
- **No** `## Approach` heading at the top.
- **No** `## Caveats\n- none` placeholder at the bottom (a simple
  question has nothing to caveat — the section should be omitted
  entirely).

If you DO see `## Approach`, the synth model is being more conservative
than the prompt requests — that's an annoyance, not a bug. Tighten the
prompt language in a follow-up.

### 2.3 Synth knows the current date

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":false,"messages":[{"role":"user","content":"what year is it right now? answer in one sentence."}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content'
```

Expect: a confident answer naming the current year (whatever the
container TZ resolves to — should be 2026 in `America/Denver`).

Pre-Phase-25, the synth model would say "I don't have real-time
data" or hedge to its training cutoff (because the synth call had no
datetime context). Phase 25's prior-messages forwarding is what fixes
this.

To prove the datetime is actually being forwarded, check the synth
prompt size:

```bash
docker compose logs --since 1m audrey-ai | grep -E 'synth: .* ok'
```

The `prompt_eval_count` in the synth response should be slightly
higher than pre-fix (~30-50 extra tokens for the datetime + memory
system messages forwarded from the pipeline). Not a hard threshold —
just sanity.

### 2.4 Caveats appear when warranted

A question with deliberately incomplete evidence:

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":false,"messages":[{"role":"user","content":"give me a year-by-year rundown of the most notable sports moments over the last five years"}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content' | tail -15
```

Expect: probably DOES end with a `## Caveats` section noting that some
years rely on summary recall vs. tool-grounded results. That's the
right time for caveats — there's a real factual gap the user should
know about. Compare to test 2.2's screwdriver question, which has no
factual gap and shouldn't trigger caveats.

This is a soft expectation — the synth might or might not emit
caveats for a given query. The strict requirement is just that it
**doesn't emit `- none`** as a placeholder.

### 2.5 Reflect doesn't fail on missing sections

```bash
docker compose logs --since 5m audrey-ai | grep "reflect:"
```

Expect: `reflect: attempt=1 passed=True reason=ok` for the test
queries above. **No** `reason=missing_sections` lines (that path
doesn't exist anymore).

If you see `reason=too_short`, check that the synth model produced
real content — that path still exists for genuinely-short answers.

### 2.6 Streaming path also drops Approach

```bash
curl -sS -N -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"messages":[{"role":"user","content":"two-paragraph history of TCP/IP"}]}' \
  http://localhost:8000/v1/chat/completions \
  > /tmp/stream25.txt

# Pull out the synth content from the SSE frames
grep -o '"content": "[^"]*"' /tmp/stream25.txt | head -5
```

Expect: the first content delta after the `BANNER_SEPARATOR` is the
start of the answer (prose), NOT `## Approach`. The streaming path
uses the same `_build_synth_messages` and the same synth prompt, so
this should "just work" — the test is mostly to catch the case where
streaming and non-streaming diverged unintentionally.

### 2.7 Deep-panel non-regression

```bash
time curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":false,"messages":[{"role":"user","content":"give me a 200-word comparison of ZFS and BTRFS"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/local_deep25.json
```

Expect: similar wall-clock to Phase 23's baseline (~150-180s for
`audrey_local`). Synth should still complete in ~50-60s.

---

## 3. Rollback

```bash
git checkout <previous-sha> -- \
  src/audrey/pipeline/synthesize.py \
  src/audrey/pipeline/reflect.py \
  src/audrey/pipeline/graph.py
docker compose up -d --build audrey-ai
```

Three-file revert. No config / env / schema changes.

---

## 4. Operational notes

- **If users report "answers feel less structured":** the old
  `## Approach` / `## Answer` / `## Caveats` triple gave a clear
  visual rhythm. Some users may have been parsing it. Phase 25 trades
  that for prose-first delivery — if a particular user/usecase wants
  structured output, ask them to put the structure in their own
  prompt ("Answer in three sections: Approach, Answer, Caveats").
- **The synth prompt is shared across all five virtual models.** Any
  follow-up tweaks happen in `_SYNTH_SYSTEM` in `synthesize.py`. The
  per-task differences live in worker pools (`deep_panel_*`, `fast_path`)
  and synthesizer model selection (`pick_synthesizer`), not the prompt.
- **Token cost from forwarding system messages:** ~30-50 extra prompt
  tokens per synth call (datetime ~25 tokens + memory recall block when
  present). Negligible compared to the drafts block (~2000-8000 tokens).
- **OWUI's `{{CURRENT_DATETIME}}` template variable:** if you have it
  set in OWUI's per-model system prompts, that ALSO flows into synth
  now (it's a system message in the original conversation). Audrey's
  server-side datetime is still the load-bearing one; OWUI's is
  complementary.
