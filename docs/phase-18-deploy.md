# Phase 18 — streaming progress banners for deep requests

**Goal:** show real-time progress while the deep panel is running, so a 30–90s
`audrey_cloud` request stops looking like a frozen tab. Each pipeline phase
emits a one-line blockquote that grows in place — header on entry, a `.`
appended every 5s, worker checkmarks/crosses interleaved during dispatch,
closing ✅ or ❌ on phase exit.

Why now: phase 17 made deep latency *visible* in metrics. Users still saw
nothing during the wait. This is the smallest UX improvement that pays back
the most. Token-streaming the synthesizer (the natural next step) stays
queued for phase 19.

What the user sees:

```
> _Thinking..._
> _Thinking ✅_
> _Dispatching panel_
> _Dispatching panel.._
> _Dispatching panel...  ✅ kimi-k2.6:cloud_
> _Dispatching panel....  ✅ kimi-k2.6:cloud  ✅ glm-4.5:cloud_
> _Dispatching panel.....  ✅ kimi-k2.6:cloud  ✅ glm-4.5:cloud  ✅ qwen3.6:35b_
> _Dispatching panel ✅_
> _Synthesizing_
> _Synthesizing._
> _Synthesizing ✅_

---

## Approach
Both drafts converged on...
```

What changed:

- **`src/audrey/pipeline/banners.py` (new)** — `PhaseTicker` async context
  manager, banner header strings, and worker_ok / worker_fail formatters.
  The ticker spawns a background task that pushes a `.` onto a bounded
  asyncio queue every 5s. The route generator drains the queue between
  awaits and yields SSE deltas. Slow consumers backpressure the ticker,
  never the model.
- **`src/audrey/pipeline/deep_panel.py`** — new `run_panel_streaming()`
  variant of `run_panel`. Same scheduling (parallel, gate-bounded for
  local) — `asyncio.as_completed` only changes *reception* order so we
  can banner per worker as it finishes. Total wall-clock time is identical
  to `run_panel`.
- **`src/audrey/routes/openai.py`** — `_stream_deep_with_banners()` replaces
  the old "run graph, emit one chunk" deep streaming branch. Sequentially
  drives memory_recall → planner → run_panel_streaming → synthesize, with
  a banner per phase. Non-streaming requests still use the compiled graph
  unchanged.
- **Virtual-model lineup expanded** to five names with strict semantics:
  `audrey_deep` (always deep, mixed pool), `audrey_cloud` (always deep,
  cloud pool), `audrey_local` (always deep, local pool), `audrey_auto`
  (adaptive — fast for short prompts, deep for long), and `audrey_fast`
  (always fast, no escalation). Previously `audrey_deep` was adaptive;
  it's now strict so the name matches the behavior. Migration: any OWUI
  alias that pointed at `audrey_deep` and expected fast short responses
  should switch to `audrey_auto` or `audrey_fast`.

What stays the same:

- Non-streaming `audrey_deep` / `audrey_cloud` / `audrey_local` — identical
  behavior, identical metrics. The compiled graph is untouched.
- Streaming fast-path requests — the original token streaming still runs
  for non-deep prompts.
- All Phase 17 metrics — `audrey_pipeline_seconds`, `audrey_dispatch_total`,
  `audrey_model_seconds`, `audrey_gpu_gate_wait_seconds`, etc. fire at the
  same sites. `_stream_deep_with_banners()` observes pipeline metrics
  manually since it bypasses the graph.

Out of scope (deliberately):

- **Synth token streaming.** Phase 19. The synthesizer still runs to
  completion, then emits as one chunk after the separator.
- **Banners on fast-path requests.** Fast-path is sub-3s; banners would
  close before the user could read them.
- **Banners on the non-streaming endpoint.** Programmatic clients hit
  `_generate_via_pipeline` and get a single JSON blob — banners would
  pollute the content field.
- **Reflect retry loop in streaming.** Once we've started emitting tokens
  we can't take them back. If the panel produces weak drafts, the
  streaming path runs synth on whatever it has and the user sees a worse
  answer once. Non-streaming requests still get the retry loop.
- **Configurable banner verbosity.** Default "show all" until there's a
  user-driven reason to dial it back.

**Prereqs:**

- Phase 17 verified.
- OWUI rendering markdown blockquotes (it does — that's how it formats
  system messages already).

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 20 audrey-ai | grep ready
```

No new env vars. No image-level dependency changes (no Dockerfile edit
required this time — the new code only uses stdlib `asyncio`).

---

## 2. Smoke tests

### 2.1 Curl — see the line grow in real time

`curl -N` disables output buffering so you watch the banner deltas land
as they're sent, instead of all at once at the end.

```bash
curl -N -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":true,"messages":[{"role":"user","content":"compare React and Vue in detail, with a focus on state management"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions
```

Expected: a stream of `data: {…}` SSE frames. The `delta.content` fields
spell out, in order:

1. `> _Thinking_` immediately
2. Possibly one or two `.` after 5s / 10s
3. ` ✅\n` when memory recall + planner finish
4. `> _Dispatching panel_`
5. Periodic `.` plus `  ✅ {model}` (or `  ❌ {model}` on a failure) as
   each worker completes
6. ` ✅\n` when the panel completes
7. `> _Synthesizing_` plus `.` ticks
8. ` ✅\n` when synth finishes
9. `\n\n---\n\n` separator
10. The full synthesized answer as one chunk (phase 18 — streamed
    chunk-by-chunk in phase 19)
11. The final stop frame and `data: [DONE]\n\n`

If you see steps 1–11 in roughly that order with visible delays between
the worker checkmarks, banners are working.

### 2.2 OWUI visual check

Send the same prompt from the OWUI chat UI. The blockquoted progress lines
should render distinct from the answer body (typically italicized, slightly
indented, possibly grey). If OWUI collapses adjacent blockquote chunks
into one rendered block, that's fine — the *line* growing in place is the
desired effect.

If OWUI strips the blockquote markers or renders them as plain text, the
markdown renderer in OWUI is configured oddly — log a follow-up but don't
roll back; the content is still readable.

### 2.3 Worker failure banner

Hard to trigger on purpose without breaking your environment. The code path
is exercised in `_stream_deep_with_banners`'s call to `run_panel_streaming`:
when a worker raises `OllamaError`, `_run_one_worker` catches it and returns
a `WorkerDraft` with empty content + `error` set, which the streaming wrapper
classifies as `ok=False` → `worker_fail()`. Confirm by code-reading
[deep_panel.py:178-185](src/audrey/pipeline/deep_panel.py#L178-L185) once.

To force one in production for verification, take one cloud model offline
in `config.yaml` (mark `enabled: false`) for the deep-panel pool, redeploy,
and watch for ❌ in the dispatching banner.

### 2.4 Virtual-model routing matrix

Verify each virtual model picks the expected mode. The four cases below
exercise both forced-mode bypasses and the adaptive complexity gate.

The **complexity gate fires at 500 tokens** (`config.yaml:complexity.token_threshold`).
A prompt that *asks for* an 800-word essay is only ~70 tokens itself — not
long. To trigger `mode=deep` on `audrey_auto` you need to actually paste
~500+ tokens of content. The drill below uses
`tr -dc '[:print:]' </dev/urandom | head -c 3000` to fabricate that.

Run the four curls one at a time on Unraid (or any laptop with
`$ADMIN_TOKEN` exported):

```bash
# 1. audrey_deep with a trivial prompt — should be deep (forced)
curl -sS -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions > /dev/null

# 2. audrey_fast with a long pasted prompt — should be fast (forced, no escalation)
LONG=$(tr -dc '[:print:]' </dev/urandom | head -c 3000)
curl -sS -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg c "summarize this: $LONG" '{model:"audrey_fast",stream:true,messages:[{role:"user",content:$c}]}')" \
  https://chat.builtryte.xyz/v1/chat/completions > /dev/null

# 3. audrey_auto with a trivial prompt — should be fast (under 500 tokens)
curl -sS -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_auto","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions > /dev/null

# 4. audrey_auto with the same long pasted prompt — should be deep (over 500 tokens)
curl -sS -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg c "summarize this: $LONG" '{model:"audrey_auto",stream:true,messages:[{role:"user",content:$c}]}')" \
  https://chat.builtryte.xyz/v1/chat/completions > /dev/null
```

Then check the audrey log:

```bash
docker compose logs --tail 200 audrey-ai 2>&1 | grep 'chat.completions (stream)'
```

Expected (in order — one line per curl):

```
chat.completions (stream) model=audrey_deep ... tokens=1   ... mode=deep
chat.completions (stream) model=audrey_fast ... tokens=900 ... mode=fast
chat.completions (stream) model=audrey_auto ... tokens=1   ... mode=fast
chat.completions (stream) model=audrey_auto ... tokens=900 ... mode=deep
```

(Token counts will vary based on the random content — `tokens=` for the
two long prompts should be well above 500.)

Failure-mode interpretation:

- `audrey_deep` shows `mode=fast` → `node_complexity` didn't update;
  the forced-deep set in [graph.py](src/audrey/pipeline/graph.py) is
  missing `audrey_deep`.
- `audrey_fast` long-prompt shows `mode=deep` or you see an
  `escalate: fast→deep` line right after it → escalation-suppression
  in `route_after_fast_path` didn't land.
- `audrey_auto` long-prompt shows `mode=fast` despite high token count
  → check `tokens=` in the log; if it's under 500 the prompt didn't
  paste through. Re-run with a longer `head -c` value.

Also confirm `/v1/models` lists all five:

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

Expected: `"audrey_deep"`, `"audrey_cloud"`, `"audrey_local"`, `"audrey_auto"`,
`"audrey_fast"` (in some order).

### 2.5 Performance regression check (most important)

The whole point of this phase was to add visibility *without* slowing
anything down. Compare a non-streamed and a streamed run of the same
prompt and watch the metrics:

```bash
PROMPT='{"model":"audrey_cloud","stream":false,"messages":[{"role":"user","content":"compare React and Vue, focus on state management"}]}'

# Snapshot metrics before
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep -E '^audrey_(pipeline_seconds_sum|model_seconds_sum)' \
  | grep -E 'mode="deep"|model="' > /tmp/m-before.txt

# Non-streaming run
curl -sS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "$PROMPT" https://chat.builtryte.xyz/v1/chat/completions > /dev/null

# Streaming run (note stream:true)
PROMPT_STREAM='{"model":"audrey_cloud","stream":true,"messages":[{"role":"user","content":"compare React and Vue, focus on state management"}]}'
curl -N -sS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "$PROMPT_STREAM" https://chat.builtryte.xyz/v1/chat/completions > /dev/null

# Snapshot after
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep -E '^audrey_(pipeline_seconds_sum|model_seconds_sum)' \
  | grep -E 'mode="deep"|model="' > /tmp/m-after.txt

diff /tmp/m-before.txt /tmp/m-after.txt
```

The deltas should split roughly evenly between the two runs — one
non-streamed contribution + one streamed contribution per metric. If
streaming adds significantly to `model_seconds_sum` (more than a few
percent over non-streaming), that's the canary that the queue / ticker
is somehow blocking model work — investigate.

### 2.6 Disconnect mid-stream

Verify cleanup. From a separate terminal start a long run, then ^C it
during the dispatch phase:

```bash
curl -N -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":true,"messages":[{"role":"user","content":"write a 1500-word essay about cycle racing"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions
# ^C during dispatch
```

Then on Unraid, check audrey logs:

```bash
docker compose logs --tail 50 audrey-ai | grep -E "stream deep|tail queue|banner"
```

There should be no orphan-task warnings and no "tail queue full" lines
piling up. The ticker tasks get garbage-collected when the route generator
is cancelled. If you see a flood of warnings, the cleanup path needs
attention.

---

## 3. Rollback

Pure additive at the route level — the deep streaming branch is the only
behavior change. Roll back by reverting the wiring commit:

```bash
git checkout <previous-sha> -- src/audrey/routes/openai.py
docker compose up -d --build audrey-ai
```

`pipeline/banners.py` and `pipeline/deep_panel.py`'s new `run_panel_streaming`
are inert without the route wiring; leaving them in place is harmless. The
non-streaming graph and every other path are unchanged.

---

## 4. Follow-ups (not Phase 18)

- **Phase 19 — synth token streaming.** Replace the single-chunk synth
  emission after `BANNER_SEPARATOR` with a streamed `chat_stream`. Most
  of the structure is already here.
- **Configurable banner verbosity.** A `?banners=off` query param or
  `audrey-banners: off` header for programmatic clients that want clean
  output. Today it's all-or-nothing.
- **Banner rendering check in OWUI.** If OWUI's markdown collapses
  adjacent blockquote deltas in a way that loses the in-place feel,
  switch to a single-line non-blockquote format with explicit padding.
- **Reflect retry in streaming.** Today's tradeoff: streaming forfeits
  the reflect → retry loop. If a deep run produces poor drafts, no
  retry. Could keep the retry by emitting "↻ retrying" banner and
  re-running the dispatch phase, but the UX of a redo mid-stream is
  awkward — leave for now.
