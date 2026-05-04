# Phase 19 — synth token streaming

**Goal:** stream the synthesizer's tokens live to the client instead of
holding the answer back until synth finishes. The Synthesizing banner
runs only until the first synth token arrives; from there, the
separator and the answer body stream paragraph-by-paragraph as the
model produces them.

Why now: synth was the dominant remaining UX gap from phase 18.
Banners cover Thinking / Dispatching nicely, but the third banner
(Synthesizing) ran for ~2–4 minutes of silent dots before any answer
text appeared. That's the same "is this still working?" feel phase 18
was supposed to kill. Streaming synth tokens replaces the silent
window with paragraphs forming live.

What changed:

- **`src/audrey/pipeline/synthesize.py`** — new
  `synthesize_stream()` async generator alongside the existing
  `synthesize()`. Same selection logic (primary → fallback → longest
  draft), same prompt, but emits a sequence of events:
    - `first_token` — the chosen synth produced its first content byte
    - `delta` — a content chunk
    - `fallback_attempt` — primary errored before any tokens; fallback
      starting (informational)
    - `done` — final event with full content + metadata
  Mid-stream errors append `[synth error mid-stream: ...]` to the
  output and emit `done` with `synth_error=stream_truncated`. Pre-token
  errors fall through to the next candidate, exactly as the
  non-streaming path does.
- **`src/audrey/routes/openai.py`** — `_stream_deep_with_banners()`
  now wraps `synthesize_stream` in a producer task. The Synthesizing
  PhaseTicker stays open until `first_token` arrives; once it does,
  the banner closes with ✅, the separator emits, and each `delta`
  is forwarded to the client as its own SSE frame.

What stays the same:

- The non-streaming graph (`pipeline/graph.py` → `node_synthesize`)
  still calls `synthesize()` unchanged. The same selection logic is
  shared — `synthesize_stream` doesn't replace it.
- All metrics (`audrey_pipeline_seconds`, `audrey_pipeline_total`,
  `audrey_dispatch_total{path="synth_primary|synth_fallback"}`,
  `audrey_model_seconds`) fire at the same sites with the same labels.
- The banner protocol on the wire is identical for Thinking and
  Dispatching; only the Synthesizing phase changes — and even there,
  only the *content body* streams differently. The banner header,
  dots, and closing ✅ are unchanged.

Out of scope (deliberately):

- **Streaming for non-streaming clients.** Programmatic callers that
  send `stream=false` go through the LangGraph and the unchanged
  `synthesize()` call. They get the answer in one shot.
- **Streaming for `audrey_fast` / `audrey_auto` short-prompt path.**
  Fast-path tool-capable models still emit one chunk at the end of
  the ReAct loop (mid-stream tool dispatch isn't supported). Only
  the deep panel synth streams.
- **Reflect retries.** The streaming path doesn't run a reflect step.
  Once tokens are on the wire, we can't un-emit them; the only
  reasonable fallback is to surface a partial answer with a note,
  which is what `stream_truncated` does.

**Prereqs:** Phase 18 + 18a verified (banner streaming + datetime
injection working). Open WebUI v0.9.x with **Stream Chat Response: On**
configured per audrey alias (otherwise OWUI sends `stream:false` and
this phase has no effect for that client).

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 20 audrey-ai | grep ready
```

No env vars. No image-level changes. No new dependencies.

---

## 2. Smoke tests

### 2.1 Streaming deep — tokens land before synth finishes

Run on **laptop** with `$ADMIN_TOKEN` exported. Use a real prompt
that exercises the deep panel:

```bash
curl -sS --no-buffer -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":true,"messages":[{"role":"user","content":"give me a thorough comparison of zfs and btrfs for a home server, with concrete tradeoffs"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions
```

What you should see:

1. Role delta frame.
2. `> _Thinking_` banner growing with dots, then ` ✅`.
3. `> _Dispatching panel_` banner with per-worker `✅` / `❌` tail
   marks, then ` ✅`.
4. `> _Synthesizing_` banner with dots, then ` ✅`. **Should close
   in well under 30s** — that's the time-to-first-token, not synth
   total runtime.
5. Separator (`\n\n---\n\n`).
6. **Answer text streaming in chunks** — `## Approach`, then a
   paragraph, then `## Answer`, then more paragraphs. Each chunk
   should land within a few hundred ms of being produced.
7. Final `[DONE]` frame.

If step 4 takes 2+ minutes (the old behavior), the streaming path
isn't kicking in. Check the audrey log for `synth: <model> ok in Xs
(attempt 1, streamed=True)` — if `streamed=False`, the synth
produced no tokens before completion (model issue, not pipeline).

### 2.2 Confirm log shape

```bash
docker compose logs --since 2m audrey-ai | grep -E 'synth|stream deep'
```

Expected lines for a successful deep stream:

```
synth: deepseek-v4-pro:cloud ok in 142.50s (attempt 1, streamed=True)
stream deep done model=audrey_cloud task=general synth=deepseek-v4-pro:cloud outcome=ok elapsed=178.20s
```

The new bit is `streamed=True` in the synth line — that's the marker
that tokens flowed through `synthesize_stream` rather than the
non-streaming fallback.

### 2.3 Pre-token fallback path

Stop the primary cloud synth temporarily by removing the API key (or
just observe it organically — `kimi-k2.6:cloud` historically times
out before producing tokens). When primary errors *before* any
content, the route should silently switch to the fallback synth:

```
synth: <primary> failed in Xs (attempt 1, streamed=False): <err>
synth: <fallback> ok in Ys (attempt 2, streamed=True)
```

User-visible result: a slightly longer Synthesizing banner, then
streaming answer body — no error in the chat itself.

### 2.4 Mid-stream error handling

Hard to reproduce intentionally. The expected behavior if a synth
errors *after* tokens have been emitted:

- Partial content stays on the wire (already delivered).
- A line `\n\n[synth error mid-stream: <err>]` appended.
- Final SSE frame closes normally with `finish_reason=stop`.
- Pipeline outcome metric records `error`, not `ok`.

Log shape:

```
synth: <model> failed in Xs (attempt 1, streamed=True): <err>
stream deep done model=audrey_cloud task=general synth=<model> outcome=error elapsed=Ys
```

### 2.5 Non-streaming path is untouched

Sanity check that the non-streaming graph still works exactly as
before — nothing in `pipeline/graph.py` or `pipeline/synthesize.py`
non-stream path changed:

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","stream":false,"messages":[{"role":"user","content":"one sentence: what is rsync?"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq -r '.choices[0].message.content' | head -20
```

Should return the standard `## Approach / ## Answer / ## Caveats`
synth output in one response, same as before phase 19.

### 2.6 OWUI end-to-end

Open the OWUI chat, pick an audrey alias that points at
`audrey_deep` / `audrey_cloud` / `audrey_local`, and send a real
prompt. Confirm:

- Banners render correctly (each on its own blockquote line).
- The Synthesizing banner closes within ~30s.
- Answer text starts forming live below the separator, not in one
  giant block at the end.

If banners render fine but answer text still appears in one chunk
at the end, check the OWUI per-model **Stream Chat Response** toggle
(Admin Panel → Models → expand → Advanced Params → set to **On**, not
Default). OWUI v0.9.2 sends `stream:false` by default unless this is
explicitly flipped.

---

## 3. Rollback

Phase 19 is two-file:

```bash
git checkout <previous-sha> -- \
  src/audrey/pipeline/synthesize.py \
  src/audrey/routes/openai.py
docker compose up -d --build audrey-ai
```

The non-streaming code path doesn't depend on `synthesize_stream`, so
removing it has no graph-side impact.

---

## 4. Follow-ups (not phase 19)

- **Per-tool dispatch metrics inside ReAct.** Today we see model
  dispatches via `audrey_dispatch_total`, but not which tool fired
  inside the loop. A `audrey_tool_calls_total{tool=...,outcome=...}`
  counter would close the observability gap.
- **Grafana alert on `audrey_pipeline_total{outcome="error"}` rate.**
  With streaming synth, mid-stream errors now flag as `error` instead
  of degrading silently to "longest draft." Worth alerting on a
  sustained nonzero rate.
- **Convert Dockerfile to install from `pyproject.toml`.** Still
  open from phase 18; bit us twice with hardcoded deps lists.
- **Reflect-on-stream.** Out of scope for phase 19, but if the synth
  produces a too-short answer, today we just ship it. A future phase
  could buffer the first N tokens, run reflect, then either stream
  the buffered tokens or restart synth — non-trivial.
