# Campaign 2 Phase 7 - Fast-path Thinking banner

UX-only change. Adds a one-line `> _Thinking_` banner to the streaming
fast-path branch so `audrey_auto` (when it routes fast) and
`audrey_fast` no longer look frozen between request submission and the
first answer token.

Deep routing's first banner is renamed from `Thinking` to `Planning`
so a quick glance at the chat tells the user which branch ran:
`Thinking` → fast, `Planning` → deep. The rest of the deep sequence
(Dispatching panel → Synthesizing) is untouched.

## Why

Phase 6 smoke testing surfaced a UX gap: on `audrey_auto` and
`audrey_fast`, the user sends a prompt and stares at a blank chat
until tokens arrive. On plain chat this is ~200-500ms — quick but
silent. On the tool-capable fast path (model emits `kb_search`, the
ReAct loop runs server-side, then one chunk arrives) it is 1-3s of
silence. Both surfaces feel slow even when latency is normal.

Deep mode never had this problem — its banner sequence covers the
wait. Now the fast path matches.

## Before / after

**Before, fast plain chat (audrey_fast or audrey_auto → fast):**

```text
(user submits)
(silence, 200-500ms)
<tokens stream>
```

**After, fast plain chat:**

```text
(user submits)
> _Thinking_
> _Thinking ✅_

---

<tokens stream>
```

**Before, fast tool-capable (audrey_auto → kb_search round-trip):**

```text
(user submits)
(silence, 1-3s)
<one-chunk answer arrives>
Tools used: kb_search
```

**After, fast tool-capable:**

```text
(user submits)
> _Thinking_
> _Thinking._    (dots appear every 5s while the ReAct loop runs)
> _Thinking ✅_

---

<one-chunk answer>

---
> _Tools used:_
> - **kimi-k2.6:cloud** — `kb_search`
```

**Deep mode (audrey_deep / audrey_cloud / audrey_local / audrey_auto → deep):**

First banner renamed from `Thinking` to `Planning` (Stage 1 covers
memory recall + planner). Stages 2 and 3 unchanged. The full deep
sequence now reads:

```text
> _Planning ✅_       (memory recall + planner)
> _Dispatching panel ✅ kimi-k2.6:cloud  ✅ qwen3.6:35b_
> _Synthesizing ✅_
```

## What changed

**`src/audrey/routes/openai.py`** — `_stream_via_pipeline` only.

- After the `forced_deep` / `use_deep` decision branches into "fast,"
  build a per-request `_fast_delta` helper and yield an OpenAI-spec
  role frame so the SSE stream has a valid leading chunk.
- **Tool-capable fast path** (`tools_active`): open a full
  `PhaseTicker(BANNER_THINKING, ...)` that emits dots every 5s while
  `_run_graph_with_metrics` runs. Close on completion, drain the
  ticker's `✅` frame, then emit `BANNER_SEPARATOR` before the
  one-chunk message. Reuses the `_drain_q_until_task` / `_drain_q_now`
  helpers the deep branch already uses, so the queue + ticker
  lifecycle matches.
- **Plain-chat fast path** (no tools): no `PhaseTicker` machinery — a
  full ticker would emit a header and close immediately because first
  token usually arrives in <200ms. Instead yield three static frames
  in sequence: header, ` ✅\n`, separator. User sees the banner flash
  briefly then tokens start.

**`src/audrey/pipeline/banners.py`** — added a new `BANNER_PLANNING =
"> _Planning_"` constant used by the deep helper's first stage. The
existing `BANNER_THINKING` constant stays put and is now reserved for
the fast branch only.

**`tests/test_banners.py`** — added five constant-pin tests so a
silent rename to `BANNER_THINKING`, `BANNER_PLANNING`,
`BANNER_DISPATCHING`, `BANNER_SYNTHESIZING`, or `BANNER_SEPARATOR`
would fail loudly. These strings are part of the SSE protocol
contract — what the chat UI renders — and must not drift
unintentionally.

## What did not change

- Deep streaming. `_stream_deep_with_banners` is untouched.
- Non-streaming routes (`POST /v1/chat/completions` without
  `stream=true`). Banners are stream-only by design.
- The graph itself. ReAct loop, classify, complexity gate, model
  registry — all the same.
- Wire format outside the banner frames. Tokens still stream
  identically.
- Auth, inflight scheduling, archive write. All untouched.

## 1. Deploy

Local first (laptop):

```bash
git pull
.venv/bin/python -m pytest -q   # 252 should pass (+4 banner-constant tests)
.venv/bin/ruff check src/audrey/routes/openai.py
```

Unraid (from `/mnt/user/appdata/audrey_ai_2.0`):

```bash
git pull
docker compose up -d --build audrey-ai
docker compose logs --since 1m audrey-ai | grep -E "ready|tools="
```

Custom-tools does not need rebuilding — only `src/audrey/routes/openai.py`
and `tests/test_banners.py` changed.

Expected:

- Image rebuilds. Roughly the same size.
- Startup line still shows `tools=7` and the existing readiness shape.
- Old chat conversations open and render normally.

## 2. Smoke tests

This is a UX change. The pytest suite covers the constant pins; the
real test is "does the banner appear in the chat UI." All checks are
manual UI inspection in Open WebUI.

### 2.1 Fast plain chat (audrey_fast)

In OWUI, pick the `audrey_fast` virtual model. Send a short prompt
that won't trigger tools, e.g. "say hello in three words."

Expected sequence in the chat bubble:

```text
> _Thinking ✅_

---

<the three-word reply>
```

The `> _Thinking_` line appears for ~50-200ms before the `✅` closes
it. The horizontal rule (`---`) separates the banner from the reply.

### 2.2 Fast tool-capable (audrey_auto → kb_search)

Pick `audrey_auto` and send a short prompt the classifier will route
through fast with a tool, e.g. "search my KB for BTRFS." On a fresh
conversation, the complexity gate should pick fast (low token
count). The model emits a `kb_search` tool call, the ReAct loop runs
the dispatch, and the answer comes back as one chunk.

Expected:

```text
> _Thinking_
> _Thinking._         (dots every ~5s if the loop takes that long)
> _Thinking ✅_

---

<answer about BTRFS>

---
> _Tools used:_
> - **kimi-k2.6:cloud** — `kb_search`
```

If the ReAct loop is fast (<5s), no dots — just header → close →
separator → answer.

### 2.3 Deep mode renamed first banner (audrey_deep)

Pick `audrey_deep` directly. Send any prompt. Confirm the banner
sequence matches the new shape:

```text
> _Planning ✅_       (memory recall + planner — was Thinking before Phase 7)
> _Dispatching panel ✅ kimi-k2.6:cloud  ✅ qwen3.6:35b_
> _Synthesizing ✅_

---

<answer>

---
> _Tools used:_
> ...
```

Stages 2 and 3 are unchanged from before Phase 7. Only the first
banner's label changed (Thinking → Planning) to disambiguate fast
vs. deep at a glance.

### 2.4 audrey_auto routes deep (long prompt)

Paste a 600+ token prompt with `audrey_auto`. The complexity gate
will route to deep. Confirm the deep banner sequence renders
(`Planning → Dispatching panel → Synthesizing`), not the fast one
(`Thinking`). The first banner being `Planning` is how you can tell
at a glance the auto router picked deep.

### 2.5 Mid-stream cancel

Send a long prompt on `audrey_fast`, then click stop in OWUI before
the answer finishes. The banner should close with `❌\n` instead of
`✅\n`. Behavior is the same as deep's cancel path today.

```text
> _Thinking ❌_
```

If you see `Thinking ✅` followed by a partial answer, that's also
fine — the cancel landed after the ticker closed but before tokens
finished. Either is the new working behavior; neither is the old
"silent freeze."

### 2.6 Non-streaming still works

The non-streaming chat completions endpoint (no `stream=true`) does
not use banners. Smoke-check by hitting `/v1/chat/completions`
directly with curl:

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer <test-jwt>' \
  -d '{"model": "audrey_fast", "messages": [{"role": "user", "content": "hi"}], "stream": false}' \
  | jq .
```

Expected: a normal completion response with a `choices[0].message.content`
of "hi"-shaped output. No banner text in the response body.

## 3. Rollback

Plain git revert. No state, no schema, no data.

```bash
git revert <phase-7-commit>
docker compose up -d --build audrey-ai
```

The previous silent-fast-path behavior re-deploys. No collections or
SQLite changed.

## 4. Operational notes

- **The fast path's role frame now arrives ~50ms earlier** than the
  first token, because it precedes the banner. Clients that buffer
  until they see content should be unaffected — the role frame is
  metadata-only.
- **OWUI's title-generation auto-task** fires a second chat
  completion in the background after the first 2-3 messages. It will
  now also see the new banner frames if it streams. Most title
  prompts route through `audrey_fast` (per Phase 4's task-model
  configuration); the banner adds ~3 frames to the title-gen
  conversation, which OWUI should ignore.
- **Mid-stream cancellation** on fast path now produces a visible
  `❌` close on the banner. Previously the response stopped silently.
  This is more transparent but a behavior change worth knowing if
  anything scrapes SSE bodies.
- **Monitoring impact**: prometheus metrics are unaffected.
  `pipeline_seconds` and `pipeline_total` labels stay the same. No
  new metrics added in this phase.

## 5. Followups

- The Phase 6a complexity-gate investigation still wants to run.
  Banner work doesn't change the underlying "tool-bloat pushes
  follow-up turns to deep" pattern — it just makes the wait feel
  less frozen.
- Plain-chat fast path's static 3-frame banner is the simplest
  shape that worked, but for slow first-token cases (>500ms) the
  user briefly sees `> _Thinking ✅_` followed by silence again.
  Rare in practice with cl100k models, but if it shows up in real
  traffic the fix is to use the full PhaseTicker for plain chat
  too. Not done now — adds queue plumbing for marginal benefit.
- Deep mode's three-banner sequence is information-dense
  (`Dispatching panel ✅ kimi-k2.6:cloud  ✅ qwen3.6:35b`). The
  fast path's single Thinking banner is lighter on purpose — fast
  responses don't have per-worker results to surface. If deep ever
  gets simplified, the two paths could converge on a shared shape.
