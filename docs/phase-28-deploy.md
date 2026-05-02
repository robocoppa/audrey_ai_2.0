# Phase 28 — Tools-used footer

Deep-panel and tool-capable fast-path responses already stream progress
banners (Phase 18) and per-worker checkmarks during dispatch. They
don't surface **which tools each worker actually called**. The data is
collected (`WorkerDraft.tool_calls`, `PipelineState.tool_calls_log`)
but only ever lands in the server log line — never in the response the
user sees.

Phase 28 appends a markdown footer after the answer body listing
tool usage broken down per worker.

What stays the same:
- Streaming protocol (still SSE deltas; the footer is one more delta).
- Banner phase headers (Thinking → Dispatching → Synthesizing).
- `WorkerDraft` shape — already carried `tool_calls`; nothing new.
- Non-streaming JSON responses — no banners, no footer (log line
  unchanged).
- Tool-free requests — no footer, no visible change.

What changed:
- **`src/audrey/pipeline/banners.py`** — added `tool_summary_block()`
  formatter (+ private `_format_calls()` helper). Builds the markdown
  block from a `[(model, calls), ...]` list. Returns empty string when
  no worker called a tool.
- **`src/audrey/routes/openai.py`** —
  - Streaming deep path (`_stream_deep_with_banners`): after synth
    drains, build the footer from `drafts` and yield it as one
    `_delta_frame` before `_stop_frame()`.
  - Streaming fast path, tool-capable branch: append the footer to
    `content` before passing to `_emit_single_message`.

Footer rendering:

```
---
> _Tools used:_
> - **qwen3.6:35b** — `kb_search` ×2
> - **deepseek-v4-pro:cloud** — `web_search`, `kb_search` ❌
```

Rules:
- One row per worker that called at least one tool. Workers with zero
  tool calls are omitted (keeps the block tight).
- Within a worker, repeat-named calls collapse to `name ×N`.
- A name with any error during this worker's run gets ❌ appended.
- Order is workers in completion order (matches the dispatch banner),
  tool names in first-seen order within each worker.
- Block is preceded by `\n\n---\n` so OWUI's markdown renderer treats
  it as a horizontal rule below the answer body, not as part of the
  prose.

Out of scope (deliberately):

- **Per-call timing in the footer.** `tool_calls` carries `elapsed_s`
  per call but rendering it inflates the block. The values still go
  into Prometheus (`audrey_tool_call_seconds`) — Grafana is the right
  place for latency.
- **Tool args / results in the footer.** Would leak query content (KB
  searches contain user queries verbatim; web_search may surface
  private info). Names + counts only.
- **Footer for fast-path no-tools branch.** That branch never calls
  ReAct (`use_tools=False` in `run_fast_path`); there's no data to
  render. Token-stream pass-through stays untouched.
- **Footer for the non-streaming JSON path.** The `chat.completions`
  return shape is OpenAI-compatible; bolting a banner-style footer
  onto `choices[0].message.content` would surprise programmatic
  clients. Logs already show `tool_rounds=N tool_calls=[...]` for
  that path.

**Prereqs:** all phases through 27 verified. No env vars, no compose
changes, no model changes.

---

## 1. Deploy

```bash
# Laptop:
git pull   # after the user has committed Phase 28

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build audrey-ai
docker compose logs -f --since 1m audrey-ai | grep -i ready
```

Expect a clean `ready: ...; pipeline=compiled` line — Phase 28 only
touches Python files in `src/audrey/`, no startup behavior changes.

---

## 2. Smoke tests

Set the auth env once for the session:

```bash
ADMIN_TOKEN="<your OWUI bearer token, role=admin>"
```

### 2.1 Tool-using deep request shows the footer

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_cloud",
    "stream": true,
    "messages": [
      {"role":"user","content":"Use kb_search and web_search to find recent news about geology in Iceland."}
    ]
  }' > /tmp/phase28-deep.sse

# Footer presence + last bit of assembled body (so you can eyeball it):
grep -c "Tools used" /tmp/phase28-deep.sse

# Reassemble all SSE deltas into the rendered answer body. Pipes the
# raw SSE into the audrey-ai container's python (Unraid host has
# neither python3 nor jq's --slurp newline handling that we'd need).
cat /tmp/phase28-deep.sse | docker exec -i audrey-ai python -c '
import json, sys
body = []
for line in sys.stdin:
    if not line.startswith("data: "): continue
    payload = line[6:].strip()
    if payload == "[DONE]": continue
    try: evt = json.loads(payload)
    except Exception: continue
    delta = (evt.get("choices") or [{}])[0].get("delta", {})
    text = delta.get("content")
    if text: body.append(text)
print("".join(body)[-800:])
'
```

Expected: `grep -c "Tools used"` returns `1` or higher, and the
printed tail ends with the footer:

```
---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `kb_search` ×2, `web_search`
> - **kimi-k2.6:cloud** — `web_search`
```

### 2.2 Tool-free deep request omits the footer

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_cloud",
    "stream": true,
    "messages": [
      {"role":"user","content":"Explain BTRFS copy-on-write semantics in two paragraphs. Do not use any tools."}
    ]
  }' | grep -c "Tools used"
```

Expected: `0` — workers can decline to call tools; when they do, no
footer is rendered.

### 2.3 Tool-using fast request shows the footer

`audrey_fast` is always-fast and goes through the tool-capable graph
branch when the picked model is in `fast_path.tool_capable_models`.

```bash
curl -sN -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_fast",
    "stream": true,
    "messages": [
      {"role":"user","content":"Use kb_search to find a survival shelter knot."}
    ]
  }' > /tmp/phase28-fast.sse

grep -c "Tools used" /tmp/phase28-fast.sse
cat /tmp/phase28-fast.sse | docker exec -i audrey-ai python -c '
import json, sys
body = []
for line in sys.stdin:
    if not line.startswith("data: "): continue
    payload = line[6:].strip()
    if payload == "[DONE]": continue
    try: evt = json.loads(payload)
    except Exception: continue
    delta = (evt.get("choices") or [{}])[0].get("delta", {})
    text = delta.get("content")
    if text: body.append(text)
print("".join(body)[-800:])
'
```

Expected: `grep -c "Tools used"` returns `1`, and the printed tail
ends with one footer row for the picked fast-path model.

### 2.4 Footer never breaks the answer

Open `/tmp/phase28-deep.sse` and `/tmp/phase28-fast.sse` in OWUI's
chat history (or any markdown previewer). Confirm:

- The horizontal rule between the answer and the footer renders.
- Tool names are inline-code formatted.
- Worker name is bold.
- ❌ only appears for tools that errored (verify by stopping
  `custom-tools` mid-test if you want to force one).

### 2.5 Non-streaming JSON path unchanged

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_fast",
    "stream": false,
    "messages": [
      {"role":"user","content":"Use kb_search to find a survival shelter knot."}
    ]
  }' | jq '.choices[0].message.content' | grep -c "Tools used"
```

Expected: `0`. Footer is streaming-only by design.

### 2.6 Log line still records tool calls (regression check)

```bash
docker compose logs --since 2m audrey-ai | grep -E "tool_rounds=" | tail -5
```

Expected: lines like `... tool_rounds=1 tool_calls=[kb_search]` still
present for fast-path tool-using requests (the streaming deep path's
tool detail goes into Prometheus, not into the per-request log).

---

## 3. Rollback

Revert the two file edits (`src/audrey/pipeline/banners.py` and
`src/audrey/routes/openai.py`), rebuild:

```bash
docker compose up -d --build audrey-ai
```

The footer is purely additive — the rollback impact is zero (footer
just disappears).

---

## 4. Operational notes

- **The footer is a `_delta_frame`, not a separate event type.** OWUI
  treats it like any other content delta. If you ever wire a
  programmatic client that wants to ignore the footer, the
  `\n\n---\n> _Tools used:_` substring is a stable parse marker.
- **Tool-call counts in the footer match the metric counter.**
  `audrey_tool_calls_total{tool="kb_search"}` per request equals the
  sum of `kb_search` counts across all rendered worker rows for that
  request. Useful cross-check if you ever doubt the footer.
- **The `❌` marker is OR'd across calls within a worker.** A worker
  that called `kb_search` four times — three OK, one error — renders
  as `kb_search ×4 ❌`. This is intentional: the user's answer is
  partly tool-grounded and partly not, and the marker flags that.
- **No Prometheus metric was added.** The data feeding this footer is
  already counted in `audrey_tool_calls_total` and timed in
  `audrey_tool_call_seconds` (both Phase 22). No new cardinality.
