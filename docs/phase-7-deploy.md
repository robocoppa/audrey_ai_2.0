# Phase 7 — Custom-tools wiring + ReAct loop on the fast path (Unraid)

**Goal:** the orchestrator auto-discovers tools from `custom-tools` and lets
tool-capable models call them via a ReAct loop on the fast path. Deep mode
still runs the panel; tools are a fast-path concern in this phase.

What's new vs Phase 6:

- **`tools/discovery.py`** — fetches `/openapi.json` from every server in
  `tools.servers`, walks `$ref`s to inline schemas, strips JSON-Schema
  keywords Ollama doesn't accept, and produces an `Ollama tool[]` payload.
  Skips `operation_id == "health"` and endpoints without the `tools` tag.
- **`tools/dispatch.py`** — invokes one tool over HTTP, decodes JSON-string
  arguments, truncates oversized results to `agentic.react.max_tool_result_chars`
  (default 2000) with a `…[truncated]` suffix. Never raises — failures
  come back as `is_error=True` `ToolResult`s the model can read.
- **`pipeline/react.py`** — wraps `ollama.chat` in a ReAct loop:
  1. send messages + `tools=[…]`
  2. if no `tool_calls` in the response, return the answer
  3. dispatch all `tool_calls` in parallel (`asyncio.gather`)
  4. append `role=tool` messages, repeat
  5. at `max_rounds`, call once more *without* `tools` to force a final answer
- **Context compression** — after `compress_after_round` (default 2), older
  `role=tool` messages are replaced with stub `system` messages
  (`[earlier tool call: NAME -> N chars elided]`) so the conversation
  doesn't blow past the model's context window.
- **`/v1/tools` and `/v1/tools/rediscover`** — list and refresh registered
  tools at runtime. Rediscovery mutates the in-process `ToolRegistry` in
  place; the graph keeps its closure over the same instance, so changes
  apply on the next request without a graph rebuild.
- **Streaming with tools** — tool-capable model + non-empty registry routes
  the streaming request through the graph and emits one consolidated chunk
  (mid-stream tool dispatch isn't supported in Phase 7).

**Prereqs confirmed:**

- Phase 6 done; `audrey-ai` healthy on `ollama-net`.
- `custom-tools` container running and reachable from `audrey-ai` on
  `http://custom-tools:8001`.
- `BRAVE_API_KEY` is set in `.env` (otherwise `web_search` returns 502).
- `config.yaml > tools.enabled: true` and the `tools.servers` list contains
  `http://custom-tools:8001`.

---

## Step 1 — Rebuild & restart

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build           # rebuilds audrey-ai + custom-tools
```

If only `audrey-ai` changed:

```bash
docker compose up -d --build audrey-ai
```

No new Python deps in the orchestrator. `custom-tools` is unchanged.

If the new log shapes (Step 3) don't appear, rebuild with `--no-cache`.

---

## Step 2 — Smoke tests

### 2.1 — Tools registered at startup

```bash
docker logs audrey-ai 2>&1 | grep -E "tools:|ready:" | tail -5
# Expected:
#   tools: discovered N tool(s) from http://custom-tools:8001
#   ready: ... tools=5 (['web_search','kb_search','kb_image_search','memory_store','memory_recall'])
```

If `tools=0`, `custom-tools` is unreachable or `tools.enabled: false`.
Verify: `docker exec audrey-ai curl -fsS http://custom-tools:8001/openapi.json | jq '.paths | keys'`.

### 2.2 — `GET /v1/tools` lists what was discovered

```bash
curl -s http://localhost:8000/v1/tools | jq '.tools[] | {name, server_url, path}'
# Expected: 5 entries — web_search, kb_search, kb_image_search,
# memory_store, memory_recall — all pointing at http://custom-tools:8001
```

### 2.3 — `web_search` triggers on a time-sensitive prompt

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"what is the latest stable release of postgresql as of today? cite a source url"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, head: .choices[0].message.content[0:240]}'
# Expected:
# - fp contains a tool-capable model (e.g. qwen3.6:35b or kimi-k2.6:cloud)
# - answer mentions a current PostgreSQL major version + a real URL
```

Then check the log:

```bash
docker logs audrey-ai --tail 40 | grep -E "react:|tool_dispatch:|chat\.completions"
# Expected:
#   react: round 1: model produced 1 tool_call(s): ['web_search']
#   tool_dispatch: web_search ok in 0.8s (1740 chars)
#   react: round 2: no tool_calls → returning answer
#   chat.completions model=audrey_deep task=general(...) mode=fast -> qwen3.6:35b tool_rounds=1 tool_calls=[web_search]
```

### 2.4 — `memory_store` round-trips with `memory_recall`

Two-call flow: first store a fact, then in a separate request ask for it back.

```bash
# Store
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"please remember this: my preferred shell is fish (key: shell_pref). store it via the memory tool and confirm."}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:200]'

# Recall (new conversation, no prior history)
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"using your memory tool, recall the key shell_pref and tell me what it says."}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:240]'
# Expected: the recall answer mentions "fish"
```

Log check:

```bash
docker logs audrey-ai --tail 60 | grep -E "tool_dispatch:|tool_calls"
# Expected at least:
#   tool_dispatch: memory_store ok in 0.0s
#   tool_dispatch: memory_recall ok in 0.0s
```

Also verify the SQLite row landed (the slim image has no `sqlite3` CLI,
so use Python's stdlib module):

```bash
docker exec custom-tools python3 -c "
import sqlite3
c = sqlite3.connect('/app/data/memory.db')
print(c.execute(\"select key, substr(value,1,40) from memory where key='shell_pref'\").fetchall())
"
# Expected: [('shell_pref', 'fish')]
```

### 2.5 — Result truncation kicks in for long tool output

Pick a query that returns a lot of text — Brave with `max_results=5` plus
descriptions usually does. Then confirm in the log that the dispatched
result was clamped:

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"web search: complete history of the linux kernel project, give me 5 paragraphs of context"}],
    "stream": false
  }' > /dev/null

docker logs audrey-ai --tail 20 | grep "tool_dispatch: web_search"
# Expected: "(... chars)" value is ≤ agentic.react.max_tool_result_chars (2000 by default)
```

### 2.6 — ReAct round cap forces a final answer

Default `max_rounds: 3`. Easiest way to verify the cap fires is to ask a
question that keeps prompting the model to call tools:

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"web search the latest changelog for postgres, then web search release notes for redis, then web search release notes for sqlite, then summarize all three"}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:200]'

docker logs audrey-ai --tail 60 | grep -E "react:|tool_rounds"
# Expected (one possibility):
#   react: round 1: 1 tool_call(s)
#   react: round 2: 1 tool_call(s)
#   react: round 3: 1 tool_call(s)
#   react: max_rounds reached, forcing final answer without tools
```

### 2.7 — Streaming with a tool-capable model emits one chunk

```bash
curl -s -N -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"web search who won the most recent F1 race"}],
    "stream": true
  }' | tail -3
# Expected:
# - second-to-last frame contains delta.content with a real answer
# - final frame is `data: [DONE]`
```

### 2.8 — Non-tool prompt skips the loop entirely

Short factual prompt to a tool-capable model — should *not* trigger a tool
call (model decides), and `tool_rounds=0` in the log line.

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"what is 2+2?"}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:80]'

docker logs audrey-ai --tail 5 | grep "chat.completions"
# Expected: no `tool_rounds=` suffix on the chat.completions line
# (the suffix only appears when rounds > 0)
```

### 2.9 — `POST /v1/tools/rediscover` picks up changes live

```bash
curl -s -XPOST http://localhost:8000/v1/tools/rediscover | jq
# Expected: {"tools":[...5 names...],"count":5}

docker logs audrey-ai --tail 5 | grep "rediscover"
# Expected: tools: rediscover -> 5 tool(s): [...]
```

If you stop `custom-tools` and rediscover, `count` should drop to `0` —
prove it tolerates a downed server without crashing the orchestrator:

```bash
docker stop custom-tools
curl -s -XPOST http://localhost:8000/v1/tools/rediscover | jq
# Expected: {"tools":[],"count":0}
docker start custom-tools
sleep 3
curl -s -XPOST http://localhost:8000/v1/tools/rediscover | jq '.count'
# Expected: 5
```

---

## Step 3 — Report back

If 2.1–2.9 all pass, Phase 7 is done. Reply with **"phase 7 smoke tests passed"**
and we'll move to Phase 8 (KB ingest + `kb_search` end-to-end with Qdrant).

If anything fails, paste:

- `docker logs audrey-ai --tail 120`
- `docker logs custom-tools --tail 60`
- The failing curl output
- Which test number failed

---

## Troubleshooting

- **`tools=0` at startup but `custom-tools` is up:** check the URL is
  reachable from inside the audrey container:
  `docker exec audrey-ai curl -fsS http://custom-tools:8001/openapi.json | head -c 200`.
  If that fails, confirm both containers are on `ollama-net` and that
  the env override `TOOL_SERVERS` (if set in `.env`) hasn't pointed away
  from `http://custom-tools:8001`.

- **Model never calls a tool even on `web search …` prompts:** the model
  isn't in `fast_path.tool_capable_models`. Check the chosen model in the
  log line (`-> <model>`); add it to the list in `config.yaml` and rebuild.
  Some smaller local models also don't reliably emit `tool_calls`; try
  forcing one of the cloud models for that query.

- **`tool_dispatch: web_search error: 401` (or 502):** Brave key missing
  or wrong. Verify with
  `docker exec custom-tools env | grep BRAVE_API_KEY` and
  `docker logs custom-tools | grep -i brave`.

- **`memory_recall` returns "no entry" right after a successful store:**
  the data dir bind is broken or the second call hit a different container.
  Verify the row exists in SQLite (Step 2.4 query). If absent, `docker
  inspect custom-tools | grep -A 3 Mounts` to confirm `/app/data` is bound
  to `/mnt/user/appdata/custom-tools` on the host.

- **Long ReAct loops stall conversations after several turns:** lower
  `agentic.react.compress_after_round` from 2 to 1 in `config.yaml`, or
  drop `max_tool_result_chars` from 2000 to 1000. Both restart-required.

- **`/v1/tools/rediscover` returns the old count after a tool was added:**
  the orchestrator caches *nothing* about discovery beyond the
  `ToolRegistry`. If a tool is missing, the new server's `/openapi.json`
  doesn't include it (or its endpoint lacks the `tools` tag / has
  `operation_id: "health"`).

- **Streaming with a tool-capable model returns the full answer in one
  shot:** that's the documented behavior in Phase 7. Mid-stream tool
  dispatch lands in a later phase if/when needed.
