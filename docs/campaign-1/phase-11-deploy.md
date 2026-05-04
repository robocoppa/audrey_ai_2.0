# Phase 11 — Per-user memory recall + memory_store hint

**Goal:** teach the pipeline to remember facts across conversations. Every
request is keyword-searched against the caller's prior memories (scoped by
`user:<id>` tag) and any hits get injected as a system message before
classify runs. Tool-capable models also get a short hint telling them when
to call `memory_store` with the right user tag.

What's new vs Phase 10:

- **`tools-server` gains `/memory_search`** — POSTs `{user, query, top_k}`,
  returns keyword-matched entries ordered by `updated_at DESC`. Matches any
  whitespace token against `key`, `value`, or `tags`. Tag scope is
  enforced by a `LIKE '%user:<id>%'` clause.
- **`ChatCompletionRequest` accepts `user`** (OpenAI-spec field). Open WebUI
  forwards the signed-in user's id here. Missing/empty `user` → memory
  step is skipped entirely (no recall, no store hint, no writes).
- **New `node_memory_recall` at the top of the graph** — runs before
  classify. Calls `memory_search` (via the same `dispatch_one` path the
  ReAct loop uses), prepends results as a system message. Also prepends the
  `memory_store` usage hint when that tool is registered. Errors/timeouts
  are non-fatal — best-effort.
- **Two state fields** added: `user_id: str` and `memory_hits: list[dict]`.
- **`config.yaml` `agentic.memory`** block: `enabled` / `top_k` / `timeout_s`.

**What didn't change:** the existing `memory_store` / `memory_recall` tools
still work as before (exact-key writes/reads). Models can call them via
ReAct exactly as they do with any other tool. The new piece is only the
auto-recall step at request start and the prompt nudge that tells the model
what tag convention to use.

---

## Prereqs

- Phase 10 green (Open WebUI pointed at v2, Cloudflare Tunnel live).
- `custom-tools` and `audrey-ai` rebuilt from this commit (compose).
- `curl` + `jq` on Unraid. `docker exec audrey-ai` for Python one-liners.

> **Note on URLs below.** All `curl` examples use `http://localhost:<port>`
> from the Unraid host shell (both services publish their ports to the
> host). If you run these *inside* a container on `ollama-net` instead,
> substitute `custom-tools:8001` / `audrey-ai:8000` — those container-DNS
> names don't resolve from the host.

Rebuild + restart:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build custom-tools audrey-ai
docker compose logs -f audrey-ai   # in one pane
```

---

## 1. Tool registration

Confirm `memory_search` is advertised alongside the existing tools:

```bash
curl -s http://localhost:8001/openapi.json \
  | jq '.paths | keys'
```

Expected: `["/health", "/kb_image_search", "/kb_search", "/memory_recall", "/memory_search", "/memory_store", "/web_search"]` (order varies).

Confirm Audrey picked it up (note: rediscover is POST):

```bash
curl -s -X POST http://localhost:8000/v1/tools/rediscover | jq '.tools'
```

Expected: array containing `"memory_search"` and `"memory_store"`.

---

## 2. Smoke tests

### 2.1 Empty user → memory step is a no-op

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","messages":[{"role":"user","content":"hello"}]}' \
  > /dev/null
```

Then tail logs:

```bash
docker compose logs --tail 20 audrey-ai | grep -E "memory:|classify:"
```

Expected: no `memory:` lines. Classify runs as normal. Proves the
skip-when-unidentified path.

### 2.2 First write (direct tool call, via custom-tools)

Seed a memory for a test user so 2.3 has something to find:

```bash
curl -s http://localhost:8001/memory_store \
  -H 'content-type: application/json' \
  -d '{"key":"prefers_rust","value":"Bart prefers Rust over Go for systems work, cites better type inference","tags":"user:bart@proton.me,topic:languages"}' \
  | jq
```

Expected: 200, body echoes back the entry with `created_at` / `updated_at`.

### 2.3 Recall fires on a matching query

Send a request whose `user` matches the tag and whose text overlaps
("rust", "languages"):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"what language should I pick for a new systems project?"}]}' \
  | jq -r '.choices[0].message.content' | head -40
```

Then check the log:

```bash
docker compose logs --tail 40 audrey-ai | grep "memory:"
```

Expected: `memory: user=bart@proton.me hits=1 keys=['prefers_rust'] store_hint=on`.

The answer itself should mention Rust (the recalled memory is in the
prompt as a system message). If it doesn't, re-read the injected hint —
the model may have deemed it irrelevant.

### 2.4 Non-matching query → zero hits (hint still attached)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"what is the boiling point of water?"}]}' \
  > /dev/null
docker compose logs --tail 80 audrey-ai | grep "memory:" | tail -5
```

Expected: `memory: user=bart@proton.me hits=0 keys=[] store_hint=on`. The
store hint alone is enough to prepend a system message.

> `--tail 10` is too narrow if the request runs for more than a few
> seconds — `audrey_deep` on a cold prompt can take 15–30s and the
> `memory:` line scrolls out of a 10-line window before the response
> lands. Use `--tail 80` (or `--since 1m` for time-scoped filtering).

### 2.5 Cross-user isolation

Store a memory for a *different* user and verify the first user can't see
it:

```bash
curl -s http://localhost:8001/memory_store \
  -H 'content-type: application/json' \
  -d '{"key":"hates_rust","value":"Alice thinks Rust is overhyped","tags":"user:alice@example.com,topic:languages"}' \
  > /dev/null

curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"rust languages","top_k":5}' \
  | jq '{count: (.results | length), keys: [.results[].key]}'
```

Expected: `{"count": 1, "keys": ["prefers_rust"]}`. `"hates_rust"` must
NOT appear — it's tagged `user:alice@example.com`.

### 2.6 Auto-write via model tool call

Send a prompt that hands the model a fact and a save instruction. Use a
tool-capable model explicitly by picking `audrey_deep` with a short prompt
(stays on fast path):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"Please remember that my daily driver is an AMD Threadripper 7970X workstation."}]}' \
  | jq -r '.choices[0].message.content' | head -10
```

Then confirm the model actually wrote a memory:

```bash
curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"threadripper workstation","top_k":5}' \
  | jq '.results[] | {key, value, tags}'
```

Expected: at least one new entry whose `tags` contain `user:bart@proton.me`
and whose `value` mentions the Threadripper. The exact key depends on the
model's choice — `daily_driver`, `hardware`, `workstation`, etc. are all
fine.

If the entry's `tags` are missing `user:bart@proton.me`, the model
ignored the hint — re-read the injected system message (log line should
show `store_hint=on`) and consider strengthening the hint wording. This
is the smoke test most sensitive to model compliance.

### 2.7 Round-trip: write in turn N, recall in turn N+1

Same `user`, new request. The memory written in 2.6 should now surface:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"what hardware am I running?"}]}' \
  | jq -r '.choices[0].message.content' | head -10
```

Expected: the answer references the Threadripper 7970X (pulled from the
memory written in 2.6, not from training data).

### 2.8 Graceful degradation when custom-tools is down

Simulate an outage:

```bash
docker stop custom-tools
curl -s -w "\nhttp=%{http_code}\n" http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"say hi"}]}' \
  | tail -5
docker start custom-tools
```

Expected: `http=200`. Audrey answers without memory (the recall call
returns an error which becomes zero hits; the pipeline continues). The
log line should show `memory: recall skipped (search error: …)`.

---

## Troubleshooting

- **`memory_search` not in `/openapi.json`**: the `custom-tools` image is
  stale. Rebuild with `docker compose up -d --build custom-tools`.
- **`memory: user=… hits=0 keys=[] store_hint=off`** on a request you
  expected to hit memory: the tool registry didn't pick up
  `memory_store`. Run `POST /v1/tools/rediscover` and check logs for
  OpenAPI errors.
- **Model doesn't save anything on 2.6**: confirm the chosen model is in
  `fast_path.tool_capable_models`. Tool-blind models (e.g. `olmo-3.1:32b`)
  can't call `memory_store` even with the hint. Check the
  `fast_path task=… -> <model> (tools=on)` log line.
- **`user` isn't getting through**: Open WebUI ≥ v0.5 forwards it by
  default. For older versions you may need `ENABLE_FORWARD_USER_INFO=true`
  or equivalent. Test by logging the `user_id` at the top of
  `node_memory_recall` — if it's always empty, Open WebUI isn't
  forwarding.
- **Memory DB location**: `/mnt/user/appdata/custom-tools/memory.db` on
  Unraid (SQLite). Safe to `sqlite3` against directly for inspection;
  writes should go through the HTTP endpoint to keep timestamps consistent.
