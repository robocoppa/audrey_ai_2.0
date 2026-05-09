# Campaign 2 Phase 1 - Searchable chat archive

This phase ships a per-user chat archive that integrates with Audrey's
existing memory/tool path. Every chat completion (streaming or not) is
archived after the response is produced; the model can later look
conversations back up via the new user-scoped `chat_history_search`
tool. Durable memory and `memory_recall` are unchanged.

The headline behavior change: chat history is now persisted on every
turn and searchable on demand. Token cost on ordinary requests does not
change — chat-archive recall is tool-driven, not auto-injected.

What stays the same:

- Five virtual models: `audrey_auto`, `audrey_fast`, `audrey_deep`,
  `audrey_cloud`, `audrey_local`.
- Pipeline shape: classify, complexity gate, fast path or deep panel,
  synth, reflect.
- Durable memory (`memory_store` / `memory_recall` / `memory_search`)
  unchanged. Auto-recall before classify still uses durable memory
  only.
- Existing tool surface unchanged for the model: web_search, kb_search,
  kb_image_search, memory_*. `chat_history_search` is added.
- No changes to Ollama, Qdrant deployment, or Open WebUI.
- All service containers still join `ollama-net`.

What changed:

- **`tools-server/settings.py`** -
  - New env-driven settings: `CHAT_ARCHIVE_COLLECTION`,
    `CHAT_ARCHIVE_CHUNK_MAX_CHARS`, `CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS`,
    `CHAT_ARCHIVE_SEARCH_THRESHOLD`, `CHAT_ARCHIVE_RETENTION_DAYS`,
    `CHAT_ARCHIVE_MAX_BYTES`. All optional with safe defaults.
  - Defaults: collection `kb_chat_archive`, max chunk 1500 chars,
    overlap 100 chars, search threshold 0.4, retention 0 (forever),
    max bytes 0 (no cap). Reuses the durable-memory embed model and
    dimension so a missing Ollama only breaks one subsystem.
- **`tools-server/chat_archive.py`** (new) -
  - `ChatArchiveStore`: SQLite source of truth + Qdrant search index.
  - Schema: `conversations`, `messages`, `archive_chunks`. Indexed by
    `(user, conversation_id)` and `(user, created_at)`.
  - Q+A-pair chunking: one chunk per (user-turn, following-assistant-turn).
    Oversize pairs split at sentence boundaries with overlap.
  - `derive_message_id` is a deterministic hash of
    `sha256(user|conversation_id|role|content|minute_bucket)[:32]` so
    retries within the same minute and full-history re-sends collapse
    via `INSERT OR IGNORE`.
  - `prune()` honors `CHAT_ARCHIVE_RETENTION_DAYS` and drops both
    SQLite rows and Qdrant points.
- **`tools-server/app.py`** -
  - New tool route `POST /chat_history_search` (model-callable).
  - New internal routes hidden from `/openapi.json` so the model
    cannot call them: `POST /chat_history/archive`,
    `POST /chat_history/prune`, `GET /chat_history/stats`.
  - Lifespan now constructs and tears down the `ChatArchiveStore`.
- **`src/audrey/tools/dispatch.py`** -
  - Added `chat_history_search` to `_USER_SCOPED_TOOLS`. The dispatcher
    overwrites any model-supplied `user` argument with the
    authenticated pipeline user, same posture as `memory_search`.
- **`src/audrey/pipeline/chat_archive.py`** (new) -
  - `ChatArchiveClient`: best-effort writer to custom-tools. Looks up
    the host server from the registry's `chat_history_search` entry on
    every call so a tools-server reload doesn't strand the writer.
    Never raises; logs and counts failures.
  - `StreamCollector`: wraps an SSE generator, passes frames through
    unchanged, accumulates only `delta.content` strings. Banner and
    tool-call frames are filtered out automatically.
  - `resolve_conversation_id`: reads OWUI `chat_id` from the request
    body when present, then `metadata.chat_id`, then the last
    message's `metadata.chat_id`, otherwise derives a deterministic id
    from the message-history prefix; UUID is the last resort.
- **`src/audrey/main.py`** -
  - Lifespan now constructs a shared httpx client and a
    `ChatArchiveClient`, exposes both on `app.state`.
- **`src/audrey/routes/openai.py`** -
  - `chat_completions` resolves the conversation id and the last user
    text once, before pipeline branches.
  - Non-streaming path archives after the response is produced.
  - Fast/tool-capable streaming path wraps the SSE generator with
    `StreamCollector` and archives in a `finally` so a client
    disconnect mid-stream still archives what was emitted, marked
    `partial=True`.
  - Deep streaming path archives `final_content` (synth deltas only,
    no banners) inside its existing `finally`.
- **`src/audrey/routes/admin.py`** -
  - `POST /v1/admin/chat_archive/prune` triggers the retention sweep.
  - `GET /v1/admin/chat_archive/stats` returns row counts.
- **`src/audrey/metrics.py`** -
  - New: `audrey_chat_archive_writes_total{result}` counter (results:
    `ok`, `partial`, `fail`, `skipped`) and
    `audrey_chat_archive_write_seconds` histogram.
- **`tests/test_chat_archive.py`** -
  - Hermetic coverage for `derive_message_id` dedup, Q+A chunking,
    `StreamCollector` content-only capture and partial-on-cancel,
    OWUI conversation-id resolution, archive-client skip/post/error
    paths, and dispatcher user-scope inclusion.

Out of scope:

- Automatic chat-archive recall on every prompt. The future
  `agentic.memory.archive_auto_recall` knob from the design doc is not
  wired yet; archive-derived context only enters a prompt when the
  model deliberately calls `chat_history_search`.
- Importing historical Open WebUI conversations. Only conversations
  starting after this phase ships are archived.
- A user-facing archive browser.
- Cross-user / admin search.
- Summarizing every conversation after completion.
- The `chat_history_context` window-fetch tool (deferred until search
  snippets prove insufficient).
- Phase 2's `prompts.py` centralization. The tool-description string
  alone steers the model's use of `chat_history_search` in this phase.

## 1. Deploy

No data migration required for existing users. New SQLite file is
created on first custom-tools start; new Qdrant collection is created
on first connection.

```bash
# Laptop:
git pull   # after the Phase 1 commit lands

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build custom-tools
docker compose up -d --build audrey-ai
docker compose logs --since 2m custom-tools | grep -E "chat_archive|ready"
docker compose logs --since 2m audrey-ai | grep -E "ready|tools="
```

Expected:

- A `chat_archive: ready sqlite=/app/data/chat_archive.db
  qdrant_collection=kb_chat_archive dim=768 retention_days=0` line in
  custom-tools logs.
- A `custom-tools ready. ... archive=kb_chat_archive` line in
  custom-tools logs.
- A normal Audrey readiness line ending with `pipeline=compiled`. The
  tool count should now show `chat_history_search` alongside the
  existing tools.

## 2. Smoke tests

### 2.1 Confirm Audrey discovered the new tool

```bash
docker compose logs --since 2m audrey-ai | grep "tools="
```

Expected: a tool count one higher than before, including
`chat_history_search` in the names list.

### 2.2 Confirm archive writes happen on a normal request

```bash
USER_TOKEN="<valid OWUI bearer token>"
AUDREY_URL="http://localhost:8000"

# Baseline counts (custom-tools host, e.g. via mapped port):
curl -sS http://localhost:8001/chat_history/stats | jq

curl -sS -o /tmp/audrey-fast.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_fast",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Briefly explain what BTRFS is."}
    ]
  }'

curl -sS http://localhost:8001/chat_history/stats | jq
```

Expected: the second `stats` call shows `messages` up by 2 (one user
turn, one assistant turn) and `chunks` up by 1. Audrey's
`audrey_chat_archive_writes_total{result="ok"}` Prometheus counter
should also increment.

### 2.3 Confirm dedup on a re-sent history

Open WebUI re-sends the entire conversation on each turn. Repeat the
exact same request from 2.2:

```bash
curl -sS -o /dev/null "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "model": "audrey_fast", "stream": false,
        "messages": [{"role": "user", "content": "Briefly explain what BTRFS is."}] }'

curl -sS http://localhost:8001/chat_history/stats | jq
```

Expected: `messages` increases by at most 2 (a fresh assistant response
generated from the same minute may share a `message_id` with the
previous one and be ignored). `chunks` may stay flat or increase by 1.
What must not happen: `messages` doubling on every retry.

### 2.4 Confirm streaming capture

```bash
curl -sS -N -o /tmp/audrey-stream.sse \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "model": "audrey_fast", "stream": true,
        "messages": [{"role": "user", "content": "Give me three uses for ZFS."}] }'

# Audrey logs should include exactly one archive write for this turn.
docker compose logs --since 1m audrey-ai | grep -E "chat_archive"

curl -sS http://localhost:8001/chat_history/stats | jq
```

Expected: `messages` increased by 2, `chunks` by 1. The archive write
should not have appeared mid-stream, only after the stream completed.

### 2.5 Confirm `chat_history_search` is user-scoped

Run the search with one OWUI user, then the same search with a
different OWUI user. Each user should see only their own conversation
history.

```bash
curl -sS -o /tmp/search-alice.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_fast",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Search my prior chats for what I asked about BTRFS."}
    ]
  }'

cat /tmp/search-alice.json | jq '.choices[0].message.content'
```

Expected: the answer cites prior turns from this user's archive only.
Switching `USER_TOKEN` to a second OWUI user and running the same
prompt should return either an empty result or only that user's
unrelated chat history.

### 2.6 Confirm partial-on-disconnect

Issue a streaming request and cancel it mid-stream:

```bash
curl -sS -N -m 2 \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "model": "audrey_deep", "stream": true,
        "messages": [{"role": "user", "content": "Write a long detailed essay about ZFS."}] }' || true

docker compose logs --since 1m audrey-ai | grep -E "chat_archive"
```

Expected: an archive write with `partial=true` recorded for whatever
synth tokens streamed before the cutoff. The
`audrey_chat_archive_writes_total{result="partial"}` counter should
increment.

### 2.7 Optional: prune

Set `CHAT_ARCHIVE_RETENTION_DAYS=30` in the custom-tools env and
restart, then trigger a prune via the admin route (admin OWUI bearer
required):

```bash
ADMIN_TOKEN="<admin OWUI bearer token>"
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$AUDREY_URL/v1/admin/chat_archive/prune" | jq
```

Expected: a `{ "messages_deleted": N, "chunks_deleted": M,
"qdrant_deleted": M }` payload. With retention 0 (default), the
counts should all be zero.

### 2.8 Local verification

On the laptop:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src/audrey/pipeline/chat_archive.py \
  tests/test_chat_archive.py tools-server/chat_archive.py \
  tools-server/settings.py src/audrey/tools/dispatch.py \
  src/audrey/metrics.py src/audrey/main.py \
  src/audrey/routes/admin.py src/audrey/routes/openai.py
```

Expected:

- All tests pass (the new file adds 18 cases).
- Ruff: `All checks passed!` for the listed paths. The pre-existing
  ASYNC240 perf hints in `kb/ingest.py` are unchanged and accepted.

## 3. Rollback

Plain git rollback. SQLite file and Qdrant collection are additive —
they do not displace existing data — so a revert is safe:

```bash
git revert <phase-1-commit>
docker compose up -d --build custom-tools
docker compose up -d --build audrey-ai
```

The `chat_archive.db` file under `TOOLS_DATA_DIR` and the
`kb_chat_archive` Qdrant collection can be left in place after revert.
Re-deploying the phase later picks up where it left off; deleting them
between revert and re-deploy is harmless and forces a clean slate.

If you want to remove them anyway:

```bash
# Inside custom-tools data volume:
rm /app/data/chat_archive.db /app/data/chat_archive.db-wal /app/data/chat_archive.db-shm

# From any container on ollama-net:
curl -X DELETE http://qdrant:6333/collections/kb_chat_archive
```

## 4. Operational notes

- Archive writes are best-effort. An `audrey_chat_archive_writes_total{
  result="fail"}` increase indicates a transport / 4xx /5xx between
  Audrey and custom-tools, not a chat failure — chats still complete
  normally.
- A non-zero `chunks_unindexed` count from
  `/v1/admin/chat_archive/stats` means an embed or Qdrant upsert
  failed at write time (Ollama outage, Qdrant flapping). The chunks
  remain in SQLite; a future reconcile pass can re-index them. Phase 1
  does not yet ship that reconcile.
- The internal `/chat_history/archive`, `/chat_history/prune`, and
  `/chat_history/stats` routes are deliberately hidden from
  `/openapi.json`. Audrey's tool discovery only sees the
  `chat_history_search` route, so the model cannot call the write or
  admin paths.
- The model-facing tool description steers use to deliberate lookup
  ("only when the user references something previously discussed").
  Phase 2's `CHAT_HISTORY_SEARCH_SYSTEM` prompt will reinforce the
  same rule from the system-message side; until then, the description
  is the only steering.
- A `chat_archive: write transport error` warning on the audrey-ai
  side typically means custom-tools restarted between requests. The
  `ChatArchiveClient` resolves the host URL from the registry on each
  call, so the next request after custom-tools settles will succeed.

## 5. Followups

- Phase 2 (prompt centralization + system-message composer) is the
  next planned phase. Open
  `docs/campaign-2/phase-2-task-role-prompts.md` for scope.
- Optional Grafana panel for the new
  `audrey_chat_archive_writes_total` and
  `audrey_chat_archive_write_seconds` series.
- Reconcile pass for `chunks_unindexed` rows (re-embed and upsert).
- Decide whether to keep partial-turn rows searchable or hide them.
  Default in this phase: searchable.
