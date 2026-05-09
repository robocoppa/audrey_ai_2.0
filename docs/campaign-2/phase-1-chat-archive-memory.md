# Campaign 2 Phase 1 - Searchable chat archive

## Goal

Add a per-user chat archive that saves conversation history and makes it
searchable through Audrey's existing memory/tool architecture.

This phase is about **archive + search**, not automatic prompt bloat.

Audrey already has durable per-user memory:

- `memory_store` saves explicit user memories.
- `memory_search` searches those memories.
- `pipeline/memory.py` can inject a few durable memory hits before classify.

What is missing is a searchable record of ordinary chat history. Campaign 2
Phase 1 should add that archive without turning every prior conversation into
automatic context.

## Product rules

- Save chat history automatically as an archive.
- Save durable memories only when the user asks or the model uses
  `memory_store` for an explicit durable fact.
- Keep token use a primary constraint.
- Do not inject chat-history hits into every prompt by default.
- Search chat history through a user-scoped tool call.
- Return compact snippets by default, not full transcripts.
- Enforce the authenticated pipeline user server-side; the model must never be
  able to search another user's archive by supplying a different `user`.
- Archive writes are best-effort: log, count, continue. Chat must never fail
  because the archive failed.

The mental model:

```text
durable memory
  -> small, curated, explicit-save
  -> eligible for today's memory_recall prompt hint

chat archive
  -> broad, automatic, transcript-derived
  -> tool-searchable on demand
  -> not auto-injected by default
```

## Architecture

Keep this inside the existing memory/tool shape instead of creating a parallel
system.

### Storage owner

`custom-tools` should own the chat archive storage and search API, alongside
the current Qdrant-backed memory store.

Recommended shape:

- SQLite is the source-of-truth archive for raw conversations and messages.
- Qdrant is the semantic search index for archive chunks.
- The archive lives under `TOOLS_DATA_DIR`, near the existing memory data.
- Qdrant collection name should be configurable, for example
  `CHAT_ARCHIVE_COLLECTION=kb_chat_archive`.

Why SQLite plus Qdrant:

- SQLite is good for durable raw records, deletion, export, retention, and
  exact conversation lookup.
- Qdrant is good for semantic search.
- If the vector index needs rebuilding later, SQLite can repopulate it.

### Audrey capture path

Audrey sees the authenticated user, request messages, selected virtual model,
concrete model, and final response. That makes Audrey the right place to decide
what should be archived.

Plan:

1. Add a small `ChatArchiveClient` in Audrey that calls an internal
   custom-tools archive endpoint.
2. Wrap the response generator (streaming and non-streaming) with a single
   `StreamCollector` helper so capture lives in one place. See "Streaming
   capture" below for details.
3. After the response is fully produced (or the client disconnects), call the
   archive client with the new user turn, the assistant response, and the
   captured stream metadata.
4. If the archive write fails, log, increment a metric, and continue. Chat
   archive must never break chat.

The archive write endpoint should be internal, not a model-callable tool.
Do not expose "store arbitrary chat history" to the model.

### Streaming capture

[src/audrey/routes/openai.py](src/audrey/routes/openai.py) has three
streaming branches today: deep-with-banners, fast token stream, and the
ReAct one-chunk-after-loop. Each emits text differently and at least one
already handles client-disconnect explicitly. Capture must live in one
place or it will drift.

Approach:

- Add a `StreamCollector` (or `archive_capture()` async-generator wrapper)
  that wraps the SSE generator. It yields each SSE frame unchanged to the
  client and, in parallel, accumulates only the assistant *content deltas*
  into a buffer.
- Banner frames, role deltas, tool-call frames, and finish-reason frames
  are not appended to the buffer. Only `choices[0].delta.content` strings
  are.
- On normal completion, the wrapper triggers an archive write with the
  full buffered text and `partial=False`.
- On client disconnect (the existing `asyncio.CancelledError` path in
  `_stream_deep_with_banners`), the wrapper still triggers an archive
  write with whatever text was emitted, marked `partial=True`. A partial
  turn is more useful than no turn.
- Non-streaming requests use the same client by calling it directly with
  the full assistant text and `partial=False`.

### Tool search path

Expose a model-callable tool for deliberate lookup:

```text
chat_history_search(query, limit=5, date_from=None, date_to=None)
```

The Audrey tool dispatcher must treat it as user-scoped. Add
`"chat_history_search"` to `_USER_SCOPED_TOOLS` in
[src/audrey/tools/dispatch.py](src/audrey/tools/dispatch.py) — the
existing `args["user"] = user_id` overwrite then applies automatically.

The custom-tools `MemorySearchRequest`-style schema makes `user` a required
field that the dispatcher overrides; `chat_history_search` should follow
the same pattern.

Tool description text the model sees must steer use:

```text
Search this user's prior conversations with you. Use only when the user
references something previously discussed, or when the question requires
a specific prior decision. Do not call to personalize ordinary answers.
```

Tool results should be capped and snippet-first:

```json
{
  "query": "what did we decide about chat archive?",
  "results": [
    {
      "conversation_id": "...",
      "chunk_id": "...",
      "created_at": "2026-05-08T...",
      "snippet": "We decided to save all chat history as a searchable archive...",
      "score": 0.78
    }
  ]
}
```

Default caps:

- `limit`: default 5, max 10.
- `snippet_chars`: default 600.
- `max_result_chars` in ReAct dispatch remains the final guardrail.

### Optional context fetch

Search snippets may not be enough for follow-up questions. Add a second tool
only if needed:

```text
chat_history_context(conversation_id, chunk_id=None, before=3, after=3)
```

This should fetch a small window around a known search result. It should not
return an entire long conversation by default. Defer to a follow-up unless
search snippets prove insufficient in real use.

## Integration with existing memory recall

Do **not** replace the current durable memory behavior.

Current behavior:

```text
memory_recall node
  -> calls memory_search
  -> injects a small durable-memory system message
```

Campaign 2 Phase 1 behavior:

```text
memory_recall node
  -> still calls durable memory_search only
  -> does not search chat archive by default

ReAct tool use
  -> model can call chat_history_search when the user asks about prior chats
```

This keeps normal token use unchanged. Chat archive tokens appear only when a
tool call is actually useful.

Future option, not Phase 1 default:

```yaml
agentic:
  memory:
    archive_auto_recall:
      enabled: false
      top_k: 1
      max_snippet_chars: 300
      min_score: 0.75
```

Leave that disabled unless real use shows that tiny archive auto-recall is
worth the token cost.

## Conversation identity

OpenAI chat completions do not guarantee a stable conversation id, but Open
WebUI does send one when present. Phase 1 must check first, not skip.

Resolution order, in this order:

1. Top-level `chat_id` field in the OWUI request body.
2. `metadata.chat_id` if OWUI nests it there.
3. `messages[-1].metadata.chat_id` (per-message metadata).
4. Fallback: deterministic id derived from `(user, sorted prefix-hash of
   message contents)` so a continued conversation in the same OWUI tab
   still stitches together across requests.
5. Last resort: fresh server-side UUID.

Treat steps 1–3 as the primary path. Spike them at the start of
implementation; if OWUI does not send one, fall through to step 4. Do not
ship Phase 1 with only step 5 — it makes every request its own
single-message "conversation" and the archive becomes useless for thread
search.

## Message dedup

Each `/v1/chat/completions` request includes the entire prior history. Without
dedup the same user turn gets archived N times.

Rule:

- For the inbound user turn(s): only archive *new* messages — those whose
  hash is not already in `messages` for this `(user, conversation_id)`.
- Compute `message_id` as `sha256(user || conversation_id || role || content
  || created_at_minute_bucket)[:32]` so retries within the same minute
  collapse.
- Store with `INSERT OR IGNORE` on `message_id` PRIMARY KEY.
- The assistant response gets a fresh `message_id` derived from the same
  hash inputs, so a retried stream collapses too.

This makes archive writes idempotent under OWUI retries and full-history
re-sends.

## Chunking

Chunk Q+A pairs together, not single messages. A user turn alone has no
context; an assistant turn alone has no question. Search relevance suffers
either way.

Default rule:

- One chunk per user-turn + following-assistant-turn pair.
- Soft cap: 1500 chars per chunk; if the pair exceeds it, split the
  assistant turn at sentence boundaries with 100-char overlap.
- System messages are not chunked unless the user-visible content
  references them.

Embed each chunk once at write time; SQLite holds the raw text so a
reindex is possible.

## Embedding

Reuse `nomic-embed-text` at 768 dims to match durable memory. Two caveats
worth flagging up front:

- Archive volume × embed cost is meaningfully higher than memory_store
  volume. Batch-embed multiple chunks per archive write where the request
  produced multiple Q+A pairs.
- Archive embed traffic now adds to custom-tools' Ollama dependency. If
  Ollama is down, archive writes degrade to "stored in SQLite, indexed
  later" — see retention/reconcile below.

## Retention

Default to "keep forever" with a kill switch wired in from day one.

Settings:

```text
CHAT_ARCHIVE_RETENTION_DAYS=0     # 0 = forever
CHAT_ARCHIVE_MAX_BYTES=0          # 0 = no cap
```

A small `chat_archive_prune` admin route or CLI removes messages older
than the retention window and drops their Qdrant points. Keeping the
wiring in place from Phase 1 avoids a later migration.

## Observability

Audrey already runs Prometheus. Add:

- `chat_archive_writes_total{result="ok|fail|partial"}` (Audrey-side).
- `chat_archive_search_total{result="ok|fail|empty"}` (custom-tools-side).
- `chat_archive_search_latency_seconds` (custom-tools-side, histogram).
- `chat_archive_messages_total` (gauge of SQLite row count, scraped on
  request or refreshed periodically).

"Best-effort" failures must not be silent.

## Data model sketch

SQLite tables:

```sql
conversations(
  conversation_id TEXT PRIMARY KEY,
  user TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_message_at TEXT NOT NULL
);

messages(
  message_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  user TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  archived_at TEXT NOT NULL,
  partial INTEGER NOT NULL DEFAULT 0,
  virtual_model TEXT,
  concrete_model TEXT,
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0
);

archive_chunks(
  chunk_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  user TEXT NOT NULL,
  message_ids_json TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  indexed_at TEXT
);

CREATE INDEX idx_messages_user_conv ON messages(user, conversation_id);
CREATE INDEX idx_messages_user_created ON messages(user, created_at);
CREATE INDEX idx_chunks_user ON archive_chunks(user);
```

`indexed_at` is null when SQLite has the chunk but Qdrant does not — the
reconcile path picks those up.

Qdrant payload per archive chunk:

```json
{
  "user": "alice@example.com",
  "conversation_id": "...",
  "chunk_id": "...",
  "message_ids": ["..."],
  "created_at": "2026-05-08T...",
  "text": "compact searchable chunk"
}
```

## Tool description and Phase 2 hand-off

The model-facing tool description (above) is the *only* steering chat-history
search gets in Phase 1. Phase 2 will add a `CHAT_HISTORY_SEARCH_SYSTEM`
constant in `prompts.py` that reinforces the same rule from the system-message
side. Until Phase 2 ships, the tool description is load-bearing — write it
carefully.

## Implementation steps

1. Spike OWUI conversation-id resolution. Confirm which of steps 1–3 in
   "Conversation identity" actually fire. Pin the result before writing
   the rest.
2. Add chat archive settings to `tools-server/settings.py`
   (`CHAT_ARCHIVE_*`, including retention knobs).
3. Add `tools-server/chat_archive.py` with SQLite source-of-truth storage,
   chunking, Qdrant indexing, search, and the prune helper.
4. Initialize the archive store in custom-tools lifespan.
5. Add an internal archive-write endpoint in custom-tools. Keep it out of
   the OpenAPI tool schema (separate router, or `include_in_schema=False`).
6. Add model-callable `chat_history_search` with the steering description.
7. Add `"chat_history_search"` to `_USER_SCOPED_TOOLS` in
   [src/audrey/tools/dispatch.py](src/audrey/tools/dispatch.py).
8. Add a `ChatArchiveClient` in Audrey plus the `StreamCollector` wrapper.
9. Wire the wrapper into all streaming branches in
   [src/audrey/routes/openai.py](src/audrey/routes/openai.py) and the
   non-streaming response path. One call site per branch.
10. Wire the Prometheus metrics.
11. Add the prune admin route or CLI.
12. Keep archive failures best-effort: log, metric, continue.
13. Add focused tests (see "Tests").
14. Update docs and deployment notes.

## Tests

Minimum hermetic tests:

- Archive store writes conversations/messages/chunks to SQLite.
- Search filters by `user`; user A cannot see user B's archive.
- Search result `limit` and snippet caps are enforced.
- Empty queries return no results.
- Archive write failure does not fail chat completion.
- Dispatcher overwrites model-supplied `user` for `chat_history_search`.
- `StreamCollector` archives only assistant content deltas, not banner or
  tool-call frames.
- Client-disconnect mid-stream still triggers a `partial=True` archive
  write with whatever was emitted.
- Re-sending the same conversation history does not duplicate messages
  (dedup by deterministic `message_id`).
- Q+A pair chunking: a request with one user turn and one assistant turn
  produces exactly one chunk; an oversized assistant turn splits with
  overlap.
- OWUI `chat_id` resolution: when present in the request body, it wins
  over the deterministic-hash fallback.
- Prune removes messages older than `CHAT_ARCHIVE_RETENTION_DAYS` and
  drops their Qdrant points.

Qdrant and Ollama calls should use fakes or mock transports in tests; the
pytest suite should remain offline.

## Deployment notes

Expected deployment impact:

- Rebuild `custom-tools`.
- Rebuild `audrey-ai`.
- New SQLite file under the custom-tools data directory.
- New Qdrant collection for chat archive chunks.
- New Prometheus series under `chat_archive_*`.
- No migration of existing chat history in Phase 1 unless Open WebUI export is
  handled separately.

Suggested smoke tests:

1. Send a normal `audrey_fast` request. Confirm `chat_archive_writes_total`
   increases and the SQLite row count goes up.
2. Send the same request again with the full prior history. Confirm the
   message count goes up by exactly 2 (new user turn + new assistant
   turn), not by the full history length.
3. Cancel a streaming `audrey_deep` request mid-flight. Confirm a
   `partial=True` row was written.
4. Ask Audrey, "Search my previous chats for what I just asked." Confirm
   the model calls `chat_history_search` and the result is user-scoped
   and compact.
5. With a second user (or test fixture), confirm user A's search returns
   zero hits from user B's archive.

## Out of scope

- Automatic chat-archive recall on every prompt (config knob is wired but
  defaults off).
- Importing historical Open WebUI conversations.
- User-facing archive browser.
- Cross-user/admin search.
- Summarizing every conversation after completion.
- Replacing `memory_store` or durable memory recall.
- `chat_history_context` window-fetch tool — defer until search snippets
  prove insufficient.

## Open questions

- Should archive search include system messages? Default is no, unless a
  system message contains user-visible context worth searching.
- Should `chat_history_search` ever grow into `memory_search` with a
  `source="durable|chat_archive|both"` option? Phase 1 keeps them
  separate; revisit only if real use shows the merged shape is clearer
  for the model.
- Should the partial-turn rows be searchable, or only durable rows?
  Default: searchable, since a partial answer is still evidence of what
  was discussed.
