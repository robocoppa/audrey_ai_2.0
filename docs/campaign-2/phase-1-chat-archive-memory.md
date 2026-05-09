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
2. In non-streaming `/v1/chat/completions`, archive after a response is
   produced.
3. In streaming `/v1/chat/completions`, collect the streamed assistant text
   while yielding chunks, then archive when the stream completes.
4. If archive write fails, log and continue. Chat archive must never break chat.

The archive write endpoint should be internal, not a model-callable tool.
Do not expose "store arbitrary chat history" to the model.

### Tool search path

Expose a model-callable tool for deliberate lookup:

```text
chat_history_search(query, limit=5, date_from=None, date_to=None)
```

The Audrey tool dispatcher must treat it as user-scoped, just like
`memory_search`, `memory_recall`, `memory_store`, `kb_search`, and
`kb_image_search`.

Tool results should be capped and snippet-first:

```json
{
  "query": "what did we decide about chat archive?",
  "results": [
    {
      "conversation_id": "...",
      "message_id": "...",
      "created_at": "2026-05-08T...",
      "role": "assistant",
      "snippet": "We decided to save all chat history as a searchable archive...",
      "score": 0.78
    }
  ]
}
```

Keep the default response small. A good first cap is:

- `limit`: default 5, max 10.
- `snippet_chars`: default around 500-800.
- `max_result_chars` in ReAct dispatch remains the final guardrail.

### Optional context fetch

Search snippets may not be enough for follow-up questions. Add a second tool
only if needed:

```text
chat_history_context(conversation_id, message_id=None, before=3, after=3)
```

This should fetch a small window around a known search result. It should not
return an entire long conversation by default.

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

## Data model sketch

SQLite tables:

```sql
conversations(
  conversation_id TEXT PRIMARY KEY,
  user TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

messages(
  message_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  user TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
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
  created_at TEXT NOT NULL
);
```

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

Use the same embedding model as durable memory unless testing shows a reason
to split:

```text
nomic-embed-text, 768 dimensions
```

## Conversation identity

OpenAI chat completions do not guarantee a stable conversation id. Phase 1
needs a practical rule:

1. If Open WebUI sends a stable conversation/chat id in metadata, use it.
2. Otherwise generate a server-side conversation id per request.
3. Store request/response messages even when they cannot be stitched into a
   longer OWUI conversation yet.

Do not block the phase on perfect OWUI thread identity. Search usefulness comes
from message content, timestamps, and user scope even before threading is
perfect.

## Implementation steps

1. Add chat archive settings to `tools-server/settings.py`.
2. Add `tools-server/chat_archive.py` with SQLite source-of-truth storage,
   Qdrant indexing, search, and optional context-window lookup.
3. Initialize the archive store in custom-tools lifespan.
4. Add an internal archive-write endpoint in custom-tools. Keep it out of the
   OpenAPI tool schema if possible.
5. Add model-callable `chat_history_search`.
6. Add `chat_history_search` to Audrey's user-scoped tool set in
   `src/audrey/tools/dispatch.py`.
7. Add an Audrey-side archive client and call it from non-streaming and
   streaming chat completion paths.
8. Keep archive failures best-effort: log, metric, continue.
9. Add focused tests for user scoping, result caps, archive write failure
   degradation, and search result shape.
10. Update docs and deployment notes.

## Tests

Minimum hermetic tests:

- Archive store writes conversations/messages/chunks to SQLite.
- Search filters by `user`; user A cannot see user B's archive.
- Search result `limit` and snippet caps are enforced.
- Empty queries return no results.
- Archive write failure does not fail chat completion.
- Dispatcher overwrites model-supplied `user` for `chat_history_search`.
- Streaming archive path records the final assistant text after stream
  completion.

Qdrant and Ollama calls should use fakes or mock transports in tests; the
pytest suite should remain offline.

## Deployment notes

Expected deployment impact:

- Rebuild `custom-tools`.
- Rebuild `audrey-ai`.
- New SQLite file under the custom-tools data directory.
- New Qdrant collection for chat archive chunks.
- No migration of existing chat history in Phase 1 unless Open WebUI export is
  handled separately.

Suggested smoke tests:

1. Send a normal `audrey_fast` request.
2. Confirm the archive write count increases.
3. Ask Audrey, "Search my previous chats for what I just asked."
4. Confirm the model calls `chat_history_search`.
5. Confirm the result is user-scoped and compact.

## Out of scope

- Automatic chat-archive recall on every prompt.
- Importing historical Open WebUI conversations.
- Long-term retention policy UI.
- User-facing archive browser.
- Cross-user/admin search.
- Summarizing every conversation after completion.
- Replacing `memory_store` or durable memory recall.

## Open questions

- Does Open WebUI send a stable conversation id in the request metadata Audrey
  can trust?
- Should assistant responses be chunked alone, or should chunks combine the
  nearby user question and assistant answer?
- Should archive search include system messages? Default should probably be no,
  unless a system message contains user-visible context worth searching.
- What retention policy does Bart want: keep forever, configurable days, or
  manual prune only?
- Should `chat_history_search` be a separate tool name, or should a future
  `memory_search` grow a `source="durable|chat_archive|both"` option?
