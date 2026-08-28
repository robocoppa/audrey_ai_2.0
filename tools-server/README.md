# custom-tools

Minimal FastAPI service that Audrey auto-discovers via `/openapi.json`. Any
endpoint defined here becomes a tool the models can call.

## Endpoints

- `POST /web_search` — Brave Search API proxy
- `POST /web_fetch` — SSRF-guarded readable-page fetch
- `POST /kb_search` — text query → Audrey `/v1/kb/query`
- `POST /kb_image_search` — image query → Audrey `/v1/kb/query/image`
- `POST /memory_store` — save `(key, value, tags)` to durable memory
- `POST /memory_recall` — fetch by key
- `POST /memory_search` — user-scoped semantic memory search
- `POST /chat_history_search` — user-scoped semantic chat-history search
- `POST /list_my_files` — list the authenticated user's Audrey uploads
- `POST /get_file_text` — read one bounded page of a user's file artifact

Internal archive routes are excluded from OpenAPI so models cannot call them:

- `POST /chat_history/archive` — idempotently persist and index one Q+A pair
- `POST /chat_history/prune` — run retention and reset exhausted repairs
- `GET /chat_history/stats` — counts, retry state, latest attempts and errors

The archive keeps SQLite as its source of truth. A scheduled maintainer retries
failed deterministic Qdrant upserts, processes deletion tombstones only after
Qdrant acknowledges them, and then removes the SQLite source rows. Configure
the interval, bounded batch, and retry ceiling with
`CHAT_ARCHIVE_MAINTENANCE_INTERVAL_S`, `CHAT_ARCHIVE_REPAIR_BATCH_SIZE`, and
`CHAT_ARCHIVE_MAX_RETRY_ATTEMPTS`. Nonzero `CHAT_ARCHIVE_MAX_BYTES` is
rejected at startup because a byte cap is not implemented.

Audrey supplies a stable `archive_id` and `created_at` on this internal write.
If its local delivery outbox retries after a timeout or restart, message and
chunk ids collide deliberately instead of duplicating the turn.

## Adding new tools later

Append a route to `app.py`, re-hit `POST /v1/tools/rediscover` on the
orchestrator, done. See `../docs/future-tools.md`.
