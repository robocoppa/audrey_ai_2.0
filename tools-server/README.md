# custom-tools

Minimal FastAPI service that Audrey auto-discovers via `/openapi.json`. A
model-visible endpoint requires both the explicit `tools` OpenAPI tag and a
matching declaration in Audrey's central capability policy.

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
- `POST /calculate` — decimal arithmetic, unit conversion, linear/quadratic equations
- `POST /analyze_table` — read-only owned CSV/XLSX rows, totals, groups and duplicates
- `POST /inspect_pdf_page` — visual observations from one owned PDF page
- `POST /open_chat_history` — paged owned saved conversations, respecting deletion cutoffs
- `POST /inspect_video_frame` — visual observations from a retained video at a timestamp

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

Runtime component state is published in `/openapi.json`. Qdrant failure leaves
the process health endpoint, stateless web tools, file catalogue, artifact
reads, and the SQLite archive source available. Qdrant-dependent model routes
return a component-specific HTTP 503 and disappear from Audrey's callable set
until the background supervisor and Audrey rediscovery observe recovery.

Audrey supplies a stable `archive_id` and `created_at` on this internal write.
If its local delivery outbox retries after a timeout or restart, message and
chunk ids collide deliberately instead of duplicating the turn.

## Memory backups

The configured memory collection is snapshotted in the background at startup
and daily. Seven private bundles are retained under the persistent
`/app/data/backups/memory`; each also includes an online archive SQLite copy
preserving account purge receipts. Backup failures retry without blocking chat.
`python /app/memory_backup.py verify --latest` checks a backup by restoring a
unique disposable collection, validating its payload ownership, and deleting it.
It cannot restore production. Deployment, verification and production recovery
limits are in [Box operations](../docs/reference/box-operations.md#10-memory-backups-and-isolated-restore-verification).

## Adding new tools later

Add a tagged route or include its router in `app.py`, declare its identity
and dependencies in `src/audrey/tools/discovery.py`, and extend focused checks.
Rebuild the owning services and rediscover tools. The
[main plan](../docs/campaign-4/README.md) owns remaining work;
[Phase 19](../docs/campaign-4/phase-19-evidence-tools.md) defines the evidence tools.
