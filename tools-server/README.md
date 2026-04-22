# custom-tools

Minimal FastAPI service that Audrey auto-discovers via `/openapi.json`. Any
endpoint defined here becomes a tool the models can call.

## v1 endpoints (built in Phase 2)

- `POST /web_search` — Brave Search API proxy
- `POST /kb_search` — text query → Audrey `/v1/kb/query`
- `POST /kb_image_search` — image query → Audrey `/v1/kb/query/image`
- `POST /memory_store` — save `(key, value, tags)` to local SQLite
- `POST /memory_recall` — fetch by key

## Adding new tools later

Append a route to `app.py`, re-hit `POST /v1/tools/rediscover` on the
orchestrator, done. See `../docs/future-tools.md`.
