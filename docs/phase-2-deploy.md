# Phase 2 — custom-tools v1 (Unraid)

**Goal:** build and run the `custom-tools` FastAPI container on `ollama-net`.
It exposes five tool endpoints (web_search, kb_search, kb_image_search,
memory_store, memory_recall) plus `/health`. The orchestrator (built in later
phases) will auto-discover them from `/openapi.json`.

**Prereqs confirmed:**
- `ollama-net` Docker network exists (Phase 1 done).
- Repo pulled to Unraid at `/mnt/user/appdata/audrey_ai_2.0` (your actual
  layout). Adjust paths in commands below if yours differs.
- **Name collision check:** you already have a `custom-tools` container from
  the old `audrey_ai/` codebase running on port `8001`. Stop and remove it
  (Docker tab → `custom-tools` → Stop → Remove) **before** building the new
  one, or rename the new one (e.g. `custom-tools-v2`). The appdata directory
  `/mnt/user/appdata/custom-tools/` is reusable — the `memory.db` schema is
  identical to whatever v1 used, or empty if that container never wrote to
  it.
- You have a Brave Search API key (free tier is fine to start).

---

## Step 1 — Create the `.env` on Unraid

The `.env` is **not** committed. Copy the example and fill it in:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
cp .env.example .env
nano .env
```

Set at minimum:
```
BRAVE_API_KEY=<your real key>
```

The other values in `.env.example` are fine as defaults for now — the tool
server only reads `BRAVE_API_KEY`, `BRAVE_CACHE_TTL_HOURS`, `AUDREY_URL`,
`AUDREY_KB_TIMEOUT_SECONDS`, and `TOOLS_DATA_DIR` from its environment.

`AUDREY_URL` defaults to `http://audrey-ai:8000` (resolves via `ollama-net`
DNS). You don't need to change it — Audrey just isn't built yet, so
`kb_search` / `kb_image_search` will return 502 until Phase 4+.

---

## Step 2 — Create the appdata directory

```bash
mkdir -p /mnt/user/appdata/custom-tools
```

This is where `memory.db` (SQLite) will live. Survives container rebuilds.

---

## Step 3 — Build the image

From the repo root on Unraid:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker build -f docker/custom-tools.Dockerfile -t audrey-custom-tools .
```

First build will take ~1–2 minutes (pip install). Subsequent builds reuse the
dependency layer and only rebuild if the source files change.

> **Unraid quirk:** don't put `:latest` in the `-t` tag. Unraid's Docker UI
> auto-appends `:latest` to the repository string when you create a
> container, so tagging as `foo:latest` produces `foo:latest:latest` and the
> container won't start. Tag with just the name.

---

## Step 4 — Create the container in Unraid UI

Docker tab → **Add Container**.

- **Name:** `custom-tools`
- **Repository:** `audrey-custom-tools`
- **Network Type:** `Custom : ollama-net`
- **Console shell command:** `bash`

Click **+ Add another Path, Port, Variable, Label or Device** as needed.

**Port:**
- Container Port `8001`, Host Port `8001`, Connection Type `TCP`

**Path:**
- Container Path `/app/data` → Host Path `/mnt/user/appdata/custom-tools`
  (access mode `Read/Write`)

**Variables:** pass the values you put in `.env`. For a simple setup, add:
- `BRAVE_API_KEY` → your key
- `AUDREY_URL` → `http://audrey-ai:8000`
- `TOOLS_DATA_DIR` → `/app/data`

(Alternatively, set **Post Arguments** → `--env-file /mnt/user/appdata/audrey_ai_2.0/.env`
so the container reads the single `.env` file. Either works; env vars in the
UI override the file.)

**No Device entries.** This container has no GPU dependency — leave it
blank. No `--gpus` / `--runtime=nvidia` needed.

Click **Apply**. Unraid will create + start the container.

---

## Step 5 — Smoke tests

From the Unraid terminal (or any LAN host):

```bash
# 1. Container is healthy
curl -s http://localhost:8001/health
# → {"status":"ok"}

# 2. OpenAPI schema exposes all five operation IDs
curl -s http://localhost:8001/openapi.json | jq -r '.paths | to_entries[] | "\(.key) \(.value | to_entries[0].value.operationId)"'
# expected (in any order):
#   /health            health
#   /web_search        web_search
#   /kb_search         kb_search
#   /kb_image_search   kb_image_search
#   /memory_store      memory_store
#   /memory_recall     memory_recall

# 3. Brave search returns real results
curl -s -XPOST http://localhost:8001/web_search \
  -H 'content-type: application/json' \
  -d '{"query":"what day is it today","count":3}' | jq '.results | length'
# → 3  (non-zero = Brave API key is valid and reachable)

# 4. Memory round-trip
curl -s -XPOST http://localhost:8001/memory_store \
  -H 'content-type: application/json' \
  -d '{"key":"smoke","value":"hello phase 2","tags":"test"}' | jq
curl -s -XPOST http://localhost:8001/memory_recall \
  -H 'content-type: application/json' \
  -d '{"key":"smoke"}' | jq '.value'
# → "hello phase 2"

# 5. Persistence check — restart and recall again
docker restart custom-tools
sleep 3
curl -s -XPOST http://localhost:8001/memory_recall \
  -H 'content-type: application/json' \
  -d '{"key":"smoke"}' | jq '.value'
# → "hello phase 2"  (SQLite file survived restart)
```

KB search endpoints will return 502 until Audrey is built (Phase 4+). That's
expected — skip them for now.

---

## Step 6 — Report back

If all five smoke tests pass, Phase 2 is complete. Reply with
"phase 2 smoke tests passed" and we'll move to Phase 3 (Qdrant).

If anything fails, paste:
- Container logs: `docker logs custom-tools --tail 100`
- The failing curl output
- `docker ps --filter name=custom-tools` so I can see the actual port/network

---

## Ops reference

- **Rebuild after code change:** pull on Unraid, then
  `docker build -f docker/custom-tools.Dockerfile -t audrey-custom-tools . && docker restart custom-tools`
- **Force dependency rebuild:** add `--no-cache` to the build command
- **View logs live:** `docker logs -f custom-tools`
- **Open a shell inside:** `docker exec -it custom-tools bash`
- **Inspect SQLite:** `sqlite3 /mnt/user/appdata/custom-tools/memory.db 'select * from memory'`
