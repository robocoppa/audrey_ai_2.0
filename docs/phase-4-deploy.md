# Phase 4 — Audrey orchestrator core (Unraid)

**Goal:** build and run the `audrey-ai` container (v2 from this repo) on
`ollama-net`. This is a **minimal pass-through** — no classification, no
panels, no tools, no KB. Success = Open WebUI can pick one of
`audrey_deep` / `audrey_cloud` / `audrey_local` from the model dropdown and
get a streamed response back.

**Prereqs confirmed:**
- `ollama` (Phase 1), `custom-tools` (Phase 2), `qdrant` (Phase 3) all
  running on `ollama-net`.
- Repo pulled to `/mnt/user/appdata/audrey_ai_2.0`.
- The **legacy** `audrey-ai` container (from the old `audrey_ai/` codebase)
  is currently running on port 8000 — we'll replace it in Step 3.

---

## Step 1 — Build the image

From the repo root on Unraid:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker build -f docker/audrey.Dockerfile -t audrey-ai:latest .
```

First build installs FastAPI + a handful of small deps (≈30 s). No ML
libraries yet — those come in Phase 8 when the KB lands.

---

## Step 2 — Create the appdata directory

```bash
mkdir -p /mnt/user/appdata/audrey
```

Audrey itself keeps very little local state in Phase 4 (just logs), but
this directory is the future home for any SQLite / cache files the
orchestrator owns.

---

## Step 3 — Replace the legacy `audrey-ai` container

Your old `audrey-ai` from `audrey_ai/` is on port 8000. We need the name +
port, so stop and remove it before starting v2.

Unraid Docker tab → legacy `audrey-ai` → **Stop** → wait → click icon → **Remove**.
(Container only — the image + any old appdata stay on disk, harmless.)

---

## Step 4 — Create the new `audrey-ai` container in Unraid UI

Docker tab → **Add Container**.

- **Name:** `audrey-ai`
- **Repository:** `audrey-ai:latest`
- **Network Type:** `Custom : ollama-net`
- **Console shell command:** `bash`

**Port:**
- Container Port `8000`, Host Port `8000`, Connection Type `TCP`

**Paths:**
- Container Path `/app/config.yaml` → Host Path
  `/mnt/user/appdata/audrey_ai_2.0/config.yaml` (access mode `Read Only`)
- Container Path `/data` → Host Path `/mnt/user/appdata/audrey`
  (access mode `Read/Write`)

**Variables** (at minimum):
- `OLLAMA_HOST` → `http://ollama:11434`
- `TOOL_SERVERS` → `http://custom-tools:8001`
- `QDRANT_HOST` → `qdrant`
- `QDRANT_PORT` → `6333`
- `AUDREY_CONFIG` → `/app/config.yaml`
- `AUDREY_DATA_DIR` → `/data`

(Or `--env-file /mnt/user/appdata/audrey_ai_2.0/.env` via Post Arguments.)

**No Device entries. No `--gpus`.** Audrey itself doesn't touch the GPU —
Ollama does all the model work.

Click **Apply**.

---

## Step 5 — Smoke tests

From Unraid terminal (or any LAN host):

```bash
# 1. Health
curl -s http://localhost:8000/health | jq
# → {"status": "ok", "version": "7.0.0"}

# 2. /v1/models lists the three virtual models
curl -s http://localhost:8000/v1/models | jq '.data[].id'
# → "audrey_deep"
#   "audrey_cloud"
#   "audrey_local"

# 3. Non-streaming pass-through hits Ollama and returns an OpenAI envelope
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "audrey_deep",
    "messages": [{"role":"user","content":"reply with just the word: pong"}],
    "stream": false
  }' | jq '.choices[0].message.content, .usage, .model'
# → some variant of "pong"  (whitespace-insensitive)
#   { prompt_tokens, completion_tokens, total_tokens }
#   "audrey_deep"

# 4. Streaming works and emits an OpenAI-compatible SSE stream
curl -s -N -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "audrey_local",
    "messages": [{"role":"user","content":"count from 1 to 5"}],
    "stream": true
  }' | head -20
# → sequence of `data: {...}` lines, ending with `data: [DONE]`

# 5. Confirm Ollama received the request
docker logs ollama --tail 10 | grep -iE "chat|qwen|kimi" | tail -3
# → shows a recent /api/chat call from the audrey-ai container

# 6. Unknown model is rejected with 400
curl -s -o /dev/null -w '%{http_code}\n' -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"not-a-real-model","messages":[{"role":"user","content":"hi"}]}'
# → 400
```

If all six pass, Phase 4 is complete.

---

## Step 6 — Point Open WebUI at v2 (optional, quick)

Your `open-webui` container is already running but configured for the legacy
Audrey. To swap it over:

Unraid → `open-webui` → **Edit** →
- `OPENAI_API_BASE_URL` → `http://audrey-ai:8000/v1`
- `OPENAI_API_KEY` → `unused` (any non-empty string)

**Apply**. Refresh Open WebUI in the browser. The model dropdown should now
show `audrey_deep`, `audrey_cloud`, `audrey_local`. Pick one and send a
prompt — you should get a streamed reply end-to-end through the new stack.

(If you'd rather leave the legacy wiring in place until later phases are
feature-complete, skip this step and point back to v2 at Phase 9.)

---

## Step 7 — Report back

Reply with "phase 4 smoke tests passed" (and any WebUI observations) and
we'll move to Phase 5 (classification + complexity gate + fast path).

If anything fails, paste:
- `docker logs audrey-ai --tail 100`
- The failing curl output
- `docker ps --filter name=audrey-ai`

---

## Ops reference

- **Hot-reload config:** `docker restart audrey-ai` (config.yaml is read at
  startup; no hot-reload yet).
- **Rebuild after code change:** `git pull && docker build -f docker/audrey.Dockerfile -t audrey-ai:latest . && docker restart audrey-ai`
- **Live logs:** `docker logs -f audrey-ai`
- **Shell inside:** `docker exec -it audrey-ai bash`
- **Which concrete model did it use?** Check the log line
  `chat.completions model=audrey_deep -> <concrete> stream=...` or the
  `system_fingerprint` field in the response (`audrey-7.0.0/<concrete>`).
