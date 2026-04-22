# Phase 3 — Qdrant vector DB (Unraid)

**Goal:** run the `qdrant/qdrant` container on `ollama-net` with persistent
storage under `/mnt/user/appdata/qdrant`. Two collections will be created
later by Audrey at startup (Phase 8): `kb_text` (768-d, nomic-embed-text)
and `kb_images` (512-d, CLIP ViT-B/32). For now we just need the server up
and reachable from the other containers.

**No Dockerfile** — Qdrant publishes a ready-to-run image, so this phase is
100% Unraid UI.

**Prereqs confirmed:**
- `ollama-net` Docker network exists (Phase 1).
- Port `6333` (HTTP) and `6334` (gRPC) are free on the host.

---

## Step 1 — Create the appdata directory

```bash
mkdir -p /mnt/user/appdata/qdrant
```

Qdrant will write `storage/`, `snapshots/`, and its WAL here. Survives
container rebuilds; the data in this directory is the actual KB index.

---

## Step 2 — Create the container in Unraid UI

Docker tab → **Add Container**.

- **Name:** `qdrant`
- **Repository:** `qdrant/qdrant` *(leave the tag blank — Unraid auto-appends `:latest` and for a public image that's what we want)*
- **Network Type:** `Custom : ollama-net`
- **Console shell command:** `sh` (Qdrant image is minimal, no bash)

**Ports** (two entries):
- Container Port `6333`, Host Port `6333`, `TCP` — HTTP / REST API
- Container Port `6334`, Host Port `6334`, `TCP` — gRPC (Audrey uses this)

**Path:**
- Container Path `/qdrant/storage` → Host Path `/mnt/user/appdata/qdrant`
  (access mode `Read/Write`)

**Variables:** none required. Optional:
- `QDRANT__SERVICE__GRPC_PORT` = `6334` (default; only set if you change it)
- `QDRANT__LOG_LEVEL` = `INFO` (default)

**No Device entries. No `--gpus`.** Qdrant is CPU-only in this deployment —
the 768-d / 512-d vectors are small enough that HNSW on CPU is fast.

Click **Apply**. First-run downloads the image (~120 MB) and boots in a few
seconds.

---

## Step 3 — Smoke tests

From the Unraid terminal (or any LAN host):

```bash
# 1. Service alive
curl -s http://localhost:6333/ | jq '{title, version}'
# → {"title": "qdrant - vector search engine", "version": "..."}

# 2. Readiness (returns a plain string "ok")
curl -s http://localhost:6333/readyz
# → all shards are ready

# 3. Collections list (empty on fresh install — this is the expected state;
#    Audrey will create kb_text and kb_images at startup in Phase 8)
curl -s http://localhost:6333/collections | jq
# → {"result":{"collections":[]},"status":"ok","time":...}

# 4. Create-and-delete round-trip proves the storage mount is writable
curl -s -XPUT http://localhost:6333/collections/smoke \
  -H 'content-type: application/json' \
  -d '{"vectors":{"size":4,"distance":"Cosine"}}' | jq '.status'
# → "ok"

curl -s http://localhost:6333/collections | jq '.result.collections[].name'
# → "smoke"

curl -s -XDELETE http://localhost:6333/collections/smoke | jq '.status'
# → "ok"

# 5. Verify from inside ollama-net (so we know Audrey will resolve it later).
#    Uses the custom-tools container as a convenient in-network probe.
docker exec custom-tools python -c "
import httpx
r = httpx.get('http://qdrant:6333/readyz', timeout=5)
print(r.status_code, r.text)
"
# → 200 all shards are ready
```

---

## Step 4 — Report back

If tests 1–5 all pass, Phase 3 is complete. Reply with
"phase 3 smoke tests passed" and we'll move to Phase 4 (orchestrator core:
FastAPI app, `/v1/models`, pass-through `/v1/chat/completions`).

If anything fails, paste:
- `docker logs qdrant --tail 100`
- The failing curl output
- `docker ps --filter name=qdrant` so I can see port bindings

---

## Ops reference

- **Storage size:** small at first (empty collections). Once the geology KB is
  ingested (Phase 8) it will grow with your dataset. BTRFS RAID0 has no
  redundancy — if this data matters long-term, snapshot it occasionally:
  `curl -XPOST http://localhost:6333/collections/kb_text/snapshots` (per
  collection). Snapshots land under `/mnt/user/appdata/qdrant/snapshots/`.
- **Upgrade:** Qdrant moves fast. Pin a version if stability matters
  (`qdrant/qdrant:v1.12.4` etc.). For a dev setup, `:latest` is fine.
- **Web UI:** Qdrant ships a built-in dashboard at
  `http://<unraid-ip>:6333/dashboard` — handy for eyeballing collections
  once the KB is populated.
- **Reset the DB:** `docker stop qdrant && rm -rf /mnt/user/appdata/qdrant/* && docker start qdrant`.
  Empty on next boot; Audrey will recreate collections at startup.
