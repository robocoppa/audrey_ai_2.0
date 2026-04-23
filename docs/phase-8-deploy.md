# Phase 8 — KB ingest + `kb_search` end-to-end with Qdrant (Unraid)

**Goal:** a first-class knowledge base: point Audrey at a directory of
docs/images, she chunks + embeds them, writes vectors to Qdrant, and the
fast-path ReAct loop (Phase 7) pulls them back via `kb_search` /
`kb_image_search`.

What's new vs Phase 7:

- **`kb/qdrant.py`** — async wrapper over `qdrant-client`. Two collections:
  `kb_text` (768-d, cosine, nomic-embed-text via Ollama) and `kb_images`
  (512-d, cosine, CLIP ViT-B-32 via sentence-transformers). Collections
  are created eagerly at startup so dim mismatches surface now, not on
  first query. Deterministic point IDs (`uuid5(source:kind:idx)`) make
  re-ingest idempotent.
- **`kb/embed.py`** — `TextEmbedder` batches through `/api/embed`;
  `ImageEmbedder` loads CLIP once (cached in `/root/.cache/clip`,
  bind-mounted) and runs `encode` on `asyncio.to_thread`.
- **`kb/chunk.py`** — cl100k_base tokenizer, `chunk_tokens=1000 /
  overlap=100` (same tokenizer the complexity gate uses). Loaders for
  `.md .txt .rst .log .csv .pdf .docx .html`.
- **`kb/ingest.py`** — recursive walker; per-file `ingest_text_file` does
  `delete_by_source` before upsert so shrinking files don't leave orphan
  chunks. Images are one point per file (CLIP returns one vector).
- **`kb/watcher.py`** — watchdog observer bridged to asyncio via
  `call_soon_threadsafe`; debounces per-path (default 2 s) so `cp -r`
  becomes O(files) ingests instead of O(events). Env-gated behind
  `KB_WATCHER_ENABLED=1`.
- **`routes/kb.py`** — `POST /v1/kb/query`, `POST /v1/kb/query/image`,
  `POST /v1/kb/ingest`, `GET /v1/kb/stats`. The Phase 7 `kb_search` /
  `kb_image_search` tools in `custom-tools` proxy to these two query
  endpoints, so the ReAct loop comes right back here.
- **`audrey-ingest` CLI** — `audrey-ingest [PATH ...]`, `--stats`,
  `--purge SOURCE`. Runs inside the `audrey-ai` container where Qdrant
  and Ollama are reachable.

**Prereqs confirmed:**

- Phase 7 done; `audrey-ai` healthy, 5 tools registered.
- `qdrant` container running under Unraid UI, reachable as
  `qdrant:6333` on `ollama-net`.
- `nomic-embed-text` pulled in Ollama (`docker exec ollama ollama list |
  grep nomic`). If missing: `docker exec ollama ollama pull nomic-embed-text`.
- Dataset mount exists:
  host `/mnt/user/knowledge` → container `/datasets`. Subdirs under
  `/mnt/user/knowledge/` (geology, botany, bushcraft, fishing, first-aid,
  wilderness-first-aid, herbal-medicine, hunting, survival, bjj, mma,
  powerapps, servicenow) map to `/datasets/<topic>`. Drop at least one
  `.md`/`.pdf` and one `.jpg` into `geology/` for the smoke tests.
- CLIP model cache mount exists:
  host `/mnt/user/appdata/clip-cache` → container `/root/.cache/clip`.
  First image ingest downloads ~380 MB; subsequent starts reuse the cache.

---

## Step 1 — Rebuild & restart

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
```

New Python deps in `audrey.Dockerfile`: `qdrant-client`,
`sentence-transformers`, `pypdf`, `python-docx`, `beautifulsoup4`,
`lxml`, `watchdog`, `pillow`. First build pulls sentence-transformers
+ torch CPU — allow ~5 min.

If the new log shape (Step 2.1) doesn't appear, rebuild with `--no-cache`.

---

## Step 2 — Smoke tests

### 2.1 — Collections exist and app reports them at boot

```bash
docker logs audrey-ai 2>&1 | grep -E "qdrant:|ready:" | tail -5
# Expected:
#   qdrant: created collection kb_text (dim=768)      (first boot only)
#   qdrant: created collection kb_images (dim=512)    (first boot only)
#   ready: ... qdrant=qdrant:6333; kb_watcher=off; pipeline=compiled
```

Direct check:

```bash
curl -s http://localhost:8000/v1/kb/stats | jq
# Expected:
# {
#   "collections": {"kb_text": 0, "kb_images": 0},
#   "text_collection": "kb_text",
#   "image_collection": "kb_images"
# }
```

If `kb_text: -1`, Qdrant is unreachable. Verify from inside the
container: `docker exec audrey-ai curl -fsS http://qdrant:6333/collections`.

### 2.2 — Ingest a dataset directory

```bash
curl -s -XPOST http://localhost:8000/v1/kb/ingest \
  -H 'content-type: application/json' \
  -d '{"paths":["/datasets/geology"]}' | jq
# Expected (shape):
# {
#   "roots": ["/datasets/geology"],
#   "files_seen": N,
#   "files_text": T,
#   "files_image": I,
#   "chunks_text": C,
#   ...
# }
```

Log:

```bash
docker logs audrey-ai --tail 60 | grep -E "kb\.ingest|kb\.chunk"
# Expected:
#   kb.ingest: root=/datasets/geology seen=... text=...(... chunks) images=...
#   kb.ingest (http): {...}
```

Re-run the same ingest — counts in `/v1/kb/stats` should not double.
That's the deterministic-UUID idempotence at work.

### 2.3 — `audrey-ingest` CLI works

```bash
docker exec audrey-ai audrey-ingest --stats
# Expected:
#   kb_text: <N>
#   kb_images: <M>

docker exec audrey-ai audrey-ingest /datasets/geology
# Expected: "ingest complete: seen=... text=...(chunks) images=... skipped=... errors=0"
```

### 2.4 — Direct text query returns hits

```bash
curl -s -XPOST http://localhost:8000/v1/kb/query \
  -H 'content-type: application/json' \
  -d '{"query":"what rocks form under high pressure","top_k":3}' | jq '.results[] | {score, source, chunk_idx, text: .text[0:120]}'
# Expected: up to 3 hits, scores in (0,1], `source` is an absolute path
# under /datasets/geology, `text` is a real excerpt.
```

If `results: []`, either the ingest didn't run or the query is
genuinely unrelated — try a prompt that quotes a phrase you know is in
one of the files.

### 2.5 — `kb_search` via the ReAct loop

Tool-routed version of 2.4: let a tool-capable model decide to call it.

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"use your kb_search tool to look up what the dataset says about metamorphic rocks, then summarize in 2 sentences"}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:240]'

docker logs audrey-ai --tail 80 | grep -E "react:|dispatch:|chat\.completions"
# Expected:
#   react: round 1: model produced 1 tool_call(s): ['kb_search']
#   dispatch: kb_search ok in ..s (... chars)
#   react: round 2: no tool_calls → returning answer
#   chat.completions ... mode=fast ... tool_rounds=1 tool_calls=[kb_search]
```

### 2.6 — Image query (URL)

Pick an image in `/datasets/geology`, or use any public image URL that
resembles what you ingested.

```bash
curl -s -XPOST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://upload.wikimedia.org/wikipedia/commons/9/91/Granite_Yosemite_P1160483.jpg","top_k":3}' | jq '.results[] | {score, source}'
# Expected: up to 3 image hits from kb_images, highest-scoring first.
```

First call is slow (~30–60 s) — CLIP weights download from
sentence-transformers into `/root/.cache/clip`. Subsequent calls are
sub-second.

### 2.7 — `kb_image_search` via the ReAct loop

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"use kb_image_search to find an image of granite in the knowledge base and describe what source matched"}],
    "stream": false
  }' | jq -r '.choices[0].message.content[0:240]'

docker logs audrey-ai --tail 80 | grep "dispatch: kb_image_search"
# Expected: dispatch: kb_image_search ok in ...
```

### 2.8 — Watcher picks up new files (env-gated)

Enable the watcher via `.env`:

```bash
echo 'KB_WATCHER_ENABLED=1' >> .env
docker compose up -d --build audrey-ai

docker logs audrey-ai 2>&1 | grep "kb.watcher" | tail -3
# Expected: kb.watcher: watching 1 root(s): ['/datasets/geology']
```

Drop a new markdown file and wait past the 2 s debounce:

```bash
echo "Obsidian is a dark volcanic glass." > \
  /mnt/user/knowledge/geology/obsidian.md
sleep 4

docker logs audrey-ai --tail 20 | grep "kb.watcher"
# Expected: kb.watcher: reingested text /datasets/geology/obsidian.md -> 1 chunks
```

Then query it:

```bash
curl -s -XPOST http://localhost:8000/v1/kb/query \
  -H 'content-type: application/json' \
  -d '{"query":"volcanic glass","top_k":1}' | jq '.results[0] | {score, source, text}'
# Expected: obsidian.md as the top hit.
```

### 2.9 — Purge removes a source

```bash
docker exec audrey-ai audrey-ingest --purge /datasets/geology/obsidian.md
curl -s -XPOST http://localhost:8000/v1/kb/query \
  -H 'content-type: application/json' \
  -d '{"query":"volcanic glass","top_k":1}' | jq '.results'
# Expected: either [] or a non-obsidian hit — the obsidian.md point is gone.
```

---

## Step 3 — Report back

If 2.1–2.9 all pass, Phase 8 is done. Reply with
**"phase 8 smoke tests passed"** and we'll move to Phase 9.

If anything fails, paste:

- `docker logs audrey-ai --tail 120`
- `docker logs qdrant --tail 40`
- The failing curl output
- Which test number failed

---

## Troubleshooting

- **`qdrant: ensure_collections failed`** at boot: Qdrant is down or on
  a different network. Check `docker inspect qdrant | grep -A3
  Networks` — it must include `ollama-net`.

- **`400 Bad Request: wrong vector size`** on upsert: a collection was
  created with a different dim before Phase 8 deploy. Drop it and let
  startup recreate it:
  `curl -XDELETE http://qdrant:6333/collections/kb_text` (from inside
  the container), then restart `audrey-ai`.

- **Text embedding returns 404/timeout**: `nomic-embed-text` not pulled
  in Ollama. `docker exec ollama ollama pull nomic-embed-text`.

- **First image call hangs / times out**: CLIP weights download. Tail
  `docker logs audrey-ai -f` — you'll see `clip: loading
  clip-ViT-B-32`. Let it finish once; the cache persists across
  restarts because `/root/.cache/clip` is bind-mounted.

- **Watcher fires but ingest does nothing**: the file suffix isn't in
  the supported list. `kb/chunk.py` supports md/txt/rst/log/csv/pdf/
  docx/html for text and jpg/jpeg/png/webp/bmp/gif/tif/tiff for images.

- **`kb_search` tool-call returns no hits while direct `/v1/kb/query`
  does**: the tool in `custom-tools` calls back into
  `http://audrey-ai:8000/v1/kb/query`. Verify that URL is reachable
  from the tools container:
  `docker exec custom-tools curl -fsS http://audrey-ai:8000/health`.

- **Re-ingest doubles counts**: point IDs aren't deterministic — check
  that `normalize_source()` resolved to the same absolute path on both
  runs. Symlink flips or mount-path changes between containers will
  break idempotence.
