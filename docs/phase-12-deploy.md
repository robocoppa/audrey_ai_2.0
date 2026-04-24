# Phase 12 — Semantic memory (Qdrant + nomic-embed-text)

**Goal:** upgrade Phase 11's SQL-`LIKE` keyword memory to cosine-similarity
search over nomic-embed-text vectors stored in Qdrant. "What **hardware**
am I running?" now finds a memory stored as "AMD **Threadripper 7970X
workstation**" without the words overlapping.

What's new vs Phase 11:

- **`custom-tools` uses Qdrant as the memory store.** A new collection
  `kb_memory` (768-d, cosine) holds one point per `(user, key)`. Point IDs
  are deterministic `uuid5(user, key)` so re-stores overwrite.
- **Embedding**: `memory_store` embeds `f"{key}: {value} [tags: …]"` via
  the Ollama `/api/embed` endpoint using `nomic-embed-text` (same model
  the KB uses). `memory_search` embeds the query, does a cosine search
  with `user == <id>` payload filter, drops hits below threshold.
- **Similarity threshold (0.4 default, env `MEMORY_SIMILARITY_THRESHOLD`)**
  keeps false positives out. Tighter than KB's (no threshold) because
  memory false-positives poison the prompt as "facts about the user" —
  more damaging than an irrelevant KB snippet.
- **Refuses untagged writes.** `memory_store` now returns 422 when `tags`
  has no `user:<id>` token. Prevents memories that can't be recalled
  (search requires a user filter) and prevents leakage across scopes.
- **One-shot migration**: on first startup, any legacy
  `/app/data/memory.db` (Phase 11 SQLite) is read, every row re-embedded
  and upserted to Qdrant, and the file renamed to `memory.db.migrated`.
  Idempotent — running again is a no-op. Rows without a `user:<id>` tag
  are skipped (and logged as WARNING).
- **`memory_recall(key)` still works** as an exact-key path (payload
  scroll, no vector). Kept for backwards compat; `search` is the primary
  path.
- **No changes to Audrey-side code.** `recall_for_request` calls
  `memory_search` exactly as before; it just gets smarter results.

**Prereqs:**

- Phase 11 green.
- `qdrant` container running on `ollama-net` and reachable at
  `http://qdrant:6333`.
- `nomic-embed-text` model pulled in Ollama (`ollama list | grep nomic`).
- `custom-tools` + `audrey-ai` rebuilt from this commit.

Rebuild + restart:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build custom-tools
docker compose logs --tail 40 custom-tools | grep -E "memory:|ready|migrating"
```

> Note on URLs: `curl` examples use `http://localhost:<port>` from the
> Unraid host. Container-DNS names (`custom-tools:8001`, `qdrant:6333`,
> `ollama:11434`) only work from *inside* `ollama-net`.

---

## 1. Sanity — collection created, migration ran

```bash
docker compose logs --tail 50 custom-tools | grep -E "created Qdrant|migrating|migrated"
```

Expected (first boot after deploy):
```
memory: migrating legacy SQLite store at /app/data/memory.db
memory: migrated N rows, failed 0, renamed memory.db -> memory.db.migrated
memory: created Qdrant collection 'kb_memory' (dim=768)
```

(Order of "migrating" vs "created" depends on whether the collection existed
before — fresh Qdrant sees "created" before "migrating.")

Confirm the collection in Qdrant:

```bash
curl -s http://localhost:6333/collections/kb_memory | jq '.result | {status, points_count, config}'
```

Expected: `status: "green"`, `points_count` ≥ the number of rows you had
pre-migration (3 if you ran the full Phase 11 smoke tests: `prefers_rust`,
`daily_driver_workstation`, `hates_rust`).

Confirm the migrated file:

```bash
docker exec custom-tools ls -la /app/data/
```

Expected: `memory.db.migrated` exists, no `memory.db`.

---

## 2. Smoke tests

### 2.1 Untagged writes are rejected

```bash
curl -s -w "\nhttp=%{http_code}\n" http://localhost:8001/memory_store \
  -H 'content-type: application/json' \
  -d '{"key":"no_user","value":"should not land in qdrant","tags":"topic:test"}'
```

Expected: `http=422`, body mentions `user:<id>`. Then confirm nothing was
written:

```bash
curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"ghost","query":"should not land","top_k":5}' | jq '.results | length'
```

Expected: `0`.

### 2.2 Semantic hit where keyword missed

The whole point of Phase 12. The prior-phase memory `daily_driver_workstation`
with value "AMD Threadripper 7970X workstation" was invisible to keyword
search for "hardware." Now it should surface:

```bash
curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"what hardware am I running","top_k":5}' \
  | jq '[.results[] | {key, value}]'
```

Expected: includes the `daily_driver_workstation` entry. If not:

- Check that the migration actually ran (step 1).
- Lower the threshold temporarily:
  `docker compose exec custom-tools env | grep MEMORY_SIMILARITY_THRESHOLD`
  and try with `0.2`. nomic-embed sometimes scores cross-domain language
  lower than expected; if 0.2 works but 0.4 doesn't, consider the
  threshold too tight.

### 2.3 Threshold filters unrelated queries

```bash
curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"what is the boiling point of water","top_k":5}' \
  | jq '.results | length'
```

Expected: `0`. Everything about cooking water should score well below
threshold against Rust preferences and a workstation spec. If this returns
hits, the threshold is too loose.

### 2.4 Cross-user isolation (still works)

```bash
curl -s http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"rust programming","top_k":5}' \
  | jq '[.results[] | {key, value, tags}]'
```

Expected: `prefers_rust` (Bart's), NOT `hates_rust` (Alice's). The payload
filter on `user` is enforced at the Qdrant query layer, not post-filter —
no chance of leakage.

### 2.5 Round-trip: end-to-end recall in a full chat request

Same shape as Phase 11 2.7 — previously the answer was correct but escalated.
Now it should hit memory, stay on the fast path, and reference the Threadripper:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"what hardware am I running?"}]}' \
  | jq -r '.choices[0].message.content' | head -10
```

And the log:

```bash
docker compose logs --tail 80 audrey-ai | grep -E "memory:|classify:|escalate:" | tail -10
```

Expected: `memory: user=bart@proton.me hits=1+ keys=[…]`. No `escalate:`
line (the memory-hits guard from Phase 11 holds). Answer mentions the
Threadripper.

### 2.6 Overwrite preserves created_at

```bash
curl -s http://localhost:8001/memory_store \
  -H 'content-type: application/json' \
  -d '{"key":"prefers_rust","value":"Bart prefers Rust (updated)","tags":"user:bart@proton.me,topic:languages"}' \
  | jq '{key, created_at, updated_at}'
```

Expected: `created_at` from the original write (pre-Phase-12 or 2026-04-24),
`updated_at` = now. Proves the `(user, key)` → UUIDv5 upsert path preserves
history.

### 2.7 New write via chat → semantic recall next turn

```bash
# Write via the model
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"Remember that my dog is a 4-year-old border collie named Ripley."}]}' \
  > /dev/null
docker compose logs --tail 20 audrey-ai | grep "memory_store"
```

Expected: a `dispatch: memory_store ok` line.

```bash
# Now recall via a semantically-related question (not a literal one)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_deep","user":"bart@proton.me","messages":[{"role":"user","content":"what pets do I have?"}]}' \
  | jq -r '.choices[0].message.content' | head -5
```

Expected: references Ripley the border collie. "pets" and "dog / border
collie" are the hardest test — zero literal overlap.

### 2.8 Embedding failure → graceful zero hits

Simulate Ollama unreachable by blocking the embed path. Easiest: stop
`ollama`, run a search, start it again.

```bash
docker stop ollama
curl -s -w "\nhttp=%{http_code}\n" http://localhost:8001/memory_search \
  -H 'content-type: application/json' \
  -d '{"user":"bart@proton.me","query":"rust","top_k":5}'
docker start ollama
```

Expected: `http=200`, `results: []`, and a log line on `custom-tools`:
`memory: embed failed for search query: …`. Don't 5xx on a best-effort
feature.

### 2.9 Full chat path survives embedding failure

```bash
docker stop ollama
curl -s -w "\nhttp=%{http_code}\n" http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"kimi-k2.6:cloud","user":"bart@proton.me","messages":[{"role":"user","content":"say hi"}]}' \
  | tail -5
docker start ollama
```

Wait — that routes through Audrey's Ollama client too, which needs Ollama
to chat. Use `audrey_cloud` with a cloud model so the chat itself runs
without local Ollama:

```bash
docker stop ollama
curl -s -w "\nhttp=%{http_code}\n" http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"audrey_cloud","user":"bart@proton.me","messages":[{"role":"user","content":"say hi"}]}' \
  | tail -5
docker start ollama
```

Expected: `http=200`. Memory recall fails silently (`dispatch: memory_search
network_error`), pipeline continues, cloud synthesizer answers normally.

---

## Troubleshooting

- **`collection 'kb_memory' not found` or shape mismatch**: a previous
  dev run created the collection with a different dimension. Drop it from
  *inside* a container on `ollama-net`:
  `docker exec custom-tools curl -XDELETE http://qdrant:6333/collections/kb_memory`.
  Then restart `custom-tools` to re-create it fresh.
- **Migration didn't run**: check `docker exec custom-tools ls /app/data/`.
  If `memory.db.migrated` exists but `memory.db` doesn't, it already ran
  on a prior startup — fine, this is idempotent.
- **2.2 finds zero hits**: similarity below threshold. Check the raw
  score by temporarily lowering `MEMORY_SIMILARITY_THRESHOLD=0.0`, re-run,
  then call the Qdrant REST directly:
  `curl -s http://localhost:6333/collections/kb_memory/points/search ...`
  to see actual scores. 0.3–0.5 range is typical for cross-domain
  phrasings with nomic-embed.
- **2.6 overwrite creates a *new* point instead of updating**: check the
  `_point_id` helper — both writes must agree on user+key. If the tags
  string changed the parsed user (e.g. extra whitespace), the hashes
  differ. Normalize the user tag on the client side.
- **Model picks a weird key that already exists for another fact**:
  overwrite wins. Not a bug — same `(user, key)` intentionally collapses.
  To avoid, encourage more specific keys in the store-hint (already does:
  "short **descriptive** key").
