# Phase 15 — sqlite index over per-user upload metadata

**Goal:** make `GET /v1/files` and the upload quota gate O(1) instead of
O(every chunk in the user's Qdrant collection). Qdrant stays
authoritative for *content*; sqlite is just an index over the *metadata*
(filename, mime, bytes, chunks, uploaded_at).

Why now: Phase 13 stored per-file metadata only as repeated payload
fields on every chunk point. Listing files meant scrolling the whole
user collection and grouping by `file_id`. Cheap at 6 files, painful
at 600. Same scroll happens once per upload (quota gate), so this
quietly amplifies.

What changed:

- **`src/audrey/kb/uploads_db.py` (new)** — single-file sqlite wrapper.
  Thread-safe via a per-instance lock; every call hops through
  `asyncio.to_thread`. WAL mode, NORMAL sync. One table:
  `uploads(file_id PK, user, filename, mime, bytes, kind, collection,
  chunks, uploaded_at)` plus `idx_uploads_user`.
- **`reconcile_with_qdrant`** runs on every boot. Backfills sqlite from
  any `kb_user_*` collection it can read; prunes sqlite rows whose
  file_id no longer has any points in Qdrant. Idempotent — safe to run
  every boot, no-op once everything's in sync.
- **`POST /v1/files`** — quota gate now reads from sqlite. After Qdrant
  ingest succeeds, sqlite gets the row; if the sqlite write fails,
  the Qdrant points are deleted (split-brain prevention).
- **`GET /v1/files`** — reads sqlite only, no Qdrant scroll.
- **`DELETE /v1/files/{id}`** — sqlite delete first (so list/quota
  immediately reflect it), then the Qdrant double-filter delete.
- **`config.yaml`** — `kb.uploads_db_path: /data/uploads.sqlite`. Lives
  in the same `/data` bind mount as the upload bytes — backup once,
  restore once.

**Prereqs:**

- Phase 14 verified (auth + same-origin routing).
- `/data` bind-mount writable by audrey (already — Phase 8).
- Existing `kb_user_*` collections in Qdrant. The boot reconcile
  populates sqlite from them; you don't need to re-upload anything.

---

## 1. Deploy

Laptop:

```bash
git pull    # or push if you committed locally
```

Unraid:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 30 audrey-ai | grep -E "uploads_db|reconcile|ready"
```

You should see one line:

```
uploads_db: reconcile complete (backfilled=N collections, pruned=0 ghost rows)
```

`N` = the count of `kb_user_*` collections in Qdrant for which at least
one row was indexed. First boot after deploy is when the bulk happens;
every boot after is a no-op.

---

## 2. Verify the index populated

The audrey image doesn't ship the `sqlite3` CLI — only the Python
module. From the Unraid host:

```bash
docker exec audrey-ai python -c "
import sqlite3
c = sqlite3.connect('/data/uploads.sqlite')
for row in c.execute('SELECT user, count(*), sum(bytes) FROM uploads GROUP BY user'):
    print(row)
"
```

Expected: one row per user who has uploaded files, with a sane count
and bytes total. Should match what `GET /v1/files` returned for those
users in Phase 14.

---

## 3. List still works (regression check)

From a machine that can reach `chat.builtryte.xyz`, with a fresh OWUI
JWT in `$TOKEN`:

```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files | jq '.user, (.files | length), .total_bytes'
```

Expected: same email + count + total bytes you saw before the upgrade.
That confirms backfill ⇒ list works without ever reading from Qdrant
anymore.

---

## 4. Upload writes both stores

```bash
echo "phase 15 sqlite index" > /tmp/p15.txt
curl -sS -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/p15.txt" \
  https://chat.builtryte.xyz/v1/files | jq
```

Then immediately:

```bash
docker exec audrey-ai python -c "
import sqlite3
c = sqlite3.connect('/data/uploads.sqlite')
for row in c.execute(\"SELECT file_id, filename, bytes FROM uploads WHERE filename = 'p15.txt'\"):
    print(row)
"
```

Expected: one row, matching `file_id` from the upload response.

---

## 5. Delete drops the row

```bash
FILE_ID='<id from step 4>'
curl -sS -X DELETE -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files/$FILE_ID | jq

docker exec audrey-ai python -c "
import sqlite3
c = sqlite3.connect('/data/uploads.sqlite')
for row in c.execute(\"SELECT file_id FROM uploads WHERE file_id = '$FILE_ID'\"):
    print(row)
print('done')
"
```

Expected: only `done` is printed — the row is gone. List from the UI
should reflect it without delay.

---

## 6. Reconcile is idempotent

```bash
docker compose restart audrey-ai
docker compose logs --tail 30 audrey-ai | grep "reconcile complete"
```

Expected: `pruned=0 ghost rows`. If you see a non-zero prune on a
fresh restart with no out-of-band Qdrant edits, something is dropping
sqlite rows that still have live Qdrant points — investigate.

---

## 7. Rollback

The upload bytes and Qdrant collections are unchanged from Phase 14.
Sqlite is purely additive; deleting it just means the next boot
re-runs the backfill from Qdrant.

If something goes wrong:

```bash
docker compose stop audrey-ai
mv /mnt/user/appdata/audrey/uploads.sqlite{,.bak}
git checkout <previous-sha> -- src/audrey/routes/files.py src/audrey/main.py
docker compose up -d --build audrey-ai
```

Phase 14 routes still work standalone — they just have to scroll Qdrant
for list/quota again.

---

## 8. Follow-ups (not Phase 15)

- Persist a `created_at` distinct from `uploaded_at` once we add
  re-ingest (right now they're the same).
- An admin endpoint that returns the reconcile stats without a
  restart — useful if a future migration touches Qdrant directly.
- A `/metrics` Prometheus endpoint exposing `uploads_indexed_total`,
  `quota_bytes_used` per user.
