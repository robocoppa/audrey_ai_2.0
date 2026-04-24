# Phase 13 — Per-user uploads + server-side RAG

**Goal:** own file ingest end-to-end. User drops files into a small
internal Audrey page → Audrey saves and ingests into per-user Qdrant
collections with `nomic-embed-text` → every OWUI chat turn (same user,
any chat) transparently searches both the global KB and that user's
private docs via `kb_search`.

What's new vs Phase 12:

- **`POST /v1/files` (multipart upload).** Streams to disk with a 50 MB
  per-file cap, libmagic sniffs the saved bytes, a 1 GiB per-user quota
  gate runs before ingest. Extracts with pypdf / python-docx / the
  existing loaders, chunks at 1000 tokens, embeds with
  `nomic-embed-text`, upserts into `kb_user_text_<sanitized_user>`.
  Images land in `kb_user_images_<sanitized_user>` via CLIP (same model
  as the global KB, same 512-d space).
- **`GET /v1/files` / `DELETE /v1/files/<file_id>`.** List the caller's
  files; delete is double-filtered on both `file_id` *and* `user` so a
  leaked id can never cross-scope delete.
- **`GET /upload` — internal-only HTML page** for drag-drop upload, list,
  delete. Ships inside the image at `src/audrey/static/upload.html`.
- **`kb_search` is user-aware.** When the ReAct loop has a `user_id`,
  dispatch auto-injects `user` into the tool arguments. The `/v1/kb/query`
  handler then searches global `kb_text` *and* `kb_user_text_<sanitized>`
  concurrently and merges by score.
- **No changes to OWUI** — the existing chat path keeps working. OWUI's
  own file-attach button stays disabled at the admin level so there's a
  single source of truth for user docs.

**Prereqs:**

- Phase 12 green (semantic memory verified, `kb_memory` populated).
- `libmagic1` installed in the `audrey-ai` image (already in
  `docker/audrey.Dockerfile` as of this commit).
- `python-magic`, `python-multipart` in the Python deps (already in
  `pyproject.toml` and the Dockerfile pin list).
- `/mnt/user/appdata/audrey` bind-mount existing (already — reused from
  Phase 8 as `/data` in-container).

Rebuild + restart:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 40 audrey-ai | grep -E "ready|upload|kb_user|files:"
```

> URL note: use `http://<unraid-ip>:8000/...` from the LAN for the
> upload UI. Audrey is **not** exposed via the Cloudflare tunnel — only
> OWUI is public. Confirmed in step 12.

---

## 1. Sanity — upload page reachable

From a machine on the LAN:

```bash
curl -sI http://localhost:8000/upload | head -1
# expected: HTTP/1.1 200 OK
```

Open `http://<unraid-ip>:8000/upload` in a browser — you should see the
"Audrey · Knowledge Upload" page. Enter your user id (e.g. your OWUI
email). It's persisted to `localStorage` so you don't retype it.

## 2. Upload a small text file via the UI

Drag-drop a short `.txt` or `.md`. The status line should read
`Uploaded <name> → N chunks.` within a second or two. The files table
refreshes and shows the row.

Then on Unraid:

```bash
docker exec -it audrey-ai curl -s http://qdrant:6333/collections \
  | python3 -m json.tool | grep kb_user_text_
# expected: a line like "name": "kb_user_text_bart_proton_me"

docker exec -it audrey-ai curl -s \
  "http://qdrant:6333/collections/kb_user_text_bart_proton_me" \
  | python3 -m json.tool | grep -E "points_count|status"
# expected: points_count >= 1, status "green"
```

## 3. Upload via `curl` (no UI)

```bash
USER=bart@proton.me
curl -sX POST "http://localhost:8000/v1/files?user=$USER" \
     -H "X-User: $USER" \
     -F "file=@/mnt/user/knowledge/first-aid/snakebite.pdf"
```

Expected JSON: `{"file_id": "...", "filename": "snakebite.pdf",
"mime": "application/pdf", "bytes": ..., "kind": "text",
"collection": "kb_user_text_bart_proton_me", "chunks": N}`.

## 4. List shows both uploads

```bash
curl -s "http://localhost:8000/v1/files?user=$USER" -H "X-User: $USER" \
  | python3 -m json.tool
```

Expected: two `files[]` entries, `total_bytes` sums them.

## 5. Retrieval — ask OWUI about the upload

In OWUI, pick `audrey_deep` (or `audrey_cloud`), start a new chat, ask
a question only answerable from the uploaded doc. In `audrey-ai` logs:

```bash
docker compose logs --tail 200 audrey-ai | grep -E "kb_search|dispatch: kb_search|tool_rounds"
```

Expected:
- `dispatch: kb_search ok in X.XXs` (real tool call).
- Arguments include `"user": "<your id>"` — auto-injected by dispatch.
- Final answer cites the filename or content, prompt token count is
  under ~3k (proof of retrieval, not full-doc inlining). Pre-Phase-13
  with OWUI bypass on, the same doc ballooned the prompt to 20k+.

## 6. Cross-user isolation

Upload a doc as `user_a@test`, then in a different OWUI session (or via
curl with a different user id) ask about it. `kb_search` should return
zero hits from `kb_user_text_user_a_test` for `user_b`. Logs show the
search proxy only included the current user's collection.

```bash
# verify from the server side:
curl -sX POST http://localhost:8000/v1/kb/query \
  -H "content-type: application/json" \
  -d '{"query": "snakebite treatment", "top_k": 5, "user": "user_b@test"}' \
  | python3 -m json.tool | head -30
# expected: hits only from kb_text (global), none from user_a's collection
```

## 7. Delete scrubs both Qdrant and bytes

Click the `delete` button on the upload page, or:

```bash
FID=<file_id from step 4>
curl -sX DELETE "http://localhost:8000/v1/files/$FID?user=$USER" -H "X-User: $USER"
# expected: {"file_id": "...", "deleted": true}

ls /mnt/user/appdata/audrey/uploads/bart_proton_me/ | grep $FID
# expected: empty (bytes gone)

curl -s "http://qdrant:6333/collections/kb_user_text_bart_proton_me" \
  | python3 -m json.tool | grep points_count
# expected: one less than before
```

## 8. Size cap (413)

```bash
# 60 MB dummy file — above the 50 MB cap
dd if=/dev/urandom of=/tmp/big.bin bs=1M count=60
curl -sX POST "http://localhost:8000/v1/files?user=$USER" \
     -H "X-User: $USER" \
     -F "file=@/tmp/big.bin" -w '\nHTTP %{http_code}\n'
# expected: HTTP 413, "Upload exceeds 50 MB limit."
# verify nothing landed:
ls /mnt/user/appdata/audrey/uploads/bart_proton_me/ | grep -c '^' # should not include big.bin
```

## 9. Mime reject (415)

```bash
# rename a binary to look like a PDF — libmagic sniffs the bytes
cp /bin/ls /tmp/fake.pdf
curl -sX POST "http://localhost:8000/v1/files?user=$USER" \
     -H "X-User: $USER" \
     -F "file=@/tmp/fake.pdf" -w '\nHTTP %{http_code}\n'
# expected: HTTP 415, "Unsupported mime: 'application/x-executable'..."
```

## 10. Scanned PDF (no text layer) → 422

Upload a scanned PDF (image-only pages). Expected:

```
HTTP 422, "no extractable text from <name>; scanned PDFs without a text layer are not yet supported"
```

And nothing written to Qdrant (confirm the `points_count` did not change).

## 11. User isolation on delete

```bash
FID=<file_id owned by user_a>
curl -sX DELETE "http://localhost:8000/v1/files/$FID?user=user_b@test" \
     -H "X-User: user_b@test"
# returns {"deleted": true} but…
# verify nothing actually went away — the Qdrant filter is (file_id AND user)
# so the delete scoped to the wrong user is a no-op, not an error
curl -s "http://qdrant:6333/collections/kb_user_text_user_a_test" \
  | python3 -m json.tool | grep points_count
# expected: unchanged
```

## 12. Tunnel isolation — `/v1/files` is internal only

From a machine **off** your LAN:

```bash
curl -sI https://<your-tunnel-host>/upload
curl -sI https://<your-tunnel-host>/v1/files
# expected: the tunnel either 404s or returns OWUI's frontend;
# it must NOT return Audrey's upload page or 401/200 from /v1/files
```

If you accidentally exposed `:8000` via Cloudflare, revoke it now. Only
`open-webui:8080` should be published.

## 13. Post-deploy: add a link from OWUI to `/upload`

OWUI's admin has a **Sidebar Links** field (path varies by version:
Settings → Interface → `WEBUI_EXTRA_LINKS`, or `custom_links` in older
builds). Add one:

```
Label: Upload documents
URL:   http://<unraid-ip>:8000/upload
```

And disable the chat composer's attach button in the same panel so
users aren't split between two upload paths:

- **Settings → Interface → Disable file upload** (or equivalent
  `ENABLE_FILE_UPLOAD=false`).

If neither toggle exists in your OWUI build, fall back to CSS in the
admin **Custom CSS** field:

```css
button[aria-label="Attach file"] { display: none !important; }
```

---

## Rollback

All new routes are additive. To fully disable:

1. Stop accepting uploads — remove the `files_router` include from
   `src/audrey/main.py` (or set `max_upload_mb: 0` in `config.yaml`).
2. Drop the user collections (irreversible):
   ```bash
   for c in $(curl -s http://qdrant:6333/collections \
       | python3 -c 'import sys, json; [print(x["name"]) for x in json.load(sys.stdin)["result"]["collections"] if x["name"].startswith("kb_user_")]'); do
     curl -sX DELETE "http://qdrant:6333/collections/$c"
   done
   ```
3. `rm -rf /mnt/user/appdata/audrey/uploads/` to reclaim disk.

`kb_search` without a `user` arg still works identically to Phase 12
(searches only global `kb_text`), so the orchestrator behaves correctly
whether user collections exist or not.

---

## Troubleshooting

- **`ModuleNotFoundError: magic`** in `audrey-ai` logs → `libmagic1`
  wasn't installed. Rebuild: `docker compose build --no-cache audrey-ai`.
- **`python-multipart not installed`** on first upload → same fix
  (the Dockerfile pin was missed). Happens only on stale images.
- **`kb_search` returns only global hits when the user has uploaded
  files** → check the ReAct loop got a `user_id`. The OWUI request body
  must include `"user": "<email>"`. Audrey logs the dispatch args on
  DEBUG; raise log level temporarily if unsure.
- **Upload succeeds but `points_count` stays 0** → extraction returned
  empty (scanned PDF, empty docx). Route returns 422 in this case; if
  you see a 200 instead, check `extract_text` didn't get bypassed.
- **Per-user quota count is wrong after many add/delete cycles** → the
  quota sums `bytes` across points grouped by `file_id`. If a prior
  delete only removed bytes but left Qdrant points (impossible via this
  API, possible via direct Qdrant fiddling), counts drift. Cleanup:
  delete the affected user collection and re-upload.
