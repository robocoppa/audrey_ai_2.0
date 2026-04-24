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
# (if you see 405 Method Not Allowed, the running image predates the
#  HEAD/GET change in routes/upload_ui.py — rebuild audrey-ai)
```

Open `http://<unraid-ip>:8000/upload` in a browser — you should see the
"Audrey · Knowledge Upload" page. Enter your user id (e.g. your OWUI
email). It's persisted to `localStorage` so you don't retype it.

## 2. Upload a small text file via the UI

Drag-drop a short `.txt` or `.md`. The status line should read
`Uploaded <name> → N chunks.` within a second or two. The files table
refreshes and shows the row.

Then on the Unraid host (Qdrant is mapped to `:6333` by the Unraid UI).
Unraid's root shell doesn't have `python3`, so use raw grep — no
dependencies, works everywhere:

```bash
curl -s http://localhost:6333/collections | grep -o 'kb_user_text_[a-z0-9_]*'
# expected: kb_user_text_bart_proton_me

curl -s http://localhost:6333/collections/kb_user_text_bart_proton_me \
  | grep -oE '"(points_count|status)":[^,}]*'
# expected:
#   "status":"green"
#   "points_count":N   (N >= 1)
```

If you have `jq` available, nicer:

```bash
curl -s http://localhost:6333/collections | jq -r '.result.collections[].name' | grep kb_user_
curl -s http://localhost:6333/collections/kb_user_text_bart_proton_me \
  | jq '{status: .result.status, points_count: .result.points_count}'
```

## 3. Upload via `curl` (no UI)

```bash
USER=your-owui-email@example.com    # whatever you entered on the upload page
curl -sX POST "http://localhost:8000/v1/files?user=$USER" \
     -H "X-User: $USER" \
     -F "file=@/mnt/user/knowledge/first-aid/snakebite.pdf"
```

Expected JSON: `{"file_id": "...", "filename": "snakebite.pdf",
"mime": "application/pdf", "bytes": ..., "kind": "text",
"collection": "kb_user_text_bart_proton_me", "chunks": N}`.

## 4. List shows both uploads

```bash
curl -s "http://localhost:8000/v1/files?user=$USER" -H "X-User: $USER" | jq
# or without jq:
# curl -s "http://localhost:8000/v1/files?user=$USER" -H "X-User: $USER"
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
  | jq '.results[] | {score, source}'
# expected: sources only from /datasets/... (global), none from /data/uploads/user_a_test/...
```

## 7. Delete scrubs both Qdrant and bytes

Click the `delete` button on the upload page, or:

```bash
FID=<file_id from step 4>
curl -sX DELETE "http://localhost:8000/v1/files/$FID?user=$USER" -H "X-User: $USER"
# expected: {"file_id": "...", "deleted": true}

ls /mnt/user/appdata/audrey/uploads/bart_proton_me/ | grep $FID
# expected: empty (bytes gone)

curl -s http://localhost:6333/collections/kb_user_text_bart_proton_me \
  | grep -oE '"points_count":[0-9]+'
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
curl -s http://localhost:6333/collections/kb_user_text_user_a_test \
  | grep -oE '"points_count":[0-9]+'
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

## 13. Post-deploy: point OWUI users at `/upload` + disable native attach

**Verified on OWUI v0.9.2.** Older/newer builds may differ — check the
v0.9.2 source paths cited below if a setting has moved.

### 13a. Surface the upload URL inside OWUI

OWUI v0.9.2 has **no "Sidebar Links" / `WEBUI_EXTRA_LINKS` / `custom_links`
feature** — earlier versions of this doc were wrong. The only built-in
way to show every user a custom URL is the **Banners** feature, which
renders a dismissible strip at the top of the chat view.

**Admin Panel → Settings → Interface → Banners**, or set the
`WEBUI_BANNERS` env var on the `open-webui` container:

```json
[{"id":"upload","type":"info","title":"Upload files",
  "content":"Upload docs to your private KB: http://<unraid-ip>:8000/upload",
  "dismissible":true,"timestamp":1761350000}]
```

Fallbacks if you want an actual sidebar entry:

- **Custom CSS/JS injection** — Admin Panel → Settings → General has a
  custom-CSS field. You can inject a sidebar link via JS that appends
  to OWUI's nav.
- **Reverse-proxy HTML rewrite** — inject a `<a>` tag into OWUI's
  `index.html` before it reaches the browser. Brittle across OWUI
  upgrades; skip unless you already run a reverse proxy.

### 13b. Hide the chat composer's file-attach (paperclip) button

OWUI v0.9.2 has **no `ENABLE_FILE_UPLOAD` env var and no
"Disable file upload" toggle** under Interface. File attach is gated
by a group permission:

**Admin Panel → Users → Groups → (your default user group) → Chat
Permissions → Allow File Upload** (toggle off).

Adjacent toggle "Allow Web Upload" controls webpage attach; turn that
off too if you don't want that path either.

Important caveats:

- **Admins always see the paperclip** regardless of this setting
  (v0.9.2 InputMenu.svelte line 56). Test with a non-admin account to
  verify the button disappears.
- The server-side block is at `MessageInput.svelte:590` — even if a
  stale client still shows the button, the upload POST is rejected.
- If you don't use Groups yet, create one (e.g. "users"), set the
  permission, and assign existing accounts to it.

**CSS fallback** (admins only, Admin Panel → Settings → General →
Custom CSS) if you need the paperclip hidden in admin accounts too.
Selectors drift across OWUI releases — verify in browser devtools for
your build:

```css
/* v0.9.2 — verify aria-label matches your locale */
button[aria-label="Attach file"] { display: none !important; }
```

Source references (inside the OWUI v0.9.2 repo):

- `src/lib/components/admin/Users/Groups/Permissions.svelte` L421-454
- `src/lib/components/chat/MessageInput/InputMenu.svelte` L56
- `src/lib/components/chat/MessageInput.svelte` L590
- `src/lib/components/admin/Settings/Interface/Banners.svelte`
- `backend/open_webui/config.py` (`WEBUI_BANNERS`)

---

## Rollback

All new routes are additive. To fully disable:

1. Stop accepting uploads — remove the `files_router` include from
   `src/audrey/main.py` (or set `max_upload_mb: 0` in `config.yaml`).
2. Drop the user collections (irreversible):
   ```bash
   for c in $(curl -s http://localhost:6333/collections \
       | grep -oE 'kb_user_[a-z0-9_]+'); do
     curl -sX DELETE "http://localhost:6333/collections/$c"
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
