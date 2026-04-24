# Phase 13 — Conversation-scoped KB ingest (server-side RAG)

**Goal:** own ingest end-to-end. User uploads files through a tiny
internal-only Audrey page → Audrey ingests into a per-user Qdrant
collection with `nomic-embed-text` → every future turn in OWUI (same
user, same chat or not) transparently retrieves against both the global
KB and the user's private docs via `kb_search`.

## Upload UX decision (spike result)

Open WebUI cannot forward uploads to an external backend. There is no
OpenAI Files API passthrough, no config toggle, no env var — the OWUI
team has declined/deferred every GitHub request for this, and
"pipelines" as a workaround means running another service with its own
API churn. We're going with **path B**: a small internal-only Audrey
upload page, plus OWUI's file-attach button disabled at the admin
level so there's a single source of truth for user docs.

Decided UX:
- OWUI is chat-only. Attach button hidden via admin config.
- `audrey-ai:8000/upload` is a single-page HTML form (internal-network
  only, not tunneled). Lists the user's existing files, drag-drop to
  upload, X to delete.
- User identity passed via the same OpenAI-spec `user` field used for
  memory. OWUI already forwards it on chat; the upload page reads it
  from a URL param (`?user=<id>`) or a simple signed cookie set on first
  visit. Either way, the page is only reachable from inside the LAN.
- Retrieval is invisible: next chat turn, `kb_search` hits both the
  global KB and the user's collection and merges. No model-side change.

## Why

- OWUI's RAG uses MiniLM-L6 (384-d). Audrey's KB uses nomic-embed-text
  (768-d). Two different embedding spaces mean we can't merge results
  across them — and MiniLM is measurably weaker on technical text.
- OWUI's RAG is per-chat. Users expect "I uploaded this last week" to
  persist. Per-user collections give that for free.
- Currently, with Bypass on, a 20-page PDF = 20k tokens in the prompt →
  deep mode every time, minutes per response. With proper RAG, top-k
  chunks = ~2-3k tokens → fast path, seconds.
- Audrey's pipeline (classify, escalate, ReAct, deep panel) can't see
  that the user attached a document when OWUI handles it upstream. With
  server-side ingest, the file is first-class — tool-capable models can
  decide to `kb_search` it, cite it, etc.

## Why not path A (OWUI pipelines intercept)

Considered and rejected:
- Pipelines is a separate service with its own Docker container, its
  own API surface, its own upgrade cadence. Adds two hops
  (OWUI → pipelines → Audrey) to every request, not just uploads.
- Pipelines API churns. Community reports breaking changes between OWUI
  versions.
- The "just disable OWUI uploads entirely" path (B) is simpler, gives
  us full control of the UX, and keeps infrastructure count flat.

## Scope

Four new user-facing capabilities:

1. **Upload API**: `POST /v1/files` — multipart, user-scoped, ingested
   synchronously if small (<5 MB) or queued if larger. Returns a file id.
2. **List / delete API**: `GET /v1/files?user=<id>`, `DELETE /v1/files/<id>?user=<id>`.
3. **Upload UI**: `GET /upload` — single HTML page served by Audrey.
   Lists the user's files, drag-drop upload, delete button. Internal
   network only.
4. **Retrieval**: `kb_search` (the existing tool) becomes user-aware —
   when `user` is present, it queries the global `kb_text` *and* the
   user's `kb_user_text_<sanitized>` collection, merging by score.

Not in scope for Phase 13:
- Scanned-PDF OCR beyond what pypdf can extract (leave a seam for
  later — Tesseract in a separate container if needed).
- Cross-user sharing / "team collections."
- Re-enabling OWUI's attach button. Admins disable it once in OWUI's
  Settings → Interface; users use Audrey's upload page instead.

## Architecture

### Collections

- `kb_user_text_<sanitized_user>` (768-d cosine, nomic-embed)
- `kb_user_images_<sanitized_user>` (512-d cosine, CLIP)

Sanitization: `user` → `re.sub(r'[^a-z0-9]+', '_', user.lower()).strip('_')`.
So `bart@proton.me` → `bart_proton_me`. The raw `user` is preserved in
each point's payload for display and filtering.

Lazy-create on first upload. Payload indexes on `user` and `file_id` so
filters and deletes are O(index lookup) not O(scan).

Payload shape on every point:

```json
{
  "user": "bart@proton.me",
  "file_id": "<uuid4>",
  "source": "/data/uploads/bart_proton_me/<uuid>.pdf",
  "filename": "RA_management.pdf",
  "chunk_idx": 12,
  "mime": "application/pdf",
  "bytes": 1048576,
  "uploaded_at": "2026-04-24T19:00:00Z"
}
```

### Upload flow

```
user browser → GET /upload          (HTML form, reads ?user=<id> from URL)
             → POST /v1/files       (multipart; hidden form field carries user)
  ├─ user identity: from form field (page is internal-only, so we trust it
  │   — same trust model as Audrey's existing /v1/chat/completions, which
  │   also takes `user` from the request body)
  ├─ validate: size cap, mime whitelist, per-user quota
  ├─ save bytes to /data/uploads/<sanitized_user>/<uuid><ext>
  ├─ if size < 5 MB: extract + embed + upsert synchronously
  └─ else: enqueue background task, return 202 with file_id

browser polls GET /v1/files/<id>    (shows "ingesting…" → "ready" in the list)
```

### Retrieval flow (unchanged contract, smarter implementation)

```
orchestrator dispatches kb_search({query, top_k, user})
  ├─ proxies to /v1/kb/query on audrey-ai (already exists)
  └─ /v1/kb/query queries global kb_text
      ├─ if user is set and kb_user_text_<sanitized> exists: also query it
      └─ merge results by score (both are cosine over nomic-embed — same
         space, safe to merge)
```

No changes to the orchestrator pipeline. `recall_for_request` already
passes `user_id` into tool dispatch — we just thread it into `kb_search`
the same way Phase 12 threaded it into `memory_search`.

## Safety measures (baked in, not added later)

1. **Max upload size**: 50 MB default, configurable via `KB_MAX_UPLOAD_MB`.
   Return `413 Request Entity Too Large` above the cap.
2. **Mime whitelist**: `application/pdf`, `text/plain`, `text/markdown`,
   `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
   (docx), and common image types. Reject everything else with `415`.
   *Don't* trust the client-declared mime alone — use `python-magic` /
   libmagic to sniff actual bytes.
3. **Per-user byte quota**: 1 GB default, `KB_MAX_USER_BYTES`. Block upload
   with `429` + a clear message if user is at cap.
4. **Filename sanitization**: on-disk name is always `<uuid><ext>`.
   Original filename is stored in payload metadata only. No
   user-controlled path component anywhere.
5. **PDF parser choice**: `pypdf` (pure Python, actively maintained, no
   native deps). Explicitly avoid `unstructured` and `pdfminer.six` for
   unsanitized input. OCR deferred — scanned PDFs without text layer
   return an empty-extraction error; user sees a clear 422 rather than
   a silent empty ingest.
6. **Quota check happens *before* write**: read current user total from
   Qdrant payload sum (or a sidecar counter — faster, updated atomically
   on upload/delete). Prevents the "check, then exceed" race.
7. **Background ingest is bounded**: single asyncio task queue,
   max 2 concurrent ingests, tail-drops with 503 if queue is full. A
   malicious user can't spawn arbitrary concurrent embed work.
8. **Audrey's `/v1/files` is internal-only**. The compose config keeps
   `audrey-ai` off the Cloudflare tunnel; only OWUI is public. **Don't
   change that in Phase 13.** (Worth a test in `docs/phase-13-deploy.md`:
   confirm the tunnel doesn't route to `:8000`.)
9. **Delete is scoped**: `DELETE /v1/files/<id>?user=<id>` requires the
   user arg, and the underlying Qdrant delete uses a `must` filter on
   both `file_id` and `user` — no "delete by file_id alone" path exists.
10. **Logging**: every upload/delete logs `user`, `file_id`, `bytes`,
    `chunks`, `mime`. Embedding failures include the filename for triage.
    No file *contents* in logs.

## OWUI configuration

Path B means we're bypassing OWUI's file handling entirely. Settings to
flip in OWUI's admin panel:

- **Interface → Disable file upload**: hides the attach button in the
  chat composer. Prevents the UX split where users try to upload via
  OWUI and wonder why retrieval differs. If this exact toggle doesn't
  exist in your OWUI version, equivalent is setting
  `ENABLE_RAG_HYBRID_SEARCH=false` and `ENABLE_RAG_LOCAL_WEB_FETCH=false`
  plus removing the files permission from user groups.
- **Documents → Bypass Embedding and Retrieval**: set to whatever; it's
  no longer in the path. Leave default.
- Chat still uses `/v1/chat/completions` normally — no OWUI-side changes
  needed for retrieval to work. Audrey handles the `user` field we already
  threaded through in Phase 11.

A small admin note in the Phase 13 deploy doc will spell out these
clicks for future deployers.

## File layout

```
src/audrey/
├── kb/
│   ├── user_store.py        # NEW — per-user collection naming + ensure
│   ├── extract.py           # NEW — pdf/docx/text extractors
│   ├── ingest.py            # MODIFIED — accept collection override
│   └── qdrant.py            # MODIFIED — generic ensure_collection(name, dim)
├── routes/
│   ├── files.py             # NEW — POST/GET/DELETE /v1/files
│   ├── upload_ui.py         # NEW — GET /upload (HTML page)
│   └── kb.py                # MODIFIED — /v1/kb/query merges user + global
├── static/
│   └── upload.html          # NEW — single-page drag-drop UI
└── main.py                  # MODIFIED — wire files + upload_ui routers
```

No changes to `tools-server/` — the existing `kb_search` tool already
sends the `user` field if the orchestrator includes it.

## Config additions

`config.yaml`:
```yaml
kb:
  user_collections_enabled: true
  max_upload_mb: 50
  max_user_bytes: 1_073_741_824        # 1 GB
  max_concurrent_ingests: 2
  ingest_queue_max: 16
  user_text_prefix: "kb_user_text_"
  user_image_prefix: "kb_user_images_"
```

`.env.example`:
```
KB_MAX_UPLOAD_MB=50
KB_MAX_USER_BYTES=1073741824
```

## Smoke tests (`docs/phase-13-deploy.md`)

1. **Upload-page smoke** — browse to `http://<unraid>:8000/upload?user=bart@proton.me`
   from a LAN machine, drag-drop a small PDF, verify:
   - Page shows "ingesting" then "ready" within a few seconds.
   - `kb_user_text_<sanitized>` collection exists with N points.
   - `GET /v1/files?user=bart@proton.me` returns the file.
2. **API upload smoke** — POST a PDF via curl (bypassing the UI),
   verify the same end state.
3. **Retrieval smoke** — in OWUI, ask a question about the uploaded doc.
   Log shows `kb_search` tool called, answer cites the filename, prompt
   is ≤3k tokens (proves retrieval, not full-doc inlining). Contrast
   with pre-Phase-13 numbers where the same doc made the prompt 20k+.
4. **Cross-user isolation** — user A uploads doc, user B asks about it
   in OWUI, user B gets no hits. Identical to Phase 12's cross-user test.
5. **Delete** — click X on the upload page, verify both Qdrant points
   and disk bytes gone, file disappears from the list.
6. **Size cap** — upload >50 MB, expect `413`, nothing written, UI shows
   a clear error.
7. **Mime reject** — POST an exe/zip, expect `415`, nothing written.
8. **Quota cap** — simulate user at quota, next upload → `429` with
   bytes-remaining hint.
9. **Scanned PDF (no text layer)** — expect `422` with clear message,
   nothing written to Qdrant.
10. **Background ingest** — upload 20 MB file, verify `202` + file_id,
    poll `/v1/files/<id>` until `status: ready`, upload page reflects it.
11. **Embed failure graceful** — stop Ollama mid-ingest, upload, expect
    the file to be marked `failed`, not crash the container.
12. **Tunnel isolation** — curl the public tunnel host at both `/v1/files`
    and `/upload` → should not reach Audrey (cloudflared routes to OWUI only).
13. **OWUI attach disabled** — confirm the paperclip button is gone in
    the OWUI chat composer. If a user somehow sends a file through OWUI
    anyway (old client version, API call), it should degrade to a normal
    chat request with whatever text OWUI inlined, not crash or duplicate.

## Risks / open questions

- **Parser robustness on messy PDFs**. pypdf is not infallible. Medical
  PDFs with columns, tables, and footnotes may extract as garbage. Worth
  a visual check on the same PANLAR PDF you already uploaded once the
  ingest path exists.
- **Queue persistence**. If `audrey-ai` restarts mid-ingest, in-flight
  uploads are lost. Acceptable for v1 — user re-uploads. If it becomes
  painful, persist the queue to a SQLite file at `/data/ingest_queue.db`.
- **CLIP for user images.** First-time download blocks the ingest for
  ~30s on a fresh `clip-cache`. Already cached per compose, so only an
  issue on first deploy after a cache wipe.
- **User identity on the upload page.** We're trusting the `user` URL
  param because the page is internal-only, same trust model as the
  chat endpoint. If/when Audrey is ever put behind real auth, both
  paths need to switch to an auth-derived identity. Worth noting in
  `CONTINUITY.md`.
- **OWUI version-pinning of the attach-disable toggle.** The exact
  setting name has drifted across OWUI releases. Deploy doc will need
  the current path at deploy time; if it moves again, the fallback is
  CSS injection via OWUI's custom-CSS admin field to `display:none` the
  attach button. Ugly but reliable.

## Rollback

All changes are additive: new endpoints, new Qdrant collections (`kb_user_*`),
new config keys default-off via `kb.user_collections_enabled: false`. Flip
that to false and restart — behaves exactly like Phase 12.

## Estimated effort

~1.5 days of writing:
- `user_store.py` + `extract.py`: ~200 lines each.
- `routes/files.py`: ~250 lines.
- `routes/upload_ui.py` + `static/upload.html`: ~200 lines total
  (plain HTML/CSS/fetch, no framework).
- `kb.py` merge logic: ~50 lines.
- Config + env: ~20 lines.
- Smoke doc: ~250 lines.
- Tests: manual via deploy doc (matches project convention).

Probably ~2 days wall-clock with smoke cycles on Unraid.
