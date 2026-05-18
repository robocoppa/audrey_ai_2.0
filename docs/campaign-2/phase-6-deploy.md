# Campaign 2 Phase 6 - KB audit fixes (Lesson 9)

Pre-lesson audit pass on the KB ingest + search surface. Six findings
resolved, one accepted (re-confirmed), one deferred for measurement. The
changes are small, local to `kb/` and `routes/kb.py`, and add seven
hermetic tests.

No new behavior. The user-facing improvements are an unambiguous error
when an `image_url` redirects, and real failures from
`create_payload_index` no longer get swallowed.

## What changed

**`src/audrey/kb/qdrant.py`**

- Module docstring now names the correct Ollama endpoint
  (`/api/embed`, not the older `/api/embeddings`).
- `QdrantKB.delete_by_source` returns `None`. It used to return a `0` /
  `-1` sentinel that no caller read — qdrant-client's `delete()` doesn't
  expose a count.
- `_ensure_user_indexes_sync` catches `UnexpectedResponse` (not bare
  `Exception`) and only swallows it when the status is < 500 and the body
  mentions "exist". Genuine schema errors and transport failures now
  propagate, so a per-user collection can't ship with missing payload
  indexes silently.

**`src/audrey/kb/embed.py`**

- `_fetch_image` checks `len(buf) + len(chunk) > _IMAGE_FETCH_BYTE_CAP`
  *before* extending `buf`. The previous order let a single oversized
  streamed chunk overshoot the 25 MB cap by its own length.
- `_fetch_image` detects 3xx responses before `raise_for_status()` and
  raises `ValueError` naming the redirect target. The user now sees
  `image_url returned redirect (302) to '...'; supply the final URL
  directly` instead of an opaque `422 image embed failed: 302 Found`.

**`src/audrey/routes/kb.py`**

- `_search_text_merged` / `_search_images_merged` rename `tasks` → `coros`
  to match what those objects actually are (coroutines, not Tasks).
- Inline note documents the precondition that lets the raw-score merge
  work: both global and per-user collections use the same embedder and
  distance metric. If that ever stops being true, switch to RRF.

**`tests/test_kb_reconcile.py`**

- Fake `delete_by_source` updated to the new `-> None` signature.

**`tests/test_kb_embed_ssrf.py`**

- Three new cases. Oversized-chunk rejection confirms the cap can't be
  overshot. Redirect test confirms the new error message includes the
  Location target. Non-image-content-type test guards against a
  regression of the existing content-type filter under the new control
  flow.

**`tests/test_kb_qdrant.py`** (new file)

- Four cases covering `_ensure_user_indexes_sync`: idempotent
  already-exists swallow, 5xx propagation, unrelated-4xx propagation,
  non-qdrant exception propagation.

## What did not change

- Wire format and tool schemas — `/v1/kb/query`, `/v1/kb/query/image`,
  `/v1/kb/ingest`, `/v1/kb/stats` all behave identically.
- Embedder choices and dimensions (768-d nomic-embed-text, 512-d CLIP
  ViT-B-32).
- Qdrant collection layout. No migration.
- Dockerfiles, `compose.yaml`, `pyproject.toml`, `uv.lock`. Pure source
  edits.
- Custom-tools side. The `kb_search` and `kb_image_search` proxies still
  call the same audrey endpoints.

## 1. Deploy

Local first (laptop):

```bash
git pull
.venv/bin/python -m pytest -q     # 248 should pass
.venv/bin/ruff check src/audrey/kb src/audrey/routes/kb.py
```

The expected count is 248 (was 241; +4 from `test_kb_qdrant.py`, +3 from
new cases in `test_kb_embed_ssrf.py`).

Unraid (from `/mnt/user/appdata/audrey_ai_2.0`):

```bash
git pull
docker compose up -d --build audrey-ai
docker compose logs --since 1m audrey-ai | grep -E "ready|kb"
```

Custom-tools does not need rebuilding — only `src/audrey/kb/*` and
`src/audrey/routes/kb.py` changed.

Expected:

- Image rebuilds cleanly. Roughly the same size.
- The startup readiness line still shows `tools=7` and `kb_text` /
  `kb_images` counts at the values from the previous deploy.

## 2. Smoke tests

The changes are narrow but they touch the hot search path and the
user-upload path. The pytest suite covers the unit-level behavior; the
checks below confirm nothing regressed in the wired stack.

### 2.1 Health and tool registry

```bash
curl -sS http://localhost:8000/health | jq .
curl -sS http://localhost:8000/v1/tools | jq '.tools[].name'
```

Expected: seven tools as before. `kb_search` and `kb_image_search`
present.

### 2.2 KB text search (global collection)

Pick a query you know hits the geology corpus or one of the other topic
subdirs.

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query \
  -H 'content-type: application/json' \
  -d '{"query": "what is BTRFS", "top_k": 3}' | jq .
```

Expected: 1-3 hits, each with `score`, `source`, `kind="text"`,
`chunk_idx`, `text`. The merge code is unchanged in behavior; this just
confirms it still runs.

### 2.3 KB image search via text query

CLIP text-to-image still works after the embed-path edits.

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"query": "person in guard position", "top_k": 3}' | jq .
```

Expected: hits with `kind="image"`. Low cosine scores (0.15-0.30) are
normal for CLIP text-to-image — not a regression.

### 2.4 KB image search via URL (exercises `_fetch_image` cap + redirect)

Pick a host that returns a small image directly with no redirect.
Wikimedia has flaked here in the past (403 on the default UA, occasional
thumb-path redirects); avoid it for this check. `httpbin.org/image/png`
is purpose-built for this and returns an ~8 KB PNG with `content-type:
image/png` and no redirect.

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://httpbin.org/image/png", "top_k": 3}' | jq .
```

Expected: top hits returned. Confirms the byte-cap re-ordering didn't
break the happy path.

To verify the new redirect message, point at a URL that 302s.
`picsum.photos/200` redirects to a Cloudinary CDN URL every request:

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://picsum.photos/200", "top_k": 3}' | jq .
```

Expected: `422` with detail `image embed failed: image_url returned
redirect (302) to '<final URL>'; supply the final URL directly`. The
old message was the opaque `422 image embed failed: Client error '302
Found'`.

If `httpbin.org` is unreachable on the deploy day, a stable alternative
is a direct GitHub raw URL pointing at a small PNG in any public repo
(e.g. an octocat avatar). GitHub raw returns `content-type: image/png`
and does not redirect.

### 2.5 Per-user collection merge still works

If a logged-in user has uploaded files (so `kb_user_text_<user>` exists),
issue a `kb_search` through Open WebUI for content known to be in their
uploads. The pipeline should pull from both collections and rank by raw
score. Watch the audrey log line:

```bash
docker compose logs --since 30s audrey-ai | grep "kb_search"
```

Expected: the `had_user_collection=true` label appears in the
`kb_search_seconds` metric, and at least one hit's `source` references
the user-uploaded file.

### 2.6 Upload flow exercises `_ensure_user_indexes_sync`

The narrowed exception handler is on the upload path. If you have a
test user without an existing per-user collection, upload one small text
file via the OWUI uploads UI. Then:

```bash
docker compose logs --since 1m audrey-ai | grep -iE "qdrant|payload index"
```

Expected: no warning about index creation failing. The collection now
exists with `user` and `file_id` keyword indexes. Re-uploading another
file for the same user must not log any error either (the idempotent
"already exists" path).

If you see an `UnexpectedResponse` propagating, that's a real Qdrant-side
problem the previous code was hiding — investigate Qdrant logs before
rolling back.

### 2.7 Quick Phase 4 Category 4 sweep (optional)

`docs/campaign-2/phase-4-testing.md` Category 4 (KB use) exercises
`kb_search` end-to-end through a chat completion. One trivial prompt
("look up BTRFS in the KB") is enough to confirm the model can still
dispatch the tool and synthesize an answer from the hits.

## 3. Rollback

Plain git revert. No state, no schema, no data touched.

```bash
git revert <phase-6-commit>
docker compose up -d --build audrey-ai
```

The previous code paths re-deploy. The Qdrant collections and per-user
indexes stay as they are — they were correct under both versions.

## Verification status

All Phase 6 smoke tests (2.1-2.7) verified on Unraid 2026-05-17 as
part of the Phase 8 cleanup deploy. Highlights:

- **2.4 happy path:** `httpbin.org/image/png` returned non-empty
  `results` array.
- **2.4 redirect arm:** `picsum.photos/200` returned the new clear
  redirect-with-target message rather than the old opaque 302.
- **2.5 per-user collection merge:** `kb_search_seconds` histogram
  shows non-zero `had_user_collection="true"` count for text
  searches.
- **2.6 upload flow:** `PUT .../index` calls all returned 200; no
  `UnexpectedResponse` propagated through the narrowed handler.
- **2.7 Phase 4 Cat 4 sweep:** end-to-end `kb_search` dispatch
  through OWUI confirmed working with model synthesis from hits.

## 4. Followups

- The deferred chunk-tail finding still wants measurement before any
  fix. A `scripts/measure_chunk_tails.py` walking `/datasets` and
  counting files whose last chunk is mostly overlap would settle whether
  the redundant trailing chunk is worth changing. Cheap to write, but
  has to run on Unraid (the laptop has no `/datasets`).
- The `qdrant.py` module docstring says "Both embedders normalize
  outputs to unit length" — true today, but it now lives next to a
  precondition note in `routes/kb.py` about the same invariant. If the
  embedder layer ever changes, both spots want updating.

## 5. Operational notes

- **The narrowed `except` may now surface failures the old code hid.** If
  you start seeing `UnexpectedResponse` on uploads after deploy, that is
  the fix working, not a new bug. Capture the response body and investigate
  the Qdrant side. The most likely cause is a schema mismatch from a
  collection created under an older version of qdrant-client.
- **`delete_by_source` now returns `None`.** If you have any local code
  outside this repo that called it and asserted on the return, update
  it. Inside the repo, all callers ignored the return already.
- **The redirect error message is new copy.** If anything downstream
  scrapes audrey's error strings for monitoring, the `image embed
  failed: 302 Found` substring is gone; the new substring is `image_url
  returned redirect`.
