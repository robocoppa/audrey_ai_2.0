# Phase 27 — SSRF guards + watcher on_deleted

Two unrelated medium-severity findings from the 2026-04-29 audit,
bundled into one phase because both are bounded-scope hardening with
no architectural overlap.

**A. SSRF guards on `/v1/kb/query/image`** — `_fetch_image` accepted
arbitrary URLs with redirects and no byte cap. An authenticated user
could probe internal services via 302s, hit Audrey's own loopback
endpoints, or OOM the container with a multi-GB response. Phase 26
required auth on the route; Phase 27 adds defense-in-depth at the
fetch layer.

**B. KB watcher missing `on_deleted` + broken `on_moved`** — files
removed from `/mnt/user/knowledge/...` left stale vectors in qdrant
forever, slowly polluting retrieval. Renames left vectors at the
old path AND created new vectors at the destination (double indexing).

What stays the same:
- Public KB query route surface (still POST `/v1/kb/query/image` with
  the same `ImageQuery` schema). Errors still surface as 422.
- Image embed flow for `image_b64` and `query` (text→image) — these
  paths don't fetch URLs and aren't affected.
- Watcher startup, debounce window, observer threading, lifecycle.
- All other watcher-handled events (create, modify) — same behavior.

What changed:
- **`src/audrey/kb/embed.py`** — added `_validate_image_url()` +
  `_is_unsafe_address()` helpers. `_fetch_image` now: validates URL
  before any I/O, disables redirects, streams response with a 25 MB
  cap, requires `image/*` content-type.
- **`src/audrey/kb/watcher.py`** — `_QueueHandler` adds `on_deleted`,
  emits `(kind, path)` tuples instead of bare `Path`. `on_moved`
  emits two events (delete src, ingest dest). `KBWatcher._run` keys
  the debounce dict on `(kind, path)` and dispatches to either
  `_handle_ingest` or `_delete_vectors`. Deletes processed before
  ingests within the same debounce flush so rename ordering is sane.

Out of scope (deliberately):

- **DNS rebinding defense.** A malicious server can return a public IP
  on first lookup (passes our check), then a private IP on the actual
  fetch. Real mitigation requires resolving once and connecting to the
  resolved IP with the hostname in SNI. Adds connection-pooling
  complexity. Bounded by Phase 26 requiring auth on the route.
- **Periodic reconcile pass.** If the watcher misses an event (process
  restart, container down when a file gets deleted), stale vectors
  persist. A background "scroll qdrant, check disk, prune orphans"
  job is a future phase. Not added because: typical operating mode is
  watcher always-on, and Phase 27 fixes the steady-state leak.
- **`http://` allowlist for trusted hosts.** Some image CDNs only
  serve over plain HTTP. Phase 27 hardcodes https-only for simplicity;
  if a real workflow needs http, add a `KB_IMAGE_ALLOW_HTTP=1` env or
  a per-request flag, but only if there's actual demand.
- **Auditing the existing global `kb_text` / `kb_images` for stale
  vectors.** This phase only prevents NEW staleness from accumulating.
  If you want to clean current staleness, delete + re-ingest the
  affected dataset with `audrey-ingest --purge`.

**Prereqs:** all phases through 26 verified. No env vars, no
migrations, no schema changes.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

Phase 24b's split means rebuild is ~6 seconds.

---

## 2. Smoke tests — Piece A (SSRF)

Each curl is a single line — copy-paste each as a unit.

### 2.1 Loopback rejected

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"https://127.0.0.1/img.jpg"}' http://localhost:8000/v1/kb/query/image
```

Expect: HTTP 422 with detail like `"image embed failed: image_url
host '127.0.0.1' resolves to a private, loopback, link-local, or
otherwise non-public address"`. Pre-fix this would have actually
attempted to GET https://127.0.0.1/img.jpg from inside the container
(which would fail with a connection error, but the *fact that we
tried* is the leak — could have hit a real internal service).

### 2.2 Docker-internal hostname rejected

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"https://qdrant:6333/collections"}' http://localhost:8000/v1/kb/query/image
```

Expect: 422. Either the scheme check fires (if qdrant only speaks
http) or the IP check fires (qdrant resolves to a 172.x docker IP
which `_is_unsafe_address` correctly classifies as private).

### 2.3 RFC1918 private IP rejected

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"https://10.0.0.5/img.jpg"}' http://localhost:8000/v1/kb/query/image
```

Expect: 422 with the same private-address detail.

### 2.4 http:// scheme rejected

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"http://example.com/img.jpg"}' http://localhost:8000/v1/kb/query/image
```

Expect: 422 with detail mentioning the https-only requirement.

### 2.5 Real public image works

A small public image — the point is to confirm we didn't break
legitimate fetches. Use a Brave Search image-CDN URL (these are
served behind Brave's CDN and are reliably reachable from Unraid's
network; the older Wikimedia thumbnail-URL form 403'd or DNS-failed
during testing).

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"https://imgs.search.brave.com/5sa_7ONUZiCVUdjeQO7WkvaSy3RWsIdrDfcJmZ44gWA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvMjE3/NzE3NDQ5Ny9waG90/by9sZXR0ZXItcy1w/YWludGVkLWluLWdv/bGQtYW5kLXdoaXRl/LW9uLWEtYmxhY2st/YmFja2dyb3VuZC1h/bmQtcGhvdG9ncmFw/aGVkLW9uLXRoZS1m/YWNhZGUtb2YtYS5q/cGc_cz02MTJ4NjEy/Jnc9MCZrPTIwJmM9/ZVRkd25uN1phNFJz/QXliSzBUcVV4bUpW/LUZUQ21ZMHBnNHVF/ZU9wV1hzVT0","top_k":3}' http://localhost:8000/v1/kb/query/image | head -c 400
```

Expect: HTTP 200 with a JSON response containing `"results": [...]`
(may be empty if no images in `kb_images` match). The presence of a
JSON body — not 422 — proves the fetch succeeded and embedding ran.

If you want to swap to a different image source, any small `https://`
URL serving an `image/*` content-type from a non-private IP works.
Brave's CDN happens to be convenient because every Brave Search
image hit produces a stable proxy URL.

### 2.6 Redirect rejected

A request that would normally 302 to somewhere; httpx with
`follow_redirects=False` should fail because the response status is
3xx, not 2xx, so `r.raise_for_status()` triggers.

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"image_url":"https://httpbin.org/redirect-to?url=https%3A%2F%2Fexample.com%2Fimg.jpg"}' http://localhost:8000/v1/kb/query/image
```

Expect: 422 with detail mentioning a 3xx HTTP error from httpx.
Pre-fix would have followed the redirect to whatever the attacker
chose.

---

## 3. Smoke tests — Piece B (watcher on_deleted)

> **Run these from the repo directory.** `docker compose logs ...`
> reads the local `compose.yaml`, so if you're not in
> `/mnt/user/appdata/audrey_ai_2.0/` you'll get
> `no configuration file provided: not found`. Either `cd` there
> first or use `docker compose -f
> /mnt/user/appdata/audrey_ai_2.0/compose.yaml logs ...`.

> **Watcher must be enabled.** Check the audrey-ai readiness log:
> ```bash
> cd /mnt/user/appdata/audrey_ai_2.0
> docker compose logs --tail 30 audrey-ai | grep "kb_watcher="
> ```
> If you see `kb_watcher=off`, the watcher isn't running and the
> on_deleted handler can't fire (the code is loaded; it's just not
> observing). To turn it on, add `KB_WATCHER_ENABLED=1` to the
> `audrey-ai` service environment in `compose.yaml`, then
> `docker compose up -d audrey-ai`. After restart, the readiness log
> should show `kb_watcher=on` plus a `kb.watcher: watching N root(s)`
> line shortly after.
>
> If you don't want auto-reingest at all (some operators prefer
> running `audrey-ingest` manually), Phase 27 piece B's runtime
> verification is N/A — the code is shipped + AST-clean but won't
> exercise. SSRF (Piece A) verifies independently.

Picking a real test file is the trickiest part. Use a throwaway under
one of the existing `KB_DATASET_PATHS` — `/datasets/geology/` is a
safe bet since it's already populated. **Don't delete a real KB
file.**

### 3.1 Stage a throwaway file + ingest it

```bash
docker exec audrey-ai sh -c "echo 'phase-27 test content about granite formations' > /datasets/geology/_phase27_test.md"
```

The watcher should pick this up within `debounce_s` (default 2s) and
ingest. Wait ~10s and check:

```bash
docker compose logs --since 30s audrey-ai | grep "kb.watcher: reingested text /datasets/geology/_phase27_test.md"
```

Expect: a line like `kb.watcher: reingested text
/datasets/geology/_phase27_test.md -> 1 chunks`. If you don't see it,
the watcher isn't picking up events — verify
`docker exec audrey-ai env | grep KB_WATCHER` shows `KB_WATCHER_ENABLED=1`.

### 3.2 Confirm vectors exist

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"query":"phase-27 test granite","top_k":5}' http://localhost:8000/v1/kb/query | jq -r '.results[] | "\(.score)  \(.source)"'
```

Expect: at least one hit with `_phase27_test.md` in the source path
and a score > 0.4.

### 3.3 Delete the file → vectors should go

```bash
docker exec audrey-ai rm /datasets/geology/_phase27_test.md
```

Wait ~10s for the debounce to flush. Check the logs:

```bash
docker compose logs --since 30s audrey-ai | grep "kb.watcher: requested delete"
```

Expect: `kb.watcher: requested delete of vectors for
/datasets/geology/_phase27_test.md`. Pre-fix this log line wouldn't
have existed and the vectors would still be in qdrant.

Re-run the query from 3.2:

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"query":"phase-27 test granite","top_k":5}' http://localhost:8000/v1/kb/query | jq -r '.results[] | "\(.score)  \(.source)"'
```

Expect: zero hits with `_phase27_test.md` in the source path. If the
hits remain, the qdrant `delete_by_source` call didn't take effect
(check qdrant logs in case the request errored).

### 3.4 Rename moves vectors (optional)

Only run this if you want full coverage of the on_moved fix.

```bash
docker exec audrey-ai sh -c "echo 'phase-27 rename test about basalt and igneous rocks' > /datasets/geology/_phase27_rename_a.md"
sleep 10
```

Confirm ingest:
```bash
docker compose logs --since 30s audrey-ai | grep "_phase27_rename_a.md"
```

Then rename:
```bash
docker exec audrey-ai mv /datasets/geology/_phase27_rename_a.md /datasets/geology/_phase27_rename_b.md
sleep 10
```

Check logs for both events:
```bash
docker compose logs --since 30s audrey-ai | grep "_phase27_rename"
```

Expect to see:
- `requested delete of vectors for /datasets/geology/_phase27_rename_a.md`
- `reingested text /datasets/geology/_phase27_rename_b.md -> 1 chunks`

Both events come from the same `on_moved` callback, dispatched as two
queue items. Pre-fix only the second line would have appeared — vectors
for the `_a.md` source would have stayed forever.

Cleanup:
```bash
docker exec audrey-ai rm -f /datasets/geology/_phase27_rename_b.md
```

---

## 4. Rollback

```bash
git checkout <previous-sha> -- src/audrey/kb/embed.py src/audrey/kb/watcher.py
docker compose up -d --build audrey-ai
```

Two-file revert. SSRF guards just go away (back to "fetch any URL");
watcher loses on_deleted handling (back to "leaks vectors on delete").
No data, config, or schema state changes.

---

## 5. Operational notes

### When a real workflow needs http://

If you ever need to embed an image from an http-only host (rare in
2026 but happens — some old hobbyist mirrors still don't have TLS),
the surgical fix is to add `"http"` to `_ALLOWED_IMAGE_SCHEMES`. The
private-IP check still applies, so you don't lose the SSRF defense
— just the transport-encryption requirement.

### Watcher delete log noise

`kb.watcher: requested delete of vectors for <path>` fires on every
delete event, including for files we never ingested (qdrant returns
silently for no-op deletes, but we don't know that ahead of time).
The suffix-allowlist filter in `_QueueHandler._enqueue` already
filters out `~`/`.swp`/dotfiles, so noise should be limited. If you
add a new editor that creates differently-named scratch files, the
filter may need tightening.

### qdrant-client doesn't expose deleted-count

`delete_by_source` in `kb/qdrant.py` returns `0` or `-1` as a
"did the call succeed" flag — qdrant's UpdateResult doesn't carry
the actual count of deleted points. So the watcher log says
"requested delete" rather than "deleted N points." If we want the
real count we'd need a `count + delete + count` sequence, which is
a future quality-of-life thing.

### What this doesn't cover

- The audit's medium #4 (`/v1/tools` listing leaks tool/server
  details) was not addressed — the listing is read-only and Phase 26
  added admin requirement to `/v1/tools/rediscover`. Listing remains
  available to `require_user`-level callers; if that's a concern,
  promote it to admin-only in a future phase.
- The audit's medium #6 (broad ports) was partially addressed in
  Phase 26 (Grafana password). Custom-tools port 8001 binding to host
  remains for debugging convenience.

### Watcher event sequencing

`_run` sorts pending events with deletes-before-ingests within each
flush. This handles the typical `mv old.pdf new.pdf` case (delete
old's vectors before ingesting new). It does NOT handle the inverse
case where an editor uses save-as-temp + rename — in that case the
event ordering is `(create, tmpfile)`, `(modify, tmpfile)`,
`(move, tmpfile→target)`. The move's delete-of-tmpfile wouldn't
happen because the tmpfile suffix is filtered out by the allowlist.
For target ingest to work, the editor needs to write directly with
the target's final suffix.
