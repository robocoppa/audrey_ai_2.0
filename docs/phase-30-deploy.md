# Phase 30 — KB reconcile pass

The Phase 27 watcher fixes the *steady-state* leak: while running, a
deleted file's vectors get cleaned up via `on_deleted` / `on_moved`.
But anything that happens while the watcher isn't running — container
restart, `KB_WATCHER_ENABLED=0` for a stretch, bulk `rm` on
`/mnt/user/knowledge/...` — leaves stale vectors in `kb_text` and
`kb_images` forever, slowly polluting retrieval results.

Phase 30 adds a periodic reconciler that scrolls every point in the
two global collections, groups by `payload.source`, checks each unique
source against `Path.exists()`, and calls `delete_by_source` for any
orphan.

What stays the same:
- Watcher behavior (Phase 27) — still event-driven, debounced,
  emits delete-before-ingest within a flush.
- Per-user collections (`kb_user_text_*`, `kb_user_images_*`) —
  managed by Phase 15's sqlite + startup reconcile, NOT touched by
  Phase 30. Don't add them to `_scroll_sources`; the per-user upload
  flow stores chunks with no `payload.source`, so the reconciler
  would happily delete every per-user file.
- All search / ingest / admin / chat routes.

What changed:
- **`src/audrey/kb/reconcile.py`** (new) — `reconcile_once(qdrant)`
  pure function + `KBReconciler` lifecycle wrapper. Returns a
  structured `ReconcileResult` with per-collection breakdown
  (checked / orphans_deleted / points_in_orphans / elapsed_s / error).
- **`src/audrey/main.py`** — instantiate `KBReconciler` in lifespan
  alongside the watcher; start in `try`, stop in `finally`. Readiness
  log gains `kb_reconcile=on|off`.
- **`config.yaml`** — new block `kb.reconcile.{enabled: true, interval_s: 1800}`.
- **`src/audrey/routes/admin.py`** — new `POST /v1/admin/kb/reconcile`
  endpoint (`require_admin`) for ad-hoc sweeps. Returns the same
  `ReconcileResult.to_dict()` payload the periodic loop logs.
- **`tests/test_kb_reconcile.py`** (new) — 9 cases covering: no orphans,
  orphan deletion, both collections, points-without-source skip,
  missing-collection-handled, duplicate-source collapse, response shape,
  `interval_s=0` disables loop, clean shutdown.

Out of scope (deliberately):

- **Per-user collections.** Phase 15's sqlite uploads index already
  reconciles those at startup. Mixing the two paths would let this
  reconciler delete legitimately-uploaded files just because the host
  can't see the user's filename (which it can't — uploads live in
  qdrant only, no on-disk source). Adding `kb_user_*` to the sweep
  would corrupt per-user state.
- **Per-source change detection.** The reconciler only catches
  *deletions*. File modifications still rely on watcher events. mtime-
  based reingest is a future phase if it earns its place.
- **Cloudflared exposure.** The admin endpoint stays behind the
  existing `^/v1/admin` rule (already tunneled, requires admin token).
- **Metrics.** No new Prometheus counter — `audrey_kb_reconcile_total`
  was considered but added cardinality without a clear question to
  answer. Periodic logs cover the observability need.

**Prereqs:** all phases through 29 verified. No env vars, no compose
changes. Config defaults `enabled: true`; opt out with `enabled: false`
or `interval_s: 0` if you want the admin endpoint without the loop.

---

## 1. Deploy

```bash
# Laptop:
git pull   # after the user has committed Phase 30

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build audrey-ai
docker compose logs --since 1m audrey-ai | grep -E "ready|kb.reconcile"
```

Expect a clean readiness line ending with `kb_watcher=on; kb_reconcile=on; pipeline=compiled`,
followed by `kb.reconcile: periodic sweep every 1800s`.

---

## 2. Smoke tests

Set the auth env once:

```bash
ADMIN_TOKEN="<your OWUI bearer token, role=admin>"
```

### 2.1 Reconciler started cleanly

```bash
docker compose logs --since 2m audrey-ai | grep -E "kb.reconcile"
```

Expected: `kb.reconcile: periodic sweep every 1800s` (or whatever
`interval_s` you set).

### 2.2 Admin endpoint with no orphans

Run a sweep before deleting anything. Should report zero orphans on
both collections.

```bash
curl -s -X POST http://localhost:8000/v1/admin/kb/reconcile \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq
```

Expected JSON shape:

```json
{
  "by_collection": {
    "kb_text": {
      "checked": 12345,
      "orphans_deleted": 0,
      "points_in_orphans": 0,
      "elapsed_s": 0.342,
      "error": ""
    },
    "kb_images": {
      "checked": 89,
      "orphans_deleted": 0,
      "points_in_orphans": 0,
      "elapsed_s": 0.041,
      "error": ""
    }
  },
  "total_orphans_deleted": 0,
  "total_elapsed_s": 0.387
}
```

### 2.3 Stage an orphan and verify cleanup (the headline test)

This is the load-bearing smoke check — proves the watcher-can't-see
case actually heals.

**Key constraints learned from the first verification attempt:**
- The host path `/mnt/user/knowledge/geology/` IS the watched directory
  (bind-mounted to `/datasets/geology` in audrey-ai). Write/delete on
  the host side; the container sees it via the bind.
- `echo 'one short line' > file.md` may not produce any chunks (chunker
  min-tokens). Use a **multi-paragraph** file so ingestion produces
  visible chunks you can verify.
- After staging, **wait for the watcher log line BEFORE stopping the
  container.** Don't move on until ingestion is confirmed.

```bash
# 1. Stage a real multi-paragraph file in a watched directory. Use the
#    host path; the watcher inside the container sees it via the bind.
cat > /mnt/user/knowledge/geology/_phase30_orphan.md <<'EOF'
# Phase 30 reconcile smoke test

This is the first paragraph. It needs to be substantial enough that the
chunker produces at least one chunk (default min_tokens around 100).

This is the second paragraph, just to be sure we cross the chunk
threshold without depending on tokenizer specifics. Geology, basalt,
quartzite, schist — filler text that's domain-relevant in case the
chunker filters by topic, which it doesn't, but defensively anyway.
EOF

# 2. Tail the watcher log until it ingests. Don't proceed until you
#    see the "reingested text" line. Up to ~10s on a cold embedder.
docker compose logs -f --since 5s audrey-ai &
LOG_PID=$!
sleep 12   # generous; watcher debounce_s=2 + embed time
kill $LOG_PID 2>/dev/null
# You must see: "kb.watcher: reingested text /datasets/geology/_phase30_orphan.md -> N chunks"
# If you don't, STOP and investigate before continuing — there's no orphan to clean.

# 2b. Confirm via qdrant count delta (kb_text grew by N chunks):
curl -s http://localhost:8000/v1/kb/stats | jq

# 3. Stop audrey, delete the file from disk, restart audrey.
#    With audrey down, the watcher can't see the delete — exactly the
#    drift case Phase 30 fixes.
docker compose stop audrey-ai
rm /mnt/user/knowledge/geology/_phase30_orphan.md
docker compose start audrey-ai
sleep 8   # let ready log emit

# 4. Verify the orphan vectors are still in qdrant (watcher missed it).
curl -s http://localhost:8000/v1/kb/stats | jq
#    kb_text count should be unchanged from step 2b (the points are
#    still there even though the file is gone).

# 5. Trigger reconcile.
curl -s -X POST http://localhost:8000/v1/admin/kb/reconcile \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq

# 6. Look for the delete log line.
docker compose logs --since 30s audrey-ai | grep "_phase30_orphan"
# Expected: "kb.reconcile: deleted orphan /datasets/geology/_phase30_orphan.md (N points) from kb_text"
# Expected response: total_orphans_deleted >= 1, points_in_orphans = N

# 7. Confirm cleanup via qdrant count.
curl -s http://localhost:8000/v1/kb/stats | jq
#    kb_text count should be back to the pre-step-1 baseline.
```

**Troubleshooting:**
- **No "reingested text" line in step 2:** the watcher didn't ingest.
  Check `KB_WATCHER_ENABLED=1` is set (readiness log shows
  `kb_watcher=on`). Check the file path is under `/datasets/...`
  inside the container, not under `/mnt/user/...` (host path → no
  ingest). Look for `kb.watcher: no valid roots, not starting`.
- **Reconcile reports 0 orphans in step 5 even though the file is gone:**
  the orphan was never ingested in step 1. The reconcile is working;
  there's nothing to clean. Re-run from step 1 with a longer file or
  longer wait.
- **`kb_text` count didn't drop after reconcile:** `delete_by_source`
  in qdrant is async-style; the `wait=True` flag in `delete_by_source`
  should make it visible immediately, but if you scroll fast on a busy
  qdrant you might see stale counts. Wait 1-2s and re-check.

### 2.4 Per-user uploads NOT touched

Confirm the reconciler doesn't delete uploaded files (which have no
`payload.source`).

```bash
# Get baseline upload count for a user.
curl -s http://localhost:8000/v1/files \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq '. | length'

# Run reconcile.
curl -s -X POST http://localhost:8000/v1/admin/kb/reconcile \
  -H "Authorization: Bearer $ADMIN_TOKEN" > /dev/null

# Re-check upload count — must be unchanged.
curl -s http://localhost:8000/v1/files \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq '. | length'
```

Expected: identical numbers before and after. Per-user collections are
NOT in `_scroll_sources`'s scope (only `kb_text` + `kb_images`).

### 2.5 Periodic sweep fires on its own

To verify the loop without waiting 30 minutes, drop `interval_s` to 60s
in `config.yaml` temporarily, restart, wait ~70s, and check logs.

```bash
# Edit config.yaml: kb.reconcile.interval_s: 60
docker compose up -d --build audrey-ai
sleep 70
docker compose logs --since 90s audrey-ai | grep "kb.reconcile: pass complete"
```

Expected: a `kb.reconcile: pass complete; orphans_deleted=N elapsed=...`
log line. Restore `interval_s: 1800` after.

### 2.6 Auth gating

```bash
# Without auth token → 401:
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/v1/admin/kb/reconcile

# Non-admin user token → 403:
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/v1/admin/kb/reconcile \
  -H "Authorization: Bearer $REGULAR_USER_TOKEN"
```

Expected: `401` and `403` respectively. Same chain that protects
`/v1/admin/auth/clear`.

---

## 3. Rollback

The reconciler is purely additive. To disable without rebuild:

```yaml
# config.yaml:
kb:
  reconcile:
    enabled: false
```

Then `docker compose up -d audrey-ai` to reload config. The admin
endpoint stays available either way (it doesn't check the enabled
flag — only the periodic loop does).

To remove the code entirely, revert the four touched files
(`src/audrey/kb/reconcile.py`, `src/audrey/main.py`,
`src/audrey/routes/admin.py`, `config.yaml`) plus delete
`tests/test_kb_reconcile.py`, then rebuild.

---

## 4. Operational notes

- **First sweep waits one full `interval_s`.** Intentional — gives
  qdrant init time to settle, and the admin endpoint is available
  for an immediate sweep if you need one sooner.
- **Reconcile and watcher coexist.** They run on the same qdrant
  client. If a watcher delete fires for a file mid-reconcile, the
  reconciler's subsequent `delete_by_source` is a no-op (qdrant
  treats deleting nonexistent vectors as success). Order doesn't
  matter.
- **`interval_s=0` disables the loop entirely** but keeps the admin
  endpoint live. Useful if you want manual-only sweeps and don't
  want a background task burning even the small CPU of an idle
  `asyncio.sleep`.
- **`enabled: false` is the cleaner off-switch** — no `KBReconciler`
  instance is created at all. Use this if you're truly opposed to
  the feature; `interval_s=0` is "loop off, endpoint on."
- **Sweep cost on this scale:** with ~10k points across both
  collections, a sweep takes ~300ms (verified on the laptop tests
  scaled up). The 30-minute default has ~1800x headroom. If KB grows
  past 100k points and sweeps start hitting multi-second latency,
  consider lowering `_SCROLL_PAGE_SIZE` (currently 256) or moving to
  an event-driven reconcile triggered by container restart events.
- **Log lines this phase introduces:**
  - Startup: `kb.reconcile: periodic sweep every <N>s` (when enabled)
  - Startup with `interval_s=0`: `kb.reconcile: interval_s=0, periodic loop disabled`
  - Per orphan deleted: `kb.reconcile: deleted orphan <path> (<N> points) from <collection>`
  - End of each sweep: `kb.reconcile: pass complete; orphans_deleted=<N> elapsed=<s>s (kb_text: x/y, kb_images: x/y)`
  - Failure (rare): `kb.reconcile: <collection> sweep failed: <error>` — loop survives, retries next interval.
- **The 9 `ASYNC240` ruff warnings are accepted.** Calling `Path.exists()`
  inside an async function technically blocks the event loop. We're
  not on a tight latency budget here (this loop runs every 30 min,
  not per-request) and local-SSD `exists()` is sub-millisecond. If
  you ever notice event-loop stalls during reconcile, wrap each
  `Path(source).exists()` in `asyncio.to_thread`. For now the warnings
  stay as pressure rather than getting masked.
- **No new Prometheus metric.** Considered `audrey_kb_reconcile_total`
  but the question it would answer ("are reconciles running?") is
  better answered by `kb.reconcile: pass complete` log lines feeding
  into Loki/journald. Add the metric only if there's a real "is the
  reconciler dead?" alert that needs it.
