# Campaign 2 Phase 32 — video upload transport (chunked parts past the edge cap)

Make a 300 MB video *arrive*. Not understood — arrived: stored, indexed, and
honest about the fact that nothing has read it yet. Understanding it is
[phase 33](phase-33-video-job-lifecycle.md) onward.

**Status: DEPLOYED + verified on box 2026-08-02.** A 288 MB mp4 crossed the
Cloudflare tunnel in 36 parts, assembled server-side, and indexed with
`status: "pending"`. `audrey_passthrough/qwen3-vl:32b` confirmed reachable
through `/v1/models`. 908 hermetic tests pass, ruff clean.

**Not yet deployed:** the upload progress bar and
[`media/framegate.py`](../../src/audrey/media/framegate.py), both landed after
the box rebuild. Neither changes server behaviour — the gate has no caller
until [phase 36](phase-36-video-visual-assessment.md).

---

## Why this exists

Two hard walls, both hit by a single 300 MB upload:

1. **Cloudflare caps request bodies at 100 MB** on Free/Pro/Business. That is a
   plan limit, not a config we own, and `/upload` + `/v1/files/*` route through
   the tunnel (Phase 14 ingress). A 300 MB `POST` is refused at the edge and
   never reaches `audrey-ai` — the user sees a 413 whose body is Cloudflare's
   HTML error page, not our JSON. That is the literal bug that opened this
   phase: `Upload failed (413): unknown error`, with no `detail` to report
   because there was no JSON to read.
2. **`POST /v1/files` does everything inline** — stream, sniff, extract, embed,
   upsert Qdrant, write sqlite, respond. Transcribing an hour of footage is
   minutes of GPU at best. It cannot live inside a request, so the upload and
   the processing have to become two different things.

This phase solves the first wall completely and prepares for the second by
giving every upload a `status`.

## What shipped

```text
browser ──▶ POST /v1/files/upload-sessions              → {upload_id, part_size, parts_total}
        ──▶ PUT  /v1/files/upload-sessions/{id}/parts/{n}    (8 MB each, under the edge cap)
        ──▶ POST /v1/files/upload-sessions/{id}/complete
                → {file_id, status: "ready" | "pending"}
```

Files at or under the single-shot cap still take the ordinary `POST /v1/files`
path untouched. The client picks by size.

**Honest limits, published by the server.** `GET /v1/files` carries a `Limits`
block — `max_upload_bytes`, `max_user_bytes`, `allowed_extensions`,
`chunked_max_bytes`, `part_size`. The page can no longer advertise a cap that
`config.yaml` does not set, and the pre-check has real numbers to refuse
against.

**A pre-check that fails open.** The client refuses empty, oversized,
unsupported-extension and over-quota files before putting bytes on the wire,
accumulating pending sizes across a batch so ten files that each individually
fit cannot all pass and then fail server-side. When `limits` is null it defers
to the server — a client that cannot see the limits must never be the reason an
upload is refused.

**A 413 that says who rejected it.** `describeFailure` reads the body once,
tries `JSON.parse`, and special-cases a non-JSON 413 as *"rejected before it
reached Audrey"*. The fix for that lives in a different place than a limit the
app controls, so the message has to distinguish them.

**mp4 accepted for storage.** Scope grew mid-phase, deliberately: rather than
defer video mime to the worker phases, mp4 is accepted now and parked at
`status: "pending"`. That let the transport be proved on the real payload
instead of a stand-in. The row reports `chunks: 0` and an empty collection, and
the UI says *"awaiting processing, not searchable yet"* — reporting "0 chunks"
as a success would read as a silent failure.

**The `vl` models allowed through passthrough.** `passthrough.allowed_models`
was text-only, so a future describe call would have returned `403 Passthrough
not allowed for model 'qwen3-vl:32b'`. Both members of the `vl` pool are now
allowed, unblocking [phase 36](phase-36-video-visual-assessment.md)'s model path
before any worker exists. Side effect accepted deliberately: `/v1/models` lists
one entry per allowed concrete, so both are now directly targetable by any
authenticated OWUI user.

**The keyframe gate**, built early because it needed no container. See
[phase 38](phase-38-video-optimise.md) for what it is and why it landed
out of order.

## What's in scope

- **[`kb/extract.py`](../../src/audrey/kb/extract.py)** — `ALLOWED_VIDEO_MIMES`
  (`video/mp4` only) folded into `ALLOWED_MIMES`; the suffix map hoisted to
  module level as `SUFFIX_MIMES` so `ALLOWED_EXTENSIONS` is **derived** from it
  and the two can never drift.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** — `status TEXT NOT
  NULL DEFAULT 'ready'` on `uploads`, plus `upload_sessions` / `upload_parts`.
  Migration via `PRAGMA table_info` + `ALTER TABLE`, because the deployed sqlite
  predates the column and `CREATE TABLE IF NOT EXISTS` will not add it.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — the three session
  routes, plus `_validate_and_ingest` factored out so single-shot and chunked
  share one sniff/quota/ingest/rollback path rather than drifting apart.
- **[`static/upload.html`](../../src/audrey/static/upload.html)** — pre-check,
  failure reporting, chunked slice loop, per-file result lines, progress bar.
- **`config.yaml`** — a `kb.chunked` block (`part_size_mb: 8`,
  `max_upload_mb: 2048`, `session_ttl_minutes: 120`) and the two `vl` entries in
  `passthrough.allowed_models`.

## What's NOT in scope

- **No cloudflared ingress change.** Chunked parts are ordinary `PUT`s to
  `/v1/files/*`, already routed by the Phase 14 rule.
- **No raising `max_upload_mb` past 100.** Once chunking exists, per-part size
  is what matters. A single-shot cap above the edge limit is a limit the edge
  silently overrides — the exact failure this phase exists to remove.
- **No processing of the stored video.** That is deliberate and visible: the row
  says `pending` and the UI says so too.
- **No video mimes beyond mp4.** quicktime, webm and matroska stay refused until
  something actually reads them.

## The parts that bit

- **Part ordering.** Assembly iterates `range(parts_total)` by number, not a
  directory glob — lexical sort puts part 10 before part 2 and produces a
  corrupt file that passes every size check.
- **Retried parts.** `record_part` is `INSERT OR REPLACE`, so a client that
  retries a part after a transient failure does not double-count the bytes
  toward the quota.
- **Re-picking a rejected file.** Selecting the same file again fires no
  `change` event unless the input is cleared, which makes a rejection look like
  a hang. `picker.value = ""` after every batch.
- **The sniff gate still holds.** Adding a video mime widens `ALLOWED_MIMES`;
  libmagic stays the gate, the extension stays a hint.
- **Orphaned partials.** A closed tab leaves parts in the session dir.
  `session_ttl_minutes` and `stale_sessions` exist for the sweep; wiring the
  sweep to boot is still open.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
```

Single image. No new container, no compose change, no ingress change.

## Verification

**1. The limits are published.**

```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files | jq '.limits'
```

**2. A large file crosses in parts.** ✅ Verified — 288 MB mp4, 36 parts,
assembled, indexed `pending`.

```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files \
  | jq '.total_bytes, (.files[] | select(.status == "pending"))'
```

**3. Chunked transport round-trips a file that already worked.** A ~5 MB PDF
through the session routes must land with the same chunk count as single-shot.

**4. The vl models are reachable through passthrough.** ✅ Verified.

```bash
curl -sS https://chat.builtryte.xyz/v1/models | jq -r '.data[].id' | grep -Ei 'vl|llava'
```

Note `/v1/models` is unauthenticated ([routes.py:49](../../src/audrey/routes/openai/routes.py#L49)),
so a pass here says nothing about your token.

**5. User isolation.** A second user must not see another's sessions or parts.
The session routes are fresh surface and key on `me.email` throughout.

### Rollback

Revert `audrey-ai`. Single-shot uploads keep working — this phase leaves that
route in place and untouched.

## What this unblocks

Video stops failing with an opaque 413 after a long wait. It now arrives, and
sits in a state that names itself. [Phase 33](phase-33-video-job-lifecycle.md)
gives that state a lifecycle and something to claim it.
