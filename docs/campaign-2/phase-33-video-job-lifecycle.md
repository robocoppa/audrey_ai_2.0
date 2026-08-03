# Campaign 2 Phase 33 — video job lifecycle (claim, lease, complete, fail)

[Phase 32](phase-32-video-upload-transport.md) leaves a video sitting at
`status: "pending"` with nothing that will ever pick it up. This phase builds
the lifecycle that lets something claim it — and nothing else. No ffmpeg, no
whisper, no container. A stub worker claims a job and returns a fixed string.

**Status: LANDED AND FULLY VERIFIED** on the box, 2026-08-03. All seven
verification steps pass, including the lease batch: a stale lease refused with
409, two expired leases swept back to `pending` (proved by the attempt counter,
which only a claim increments), and the attempt cap turning a poison job into a
terminal `failed` rather than an endless cycle.

Gained a fourth route during verification — `POST /{file_id}/requeue`. Nothing
could put a processed video back in the queue, so every iteration of Phases
34-38 would have meant deleting the row and re-uploading the source.

Building the state machine before the media that flows through it is the same
call [phase 32](phase-32-video-upload-transport.md) made by proving chunked
transport on file types where a failure cost nothing. When a real transcode
fails later, the question should be "why did ffmpeg fail", not "is it ffmpeg or
is it the lease logic".

---

## Design decisions

### The worker does not write sqlite

[`UploadsDB`](../../src/audrey/kb/uploads_db.py) is deliberately a single
connection, no WAL, guarded by an in-process `threading.Lock` — its docstring
spells out the single-writer contract `reconcile_with_qdrant` depends on. A
second container writing that file breaks the invariant the module is built on,
and breaks it quietly: two writers on a no-WAL sqlite do not fail loudly, they
interleave.

Results come back over HTTP with an `X-Audrey-Service-Token`, exactly as
`custom-tools` already calls home (Phase 31) — `verify_service_token` and
`resolve_kb_caller` in [`auth.py`](../../src/audrey/auth.py) already exist and
already use `hmac.compare_digest`. The single writer stays single.

### The worker pulls work; audrey never pushes

The worker polls for work. Audrey leases the oldest `pending` row and hands back
the file path and metadata. Nothing in audrey needs to know the worker's
address, how many of them there are, or whether one is up.

Push would invert that. Audrey would hold a queue, a retry policy, and a backoff
timer for a container it does not own, and a worker restart mid-video would
strand state on the wrong side of the boundary. With pull, a worker that is down
means jobs accumulate as `pending` — which is exactly where they sit today, so
nothing new breaks when the worker is absent.

### The lease id must still match on completion

This is the failure worth designing for explicitly. A worker stalls past its
lease. The sweep returns the job to `pending`. A second worker claims it and
finishes. Then the first worker wakes up and posts its result.

Without a lease check, that late post overwrites the newer transcript — and the
row looks perfectly healthy afterwards, which is what makes it dangerous. There
is no error, no log line, and no way to notice except by reading a transcript
that does not match its video.

So `ingest-result` presents the `lease_id` it was issued, and a mismatch is a
`409` with the late output dropped.

### Failure is a state, not a silence

`pending` forever is the failure mode to design out. A 300 MB video whose codec
turns out to be unreadable has to be able to say so, in the file list, to the
person who uploaded it. `ingest-failed` takes a reason string and the row
carries it.

## The routes

Four, all service-token, all under the existing `/v1/files` prefix:

- **`POST /v1/files/jobs/claim`** — lease the oldest `pending` row. Sets
  `status` to `processing`, generates a `lease_id`, stamps `leased_at`,
  increments `attempts`. Returns the file path, mime, size, and owning user.
  `204` when the queue is empty, which is the common case and must not be an
  error.
- **`POST /v1/files/{file_id}/ingest-result`** — carries the `lease_id` and the
  produced artifacts. Audrey runs them through the existing
  [`ingest_user_text_file`](../../src/audrey/kb/ingest.py) path and flips the row
  to `ready`. `409` on a stale lease.
- **`POST /v1/files/{file_id}/ingest-failed`** — carries the `lease_id` and a
  reason. `409` on a stale lease.
- **`POST /v1/files/{file_id}/requeue`** — put a processed row back to
  `pending`, clearing the lease, `attempts`, the failure reason, and the
  claimed collection. Added after the first on-box run, when it became obvious
  there was no route back into the queue at all: a video that failed for a
  since-fixed reason could only be deleted and re-uploaded, and every phase
  from 34 on runs a new worker over the same file repeatedly.

### Requeue deletes the Qdrant points first, and fatally

The row reset is the easy half. The subtle half is that the previous run's
chunks have to go, and they have to go *before* the row is touched.

`ingest_user_text_file` deletes by `file_id` before upserting, so a re-run that
produces a transcript cleans up after itself. But a re-run that produces *no*
transcript never calls that path — and the ghost sweep in
`reconcile_with_qdrant` won't collect the leftovers either, because it exempts
`chunks = 0` rows by design (that exemption is what stops it eating the video
queue). Nothing else in the system would ever remove them, so they would stay
searchable under a row claiming to have none.

Doing the delete first also means a Qdrant failure leaves the row exactly as it
was, so the caller retries against unchanged state. The reverse order fails
badly rather than safely: the row would already be `pending` with its points
still live, and the operator would have no signal that anything was left behind.

## What's in scope

- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** — additive
  migration adding `lease_id`, `leased_at`, `attempts`, `failure_reason`, in the
  same `PRAGMA table_info` pattern that added `status`. New methods:
  `claim_job`, `complete_job`, `fail_job`, `sweep_expired_leases`,
  `requeue_job`.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — the four routes,
  guarded by the service token. `FileRow` grows `failure_reason`.
- **`config.yaml`** — a `kb.video` block with `lease_minutes` and
  `max_attempts`, both overridable from `.env` as `VIDEO_LEASE_MINUTES` /
  `VIDEO_MAX_ATTEMPTS`. The override exists because verifying the sweep means
  shortening the lease, and doing that by editing a tracked file on the box
  dirties the deployed tree and has to be reverted by hand — a step that gets
  forgotten, leaving production on a one-minute lease with no symptom beyond
  jobs occasionally running twice.
- **[`static/upload.html`](../../src/audrey/static/upload.html)** — show
  `processing` and `failed` in the file list, with the reason on a failure.
- **`scripts/stub_media_worker.py`** (new) — claims a job, returns a fixed
  string, exits. Exists to exercise the lifecycle end to end without a
  container, and to stay useful afterwards as a way to reproduce lease bugs.

## What's NOT in scope

- **No container.** [Phase 34](phase-34-media-worker-container.md).
- **No ffmpeg, no whisper, no model calls.** Phases 34–36.
- **No concurrency beyond one claim at a time.** Multiple workers are a real
  possibility later; the lease design permits it, but nothing here tests it and
  nothing here should claim it works.

## The parts that will bite

- **Stuck jobs.** A worker crash mid-job leaves `status=processing` forever.
  The lease sweep is the answer, and it needs to run somewhere — boot is the
  obvious place, in the spirit of `reconcile_with_qdrant`.
- **Attempt exhaustion has to terminate.** A video that crashes the worker every
  time must land in `failed` rather than cycling forever and re-crashing the
  worker on every sweep.
- **The claim must not leak across users.** These are fresh internal routes and
  the file path they hand back is a real path on disk. A claim returns whatever
  is oldest, so the *worker* is trusted, but the result path must still write
  into the owning user's collections and nowhere else.
- **`204` is not an error.** An empty queue is the steady state. A worker
  polling an idle audrey must not log an error every few seconds.
- **Boot used to eat the queue.** *(found in testing, fixed here.)*
  `reconcile_with_qdrant`'s prune deletes any sqlite row with no Qdrant
  points. A pending video has none by definition, so every restart emptied
  the queue — silently, leaving the mp4 orphaned on disk. The prune is now
  scoped to rows that claim Qdrant content (`status='ready'` and
  `chunks > 0`), which also spares a completed silent video. Without this,
  the Phase 34 worker would poll an empty queue after every deploy and the
  cause would look like the worker. Pinned by `tests/test_uploads_reconcile.py`.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
```

Still a single image — the stub worker runs from the laptop against the box.
`KB_SERVICE_TOKEN` already exists in `.env` from Phase 31.

## Verification

Run from the laptop. Everything here is hermetically tested already — what
these check is the deployed wiring: the service token the container actually
loaded, the uploads volume as the container sees it, and the boot path.

```bash
export BOX=http://192.168.1.11:8000
export SVC=<KB_SERVICE_TOKEN from .env>
export TOKEN=<a normal user token>  # the OWUI JWT
alias claim='curl -s -o /tmp/j.json -w "%{http_code}\n" -X POST $BOX/v1/files/jobs/claim -H "X-Audrey-Service-Token: $SVC"'
```

**0. The queue survives a restart.** The regression that motivated the
reconcile fix, and it has to be checked first because every step below assumes
a row to claim. Upload a video, confirm it is `pending`, restart `audrey-ai`,
then list again — the row must still be there.

```bash
curl -s $BOX/v1/files -H "Authorization: Bearer $TOKEN" \
  | jq '.files[] | {filename, status, chunks}'
```

**1. Claim on an empty queue returns 204.** With no pending video, `claim`
prints `204` and `/tmp/j.json` is empty. Check the container log stays quiet —
this is the steady state a polling worker produces all day.

**2. A pending video is claimed exactly once.** `claim` twice. First is `200`
with a `lease_id`; second is `204` (or a different `file_id` if more than one
video is queued). The same `file_id` twice is the bug this phase exists to
prevent.

**3. A completed job flips to `ready` and becomes searchable.** Claim, then
post a result with the lease you were given:

```bash
FID=$(jq -r .file_id /tmp/j.json); LEASE=$(jq -r .lease_id /tmp/j.json)
curl -s -X POST $BOX/v1/files/$FID/ingest-result \
  -H "X-Audrey-Service-Token: $SVC" -H 'Content-Type: application/json' \
  -d "{\"lease_id\":\"$LEASE\",\"duration_s\":12.0,\"segments\":[
       {\"t_start\":0,\"t_end\":5,\"text\":\"a distinctive phrase to search for\"}]}"
```

Then confirm the text is retrievable *as that user*, which is the part that
proves attribution rather than mere storage:

```bash
curl -s -X POST $BOX/v1/kb/query -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"distinctive phrase","top_k":3}' \
  | jq '.results[] | {score, source, text}'
```

`source` should name the `<file_id>.transcript.txt` sidecar, which is what
distinguishes "the transcript was ingested" from "something else matched".
Expect global `kb_text` hits alongside it — `_search_text_merged` searches the
user collection and the global one together.

**3b. Requeue puts it back.** Run this before the lease checks below — they all
need a pending row, and by this point the queue is empty.

```bash
python scripts/stub_media_worker.py --endpoint $BOX --requeue $FID
```

The row returns to `pending` with `chunks: 0`, and the phrase from step 3 stops
being returned by `/v1/kb/query` — that second half is the one worth checking,
because it is the only proof the old points actually went.

### Steps 4-6: the lease batch

These three share a setup, so run them as one pass. It also covers phase 34's
step 6 — a killed worker and an abandoning one leave *identical* state (a lease
with no reporter), and `--abandon` reaches it deliberately instead of racing a
sub-second demux with `docker kill`.

**Setup.** Shorten the lease — the default 30 makes this take two hours — and
get the real worker out of the way so it doesn't claim rows from under the
stub. On the box, add one line to `.env`:

```bash
echo 'VIDEO_LEASE_MINUTES=1' >> .env
docker compose up -d audrey-ai      # env + config.yaml are read at boot
docker compose stop media-worker
```

Use the env var, **not** an edit to `config.yaml`. That file is tracked, so
editing it on the box dirties the deployed working tree and has to be undone by
hand before the next pull; `.env` is gitignored and holds deployment-specific
values already. A `config.yaml` edit left behind is also invisible — nothing
reports the effective lease, so production would sit on a one-minute lease and
the only symptom would be jobs occasionally running twice.

From the laptop, using the small `silent.mp4` fixture (a 5s file makes each
cycle instant; the only slow part is waiting out the lease):

```bash
export BOX=http://192.168.1.11:8000
W="python3 scripts/stub_media_worker.py --endpoint $BOX"
FID=$(curl -s $BOX/v1/files -H "Authorization: Bearer $TOKEN" \
      | jq -r '.files[] | select(.filename=="silent.mp4") | .file_id')

$W --requeue $FID          # pending, attempts 0
```

**4. A stale lease is refused.** Claims normally, then posts against a lease id
that was never issued:

```bash
$W --lease bogus           # expect: reported: 409
```

The row stays `processing` under its *real* lease — the 409 must not have
released it. This is the guard against a stalled worker overwriting a newer
run's result with stale output, which leaves no trace afterwards.

**5. A crashed worker's job comes back.**

```bash
sleep 70 && $W --abandon   # sweeps the expired lease, re-claims: "attempt 2"
sleep 70 && $W --abandon   # "attempt 3"
```

The attempt counter rising is the proof the sweep ran — the claim increments
it, so `attempt 2` can only happen if the row was returned to `pending` first.

**6. Attempts terminate.**

```bash
sleep 70 && $W             # sweep sees attempts 3 >= max_attempts, fails it
                           # then prints "queue empty" — nothing left to claim
curl -s $BOX/v1/files -H "Authorization: Bearer $TOKEN" \
  | jq '.files[] | select(.filename=="silent.mp4") | {status, failure_reason}'
```

Expect `failed` with `abandoned by the media worker after 3 attempt(s)`. A
poison video that cycled forever would take the worker down on every pass, so
this terminating is what makes the queue safe to leave running unattended.

**Teardown.** Drop the override, restart, requeue the fixture, and bring the
worker back. The requeue is not optional — step 6 deliberately leaves the row
`failed`, and a `failed` row is never claimed again:

```bash
# on the box
sed -i '/^VIDEO_LEASE_MINUTES=/d' .env
docker compose up -d audrey-ai
docker compose start media-worker

# from the laptop
$W --requeue $FID
```

Confirm the override is actually gone — this is the step that gets skipped:

```bash
docker exec audrey-ai python3 -c "import os; print(os.environ.get('VIDEO_LEASE_MINUTES','unset'))"
```

**7. The routes are service-only.** All three must be `401` with no token, with
a wrong token, and with a *valid user JWT* — the last is the one worth checking,
because these routes hand back real filesystem paths and `resolve_kb_caller`
would have accepted that JWT quite happily.

```bash
curl -s -o /dev/null -w "no token:    %{http_code}\n" -X POST $BOX/v1/files/jobs/claim
curl -s -o /dev/null -w "wrong token: %{http_code}\n" -X POST $BOX/v1/files/jobs/claim \
  -H "X-Audrey-Service-Token: nope"
curl -s -o /dev/null -w "user JWT:    %{http_code}\n" -X POST $BOX/v1/files/jobs/claim \
  -H "Authorization: Bearer $TOKEN"
```

### Rollback

Revert `audrey-ai`. The added columns are additive and harmless to older code —
`status` still reads `pending`, which is where those rows already were.

## What this unblocks

Something can now own a video and be held to it. [Phase
34](phase-34-media-worker-container.md) makes that something a real container.
