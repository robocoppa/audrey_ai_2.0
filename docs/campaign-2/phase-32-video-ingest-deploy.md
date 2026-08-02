# Campaign 2 Phase 32 — video ingest via a media-worker sidecar

Make video a first-class upload type. A user drops a 300 MB mp4 on `/upload`
and Audrey ends up with four artifacts, all searchable and all attributed to a
timestamp in the source:

1. **A transcript** of the audio.
2. **A visual assessment** — what is actually on screen, in words.
3. **Keyframe embeddings** for image-similarity retrieval.
4. **A summary** of the whole thing, shown in the file list.

…without the chat pipeline stalling for the minutes-to-hours that takes.

**Status: 32a landed, 32b–32f PLANNED.** This doc is the design plus the
deploy/verify plan the video work will ship with. What exists today is 32a: the
client pre-check, honest upload errors, and the `vl` models allowed through
passthrough. No chunked transport, no worker, no video mime yet.

**New container — adds a `media-worker` service to `compose.yaml`.** Deploy is
`up -d --build media-worker` plus a rebuild of `audrey-ai` for the route
changes.

## What it does

```text
browser ──chunks──▶ POST /v1/files/upload-sessions            → {upload_id}
        ──chunks──▶ PUT  /v1/files/upload-sessions/{id}/parts/{n}   (≤ 8 MB each)
        ──────────▶ POST /v1/files/upload-sessions/{id}/complete
                        → 202 {file_id, status: "pending"}
                              │
                    media-worker  (new container: ffmpeg + whisper ONLY)
                        │
                        ├─ AUDIO  ffmpeg -vn -ar 16000 -ac 1 → wav
                        │           └─ faster-whisper → [(t_start, t_end, text)]
                        │
                        └─ VISUAL ffmpeg scene-detect → ≤ keyframes_max stills
                                    │
                        every model call goes back through audrey-ai as an
                        ordinary passthrough request, acting-as the uploader:
                          POST /v1/chat/completions
                            model: audrey_passthrough/qwen3-vl:32b   ← frames
                            model: audrey_passthrough/glm-5.2:cloud  ← summary
                              │
                    POST /v1/files/{file_id}/ingest-result  (service token)
                        ├─ transcript chunks    → ingest_user_text_file()
                        ├─ frame descriptions   → ingest_user_text_file()
                        ├─ summary              → ingest_user_text_file()
                        └─ keyframes            → ingest_user_image_file()
                              │
browser polls GET /v1/files → status: pending | processing | ready | failed
```

The KB half stays free. All four artifacts land through
[`ingest_user_text_file`](../../src/audrey/kb/ingest.py) and
[`ingest_user_image_file`](../../src/audrey/kb/ingest.py) exactly as they exist
today — no new collections, no new point schema beyond a `t_start`/`t_end`
payload pair and an `artifact` discriminator. The work is all in the front
half: transport, job lifecycle, and the sidecar.

## Why this exists

Two hard walls, both hit by a single 300 MB upload:

1. **Cloudflare caps request bodies at 100 MB** on Free/Pro/Business. That is a
   plan limit, not a config we own, and `/upload` + `/v1/files/*` route through
   the tunnel (Phase 14 ingress). A 300 MB `POST` is refused at the edge and
   never reaches `audrey-ai` — the user sees a 413 whose body is Cloudflare's
   HTML error page, not our JSON.
2. **`POST /v1/files` does everything inline** — stream, sniff, extract, embed,
   upsert Qdrant, write sqlite, respond. Transcribing and describing an hour of
   footage is minutes of GPU at best. It cannot live inside a request.

## Design decisions

### Visual assessment is VL prose, not CLIP vectors

CLIP embeddings make a frame *findable* by image similarity. They do not tell
you what is in it — you cannot ask "what did the whiteboard say" and have a
vector answer. So the visual pass runs each keyframe through the **`vl` pool**
and stores the resulting prose as ordinary text chunks, embedded and searchable
alongside the transcript.

[`vision.py`](../../src/audrey/pipeline/vision.py) already does exactly this
for chat attachments, and its `DESCRIBE_SYSTEM` prompt is the right one to
reuse: it forbids the model from answering or speculating, and leans hard on
verbatim text capture — which is what slides, screen recordings, and captions
in a video actually need. Keyframes are just images arriving from a different
door.

Keyframe CLIP vectors are still written. The two are complementary, not
alternatives.

### Both transcript AND summary, not either/or

The transcript is the retrieval substrate — it is what makes a specific claim
findable at a specific timestamp. The summary is one extra model call once the
transcript exists, and it earns its keep twice: it gives `GET /v1/files`
something readable to show for a video, and it answers "what is this video"
without pulling 200 chunks into context.

Cheap enough to always do; not a substitute for the transcript.

### The worker reaches models through passthrough — no new model surface

`FairLocalGate` is **in-process to `audrey-ai`**. A worker calling Ollama
directly would not share that gate — it would contend with live chat at the
Ollama level with no fairness whatsoever, and a long ingest would starve the
box.

The fix needs no new endpoint, because `audrey_passthrough/<concrete>` already
exists for exactly this. Its config comment states the intent outright: *"Both
fair-scheduling layers (FairLocalGate, UserInflightRegistry) still fire so
direct-Ollama clients can be brought under Audrey's GPU contention story."*
[`passthrough.py`](../../src/audrey/routes/openai/passthrough.py) wraps every
forward in `inflight.slot(me.email)` and hands `gate` to `passthrough_chat`.

So the worker is an ordinary OpenAI-API client:

```
POST /v1/chat/completions
  { "model": "audrey_passthrough/qwen3-vl:32b",
    "messages": [{ "role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}]}] }
```

Two properties fall out for free:

- **Images route correctly by target.** `_handle_passthrough` calls
  `describe_for_text_model(..., target_model=concrete, ...)`, which rewrites an
  image to prose only for *non-vl* targets. Naming a `vl` model sends the frame
  straight to the vision encoder as Ollama's `images: [...]`.
- **The summary can be a cloud model** — `audrey_passthrough/glm-5.2:cloud` is
  already in `allowed_models` and touches no GPU at all.

**Act as the uploading user**, via the Phase 31 service-token act-as, not as a
distinct service identity. Both fairness layers key on `me.email`, so acting-as
the uploader puts the ingest in *that user's* round-robin slice: a giant video
slows its own owner's chat and leaves everyone else's alone. A shared service
identity would pool all ingests into one slice and blur exactly the distinction
the gate exists to draw.

**The enabling config change has LANDED (32a).** `passthrough.allowed_models`
was text-only, so a describe call would have returned `403 Passthrough not
allowed for model 'qwen3-vl:32b'`. Both members of the `vl` pool —
`qwen3-vl:32b` and its `llava:34b` fallback — are now allowed, so the worker's
model path is unblocked before any worker exists. Side effect accepted
deliberately: `/v1/models` lists one entry per allowed concrete, so both are
now directly targetable by any authenticated OWUI user.

The sidecar still earns its place — it isolates the *CPU* work (demux,
resample, whisper) that would otherwise sit in the request path, while every
*GPU* call goes back through the fairness machinery that already exists.

### Ship it slow, measure it, then make it fast

The `vl` pool is local-only on purpose (`qwen3-vl:32b` primary, `llava:34b`
fallback — the config comment records that unverified cloud entries were
removed after an image got answered blind). With `gpu.concurrency: 1`, every
frame description serializes against every chat turn. `vision.timeout_s` is
120s because "a dense screenshot is a slow decode", and `max_images_per_turn:
4` already treats four images as a lot for a single turn.

Sixty keyframes at ~30s each is half an hour of contended GPU for one upload.
That estimate is a guess, and guessing is the problem — **the first version
ships instrumented and slow, and the optimisation work is driven by what the
numbers actually say.** A conservative default that never gets measured just
hides the cost.

The one thing to hold from the start: `keyframes_max` is a **hard cap**, never
a scene-detection byproduct. Scene detection *ranks* candidates; the cap decides
how many survive. Start at 24 so the first runs finish in a knowable time, and
raise it once the per-frame cost is known.

### The worker does not write sqlite

[`UploadsDB`](../../src/audrey/kb/uploads_db.py) is deliberately a single
connection, no WAL, guarded by an in-process `threading.Lock` — its docstring
spells out the single-writer contract `reconcile_with_qdrant` depends on. A
second container writing that file breaks the invariant the module is built on.

Results come back over `/v1/files/{file_id}/ingest-result` with an
`X-Audrey-Service-Token`, exactly as `custom-tools` already calls home
(Phase 31). The single writer stays single.

### Source bytes are discarded after success

`max_user_bytes` is 1 GiB. At 300 MB a piece, three videos exhaust a user's
whole quota — and nothing downstream ever re-reads the original. The ingest
functions read their inputs once.

So `kb.video.keep_source: false` by default: on success, unlink the video and
charge the quota for the transcript, descriptions, summary, and keyframes.

## Making it efficient

Instrument first. `pipeline_seconds` / `pipeline_total` already exist and feed
`monitoring/`; ingest gets the same treatment with a `stage` label — `demux`,
`stt`, `keyframe_select`, `describe`, `summarise`, `embed`, `upsert` — plus a
per-frame histogram. Without that split, "video ingest is slow" is not an
actionable sentence.

Then the levers, in the order I would try them:

1. **Cut the frame count before spending a decode on it.** Perceptual-hash or
   frame-distance dedup ahead of the describe call. A static talking-head or a
   slide held for two minutes yields many near-identical keyframes, and
   `vision.py`'s cache keys on `sha256` of the data URI — exact bytes — so it
   catches none of them. Pure win, no quality cost. Likely the largest single
   reduction.
2. **Run audio and visual stages concurrently.** Whisper is CPU, frame
   description is GPU-via-passthrough. Serialising them wastes the idle
   resource; overlapping them is close to free wall-clock.
3. **Batch frames per call.** `max_images_per_turn: 4` implies multi-image
   requests work. Either several `image_url` parts per request or a tiled
   contact sheet cuts call count 4–9×. Costs fidelity on small on-screen text,
   so A-B it against single-frame output before adopting.
4. **Move description to a cloud vl model.** Decouples ingest throughput from
   the chat GPU completely, and cloud calls run ~3-way parallel. Highest
   ceiling of anything here — gated on verifying the model genuinely accepts
   images, which is the exact mistake the pool comment warns about.
5. **Downscale frames before sending.** A 4K keyframe carries no more legible
   text than a 1080p one after the encoder resizes it anyway.
6. **Tune the whisper tier.** `small` vs `medium`, and int8 on CPU. Only worth
   doing once stage timings show STT is actually a material share.

Expect the ranking to survive contact with real numbers only partially — that
is what the instrumentation is for.

## What's in scope

- **[`src/audrey/kb/extract.py`](../../src/audrey/kb/extract.py)** — new
  `ALLOWED_VIDEO_MIMES` (mp4, quicktime, webm, x-matroska) folded into
  `ALLOWED_MIMES`, suffixes into `SUFFIX_MIMES`. `ALLOWED_EXTENSIONS` picks
  them up automatically (32a made it derived).
- **[`src/audrey/kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** —
  migration adding `status`, `error`, `duration_s`, `summary`; `kind` gains
  `"video"`. Plus an upload-session table for in-flight chunk assembly.
- **[`src/audrey/routes/files.py`](../../src/audrey/routes/files.py)** — three
  session routes, `/complete` returning 202, the internal `ingest-result`
  callback. `FileRow` grows `status` + `summary`.
- **`src/audrey/metrics.py`** — a `stage` label on the ingest timings so the
  optimisation work has numbers to argue from.
- **[`src/audrey/static/upload.html`](../../src/audrey/static/upload.html)** —
  slice loop with per-part progress, polling for terminal status, summary in
  the file row.
- **`src/audrey/media/`** (new) — ffmpeg invocation, whisper driver, keyframe
  selection, segment→chunk mapping, result callback.
- **`docker/media-worker.Dockerfile`** (new) — `ffmpeg` (not installed in any
  current image) + faster-whisper + baked weights.
- **`compose.yaml`** — the `media-worker` service.
- **`config.yaml`** — a `kb.video` block (`max_upload_mb`, `stt_model`,
  `keyframes_max`, `keep_source`, `chunk_seconds`, `describe_model`,
  `summarise_model`, `frame_dedup_distance`). The
  `passthrough.allowed_models` half is ✅ **done** — see 32a.

## What's NOT in scope

- **No cloudflared ingress change.** Chunked parts are ordinary `PUT`s to
  `/v1/files/*`, already routed by the Phase 14 rule.
- **No raising `max_upload_mb` past 100.** Once chunking exists, per-part size
  is what matters. A single-shot cap above the edge limit is a limit the edge
  silently overrides — the exact failure this phase exists to remove.
- **No cloud vl model added to the `vl` pool.** Efficiency lever 4 — a real
  follow-up, but it needs its own image-capability verification and shouldn't
  ride in on this phase.
- **No re-encoding or proxy transcodes.** We extract, we don't transform.

## Phasing

- **32a — client pre-check, honest errors, vl passthrough.** ✅ **LANDED.**
  Server publishes its real limits on `GET /v1/files`; the page refuses
  oversized, unsupported, over-quota, and empty files before sending bytes, and
  a non-JSON 413 now reads "rejected before it reached Audrey" instead of
  "unknown error". `qwen3-vl:32b` and `llava:34b` added to
  `passthrough.allowed_models` so 32d's model path is open in advance.
- **32b — chunked upload + async job status.** No video yet. Existing types get
  session transport and a `status` column. Proves the hardest part where the
  failure modes are cheap.
- **32c — audio → transcript.** ffmpeg + whisper in the sidecar, transcript
  through the existing text path. First real user-facing win.
- **32d — visual assessment.** Keyframes → `audrey_passthrough/qwen3-vl:32b` →
  prose chunks + CLIP vectors. Ships **on**, instrumented, with `keyframes_max`
  low. The point of this slice is to produce the first real timings.
- **32e — summary.** One call over 32c + 32d output; surfaces in the file list.
- **32f — optimise.** Driven by 32d's numbers, working the lever list above.
  Scoped once there is something to scope it against.

## The parts that will bite

- **Near-duplicate frames** are the likely top cost. A static scene yields many
  near-identical keyframes, each paying a full vl decode. `vision.py`'s cache
  keys on `sha256` of the data URI — exact bytes — so it catches none of them.
  The filter has to sit *before* the describe call. See efficiency lever 1.
- **A long ingest will visibly slow its owner's chat.** That is the fair
  outcome (round-robin keys on `me.email`, and the ingest acts-as the
  uploader), not a bug — but it should be a known, measured number rather than
  a surprise. Watch `gate.snapshot()` during the first real run.
- **Orphaned partials.** A closed tab leaves half a video in the session dir.
  Needs a TTL sweep, in the spirit of the boot-time `reconcile_with_qdrant`.
- **Stuck jobs.** A worker crash mid-transcription leaves `status=processing`
  forever. Needs a lease with a timeout that flips the row to `failed`.
- **Late rejection is worse with chunking.** A user could push 300 MB across 40
  requests before we notice the codec is unreadable. **Probe the first part's
  container header at session start** and reject there.
- **Silent or music-only video** yields an empty transcript — but may still
  have rich visual content. Do NOT reuse `EmptyExtractionError` as a hard
  failure for video; an empty audio track with usable frames is a success.
- **The sniff gate must still hold.** Adding video mimes widens
  `ALLOWED_MIMES`; libmagic stays the gate, extension stays a hint.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build media-worker audrey-ai
```

Both containers have no `curl`; probes run inside via `python3`.

## Verification (to be written against the built phase)

**0. ffmpeg + weights present.**

```
docker exec media-worker sh -c 'command -v ffmpeg || echo MISSING'
docker exec media-worker python3 -c "import faster_whisper; print(faster_whisper.__version__)"
```

**1. Chunked transport round-trips a file that already worked.** A ~5 MB PDF
through the session routes must land with the same chunk count as single-shot.

**2. The worker makes no direct Ollama calls.** Its only outbound target is
`audrey-ai`'s `/v1/chat/completions` via `audrey_passthrough/…`; it holds no
`OLLAMA_HOST` and is not on a network path that reaches Ollama directly. This
is the fairness invariant — if it regresses, the gate stops meaning anything.

**2b. The vl model is reachable through passthrough.** Without the
`allowed_models` entry the describe call 403s:

```
docker exec audrey-ai python3 -c "import json,urllib.request; \
  print([m['id'] for m in json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models'))['data'] \
  if 'qwen3-vl' in m['id']])"
```

**2c. Stage timings are being recorded.** `/metrics` must show the ingest
stages (`demux`, `stt`, `describe`, `summarise`, `embed`, `upsert`) after one
run. If they aren't there, 32f has nothing to work from and the optimisation
pass is guesswork.

**3. Gate fairness under load.** Start a video ingest with `describe_frames`
on, then hold a normal chat conversation. Chat latency must stay within one
frame-decode; check `gate.snapshot()` shows interleaving, not a monopoly.

**4. Quota accounting after `keep_source: false`.** Confirm `GET /v1/files`
charges the derived artifacts and the source is gone from `/data/uploads`.

**5. User isolation.** A second user must not see the video, transcript,
descriptions, summary, or keyframes. The new internal routes are fresh surface
and need their own 401 tests.

**6. Crash recovery.** `docker kill media-worker` mid-job; the row must reach
`failed` via lease timeout rather than sitting in `processing`.

### Rollback

`docker compose stop media-worker` and revert `audrey-ai`. Uploads of existing
types keep working through the single-shot route, which 32b leaves in place.

## What this unblocks

Video stops being a dead end that fails with an opaque 413 after a long wait,
and becomes askable: "what did they say about pricing" hits the transcript,
"what was on the slide at that point" hits the frame descriptions, and "what is
this video" hits the summary. The sidecar is also the natural home for future
heavy media work — audio-only files, and OCR for the scanned PDFs that
`EmptyExtractionError` currently turns away — without any of it entering the
request path.
