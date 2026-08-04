# Campaign 2 Phase 36 — visual assessment (keyframes through the vl pool)

What is actually on screen, in words. Keyframes come out of the video, through
the [keyframe gate](phase-38-video-optimise.md), into
`audrey_passthrough/qwen3-vl:32b`, and the resulting prose is ingested as
ordinary searchable text alongside the transcript.

**Status: BUILT, NOT YET DEPLOYED.** The whole path exists and is tested:
sample → thin → describe → ingest, with descriptions landing beside the
transcript in the same per-user collection under an `artifact: "visual"`
discriminator.

The model path was already proved by hand against the deployment — frames
extracted from a real upload came back correctly described. What is unproved
is doing it unattended, at volume, without starving chat. That is what the
verification below is for, and it needs the box.

---

## Design decisions

### Visual assessment is VL prose, not CLIP vectors

CLIP embeddings make a frame *findable* by image similarity. They do not tell
you what is in it — you cannot ask "what did the whiteboard say" and have a
vector answer. So the visual pass runs each keyframe through the **`vl` pool**
and stores the resulting prose as ordinary text chunks, embedded and searchable
alongside the transcript.

[`vision.py`](../../src/audrey/pipeline/vision.py) already does exactly this for
chat attachments, and its `DESCRIBE_SYSTEM` prompt is the right one to reuse: it
forbids the model from answering or speculating, and leans hard on verbatim text
capture — which is what slides, screen recordings and captions actually need.
Keyframes are just images arriving from a different door.

Keyframe CLIP vectors are still written. The two are complementary, not
alternatives.

### The worker reaches models through a narrow service route (corrected)

`FairLocalGate` is **in-process to `audrey-ai`**. A worker calling Ollama
directly would not share that gate — it would contend with live chat at the
Ollama level with no fairness whatsoever, and a long ingest would starve the
box. So the worker must reach the model *through Audrey*. That part stands.

**The transport this plan named does not exist.** It said the worker would
`POST /v1/chat/completions` with `audrey_passthrough/qwen3-vl:32b`, "acting as
the uploading user via the Phase 31 service-token act-as". That route depends
on `require_user`, which demands a real OWUI bearer. Phase 31's act-as is
`resolve_kb_caller` and it lives only on the KB *query* routes. The worker
holds `KB_SERVICE_TOKEN` and cannot obtain a user JWT, so it would have got a
401 on the first frame.

The obvious repair — teach `require_user` to accept a service token plus an
act-as header — was rejected. `/v1/chat/completions` is the endpoint every
OWUI user hits, and putting the entire chat surface behind a header that
grants any identity, to close a gap for one background client, is a bad trade
at any size.

Instead: **[`routes/media.py`](../../src/audrey/routes/media.py), service-token
only, one verb.** `POST /v1/media/describe` takes an image and a user, and
returns prose. No message history, no streaming, no model selection — nothing
that would make it a second chat endpoint.

**The fairness property is fully preserved**, which was the only reason the
identity mattered. The route passes the uploader's email to both
`inflight.slot(...)` and `gate.acquire(..., user_id=...)`, so a giant video
slows its own owner's chat and leaves everyone else's alone. A shared service
identity would pool every ingest into one slice and blur exactly the
distinction the gate exists to draw.

Two properties the plan expected from passthrough, and where they went:

- **Images route correctly by target.** Moot — this route always calls the
  `vl` pool directly, so there is no text-target rewrite to get right.
- **The image must be a `data:` URI.**
  [`ollama.py:29-43`](../../src/audrey/models/ollama.py#L29-L43) silently drops
  `http(s)://` URLs, which yields a confidently blind answer rather than an
  error — verified the hard way during phase 32's manual testing. Now
  structurally impossible: the route takes raw base64 and builds the `data:`
  URI itself, so there is no field a caller could put a URL in.

### Ship it slow, measure it, then make it fast

The `vl` pool is local-only on purpose (`qwen3-vl:32b` primary, `llava:34b`
fallback — the config comment records that unverified cloud entries were removed
after an image got answered blind). With `gpu.concurrency: 1`, every frame
description serialises against every chat turn. `vision.timeout_s` is 120s
because "a dense screenshot is a slow decode", and `max_images_per_turn: 4`
already treats four images as a lot for a single turn.

**The first version ships on and instrumented, not behind a flag defaulting
off.** A conservative default that never gets measured just hides the cost, and
[phase 38](phase-38-video-optimise.md) has nothing to work from. The
point of this slice is to produce the first real per-frame timings.

The one thing to hold from the start: `keyframes_max` is a **hard cap**, never a
scene-detection byproduct. Scene detection and the gate *rank and thin*
candidates; the cap decides how many survive. Start at 24 so the first runs
finish in a knowable time, and raise it once the per-frame cost is known.

### The gate runs before the describe call, not after

[`media/framegate.py`](../../src/audrey/media/framegate.py) landed early, in
phase 32, and finally gets a caller here. It must sit *before* the model call:
`vision.py`'s description cache keys on the `sha256` of the data URI — exact
bytes — so two frames from a locked-off camera differing only in sensor noise
hash differently and both pay full price. The cache cannot catch this class of
redundancy; only the gate can.

Measured on real footage: 6 of 19 frames kept, 68% fewer describe calls, with
the static stretch collapsing and the head and tail preserved.

## What's in scope

- **[`src/audrey/media/frames.py`](../../src/audrey/media/frames.py)** (new,
  **done**) — ffmpeg sampling and the thinning that feeds
  [`framegate.py`](../../src/audrey/media/framegate.py). Sampling is by time
  rather than `-skip_frame nokey`, because I-frame spacing is a function of
  the encoder's GOP settings and bitrate ladder, not of the content.
- **[`src/audrey/routes/media.py`](../../src/audrey/routes/media.py)** (new,
  **done**) — the model door, replacing the `describe.py` passthrough client
  this plan assumed. See the corrected decision above.
- **`Probe.has_video`** (**done**) — an audio-only file yields zero frames
  rather than failing, mirroring phase 35's treatment of a silent video. This
  refines verification step 6 below: a *corrupt* video stream must fail, a
  genuinely absent one must not, or every podcast upload becomes a failed job.
- **`docker/media-worker.Dockerfile`** (**done**) — Pillow, which
  `framegate.dhash` needs. The Dockerfile had carried a note predicting this
  exact trap since phase 34; the failure mode is an ImportError at *claim*
  time on the box, not at build time, because `framegate` imports PIL inside
  its functions.
- **[`src/audrey/media/describe.py`](../../src/audrey/media/describe.py)**
  (new, **done**) — the worker's side of the describe call, one frame at a
  time, with the budget.
- **`src/audrey/media/worker.py`** (**done**) — the frame stage in the claim
  loop, between transcription and the result post.
- **[`kb/ingest.py`](../../src/audrey/kb/ingest.py)** (**done**) —
  `ingest_frame_descriptions`, chunked per description and marked
  `artifact: "visual"`.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** (**done**) —
  `ingest-result` accepts `frames`, and the claim now carries `FrameSettings`
  and `lease_seconds`.

### Budgets have to be checked against the lease, not against each other

`TRANSCRIBE_BUDGET_S` is 1440s and `FRAME_BUDGET_S` is 900s. That is 39
minutes against a 30-minute lease — so a long video would be swept
mid-describe, re-claimed, and burn its attempts doing exactly the same thing
every time. Two stage budgets that are each individually reasonable are not
jointly reasonable, and nothing in either one can see the other.

The lease is the only authority, so it travels with the job:
`JobClaim.lease_seconds`. The worker treats `FRAME_BUDGET_S` as a **ceiling**
and cuts it to whatever the lease has left, holding back
`LEASE_RESERVE_S` = 120s so the result post has somewhere to land. A lease
already spent yields a budget of 0.0, which describes nothing and still posts
the transcript — the video comes back searchable by speech rather than not at
all.

### A partial visual pass is fine; a partial transcript is not

Phase 35 holds that a truncated transcript is worse than none, because
nothing in the artifact says where it stopped: it is silently wrong in the way
that never gets noticed.

Frame descriptions are the opposite shape. Each is independently timestamped
and stands alone, so a set that stops early is *correct about what it covers*
and merely incomplete. `frames_planned` travels with the result so the gap is
recorded rather than inferred from a count that looks complete.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — `ingest-result`
  learns frame descriptions and keyframe images.
- **`src/audrey/metrics.py`** — a `stage` label on the ingest timings
  (`demux`, `stt`, `keyframe_select`, `describe`, `embed`, `upsert`) plus a
  per-frame histogram. Without that split, "video ingest is slow" is not an
  actionable sentence, and phase 38 is guesswork.
- **`config.yaml`** — `kb.video.keyframes_max`, `describe_model`,
  `frame_dedup_distance`.

## What's NOT in scope

- **No summary.** [Phase 37](phase-37-video-summary.md).
- **No batching frames per call.** A real lever, but it costs fidelity on small
  on-screen text and belongs in phase 38 where it can be A-B'd against
  single-frame output that actually exists.
- **No cloud vl model.** It needs its own image-capability verification and
  should not ride in on this phase — that is the exact mistake the pool comment
  warns about.

## The parts that will bite

- **A long ingest will visibly slow its owner's chat.** That is the fair
  outcome, not a bug — but it should be a known, measured number rather than a
  surprise. Watch `gate.snapshot()` during the first real run.
- **Frame size matters more than expected.** A 4K keyframe carries no more
  legible text than a 1080p one after the encoder resizes it anyway, and costs
  decode time both ways.
- **The gate's threshold is a starting point, not a finding.** 8 bits was
  calibrated against one video. Re-check that the head and tail survive whenever
  it is retuned.
- **dHash cannot see a cut between two flat colour fields.** It asks "is this
  pixel brighter than the one to its right", so a solid colour has zero
  gradient everywhere and red, blue and white all hash identically. Found by
  building a test fixture out of colour cards and watching three scenes merge
  into one. Pinned as a test rather than fixed: real footage has texture, and
  the alternatives (average hash, colour histograms) trade this rare case for
  sensitivity to exposure and colour-grade drift, which is the common one.
  Worth knowing before someone else tests with solid colours and concludes the
  gate is broken.
- **`keyframes_max` interacts with the gate.** The gate may return fewer than
  the cap, which is the good case. It must never return *more*.
- **A clip shorter than the sample interval yields no frames at all.** *(hit
  during the build.)* `fps=1/30` on a 10-second upload emits nothing — the
  filter never reaches its first output time — so the job failed with "the
  video stream is present but did not decode", which is both wrong and fatal.
  `extract_frames` falls back to taking a single frame, which also covers a
  source whose duration ffmpeg does not report (routine in matroska), so
  deciding up front from the probe would have been its own bug.
- **The two artifacts share a `file_id` and can delete each other.**
  `delete_by_file_id` removes *every* point for a file, so the transcript
  ingest clearing on its own way in would take out frame descriptions written
  moments earlier — order-dependently. The route now clears once and both
  write into the space it made. For the same reason the frames get their own
  sidecar name: `point_id` is `(source, kind, chunk_idx)`, so a shared source
  would make frame chunk 0 and transcript chunk 0 *the same point*.
- **A budget of zero must not read as "no budget".** `if budget_s and ...` is
  the natural way to write the check and is wrong: 0.0 is falsy, so a lease
  with nothing left would skip the check and describe every frame — the exact
  failure the 0.0 was computed to prevent.

## Deploy on Unraid

```
docker compose up -d --build media-worker audrey-ai
```

## Verification

**0. Pillow is in the worker image.** The one that fails at *claim* time
rather than at build, because `framegate` imports PIL inside its functions.

```
docker exec media-worker python3 -c "import PIL; print(PIL.__version__)"
```

**1. The worker still makes no direct Ollama calls.** Re-run the phase 34
network probe. This is the phase that gives it a reason to, so it is the phase
where the invariant matters.

```
docker exec media-worker python3 -c "
import socket; socket.gethostbyname('ollama')"           # must fail
docker exec media-worker python3 -c "
import socket; socket.create_connection(('192.168.1.11', 11434), timeout=3)"
```

Both must fail. The second is the one that matters — a separate bridge network
scopes DNS, not routes, and `internal: true` is what removes the route to the
host's published port.

**2. Per-frame timings are recorded.** `curl -s localhost:8000/metrics | grep
video_describe`. Without a per-frame number, "video ingest is slow" is not an
actionable sentence and [phase 38](phase-38-video-optimise.md) is guesswork.

**3. The gate is actually thinning frames.** The worker logs `keyframes N of M
sampled frames`. On real footage N must be well under M — equality means the
gate is not wired in, and the phase-32 measurement was 6 of 19.

**4. Descriptions are attributed to timestamps** and searchable by the
uploading user, and by no one else. A `kb/query` hit should carry
`artifact: "visual"` with a `t_start` that matches what is on screen at that
point in the video.

**5. Gate fairness under load.** Start a video ingest, then hold a normal chat
conversation. Chat latency must stay within one frame-decode; the ingest runs
in the *uploader's* slice, so this is the test that the `user` field on
`/v1/media/describe` is doing its job.

**6. A source with no *decodable* video stream fails cleanly** — but an
audio-only file must **not**. The distinction is the point: a podcast upload
has no video stream and must still complete with its transcript, while a
corrupt stream must fail rather than reporting success with zero descriptions.
A short clip is the third case and must yield one frame, not zero.

**7. A described video survives a requeue.** Both artifacts live under one
`file_id`, and the delete-before-upsert that keeps a re-run from doubling them
is the same call that could wipe one of them. Requeue a described video and
confirm both come back.

### Rollback

Revert `media-worker` to phase 35. Transcripts keep working; videos lose their
visual half and stay searchable by speech.

## What this unblocks

"What was on the slide at that point" becomes answerable, and phase 38 finally
has numbers instead of estimates. [Phase
37](phase-37-video-summary.md) puts a readable face on all of it.
