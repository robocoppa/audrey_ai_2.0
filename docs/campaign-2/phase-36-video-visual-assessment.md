# Campaign 2 Phase 36 — visual assessment (keyframes through the vl pool)

What is actually on screen, in words. Keyframes come out of the video, through
the [keyframe gate](phase-38-video-optimise.md), into
`audrey_passthrough/qwen3-vl:32b`, and the resulting prose is ingested as
ordinary searchable text alongside the transcript.

**Status: IN PROGRESS.** The CPU half is built and tested —
[`media/frames.py`](../../src/audrey/media/frames.py) samples and thins, and
[`routes/media.py`](../../src/audrey/routes/media.py) is the model door. What
remains is the worker stage that joins them, and ingesting the result.

The model path was already proved by hand against the deployment — frames
extracted from a real upload came back correctly described. What is unproved
is doing it unattended, at volume, without starving chat.

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
- **`src/audrey/media/worker.py`** — the frame stage in the claim loop.
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

## Deploy on Unraid

```
docker compose up -d --build media-worker audrey-ai
```

## Verification (to be written against the built phase)

**1. The worker still makes no direct Ollama calls.** Re-run the phase 34
network probe. This is the phase that gives it a reason to, so it is the phase
where the invariant matters.

**2. Stage timings are recorded.** `/metrics` shows every ingest stage after one
run. If they are not there, phase 38 has nothing to work from.

**3. The gate is actually thinning frames.** `keyframe_select` reports both
candidate and kept counts; a run where they are equal on static footage means
the gate is not wired in.

**4. Descriptions are attributed to timestamps** and searchable by the
uploading user, and by no one else.

**5. Gate fairness under load.** Start a video ingest, then hold a normal chat
conversation. Chat latency must stay within one frame-decode; `gate.snapshot()`
should show interleaving, not a monopoly.

**6. A video with no decodable video stream fails cleanly** rather than
producing zero descriptions and reporting success.

### Rollback

Revert `media-worker` to phase 35. Transcripts keep working; videos lose their
visual half and stay searchable by speech.

## What this unblocks

"What was on the slide at that point" becomes answerable, and phase 38 finally
has numbers instead of estimates. [Phase
37](phase-37-video-summary.md) puts a readable face on all of it.
