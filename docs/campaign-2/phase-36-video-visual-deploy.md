# Campaign 2 Phase 36 — visual assessment (keyframes through the vl pool)

What is actually on screen, in words. Keyframes come out of the video, through
the [keyframe gate](phase-38-video-optimise-deploy.md), into
`audrey_passthrough/qwen3-vl:32b`, and the resulting prose is ingested as
ordinary searchable text alongside the transcript.

**Status: PLANNED.** The model path is already proved by hand against the
deployment — frames extracted from a real upload came back correctly described.
What is unproved is doing it unattended, at volume, without starving chat.

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

### The worker reaches models through passthrough — no new model surface

`FairLocalGate` is **in-process to `audrey-ai`**. A worker calling Ollama
directly would not share that gate — it would contend with live chat at the
Ollama level with no fairness whatsoever, and a long ingest would starve the
box.

This needs no new endpoint, because `audrey_passthrough/<concrete>` already
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
- **The image must be a `data:` URI.**
  [`ollama.py:29-43`](../../src/audrey/models/ollama.py#L29-L43) silently drops
  `http(s)://` URLs, which yields a confidently blind answer rather than an
  error. Verified the hard way during phase 32's manual testing.

**Act as the uploading user**, via the Phase 31 service-token act-as, not as a
distinct service identity. Both fairness layers key on `me.email`, so acting-as
the uploader puts the ingest in *that user's* round-robin slice: a giant video
slows its own owner's chat and leaves everyone else's alone. A shared service
identity would pool all ingests into one slice and blur exactly the distinction
the gate exists to draw.

The `allowed_models` half of this landed in
[phase 32](phase-32-video-ingest-deploy.md) — both `vl` members are already
permitted, so the model path is open before this phase starts.

### Ship it slow, measure it, then make it fast

The `vl` pool is local-only on purpose (`qwen3-vl:32b` primary, `llava:34b`
fallback — the config comment records that unverified cloud entries were removed
after an image got answered blind). With `gpu.concurrency: 1`, every frame
description serialises against every chat turn. `vision.timeout_s` is 120s
because "a dense screenshot is a slow decode", and `max_images_per_turn: 4`
already treats four images as a lot for a single turn.

**The first version ships on and instrumented, not behind a flag defaulting
off.** A conservative default that never gets measured just hides the cost, and
[phase 38](phase-38-video-optimise-deploy.md) has nothing to work from. The
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

- **`src/audrey/media/frames.py`** (new) — ffmpeg frame extraction and scene
  ranking, feeding [`framegate.py`](../../src/audrey/media/framegate.py).
- **`src/audrey/media/describe.py`** (new) — the passthrough client, acting-as
  the uploader, one frame per call to start.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — `ingest-result`
  learns frame descriptions and keyframe images.
- **`src/audrey/metrics.py`** — a `stage` label on the ingest timings
  (`demux`, `stt`, `keyframe_select`, `describe`, `embed`, `upsert`) plus a
  per-frame histogram. Without that split, "video ingest is slow" is not an
  actionable sentence, and phase 38 is guesswork.
- **`config.yaml`** — `kb.video.keyframes_max`, `describe_model`,
  `frame_dedup_distance`.

## What's NOT in scope

- **No summary.** [Phase 37](phase-37-video-summary-deploy.md).
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
37](phase-37-video-summary-deploy.md) puts a readable face on all of it.
