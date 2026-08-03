# Campaign 2 Phase 35 — audio to transcript (whisper in the sidecar)

The first slice a user can see. [Phase 34](phase-34-media-worker-container.md)'s
worker extracts audio and reports a duration; this one turns that audio into a
timestamped transcript, ingests it through the existing text path, and flips the
row to `ready`. The video becomes searchable.

**Status: PLANNED.**

---

## Design decisions

### The transcript is the retrieval substrate

It is what makes a specific claim findable at a specific timestamp. Everything
later in this phase family — the visual descriptions, the summary — is
additional detail hung off a video that is already searchable. If only one of
the four artifacts ever shipped, this is the one worth having.

### Segments become chunks, with timestamps preserved

faster-whisper returns `(t_start, t_end, text)`. Those go through
[`ingest_user_text_file`](../../src/audrey/kb/ingest.py) exactly as any text
file does — no new collection, no new point schema beyond a `t_start`/`t_end`
payload pair and an `artifact` discriminator.

The KB half of this work is nearly free, and that is by design. The expensive
decision was made in phase 32: put the artifacts through the paths that already
exist rather than building a parallel video-shaped one.

### An empty transcript is a success

A silent or music-only video yields no text. That is not a failure — the file
may still have rich visual content that
[phase 36](phase-36-video-visual-assessment.md) will find.

Specifically: do **not** reuse `EmptyExtractionError` as a hard failure for
video. That error exists to turn away a scanned PDF with no text layer, where
empty means "we cannot use this". For video, empty audio is a fact about the
file, not a defect in it.

### Source bytes are discarded after success

`max_user_bytes` is 1 GiB. At 300 MB a piece, three videos exhaust a user's
whole quota — and nothing downstream ever re-reads the original. The ingest
functions read their inputs once.

So `kb.video.keep_source: false` by default: on success, unlink the video and
charge the quota for the transcript instead. On failure the source stays, or
the retry in phase 33 would have nothing to retry against.

## What's in scope

- **`docker/media-worker.Dockerfile`** — faster-whisper plus **baked weights**.
  Downloading a model on first job turns a cold start into a silent multi-minute
  stall inside a lease window.
- **`src/audrey/media/stt.py`** (new) — the whisper driver and segment mapping.
- **`src/audrey/media/worker.py`** — the transcript stage added to the claim
  loop, posted through `ingest-result`.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — `ingest-result`
  learns to accept transcript segments and route them to the text ingest path.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** — `duration_s` on
  the row, so the file list can show something true about a video.
- **`config.yaml`** — `kb.video.stt_model`, `chunk_seconds`, `keep_source`.

## What's NOT in scope

- **No visual assessment.** [Phase 36](phase-36-video-visual-assessment.md).
- **No summary.** [Phase 37](phase-37-video-summary.md).
- **No speaker diarisation.** Worth wanting, not worth blocking a first
  transcript on.
- **No translation.** Transcribe in the source language.

## The parts that will bite

- **Baked weights make the image large.** Accepted deliberately — the
  alternative is a cold start inside a lease window, which reads as a stuck job.
- **Whisper tier is a real cost lever** (`small` vs `medium`, int8 on CPU), but
  tuning it before [phase 38](phase-38-video-optimise.md) has stage
  timings is guesswork. Pick a tier, record it, move on.
- **Quota accounting flips mid-phase.** Once `keep_source: false` lands, a
  user's stored bytes *drop* when a video finishes. The file list has to not
  look broken while that happens.
- **A partial transcript is worse than none.** If whisper dies halfway, the job
  must fail rather than ingest half a video and report `ready` — a half
  transcript is silently wrong in exactly the way that never gets noticed.
- **Long videos exceed one result post.** An hour of speech is a lot of JSON.
  Either the result post streams or the worker posts in batches; decide before
  the first two-hour file rather than after.

## Deploy on Unraid

From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build media-worker audrey-ai
```

Both: the worker gains whisper, `audrey-ai` gains the transcript half of
`ingest-result`.

## Verification (to be written against the built phase)

**0. Whisper and its weights are present, without network.**

```
docker exec media-worker python3 -c "import faster_whisper; print(faster_whisper.__version__)"
```

**1. A short spoken video produces a transcript** whose text matches what was
said, with timestamps that line up against the source.

**2. The row reaches `ready`** and the transcript is searchable through an
ordinary KB query by the uploading user.

**3. A second user cannot retrieve it.** Fresh artifacts in a per-user
collection; isolation needs its own test rather than an assumption.

**4. A silent video succeeds** with an empty transcript and a `ready` row —
not a `failed` one.

**5. Quota accounting after `keep_source: false`.** `GET /v1/files` charges the
transcript and the source is gone from the uploads volume.

**6. A killed worker mid-transcription** leaves no partial transcript in Qdrant
and returns the job to `pending`.

### Rollback

Revert `media-worker` to the phase 34 image and `audrey-ai` to phase 33.
Videos go back to sitting at `pending`; nothing already ingested is lost.

## What this unblocks

A video is now searchable by what was said in it. [Phase
36](phase-36-video-visual-assessment.md) makes it searchable by what was *shown* in
it, which is the half a transcript cannot reach.
