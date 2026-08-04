# Campaign 2 Phase 35 — audio to transcript (whisper in the sidecar)

The first slice a user can see. [Phase 34](phase-34-media-worker-container.md)'s
worker extracts audio and reports a duration; this one turns that audio into a
timestamped transcript, ingests it through the existing text path, and flips the
row to `ready`. The video becomes searchable.

**Status: BUILT.** Hermetic tests pass; unverified on the box — this is the
first phase whose output quality cannot be checked by a test, only by reading a
transcript of something you know the contents of.

**Scope cut: `keep_source: false` is deferred to
[phase 38](phase-38-video-optimise.md).** The plan below called for unlinking
the video after a successful transcript, to stop 300 MB files eating a 1 GiB
quota. That is incompatible with everything that comes after it: phase 36
extracts *frames from the source*, phase 37 summarises both artifacts, and
phase 33's `requeue` route — added after this plan was written — points a
re-run at a path that would no longer exist. Deleting the bytes is only safe
once nothing downstream needs them, which is after 36 has taken its frames.
The quota pressure is real and the phase 38 doc now owns it.

---

## Design decisions

### The transcript is the retrieval substrate

It is what makes a specific claim findable at a specific timestamp. Everything
later in this phase family — the visual descriptions, the summary — is
additional detail hung off a video that is already searchable. If only one of
the four artifacts ever shipped, this is the one worth having.

### Segments become chunks, with timestamps preserved

faster-whisper returns `(t_start, t_end, text)`. These go through
[`ingest_transcript_segments`](../../src/audrey/kb/ingest.py) into the same
per-user collection as any other upload — no new collection, and the only new
payload fields are `t_start`, `t_end` and an `artifact` discriminator.

**Chunk size stays at 250, deliberately.** After the measurements in
[phase 39](phase-39-hybrid-retrieval.md), dropping to ~120 tokens would make
exact quoting mostly work — and would make broad questions worse, because a
one-sentence chunk cannot hold several people's answers. There is no chunk size
good at both, so tuning toward a compromise about to be deleted was rejected.
Hybrid retrieval removes the trade-off; expect this number to go *up* once it
lands.

**The first build reused `ingest_user_text_file` and that was wrong.** It
joined the segments into one `[HH:MM:SS] line` document and let `chunk_text`
split it at the 1000-token document default, on the reasoning that reusing the
existing path was free. It is free in code and expensive in retrieval, and the
first real video showed both costs: 1000-token chunks are three-plus minutes of
speech spanning several speakers, and the `[HH:MM:SS] ` prefixes were ~1,700 of
7,318 characters — 23% of every embedding spent on strings with no meaning.

A 25-word verbatim quote scored **0.586** against its own chunk. An exact quote
that a search cannot find is the worst failure a retrieval substrate can have.

So transcripts get their own path: 250-token chunks grouped on segment
boundaries (whisper already split on natural pauses, which are better cut
points than a token count), and timestamps in the payload rather than the text.
The sidecar keeps its `[HH:MM:SS]` lines — it is the human-readable artifact
and the identity anchor for `delete_by_file_id` — but its contents are not what
gets embedded.

The wider principle from phase 32 still holds: put artifacts through paths that
already exist. The correction is that "reuse the path" is not the same as
"reuse its defaults", and the difference is only visible if you measure.

### An empty transcript is a success

A silent or music-only video yields no text. That is not a failure — the file
may still have rich visual content that
[phase 36](phase-36-video-visual-assessment.md) will find.

Specifically: do **not** reuse `EmptyExtractionError` as a hard failure for
video. That error exists to turn away a scanned PDF with no text layer, where
empty means "we cannot use this". For video, empty audio is a fact about the
file, not a defect in it.

### Source bytes are NOT discarded here (changed)

The original plan unlinked the video on success, on the grounds that nothing
downstream re-reads it. That was wrong, and the error is worth keeping visible
because it is the kind that only shows up two phases later.

Phase 36 extracts keyframes *from the source video*. Phase 37 summarises what
36 produced. And `requeue` — which did not exist when this was planned — puts a
row back to `pending` for a worker that would then find no file. Unlinking
after transcription would have left the pipeline able to run exactly once,
with no way to re-run it and no way to add visual data to anything already
processed.

The quota problem is real: at 300 MB a piece, three videos exhaust a 1 GiB
allowance. It now belongs to [phase 38](phase-38-video-optimise.md), which is
the first point where nothing else needs the bytes.

## What's in scope

- **`docker/media-worker.Dockerfile`** — faster-whisper plus **baked weights**.
  Downloading a model on first job turns a cold start into a silent multi-minute
  stall inside a lease window.
- **[`src/audrey/media/stt.py`](../../src/audrey/media/stt.py)** (new) — the
  whisper driver, the transcription budget, and the repetition collapse. The
  `faster_whisper` import is lazy so the module stays importable (and testable)
  outside the worker image.
- **`src/audrey/media/worker.py`** — the transcript stage added to the claim
  loop, posted through `ingest-result`.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — `ingest-result`
  learns to accept transcript segments and route them to the text ingest path.
- **`compose.yaml`** — `WHISPER_MODEL` and `TRANSCRIBE_BUDGET_S` on the
  worker. Not `config.yaml`: the worker reads env, per the Phase 34 decision.
  `WHISPER_MODEL` can only *select* a model baked into the image at build time
  (`WHISPER_BAKE`), because the worker has no network to fetch another.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** — `duration_s`,
  the sixth additive column, so the file list can say how long a video is.

## What's NOT in scope

- **No visual assessment.** [Phase 36](phase-36-video-visual-assessment.md).
- **No summary.** [Phase 37](phase-37-video-summary.md).
- **No speaker diarisation.** Worth wanting, not worth blocking a first
  transcript on.
- **No translation.** Transcribe in the source language.

## The parts that will bite

- **faster-whisper needs `requests` and does not declare it.** *(hit on the
  first build.)* Its `utils.py` does `import requests`, which used to arrive
  transitively through `huggingface_hub` — that package has since moved off it.
  So `pip install faster-whisper` succeeds and `import faster_whisper` raises
  `ModuleNotFoundError: No module named 'requests'`, at build time, naming
  neither package involved. The pip line installs it explicitly; **do not
  remove it as redundant**, which is exactly what it looks like.
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

## Verification

**0. Whisper and its weights are present, without network.** The second command
is the one that matters — `local_files_only=True` means a missing bake raises
immediately instead of hanging on a download that can never succeed.

```
docker exec media-worker python3 -c "import faster_whisper; print(faster_whisper.__version__)"
docker exec media-worker du -sh /opt/whisper
docker exec media-worker python3 -c "
from audrey.media.stt import load_model; load_model('small'); print('weights load OK')"
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
