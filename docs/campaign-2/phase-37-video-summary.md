# Campaign 2 Phase 37 — video summary (one call, shown in the file list)

One model call over the transcript and frame descriptions that
[phase 35](phase-35-video-transcript.md) and
[phase 36](phase-36-video-visual-assessment.md) produced, stored on the row and
shown in `GET /v1/files`.

**Status: PLANNED.** The smallest phase of the video work, and the one that
makes the rest legible.

---

## Design decisions

### Both transcript AND summary, not either/or

The transcript is the retrieval substrate — it is what makes a specific claim
findable at a specific timestamp. The summary is one extra model call once the
transcript exists, and it earns its keep twice: it gives `GET /v1/files`
something readable to show for a video, and it answers "what is this video"
without pulling 200 chunks into context.

Cheap enough to always do; not a substitute for the transcript. A file list that
shows `jasonRetirement.mp4 · 288 MB · ready` tells you nothing you did not
already know.

### The summary can be a cloud model

`audrey_passthrough/glm-5.2:cloud` is already in `allowed_models` and touches no
GPU at all. Summarising is a text task over text inputs, so unlike
[phase 36](phase-36-video-visual-assessment.md)'s frame descriptions it carries no
image-capability risk — the failure mode that made the `vl` pool local-only does
not apply here.

That makes this the one stage of video ingest that costs the box nothing, and
it should stay that way.

### The summary is also ingested, not only stored

Stored on the row it answers "what is this video" in the UI. Ingested as a text
chunk it also answers that question in chat, without retrieving the whole
transcript. One extra chunk per video is a rounding error against a transcript.

## What's in scope

- **`src/audrey/media/summarise.py`** (new) — one passthrough call over the
  transcript and descriptions, acting-as the uploader as in phase 36.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** — `summary` on the
  row.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** — `ingest-result`
  accepts the summary; `FileRow` grows `summary`.
- **[`static/upload.html`](../../src/audrey/static/upload.html)** — show it in
  the file row.
- **`config.yaml`** — `kb.video.summarise_model`.

## What's NOT in scope

- **No summarisation of non-video uploads.** A PDF summary is a reasonable want
  and a different phase; nothing here should be video-shaped by accident.
- **No chaptering or section breaks.** One summary per video.

## The parts that will bite

- **A two-hour transcript will not fit one context.** The summariser needs a
  reduce step, or a cap on what it reads, decided before the first long file
  rather than discovered by a truncation.
- **An empty transcript still deserves a summary** — from the frame
  descriptions alone. A silent video is a normal case (phase 35), and the file
  list should not go blank for it.
- **A failed summary must not fail the video.** The transcript and descriptions
  are already ingested and already useful by this point. Summary failure is a
  missing field, not a `failed` row.

## Deploy on Unraid

```
docker compose up -d --build media-worker audrey-ai
```

## Verification (to be written against the built phase)

**1. A processed video shows a summary** in `GET /v1/files` and in the upload
page's file list.

**2. The summary is searchable** as its own chunk, attributed to the uploading
user.

**3. A silent video gets a summary** built from frame descriptions alone.

**4. A deliberately failed summary call leaves the row `ready`**, with the
transcript intact and the summary field empty.

**5. The summariser touched no GPU.** Confirm the call went to the cloud model
and `gate.snapshot()` shows no local occupancy for it.

### Rollback

Revert `media-worker` to phase 36. Videos keep their transcript and
descriptions and lose the summary field.

## What this unblocks

All four artifacts now exist for a video: transcript, visual assessment,
keyframe embeddings, summary. [Phase
38](phase-38-video-optimise.md) makes producing them affordable.
