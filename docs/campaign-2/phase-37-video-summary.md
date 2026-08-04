# Campaign 2 Phase 37 — video summary (one call, shown in the file list)

One model call over the transcript and frame descriptions that
[phase 35](phase-35-video-transcript.md) and
[phase 36](phase-36-video-visual-assessment.md) produced, stored on the row and
shown in `GET /v1/files`.

**Status: BUILT, NOT YET DEPLOYED.** The smallest phase of the video work, and
the one that makes the rest legible.

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

### It runs in `audrey-ai`, not the worker (corrected)

The plan put this in `media/summarise.py`, "acting-as the uploader as in phase
36". Phase 36 found there is no act-as on `/v1/chat/completions` and answered
it with a narrow service route; a second route would work here too, and would
still be the wrong shape.

**A summary is derived from the artifacts, not from the video file.** By the
time one can be written, `ingest_result` is already holding the segments and
the descriptions in memory. Asking the worker to do it would mean shipping the
whole transcript to a summarise endpoint and then shipping it again in the
result post — to produce something the worker never looks at.

So the worker's job ends where the artifacts end. This runs where they land,
which also means it needs no new endpoint, no new auth surface, and no second
copy of a 7,000-character transcript on the wire.

### The input is bounded here, and sampled rather than truncated

A two-hour transcript is ~100k characters. `summary_input_chars` (24k, roughly
45 minutes of speech) caps what the model sees, and when the cap bites the
excerpts are sampled **evenly across the video** rather than taken from the
front — a summary built from the first fifteen minutes is confidently wrong
about the other forty-five and says nothing to indicate it. Same reasoning as
the keyframe cap in [phase 36](phase-36-video-visual-assessment.md).

The prompt is also told when it is reading excerpts, because a model that
believes it has the whole transcript will happily assert what the video
concluded.

The two artifacts are labelled separately in the prompt — what was **said** and
what was **on screen**. A summary reporting a whiteboard as something a person
stated is worse than one that omits it.

## What's in scope

- **[`pipeline/summarise.py`](../../src/audrey/pipeline/summarise.py)** (new,
  **done**) — one call over the transcript and descriptions, in `audrey-ai`.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** (**done**) —
  `summary`, the seventh additive column, written by `complete_job` and
  cleared by `requeue_job` so a re-run cannot show the previous run's text.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** (**done**) —
  `ingest_result` summarises after both other artifacts land; `FileRow` grows
  `summary`.
- **[`kb/ingest.py`](../../src/audrey/kb/ingest.py)** (**done**) —
  `ingest_summary`, deliberately unchunked: a summary that needed splitting
  would no longer be a summary, and half of one answers nothing.
- **[`static/upload.html`](../../src/audrey/static/upload.html)** (**done**) —
  a second row under the file, absent when there is no summary.
- **`config.yaml`** (**done**) — `kb.video.summarise_model`,
  `summary_input_chars`, `summary_timeout_s`.

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

## Verification

**1. A processed video shows a summary** in `GET /v1/files` and as a second
row in the upload page's file list.

```
curl -s http://192.168.1.11:8000/v1/files -H "Authorization: Bearer $TOKEN" \
  | jq -r '.files[] | select(.summary != "") | "\(.filename): \(.summary)"'
```

**2. The summary is searchable** as its own chunk, attributed to the uploading
user. A `kb/query` hit whose `source` ends `.summary.txt`.

**3. A silent video gets a summary** built from frame descriptions alone —
`silent.mp4` is the fixture, and the budget split means it spends its whole
allowance on descriptions rather than reserving half for a transcript that
does not exist.

**4. A failed summary leaves the row `ready`.** The cheapest way to force it
is `summarise_model: "nope:doesnotexist"` for one run: the transcript and
descriptions must still ingest, `chunks` must still be right, and the summary
field must be empty rather than the row `failed`.

**5. The summariser touched no GPU.** `audrey_gpu_gate_wait_seconds` must not
move across a summary, and the log line names the model. `FairLocalGate` is a
no-op for a non-local location, so a cloud summariser cannot queue behind
chat — but the assertion is worth making, because the failure is silent and
only visible under load.

**6. A requeue does not leave the old summary behind.** `requeue_job` clears
the field; a row that kept last run's text while re-processing would be
describing a video it no longer matches.

### Rollback

Revert `media-worker` to phase 36. Videos keep their transcript and
descriptions and lose the summary field.

## What this unblocks

All four artifacts now exist for a video: transcript, visual assessment,
keyframe embeddings, summary. [Phase
38](phase-38-video-optimise.md) makes producing them affordable.
