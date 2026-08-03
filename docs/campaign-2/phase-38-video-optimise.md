# Campaign 2 Phase 38 — making video ingest affordable

By the time this phase starts, video ingest works and is slow. This is the phase
that makes it fast, driven by the stage timings
[phase 36](phase-36-video-visual-assessment.md) shipped rather than by estimates.

**Status: PLANNED — except the keyframe gate, which LANDED early.**

---

## The keyframe gate (landed 2026-08-02)

Built out of order because it needed no container, no GPU, no ffmpeg and no
model — the one part of the video work with no deployment dependency, so it
could be built the moment real footage showed the redundancy. It has no caller
until [phase 36](phase-36-video-visual-assessment.md).

**The observation.** A 288 MB, ~9½-minute retirement video sampled at 1
frame/30s gave 19 frames. Three consecutive frames — 90 seconds apart — went to
`audrey_passthrough/qwen3-vl:32b` and came back describing the same two men in
the same red chairs against the same printed fireplace backdrop, differing only
in a gesture and one hair detail. At `gpu.concurrency: 1` every one of those
calls was exclusive GPU that bought nothing.

This is the common case, not an edge case: talking-head recordings, ceremony
footage, screen recordings of a held slide. Sampling rate cannot fix it — drop
the rate and you miss the cuts that matter, raise it and you pay more for the
same answer. The decision has to be content-driven.

**The mechanism.**
[`media/framegate.py`](../../src/audrey/media/framegate.py): dHash plus Hamming
distance, each candidate compared against the last frame **kept** rather than
its immediate predecessor. That distinction is the whole design — under pairwise
comparison a slow pan differs from its neighbour by almost nothing at every
step, so the entire shot collapses and an ending that looks nothing like the
beginning is never described. Measuring against the last kept frame lets drift
accumulate until it crosses the threshold on its own.

`max_run` caps how many consecutive frames one keyframe may stand for, so a
three-hour lecture on one slide still gets periodic coverage instead of reducing
to a single description.

**Measured: 6 of 19 frames kept, 68% fewer describe calls.** Frames 4–16 — the
static seated conversation — collapsed to one; frames 1, 2, 17, 18 and 19
survived as distinct. The reduction lands where the redundancy is and leaves the
head and tail alone, which is the behaviour to re-check whenever the threshold
is retuned.

The default distance of 8 bits is a starting point, not a finding. Run it
against a corpus before trusting it:

```bash
PYTHONPATH=src python -m audrey.media.framegate /tmp/frames/*.jpg
```

## Instrument first

`pipeline_seconds` / `pipeline_total` already exist and feed `monitoring/`;
ingest gets the same treatment with a `stage` label — `demux`, `stt`,
`keyframe_select`, `describe`, `summarise`, `embed`, `upsert` — plus a per-frame
histogram. That lands in phase 36. Without it, "video ingest is slow" is not an
actionable sentence and every lever below is a guess.

## The remaining levers, in the order I would try them

1. **Run audio and visual stages concurrently.** Whisper is CPU, frame
   description is GPU-via-passthrough. Serialising them wastes the idle
   resource; overlapping them is close to free wall-clock.
2. **Batch frames per call.** `max_images_per_turn: 4` implies multi-image
   requests work. Either several `image_url` parts per request or a tiled
   contact sheet cuts call count 4–9×. Costs fidelity on small on-screen text,
   so A-B it against the single-frame output phase 36 produced before adopting.
3. **Move description to a cloud vl model.** Decouples ingest throughput from
   the chat GPU completely, and cloud calls run ~3-way parallel. Highest ceiling
   of anything here — gated on verifying the model genuinely accepts images,
   which is the exact mistake the pool comment warns about.
4. **Downscale frames before sending.** A 4K keyframe carries no more legible
   text than a 1080p one after the encoder resizes it anyway.
5. **Tune the whisper tier.** `small` vs `medium`, and int8 on CPU. Only worth
   doing once stage timings show STT is a material share.

Expect this ranking to survive contact with real numbers only partially. That is
what the instrumentation is for — the gate got promoted to "landed early"
precisely because real footage moved it from a guess to a measurement, and the
same should happen to the rest of this list.

## What's NOT in scope

- **No re-encoding or proxy transcodes.** We extract, we don't transform. A
  cheaper decode is not worth a rewritten source.
- **No changes to the fairness model.** Faster ingest, same gate. If a lever
  only works by leaving the round-robin, it is not a lever, it is a regression.

## The parts that will bite

- **Concurrency reintroduces contention.** Running audio and visual stages at
  once means one job holds both a CPU-heavy whisper pass and a GPU slot. Fair
  against other users, less fair against the uploader's own chat.
- **Batching hides per-frame timings.** The moment frames go out four at a time,
  the per-frame histogram stops meaning what it meant. Keep a single-frame path
  for measurement.
- **Every lever changes output, not just speed.** Batching, downscaling and a
  different vl model all change what the descriptions say. Each needs an output
  comparison, not just a stopwatch.

## Deploy on Unraid

```
docker compose up -d --build media-worker audrey-ai
```

## Verification (to be written against the built phase)

**1. Stage timings before and after**, from `/metrics`, on the same source
video. A lever with no recorded before is not a measured improvement.

**2. Output comparison per lever.** The same video's descriptions before and
after, read side by side. Speed that costs accuracy has to be a deliberate
trade, not an accident.

**3. Fairness is unchanged.** `gate.snapshot()` during an ingest still shows
interleaving with chat.

**4. The gate's head-and-tail behaviour survives** any threshold retune.

### Rollback

Each lever is independent and revertible on its own. That is the reason for the
ordering — nothing here should require reverting the phase as a unit.

## What this unblocks

Video ingest stops being something you start and walk away from. The sidecar is
also the natural home for future heavy media work — audio-only files, and OCR
for the scanned PDFs that `EmptyExtractionError` currently turns away — without
any of it entering the request path.
