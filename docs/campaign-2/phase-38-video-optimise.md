# Campaign 2 Phase 38 — making video ingest affordable

By the time this phase starts, video ingest works and is slow. This is the phase
that makes it fast, driven by the stage timings
[phase 36](phase-36-video-visual-assessment.md) shipped rather than by estimates.

**Status: MEASURED, and the plan's whole lever ranking is dead.**

- The keyframe gate landed early (2026-08-02).
- Cost attribution landed and **ran on the box** (2026-08-04).
- **The measurement: generation is 87% of the visual pass**, load is a
  once-per-job cold start, prefill is 5%, and queue is zero. See
  [The measurement](#the-measurement-on-the-box-2026-08-04).
- **Levers 1-4 and 6 are all dead on the numbers**, including `keep_alive`,
  which this document added. Together they address at most 13%.
- **Lever 7 was got wrong once before it was got right.** The first attempt
  blamed thinking tokens, changed nothing measurable, and lost three of six
  keyframes to a `num_predict` cap derived from a character count. Recorded in
  full at [Lever 7, first
  attempt](#lever-7-first-attempt--wrong-and-it-did-harm-2026-08-04), because
  the way it was wrong is more useful than the fix.
- **The real lever was the prompt, and it is VERIFIED ON THE BOX.** Phase 36
  reused the chat-screenshot prompt for video keyframes, so the model wrote
  markdown scaffolding and inventoried the furniture. `KEYFRAME_SYSTEM` plus
  the transcript hint: **291.1s → 135.7s wall, 254.5s → 98.6s generation**, and
  the descriptions now surface the on-screen company name that 12,490
  characters of room inventory had buried.
- Source reclamation is built, tested, and **off by default** — the plan was
  wrong to want it on, see
  [Why the default is `keep_source: true`](#why-the-default-is-keep_source-true).

The short version: the plan ranked five levers by expected payoff, one ingest's
worth of instrumentation retired all of them, and the thing actually costing
87% was a prompt inherited from a different feature.

---

## What the plan got wrong about its own ordering

The lever list below is ranked by expected payoff, and every entry in it is a
bet on which *part* of a describe call is large. Phase 36 measured the whole
call — 62.3s per frame, with a 4x spread it could not explain — and that number
sizes the problem without pointing at any of them:

| If the time is going to… | the fix is… | and these do nothing |
|---|---|---|
| waiting on `FairLocalGate` | lever 1 or 3 | batching, downscaling |
| loading the model | `keep_alive` — **not on the list** | batching, downscaling |
| the image, as tokens | lever 4, maybe lever 2 | `keep_alive` |
| writing prose | a shorter prompt and `num_predict` — **not on the list** | all four |

Two of the four plausible answers are not in the ranking at all, and the two
that are sit at positions 2 and 4. So the ranking is not a plan, it is a list
of guesses in a confident order — and each one costs a build, a deploy, and a
full re-ingest to test.

Ollama has been returning the answer on every response the whole time.
`OllamaClient.chat` hands back the entire response dict, and
[`pipeline/vision.py`](../../src/audrey/pipeline/vision.py) was discarding
everything but `message.content`.

**A note on the prompt, which is what prompted the suspicion.** `DESCRIBE_SYSTEM`
was written in the vision-sidecar phase for *chat screenshots*: "Transcribe
every piece of visible text VERBATIM… preserve line breaks, code indentation,
table rows, labels, legends, and axis values." Phase 36 reused it verbatim for
keyframes. Pointed at a photo of two men in chairs, that is a prompt asking for
a long answer about a frame with nothing in it to transcribe — which would make
`eval` the dominant stage, and would make the cheapest available lever one the
plan never considered. That is a hypothesis, not a finding, and the point of
the instrumentation is that it no longer has to be argued.

## The measurement (on the box, 2026-08-04)

Six keyframes of `jasonRetirement.mp4` through `qwen3-vl:32b`:

| Stage | Total | Share |
|---|---|---|
| **generation** | **254.5s** | **87%** |
| load | 21.1s | 7% — and 20.1s of that is frame 1 |
| prefill | 14.3s | 5% |
| queue | 0.0s | 0% |

Per-frame wall clock 28.8-81.4s. The stages sum to 289.9s against 291.1s
measured, so the attribution is complete.

**Every lever the plan ranked is dead, and the numbers kill them individually:**

1. **Concurrency** and **3. a cloud vl model** were both justified by GPU
   contention. `queue=0.0s` on all six calls. *(Caveat: the box was idle. This
   says the gate is not contended when nothing else runs, not that it is fair
   under load — phase 36 step 5 remains genuinely open.)*
2. **Batching frames per call** amortises per-call overhead, and there is none
   to amortise: prefill is 2,249 tokens on *every* frame, so it is linear in
   images, and generation would scale with the batch too.
4. **Downscaling** attacks prefill — 2.3s per frame. Halving the image saves
   roughly 7s out of 291s.
6. **`keep_alive`**, the lever this document added: load is 20.1s on frame 1
   and **0.2s on every frame after**. The model stays resident. Also dead.

That is at most 13% between them, and the plan had nothing pointed at the
other 87%.

### What generation was spent on — and the wrong turn taken here

9,486 tokens produced 12,490 characters of description: **1.3 characters per
token**, where English prose runs at roughly 4.

The inference drawn from that number was that the missing two thirds were
thinking tokens — `ollama show qwen3-vl:32b` does list `thinking` among its
capabilities, and thinking tokens are counted in `eval_count` without ever
appearing in `message.content`. It was wrong, and the next section is the
measurement that says so.

**The mistake is worth naming precisely, because the number was real and only
the reasoning about it was bad.** 1.3 chars/token is a *derived* quantity, and
it was compared against an *assumed* constant to reach a confident conclusion
about a mechanism nobody had looked at. The actual descriptions were sitting in
`{file_id}.frames.txt` the whole time, and one `head` of that file answered the
question in seconds — markdown scaffolding and an inventory of the room, at
~1.9 chars/token, which is simply what this tokenizer does with `**bold**` and
nested bullets.

Read the artifact before theorising about the metric.

## Lever 7, first attempt — WRONG, and it did harm (2026-08-04)

The diagnosis above was that the missing two thirds were thinking tokens. It
was not, and the re-measurement says so plainly. Comparing frames 1-3 against
frames 1-3 of the previous run, with `think: false` in force:

| | before | after |
|---|---|---|
| wall | 107.1s | 109.5s |
| generation | 78.9s | 80.4s |
| tokens | 2,955 | 3,011 |
| chars/token, frames 2-3 | 1.90 | 1.91 |

Nothing moved. `num_predict` demonstrably took effect — frame 1 stopped at
exactly 1024 tokens — so the config was live and `think: false` was sent. It
simply was not where the time was going.

**And `num_predict: 1024` broke the ingest.** It was set from an estimate of
440-660 tokens per description, obtained by dividing characters by an assumed
~4 chars/token. The real ratio is ~1.9, so the cap bit immediately: frame 1
truncated to 636 characters against neighbours at 2,181, and **frames 4, 5 and
6 hit the cap before emitting any content at all**, returned empty, were 502'd
by the describe route, and were dropped. Three of six keyframes lost, `chunks`
15 against 25.

Three lessons, all cheap to state and none of which were:

- **Do not derive a token budget from a character count.** Measure the ratio.
  `audrey_vision_eval_tokens` exists precisely to make that a lookup.
- **A diagnosis from a derived ratio is a hypothesis.** "1.3 chars/token, so
  two thirds is thinking" reasoned from an assumed constant to a confident
  conclusion, and one look at an actual description would have settled it
  before a deploy.
- **`ingest_result`'s short-frame warning named a cause it could not know.**
  It said "the visual pass ran out of budget" for any shortfall, and pointed
  the investigation at a lease that had never been touched — the worker log
  showed the full 900s budget and three 502s. It now reports the count and
  defers to the worker's per-frame log.

What did improve: per-frame generation variance collapsed, 23.6-78.6s before
against 26.4-27.3s after. The 81.4s runaway is gone. The mean did not move.

## Lever 7, second attempt — the prompt (built 2026-08-04)

Reading one actual description answered in seconds what two rounds of
inference had not:

```
**Layout and Key Elements:**
- **Two individuals** sit on chairs with red cushions and black metal
  frames... Above the banner, a light beige wall displays a dark
  brown/bronze metal decorative cross (star-shaped with scrollwork).
```

Two independent wastes in one output:

- **Markdown.** Headings, bold markers, nested bullets. This text is stored as
  a retrieval chunk and never rendered, so every marker is tokens spent on
  formatting nobody sees — and it is most of the gap between 1.9 chars/token
  and the ~4 that plain prose runs at.
- **Inventory.** A decorative cross with scrollwork, a polo-shirt logo, two
  pieces of wood on a stand. Nobody will search for any of it. The first two
  keyframes opened with a byte-identical sentence about the backdrop.

`DESCRIBE_SYSTEM` is doing exactly what it was written to do — verbatim
transcription of a pasted screenshot — applied to footage with nothing in it
to transcribe. Phase 36 reused it unchanged, and that reuse is the 87%.

`KEYFRAME_SYSTEM` is the video version: plain prose, no markdown, **lead with
any text visible in the frame** (the slide, the whiteboard, the document —
what a transcript cannot capture and what someone will genuinely search),
then briefly what is happening, and explicitly *do not* inventory furniture,
clothing or decor. A frame with nothing written in it and nothing happening
gets one or two sentences.

`describe_one_image` defaults to it, since that entry point exists for the
keyframe pass. `describe_images` keeps `DESCRIBE_SYSTEM` — a chat turn about
an error dump wants every legible character, which is the opposite need, and
the two prompts must not converge.

### Verified on the box, 2026-08-04

Same video, same six keyframes:

| | before | after |
|---|---|---|
| wall clock | 291.1s | **135.7s** |
| generation | 254.5s | **98.6s** |
| tokens | 9,486 | 3,685 |
| characters | 12,490 | 2,008 |

**2.1x faster overall, 2.6x less generation** — 58% off the non-load work once
the once-per-job cold start is excluded from both. All six frames described, no
rejections. `chunks` 16 rather than 25: descriptions now fit one chunk each
instead of being split into fifteen.

The transcript hint is confirmed arriving by prefill alone. It was exactly
2,249 tokens on every frame before — prompt plus image, nothing else — and is
now 2,411-2,479 and varies per frame.

**The descriptions got better, not merely cheaper**, which was not guaranteed
and is the more important result. The old pass spent 12,490 characters on
carpet, chair frames and a decorative sunburst, and never once named the
company. The new ones open with `ACOM TECHNOLOGIES`, `AM enertec`, `THANK
YOU` — text that was in those frames all along and was being buried under the
inventory. That text is the entire reason a visual pass exists beside a
transcript, and until this change it was absent from the artifact completely.

### The loose end

Frames 4-6 generated 648-1,156 tokens for 234-267 characters: **0.23-0.36
chars/token**, worse than the 1.9 this started from. Frames 1-3 sit at 1.1-1.5.

The difference is content. Frames 1-3 are the same two men in the same chairs;
4-6 are new scenes with on-screen text to read. So hidden generation is real,
`think: false` is not suppressing it, and it concentrates where the model has
something to work out.

Not chased, deliberately. It is no longer the dominant cost, the 2.1x is
banked, and the honest position is that the earlier thinking diagnosis was
wrong about the magnitude and the fix rather than wrong about the phenomenon.
Anyone picking this up should start by capturing `message.thinking` from the
response rather than inferring from a ratio again.

## The sampling knobs, retained (2026-08-04)

`vision.think: false`, plus `num_predict: 1024` and `temperature: 0.3`, wired
through a new `think` parameter on `OllamaClient.chat`.

**`think` is tri-state, and that is not fussiness.** Ollama *rejects* the field
for a model that does not declare the `thinking` capability rather than
ignoring it, so a plain boolean default would break every non-thinking model in
one edit. `None` means "send no field at all" and reproduces the previous
request exactly; only an explicit setting changes anything. Check
`ollama show <model>` before adding anything to the `vl` pool.

The other two are hedges rather than the lever:

- **`num_predict: 2048`** is a ceiling. It was 1024 for one deploy and that
  was a guillotine, not a ceiling — see the previous section. 2048 clears a
  normal frame's ~1,150 tokens with headroom while still stopping the observed
  2,914-token runaway.
- **`temperature: 0.3`** because the model ships at 1.0 — a creativity setting
  applied to a transcription task, and the likeliest cause of the known
  non-determinism in description length between requeues of an unchanged video
  (15 visual chunks one run, 17 the next). This is the first knob to A-B if
  the descriptions get worse.

Applied to the chat path as well as the keyframe path. The transcription prompt
forbids answering on both, so there is nothing for thinking to contribute
either way — and the chat path is the one with a user waiting on the latency.

## Cost attribution (landed 2026-08-04)

`VisionTiming` in [`pipeline/vision.py`](../../src/audrey/pipeline/vision.py)
reads Ollama's own timings off the response and
[`routes/media.py`](../../src/audrey/routes/media.py) attributes each describe
call across four disjoint stages that sum to the wall clock:

- **`queue`** — the caller's stopwatch minus what Ollama says it did. Ollama
  cannot report this because it never sees it; it is time inside
  `FairLocalGate` and `UserInflightRegistry`.
- **`load`**, **`prompt_eval`**, **`eval`** — straight from the response.

Both a Prometheus histogram (`audrey_vision_stage_seconds{stage}`) and the
existing per-frame log line, because tuning this means reading a handful of
consecutive frames from one video and seeing which number moved — and a
histogram aggregates away exactly the per-frame variation phase 36 measured and
could not explain. `audrey_vision_eval_tokens` records how much prose was
generated, which is the number that turns "the description was long" into "the
prompt asked for it".

Two traps worth naming, both pinned by tests:

- **Ollama reports nanoseconds.** Reading them as microseconds turns 62 seconds
  into 62 milliseconds, which looks like the problem solving itself.
- **A missing `total_duration` must not become queue time.** `queue = wall −
  total` with `total = 0` attributes the entire call to gate contention, which
  fabricates the exact signal the metric exists to detect — and points at the
  most expensive lever in the list.

`describe_images` (the chat path) discards its timing deliberately. It is one
call inside a turn a user is waiting on, not a stage of a background pipeline
anyone is going to tune.

## The regression this turned up

Building the above meant reading how `uploads` rows are written, which exposed
a live defect with nothing to do with phase 38.

`record_upload` was an `INSERT OR REPLACE`. That deletes the conflicting row
and inserts a new one, so **every column the statement does not name reverts to
its schema default** — and it named none of the seven in
`_UPLOADS_ADDED_COLUMNS`. `reconcile_with_qdrant` calls it for every user file
on every boot.

So a processed video lost its `summary` and its `duration_s` at the next
restart of `audrey-ai`. Phase 37's summary was verified working on 2026-08-04
and would have been blank after the next deploy, with no error and no log line.
Reproduced against a real `UploadsDB` before the fix and after it.

The fix is an upsert whose `ON CONFLICT` clause encodes the rule the old
statement violated: **a Qdrant payload is authoritative about content, never
about job lifecycle.** It knows the filename, the size and the chunk count. It
does not know how many attempts a job took, why it failed, or what its summary
said. `status` is deliberately in the preserved group — it arrives as a default
argument rather than as something read from Qdrant, so honouring it on conflict
would let a boot flip a `failed` row to `ready` on the strength of a value the
caller never supplied.

This is the third bug in this campaign from the same root: **`reconcile_with_qdrant`
rebuilds sqlite from Qdrant on every boot, and every additive column is one
more thing it can quietly undo.** Anything that must survive a restart either
belongs in the payload or belongs in a column reconcile is explicitly told to
leave alone. There is no third option, and the failure is always silent.


## Inherited from phase 35: deleting the source (landed 2026-08-04)

`keep_source: false` was planned for [phase 35](phase-35-video-transcript.md)
and deferred here. Unlinking the video after transcription would have broken
phase 36 (which reads frames from the source), phase 37 (which summarises what
36 produced), and phase 33's `requeue` (which points a re-run at a path that
would no longer exist).

This is the first phase where nothing downstream needs the bytes. **It is still
off by default, and the plan was wrong to assume otherwise.**

### Why the default is `keep_source: true`

Both the phase-35 and phase-38 plans took reclamation as the goal and treated
the escape hatch as the concession. That inverts once you say plainly what the
feature does: **it deletes a file the user uploaded.**

The stated justification is that `max_user_bytes` is 1 GiB and three 300 MB
videos exhaust it. That is real, and it is not a reason to delete anyone's
video — 1 GiB is a number in `config.yaml`, on a NAS. The proportionate fix for
"the quota is too small" is a bigger quota, or excluding video sources from the
accounting. Reclaiming user data to fit a self-imposed limit is solving the
wrong problem with the one operation in this pipeline that cannot be undone.

There is a tempting argument the other way, and it is worth writing down
because it is the one that nearly carried: **nothing can read the bytes back.**
There is no download route; a processed video's source is write-only storage
that costs quota and serves the media worker alone. Deleting it appears to lose
nothing.

That reasoning is weak, and the weakness is instructive. It describes a feature
that does not exist yet rather than a decision that the file is disposable. Add
a download button — an obvious, cheap thing to want — and every already-reclaimed
video is permanently broken, with nothing on the row to say which ones or why.
An absent feature is not consent.

The file list makes the same point from the other end. It shows filename, size,
chunk count and a delete button, and reports "Stored: X of 1 GiB". That is the
vocabulary of a file store. Silently deleting the file while still displaying
288 MB needed a strikethrough and a tooltip to explain itself — and a UI that
has to apologise for the model underneath it is usually reporting a design
problem, not a display problem.

So: the mechanism is built, tested, and **off**. Turn it on for genuine disk
pressure on the array, which is a different situation from an accounting number
being inconvenient. What follows describes it under that assumption.

**A retention window, timed from completion.** `source_retention_hours: 24`,
measured from `completed_at` and not from `uploaded_at` — a video that sat in a
stalled queue for two days would otherwise become eligible the instant it
finished, giving a window of zero to exactly the file whose processing most
deserved a second look. That meant a new column; there was no completion time
on the row before this.

**Decision, 2026-08-04:** the box has plenty of disk, so this stays off and the
question is deferred until there is a real constraint to answer. Nothing about
video ingest depends on it.

**The thing that will actually bite first is the quota, not the disk.**
`max_user_bytes` is 1 GiB, so the fourth 300 MB video is refused while the array
has terabytes free. That is a one-line config change whenever it becomes
annoying, and it is the correct fix for the problem reclamation was reaching
for.

### Why the quota reads a flag instead of zeroing `bytes`

Zeroing `bytes` on reclamation is the obvious implementation and it does not
survive a restart: `reconcile_with_qdrant` refreshes `bytes` from the Qdrant
payload, which still carries the original size and always will. Freed space
that silently un-frees itself at the next boot is worse than no reclamation at
all — the user would see their allowance come back and go away again with
nothing to explain either.

So `bytes` stays true to what was uploaded, `source_freed_at` records that the
file is gone, and the quota sums only rows where that is empty. The file list
shows the original size struck through, because the two numbers would otherwise
just disagree — 288 MB in the list, nothing against the allowance, and no way
to tell that from a bug.

### The order of operations, and which way to lose

sqlite is marked **first**, and only then is the file unlinked. Both orders
lose something if the process dies between the two steps, and they are not the
same loss:

- unlink-then-mark leaves a row still billing the user for bytes that do not
  exist, with nothing on disk to prove otherwise and no path that would ever
  correct it;
- mark-then-unlink leaves an orphaned file that costs disk and is invisible to
  the quota — recoverable by hand, and never wrong about what the user owes.

The sweep runs on the claim path, beside phase 33's lease sweep, for the reason
that phase gave: a worker polling every ten seconds is already a heartbeat, and
a supervised background task is one more thing that can stop without telling
anyone. Reclamation has no deadline, so it inherits that heartbeat rather than
growing its own. It cannot raise — a disk error while tidying up must not stand
between a worker and its job.

### What is refused

A requeue of a reclaimed video is a `409`, and **`force` does not override it**.
The other requeue guard protects work merely in progress and can be overruled
by someone who means it; there is nothing here to overrule. Proceeding would
delete the video's existing chunks, queue a job against a path that does not
exist, and burn all three attempts failing on it — ending with a video that
*was* fully searchable and now is not.

Only `status = 'ready'`, `kind = 'video'` rows with a recorded completion time
are ever eligible. A row restored into a fresh sqlite by `reconcile_with_qdrant`
has no completion time, so it is never reclaimed: nothing knows when it
finished, and the conservative answer to "may I delete this irreversibly?" is
no.

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
# laptop, from the repo root. `.venv/bin/python`, not bare `python` —
# framegate needs Pillow and the system interpreter does not have it.
PYTHONPATH=src .venv/bin/python -m audrey.media.framegate /tmp/frames/*.jpg
```

## Instrument first

`pipeline_seconds` / `pipeline_total` already exist and feed `monitoring/`;
ingest gets the same treatment with a `stage` label — `demux`, `stt`,
`keyframe_select`, `describe`, `summarise`, `embed`, `upsert` — plus a per-frame
histogram. That lands in phase 36. Without it, "video ingest is slow" is not an
actionable sentence and every lever below is a guess.

**Phase 36 shipped half of it and that half was not enough** — see [What the
plan got wrong about its own ordering](#what-the-plan-got-wrong-about-its-own-ordering).
A per-frame wall clock says how much a frame costs. It does not say which of
four mutually exclusive fixes would make it cheaper, and four A-B deploys to
find out is exactly the guesswork this section exists to prevent. The stage
breakdown is the missing half, and it is why nothing below has been built yet:
**the next lever gets picked from one ingest's worth of numbers, not from the
ordering here.**

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

**Two candidates the ranking omits**, both cheap, both only justifiable once
the stage breakdown is read:

6. **`keep_alive` on the vl model for the duration of a visual pass.** Answers
   a large `load` stage and nothing else. The GPU holds one local model at a
   time, so a describe call landing between two chat turns pays a full model
   load — and phase 36's unexplained 4x spread (25.5–102.6s) is the shape that
   would make. Cheap to try, and it trades directly against chat latency, so it
   needs the measurement first rather than a hunch.
7. **A keyframe-specific describe prompt, and a `num_predict` cap.** Answers a
   large `eval` stage. `DESCRIBE_SYSTEM` is inherited from the vision sidecar
   and written for screenshots — verbatim transcription of every legible
   character, table rows, axis values. Against a photo of two men in chairs it
   is asking for a long answer about a frame with nothing to transcribe. This
   is the only lever that costs no GPU work to evaluate: the same frames, a
   different prompt, read side by side.

## What's in scope

- **[`pipeline/vision.py`](../../src/audrey/pipeline/vision.py)** (**done**) —
  `VisionTiming`, read off the Ollama response `chat()` was already returning.
  `describe_one_image` hands it back; the chat path discards it.
- **[`routes/media.py`](../../src/audrey/routes/media.py)** (**done**) — four
  disjoint stages to Prometheus and to the per-frame log line.
- **[`metrics.py`](../../src/audrey/metrics.py)** (**done**) —
  `audrey_vision_stage_seconds{stage}` and `audrey_vision_eval_tokens`.
- **[`kb/uploads_db.py`](../../src/audrey/kb/uploads_db.py)** (**done**) —
  `completed_at` and `source_freed_at`, a quota that skips reclaimed rows, and
  the `record_upload` upsert that stops reconcile eating job state.
- **[`routes/files.py`](../../src/audrey/routes/files.py)** (**done**) — the
  reclaim sweep on the claim path, and the requeue refusal.
- **[`static/upload.html`](../../src/audrey/static/upload.html)** (**done**) —
  a reclaimed video's size shown struck through, with the date in the tooltip.
- **`config.yaml`** (**done**) — `kb.video.keep_source`,
  `source_retention_hours`.
- **The levers themselves** — deliberately not built. See
  [Instrument first](#instrument-first).

## What's NOT in scope

- **No re-encoding or proxy transcodes.** We extract, we don't transform. A
  cheaper decode is not worth a rewritten source.
- **No reclamation of non-video uploads.** A document's chunks were extracted
  from its source and there is no worker to re-run; only video has artifacts
  complete enough to stand without the bytes.
- **No user-facing "reclaim now" control.** The sweep is policy, not a button.
  A user who wants their space back has `DELETE /v1/files/{id}`, which is the
  honest version of that request — it removes the chunks too.
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
- **Reclamation is the only irreversible operation in the whole pipeline.**
  Every other mistake in phases 32-39 costs a re-run; this one costs the file.
  It is off by default and should stay off until disk is the actual constraint.
  If it is ever turned on, consider it before every deploy that changes a
  visual or summary stage — the videos already reclaimed will never see that
  change.
- **"Nothing can read it back" is not consent.** The argument that a source is
  disposable rests entirely on there being no download route. That is an absent
  feature, not a decision, and building one later would silently break every
  video already reclaimed. Watch for the same shape elsewhere: a deletion
  justified by what the system does not do yet.
- **`keep_source: false` and a long retention window fight each other.** The
  window exists so a bad ingest can be re-run; the pressure exists because
  bytes sit around. No setting gives both — 24 hours buys one day of second
  chances at the cost of one day of allowance.

## Deploy on Unraid

```
docker compose up -d --build media-worker audrey-ai
```

## Verification

### Part 1 — cost attribution (do this first; it picks the next lever)

**1. Requeue one video and read the breakdown.** One line per described frame,
and the four stages sum to the wall clock. This is the measurement the rest of
the phase is waiting on, so read it before building anything.

```
# Unraid box
docker compose logs --tail=200 audrey-ai | grep "described a frame"
```

Each line reads `… in 62.3s (1489 chars) queue=0.4s load=4.1s
prefill=2.7s/312tok gen=55.2s/486tok`. **Whichever stage is largest names the
lever**, per the table at the top of this document. Watch for the answer being
different on the first frame than on the rest — a large `load` on frame 1 only
is a cold start and is not worth optimising; a large `load` on every frame is
the model being evicted between calls, which is lever 6.

**2. The stages agree with the wall clock.** A breakdown that sums to a third
of `elapsed` means the fields are being read wrong, and every conclusion drawn
from it would be wrong in the same direction.

```
# Unraid box
curl -s http://192.168.1.11:8000/metrics | grep -E "vision_stage_seconds_sum|vision_eval_tokens_sum"
```

`sum(vision_stage_seconds_sum)` should be within a few percent of
`audrey_video_describe_seconds_sum` over the same window.

**3. A summary survives a restart.** The regression this phase found, and the
one most likely to come back — it is invisible until someone reboots.

```
# Unraid box
docker compose restart audrey-ai
# laptop
curl -s http://192.168.1.11:8000/v1/files -H "Authorization: Bearer $TOKEN" \
  | jq -r '.files[] | select(.summary != "") | "\(.filename): \(.summary[0:60])"'
```

Non-empty after the restart. Before the fix this returned nothing at all, for
every video, with no error anywhere.

### Part 2 — source reclamation (only if it is ever turned on)

**Off by default**, so on the deployed config the check is simply that
**nothing is deleted**: upload a video, let it process, confirm the `.mp4` is
still under `/data/uploads/<user>/` afterwards. Everything below applies only
after someone sets `keep_source: false` deliberately.

**4. Nothing is reclaimed inside the window.** With `source_retention_hours: 24`
and a video processed minutes ago, the source is still on disk after a worker
poll and `source_freed_at` is empty. The window is the entire escape hatch for
"that ingest came out wrong, run it again", so this is the one to check before
trusting the sweep with anything.

**5. A video past the window is reclaimed exactly once**, and the quota moves.
Force it with `source_retention_hours: 0` on a test video rather than waiting a
day — that path is otherwise untested until tomorrow.

```
# laptop — before and after a worker poll
curl -s http://192.168.1.11:8000/v1/files -H "Authorization: Bearer $TOKEN" \
  | jq -r '.total_bytes, (.files[] | "\(.filename) \(.bytes) freed=\(.source_freed_at)")'
```

`total_bytes` drops by the video's size; `bytes` on the row does not change.
The `.mp4` is gone from disk and **`.frames.txt` and `.summary.txt` are still
there** — Qdrant payloads point at those, and the delete route's `{file_id}.*`
glob would have taken all three.

**6. The video is still searchable afterwards.** The whole premise. Its
transcript, descriptions and summary are unaffected by the source going away.

```
# laptop
curl -s http://192.168.1.11:8000/v1/kb/query -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "<a phrase from the transcript>", "top_k": 5}' | jq -r '.hits[].source'
```

**7. A requeue of a reclaimed video is refused**, with `force=true` too, and
the row is untouched by the refusal. Proceeding would delete the chunks
verified in step 6 and then fail three times against a missing file.

**8. `keep_source: true` stops all of it.** The escape hatch has to work from
config alone, because it is what someone reaches for after realising they want
old videos re-processed.

### Part 3 — lever 7 (thinking off)

**9. All six keyframes come back.** The first thing to check, because the last
attempt silently lost three of them. `chunks` should be back around 25, and
the media-worker log should carry no `describe: frame N/M rejected` lines.

```
# Unraid box
docker compose logs --tail=200 media-worker | grep "describe: frame"
docker compose logs --tail=200 audrey-ai | grep "described a frame"
```

**10. Generation falls, and chars/token rises.** The **before**, for the same
six frames: 254.5s generation, 9,486 tokens, 12,490 characters, 28.8-81.4s per
frame, ~1.9 chars/token. Dropping markdown alone should move the ratio toward
~4; dropping the scene inventory should cut the token count outright.

**11. The descriptions still carry what matters — read them.** This is the
step that decides whether the lever was worth it, and it cannot be done from a
log line.

```
# Unraid box
docker exec audrey-ai sh -c 'head -c 1200 /data/uploads/*/<file_id>.frames.txt'
```

Shorter is the point. What must NOT be lost is **text visible in the frame** —
a slide, a whiteboard, a document, a name plate. That is the whole retrieval
value of the visual pass, and it is the one thing a transcript can never
supply. Losing the decorative cross is a win; losing a slide title is a
regression that reverts the prompt.

If quality drops, revert in this order: `temperature` first (cheapest to
undo, least likely to be carrying anything), then the prompt, and `think`
last — it is the one that provably changed nothing.

**12. No non-thinking model is in the `vl` pool.** `think: false` is rejected
outright by a model without the capability, so this breaks vision entirely
rather than degrading. `docker exec ollama ollama show <model>` for anything
in the pool; set `think: null` if any lacks it.

### Part 4 — for any further lever

**13. Stage timings before and after**, on the same source video. A lever with
no recorded before is not a measured improvement.

**14. Output comparison per lever.** The same video's descriptions before and
after, read side by side.

**15. Fairness is unchanged.** `gate.snapshot()` during an ingest still shows
interleaving with chat.

**16. The gate's head-and-tail behaviour survives** any threshold retune.

### Rollback

Each lever is independent and revertible on its own. That is the reason for the
ordering — nothing here should require reverting the phase as a unit.

The two parts already built revert differently. Cost attribution is pure
instrumentation and can go at any time. Source reclamation is inert at its
default, so reverting it costs nothing — but note that **if it has been turned
on, reverting the code does not undo it.** Deleted sources stay deleted;
`keep_source: true` only stops the next one.

## What this unblocks

Video ingest stops being something you start and walk away from. The sidecar is
also the natural home for future heavy media work — audio-only files, and OCR
for the scanned PDFs that `EmptyExtractionError` currently turns away — without
any of it entering the request path.
