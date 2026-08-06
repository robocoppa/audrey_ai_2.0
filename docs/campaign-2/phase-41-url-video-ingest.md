# Campaign 2 Phase 41 — paste a link, get a video in your KB

Phases 32-40 assume the bytes start on your laptop. Most video worth asking
questions about is already on the internet, and uploading it means downloading
it by hand first, then pushing 300 MB back up a domestic uplink to a box that
has a better connection than you do.

**Status: ALL FOUR STEPS BUILT, NOT DEPLOYED.** Nothing has run on the box —
every claim below about behaviour is a claim about tests. 105 of them, across
`test_files_from_url.py`, `test_media_fetch.py`, `test_media_fetcher.py`,
`test_media_worker.py` and `test_video_jobs.py`.

**The one deliberate departure from this plan** is where the fetcher writes.
The plan had it writing into the user's upload directory and Audrey verifying
afterwards; it writes into a single staging directory instead, and Audrey moves
the file into place once it has checked it. Recorded in step 2 below with the
reason, which turned out to be a uid rather than an aesthetic.

---

## The shape, decided up front

**A field on the upload page, not a tool.** Paste a URL, the box fetches it,
and the row appears in the same list as an upload with the same statuses,
elapsed time and failure reasons.

Not model-facing **in this phase**. That is a sequencing decision, not a
verdict on the idea, and the distinction matters because the first version of
this section got it wrong.

**The weak argument, retracted:** "a model would hallucinate the URL." Too
broad. When someone pastes a link and says "add this", the URL is verbatim in
the turn and there is nothing to invent. The real risk is narrower — *"find me
a video about X and ingest it"*, or *"add that one we discussed"* — where the
URL comes from a search result or the model's recollection. Ingesting whatever
`web_search` returned is a different act from ingesting a link a human handed
over. And that is **deterministically fixable**: require the URL to appear
verbatim in the user's own latest message, checked in `react.py` (which owns
`convo`; `dispatch_one` only sees `user_id`). Same shape as the
`_USER_SCOPED_TOOLS` overwrite — a fact the dispatcher establishes, not an
instruction a model can talk itself out of.

**The argument that actually holds is waiting.** A 9-minute video is ~458
seconds and phase 38 measured 77% of that as the visual pass. A chat turn
cannot wait eight minutes, so the model's honest answer is "started, come back
later" — and then you go to the page, which is where phase 40's elapsed time
and polling already are. For the slow case the page is not merely the safer
surface, it is the better one.

**That flips if step 4 lands.** Subtitles arrive in seconds; skip the visual
pass on a captioned video and ingest is plausibly 20-30s, which *is*
chat-shaped. "Add this" → 20s → "here is what they said about X" beats a page
round trip clearly.

So the tool is deferred, not rejected — see the deferred section at the end.
Meanwhile the page keeps the SSRF surface small: the URL comes from a human
typing into their own authenticated session, which does not remove the need
for an allowlist but does change what it is defending against.

## Why this is smaller than it looks

**It is a new acquisition path, not a new pipeline.** Everything downstream of
"bytes on disk with a `pending` row" already exists and is deployed: claim →
ffmpeg → whisper → keyframes → describe → summary → ingest. This phase adds a
way to *produce* that row. Nothing in `media/worker.py` changes.

## Where the download can NOT happen

Two containers are already ruled out by decisions this campaign made on
purpose, and both would look like the obvious place.

**Not `media-worker`.** It is on `media-net` only, and `media-net` is
`internal: true` — it has no route to the internet. That is phase 34's
fairness invariant expressed as topology: giving it ordinary egress to reach
YouTube would also restore its route to `192.168.1.11:11434`, because ollama
publishes that port on the host and any bridge with egress can reach it. The
compose comment says this in full; the first on-box run proved it. Its
`/data` mount is also `:ro`, and the comment there is explicit that a writable
mount would quietly break phase 33's single-writer claim for the upload bytes.

**Not `audrey-ai`.** It has egress and it already writes to `/data`, so it
*could*. Two reasons not to. yt-dlp breaks whenever YouTube changes something
and needs updating on its own cadence — putting it here couples that to
rebuilding the API image. And a multi-minute download has no business
occupying the API container when a lease-based job queue with crash recovery
already exists next door.

**So: a third container, `media-fetcher`.** Egress, one writable directory, and
nothing else. Not on `media-net` — it has no reason to reach the worker — and
not on `ollama-net`, so nothing resolves or configures the model server's
address. It reaches `audrey-ai` over its own bridge, `fetch-net`, with
`KB_SERVICE_TOKEN`.

**A correction to the sentence above, made while wiring it.** "A bug in a
downloader cannot reach the model server" is *false as stated*, and the
compose file now says so. `ollama` publishes 11434 on the host, so any
container with ordinary egress routes to `192.168.1.11:11434` — which is
exactly what phase 34 discovered on the first on-box run, and why `media-net`
is `internal: true` rather than merely separate. That protection is
**unavailable** to a container whose entire job is downloading things from the
internet. What is actually true is narrower: the fetcher is not on
`ollama-net`, so the name does not resolve, and no configuration hands it the
address. The route exists. That is the price of the feature, and it is worth
writing down rather than inheriting a claim that was true of the other sidecar.

---

## Step 1 — the state machine and the route ✅ BUILT

The uploads row gains **two** states in front of the existing ones, not one:

    fetch_pending → fetching → pending → processing → ready
                            ↘                      ↘ failed

**This is a correction to the plan.** It called for a single `fetching` state
plus a `fetch_lease_id` column to say whether anyone held it. Two states is
better and the reason is structural: it makes the fetch stage the *same shape*
as the ingest stage, where `pending` waits and `processing` is held. The claim's
status transition, done under one lock hold, then **is** the mutual exclusion —
which is exactly how `_claim_job_sync` already works, so neither claim needs a
`lease_id = ''` predicate and there is no second way to be leased. The extra
column bought nothing that a status it already had to write did not.

Consequences worth stating:

- `lease_id` and `leased_at` are **shared by both stages**, safely, because a
  row is `fetching` before it is ever `processing` and never both. That sharing
  is what makes phase 40's elapsed-time display work for a download with no new
  column and no second code path — `_waiting_for_s` reads one stamp and does
  not care who wrote it.
- `sweep_expired_leases` is **parameterised by stage** rather than duplicated,
  via a `_LEASE_STAGES` table mapping the held status to the status a swept row
  returns to and the name of who abandoned it. The plan asked for this check
  ("prefer extending to duplicating if the shapes turn out to be the same") and
  they do.
- `fail_job` gained a `stage` argument for the same reason. It keeps a fetcher
  from failing a row that has moved on to ingest, which is the one way the two
  stages could reach across each other.

**`uploads` gains one column**, `source_url`. Adding a column means **schema,
migration, and the SELECT in `_list_user_sync`** — phase 40 removed the fourth
place by driving the route's projection from `FileRow.model_fields`, so
declaring the field on the model is enough for it to reach the response. Do not
reintroduce a hand-written key list.

`POST /v1/files/from-url`, `require_user`, in this order:

1. **Allowlist the host before anything else.** `kb.fetch.allowed_hosts`, an
   explicit set, **exact matches**, and an **empty list refuses everything** —
   an absent config must not hand a deployment an open downloader. Not
   "whatever yt-dlp accepts", which is ~1800 sites including plenty that serve
   arbitrary files. It reuses the *shape* of `tools-server/fetch.py`, not the
   function, since that one exists to allow the open web.

   Matched on `urlparse(...).hostname`, never `netloc`. `hostname` lowercases,
   drops the port, and — the part that matters — drops any `user:pass@` prefix,
   so `https://www.youtube.com@evil.test/x` is refused rather than passing a
   membership test against a string that starts with an allowed host.
2. **Duplicate URL**, cheap and the likeliest mistake: a double-clicked button
   or a link already in the list, each costing a second download of the same
   300 MB. Exact string match only — `youtu.be/X` and `youtube.com/watch?v=X`
   will not recognise each other, because canonicalising needs the video id and
   only yt-dlp knows that. Failed rows are excluded, since a failed fetch is
   precisely the one a user retries by pasting the link again.
3. **Quota before download, not after.** The size is not known yet, so this is
   a pre-check against `kb.fetch.max_bytes_mb`, re-checked for real once the
   bytes land.
4. Insert the row `fetch_pending` with `bytes=0`, a placeholder filename (the
   video id, so the row is not blank in the list), and return immediately.
   **Nothing is downloaded in the request.**

Three service routes, `require_service`: `POST /v1/files/fetch/claim`,
`POST /v1/files/fetch/{file_id}/result` and `.../failed`. They are declared
*before* the `/{file_id}/...` routes so path-matching order is a property of
the file rather than a coincidence of naming.

**The result route verifies rather than records.** `media-fetcher` is a
separate container that can be wrong, stale or broken, so it re-derives all
three things it is told, and a failed check fails the row with a reason and
unlinks the bytes:

- **the file is where the row implies** — `_source_path` rebuilds the path from
  `file_id` plus the reported filename's extension, so a filename whose
  extension disagrees with what was written hands the worker a path to nothing,
  which would otherwise surface three attempts later as an unexplained error;
- **the bytes are a video** — the same libmagic gate an upload passes. Without
  it a downloader that saved an HTML "video unavailable" page pushes that into
  the transcription queue;
- **the real size fits the quota** — the first moment the true number exists.

`attempts` resets on the handover to `pending`. The counter is per stage:
carrying a download's attempts into ingest would give a video that took two
tries to fetch only one try to transcribe, making a slow network look like a
bad video.

**40 tests** in `tests/test_files_from_url.py`.

## Step 2 — the fetcher container ✅ BUILT

`docker/media-fetcher.Dockerfile`, modelled on `docker/media-worker.Dockerfile`.
yt-dlp pinned by version, and pinned deliberately: an unpinned downloader that
auto-updates is a supply-chain path into a container with write access to
uploads. The cost is real and accepted — YouTube changes something every few
months and an unpinned yt-dlp would have fixed it before you noticed. Here the
symptom is fetches failing with a reason from `friendly_reason`, and the fix is
bumping `YTDLP_VERSION` and rebuilding.

Claim → download → post result, in `media/fetcher.py`. The judgement lives in
`media/fetch.py` so it can be tested without a container: `probe_url`,
`check_limits`, `download`, `friendly_reason`, `parse_vtt`. `post` and
`Stopping` moved to `media/service.py`, shared with the worker rather than
copied — `worker.py` re-exports both names so the test modules that patch
`audrey.media.worker.post` still work.

The caps arrive **on the claim**, config-driven from `kb.fetch`, for the same
reason `FrameSettings` does: the fetcher does not mount `config.yaml`, and a
parallel set of environment variables is a second source of truth that drifts
silently.

- **duration** — `max_duration_s`, refusing a 6-hour stream from a
  metadata-only pass, before a byte is downloaded.
- **filesize** — `max_bytes`; the metadata estimate first, then a hard
  `--max-filesize`, then a stat of what actually landed. Three checks because
  the first is frequently absent and the second does not apply to every
  downloader path.
- **wall clock** — `lease_seconds` minus a 60s reserve for the result post,
  exactly as the worker does with its own.

**One metadata call, not several.** `-J` rather than a series of `--print`
passes: it costs the same request and it is the only way to learn what caption
tracks exist, which step 4 needs *before* it can ask for the right one.

### The naming contract, and where it moved

**The plan said `<dest_dir>/<file_id><ext>`. It is `<stage_dir>/<file_id><ext>`,
and Audrey does the move.** The claim hands a directory and a `file_id`, not a
path, because the extension is not known until yt-dlp has picked a container;
the fetcher reports a `filename` whose extension is that same `<ext>`; the
result route rebuilds both paths from it and fails the row if nothing is there.
All of that is unchanged. What changed is which directory, for two reasons:

1. **A partial download never exists at the path the row implies.** The plan
   already asked for temp-then-rename; putting the temp path in its own
   directory is the same idea with the failure mode removed rather than
   avoided.
2. **The uid forced it, and this is the part the plan missed.** `audrey-ai`
   runs as root, so the user directories it creates are `root:root 0755`. A
   non-root fetcher cannot write into them — so the plan's layout had exactly
   two outcomes available: run the container with internet egress and a
   downloader as root, or give it one directory it owns. One shared
   `.staging` (mode 0777, leading dot so it can never collide with a
   `sanitize_user` output) is the second.

⚠️ **This half-landed, and it cost an afternoon on the box.** `FetchClaim`
dropped `dest_dir` for `stage_dir` and `files.py` learned to do the move — but
`fetcher.py` kept both the field read and the rename. First claim on the box:
`KeyError: 'dest_dir'`, before the metadata pass, so nothing downloaded.

**The tests did not catch it because the fixture invented the contract.**
`tests/test_media_fetcher.py` hand-wrote a job dict and, through the change,
carried *both* `stage_dir` and `dest_dir` — so the fetcher's tests passed
against a payload the route had stopped sending, and two green test files
agreed with each other about a shape neither end actually used. The fixture now
builds the job from `FetchClaim(...).model_dump()`, which makes a field only
one side knows about impossible to write down.

**The second failure was worse than the first.** The `KeyError` escaped
`handle_job`, killed the process, and the container restarted — leaving the row
in `fetching` with nobody holding it. Twenty minutes later the lease expired,
the sweep re-queued it, the next claim crashed identically. From the upload
page that is a download that has been running for an hour, indistinguishable
from a slow one, with the only evidence in a container log nobody has reason to
open. `run()` now catches anything unexpected, reports the row failed with the
error in `failure_reason`, and keeps polling. `YtDlpMissingError` still
propagates: a wrongly-built image should crash-loop visibly rather than fail
every queued URL on the way past.

The gain is bigger than the fix. **Audrey is now the only writer of the user
directories** — the same single-writer argument phase 33 made for sqlite and
phase 34 expressed as a read-only mount on the worker — and verification
happens *before* placement rather than after, so the final path means "passed
every gate" at every instant. Compose mounts only that one directory, so the
container with egress has no route to anybody's stored files at all.

**`env_file: .env` is banned here**, as it is for media-worker, and it bites
harder: that file carries `OLLAMA_HOST`, and this is the container that
actually has a route to it.

**mp4 or nothing.** `ALLOWED_VIDEO_MIMES` is exactly `{"video/mp4"}`, so the
format selector prefers mp4 streams and `--remux-video mp4` catches the rest.
Without it a webm download is refused by the same libmagic gate that stops an
HTML error page — correctly, but with a message that reads as "the download
broke". This is also why the image carries ffmpeg: the resolution cap needs a
merge, and the remux needs a container rewrite. Both are stream copies.

**480p.** The video track feeds keyframes to a vision model and nothing else —
the transcript comes from captions or from whisper on the audio. So the cap is
set by what a describe call can still read, not by what the source offers, and
1080p is roughly four times the bytes for the same description. The cost is
small text burned into a frame; raise the cap in `fetch.DEFAULT_FORMAT` if
descriptions go vague about terminals or slides.

## Step 3 — the upload page field ✅ BUILT

A URL input above the drop zone. On submit, `POST /v1/files/from-url`, then
`refreshList()` — phase 40's polling already starts on its own when any row is
`pending` or `processing`, and both fetch states join that set.

`statusCell` gains **two** cases, not one, and keeping them apart is the value:
`queued to download` means no fetcher has picked it up, `downloading — 2m 14s`
means bytes are moving. Collapsing them makes a stopped `media-fetcher` look
exactly like a slow download. Elapsed comes from the same `leased_at` /
`server_time` skew correction already built.

### The download reports itself — and this overturns a phase-40 decision

Added 2026-08-06, at the user's request: a download shows `downloading — 42%
(121.4 MB of 288.3 MB)` and the video's **real title**, rather than elapsed
seconds against a video id.

**Phase 40 explicitly declined a progress protocol**, and that refusal still
stands where it was made. It does not transfer here, and the difference is
worth stating rather than quietly overriding:

- **The ingest stage has no denominator.** "Whisper done, frames next" is three
  coarse steps with no honest fraction between them, so the choice there was
  between elapsed time and an invented percentage. A download has a real
  numerator and a real denominator, reported by the downloader — turning that
  into a percentage invents nothing.
- **The surface phase 40 was avoiding was a sibling of `ingest_result`**: a
  route that could half-complete a row, with lease logic that had to tolerate
  partial updates. `POST /v1/files/fetch/{id}/progress` writes three display
  fields under a `status = 'fetching'` predicate and **cannot transition
  anything**. There is no state for it to strand a row in, which is the whole
  reason it is safe to add.

**The title is the part that matters most and was nearly free.** The fetcher
learns it from the metadata pass before it downloads anything and simply never
told anyone until the result post. It now sends it immediately, writing over
the video-id placeholder in `filename` — safe during `fetching` precisely
because the bytes are in staging, so nothing derives a path from that column
until `complete_fetch` writes the real name with its real extension.

Three things this had to get right:

- **Progress must be monotonic.** Above a site's highest pre-muxed quality,
  video and audio arrive as two separate downloads and yt-dlp counts each from
  zero — so a naive forward reads 100%, then 3%, then 100%, which looks like a
  download that restarted. `_ProgressReporter` folds a finished stream into a
  running total. The denominator can still step *up* once, when the second
  stream's size becomes known; that is honest, and it shows as the "of 288.3
  MB" figure changing rather than as a percentage mysteriously falling.
- **One unknown total makes the whole total unknown.** Summing only the streams
  that reported a size gives a denominator the numerator will overtake — "100%
  (130 MB of 120 MB)", still downloading. The page falls back to a bare byte
  count, and to elapsed time before any bytes move.
- **Streaming needs `Popen`, not `subprocess.run`.** Progress that arrives
  after the process exits is not progress. Two traps came with it: stderr goes
  to a temp file, because reading stdout line by line while stderr fills its
  own 64 KB pipe deadlocks; and the timeout is a watchdog timer rather than a
  deadline checked between lines, because `readline` blocks and a downloader
  that hangs silently would sit past any between-lines check forever.

Updates are throttled to one every 2s — yt-dlp emits several a second, the page
polls every five, and each POST is a sqlite write on the connection every other
request shares. The first goes straight through.

Three smaller things the page needed:

- **The link is shown on the row.** The filename is a placeholder until yt-dlp
  reports the real title, so "is it fetching the video I meant?" is answerable
  from the link and from nothing else.
- **Size reads `—`, not `0 B`, before the download.** "0 B" is a measurement,
  and printing one for a number nobody has taken yet gets read as a failure.
- **The allowed hosts are published in `Limits` and named in the placeholder.**
  Describing them invites a paste of something that is not one, and the refusal
  then arrives after the click instead of before it. An empty list disables the
  field rather than offering an input that always fails.

**The failure text is the deliverable here, not the happy path.** Private
video, region blocked, members-only, age-gated, live stream, deleted — these
are the common cases, and `friendly_reason` maps them to sentences. Unmapped
output is **passed through truncated, not replaced**: the failure modes are not
a closed set, and a new one should reach the user as whatever yt-dlp said
rather than being flattened into a generic message that tells them nothing.

## Step 4 — subtitles before whisper ✅ BUILT

The largest quality win in the phase, and it is nearly free.

Prefer **manual subtitles → auto-captions → whisper**. Which one to ask for is
decided from the `-J` metadata, never from what lands on disk: `--write-subs`
and `--write-auto-subs` produce files with identical names, so asking for both
and guessing which arrived is how a row ends up claiming a human wrote its
auto-captions.

The transport already existed. `TranscriptSegment` moved up the file so the
fetch routes can name it — the same import-order lesson `JobResultResponse`
taught in step 1 — and a caption track becomes exactly the object whisper
produces, so chunking, the `[HH:MM:SS]` sidecar and the frame-description
context did not have to learn a second shape.

The handover needed one thing the plan did not mention: **the two stages never
hold the row at the same time.** So the transcript is stored on it —
`fetched_transcript`, JSON — written by `complete_fetch`, handed to the worker
on its claim, and **cleared by `complete_job`**, by which point the text is
chunked, indexed and in a sidecar and the column would be a third copy on a row
that `SELECT *` reads on every claim. Deliberately *not* cleared by `fail_job`
or the sweep: a retry should reuse the captions rather than fall back to
whisper for a video that plainly has them.

**`parse_vtt` is where the win is won or lost.** Two problems, both specific to
auto-captions:

- **Rolling repetition.** YouTube's captions "paint on" — each cue repeats the
  tail of the one before so words appear to accumulate on screen. Parsed
  literally, a ten-minute video says everything two or three times. That does
  not merely read badly: a chunk of triplicated text matches a query about that
  phrasing far more strongly than the sentence deserves. Deduplication is per
  *line* against the previous cue, which is the level the repetition happens at.
- **Granularity.** Cues arrive per phrase, sometimes per second. Adjacent ones
  merge up to 5s so a caption track and a whisper transcript are the same kind
  of object by the time anything else sees them.

**Which source produced the transcript is recorded on the row**, in
`transcript_source`, and shown in the file list. It is written by
`complete_job` from what the **worker** reports, not from what the fetch
offered — those differ the moment a claim carries captions and the worker
transcribes anyway, and the point of the column is to be right. The fetcher may
not claim `whisper`; the route refuses it, because that is the worker's answer
and accepting it would let a row claim a transcript came from a model that was
never run.

**Chapters are DEFERRED, as the plan allowed.** `probe_url` carries them on
`UrlInfo` and nothing reads them. Chapter-boundary chunking touches
`chunk_segments`, which every text ingest path shares — that is a change to the
retrieval quality of every document in the KB, made for videos, and it belongs
in its own change with its own before/after rather than riding in on this one.

---

## Verification

1. **A short public video ingests end to end** and is searchable by a quote
   from it — the same check phase 39 used, against a source nobody uploaded.
2. **The transcript came from subtitles, not whisper**, on a video that has
   them. Check the recorded source field, not the timing.
3. **A private/deleted/region-blocked URL fails with a reason a person can
   act on**, not "download failed". Try all three.
4. **A non-allowlisted host is refused at the route**, before any download.
5. **The quota is enforced.** Fill an account near its ceiling, then paste a
   large video: it must be refused, and refused *before* the download.
6. **A killed fetcher mid-download recovers** — the lease expires, the row is
   swept, the partial file is deleted from `.staging` on the next claim, and
   nothing was ever at the final path. This is the phase-35 crash test with a
   different verb. `docker kill media-fetcher` mid-download, then watch two
   claim polls.
7. **`media-worker` still cannot reach the internet.** `docker exec
   media-worker` and try. This phase adds a container with egress next to one
   that must not have it, and the invariant is worth re-proving once.
8. **The staging directory is writable by the fetcher.** The one thing that is
   a property of the deployment rather than of the code: `mkdir -p
   /mnt/user/appdata/runtime/uploads/.staging` before the first `up`, or the
   daemon creates it root-owned and the first download fails on permissions
   until audrey-ai's next claim chmods it.

## Rollback

Additive. The route is new, the container is new, the page field is new, and
the state is in front of the existing machine rather than inside it. Not
starting `media-fetcher` leaves rows stuck in `fetch_pending` and nothing else
broken; nothing is ever claimed, so nothing is ever swept.

The one part that is *not* purely additive is step 4's touch on the ingest
stage: `JobClaim` gained a `transcript` field and `IngestResultRequest` gained
`transcript_source`. Both default to absent, and the worker treats an absent
transcript as "transcribe it yourself" — which is what it did before — so an
old worker image against a new Audrey degrades to whisper rather than failing.
Setting `kb.fetch.allowed_hosts: []` disables the whole feature at the route
without touching anything else.

---

## More information for later

**Terms of service.** Downloading YouTube content is against YouTube's ToS
regardless of purpose. That is a property of the platform, not a legal opinion,
and it is recorded here so it is a decision that was made rather than a thing
discovered later. It does not change the engineering.

**Source reclamation defaults ON for this path, the opposite of the decision
for uploads. ✅ BUILT 2026-08-06.** Phase 38 turned reclamation off because it
deletes a file the user gave us and there is no download route to get it back —
"nothing can read the bytes back" described an absent feature, not consent. For
a URL-sourced video **the URL is how you read it back**, so the argument that
blocked it does not apply. `kb.video.keep_fetched_source: false` sits beside
`keep_source: true`; the two defaults stay separate rather than one global
flipping.

⚠️ **But "the URL is how you read it back" had to be made true first.**
`requeue_job` hardcoded `status = 'pending'`, which hands a media worker a path
that no longer exists — so shipping the flag on its own would have produced
exactly the irreversible delete phase 38 refused, with a better story attached.
`requeue_job(refetch=True)` sends a reclaimed URL row to `fetch_pending`
instead, and the requeue route's 409 is narrowed to rows with no way back
rather than deleted. The flag depends on that path and says so in `config.yaml`.

**DEFERRED, not rejected: the chat tool.** `ingest_video_url`, a thin
service-token sibling of `POST /v1/files/from-url`, so "add this and tell me
what they said about X" is one turn instead of a page round trip.

Three conditions, in order:

1. **Step 4 works and the fast path is real.** The tool only earns its keep if
   a captioned video ingests in tens of seconds. Against a 458-second job it
   can only say "come back later", which the page says better because it shows
   progress.
2. **The verbatim-URL check ships with it, not after.** The URL must appear
   literally in the user's latest message or the call is refused with a
   message telling the model to ask for it. This is the whole guardrail; a
   prompt instruction is not a substitute.
3. **`_USER_SCOPED_TOOLS` gains the name in the same commit.** Phase 40 built
   `audit_user_scoping()` to warn about exactly this omission at discovery
   time, and a warning is not a gate.

Estimated at well under a day once the route exists, which is the other reason
not to rush it into this phase: nothing is saved by doing it early.

**The obvious next want is a playlist or a channel**, and it should be resisted
until this works for one video. A playlist is a queue-depth and quota problem
wearing an ingest costume.

**Audio-only is a real option worth measuring.** `-f bestaudio` for a podcast
or a talking-head interview skips the visual pass entirely — which phase 38
measured at 77% of a 458-second job — and downloads a fraction of the bytes.
Whether that should be a user checkbox or inferred is open; inferring it wrong
loses the on-screen text that is the whole reason the visual pass exists.
