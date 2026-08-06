# Campaign 2 Phase 41 — paste a link, get a video in your KB

Phases 32-40 assume the bytes start on your laptop. Most video worth asking
questions about is already on the internet, and uploading it means downloading
it by hand first, then pushing 300 MB back up a domestic uplink to a box that
has a better connection than you do.

**Status: STEP 1 BUILT, NOT DEPLOYED.** Steps 2-4 are unbuilt. Nothing has run
on the box — every claim below about behaviour is a claim about tests.

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

**So: a third container, `media-fetcher`.** Egress, `/data` read-write, and
nothing else. Not on `media-net` — it has no reason to reach the worker — and
not on `ollama-net`, so a bug in a downloader cannot reach the model server.
It talks to `audrey-ai` over the published host port like any other LAN
client, with `KB_SERVICE_TOKEN`.

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

## Step 2 — the fetcher container

`docker/media-fetcher.Dockerfile`, modelled on `docker/media-worker.Dockerfile`.
yt-dlp pinned by version, and pinned deliberately: an unpinned downloader that
auto-updates is a supply-chain path into a container with write access to
every user's uploads.

Claim → download → post result. The caps arrive **on the claim**, already built
in step 1 and config-driven from `kb.fetch`, for the same reason `FrameSettings`
does: the fetcher does not mount `config.yaml`, and a parallel set of
environment variables is a second source of truth that drifts silently.

- **duration** — `max_duration_s`, refusing a 6-hour stream from a
  metadata-only `--print duration` pass, before a byte is downloaded.
- **filesize** — `max_bytes`; `--print filesize_approx` first, then a hard
  `--max-filesize` so a lying or absent estimate cannot run away.
- **wall clock** — `lease_seconds`, exactly as the worker has.

**The naming contract with step 1, which the result route enforces:** the claim
hands a `dest_dir` and a `file_id`, not a path, because the extension is not
known until yt-dlp has picked a container. The fetcher must write
`<dest_dir>/<file_id><ext>` and report a `filename` whose extension is that same
`<ext>`. Step 1 rebuilds the path from the reported name and fails the row if
nothing is there, so getting this wrong is loud rather than silent — but it is
the one place the two containers have to agree.

**Write to a temp path and rename into place only on success.** A partial
download at the final path is a file the worker will happily claim and
transcribe.

**`env_file: .env` is banned here**, as it is for media-worker, and for the
same reason: that file carries `OLLAMA_HOST`, and handing this container the
model server's address undoes the isolation the network layout is for. Name
the variables it needs.

## Step 3 — the upload page field

A URL input above the drop zone. On submit, `POST /v1/files/from-url`, then
`refreshList()` — phase 40's polling already starts on its own when any row is
`pending` or `processing`, and `fetching` joins that set.

`statusCell` gains one case. Elapsed comes from the same `leased_at` /
`server_time` skew correction already built; a download's elapsed time is the
one thing a user watching a slow fetch actually wants.

**The failure text is the deliverable here, not the happy path.** Private
video, region blocked, members-only, age-gated, live stream, deleted, "this
video is unavailable" — these are the common cases, not edge cases, and they
must land in `failure_reason` in the user's words. yt-dlp's stderr is close to
usable; map the frequent ones and pass the rest through truncated rather than
inventing a generic "download failed", which is the message that generates the
support question this field exists to prevent.

## Step 4 — subtitles before whisper

The largest quality win in the phase, and it is nearly free.

yt-dlp can fetch the caption track. Prefer **manual subtitles → auto-captions
→ whisper**. Manual subs are human-authored and routinely better than whisper
output, and both arrive in seconds against whisper's 74s for a 9-minute video.

The transport already exists: `IngestResultRequest.segments` is a list of
`{t_start, t_end, text}`, which is exactly what a caption track is after
parsing. A fetched transcript can be posted on the **fetch** result and the
worker told to skip transcription — `segments` and `frames` are already
independent, and a video with speech and no frames is already an ordinary
successful job.

**Record which source produced the transcript.** "The transcript is wrong" has
a completely different answer for auto-captions than for whisper, and without
this field nobody can tell them apart after the fact.

**Chapters, if present, are a free structural segmentation.** `chunk_segments`
currently uses fixed token windows; chapter boundaries are authored semantic
boundaries and are strictly better where they exist. This is worth doing but
is not load-bearing for the phase — if it complicates step 4, defer it and say
so.

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
   swept, and nothing is left at the final path. This is the phase-35 crash
   test with a different verb.
7. **`media-worker` still cannot reach the internet.** `docker exec
   media-worker` and try. This phase adds a container with egress next to one
   that must not have it, and the invariant is worth re-proving once.

## Rollback

Additive. The route is new, the container is new, the page field is new, and
the state is in front of the existing machine rather than inside it. Not
starting `media-fetcher` leaves rows stuck in `fetching` and nothing else
broken; the sweep will fail them.

---

## More information for later

**Terms of service.** Downloading YouTube content is against YouTube's ToS
regardless of purpose. That is a property of the platform, not a legal opinion,
and it is recorded here so it is a decision that was made rather than a thing
discovered later. It does not change the engineering.

**Source reclamation should probably default ON for this path, and that is the
opposite of the decision for uploads.** Phase 38 turned reclamation off because
it deletes a file the user gave us and there is no download route to get it
back — "nothing can read the bytes back" described an absent feature, not
consent. For a URL-sourced video **the URL is how you read it back**. The
argument that blocked it does not apply. Keep the two defaults separate rather
than flipping one global.

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
