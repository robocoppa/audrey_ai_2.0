# Plan — catching the maintainer course up with the code

The maintainer course (`docs/lesson-ai/`) ends with the answer pipeline. Since then
Audrey has gained its own application, identity, file and media pipeline,
retrieval upgrades, skills, Projects and a second API. This plan sets out the
lessons that cover that work and the refreshes the existing lessons need.

**The structure was decided 2026-10-10 and revised 2026-10-23** (see
[The course structure](#the-course-structure-revised-2026-10-23)); it replaces
the sequence of 2026-10-08. Work happens **one cycle at a time**:
each new lesson or refresh unit goes through its own gated cycle (see
[Process](#process--one-lesson-per-cycle)), and the next cycle starts only after
the user has reviewed the finished one.

## The rule (approved 2026-10-10)

Lesson numbers in this section are those of 2026-10-10. The rule still holds;
the placement it led to was revised on 2026-10-23 (next section).

### Why reassess

The 2026-10-08 sequence appends seventeen lessons after research mode and
patches Lessons 0–17 along the way. Two problems with that:

- **Much of the new code replaced what an old lesson teaches, rather than
  adding beside it.** The client and the entry path (L0, L4, L5), the tool
  user binding (L9), search itself (L11: hybrid retrieval is on by default, so
  L11's query path is the fallback), the upload flow (L12), identity (L13
  §2.1), the streaming internals (L15) and web search (L16). A later lesson
  for each means the reader first learns a retired design as current, then
  unlearns it.
- **The draft Lesson 18 repeats L4 and L5.** Its container map is L5 §2.1, its
  lifespan table is L5 §2.3–2.4, its two front doors are L4 §2.1, and its
  message trace is L4 §2.4. Only nginx and the data-authority map are new.

### The rule

1. **Replaced means rewrite.** When new code changed how an old lesson's
   subsystem works, that lesson is rewritten to teach the current design.
2. **Added means a new lesson,** placed at the first point where everything it
   needs has been taught.
3. **Orientation stays at the front.** The system map lives in L0, L4 and L5.
4. **Insert between old lessons only where appending would teach the same
   thing twice.** Identity is that case: appended, its core would have to be
   taught once compactly in L13 §2.1, then again in full later.

## The course structure (revised 2026-10-23)

### Why revise

The 2026-10-10 order had three problems, found in the review of 2026-10-23:

- **The pipeline was taught out of request order.** Routing decides the task
  type and the mode before any model is chosen, but came after the model layer.
  Fair scheduling, the GPU gate every local model call waits on, came long
  after the gate first appeared.
- **Identity and the database came too late.** They sat after the knowledge
  base, but model access by group, a tool's caller identity and everything
  per-user depend on them earlier.
- **Lessons overlapped.** Function calling was mostly a tour of three vendors'
  wire formats. The KB lifecycle lesson's upload flow would be taught again by
  the files lesson. The OpenAI-routes lesson overlapped the planned
  event-streaming lesson, and the planned browser lesson repeated Lesson 4.
  The planned jobs and media lessons serve one feature, and skills and
  Projects share their mechanics.

### The course (Lessons 0–26)

Numbers are final. "Now" is the file the content lives in today.

| # | Lesson | Now | Change |
|---|---|---|---|
| | **Part 1 — Foundations** | | |
| 0 | Introduction | L0 | Course map updated to the seven parts (2026-10-23) |
| 1 | Python language features | L1 | Checked in the 2026-10-23 accuracy pass |
| 2 | The libraries | L2, L3 | Takes httpx and a short note on the test suite from L3; L3's embeddings section goes to L13 and its Prometheus section to L26 |
| | **Part 2 — The platform** | | |
| 3 | The platform and the request lifecycle | L4 | Its native trace shrinks to map level when L17 is written |
| 4 | Configuration and startup | L5 | Checked in the accuracy pass |
| 5 | **The application database** | — | New |
| 6 | **Identity and access** | — | New; takes model access from L8 |
| | **Part 3 — The answer pipeline** | | |
| 7 | Classification and routing | L8 | Refreshed in the accuracy pass (in review) |
| 8 | The model layer I: choosing a model | L6 | Written 2026-10-23 (in review); model access moves to L6 |
| 9 | The model layer II: calling models | L7 | Written 2026-10-23 (in review); Private at a high level |
| 10 | Fair scheduling | L15 | Light refresh: grants with explicit owners; a waiter cancelled after its grant returns the slot |
| 11 | Deep mode and research mode | L9 | Rewritten 2026-10-23 (in review) |
| 12 | Tools: function calling and the ReAct loop | L10, L11 | L10 (accepted) takes L11's protocol essentials; L11's caller-executed tools go to L25, and its vendor tour is cut |
| | **Part 4 — Knowledge, memory and the tool server** | | |
| 13 | The knowledge base | L12, L13, L3 | L12 rewritten for hybrid retrieval, with L3's embeddings section and L13's watcher and reconcile; L13's upload flow goes to L19 |
| 14 | Memory and history | L14 | Refresh: §2.1 becomes a recap of L6; canonical history lives in the database |
| 15 | The custom-tools sidecar | L17 | Cycle 2b refresh |
| | **Part 5 — The native application** | | |
| 16 | **Streaming: run events and wire formats** | L16 (part) | New, with L16's streaming internals |
| 17 | **The native run** | — | New, with the browser boundary |
| 18 | **Durable side effects** | — | New |
| | **Part 6 — Files and media** | | |
| 19 | **Files as durable objects** | L13 (part) | New, with L13's upload flow |
| 20 | **Processing media: jobs, leases and the worker** | — | New |
| 21 | **Reaching outside safely** | — | New |
| 22 | **Asking about your files** | — | New |
| | **Part 7 — Shaping a turn, the bot API, and operating** | | |
| 23 | **Skills and Projects** | — | New |
| 24 | **The bot API: Chat Completions and Responses** | L16 (rest) | New, with the rest of L16 |
| 25 | **Responses: caller tools, files and stored responses** | L11 (part) | New, with L11's caller-executed tools |
| 26 | **Operating Audrey** | L3 (part) | New, the closing lesson, with L3's Prometheus section |

The bot API comes late on purpose: its Responses half takes owned file
references and public URLs, so it needs the files and outbound-request
lessons first. Lesson 3 already introduces the `/v1` door, and L16 covers the
streaming driver both doors share.

**Deferred, written once the code settles:**

- **Private processing.** L9 (now L7) §2.8 explains it at a high level today.
  A full lesson would cover the processing policy and its scope, the isolated
  runtime and its allow-list of shared services, local-only tools,
  restrictions that stick to conversations, Projects and files, and private
  task admission. It would follow L23, since Private composes with Projects and
  files.
- **Reference libraries and the tax review.** Owned reference libraries
  (indexing, a pinned embedding identity, search) and the tax review built on
  them (cases, facts, worksheets, verification). Under active development; it
  would go before the closing lesson.

### Numbering and migration

The table uses final numbers. Two scripted renumbers get the files there:

1. **When the database and identity lessons enter.** They're written back to
   back right after cycle 2b, reviewed as drafts, and enter the course after
   configuration and startup in one renumber that also puts the pipeline in
   request order. Lessons still waiting to be merged (L3, L11, L13 and L16) keep
   their own files beside the lesson that will absorb them, so the interim
   course has no gaps: 0 introduction, 1 Python, 2 libraries, 3 satellite
   libraries (L3), 4 lifecycle, 5 configuration, 6 database, 7 identity,
   8 routing, 9–10 the model layer, 11 fair scheduling, 12 deep and research,
   13 tools, 14 function calling (L11), 15 knowledge base, 16 KB lifecycle (L13),
   17 memory and history, 18 the sidecar, 19 the OpenAI routes (L16), then new
   lessons from 20.
2. **At the end.** When L3, L11, L13 and L16 have nothing left that isn't taught
   elsewhere, their files are removed and every lesson takes its final number.

New lessons written between the two renumbers take the next free number at
the time. Each merge removes the moved content from its source file in the
same cycle, so nothing is taught twice in between. Each renumber rewires every
footer. Live documents (this plan, `AGENTS.md`, `PROJECT_STATE.md`, the course
README and Lesson 0's course map) follow each renumber; the AUDIT gets a
mapping note rather than rewritten history; campaign docs and completed plans
keep the numbers they were written with.

### Writing order (one reviewed cycle each)

Numbers are final; "now" gives today's file where it differs.

Done:

1. **System map:** L0, L3 (now L4) and L4 (now L5) rewritten, client examples
   in the foundations; accepted 2026-10-15.
2. **Cycle 2, tool policy:** 2a, tools (L12, now L10), accepted 2026-10-15.
   Moved ahead on 2026-10-23 and awaiting review: the model-layer split (L8–L9,
   now L6–L7), the deep and research merge (L11, now L9), and the Lessons 1–8
   accuracy pass, which refreshed routing (L7, now L8). Their audit findings
   are in the AUDIT.

To go:

3. **2b, the sidecar** (L15, now L17): §1.2, §2.1, §2.3, §2.4, §2.6 and §2.7.
   Approved source work is complete and locally verified October 24; the local
   AUDIT holds the drain. Lesson prose remains with its author; an outline is
   handed over without changing the lesson.
   Next.
4. **New L5, the application database,** written as a draft.
5. **New L6, identity and access,** written as a draft; it takes model access
   from L8. Both enter in the first renumber, which also reorders the pipeline.
6. **The knowledge base** (L13): L12's rewrite, with L3's embeddings section and
   L13's watcher and reconcile.
7. **Tools** (L12): L11's protocol essentials folded into L10.
8. **Light refreshes:** fair scheduling (L10, now L15) and memory and history
   (L14).
9. **New L16, streaming,** taking L16's streaming internals.
10. **New L17, the native run,** with the browser boundary; L3's native trace
    shrinks.
11. **New L18, durable side effects.**
12. **New L19, files,** taking L13's upload flow.
13. **New L20, processing media.**
14. **New L21, reaching outside safely.**
15. **New L22, asking about your files.**
16. **New L23, skills and Projects.**
17. **New L24, the bot API,** taking the rest of L16.
18. **New L25, Responses,** taking L11's caller-executed tools.
19. **New L26, operating Audrey,** taking L3's Prometheus section, while L3's
    httpx and testing note go to L2. Then the second renumber.

That is 22 reviewed cycles in all: five done (the system map, 2a, the split,
the merge and the accuracy pass) and seventeen to go.

### What stays

One lesson or refresh unit per cycle, reviewed before the next. The browser
boundary keeps its TypeScript reading primer, now inside the native run lesson.
Open WebUI material is replaced, not appended. The Python course stays paused.

## Baseline

- The newest lesson is **Lesson 17 (research mode)**, first written
  2026-06-30. Its research-mode content was kept current through 2026-08-15.
  It still ends with "That's it for the course".
- Lessons 0–16 describe the code as of about 2026-07-01. Later edits were
  small patches: `web_fetch` (L16 §2.5), passthrough tool-result turns (L15),
  routing and embedder fixes (L7, L16), and cite re-anchors.
- **None of these appear anywhere in the course:** the application database,
  Cloudflare Access and personal tokens, native runs and run events, skills,
  Projects, the Responses API, outboxes and tombstones, job leases, the media
  worker and fetcher, hybrid retrieval, file-scoped search, `list_my_files`,
  `get_file_text`, the service token, the tool declaration catalogue, readiness.
- **Many lessons still teach the Open WebUI era.** Open WebUI is stopped and
  the native app is authoritative, but L0, L4, L13 and L15 frame Open WebUI as
  the client, and L13 §2.1 explains identity through it. See the
  [replacement checklist](#replacing-the-open-webui-material).

## What changed since

Since 2026-07-01: about 26,000 lines in new modules, roughly 17,000 more lines
of growth in existing ones, an ~8,000-line TypeScript/React client in `web/`,
and three new runtime containers (`audrey-ui`, `media-worker`,
`media-fetcher`).

C3 and C4 below are Campaigns 3 and 4 in `docs/campaign-3/` and
`docs/campaign-4/` (numbering explained in `docs/README.md`).

| Work (campaign, phases) | Lesson coverage today |
|---|---|
| C3 25, 26, 28: research fact-check, claim ledger, grounding diagnostic | Covered by L9 §2.5 |
| C3 27: live eval on the box | Out of scope (tooling) |
| C3 29–31: `web_fetch`, SSRF hardening, KB query auth | L17 §2.5 covers the basic page-opener only |
| C3 32–38, 41–42: video transport, jobs, worker, transcript, visual pass, summary, cost, URL ingest, `audrey_video` | None |
| C3 39, 40, 43: hybrid retrieval, file tools and scoped search, per-file coverage | None |
| Between campaigns (Aug 9–20): schema-pinned router, thinking knobs, embedder residency, model lineup | Partly patched into L6, L8, L17 |
| C4 01: ownership and cancellation, durable outboxes, data controls, tool policy, readiness | None |
| C4 02, 10–12, 15, 18: native app — state, identity, runs, events, provenance, client, Projects, admin | None |
| C4 03: skills | None |
| C4 04, 07–09: downloads, scanned-PDF OCR, audio | None |
| C4 05, 13, 17B: Responses API, reasoning controls | None |
| C4 14, 17A: bot accounts and tokens, model telemetry | None |
| C4 06, 16: router assessment (no change), retired document authoring | Nothing to teach |

## Lesson cards

Each card gives the learner's question, what the lesson covers, the files in
scope for its audit pass, what it builds on, and which older sections it
updates in the same cycle. Numbers are the final ones (the structure table maps
each to today's file). Phase docs are background for the writer only; lesson
prose must not cite phase numbers.

### Part 2 — The platform

**L5 — The application database**

- **Question:** "Where does a conversation live now, and what makes it safe to
  change that storage while real data is in it?"
- **Covers:** SQLite in WAL mode as the authority for accounts, conversations,
  messages, runs, preferences and tokens. Transactions through one store.
  Ordered, additive migrations and the schema version. Typed records and
  repositories, and why routes never write SQL. Owner scoping, where foreign
  and missing return the same 404. Authority versus projection: the search
  archive and Qdrant's search collections can be rebuilt from canonical rows
  and source files, while saved memories live only in Qdrant (L3 §2.6
  introduces the distinction). Online backups before migrations, and verified
  restores.
- **Files:** `src/audrey/app_state/` (`store.py`, `migrations.py`,
  `records.py`, `repositories.py`, `titles.py`), `src/audrey/chat_projection.py`,
  backup and restore parts of `src/audrey/admin_cli.py`.
- **Builds on:** L3 (where data lives), L4 (`lifespan`, `app.state`).
- **Note:** The learner has not met SQL transactions or WAL; teach them from
  scratch, as the foundations lessons teach their topics.
- **Updates:** L4 §2.3 (startup opens the store and runs migrations).

**L6 — Identity and access**

- **Question:** "A request arrives from the browser, a bot with a token, or the
  tools sidecar. How does Audrey decide who it is and what it may use?"
- **Covers:** Authentication versus authorization. Verifying the signed
  Cloudflare Access assertion. `Principal`: a stable Audrey user id and private
  namespace that an email change cannot rename. Account states (pending to
  active), roles, and the Bots group. Personal access tokens: hashed secrets,
  `account:read` and `compat:full` scopes, dated or permanent expiry, revocation
  checked on every use, shown once. Dependency chains (`require_user`,
  `require_principal`, `require_scope`). The internal service token and
  `resolve_kb_caller`. Model access, moved here from L8: a catalog filtered by
  access groups and rechecked at run time, because hiding a picker option is not
  enforcement. First-admin bootstrap.
- **Files:** `src/audrey/identity/`, `src/audrey/auth.py`,
  `src/audrey/model_catalog.py`, `src/audrey/routes/app/{me,admin,models}.py`,
  bootstrap in `src/audrey/admin_cli.py`, identity and token parts of
  `src/audrey/app_state/store.py`.
- **Builds on:** L2 (dependencies), L3, L5. **Background:** C4 phases 02, 14;
  C3 phase 31.
- **Updates:** L8 §2.1 keeps a short pointer here for model access; L14 §2.1
  becomes a recap of this lesson.

### Merged from existing lessons

**L2 — The libraries** (L2, plus parts of L3)

- **Covers:** FastAPI, Pydantic and LangGraph as now; httpx from L3; a short
  note on the hermetic test suite (what it guards, how a maintainer runs it),
  with no test walkthroughs.
- **When:** the closing cycle, after L13 and L26 have taken L3's other sections.

**L12 — Tools: function calling and the ReAct loop** (L10, plus parts of L11)

- **Covers:** L10 as accepted, opened by a short protocol primer from L11: one
  round on the wire (the tools list, `tool_calls`, tool messages), JSON Schema
  as what the model reads, how a model decides to call, `tool_choice`, parallel
  calls, and the failure modes the protocol exposes. L11's tour of three
  vendors' formats is cut; its caller-executed tools go to L25.

**L13 — The knowledge base** (L12, plus parts of L13 and L3)

- **Covers:** L12's rewrite (hybrid retrieval: dense plus BM25 sparse vectors,
  reciprocal rank fusion, the evidence rule, the migration; user versus service
  callers; the embedders and their pinned identity), L3's embeddings and Qdrant
  primer, and L13's watcher and reconcile, with the generic spotlights cut to
  their point. L13's upload flow goes to L19.
- **Files:** `src/audrey/kb/` (ingest, chunk, embed, bm25, fusion, qdrant,
  watcher, reconcile, reembed), `src/audrey/embedding_profile.py`, the query
  routes in `src/audrey/routes/kb.py`.
- **Builds on:** L5 (two stores), L6 (per-user collections, caller resolution).

### Part 5 — The native application

**L16 — Streaming: run events and wire formats** (new, with L16's streaming internals)

- **Question:** "The browser, a Chat Completions client and a Responses client
  all watch an answer being produced. How does one pipeline speak three
  streaming protocols without them drifting apart?"
- **Covers:** Typed, sequenced, client-neutral run events: stages, text
  deltas, tool calls, sources, model used, usage, terminal outcome. The
  emitter. Adapters that turn the same events into AG-UI, Chat Completions SSE
  and Responses events, and the rule that no adapter parses another's display
  text. Shared terminal state (success, failure, or cancellation with partial
  output) used by fast, deep and research. Run observations: how real tool
  activity becomes Sources, Models and Tool calls without leaking retrieved
  text. From L16: SSE and `StreamingResponse`, the stream session and adapter,
  the stage runner, the banner queue and `PhaseTicker`, and cancellation that
  drains owned tasks.
- **Files:** `src/audrey/pipeline/{run_events,agui,run_observations,streaming,banners}.py`,
  `src/audrey/routes/openai/streaming.py`, the streaming parts of
  `src/audrey/routes/openai/pipeline.py`.
- **Builds on:** L3 (two drivers), L11 (deep stages), L12 (tools).

**L17 — The native run** (new, with the browser boundary)

- **Question:** "What happens between pressing Send and a finished answer, why
  does closing the tab not lose it, and what does the browser itself own?"
- **Covers:** Conversations and runs as server-owned resources. Persisting the
  user message and run record before streaming. At most one active run per
  conversation. Run ownership that outlives the browser connection. Stop as
  explicit cancellation that drains owned work. Startup settling interrupted
  runs. The bounded, process-local replay buffer and cursor, and the fallback
  to durable reads. Retrying a failed turn. How the native path enters the same
  streaming driver as `/v1`. Then the browser boundary: `api.ts` as the typed
  boundary, the AG-UI transport, why the browser only renders, and how answer
  details (Sources, Models, Tool calls) arrive. React internals stay out of
  scope.
- **Files:** `src/audrey/routes/app/runs.py`,
  `src/audrey/routes/app/conversations.py`, run and message parts of
  `src/audrey/app_state/repositories.py`, `src/audrey/conversation_titles.py`,
  `web/src/api.ts`, `web/src/agentTransport.ts`, `web/src/App.tsx`, run and
  stream parts of `web/src/ChatWorkspace.tsx`.
- **Builds on:** L3, L5, L6, L16.
- **Needs:** a short TypeScript reading primer; the learner knows neither
  TypeScript nor React.
- **Updates:** L3's native trace shrinks to map level and points here.

**L18 — Durable side effects**

- **Question:** "If the box restarts halfway through archiving a chat or
  deleting someone's data, what guarantees the work finishes, and that deleted
  data never comes back?"
- **Covers:** The transactional outbox: commit the intent with the data,
  deliver later, retry until done. At-least-once delivery made safe by
  idempotency. Archive delivery that never delays the response. Repair status.
  Tombstones that hide data at once and prevent resurrection. Current-user
  export, correction and deletion. The account-purge coordinator and its
  process-local privacy gate.
- **Files:** `src/audrey/pipeline/chat_archive.py`,
  `tools-server/chat_archive.py`, `src/audrey/chat_projection.py`,
  `src/audrey/routes/user_data.py`, `src/audrey/user_data_purge.py`,
  `src/audrey/user_data_visibility.py`.
- **Builds on:** L5, L14 (archive capture), L15 §2.8 (archive read side), L17.
- **Updates:** L14 §2.3–2.6 and L15 §2.8 (delivery is now an outbox).

### Part 6 — Files and media

**L19 — Files as durable objects** (new, with L13's upload flow)

- **Question:** "A 300 MB video crosses a tunnel that caps bodies at 100 MB,
  shows Pending, can be deleted mid-processing, and never reappears. How?"
- **Covers:** Chunked part upload and server-side assembly. Admission by
  sniffed bytes rather than filename. Atomic storage reservations and quotas.
  The file status machine (pending, processing, ready, failed) and who may move
  it. Durable, idempotent deletion with per-file operation locks. Downloads:
  byte ranges, 404 for foreign or missing, 410 for a reclaimed original.
  Attachment snapshots on messages.
- **Files:** upload and download parts of `src/audrey/routes/files.py`,
  `src/audrey/kb/uploads_db.py`, `src/audrey/kb/storage_lifecycle.py`,
  `src/audrey/kb/file_deletion.py`, `src/audrey/routes/app/files.py`,
  `src/audrey/kb/extract.py`, upload ingest in `src/audrey/kb/ingest.py`.
- **Builds on:** L13, L18. **Background:** C3 phases 32, 40; C4 phases 04, 11.
- **Updates:** takes L13's upload flow, which leaves L13 with nothing of its own.

**L20 — Processing media: jobs, leases and the worker** (two planned lessons merged)

- **Question:** "Who picks up a pending video, what does processing do to it
  before a model can answer questions about it, and what happens if the worker
  dies halfway?"
- **Covers:** A job queue built on the uploads database instead of a queue
  server: claim, lease, complete, fail, requeue; refusing a stale lease and
  sweeping expired ones back to pending; attempt counts, and infrastructure
  failures that do not burn attempts; the service-token routes workers call.
  Then the work: probing and audio extraction with ffprobe and ffmpeg, Whisper
  transcription, keyframe sampling and the gate that decides which frames earn
  a describe call, brokered model access (the worker cannot reach Ollama, so
  Audrey describes frames for it), OCR one page at a time under hard bounds,
  bounded fail-soft summaries, and artifacts ingested as ordinary searchable
  text. Cost attribution, and measuring before optimizing.
- **Files:** job-lifecycle parts of `src/audrey/routes/files.py` and
  `src/audrey/kb/uploads_db.py`, `src/audrey/media/`, `src/audrey/routes/media.py`,
  `src/audrey/pipeline/vision.py`, `src/audrey/pipeline/summarise.py`,
  `require_service` in `src/audrey/auth.py`.
- **Builds on:** L9 (`OllamaClient`, describing images), L10 (the gate), L13
  (ingest), L19. **Background:** C3 phases 33–38, 41; C4 phases 07–09.

**L21 — Reaching outside safely**

- **Question:** "When a model or a user hands Audrey a URL, what stops it
  reaching inside the home network, and why does downloading live in its own
  container?"
- **Covers:** SSRF, and why checking the hostname is not enough: redirects,
  and DNS answers that change between check and connect. The `web_fetch` guard.
  Fetching through sockets pinned to vetted DNS answers. Size, time and
  content-type bounds. Parsing untrusted remote documents in a disposable
  worker. yt-dlp in a fetcher that has egress while the worker has none.
  Captions preferred over transcription when a video has them.
- **Files:** `tools-server/fetch.py`, `src/audrey/net/{public_fetch,remote_input}.py`,
  `src/audrey/kb/remote_input_worker.py`, `src/audrey/media/fetch.py`, the
  networks in `compose.yaml`.
- **Builds on:** L15 §2.5 (`web_fetch`), L20. **Background:** C3 phases 29, 30,
  41; C4 phase 13.

**L22 — Asking about your files**

- **Question:** "Find where this video mentions the budget": how does Audrey
  search one file, page through it, and still cover every file?
- **Covers:** One scope object applied to both retrievers. Filename
  resolution and its notices. Pooled results with a guaranteed slot per file.
  Deleted-file exclusion. `list_my_files`, `get_file_text` paging and
  `kb_search` filters, declared like every user-scoped tool. `audrey_video` as
  a retrieval specialist. (The fusion core is taught in L13.)
- **Files:** scope and coverage parts of `src/audrey/routes/kb.py`, file-tool
  routes in `tools-server/app.py`, the video virtual model and its skill.
- **Builds on:** L12, L13, L19, L20. **Background:** C3 phases 40, 42, 43.
- **Updates:** L15 (the file tools in the tool list).

### Part 7 — Shaping a turn, the bot API, and operating

**L23 — Skills and Projects** (two planned lessons merged)

- **Question:** "What is a skill, how is one chosen, and when a conversation
  lives in a Project, what extra context reaches the model, without either one
  granting new power?"
- **Covers:** Skills as declarative, read-only bundles: `SKILL.md` front matter
  plus bounded resources; strict loading (no executables, no paths escaping the
  bundle, a content digest); an atomic registry with per-bundle degradation;
  resolving one skill before the fast/deep branch and injecting its
  instruction once, outside the complexity decision; tools offered =
  discovered ∩ platform policy ∩ skill allowlist; deterministic English
  file-intent rules that abstain when unsure; provenance. Projects as
  owner-scoped groupings with instructions and references to files, not
  copies: snapshotting instructions and Ready file ids at run creation; prompt
  order (platform, then Project, then conversation) and exclusion from routing;
  bounded retrieval; filenames and passages treated as untrusted evidence;
  direct models receive the text context but no tools.
- **Files:** `src/audrey/skills/`, `skills/*/SKILL.md`, skill integration
  points in `src/audrey/pipeline/` and `src/audrey/tools/`,
  `src/audrey/routes/app/capabilities.py`, `src/audrey/app_state/projects.py`,
  `src/audrey/routes/app/projects.py`, `src/audrey/project_context.py`.
- **Builds on:** L7 (routing), L12 (tools), L17 (runs), L22 (file intents).
  **Background:** C4 phases 03, 15.

**L24 — The bot API: Chat Completions and Responses** (new, with the rest of L16)

- **Question:** "A bot calls `/v1/chat/completions` or `/v1/responses`. What
  does Audrey do differently for each, and how does it serve both without a
  second pipeline?"
- **Covers:** From L16: the request schemas and why unknown fields are
  ignored, the dispatch decision tree, the non-streaming path, the passthrough
  fork and its typed outcomes, and mapping options onto Ollama. Then Responses:
  mapping a request onto the shared generation path; typed streaming events
  with contiguous sequence numbers and the completed, failed and incomplete
  terminals; validating before side effects (a 400 before any SSE or file
  fetch); JSON Schema admission and output validation.
- **Files:** `src/audrey/routes/openai/{routes,schemas,passthrough,structured_outputs}.py`,
  the non-streaming and dispatch parts of `src/audrey/routes/openai/pipeline.py`,
  the Responses adapter in `src/audrey/routes/openai/streaming.py`.
- **Builds on:** L6 (tokens), L12 (tools), L16 (streaming). **Background:** C4
  phase 05.

**L25 — Responses: caller tools, files and stored responses** (new, with L11's caller-executed tools)

- **Question:** "How does a bot hand Audrey its own functions, files and URLs,
  and pick up a conversation it stored earlier?"
- **Covers:** Caller-executed function tools on passthrough models, on both
  doors (from L11), and stateless replay. Owned file references and bounded
  public URL inputs. Opt-in storage and continuation, and why that saves
  bookkeeping rather than tokens. Reasoning-effort validation against a model's
  advertised thinking values.
- **Files:** `src/audrey/routes/openai/{client_tools,client_tool_generation,file_inputs,response_storage,reasoning}.py`,
  `src/audrey/app_state/responses.py`.
- **Builds on:** L5, L19, L21, L24. **Background:** C4 phases 13, 17B.

### Closing

**L26 — Operating Audrey** (new, with L3's Prometheus section)

- **Question:** "When one component is down, how does ordinary chat keep
  working, and how do you see what is wrong and what the models cost?"
- **Covers:** Prometheus basics from L3 (counters, gauges, histograms, labels
  and their cardinality). Degraded startup and per-capability degradation,
  recapped across the course. One sanitized readiness snapshot (components,
  tools, skills, queues, workers, gate pressure) for admins and Prometheus.
  Provider telemetry: exactly one terminal outcome per call, and reported token
  usage where a valid zero differs from unknown. Reading the dashboards. The
  course wrap-up.
- **Files:** `src/audrey/readiness.py`, `src/audrey/metrics.py`, usage and
  outcome parts of `src/audrey/models/ollama.py`,
  `src/audrey/routes/app/capabilities.py`, `monitoring/`.
- **Builds on:** L4 (degraded startup), L12 (tool declarations), L15, and the
  course as a whole. **Background:** C4 phases 01, 17A.
- **Updates:** the footer of the last lesson before it (the wrap-up moves here).

## Cycle 1 — the system map (outline approved 2026-10-10; accepted 2026-10-15)

Rewrites L0 and L4, refreshes L5, swaps the client examples in L1–L3, and
updates the course README. Audit records live in the gitignored AUDIT.
The lessons describe current source behavior; L4 includes background first-turn
titles, shared stream context, backup recovery, and catalog-preserving startup.
The cites those source changes moved were re-anchored afterwards (2026-10-12),
and the staleness they caused in later lessons is in the refresh table.

**As written** (beyond the outline): L4's graph section now shows the
`research` node; §2.4 shows that the streaming driver also runs the graph for a
tool-capable fast turn; §2.5 says the browser polls a run after a reload rather
than resuming the event stream; §2.6 names saved memories as the one authority
inside Qdrant. L5 fixes the stale env-override snippet (`_set`), the service
name (`audrey`, formerly `audrey-ai`), recreate-not-restart after a
config change, and the tool-discovery answer (there is now a background retry).
The course README gained a list of published lessons, which L0 points at.

The approved outline follows.

**L0 — Introduction (rewrite)**

- What Audrey is now: a self-hosted AI application on one home server. A
  browser app, an OpenAI-compatible API for bots and scripts, and the pipeline
  that routes each turn across local and cloud models, tools and the knowledge
  base.
- The pipeline paragraph (fast, deep, research), kept and brought up to date.
- The supporting pieces: fair scheduling; identity (Cloudflare Access for the
  browser, personal tokens for bots); conversations and runs the server owns;
  files, media and the knowledge base; durable side effects; metrics.
- How it was built, briefly, and how the course is organized: its parts named
  by substance, with no lesson numbers beyond the next lesson. The author is
  described without pronouns.

**L1–L3 — client examples**

- L1: the dataclass example becomes `Principal`, the native identity record
  (frozen; durable ids versus mutable profile fields); the httpx bullet drops
  the Open WebUI probe.
- L2: the five mentions of Open WebUI as the client become the native app or
  an API client; the streaming bullet says OpenAI framing serves `/v1` clients
  while the browser receives AG-UI events.
- L3: the httpx example becomes the Cloudflare Access public-key fetch
  (`_fetch_keys` in `identity/cloudflare_access.py`): one GET, a status check,
  JSON, and network errors translated into Audrey's own exception. The
  satellite list and the test notes lose Open WebUI; the closing pointer starts
  the trace at the native app.

**L4 — The platform and the request lifecycle (rewrite)**

Opening question: "You type a question into Audrey and press Send. What runs
where, in what order, until the answer has finished streaming, and how does a
bot calling the API reach the same pipeline?"

1. **Context**
   - 1.1 *From an API behind someone else's app to an application:* why Audrey
     owns identity, conversations and files.
   - 1.2 *The whole-system map:* browser → Cloudflare Access → tunnel →
     `audrey-ui` → `audrey` → Ollama, Qdrant, custom-tools and the SQLite
     stores; the media worker and fetcher pulling jobs from `audrey`; bots on
     `/v1`. Slogan: "The browser renders; Audrey decides; the sidecars do risky
     work in a box."
2. **Read-along**
   - 2.1 *One origin for the browser* (the nginx template): the static app plus
     forwarded `/api` and `/v1`; request-time DNS; unbuffered streams; the
     Access assertion passed through for Audrey to verify. Spotlight:
     same-origin.
   - 2.2 *Two front doors, one pipeline:* the native `/api` routes and the
     OpenAI-compatible `/v1` routes. Who is calling, in two sentences.
   - 2.3 *The pipeline graph, and what a node looks like* (today's §2.2–2.3,
     kept).
   - 2.4 *Two drivers over the same building blocks:* the graph serves
     non-streaming `/v1` requests; the streaming driver serves the native app
     and streaming `/v1`, deciding fast or deep itself.
   - 2.5 *One message, end to end:* the browser posts only the newest message
     to `/api/agent`. Audrey checks the caller, loads the conversation the
     caller owns, checks model access, picks a skill, and reloads history from
     its own database (the browser's transcript is never trusted). It records
     the message and the run in one transaction, launches the run as a task the
     server owns, and streams the run's events back as AG-UI
     (`X-Audrey-Run-ID`; reconnect and Stop by run id). The run enters the
     streaming driver, fast or deep runs, and events reach the browser. Each
     later topic is named by substance.
   - 2.6 *Where data lives, and where to look:* authority versus projection
     (the application and uploads databases and the files on disk are
     authorities; Qdrant and the chat archive are projections: "If you can
     rebuild it, it's a projection"), then the "If you're asking…, look in…"
     table brought up to date.
3. **Comprehension questions** (scenarios): a page refresh mid-answer; a bot
   streaming through `/v1`, and which driver serves it; why the browser sends
   only the newest message; what a wiped Qdrant volume loses; every chat
   failing right after `audrey` is rebuilt; Ollama failing mid-answer.

**L5 — Configuration and startup (refresh)**

- §2.1 becomes *The containers as a trust map:* the five runtime services and
  what each may reach (loopback-only UI, LAN-published API, unpublished tools,
  a media worker with no route out, a fetcher with download egress only),
  read-only mounts and the single writer. Spotlight: Docker networks, and why
  DNS isolation is not network isolation. Then today's env → Python content.
- §2.2 *The config stack:* kept.
- §2.3 *What the lifespan owns now:* a table grouping identity, canonical state
  (the store, migrations, interrupted runs settled), the model layer and
  catalog, tools and skills, the KB stack, durable workers, native runs,
  titles and readiness; shutdown stops producers before the queues they feed.
- §2.4 `app.state`: the list brought up to date.
- §2.5 *Graph closures and rediscover:* kept, plus the guard that keeps the
  live registry when discovery comes back empty.
- §2.6 *The readiness log* becomes the readiness snapshot, briefly, with
  degraded startup named; the details belong to the closing lesson.
- Questions refreshed for the new startup.

**README:** the out-of-scope line about Open WebUI becomes "React internals;
the course covers the browser boundary".

**Not in this cycle:** the Open WebUI mentions in L6–L8 and dropping the
utility-prompt passage (cycle 6); identity in depth (L14); the research
lesson's footer (when L20 ships).

## Cycle 2a — tool policy, Lesson 9, now Lesson 10 (outline approved 2026-10-15; accepted)

Its audit findings were resolved in source the same day (record in the
AUDIT), and the lesson describes the code after those fixes. As written:
the declaration catalogue and failing closed per server (§2.1); discovery,
schema clean-up and availability merged into one section (§2.2); what a
model is offered, including a skill's narrower set (§2.3); the dispatcher's
steps in their real order (§2.6); identity binding, refusal without a
caller identity, and the purge block (§2.7); truncation that drops whole
items and says what was cut (§2.8); parallel calls and the per-worker
search budget (§2.9); compaction by message priority (§2.10). The Context
table now shows deep and research workers taking the GPU gate per model
call, as the fast path does. Five questions, down from ten; 635 lines, down
from 713. Scoped checks: conventions 0, cites 30/30, every snippet line
found in its cited code.

## Replacing the Open WebUI material

Lesson numbers in this section are the ones in use before the 2026-10-23 split.

Open WebUI material is **replaced, not appended to**: after its cycle, a
passage describes the native app (or an API client) as the client. Lessons
never name Open WebUI or say it still exists, and its compatibility leftovers
are not taught (user rule 2026-10-15; the user is removing it from the code).
All of them were removed from the
lessons on 2026-10-23 at the user's request (originally about 110 mentions
across 14 files); the identity-specific ones wait for L14, which
explains the mechanism they need. The lesson numbers in this table are
historical.

| Where | What it says now | Replace with | Cycle |
|---|---|---|---|
| lesson-ai README | "Frontend integration (Open WebUI)" is out of scope | The course covers the browser boundary (L22); React internals stay out of scope | Cycle 1 (done) |
| L0 (2) | Audrey sits behind Open WebUI, which authenticates every request | The native app and API clients; Audrey-owned identity in one sentence | Cycle 1 (done) |
| L1 (1), L2 (5), L5 (1) | Open WebUI named as *the* client in examples | The native app or an API client | Cycle 1 (done) |
| L6 (1), L8 (1) | Open WebUI named as *the* client in examples | The native app or an API client | Done 2026-10-23 (user request) |
| L3 (3 of 11) | "No Open WebUI required"; a trace starting "OWUI sends POST" | Current dependency list; the trace starts at the native app | Cycle 1 (done) |
| L4 (17) | The walk-through starts in Open WebUI; auth asks Open WebUI who owns the token | The native app sends the request, with `/v1` clients as the second door; identity in two sentences (Access assertion or personal token) | Cycle 1 (done) |
| L7 (3) | Open WebUI utility prompts force fast mode | Drop: a compatibility leftover, being removed from the code | Done 2026-10-23 (user request) |
| L3 (8 of 11) | The httpx "real example" is the Open WebUI token check | A current httpx example; the Open WebUI adapter is dormant | Cycle 1 (done) |
| L11 (1), L12 (2) | The user id comes from the Open WebUI session or JWT | The authenticated user's stable id and private namespace (mechanism in L14) | Done 2026-10-23 (user request) |
| L13 §2.1 | Identity via Open WebUI's `/api/v1/auths/` | A recap of L14 | Done 2026-10-23 (user request) |
| L13 §2.3–2.6 | Archive stitching keyed on Open WebUI chat ids | Native conversation ids; `/v1` stitching as compatibility | Done 2026-10-23 (user request) |
| L15 (15) | Virtual models and SSE framed for Open WebUI | The compatibility door for `/v1` clients; the native app via run events | Done 2026-10-23 (user request) |

## Refreshing the existing lessons

Besides the Open WebUI items above, these sections changed underneath their
lessons. Rows are keyed by today's file number; the arrow gives the final one.
Each row is done in the cycle named in the "When" column, without numbered
forward references in lesson prose. Items marked † are stale *snippets or
values* found by the 2026-10-08 cite pass; items marked ‡ were found by the
2026-10-10 snippet check (every snippet line compared with the cited code).

| Lesson | Now stale | When |
|---|---|---|
| L5 (→4) | Startup opens the store and migrates, loads skills and the model catalog, starts durable workers; readiness | Cycle 1 (done); a store pointer with L5 |
| L6 (→8) | Thinking descriptors and per-role thinking, model catalog and direct models, vision sidecar; † the cloud-cap snippet (the cap now logs the dropped worker) | Done in the model-layer split (in review); model access moves to L6 in the identity cycle |
| L8 (→7) | Schema-pinned router without thinking; failed classify no longer escalates. ‡ §2.3 `node_classify` and §2.7 complexity snippets predate skill routing (`routing_messages`) and research (`audrey_research` is forced deep); utility-prompt detection now applies to compatibility requests only | Done 2026-10-23 in the Lessons 1–8 accuracy pass (in review) |
| L9 (→11) | Per-role thinking, shared stage lifecycle, worker-exception handling. ‡ §2.3 pool snippet, §2.4 synthesizer example and question 2 use retired model names | Done 2026-10-23 in the deep and research merge (in review) |
| L10 (→12) | † §2.7 snippet and prose still show `_USER_SCOPED_TOOLS`; user binding is now declared per tool in `TOOL_DECLARATIONS`. † Truncation default is 6000, not 2000, and the result now says what was cut. Compaction counted in tool messages with failures evicted first; per-worker search budget. ‡ §2.8 `_truncate` snippet: the marker now gives shown and total characters and says a retry will not help, and `_truncate_payload` (drops whole list items so JSON stays valid) is tried first. ‡ §2.9 dispatch goes through `_dispatch_observed` after the search budget stubs excess calls | Cycle 2a (done, accepted); takes L11's essentials in the tools cycle |
| L11 (→12, 25) | Client tool calls and results through passthrough; Responses function tools | Essentials into L12 in the tools cycle; caller-executed tools into L25 |
| L12 (→13) | Hybrid retrieval, private-read isolation, resident embedder. ‡ §2.2 ingest snippet lacks the thread offload and sparse vectors; ‡ §2.5 `kb_query` snippet predates caller resolution, file and artifact scope, deleted-file exclusion and the hybrid branch, which is on by default, so §2.6's raw-score merge is the hybrid-off path. The text embedder is now `qwen3-embedding:4b` (one shared profile in `embedding_profile.py`, a query instruction prefix, a recoverable re-embedding job); nomic, 768-d and the 0.53 floor are stale | The knowledge-base cycle (L13) |
| L13 (→13, 19) | † §2.9 upload steps: storage is now reserved atomically and committed after ingest; statuses, tombstones, durable deletion; the upload page is retired. §2.10 the startup reconcile no longer prunes: it backfills missing legacy rows, reports missing index content and never deletes catalog rows, so §2.11's "the next boot's `reconcile_with_qdrant` cleans up the orphan points" is stale too; the embedder change also dates its nomic mentions | Watcher and reconcile in the knowledge-base cycle (§2.11 and the nomic mentions to confirm); the upload flow with L19 |
| L14 | §2.1 identity through Open WebUI; §2.3–2.6 archive capture and stitching. §2.6 the streaming bypass: plain fast streams now recall too, through `_prepare_stream_context`, which `_phase_thinking` also calls; its nomic mention is stale (one Qwen text embedding space for files, references, memories and chat history) | With L6 (§2.1); with L18 (§2.3–2.6) |
| L15 (→10) | Queue grants have explicit owners; cancellation releases slots. ‡ §2.2 step 5: a waiter cancelled after its grant re-releases the slot | The light-refresh cycle |
| L16 (→16, 24) | † §2.3 shows retired `_delta_frame`/`_stop_frame` (now `OpenAIStreamAdapter._frame`); §2.5–2.7 deep stream and cancellation run through `StreamStageRunner` and `OpenAIStreamSession`; typed passthrough outcomes; reasoning controls; a Responses route. ‡ §2.5 error handler and §2.7 cancellation snippets show `pipeline_outcome` and `synth_task.cancel()`: the deep stream now records outcomes on `runner.terminal`, drains owned tasks with `runner.cancel_and_drain()`, and archives through `runner.finalize` | Streaming internals with L16; the rest with L24 |
| L17 (→15) | `list_my_files`, `get_file_text`; † `_USER_SCOPED_TOOLS` prose; capability supervisor; archive outboxes; `kb_search` scope. ‡ §2.4 teaches Brave only, with a 429 becoming 503 and other errors escaping as 500: `web_search` now alternates Brave and SearXNG with cross-fallback, returns 503 only when both fail, and normalizes non-429 errors. ‡ §2.6 the user filter is now the shared `_user_filter` helper, and the memory collection now has scheduled snapshots and restore verification (`memory_backup.py`). §1.2 the file list lacks `capabilities.py`, `searxng.py` and `memory_backup.py`; §1.2 also lacks `text_embedding.py`, and §2.6's nomic mentions are stale (memories embed through the shared Qwen profile) | Cycle 2b (§1.2, §2.1, §2.3, §2.4, §2.6, §2.7); with L18 (§2.8); with L22 (`kb_search` scope); its footer ends the course for now |
| Research mode (was L18) | Folded into L9 §2.5 (final L11) on 2026-10-23; the course-end note moved to L17's footer | Done; the wrap-up moves to L26 |

## Out of scope

- The eval harness, probes and smoke scripts. They are measurement tooling,
  and the course explains production code, not how it is tested.
- Container, Unraid and Cloudflare operations, beyond the trust boundaries
  L4 and L21 need.
- Retired document authoring, and the router assessment that changed nothing.
- React internals (the native run lesson covers the browser boundary only).
- The Python course: paused after Lesson 6; the user will resume it later.

## Process — one lesson per cycle

From `AGENTS.md` "Lesson workflow". Do not skip or merge steps, and do not
start the next cycle until the user has reviewed this one. A refresh cycle runs
the same steps over the sections it rewrites.

1. Confirm the card's scope with the user; it may split or merge.
2. Audit the in-scope files. File findings with severity and `file:line`
   under a new lesson heading in `docs/lesson-ai/AUDIT.md` "Open" (gitignored;
   back it up before editing). Findings and drain decisions are recorded only
   there: this plan and every other tracked doc may point at the AUDIT but
   never list them.
3. Drain this lesson's findings with the user. Code changes need explicit
   approval, even for obvious nits.
4. Propose an outline, including this cycle's refresh and replacement items;
   wait for an explicit go-ahead.
5. Open the most recently published lesson and match its style. Write in the
   Lesson 4+ shape: Context, Read-along, Comprehension questions.
6. Apply this cycle's refresh and Open WebUI replacement items.
7. Run `scripts/lessons/check-lesson-conventions.py`, and the cite checker
   scoped to the changed lessons (a full sweep stays user-directed).
8. Wire it in: the previous lesson's footer, the course README, and the
   lesson count in `AGENTS.md`.
9. Hand it to the user for review.

Hard rules for the writer:

- No "Phase N" or campaign vocabulary in lesson prose. All the source material
  here is phase-tagged, so strip it and describe features by substance.
- No personal names in prose or example values; use `alice@example.com` for
  example emails.
- No exact codebase counts outside `file:line` cites.
- No numbered forward references; say "the next lesson".
- Capitalize after a colon that introduces a clause.
- No test walkthroughs: explain production code.
- No labs (2026-10-13): never ask the reader to open a terminal, install
  anything, create files or run code. Show code and what it prints, and explain
  why.
- At most five comprehension questions (2026-10-15). Every lesson was trimmed
  to five on 2026-10-23.
- Teach the principle, not the war story. The phase docs are full of
  incidents; the lesson explains why the code is shaped the way it is.
- Define every new concept on first use (see the learner profile in `AGENTS.md`).

## Done so far

- **AUDIT untracked.** `.gitignore` named the pre-rename `docs/lessons/` path;
  it now ignores `docs/lesson-ai/AUDIT.md`, and the file was removed from the
  index (kept on disk). Earlier commits still contain it.
- **AUDIT "Open" re-validated** against current source on 2026-10-08 (the
  record is in the AUDIT).
- **Cite sweep.** 270 of 370 cites had drifted. All were re-anchored by
  content, and the checker reported 305 ok, 0 broken, 64 verified soft hints
  and 1 known false positive (recorded in AUDIT "Accepted"). Snippet content
  that changed, rather than moved, is marked † in the refresh table.
- **Strict cite re-check (2026-10-10).** That "305 ok" was overstated. A
  strict re-check and a hand check of every bare `file:line` cite fixed 84 more
  cites and 4 stale snippet values; snippets and prose describing changed
  behavior are marked ‡ in the refresh table.
- **Cite checker tightened (2026-10-10).** It no longer accepts a cite that is
  merely near its anchor: a snippet may sit below the cited line only inside the
  block that opens there or inside the range the cite or its label states, and
  a symbol label must sit on its definition or the decorator above it. The
  full sweep afterwards: 370 cites, 0 drift, 0 broken.
- **Platform audit (2026-10-10).** The review record is in the local AUDIT.
- **Restructure decided (2026-10-10).** See the decisions below.
- **Model layer split (2026-10-23).** L6 rewritten as *choosing a model* and a
  new L7 *calling models* (with Private at a high level); the old 7–17 became
  8–18. All Open WebUI mentions were removed from the lessons the same day, and
  every lesson was cut to five questions.
- **Course structure revised (2026-10-23).** 34 planned lessons became 27, in
  seven parts, with the pipeline in request order and identity and the database
  in the platform part; see the structure section and decisions 5–8. Lesson
  files are unchanged until the first renumber.
- **Lessons 1–8 accuracy pass (2026-10-23).** Each lesson checked against the
  working tree, one at a time, while another session drained the AUDIT in the
  same files. Wrong facts fixed (lint cases, HTTP verbs, schema and metric
  names, per-user isolation, the pipeline order in L2, `ModelSpec` fields);
  model names and volatile values replaced by roles and reasons; war stories
  and terminal instructions removed; L8 refreshed and cut from 707 to 471
  lines; L9 re-checked against the drained code. All cites re-anchored and
  checked. Three new findings are in the AUDIT.
- **Deep and research merge (2026-10-23).** Research mode (L18) became §2.5 of
  L9, rewritten at 723 lines (from 839); L17's footer now ends the course, and
  the planned lessons after research mode moved up one (20–33). Scoped checks:
  conventions 0, cites 52 with 0 drift, every snippet line found in its cited
  code.
- **Cycle 1 accepted (2026-10-15).** The Qwen embedding switch landed the same
  day; L3's model facts were updated and the cites it moved in L4 and L5
  re-anchored (scoped check: 89 cites, 0 drift).

## Decisions (2026-10-23)

1. **Split Lesson 6** by job rather than by model: the model layer I, choosing
   a model (choices and access, config pools, `ModelSpec`, ranking, health),
   and the model layer II, calling models (the Ollama client and thinking, fast
   path, deep panel and its cloud cap, synthesis fallbacks, describing images,
   then Private, then one request end to end). At most five questions each.
2. **Private** is explained at a high level at the end of the second lesson:
   the server-owned local-only policy, local-only tools, restrictions that stick
   to conversations, Projects and files, and what "private" does and does not
   cover. Parts still being built stay out.
3. **Renumber in the split's cycle:** today's 7–17 become 8–18 (file renames,
   "Lesson N" mentions, links, a README map, this plan, `AGENTS.md`). Lesson
   numbers in the table above and the cards below change with it.
4. **Research mode joins deep mode** (user request): it doesn't need its own
   lesson, so it is a section of L9, which gets leaner at the same time. The
   standalone file is retired, nothing after it is renumbered (it was last),
   and the planned lessons that followed it move up one.
5. **Course structure revised** (user-approved, after the Lessons 1–8 pass):
   27 lessons in seven parts, as in the structure table. The pipeline is taught
   in request order (routing, the model layer, fair scheduling, deep and
   research, tools), and identity and the database join the platform part.
6. **Merges:** L11 into tools (L12) and Responses (L25), its vendor tour cut;
   L12 and L13 into one knowledge-base lesson, with the upload flow going to
   files; L3 into L2, with embeddings going to the knowledge base and
   Prometheus to the closing lesson; L16 split between streaming (L16) and the
   bot API (L24). Planned lessons paired: the native run with the browser
   boundary, jobs with media, skills with Projects, Responses I with the bot
   API. The bot API stays late, because Responses needs files and outbound
   requests.
7. **Two scripted renumbers:** when the database and identity lessons enter
   (with the pipeline reorder), and at the end, when the merged files are
   retired. Merges happen in the refresh cycles their lessons need anyway.
8. **Deferred lessons:** Private processing, and the reference libraries with
   the tax review, written once the code settles.

## Decisions (2026-10-10)

1. **The rule:** approved. Rewrite a lesson when new code replaced how its
   subsystem works; write a new lesson only for a new subsystem.
2. **Placement:** the application database and identity lessons are inserted
   as 13 and 14; the cycle that writes 13 renumbers 13–17 to 15–19. The rest
   append as 20–33.
3. **Writing order:** approved, system map first.
4. **Platform audit:** its review record is in the local AUDIT.
5. **Cite checker:** the ten-line tolerance replaced by structural rules; done.
6. **Audit findings are always gitignored:** findings and drain decisions live
   only in gitignored files, and tracked docs, this plan included, point at
   them without listing them.

## Decisions (2026-10-08)

Items 1, 4 and 5 are superseded by the decisions of 2026-10-10 above.

1. **Sequence and numbering:** approved; Lessons 18–34, appended, numbers
   fixed as each ships.
2. **Frontend:** one browser-side lesson (L23) at the boundary, with a
   TypeScript reading primer inside it; React internals out of scope.
3. **Responses API:** two lessons (L32, L33), kept late because bot API work
   is parked.
4. **Refreshes:** paired with the new lessons plus one refresh batch after
   Part 2. Open WebUI material is replaced throughout, per the checklist.
5. **Cite sweep:** done before Lesson 18.
6. **AUDIT:** untracked.
7. **Python course:** paused; to be continued later.
8. **Cadence:** one lesson at a time, reviewed by the user before the next.
