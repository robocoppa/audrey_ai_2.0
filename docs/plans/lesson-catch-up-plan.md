# Plan — catching the maintainer course up with the code

The maintainer course (`docs/lesson-ai/`) stops at research mode. Since then
Audrey has gained its own application, identity, file and media pipeline,
retrieval upgrades, skills, Projects and a second API. This plan sets out the
lessons that cover that work and the refreshes the existing lessons need.

**The structure was decided 2026-10-10** (see [Decisions](#decisions-2026-10-10));
it replaces the sequence of 2026-10-08. Work happens **one cycle at a time**:
each new lesson or refresh unit goes through its own gated cycle (see
[Process](#process--one-lesson-per-cycle)), and the next cycle starts only after
the user has reviewed the finished one.

## The restructure (approved 2026-10-10)

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

Applied:

- **Absorbed into existing lessons:** the platform map (L0, L4, L5); the tool
  declaration catalogue, bounded discovery and truncation (L9, plus the
  sidecar lesson); the hybrid-retrieval core (L11); the search fallback and the
  capability supervisor (sidecar lesson); the readiness snapshot's basics (L5).
- **Inserted after L12:** the application database, then identity and access.
  L12 ends on SQLite and "two stores agree", which the database lesson
  continues into authority versus projection. Identity needs the store
  (principals, tokens). Everything after L12 is per-user (memory, history,
  fairness buckets, `/v1` auth, sidecar scoping), so this is the first point
  where identity is needed in depth.
- **Appended:** everything else. Each of those lessons needs most of Part 2
  (the native run needs L15's streaming; files need L12 and the sidecar;
  outbound requests need `web_fetch`; skills need routing, tools and the file
  tools), so after research mode is their earliest valid place.
- **No longer standalone:** the platform map (absorbed). The readiness lesson
  becomes the closing lesson, because a readiness snapshot reports on
  everything from tools and skills to queues and workers and so can only be
  explained last.

### Numbering

Inserting two lessons after L12 moves 13–17 to 15–19. Inside the course that
means five file renames, about twenty "Lesson N" mentions and thirteen links.
Outside it, live documents (this plan, `AGENTS.md`, `PROJECT_STATE.md`) are
updated, and AUDIT gets a mapping note rather than rewritten history. Campaign
docs and completed plans that name old numbers are history; the course README
gets a one-line old-to-new map. The renumber runs, scripted and checked, in
the cycle that writes the first inserted lesson, so the course never has a gap.


### The course after the restructure (Lessons 0–33)

| # | Lesson | Change |
|---|---|---|
| | **Part 1 — Foundations and the system map** | |
| 0 | Introduction | Rewrite: a self-hosted application with a browser app, a bot API and the pipeline; course map by part |
| 1–2 | Python features; FastAPI, Pydantic, LangGraph | Light: client examples |
| 3 | Satellite libraries | Refresh: a current httpx example; the dependency list |
| 4 | The platform and the request lifecycle | Rewrite, absorbs the platform map: browser, nginx as the single origin, `/api/agent`; the `/v1` door; two drivers (graph and streaming) over one set of nodes; where data lives (authority versus projection); one native message end to end |
| 5 | Configuration and startup | Refresh, absorbs the platform map: Compose as a trust map (networks, who may reach what); what the lifespan owns now; degraded startup; the readiness snapshot that replaced the log |
| | **Part 2 — The answer pipeline** | |
| 6 | The model layer | Refresh: thinking controls, the model catalog and direct models, describing images for text-only models, current pools |
| 7 | Classification and routing | Refresh: schema-pinned router without thinking, short-prompt skip, failed classification no longer escalates, the skill instruction kept out of the gate, utility-prompt detection for compatibility requests only, research as forced deep |
| 8 | Deep mode | Refresh: per-role thinking, the worker role prompt, worker failure handling, current models |
| 9 | Tool use and the ReAct loop | Rewrite §2.1 and §2.7–2.10, absorbs tool policy: the declaration catalogue (undeclared routes refused), declared user binding, bounded concurrent discovery, truncation that says what was cut and keeps JSON valid, the per-worker search budget, personal reads blocked during a purge |
| 10 | How function calling works | Light: caller-executed tools |
| 11 | The knowledge base: ingest and search | Rewrite §2.1–2.2 and §2.5–2.6, absorbs hybrid retrieval: dense plus BM25 sparse vectors, reciprocal rank fusion, the evidence rule, the migration; user versus service callers |
| 12 | The KB lifecycle | Refresh §2.9–2.10: the upload flow as it runs now (reserve, sniff, ingest, commit, statuses) |
| 13 | **The application database** | New, inserted |
| 14 | **Identity and access** | New, inserted |
| 15 | Per-user context: memory and history (was 13) | Refresh: §2.1 becomes a recap; canonical history lives in the database; the archive is a search projection |
| 16 | Fair scheduling (was 14) | Light: grants with explicit owners; a waiter cancelled just after its grant returns the slot |
| 17 | The OpenAI-compatible routes (was 15) | Rewrite §2.3 and §2.5–2.7: stream session and adapter, the stage runner, cancellation that drains owned tasks, typed passthrough outcomes, reasoning controls; framed as the compatibility door |
| 18 | The custom-tools sidecar (was 16) | Refresh: declared tools, the file tools, Brave and SearXNG fallback, the capability supervisor, the archive read side |
| 19 | Research mode (was 17) | Light: footer |
| | **Part 3 — The native application** | |
| 20 | **The native run** | New |
| 21 | **One event spine, several wire formats** | New |
| 22 | **The browser side** | New |
| 23 | **Durable side effects** | New |
| | **Part 4 — Files and media** | |
| 24 | **Files as durable objects** | New |
| 25 | **Background jobs and leases** | New |
| 26 | **From media to text** | New |
| 27 | **Reaching outside safely** | New |
| 28 | **Asking about your files** | New |
| | **Part 5 — Shaping a turn, and the second API** | |
| 29 | **Skills** | New |
| 30 | **Projects** | New |
| 31 | **The Responses API I** | New |
| 32 | **The Responses API II** | New |
| 33 | **Operating Audrey** | New, closing lesson |

### Writing order (one reviewed cycle each)

1. **System map:** rewrite L0, L4 and L5; client examples in L1–L3. Replaces
   the L18 cycle. The platform audit of these files (filed 2026-10-09) has been
   drained; its record is in the course AUDIT.
2. **Tool policy:** L9, plus the sidecar lesson's §2.1, §2.3, §2.4 and §2.7.
3. **Search:** L11, plus L12 §2.9–2.10.
4. **New 13, the application database,** with the 13–17 renumber.
5. **New 14, identity and access,** with the L15 (was 13) recap.
6. **Light refreshes:** L6, L7, L8, L10 and L16 (was 14).
7. **L17 (was 15):** the streaming rewrite.
8. **New 20–23,** one per cycle, each with its paired refreshes.
9. **New 24–28,** one per cycle.
10. **New 29–32,** one per cycle.
11. **New 33,** the closing lesson.

That is 21 reviewed cycles: 5 refresh units and 16 new lessons (one fewer than
the 2026-10-08 sequence, because the platform map is absorbed).

### What stays and what changes

- **Stays:** one lesson or refresh unit per cycle, reviewed before the next;
  the browser lesson at the boundary with a TypeScript primer; two Responses
  lessons kept late; Open WebUI material replaced, not appended; the Python
  course paused.
- **Changes:** decision 1 (sequence and numbering), and decision 4: most
  refreshes become their own early cycles instead of a batch after Part 2.

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
| C3 25, 26, 28: research fact-check, claim ledger, grounding diagnostic | Covered by L17 |
| C3 27: live eval on the box | Out of scope (tooling) |
| C3 29–31: `web_fetch`, SSRF hardening, KB query auth | L16 §2.5 covers the basic page-opener only |
| C3 32–38, 41–42: video transport, jobs, worker, transcript, visual pass, summary, cost, URL ingest, `audrey_video` | None |
| C3 39, 40, 43: hybrid retrieval, file tools and scoped search, per-file coverage | None |
| Between campaigns (Aug 9–20): schema-pinned router, thinking knobs, embedder residency, model lineup | Partly patched into L6, L7, L16 |
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
updates in the same cycle. Numbers are the final ones: lessons 13–17 are
written as 15–19 once the renumber has run. Phase docs are background for the
writer only; lesson prose must not cite phase numbers.

### Inserted after L12

**L13 — The application database** (`lesson-13-the-application-database.md`)

- **Question:** "Where does a conversation live now, and what makes it safe to
  change that storage while real data is in it?"
- **Covers:** SQLite in WAL mode as the authority for accounts, conversations,
  messages, runs, preferences and tokens. Transactions through one store.
  Ordered, additive migrations and the schema version. Typed records and
  repositories, and why routes never write SQL. Owner scoping, where foreign
  and missing return the same 404. Authority versus projection: the search
  archive and Qdrant can be rebuilt from canonical rows. Online backups before
  migrations, and verified restores. The dormant history import as an
  idempotent, owner-bound operation.
- **Files:** `src/audrey/app_state/` (`store.py`, `migrations.py`,
  `records.py`, `repositories.py`, `titles.py`, `history_import.py`),
  `src/audrey/chat_projection.py`, backup/restore parts of
  `src/audrey/admin_cli.py`.
- **Builds on:** L5 (`lifespan`, `app.state`), L12 (two stores agree).
- **Note:** The learner has not met SQL transactions or WAL; teach them from
  scratch, as Lessons 1–3 teach foundations.
- **Updates:** L5 §2.3 (startup now opens the store and runs migrations).

**L14 — Identity and access** (`lesson-14-identity-and-access.md`)

- **Question:** "A request arrives from the browser, a bot with a token, an
  eval script, or the tools sidecar. How does Audrey decide who it is and what
  it may use?"
- **Covers:** Authentication versus authorization. Verifying the signed
  Cloudflare Access assertion. `Principal`: a stable Audrey user id and private
  namespace that an email change cannot rename. Account states (Pending to
  active), roles, and the Bots group. Personal access tokens: hashed secrets,
  `account:read` and `compat:full` scopes, dated or permanent expiry, revocation
  checked on every use, shown once. The internal service token and
  `resolve_kb_caller`. Model access: a catalog filtered by access groups and
  rechecked at run time, because hiding a picker option is not enforcement.
  First-admin bootstrap. The dormant Open WebUI adapter.
- **Files:** `src/audrey/identity/`, `src/audrey/auth.py`,
  `src/audrey/model_catalog.py`, `src/audrey/routes/app/{me,admin,models}.py`,
  bootstrap in `src/audrey/admin_cli.py`, identity and token parts of
  `src/audrey/app_state/store.py`.
- **Builds on:** L2 (dependencies), L4, L13. **Background:** C4 phases 02, 14;
  C3 phase 31.
- **Updates:** L15 §2.1 (was L13) becomes a recap of this lesson, plus the
  identity items in the [replacement checklist](#replacing-the-open-webui-material).

### Part 3 — The native application

**L20 — The native run** (`lesson-20-the-native-run.md`)

- **Question:** "What happens between pressing Send and a finished answer, and
  why does closing the tab not lose it?"
- **Covers:** Conversations and runs as server-owned resources. Persisting the
  user message and run record before streaming. At most one active run per
  conversation. Run ownership that outlives the browser connection. Stop as
  explicit cancellation that drains owned work. Startup settling interrupted
  runs. The bounded, process-local replay buffer and cursor, and the fallback
  to durable reads. Retrying a failed turn. How the native path enters the same
  streaming driver as `/v1`, and what it deliberately skips: utility-prompt
  routing and the compatibility archive hooks (its terminal transaction records
  the archive delivery instead).
- **Files:** `src/audrey/routes/app/runs.py`,
  `src/audrey/routes/app/conversations.py`, run and message parts of
  `src/audrey/app_state/repositories.py`, `src/audrey/pipeline/streaming.py`,
  `src/audrey/conversation_titles.py`.
- **Builds on:** L4, L17 (streaming, cancellation), L13, L14.
- **Updates:** L4's native trace, where anything it summarizes changed.

**L21 — One event spine, several wire formats** (`lesson-21-run-events.md`)

- **Question:** "The browser, a Chat Completions client and a Responses client
  all watch an answer being produced. How does one pipeline speak three
  streaming protocols without them drifting apart?"
- **Covers:** Typed, sequenced, client-neutral run events: stages, text
  deltas, tool calls, sources, model used, usage, terminal outcome. The
  emitter. Adapters that turn the same events into AG-UI, Chat Completions SSE
  and Responses events, and the rule that no adapter parses another's display
  text. Shared terminal state (success, failure, or cancellation with partial
  output) used by fast, deep and research. Run observations: how real tool
  activity becomes Sources, Models and Tool calls without leaking retrieved text.
- **Files:** `src/audrey/pipeline/{run_events,agui,run_observations,streaming}.py`,
  `src/audrey/routes/openai/streaming.py`.
- **Builds on:** L17 (SSE, `asyncio.Queue`, banners), L20.
- **Updates:** none. L17's streaming sections, rewritten in cycle 7, are its
  worked example.

**L22 — The browser side** (`lesson-22-the-browser-side.md`)

- **Question:** "What does the browser own, and how does it talk to Audrey?"
- **Covers:** nginx serving the static build and proxying `/api` and `/v1`
  with request-time DNS; `api.ts` as the typed boundary; the AG-UI transport;
  why the server owns history and the browser only renders; single-request
  versus chunked uploads; how answer details (Sources, Models, Tool calls)
  arrive. The boundary only: React internals stay out of scope.
- **Files:** `web/docker/default.conf.template`, `web/src/api.ts`,
  `web/src/agentTransport.ts`, `web/src/App.tsx`, run and stream parts of
  `web/src/ChatWorkspace.tsx`.
- **Builds on:** L4, L21.
- **Needs:** a short TypeScript reading primer inside the lesson; the learner
  knows neither TypeScript nor React.
- **Updates:** any client framing left in L17.

**L23 — Durable side effects** (`lesson-23-durable-side-effects.md`)

- **Question:** "If the box restarts halfway through archiving a chat or
  deleting someone's data, what guarantees the work finishes, and that deleted
  data never comes back?"
- **Covers:** The transactional outbox: commit the intent with the data,
  deliver later, retry until done. At-least-once delivery made safe by
  idempotency. Archive delivery that never delays the response, and the
  exclusion of utility turns. Repair status. Tombstones that hide data at once
  and prevent resurrection. Current-user export, correction and deletion. The
  account-purge coordinator and its process-local privacy gate.
- **Files:** `src/audrey/pipeline/chat_archive.py`,
  `tools-server/chat_archive.py`, `src/audrey/chat_projection.py`,
  `src/audrey/routes/user_data.py`, `src/audrey/user_data_purge.py`,
  `src/audrey/user_data_visibility.py`.
- **Builds on:** L15 (archive capture), L18 §2.8 (archive read side), L13.
- **Updates:** L15 §2.3–2.6 and L18 §2.8 (delivery is now an outbox).

### Part 4 — Files and media

**L24 — Files as durable objects** (`lesson-24-files.md`)

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
  `src/audrey/kb/extract.py`, sidecar ingest in `src/audrey/kb/ingest.py`.
- **Builds on:** L12 (uploads flow), L23. **Background:** C3 phases 32, 40;
  C4 phases 04, 11.
- **Updates:** L12 §2.9–2.10 pointers (cycle 3 already brought the flow
  itself up to date).

**L25 — Background jobs and leases** (`lesson-25-jobs-and-leases.md`)

- **Question:** "Who picks up a pending video, what happens if that worker
  dies mid-job, and how does Audrey stop two workers processing the same file?"
- **Covers:** A job queue built on the uploads database instead of a queue
  server. Claim, lease, complete, fail, requeue. Refusing a stale lease and
  sweeping expired ones back to pending. Attempt counts, and infrastructure
  failures that do not burn attempts. The service-token routes workers call.
  The worker and fetcher claim loops. Why an uncaught exception in a claim loop
  looks like a slow job, not a crash. Checking stage budgets against the lease.
- **Files:** job-lifecycle parts of `src/audrey/routes/files.py` and
  `src/audrey/kb/uploads_db.py`, `src/audrey/media/{worker,fetcher,service}.py`,
  `require_service` in `src/audrey/auth.py`.
- **Builds on:** L24. **Background:** C3 phases 33, 34, 41.

**L26 — From media to text** (`lesson-26-media-to-text.md`)

- **Question:** "What does 'processing' do to a video, an MP3 or a scanned PDF
  before a model can answer questions about it?"
- **Covers:** Probing and audio extraction with ffprobe and ffmpeg. Whisper
  transcription. Keyframe sampling and the gate that decides which frames earn
  a describe call. Brokered model access: the worker cannot reach Ollama, so
  Audrey describes frames for it through the vision path. OCR one page at a
  time under hard bounds. Bounded, fail-soft summaries. Artifacts (transcript,
  visual notes, summary) ingested as ordinary searchable text. Cost attribution
  (queue, load, prefill, generation) and measuring before optimizing.
- **Files:** `src/audrey/media/{audio,stt,frames,framegate,describe,ocr}.py`,
  `src/audrey/routes/media.py`, `src/audrey/pipeline/vision.py`,
  `src/audrey/pipeline/summarise.py`.
- **Builds on:** L6 (`OllamaClient`), L11 (ingest), L16 (the gate), L25.
  **Background:** C3 phases 35–38; C4 phases 07–09.

**L27 — Reaching outside safely** (`lesson-27-outbound-requests.md`)

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
- **Builds on:** L18 §2.5, L25, L26. **Background:** C3 phases 29, 30, 41;
  C4 phase 13.

**L28 — Asking about your files** (`lesson-28-asking-about-your-files.md`)

- **Question:** "Find where this video mentions the budget": how does Audrey
  search one file, page through it, and still cover every file?
- **Covers:** One scope object applied to both retrievers. Filename
  resolution and its notices. Pooled results with a guaranteed slot per file.
  Deleted-file exclusion. `list_my_files`, `get_file_text` paging and
  `kb_search` filters, declared like every user-scoped tool. `audrey_video` as
  a retrieval specialist. (The fusion core is taught in L11.)
- **Files:** scope and coverage parts of `src/audrey/routes/kb.py`, file-tool
  routes in `tools-server/app.py`, the video virtual model and its skill.
- **Builds on:** L9, L11, L24, L26. **Background:** C3 phases 40, 42, 43.
- **Updates:** L18 (the file tools in the tool list).

### Part 5 — Shaping a turn, and the second API

**L29 — Skills** (`lesson-29-skills.md`)

- **Question:** "What is a skill, how is one chosen (by the user or
  automatically), and how can it change which tools a model may call without
  ever granting new power?"
- **Covers:** Declarative, read-only bundles: `SKILL.md` front matter plus
  bounded resources. Strict loading: no executables, no paths escaping the
  bundle, a content digest. An atomic registry with per-bundle degradation and
  rediscovery. Resolving one skill before the fast/deep branch. Injecting the
  instruction once through the task-role mechanism, outside the complexity
  decision. Tools offered = discovered ∩ platform policy ∩ skill allowlist.
  Deterministic English file-intent rules that abstain when unsure; automatic
  selection is now on by default. Provenance: id, version, digest, reason.
- **Files:** `src/audrey/skills/`, `skills/*/SKILL.md`, skill integration
  points in `src/audrey/pipeline/` and `src/audrey/tools/`,
  `src/audrey/routes/app/capabilities.py`.
- **Builds on:** L7, L9, L28 (file intents choose the document and video
  workflows). **Background:** C4 phase 03.

**L30 — Projects** (`lesson-30-projects.md`)

- **Question:** "When a conversation lives in a Project, what extra context
  reaches the model, and what keeps it from leaking or overriding Audrey's rules?"
- **Covers:** Projects as owner-scoped groupings with instructions and
  references to files, not copies. Snapshotting instructions and Ready file ids
  at run creation. Prompt order (platform, then project, then conversation) and
  exclusion from routing. Bounded retrieval: best hit per file, passage and
  token caps. Filenames and passages treated as untrusted evidence. Direct
  models receive the text context but no tools. Transactional delete and ungroup.
- **Files:** `src/audrey/app_state/projects.py`,
  `src/audrey/routes/app/projects.py`, `src/audrey/project_context.py`.
- **Builds on:** L20, L28, L29. **Background:** C4 phase 15.

**L31 — The Responses API I: one pipeline, a second protocol** (`lesson-31-responses-api.md`)

- **Question:** "Bots can call `/v1/responses` instead of
  `/v1/chat/completions`. What is different about that protocol, and how does
  Audrey support it without a second pipeline?"
- **Covers:** Mapping a Responses request onto the shared generation path.
  Typed streaming events with contiguous sequence numbers and the completed,
  failed and incomplete terminals. Validating before side effects: a 400 before
  any SSE or file fetch. JSON Schema admission and output validation.
- **Files:** `/v1/responses` in `src/audrey/routes/openai/routes.py`,
  `src/audrey/routes/openai/structured_outputs.py`, the Responses adapter in
  `src/audrey/routes/openai/streaming.py`.
- **Builds on:** L10, L17, L21. **Background:** C4 phase 05.

**L32 — The Responses API II: tools, inputs and stored responses** (`lesson-32-responses-tools-and-storage.md`)

- **Question:** "How does a bot hand Audrey its own functions, files and URLs,
  and pick up a conversation it stored earlier?"
- **Covers:** Caller-executed function tools on passthrough models, and
  stateless replay. Owned file references and bounded public URL inputs. Opt-in
  storage and continuation, and why that saves bookkeeping rather than tokens.
  Reasoning-effort validation against a model's advertised thinking values.
- **Files:** `src/audrey/routes/openai/{client_tools,client_tool_generation,file_inputs,response_storage,reasoning}.py`,
  `src/audrey/app_state/responses.py`.
- **Builds on:** L13, L27, L31. **Background:** C4 phases 13, 17B.
- **Updates:** L10 and L17 (pointers; passthrough tool-result turns, reasoning
  controls, unknown fields dropped).

### Closing

**L33 — Operating Audrey** (`lesson-33-operating-audrey.md`)

- **Question:** "When one component is down, how does ordinary chat keep
  working, and how do you see what is wrong and what the models cost?"
- **Covers:** Degraded startup and per-capability degradation, recapped across
  the course. One sanitized readiness snapshot (components, tools, skills,
  queues, workers, gate pressure) for admins and Prometheus. Provider
  telemetry: exactly one terminal outcome per call, and reported token usage
  where a valid zero differs from unknown. Reading the dashboards. The course
  wrap-up, moved from the research lesson's footer. (The tool declaration
  catalogue is taught in L9.)
- **Files:** `src/audrey/readiness.py`, `src/audrey/metrics.py`, usage and
  outcome parts of `src/audrey/models/ollama.py`,
  `src/audrey/routes/app/capabilities.py`.
- **Builds on:** L3 (Prometheus), L5, L9, L18, and the course as a whole.
  **Background:** C4 phases 01, 17A.
- **Updates:** L19's footer (the wrap-up moves here).

## Cycle 1 — the system map: proposed outline (2026-10-10, awaiting go-ahead)

Rewrites L0 and L4, refreshes L5, swaps the client examples in L1–L3, and
updates the course README. The platform audit is drained. This cycle's audit
added one finding about the native entry point (in the AUDIT); its drain
decides how L4 §2.5 orders the steps of a first turn.

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

**Not in this cycle:** the Open WebUI mentions in L6–L8 and the utility-prompt
relabel (cycle 6); identity in depth (L14); the research lesson's footer (when
L20 ships).

## Replacing the Open WebUI material

Open WebUI material is **replaced, not appended to**: after its cycle, a
passage describes the native app (or an API client) as the client. It survives
only where code still carries compatibility behavior, labelled as such. About
110 mentions across 14 files; the identity-specific ones wait for L14, which
explains the mechanism they need. Existing lessons keep their current numbers
here; 13–17 become 15–19 in cycle 4.

| Where | What it says now | Replace with | Cycle |
|---|---|---|---|
| lesson-ai README | "Frontend integration (Open WebUI)" is out of scope | The course covers the browser boundary (L22); React internals stay out of scope | Cycle 1 |
| L0 (2) | Audrey sits behind Open WebUI, which authenticates every request | The native app and API clients; Audrey-owned identity in one sentence | Cycle 1 |
| L1 (1), L2 (5), L5 (1) | Open WebUI named as *the* client in examples | The native app or an API client | Cycle 1 |
| L6 (1), L8 (1) | Open WebUI named as *the* client in examples | The native app or an API client | Cycle 6 |
| L3 (3 of 11) | "No Open WebUI required"; a trace starting "OWUI sends POST" | Current dependency list; the trace starts at the native app | Cycle 1 |
| L4 (17) | The walk-through starts in Open WebUI; auth asks Open WebUI who owns the token | The native app sends the request, with `/v1` clients as the second door; identity in two sentences (Access assertion or personal token) | Cycle 1 |
| L7 (3) | Open WebUI utility prompts force fast mode | Keep, relabelled: applies to `/v1` compatibility requests; native requests skip it | Cycle 6 |
| L3 (8 of 11) | The httpx "real example" is the Open WebUI token check | A current httpx example; the Open WebUI adapter is dormant | Cycle 1 |
| L11 (1), L12 (2) | The user id comes from the Open WebUI session or JWT | The authenticated user's stable id and private namespace (mechanism in L14) | Cycle 3 |
| L13 §2.1 | Identity via Open WebUI's `/api/v1/auths/` | A recap of L14 | With L14 |
| L13 §2.3–2.6 | Archive stitching keyed on Open WebUI chat ids | Native conversation ids; `/v1` stitching as compatibility | With L23 |
| L15 (15) | Virtual models and SSE framed for Open WebUI | The compatibility door for `/v1` clients; the native app via run events | Cycle 7, then L21–L22 |

## Refreshing Lessons 0–17

Besides the Open WebUI items above, these sections changed underneath their
lessons. Each row is done in the cycle that rewrites its lesson (the "When"
column; cycle numbers are from the writing order), without numbered forward
references. Items marked † are stale *snippets or values* found by the
2026-10-08 cite pass; items marked ‡ were found by the 2026-10-10 snippet check
(every snippet line compared with the cited code).

| Lesson | Now stale | When |
|---|---|---|
| L5 | Startup opens the store and migrates, loads skills and the model catalog, starts durable workers; readiness | Cycle 1; a store pointer with L13 |
| L6 | Thinking descriptors and per-role thinking, model catalog and direct models, vision sidecar; † the cloud-cap snippet (the cap now logs the dropped worker) | Cycle 6 |
| L7 | Schema-pinned router without thinking; failed classify no longer escalates. ‡ §2.3 `node_classify` and §2.7 complexity snippets predate skill routing (`routing_messages`) and research (`audrey_research` is forced deep); utility-prompt detection now applies to compatibility requests only | Cycle 6 |
| L8 | Per-role thinking, shared stage lifecycle, worker-exception handling. ‡ §2.3 pool snippet, §2.4 synthesizer example and question 3 use retired model names | Cycle 6 |
| L9 | † §2.7 snippet and prose still show `_USER_SCOPED_TOOLS`; user binding is now declared per tool in `TOOL_DECLARATIONS`. † Truncation default is 6000, not 2000, and the result now says what was cut. Compaction counted in tool messages with failures evicted first; per-worker search budget. ‡ §2.8 `_truncate` snippet: the marker now gives shown and total characters and says a retry will not help, and `_truncate_payload` (drops whole list items so JSON stays valid) is tried first. ‡ §2.9 dispatch goes through `_dispatch_observed` after the search budget stubs excess calls | Cycle 2 |
| L10 | Client tool calls and results through passthrough; Responses function tools | Cycle 6; with L32 |
| L11 | Hybrid retrieval, private-read isolation, resident embedder. ‡ §2.2 ingest snippet lacks the thread offload and sparse vectors; ‡ §2.5 `kb_query` snippet predates caller resolution, file and artifact scope, deleted-file exclusion and the hybrid branch, which is on by default, so §2.6's raw-score merge is the hybrid-off path | Cycle 3 |
| L12 | † §2.9 upload steps: storage is now reserved atomically and committed after ingest; statuses, tombstones, durable deletion; the upload page is retired | Cycle 3 |
| L13 (→15) | §2.1 identity through Open WebUI; §2.3–2.6 archive capture and stitching | With L14 (§2.1); with L23 (§2.3–2.6) |
| L14 (→16) | Queue grants have explicit owners; cancellation releases slots. ‡ §2.2 step 5: a waiter cancelled after its grant re-releases the slot | Cycle 6 |
| L15 (→17) | † §2.3 shows retired `_delta_frame`/`_stop_frame` (now `OpenAIStreamAdapter._frame`); §2.5–2.7 deep stream and cancellation run through `StreamStageRunner` and `OpenAIStreamSession`; typed passthrough outcomes; reasoning controls; a Responses route. ‡ §2.5 error handler and §2.7 cancellation snippets show `pipeline_outcome` and `synth_task.cancel()`: the deep stream now records outcomes on `runner.terminal`, drains owned tasks with `runner.cancel_and_drain()`, and archives through `runner.finalize` | Cycle 7 |
| L16 (→18) | `list_my_files`, `get_file_text`; † `_USER_SCOPED_TOOLS` prose; capability supervisor; archive outboxes; `kb_search` scope. ‡ §2.4 teaches Brave only, with a 429 becoming 503 and other errors escaping as 500: `web_search` now alternates Brave and SearXNG with cross-fallback, returns 503 only when both fail, and normalizes non-429 errors. ‡ §2.6 the user filter is now the shared `_user_filter` helper | Cycle 2 (§2.1, §2.3, §2.4, §2.6, §2.7); with L23 (§2.8); with L28 (`kb_search` scope) |
| L17 (→19) | Footer ends the course | With L20 (points onward); the wrap-up moves to L33 |

## Out of scope

- The eval harness, probes and smoke scripts. They are measurement tooling,
  and the course explains production code, not how it is tested.
- Container, Unraid and Cloudflare operations, beyond the trust boundaries
  L5 and L27 need.
- Retired document authoring, and the router assessment that changed nothing.
- React internals (the browser lesson covers the boundary only).
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
- **Platform audit drained (2026-10-10)** by the user; the record is in the
  AUDIT.
- **Restructure decided (2026-10-10).** See the decisions below.

## Decisions (2026-10-10)

1. **The rule:** approved. Rewrite a lesson when new code replaced how its
   subsystem works; write a new lesson only for a new subsystem.
2. **Placement:** the application database and identity lessons are inserted
   as 13 and 14; the cycle that writes 13 renumbers 13–17 to 15–19. The rest
   append as 20–33.
3. **Writing order:** approved, system map first.
4. **Platform audit:** drained by the user (record in the AUDIT).
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
