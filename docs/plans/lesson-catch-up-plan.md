# Plan — catching the maintainer course up with the code

The maintainer course (`docs/lesson-ai/`) stops at research mode. Since then
Audrey has gained its own application, identity, file and media pipeline,
retrieval upgrades, skills, Projects and a second API. This plan sets out the
lessons that cover that work and the refreshes the existing lessons need.

**Decisions were made 2026-10-08** (see [Decisions](#decisions-2026-10-08)).
Lessons are written **one at a time**: each goes through its own gated cycle
(see [Process](#process--one-lesson-per-cycle)), and the next cycle starts only
after the user has reviewed the finished lesson.

**A restructure was proposed 2026-10-10 and awaits a decision** (next
section). If approved, it replaces the 2026-10-08 sequence under
[Lessons](#lessons).

## Proposed restructure (2026-10-10, awaiting decision)

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

Alternative, if numbers should never move: call the inserts 12a and 12b.

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

### New lessons in the restructure

Cards below give the question, the coverage, the files in scope and the
prerequisites. Where a lesson matches a 2026-10-08 card under
[Lessons](#lessons), only the differences are listed.

- **13 — The application database.** As the old L19 card. Builds on L5 and
  L12 §2.10.
- **14 — Identity and access.** As the old L20 card. Builds on L2, L4 and 13.
  In the same cycle: L15 §2.1 (was L13) becomes a recap, and the identity rows
  of the replacement checklist are done.
- **20 — The native run.** As the old L21 card. Builds on L4, 13, 14 and 17.
  Adds what the native path skips: utility-prompt routing and the
  compatibility archive hooks (its terminal transaction records the archive
  delivery instead).
- **21 — One event spine, several wire formats.** As the old L22 card. Builds
  on 17 and 20. L17's streaming sections, rewritten in cycle 7, become the
  worked example.
- **22 — The browser side.** As the old L23 card. Builds on L4 and 21.
- **23 — Durable side effects.** As the old L24 card. Builds on 13, 15, 18
  §2.8 and 20. Updates L15 §2.3–2.6 and L18 §2.8.
- **24–27 — Files, jobs and leases, media to text, reaching outside safely.**
  As the old L25–L28 cards, prerequisites renumbered.
- **28 — Asking about your files.** "Find where this video mentions the
  budget": searching one file, paging through it, and still covering every
  file. One scope object for both retrievers; filename resolution and its
  notices; pooled results with a guaranteed slot per file; deleted-file
  exclusion; `list_my_files`, `get_file_text` paging and `kb_search` filters,
  declared like every user-scoped tool; `audrey_video` as a retrieval
  specialist. (The fusion core moved to L11.) Files: scope and coverage parts
  of `src/audrey/routes/kb.py`, file-tool routes in `tools-server/app.py`, the
  video virtual model. Builds on L9, L11, 24 and 26.
- **29–32 — Skills, Projects, the Responses API I and II.** As the old L30–L33
  cards. Skills now build on 28 (file intents choose the document and video
  workflows).
- **33 — Operating Audrey.** "When one component is down, how does ordinary
  chat keep working, and how do you see what is wrong and what the models
  cost?" Degraded startup and per-capability degradation, recapped across the
  course; one sanitized readiness snapshot (components, tools, skills, queues,
  workers, gate pressure) for admins and Prometheus; provider telemetry (one
  terminal outcome per call, usage where zero differs from unknown); reading
  the dashboards; the course wrap-up, moved from the research lesson's
  footer. The declaration catalogue moved to L9. Files: `src/audrey/readiness.py`,
  `src/audrey/metrics.py`, usage and outcome parts of
  `src/audrey/models/ollama.py`, `src/audrey/routes/app/capabilities.py`.

### Writing order (one reviewed cycle each)

1. **System map:** rewrite L0, L4 and L5; client examples in L1–L3. Replaces
   the L18 cycle. The audit filed for L18 on 2026-10-09 covers the same files;
   its approved fixes are implemented (October 10).
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

### Decisions needed

1. The rule: rewrite replaced subsystems in place; new lessons only for new ones.
2. Placement: insert 13 and 14 with the renumber (recommended), insert as
   12a and 12b, or append as before.
3. The writing order above, system map first.
4. The cite checker's tolerance (`AUDIT.md`, course tooling): a change to a
   repository script, so it needs approval.

## Lesson 18 audit decisions — October 10, 2026

The user approved the platform audit fixes. Seven are implemented; two findings
are accepted as intentional behavior. This closes the audit prerequisite for
the platform lesson; outline/restructure approval is still required before prose.
Tower deployment and browser acceptance remain pending.

| Finding | Decision |
|---|---|
| Tool catalog authentication | Require an active account; retain token scopes |
| Empty tool rediscovery | Return 503; preserve the existing registry and skills |
| Entrypoint description | Describe the current service lifecycle |
| Compose comments | Describe current services, media jobs, and isolation |
| Missing custom-tools icon | Use the tracked image |
| Legacy upload page | Remove page, router, unused setting, and obsolete tests |
| Duplicate cache policy | Preserve upstream policy; add a fallback only when absent |
| Repeated scoping audit | Retain the defensive check |
| API schema proxy | Retain for API discovery; protected routes still require auth |

The separate cite-checker proposal and older planner/polling findings remain
undecided; they are not part of this implementation.

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

## Lessons

**Appended as Lessons 18–34; Lessons 0–17 are not renumbered.** The
[research-mode plan](lesson-research-mode-plan.md#placement--new-lesson-17-no-renumbering)
explains why renumbering is the most disruptive edit in this course. Numbers
below are provisional until a lesson ships; once published, they are fixed.

The order follows dependencies: each lesson needs only those above it. Part 2
(L18–L24) is the core a solo maintainer needs first: data safety, security,
and the main chat path.

| # | Lesson | Part |
|---|---|---|
| 18 | The platform map: Audrey after Open WebUI | 2 — Audrey as an application |
| 19 | The application database | 2 |
| 20 | Identity and access | 2 |
| 21 | The native run | 2 |
| 22 | One event spine, several wire formats | 2 |
| 23 | The browser side | 2 |
| 24 | Durable side effects: outboxes, the archive, personal data | 2 |
| 25 | Files as durable objects | 3 — Files, media and retrieval |
| 26 | Background jobs and leases | 3 |
| 27 | From media to text | 3 |
| 28 | Reaching outside safely | 3 |
| 29 | Hybrid retrieval and file-scoped search | 3 |
| 30 | Skills | 4 — Shaping a turn, and running it |
| 31 | Projects | 4 |
| 32 | The Responses API I: one pipeline, a second protocol | 4 |
| 33 | The Responses API II: tools, inputs and stored responses | 4 |
| 34 | Readiness, degradation and telemetry | 4 |

Each card gives the learner's question, what the lesson covers, the files in
scope for its audit pass, what it builds on, and which older sections it
updates in the same cycle. Phase docs are background for the writer only;
lesson prose must not cite phase numbers.

### Part 2 — Audrey as an application

**L18 — The platform map: Audrey after Open WebUI** (`lesson-18-the-platform-map.md`)

- **Question:** "Open WebUI is gone. What are Audrey's pieces now, where does
  each kind of data live, and what path does one message from the browser take?"
- **Covers:** The Compose services read as a trust map, not as Docker:
  `audrey-ui` serves the client and proxies same-origin `/api` and `/v1`;
  `audrey` is API-only; `media-worker` reaches neither the internet nor Ollama;
  `media-fetcher` has download egress only. Two front doors (native `/api`, and
  `/v1` for Chat Completions and Responses) feed one pipeline. Which store is
  authoritative and which is a rebuildable projection: application SQLite,
  uploads SQLite, Qdrant, the chat archive. One shallow trace of a native
  message from Send to the settled answer, which later lessons deepen.
- **Files:** `compose.yaml` (as a map), `web/docker/default.conf.template`,
  `src/audrey/main.py` (routers and lifespan additions),
  `src/audrey/routes/app/__init__.py`.
- **Builds on:** L4, L5. **Background:** C4 phase 02.
- **Updates:** the course README, L0's course map, L17's footer, and every
  item assigned to L18 in the
  [replacement checklist](#replacing-the-open-webui-material).

**L19 — The application database** (`lesson-19-the-application-database.md`)

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

**L20 — Identity and access** (`lesson-20-identity-and-access.md`)

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
- **Builds on:** L13 §2.1, L19. **Background:** C4 phases 02, 14; C3 phase 31.
- **Updates:** L13 §2.1 rewritten around Audrey principals, plus the identity
  items in the [replacement checklist](#replacing-the-open-webui-material).

**L21 — The native run** (`lesson-21-the-native-run.md`)

- **Question:** "What happens between pressing Send and a finished answer, and
  why does closing the tab not lose it?"
- **Covers:** Conversations and runs as server-owned resources. Persisting the
  user message and run record before streaming. At most one active run per
  conversation. Run ownership that outlives the browser connection. Stop as
  explicit cancellation that drains owned work. Startup settling interrupted
  runs. The bounded, process-local replay buffer and cursor, and the fallback
  to durable reads. Retrying a failed turn. How the native path enters the same
  graph as `/v1`, and what it deliberately excludes (utility-prompt routing).
- **Files:** `src/audrey/routes/app/runs.py`,
  `src/audrey/routes/app/conversations.py`, run and message parts of
  `src/audrey/app_state/repositories.py`, `src/audrey/pipeline/streaming.py`,
  `src/audrey/conversation_titles.py`.
- **Builds on:** L4, L15 (streaming, cancellation), L19, L20.
- **Updates:** L4 (point the walk-through's native path at this lesson's
  substance, without a numbered forward reference).

**L22 — One event spine, several wire formats** (`lesson-22-run-events.md`)

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
- **Builds on:** L15 (SSE, `asyncio.Queue`, banners), L21.
- **Updates:** L15's streaming sections (see the refresh table).

**L23 — The browser side** (`lesson-23-the-browser-side.md`)

- **Question:** "What does the browser own, and how does it talk to Audrey?"
- **Covers:** nginx serving the static build and proxying `/api` and `/v1`
  with request-time DNS; `api.ts` as the typed boundary; the AG-UI transport;
  why the server owns history and the browser only renders; single-request
  versus chunked uploads; how answer details (Sources, Models, Tool calls)
  arrive. The boundary only: React internals stay out of scope.
- **Files:** `web/docker/default.conf.template`, `web/src/api.ts`,
  `web/src/agentTransport.ts`, `web/src/App.tsx`, run and stream parts of
  `web/src/ChatWorkspace.tsx`.
- **Builds on:** L18, L22.
- **Needs:** a short TypeScript reading primer inside the lesson; the learner
  knows neither TypeScript nor React.
- **Updates:** L15's client framing (with L22's items).

**L24 — Durable side effects** (`lesson-24-durable-side-effects.md`)

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
- **Builds on:** L13 (archive capture), L16 §2.8 (archive read side), L19.
- **Updates:** L13 §2.3–2.6 and L16 §2.8 (delivery is now an outbox).

### Part 3 — Files, media and retrieval

**L25 — Files as durable objects** (`lesson-25-files.md`)

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
- **Builds on:** L12 (uploads flow), L24. **Background:** C3 phases 32, 40;
  C4 phases 04, 11.
- **Updates:** L12 §2.9–2.10.

**L26 — Background jobs and leases** (`lesson-26-jobs-and-leases.md`)

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
- **Builds on:** L25. **Background:** C3 phases 33, 34, 41.

**L27 — From media to text** (`lesson-27-media-to-text.md`)

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
- **Builds on:** L6 (`OllamaClient`), L11 (ingest), L26.
  **Background:** C3 phases 35–38; C4 phases 07–09.

**L28 — Reaching outside safely** (`lesson-28-outbound-requests.md`)

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
- **Builds on:** L16 §2.5, L26. **Background:** C3 phases 29, 30, 41;
  C4 phase 13.

**L29 — Hybrid retrieval and file-scoped search** (`lesson-29-hybrid-retrieval.md`)

- **Question:** "Why did a six-word quote from a transcript return nothing
  while a vague paraphrase found it, and what changed so both work?"
- **Covers:** Lexical versus semantic retrieval. BM25 sparse vectors beside
  dense vectors in Qdrant. Reciprocal rank fusion, and the evidence rule that
  replaced a single score floor. The migration that added sparse vectors to
  existing points. Scoping a search to one file or artifact with one scope
  object for both retrievers. Pooled results and a guaranteed slot per file.
  The file tools (`list_my_files`, `get_file_text`, `kb_search` filters) and
  how their user binding is declared. `audrey_video` as a retrieval specialist.
- **Files:** `src/audrey/kb/{bm25,fusion,qdrant}.py`, `src/audrey/routes/kb.py`,
  `scripts/ops/migrate_bm25.py`, file-tool routes in `tools-server/app.py`,
  `TOOL_DECLARATIONS` in `src/audrey/tools/discovery.py`.
- **Builds on:** L11, L16, L25. **Background:** C3 phases 39, 40, 42, 43.
- **Updates:** L11 (pointer; private-read isolation), L16 (tool list).

### Part 4 — Shaping a turn, and running it

**L30 — Skills** (`lesson-30-skills.md`)

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
- **Builds on:** L7, L9. **Background:** C4 phase 03.

**L31 — Projects** (`lesson-31-projects.md`)

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
- **Builds on:** L21, L29, L30. **Background:** C4 phase 15.

**L32 — The Responses API I: one pipeline, a second protocol** (`lesson-32-responses-api.md`)

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
- **Builds on:** L10, L15, L22. **Background:** C4 phase 05.

**L33 — The Responses API II: tools, inputs and stored responses** (`lesson-33-responses-tools-and-storage.md`)

- **Question:** "How does a bot hand Audrey its own functions, files and URLs,
  and pick up a conversation it stored earlier?"
- **Covers:** Caller-executed function tools on passthrough models, and
  stateless replay. Owned file references and bounded public URL inputs. Opt-in
  storage and continuation, and why that saves bookkeeping rather than tokens.
  Reasoning-effort validation against a model's advertised thinking values.
- **Files:** `src/audrey/routes/openai/{client_tools,client_tool_generation,file_inputs,response_storage,reasoning}.py`,
  `src/audrey/app_state/responses.py`.
- **Builds on:** L19, L28, L32. **Background:** C4 phases 13, 17B.
- **Updates:** L10 and L15 (pointers; passthrough tool-result turns, reasoning
  controls, unknown fields dropped).

**L34 — Readiness, degradation and telemetry** (`lesson-34-readiness-and-telemetry.md`)

- **Question:** "When one component is down, how does Audrey keep ordinary
  chat working, and how do you see what is wrong and what the models cost?"
- **Covers:** The tool declaration catalogue validated at discovery: a
  model-visible route that is not declared is refused. Bounded discovery and
  rediscovery. Per-capability degradation inside custom-tools. Degraded startup
  instead of refusing to boot. One sanitized readiness snapshot for admins and
  Prometheus: components, tools, skills, queues, workers, gate pressure.
  Provider telemetry: exactly one terminal outcome per call, and reported token
  usage where a valid zero differs from unknown. Reading the model dashboard.
- **Files:** `src/audrey/readiness.py`, `tools-server/capabilities.py`,
  `src/audrey/tools/discovery.py`, `src/audrey/metrics.py`, usage and outcome
  parts of `src/audrey/models/ollama.py`, `src/audrey/routes/app/capabilities.py`.
- **Builds on:** L3 (Prometheus), L5, L9, L16. **Background:** C4 phases 01, 17A.
- **Updates:** L5 §2.6 (the readiness log became a snapshot), L9, L16.

## Lesson 18 — proposed outline (2026-10-09, awaiting go-ahead)

The Lesson 18 audit is closed in source (October 10). Writing still awaits
approval of this outline or the proposed restructure above. Target length is
that of Lesson 4 (a map, not a deep dive).

**Opening question:** "Open WebUI is gone. What are Audrey's pieces now, where
does each kind of data live, and what path does one message from the browser
take?"

1. **Context**
   - 1.1 *From an API behind someone else's app to an application.* Why Audrey
     now owns identity, conversations and files: history that cannot be lost
     with another app's database, one identity for browser and bots, file
     workflows and provenance. Lessons 4–17 still describe the pipeline core;
     this lesson redraws everything around it.
   - 1.2 *The whole-system map:* browser → Cloudflare Access → cloudflared →
     `audrey-ui` → `audrey` → Ollama, Qdrant, custom-tools and two SQLite
     databases; the media worker and fetcher pulling jobs from `audrey`; bots
     and evals calling `/v1` directly. Slogan: "The browser renders; Audrey
     decides; the sidecars do risky work in a box."
2. **Read-along**
   - 2.1 *The containers as a trust map* (`compose.yaml`): what each service
     may reach, and why. Loopback-only UI, LAN-published API, unpublished
     tools, a worker with no route out, a fetcher with egress but no Ollama
     address. Spotlight: Docker networks, and why DNS isolation is not network
     isolation (`internal: true`). Read-only mounts and the single writer.
   - 2.2 *One origin for the browser* (the nginx template): static files plus
     forwarded `/api` and `/v1`; request-time DNS (containers get new
     addresses); unbuffered streaming; same-origin security headers; the
     Access assertion passed through for Audrey to verify, never trusted by
     the proxy. Spotlight: same-origin, and why it simplifies auth and CSP.
   - 2.3 *Two front doors, one pipeline* (the routers in `main.py` and
     `routes/app/__init__.py`): native `/api` resources versus `/v1`
     compatibility and service routes; health and metrics at the root.
   - 2.4 *What the lifespan owns now* (`main.py`): a table grouping identity,
     canonical state, model layer, tools and skills, KB stack, durable
     workers, native runs, titles and readiness, against Lesson 5's
     `app.state`; why shutdown stops producers before the queues they feed.
   - 2.5 *Where data lives:* authority versus projection. Application SQLite,
     uploads SQLite and the files on disk are authorities; Qdrant and the chat
     archive are projections. Slogan: "If you can rebuild it, it's a projection."
   - 2.6 *One message, end to end:* the browser posts only the newest user
     message to `/api/agent`; Audrey verifies the caller, records the message
     and run before streaming, runs the same graph, and streams typed events
     back unbuffered (`X-Audrey-Run-ID`, reconnect, Stop). Each later topic is
     named by substance, not lesson number.
3. **Comprehension questions** (scenarios): every chat 502s after `audrey` is
   rebuilt; a media job fails to reach Ollama, so should the worker join
   `ollama-net`?; an unauthenticated LAN call to `/v1/kb/query`; what a wiped
   Qdrant volume loses; a restart mid-answer; why the public entry point holds
   no Linux capabilities.

**This cycle also updates:** the course README (out-of-scope line, Part 2 in
the map); L0's framing; L17's closing section (kept as the Part 1 recap,
ending with a pointer to the next lesson); and the L18 rows of the
replacement checklist below (L1, L2, L3, L4, L5, L6, L7, L8).

## Replacing the Open WebUI material

Open WebUI material is **replaced, not appended to**: after its cycle, a
passage describes the native app (or an API client) as the client. It survives
only where code still carries compatibility behavior, labelled as such. About
110 mentions across 14 files; the identity-specific ones wait for L20, which
explains the mechanism they need.

| Where | What it says now | Replace with | Cycle |
|---|---|---|---|
| lesson-ai README | "Frontend integration (Open WebUI)" is out of scope | The course covers the browser boundary (L23); React internals stay out of scope | L18 |
| L0 (2) | Audrey sits behind Open WebUI, which authenticates every request | The native app and API clients; Audrey-owned identity in one sentence | L18 |
| L1 (1), L2 (5), L5 (1), L6 (1), L8 (1) | Open WebUI named as *the* client in examples | The native app or an API client | L18 |
| L3 (3 of 11) | "No Open WebUI required"; a trace starting "OWUI sends POST" | Current dependency list; the trace starts at the native app | L18 |
| L4 (17) | The walk-through starts in Open WebUI; auth asks Open WebUI who owns the token | The native app sends the request, with `/v1` clients as the second door; identity in two sentences (Access assertion or personal token) | L18 |
| L7 (3) | Open WebUI utility prompts force fast mode | Keep, relabelled: applies to `/v1` compatibility requests; native requests skip it | L18 |
| L3 (8 of 11) | The httpx "real example" is the Open WebUI token check | A current httpx example; the Open WebUI adapter is dormant | L20 |
| L11 (1), L12 (2) | The user id comes from the Open WebUI session or JWT | The authenticated Audrey principal and its private namespace | L20 |
| L13 §2.1 | Identity via Open WebUI's `/api/v1/auths/` | Rewritten around Audrey principals | L20 |
| L13 §2.3–2.6 | Archive stitching keyed on Open WebUI chat ids | Native conversation ids; `/v1` stitching as compatibility | L24 |
| L15 (15) | Virtual models and SSE framed for Open WebUI | The native app via run events; `/v1` clients | L22, L23 |

## Refreshing Lessons 0–17

Besides the Open WebUI items above, these sections changed underneath their
lessons. Most ride along with the new lesson that replaces the old picture.
The **refresh batch** holds lessons whose subject changed but which no new
lesson replaces; do it after Part 2, without numbered forward references.
Items marked † are stale *snippets or values* found by the 2026-10-08 cite
pass: the cites were re-anchored, but the shown code or number is out of date.
Items marked ‡ were found by the 2026-10-10 snippet check (every snippet line
compared with the cited code). The "When" column is the 2026-10-08 plan; under
the proposed restructure each row moves to the cycle that rewrites its lesson.

| Lesson | Now stale | When |
|---|---|---|
| L5 | Startup opens the store and migrates, loads skills and the model catalog, starts durable workers; readiness | L19, L34 |
| L6 | Thinking descriptors and per-role thinking, model catalog and direct models, vision sidecar; † the cloud-cap snippet (the cap now logs the dropped worker) | Refresh batch |
| L7 | Schema-pinned router without thinking; failed classify no longer escalates. ‡ §2.3 `node_classify` and §2.7 complexity snippets predate skill routing (`routing_messages`) and research (`audrey_research` is forced deep); utility-prompt detection now applies to compatibility requests only | Refresh batch |
| L8 | Per-role thinking, shared stage lifecycle, worker-exception handling. ‡ §2.3 pool snippet, §2.4 synthesizer example and question 3 use retired model names | Refresh batch |
| L9 | † §2.7 snippet and prose still show `_USER_SCOPED_TOOLS`; user binding is now declared per tool in `TOOL_DECLARATIONS`. † Truncation default is 6000, not 2000, and the result now says what was cut. Compaction counted in tool messages with failures evicted first; per-worker search budget. ‡ §2.8 `_truncate` snippet: the marker now gives shown and total characters and says a retry will not help, and `_truncate_payload` (drops whole list items so JSON stays valid) is tried first. ‡ §2.9 dispatch goes through `_dispatch_observed` after the search budget stubs excess calls | Refresh batch, L34 |
| L10 | Client tool calls and results through passthrough; Responses function tools | L33 |
| L11 | Hybrid retrieval, private-read isolation, resident embedder. ‡ §2.2 ingest snippet lacks the thread offload and sparse vectors; ‡ §2.5 `kb_query` snippet predates caller resolution, file and artifact scope, deleted-file exclusion and the hybrid branch, which is on by default, so §2.6's raw-score merge is the hybrid-off path | L29 |
| L12 | † §2.9 upload steps: storage is now reserved atomically and committed after ingest; statuses, tombstones, durable deletion | L25 |
| L14 | Queue grants have explicit owners; cancellation releases slots. ‡ §2.2 step 5: a waiter cancelled after its grant re-releases the slot | Refresh batch |
| L15 | † §2.3 shows retired `_delta_frame`/`_stop_frame` (now `OpenAIStreamAdapter._frame`); §2.5–2.7 deep stream and cancellation run through `StreamStageRunner` and `OpenAIStreamSession`; typed passthrough outcomes; reasoning controls; a Responses route. ‡ §2.5 error handler and §2.7 cancellation snippets show `pipeline_outcome` and `synth_task.cancel()`: the deep stream now records outcomes on `runner.terminal`, drains owned tasks with `runner.cancel_and_drain()`, and archives through `runner.finalize` | L22, L33 |
| L16 | `list_my_files`, `get_file_text`; † `_USER_SCOPED_TOOLS` prose; capability supervisor; archive outboxes; `kb_search` scope. ‡ §2.4 teaches Brave only, with a 429 becoming 503 and other errors escaping as 500: `web_search` now alternates Brave and SearXNG with cross-fallback, returns 503 only when both fail, and normalizes non-429 errors. ‡ §2.6 the user filter is now the shared `_user_filter` helper | L24, L29, L34 |
| L17 | Footer ends the course | L18 |

## Out of scope

- The eval harness, probes and smoke scripts. They are measurement tooling,
  and the course explains production code, not how it is tested.
- Container, Unraid and Cloudflare operations, beyond the trust boundaries
  L18 and L28 need.
- Retired document authoring, and the router assessment that changed nothing.
- React internals (the browser lesson covers the boundary only).
- The Python course: paused after Lesson 6; the user will resume it later.

## Process — one lesson per cycle

From `AGENTS.md` "Lesson workflow". Do not skip or merge steps, and do not
start the next lesson's cycle until the user has reviewed this one.

1. Confirm the card's scope with the user; it may split or merge.
2. Audit the in-scope files. File findings with severity and `file:line`
   under a new lesson heading in `docs/lesson-ai/AUDIT.md` "Open" (gitignored;
   back it up before editing).
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

## Done before Lesson 18 (2026-10-08)

- **AUDIT untracked.** `.gitignore` named the pre-rename `docs/lessons/` path;
  it now ignores `docs/lesson-ai/AUDIT.md`, and the file was removed from the
  index (kept on disk). Earlier commits still contain it.
- **AUDIT "Open" re-validated.** Three findings were already fixed by later
  work and moved to Resolved; the 2026-06-30 research-mode drain items were
  relabelled; two findings stay open (the ungated planner call, and the 50 ms
  stream poll, deferred by trigger).
- **Cite sweep.** 270 of 370 cites had drifted. All were re-anchored by
  content, and the checker reported 305 ok, 0 broken, 64 verified soft hints
  and 1 known false positive (recorded in AUDIT "Accepted"). Snippet content
  that changed, rather than moved, is marked † in the refresh table.
- **Correction, 2026-10-10.** That "305 ok" was wrong. The checker accepts a
  cite up to ten lines from its anchor, and bare `file:line` cites only need
  to land on a definition-shaped line. A strict re-check, plus a hand check of
  every bare cite, found 84 still pointing at the wrong line; all are fixed.
  Now 366 cites are exact and 4 deliberately cite a `def` line whose snippet
  shows the body. Four snippets with stale values were updated. Snippets and
  prose describing changed behavior are marked ‡ in the refresh table. The
  checker fix is filed in AUDIT under course tooling.

## Decisions (2026-10-08)

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
