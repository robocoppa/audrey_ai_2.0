# Audrey codebase review — 2026-08-24

**Campaign follow-up:**
[`../campaign-3/phase-01-audit-remediation-plan.md`](../campaign-3/phase-01-audit-remediation-plan.md)
and [`../campaign-3/phase-02-skills-capability-plan.md`](../campaign-3/phase-02-skills-capability-plan.md).

## Executive summary

Audrey's core design is stronger than a typical solo-maintained orchestrator:
identity is resolved at the edge, model-supplied user ids are overridden for
private tools, local GPU work has an explicit fairness gate, media work is
isolated by network topology, long-running jobs use leases, and the hermetic
suite is unusually broad. The full suite passes.

The highest risks are at boundaries rather than in the central graph:

1. A known lossy collection-name scheme still permits authenticated cross-user
   KB reads when two user ids sanitize to the same value.
2. Streaming background tasks and one GPU-gate cancellation race can outlive
   their requests or permanently leak the local GPU slot.
3. Passthrough emits tool calls but cannot accept the assistant/tool messages
   required to continue the tool loop.
4. Upload quotas are check-then-act rather than atomic reservations, so parallel
   uploads and queued fetches can exceed the configured limit.
5. Chat-archive repair and deletion semantics are incomplete: unindexed chunks
   have no reconciler, retention can abandon live vectors, and two configured
   retention controls do not run.
6. On a cold start, a Qdrant failure in custom-tools can prevent the entire
   Audrey service from starting even though ordinary chat and web search do not
   require Qdrant.

The best next move is a focused correctness and isolation pass, not a broad
rewrite. The current architecture can support the fixes. After that, the most
valuable product additions are user-controlled data lifecycle operations, OCR,
audio/common-media ingestion, ordinary-answer provenance, and an operator
repair/readiness surface.

## Scope and evidence

Reviewed:

- Fast, deep, research, video-role, and passthrough chat paths.
- OpenAI request/response and streaming translation.
- Tool discovery, dispatch, ReAct, memory, and chat archive.
- Global and per-user KB search, upload/session/fetch/job lifecycle, and delete.
- Configuration loading, Compose topology, Docker builds, health, metrics,
  documentation, and representative tests.
- Existing optimization and security plans, so known work is labeled rather
  than presented as new.

Verification baseline:

- `.venv/bin/pytest tests/ -q`: **2,379 passed** in 41.64 seconds, with one
  FastAPI status-constant deprecation warning.
- `.venv/bin/ruff check .`: **13 findings**. Twelve are the accepted `ASYNC240`
  `Path` calls in `kb/`; the thirteenth is an unsorted import block at
  `tests/test_fast_path.py:358`. `docs/PROJECT_STATE.md:234-242` still records a
  12-finding baseline.
- Compile and mypy invocations could not start because the execution sandbox
  failed while creating its loopback network (`bwrap: ... RTM_NEWADDR: Operation
  not permitted`). This is an environment limitation, not a code failure.
- No live Ollama, Qdrant, Open WebUI, Unraid, or answer-quality eval was run.
  Runtime/deployment behavior therefore remains unverified here.
- This was a code/configuration review, not a dependency CVE audit or a
  penetration test.

## Severity guide

- **High**: Credible confidentiality, durability, interoperability, resource,
  or whole-service availability impact. Address before adding broad features.
- **Medium**: Incorrect behavior, misleading telemetry, avoidable latency, or a
  failure that needs a narrower precondition.
- **Low**: Maintainability or documentation debt with limited immediate impact.

## High-priority findings

### H1 — Per-user Qdrant collection collisions allow cross-user reads

**Status:** Known in `docs/plans/security-review-remediation-plan.md`, still
present.

`sanitize_user()` replaces every run of non-alphanumeric characters with `_`
(`src/audrey/kb/user_store.py:44`). Addresses such as `a.b@example.com`,
`a-b@example.com`, and `a_b@example.com` therefore share the same collection
name. Private text, hybrid, and image reads select that collection but do not
add the raw authenticated user as a payload filter
(`src/audrey/routes/kb.py:484-487`, `530-533`, `646-648`). Deletes do add the raw
user filter, so read isolation is weaker than delete isolation.

**Impact:** A colliding authenticated account can receive another account's
document chunks. If signup is open, the collision can be chosen deliberately.

**Simplest safe fix:** Keep the existing collection names so deployed data is
not orphaned, but always compose `SearchScope(user=<authenticated user>, ...)`
on the private branch. Apply it to dense, lexical/hybrid, and image searches.

**Acceptance:** Two deliberately colliding users; each has distinct text and
image points. Prove isolation for unscoped text, filename/artifact-scoped text,
hybrid search, and image search, both with and without a score floor.

### H2 — A granted-then-cancelled waiter can permanently leak a GPU slot

`FairLocalGate.acquire()` cleans up a task cancelled while its future remains
queued (`src/audrey/pipeline/fair_gate.py:120-132`). It does not cover the
smaller race where `_release()` has already called `fut.set_result(None)`
(`:188`) but the waiting task is cancelled before it resumes and enters the
context body. The cancellation handler finds no queued future and re-raises;
the context's `finally: _release()` was never entered. With the default
concurrency of one, all later local generations wait forever.

**Fix:** Track whether the waiter was granted. On cancellation after grant,
transfer/release the slot before propagating `CancelledError`. Keep the state
transition under the existing lock and release outside it.

**Acceptance:** A deterministic test pauses immediately after `set_result`,
cancels the grantee, then proves a third waiter enters and `_available` never
exceeds configured concurrency.

### H3 — Deep and research streaming work can survive client disconnects

The deep stream creates planning and panel tasks at
`src/audrey/routes/openai/pipeline.py:655` and `:693`. Only the later synth task
has a cancellation cleanup. The research stream has the same planning gap and
starts one full-pipeline task at `:990` and `:1031` with no owner-side cleanup.
Panel and researcher fan-out use `asyncio.as_completed()` over implicit tasks
(`src/audrey/pipeline/deep_panel.py:748`, `:1890`), so cancelling the consumer
does not reliably cancel siblings.

The producer `finally` blocks also await a sentinel write into bounded queues
(`src/audrey/routes/openai/pipeline.py:736`, `:1029`). Once the client-side
consumer is gone, a full queue can make cancellation cleanup itself hang. A
research disconnect can therefore continue multiple cloud calls, tool calls,
verification, fact-checking, and writing; it can eventually block forever on
the abandoned queue.

**Fix:** Make every spawned task explicitly owned by one stage. Use a shared
`cancel_and_drain()` helper or structured concurrency, create fan-out tasks
explicitly, and cancel/gather them in `finally`. Do not require a blocking queue
put to signal producer completion; task completion is already a signal.

**Acceptance:** Cancel during planning, panel/research fan-out, verification,
and synthesis/writing. Assert all child tasks finish, all gate slots return,
queues do not block, archive marks the turn partial, and no later model/tool
calls occur.

### H4 — Passthrough tool calling is outbound-only

Audrey correctly translates an Ollama tool call into OpenAI-shaped non-stream
and stream responses. The next client request cannot be represented, however:
`ChatMessage` has only `role`, required non-null `content`, and `name`
(`src/audrey/routes/openai/schemas.py:16-26`). It has no assistant `tool_calls`
or tool-message `tool_call_id`. `_handle_passthrough()` serializes only those
validated fields (`src/audrey/routes/openai/passthrough.py:174`), so ignored
extras cannot survive by accident.

The OpenAI Chat Completions contract defines assistant tool calls and tool
messages linked by `tool_call_id`; it also includes `developer` messages for
current models. See the official [Chat Completions API reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions).

**Impact:** Hermes/OpenClaw/SDK-style clients can receive the first tool call
but cannot return the tool result faithfully. Audrey advertises a capability it
cannot complete end to end.

**Fix:** Use role-discriminated message models, including nullable assistant
content with `tool_calls`, tool content plus `tool_call_id`, and `developer`
messages. Define and test an explicit compatibility matrix for request fields;
do not silently ignore fields while calling the route OpenAI-compatible.

**Acceptance:** A real two-request loop: request one returns a tool call;
request two resends the assistant call and linked tool result; Ollama receives
both unchanged and produces the final answer. Cover stream and non-stream.

### H5 — Storage quota enforcement is raceable and does not reserve work

Every upload flow reads `user_total_bytes()` and mutates storage later
(`src/audrey/routes/files.py:479`, `:748`, `:875`, `:1187`, `:1501`). The check
and reservation are not one database operation.

- Two single-shot uploads can both see the same total and both commit.
- Any number of chunked sessions can be opened because `total_bytes` is checked
  but never reserved in `upload_sessions` for quota accounting.
- URL fetch rows carry `bytes=0`; multiple pending downloads each pass a ceiling
  check and can consume their full ceilings concurrently.
- Parts themselves occupy disk without contributing to `user_total_bytes()`.

**Impact:** A user or buggy client can exceed the configured limit and exhaust
the shared mount despite every individual request passing validation.

**Fix:** Add an atomic SQLite reservation operation. Account for completed
bytes, open-session declared totals, pending fetch ceilings, and in-flight
single-shot reservations in one transaction. Convert a reservation to stored
bytes on commit and release it on every failure/TTL path.

**Acceptance:** Concurrent session opens, upload finalizations, and URL queues
near the limit; exactly the set that fits succeeds. Crashes and stale-session
sweeps return reservations without double-freeing them.

### H6 — Chat archive repair and retention can violate their own durability contract

Archive writes commit SQLite first and leave failed embeddings/upserts with
`indexed_at IS NULL`, promising “later reconcile”
(`tools-server/chat_archive.py:385`). No reconciler exists; stats only count the
stranded rows (`:626`). They remain permanently unsearchable.

Retention attempts Qdrant deletion, logs and continues on failure (`:595`),
then deletes the SQLite messages and chunks (`:597-602`). That removes the
source record needed to retry while the old vector remains searchable. In
addition, `CHAT_ARCHIVE_MAX_BYTES` is stored but explicitly unimplemented
(`:568`), and retention is manual because the periodic loop is not wired
(`src/audrey/routes/admin.py:113-118`).

**Impact:** Search durability is weaker than documented, and a configured
privacy deletion can leave retrievable content with no repair record.

**Fix:** Implement an idempotent outbox/reindex loop for null `indexed_at`
rows. For deletion, keep a tombstone/outbox until Qdrant confirms deletion;
only then remove the source rows. Either implement `max_bytes` and scheduled
retention or reject/omit those settings until they are real.

**Acceptance:** Failure injection for embedding, upsert, and delete; restart
and reconcile converge both stores. A failed prune leaves retryable state and
the data is never reported deleted while its vector remains searchable.

### H7 — Stateful custom-tools startup can block the whole platform

`tools-server/app.py:65-105` initializes memory and chat archive before the app
serves. Both require Qdrant. A Qdrant outage therefore prevents unrelated
`web_search` and `web_fetch` from starting. On a cold Compose start, Audrey also
waits for custom-tools to become healthy, so ordinary no-tool chat is blocked
despite Audrey's own Qdrant path being deliberately fail-soft.

**Fix:** Initialize independent tool capabilities independently and return a
component-specific 503 only for unavailable stateful routes. Audrey already
has bounded background rediscovery, so Compose can allow Audrey to start while
custom-tools recovers. A later service split is optional; it is not required to
remove this failure fan-out.

**Acceptance:** Cold-start with Qdrant down: Audrey chat and web search/fetch
start, memory/archive/KB report unavailable, component status identifies
Qdrant, and rediscovery restores the stateful tools after recovery.

## Medium-priority findings

### M1 — Plain fast streaming is a separate, inconsistent implementation

The fast route creates a stream id and role frame at
`src/audrey/routes/openai/pipeline.py:343-368`, then its plain-chat branch calls
`_stream_openai()` (`:535`), which creates a second id and a second role frame
(`:1377`). One response therefore mixes two completion identities.

That branch also bypasses `run_fast_path()`, so it misses the bounded two-model
fallback, `fast_path.no_thinking_prose`, `dispatch_total`, and the normal
pipeline metrics. It emits `worker_ok(spec.name)` before the model stream has
actually started (`:531`). If Ollama never emits `done`, `_stream_openai()`
still sends `[DONE]` without a terminal finish frame.

**Simpler design:** One stream-attempt abstraction owns id/fingerprint,
thinking policy, gate, health, dispatch/pipeline metrics, terminal state, and
pre-first-token fallback. Banner rendering should consume that abstraction,
not wrap a second SSE generator.

### M2 — Streaming and passthrough metrics can report failures as success

`_passthrough_stream_sse()` catches `OllamaError` and turns it into an SSE
content frame. Its outer metric wrapper only changes `outcome` if that error
escapes, so the metric remains `ok` (`src/audrey/routes/openai/passthrough.py:195-222`).
Cancellation is not represented there either. Plain fast streaming has the
larger metric omissions described in M1.

**Fix:** Return/raise a typed terminal outcome from the stream layer and record
exactly one of `ok`, `error`, `cancelled`, or `truncated` in the owner.

### M3 — Archive capture adds response latency and fallback conversation ids do not stitch

The non-stream path says archive work “never delays the response” while
awaiting it before returning (`src/audrey/routes/openai/pipeline.py:135-151`).
The client call performs an awaited HTTP request with a five-second timeout
despite its “Fire-and-log” docstring
(`src/audrey/pipeline/chat_archive.py:230-256`). Custom-tools can then perform
multiple embedding/upsert operations sequentially for a long answer.

The fallback conversation id hashes `messages[:6]`
(`src/audrey/pipeline/chat_archive.py:99`). A growing history therefore gets a
new id at turns 1, 2, 3, and so on until six messages are present. The
`messages[-1].metadata` fallback (`:85`) is also effectively dead because the
validated `ChatMessage` model drops metadata before this helper is called.
Explicit OWUI `chat_id` avoids both defects, but other clients do not.

**Fix:** Use a bounded, lifecycle-managed archive queue rather than unmanaged
`create_task()` or an awaited write. Base fallback identity on user plus the
first user turn, or require/accept an explicit conversation id. Test the id
against growing prefixes, not the same static history twice.

### M4 — Partial failures can strand or resurrect stored data

- Legacy memory migration counts failed rows but unconditionally renames the
  source database to `.migrated` (`tools-server/db.py:185`, `:217-220`). A
  transient failure permanently abandons those rows. Rename only after all rows
  succeed; deterministic ids already make retries safe.
- File deletion removes the SQLite row before Qdrant
  (`src/audrey/routes/files.py:2562-2584`). A Qdrant failure leaves searchable
  points with no current row, and startup reconciliation can recreate the row.
  Use a deletion state/outbox rather than claiming either store is atomically
  deleted.

### M5 — Tool discovery's failure contract and trust boundary do not match its code

`discover_one()` says it returns an empty list on any error
(`src/audrey/tools/discovery.py:184`), but JSON parsing and shape access occur
outside the exception handler (`:193`). Invalid JSON or a non-object document
can abort `discover_all()` and startup. Discovery also accepts an untagged POST
endpoint because it rejects only a non-tools tag (`:205-206`), while the module
contract says tool endpoints live under the `tools` tag.

**Fix:** Catch response decode/schema errors per server and require the explicit
`tools` tag. A malformed or overly broad server should not poison other
servers' registries or expose an unintended POST endpoint to models.

### M6 — Health checks prove only that the process answers HTTP

Both `/health` routes unconditionally return `ok`
(`src/audrey/main.py:294-296`, `tools-server/app.py:288-290`). They do not expose
tool count, model cooldowns, Qdrant reachability, archive backlog, watcher/
reconciler state, worker queue age, or GPU-gate depth. Compose can call a
severely degraded service healthy.

**Fix:** Preserve shallow `/health` as liveness. Add authenticated/component
`/ready` or admin status with explicit required/optional dependencies and
Prometheus gauges. Do not make all optional features a single binary readiness
gate.

### M7 — Container dependency installs ignore the committed lockfile

The repository has a shared `uv.lock`, but both main Dockerfiles copy only a
`pyproject.toml` and run `uv pip compile` during the build
(`docker/audrey.Dockerfile:69-72`, `docker/custom-tools.Dockerfile:42-45`). With
`>=` constraints, a cold rebuild can resolve a new runtime set even when the
lockfile did not change. The digest-pinned base and uv binary do not make that
Python environment reproducible.

**Fix:** Copy the workspace lockfile and install/export in frozen/locked mode
for each package. Keep dependency refresh as a deliberate, tested change. Also
finish the known non-root container work from the existing security plan;
media sidecars already demonstrate the pattern.

### M8 — Blocking disk I/O remains on async request paths

Single-shot uploads and chunk parts use synchronous `Path.open()`/`write()` in
async handlers (`src/audrey/routes/files.py:368-376`, `:810-820`), and the
single-shot MIME sniff is synchronous (`:533`). Large writes to an Unraid mount
can stall unrelated requests on the event loop. By contrast, assembly and
fetch-result paths already use `asyncio.to_thread`, showing the intended
pattern.

**Fix:** Move streaming writes to an async file API or a bounded writer thread;
thread MIME sniff/stat calls. Share the app's HTTP client for hot-path memory
recall instead of creating a new `httpx.AsyncClient` per request in
`pipeline/memory.py`.

### M9 — Configuration and documentation contain operationally misleading statements

- `compose.yaml:43` says tool discovery never retries; `main.py:225-263` now
  retries for two minutes.
- Compose says `TOOL_SERVERS` and `KB_DATASET_PATHS` are deliberately omitted
  so YAML is authoritative (`compose.yaml:77-80`), but `env_file: .env` loads
  them and `.env.example:29,35` actively sets both. A copied example silently
  overrides later YAML edits.
- README says seven tool endpoints (`README.md:52-56`); tools-server exposes ten
  model tools. It says five virtual models (`README.md:76`); code exposes seven,
  including research and video (`src/audrey/routes/openai/routes.py:41-49`).
- FastAPI's description and `routes/openai/__init__.py` say six and omit video.
- `fast_path.py` still attributes roughly 0.7 seconds “once” to prose thinking,
  while the current measured project state records about 4.0 seconds.
- `_USER_SCOPED_TOOLS` says no startup audit exists immediately above the
  implemented `audit_user_scoping()` function.
- The security remediation plan still reads like port unpublishing is pending;
  custom-tools is already unpublished while the collection collision and root
  containers remain open.

**Fix:** Remove active source-of-truth overrides from `.env.example` or label
them as overrides and keep them commented. Generate model/tool inventory in
docs from the source constants/OpenAPI where practical. Sweep status comments
when behavior changes.

## Lower-priority simplification opportunities

1. **Split by lifecycle, not arbitrary file size.** `routes/files.py` currently
   combines upload transport, chunk sessions, URL fetch jobs, media-worker jobs,
   artifacts, listing, and deletion. Extract quota reservations and lifecycle
   transitions into a service layer first, then use thin routers for upload,
   fetch, worker, and file-management endpoints. This makes the H5/M4 invariants
   testable once rather than per route.
2. **Share streaming stage ownership.** Deep and research routes duplicate
   banner queues, producer tasks, terminal handling, cancellation, metrics, and
   archive finalization. A small stage runner with one ownership contract would
   remove the exact class of defects in H3/M1/M2.
3. **Separate flat panel, research pipeline, and ledger policy.** They are
   distinct concepts currently co-located in `deep_panel.py`. Split only after
   cancellation tests exist; a mechanical move before then raises regression
   risk without fixing behavior.
4. **Make capability registration declarative.** Tool name, user-scope policy,
   internal/public status, and availability currently live across OpenAPI tags,
   `_USER_SCOPED_TOOLS`, prompts, and route startup. A `ToolPolicy` record built
   during discovery would turn warnings into explicit per-tool policy and
   reduce two-file security edits.
5. **Avoid inert settings.** Configuration that is accepted but not enacted
   (`CHAT_ARCHIVE_MAX_BYTES`, periodic retention) is worse than an absent knob.
   Validate unsupported nonzero values at startup until implementation lands.

## Capability backlog

| Priority | Capability | Value | Fit / reuse | Effort |
|---|---|---:|---|---:|
| P0 | User data control: list/edit/delete memory; export/delete chat history; account-wide purge with status | Very high | Existing per-user payload filters, SQLite source stores, and admin auth | Medium |
| P0 | Archive repair and scheduled retention | Very high | Existing `indexed_at`, stats, admin routes, and deterministic point ids | Low-medium |
| P1 | OCR for scanned PDFs | High | Existing PDF ingest, image extraction/vision stack, and background media pattern | Medium |
| P1 | Audio uploads and common video containers (`m4a/mp3/wav/webm/mov`) | High | Existing ffmpeg/faster-whisper worker and transcript ingestion | Medium |
| P1 | Provenance on ordinary fast/deep tool-backed answers | High | Existing retrieved-result logs, research ledger, and Sources renderer | Medium-high |
| P1 | Operator readiness/repair console | High | Existing admin auth, metrics, health tracker, archive stats, queues, and reconciler | Medium |
| P2 | Download/export of original files and derived artifacts | Medium | Existing scoped file rows and artifact resolver; video source retention needs a policy decision | Low-medium |
| P2 | `/v1/responses` compatibility adapter | Medium | Existing model registry, streaming, and tool translation | High |

### Capability notes

**Data control comes first.** Audrey can write and semantically recall durable
memory, but the user has no supported way to inspect, correct, or delete it.
Files have deletion; memory and chat history do not have equivalent per-user
lifecycle APIs. Adding more automatic memory before adding control would make a
privacy and trust gap larger.

**OCR and audio are unusually good fits.** Scanned PDFs are explicitly rejected
as unsupported (`src/audrey/kb/extract.py:139`), while the platform already has
vision models, image processing, background jobs, ffmpeg, and Whisper. Audio
ingestion can reuse the video transcript path while skipping frame extraction.

**Ordinary provenance should reuse the research ledger, not ask prose models to
invent citations.** Fast/deep workers already record tool results, but only the
research route deterministically renders a source list. Promote retrieved URLs
and file identities through a lightweight evidence envelope and append sources
from code.

**Responses API is an integration feature, not the first repair.** The current
OpenAI Responses contract offers state continuation via `previous_response_id`
and structured function-call output items; see the official
[Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).
An adapter could broaden compatible clients, but it should be built only after
Chat Completions tool-loop and stream-terminal semantics are correct.

## Recommended implementation sequence

### Phase 0 — Isolation and cancellation

1. Add the raw-user filter to every private KB read.
2. Close the FairLocalGate granted/cancelled race.
3. Add explicit cancellation ownership to deep/research stage tasks and fan-out.
4. Add failure-injection and disconnect tests before refactoring stream code.

**Gate:** Full suite; scoped ruff; the new isolation matrix; cancellation at
each stage; no leaked task, queue, or GPU slot.

### Phase 1 — Contract and capacity correctness

1. Implement role-aware OpenAI messages and a real passthrough tool loop.
2. Add atomic quota reservations across all three ingest paths.
3. Replace success/error-as-content ambiguity with typed stream outcomes.

**Gate:** OpenAI two-turn tool loop in stream/non-stream; concurrent quota tests;
one completion id and one role frame per stream; metrics match injected errors.

### Phase 2 — Durable state lifecycle

1. Build archive reindex and deletion outboxes; wire scheduled retention.
2. Make file deletion and legacy-memory migration retryable.
3. Put archive writes behind a bounded managed queue and fix fallback
   conversation identity.
4. Add user-facing memory/chat list, correction, export, and delete operations.

**Gate:** Restart/failure-injection convergence, deletion proof across SQLite,
Qdrant, and disk, and no archive latency in the response critical path.

### Phase 3 — Runtime consistency and deployment

1. Unify plain fast streaming with fast-path policy/fallback/metrics.
2. Add component readiness and make custom-tools degrade per capability.
3. Install Docker dependencies from `uv.lock`; finish non-root containers.
4. Move hot-path file I/O off the event loop and reuse HTTP clients.
5. Correct generated/manual documentation and remove inert/default overrides.

**Gate:** Hermetic suite and ruff; container build reproducibility check; cold
boot with Qdrant/Ollama/custom-tools failures; then user-run Unraid smoke and
live evals.

### Phase 4 — Product enrichment

Implement OCR and audio/common-media ingestion first, then ordinary-answer
provenance. Consider original-file downloads and `/v1/responses` after the data
lifecycle and compatibility foundations are stable.

## What should remain unchanged

- Preserve the documented complexity-gate ordering.
- Preserve authenticated identity override for every user-scoped model tool.
- Keep media-worker off Ollama's network path and keep fetcher privileges narrow.
- Keep collection names stable while repairing collision isolation; renaming
  them first would orphan deployed data.
- Keep parser tolerance for fenced JSON and bare arrays.
- Keep tool-side effects single-shot; do not add blind model fallback after a
  ReAct loop has begun.
- Keep live answer-quality evaluation separate from hermetic plumbing tests.

## Review conclusion

There is no reason to replace LangGraph, FastAPI, Qdrant, or the sidecar model.
The platform's problems are local and repairable. The first three phases above
reduce confidentiality, availability, quota, data-lifecycle, and compatibility
risk while also simplifying the code. Product expansion should follow those
invariants so new ingestion and agent capabilities inherit reliable isolation,
repair, cancellation, and provenance instead of multiplying current edge cases.
