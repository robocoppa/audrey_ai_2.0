# Campaign 3 Phase 1 — codebase audit remediation

**Status:** Planned. No implementation has started.

**Source:** [`../reviews/codebase-review-2026-08-24.md`](../reviews/codebase-review-2026-08-24.md)

## Goal

Close the correctness, isolation, durability, compatibility, and operational
findings from the 2026-08-24 codebase review before Campaign 3 adds a general
skills layer or other broad product capabilities.

This is one campaign phase with several independently shippable waves. It is
not one large refactor or one deploy. Each wave has its own tests, laptop gate,
deploy boundary, on-box checks, and rollback point.

## Why this comes first

Skills will add another source of instructions, tool filtering, routing state,
and versioned configuration. Building that on top of the current cancellation,
stream ownership, tool policy, and degraded-startup gaps would multiply those
gaps across every skill. Phase 1 first establishes the invariants Phase 2 can
reuse:

- authenticated data isolation at every private read;
- owned and cancellable request work;
- one request/message and stream-terminal contract;
- retryable storage lifecycles;
- declarative model-visible tool policy;
- component-level readiness and graceful degradation.

## Scope

In scope:

- All high findings H1–H7.
- All medium findings M1–M9.
- The lower-priority simplifications where they directly reduce the defect
  surface being repaired.
- The audit's P0 user-data-control capability, because the storage lifecycle
  work supplies the safe primitives it needs.
- The operator readiness surface, because it is the user-facing expression of
  H7/M6 rather than a separate feature.

Deferred until after Phase 2 unless reprioritized:

- OCR and additional audio/video formats.
- Ordinary-answer provenance.
- Original-file/artifact download.
- `/v1/responses` compatibility.
- A broad UI for installing or authoring skills.

## Non-goals

- No replacement of FastAPI, LangGraph, Qdrant, Ollama, or the custom-tools
  sidecar.
- No per-user collection rename. Existing collection names remain stable;
  raw authenticated-user filters repair isolation without orphaning data.
- No blind model fallback after a ReAct loop has begun and may have performed
  a side effect.
- No mechanical module split before cancellation and lifecycle behavior has
  deterministic tests.
- No claim of Unraid/runtime verification from laptop tests.

## Invariants that every wave preserves

- The complexity-gate ordering remains unchanged.
- Model-supplied `user` values never override authenticated identity.
- User-scoped tools remain enforced in the dispatcher, not in prompt text.
- Media-worker remains off Ollama's network path; media-fetcher retains only
  the egress it needs.
- Parser tolerance for fenced JSON and bare top-level arrays remains intact.
- Tool-side effects remain single-shot.
- Live answer-quality evaluation remains separate from hermetic plumbing
  tests.
- Existing dirty working-tree files are not folded into Campaign 3 work.

## Baseline and evidence discipline

The review baseline was 2,379 passing hermetic tests. Global ruff had thirteen
findings: twelve accepted `ASYNC240` findings under `kb/` and one unsorted test
import. Treat that as historical evidence, not a permanent expected count.
Every implementation slice records its own pre-change baseline and judges its
changed files.

For each slice:

1. Write the failure or isolation test first when the defect is reproducible
   hermetically.
2. Make the smallest behavior change that satisfies the contract.
3. Run targeted tests, then the full hermetic suite.
4. Run ruff on changed files.
5. After changes under `src/audrey/`, `tools-server/`, or `config.yaml`, run
   `scripts/check-lesson-links.py` and resolve every `DRIFT` line.
6. Record what still requires the user's Unraid deployment and live smoke.

Schema changes must be forward-compatible during a rolling recreate,
idempotent on restart, and tested against an existing database fixture. A
failed migration must leave the old source recoverable.

---

## Wave 1A — isolation and cancellation

**Findings:** H1, H2, H3.

This wave removes the highest confidentiality and resource-leak risks before
request-path refactoring begins.

### 1A.1 Private KB read isolation — H1

Keep the deployed `kb_user_*` collection names. Add a raw authenticated-user
payload condition to every private read by composing
`SearchScope(user=<authenticated user>, ...)` on the per-user branch.

Cover all shapes rather than testing only the simplest dense search:

- unscoped private text search;
- filename- and artifact-scoped text search;
- dense and lexical/hybrid search;
- image search;
- score-floor enabled and disabled;
- global-plus-private merged results.

Use two deliberately colliding user ids with distinct text and image points.
Each user must receive only their own private data while global results remain
available.

### 1A.2 FairLocalGate grant/cancel race — H2

Add a deterministic scheduling hook or test seam that pauses after a waiter is
granted and before its context body resumes. Track whether the cancelled
waiter owned a granted slot. If it did, transfer or release that slot before
re-raising `CancelledError`.

Acceptance:

- cancel a queued waiter before grant: queue cleanup remains correct;
- cancel a waiter after grant but before entry: a third waiter enters;
- cancel inside the context body: the normal context cleanup releases once;
- `_available` never exceeds configured concurrency;
- no double release, starvation, or orphaned future.

### 1A.3 Streaming task ownership — H3

Introduce one `cancel_and_drain()` ownership helper or an equivalent structured
concurrency primitive. Every task created for planning, panel fan-out,
researcher fan-out, verification, fact-checking, writing, and synthesis must
have one visible owner whose `finally` cancels and awaits it.

Create fan-out tasks explicitly instead of relying on implicit tasks passed to
`asyncio.as_completed()`. Producer shutdown must not depend on a blocking put
to an abandoned bounded queue; task completion/cancellation is the terminal
signal.

Disconnect tests cancel during:

- planning;
- ordinary panel fan-out;
- research fan-out;
- verification and fact-checking;
- synthesis and writing;
- queue-full producer cleanup.

Each test proves all children settle, all GPU and per-user slots return, no
later model/tool call occurs, and partial archive capture follows the chosen
contract.

### Wave 1A gate

- New two-user isolation matrix passes.
- Deterministic grant/cancel race test passes repeatedly.
- Every disconnect stage drains without leaked tasks, queues, or slots.
- Full hermetic suite and changed-file ruff pass.
- User deploys the slice and confirms an ordinary chat, a private-file query,
  a cancelled deep request, and a later local request all work.

---

## Wave 1B — OpenAI contract and one stream ownership model

**Findings:** H4, M1, M2.
**Simplification absorbed:** shared streaming stage ownership.

Wave 1A supplies cancellation tests first. Wave 1B can then consolidate stream
behavior without hiding task leaks inside a mechanical rewrite.

### 1B.1 Role-aware Chat Completions messages — H4

Replace the single permissive `ChatMessage` shape with role-discriminated
models that preserve the current contract and add the fields needed for a real
tool loop:

- `system` and `developer` instruction messages;
- `user` content in text or multimodal parts;
- `assistant` content that may be null when `tool_calls` is present;
- `tool` content with required `tool_call_id`;
- existing optional names where the OpenAI contract permits them.

Write an explicit request-field compatibility table. Unsupported fields are
rejected or documented; they are not silently dropped while the endpoint is
described as OpenAI-compatible.

Acceptance is a two-request passthrough loop in both stream and non-stream:
request one returns a tool call; request two resends the assistant call and
linked tool result; the Ollama request receives the relationship unchanged and
returns the final answer.

### 1B.2 Typed stream terminal outcome — M2

Define one terminal result owned by the outer stream layer:

```text
ok | error | cancelled | truncated
```

The inner generator may render an error frame for client compatibility, but it
must also report the typed outcome to the metric/archive owner. Exactly one
terminal outcome and one finish decision are recorded per request.

Failure-injection tests cover pre-first-token error, mid-stream error, missing
Ollama `done`, client cancellation, and normal completion.

### 1B.3 Unify plain fast streaming — M1

Build one stream-attempt abstraction that owns:

- completion id and fingerprint;
- the single assistant-role frame;
- model selection and pre-first-token fallback;
- thinking policy;
- GPU gate and health updates;
- dispatch and pipeline metrics;
- terminal frame/outcome;
- archive capture boundary.

The banner layer consumes that abstraction. It does not wrap a second SSE
generator that creates another identity. Preserve the no-blind-retry rule once
tokens or tool side effects have been emitted.

### 1B.4 Share deep/research stage mechanics only after behavior is pinned

Extract a small stage runner for the mechanics both paths actually share:
owned producer task, bounded event delivery, cancellation/drain, terminal
outcome, metrics, and archive finalization. Keep panel policy, research policy,
and banners separate. Do not combine them into a generic framework whose
configuration is harder to review than the current branches.

### Wave 1B gate

- Passthrough tool continuation works end to end in stream/non-stream tests.
- Every stream uses one completion id and one role frame.
- Every injected failure produces exactly one matching metric outcome.
- Plain fast streaming applies the same thinking, health, fallback, and metric
  policy as non-stream fast execution where their contracts overlap.
- Existing banner ordering remains unchanged.
- Full suite, changed-file ruff, and lesson-link checks pass.
- User-run Open WebUI and agent-client smoke tests pass after deployment.

---

## Wave 1C — quota and durable data lifecycle

**Findings:** H5, H6, M3, M4.
**Capability absorbed:** user memory/chat inspection, correction, export, and
deletion.

This wave creates retryable state machines before exposing broader data
controls.

### 1C.1 Atomic quota reservation service — H5

Move quota decisions out of individual route branches into a storage lifecycle
service backed by one SQLite transaction. The reservation calculation includes:

- completed stored bytes;
- declared totals for open chunk sessions;
- on-disk chunk parts;
- pending URL-fetch ceilings;
- in-flight single-shot reservations.

Reserve before accepting work, convert the reservation on commit, and release
it on every explicit failure, cancellation, TTL sweep, or abandoned-job path.
Use stable reservation ids so recovery and repeated cleanup are idempotent.

Test concurrent reservation races near the limit across mixed upload types,
not merely several copies of one route. Exactly the set that fits succeeds.

### 1C.2 Archive reindex and deletion outboxes — H6

SQLite remains the source of truth. Add:

- an idempotent reindex loop for chunks with `indexed_at IS NULL`;
- a deletion tombstone/outbox that remains until Qdrant confirms deletion;
- bounded retries with visible last error and attempt time;
- scheduled retention wiring;
- either a real `max_bytes` implementation or startup rejection of a nonzero
  value until it exists.

Embedding, upsert, delete, restart, and repeated-run failure injection must
converge without duplicated points or data reported deleted while still
searchable.

### 1C.3 Archive response decoupling and conversation identity — M3

Put response-time archive writes behind a bounded, lifecycle-managed queue.
Define what happens when it is full: metric/log and a retryable source record,
not unbounded tasks or hidden response latency.

Resolve conversation identity from explicit OWUI/client ids first. The
fallback must remain stable as the history grows, using the authenticated user
and stable first-turn material rather than the first six current messages.
Preserve message metadata needed for identity instead of validating it away.

### 1C.4 Retryable legacy migration and file deletion — M4

- Rename the legacy memory database only after every row has migrated; failed
  rows leave the source eligible for a safe retry.
- Represent file deletion as a state/tombstone. Remove Qdrant points and disk
  artifacts idempotently, then remove/finalize the SQLite source row only when
  the deletion contract is satisfied.
- Ensure startup reconcile recognizes deletion state and cannot resurrect a
  file being deleted.

### 1C.5 User data controls

Once the lifecycle primitives above are proven, expose authenticated per-user
operations for:

- list, correct, and delete durable memories;
- export and delete chat history;
- account-wide purge with progress/status;
- retry/repair status visible to the owner or admin as appropriate.

These routes enforce the authenticated user server-side. No model-callable tool
can select another user or bypass pending deletion.

### Wave 1C gate

- Concurrent mixed upload reservations never exceed quota.
- Crash/TTL cleanup returns reservations exactly once.
- Archive index/delete failure tests converge after restart.
- A file deletion failure remains retryable and cannot be resurrected.
- Archive work is absent from response critical-path latency.
- Conversation ids stitch across growing history.
- User data-control isolation and deletion proofs pass across SQLite, Qdrant,
  and disk.
- User performs a backup, deploy, migration smoke, upload/fetch smoke, and a
  disposable-user export/delete exercise on Unraid.

---

## Wave 1D — capability policy, degraded startup, and readiness

**Findings:** H7, M5, M6.
**Simplification absorbed:** declarative capability registration.

This wave is also the direct platform prerequisite for Phase 2 skills: a skill
must be able to restrict a known capability set and explain why a requested
capability is unavailable.

### 1D.1 Declarative tool policy

Create one model-visible policy record per discovered tool. At minimum it
holds:

- tool name and source server;
- explicit model-visible/internal status;
- user-scope requirement;
- component dependencies;
- current availability and reason;
- dispatch metadata.

Build the model-visible `ToolRegistry` from these records. Startup auditing may
still warn, but a missing user-scope declaration or unexpected public tool is a
validation failure rather than a prompt/documentation convention.

### 1D.2 Harden discovery — M5

Catch HTTP, JSON decode, and schema-shape failures per server. Require the
explicit `tools` OpenAPI tag for model-visible endpoints. A malformed server or
untagged POST neither aborts discovery of other servers nor becomes callable.

### 1D.3 Degrade custom-tools per capability — H7

Initialize memory, archive, KB, and stateless web capabilities independently.
Qdrant failure disables only Qdrant-dependent routes with a component-specific
503. Web search/fetch and ordinary Audrey chat still start. Preserve Audrey's
bounded rediscovery so recovered capabilities reappear without a full stack
restart.

### 1D.4 Liveness, readiness, and operator status — M6

Keep shallow `/health` as process liveness. Add authenticated `/ready` or admin
status that reports required and optional components separately:

- discovered/available tool count and per-capability failures;
- Ollama/Qdrant state;
- archive index/delete backlog;
- watcher/reconciler activity;
- upload/fetch worker queue age;
- GPU-gate and per-user inflight pressure.

Add bounded-cardinality Prometheus gauges for the same operational state.
Optional capability failure does not make the whole service unready unless the
deployment explicitly marks that capability required.

### Wave 1D gate

- Cold start with Qdrant down serves ordinary chat and web tools.
- Stateful tools return a precise temporary-unavailable response.
- Recovery plus rediscovery restores them.
- Invalid OpenAPI from one server does not poison another.
- Untagged endpoints never enter the model-visible registry.
- Readiness and metrics agree with injected component failures.
- User confirms the degraded-start/recovery sequence on Unraid.

---

## Wave 1E — reproducibility, async I/O, and operational truth

**Findings:** M7, M8, M9.
**Known security work absorbed:** non-root Audrey and custom-tools containers.

### 1E.1 Locked and non-root containers — M7

- Build both Python services from the committed workspace `uv.lock` in frozen
  mode; dependency refresh becomes an explicit lockfile change.
- Run Audrey and custom-tools as non-root users.
- Set application, `/data`, knowledge, staging, and model-cache ownership
  deliberately rather than relying on daemon-created root paths.
- Preserve the existing external/internal network topology and health ordering.

Hermetic checks prove dependency selection. The user performs the bind-mount
ownership, clean boot, upload, watcher ingest, CLIP cache, and archive smoke on
Unraid.

### 1E.2 Remove blocking hot-path disk/network setup — M8

- Move upload and chunk streaming writes to an async file API or bounded worker
  threads with backpressure.
- Move MIME sniff/stat work off the event loop.
- Reuse the application HTTP client for memory recall instead of constructing
  one per request.
- Measure event-loop responsiveness with concurrent slow-write tests rather
  than assuming `to_thread` alone proves the outcome.

### 1E.3 Correct configuration and documentation — M9

- Make `.env.example` clearly show overrides as commented opt-ins; it must not
  silently defeat `config.yaml` source-of-truth values.
- Correct discovery retry, model inventory, tool inventory, virtual-model, and
  measured thinking-latency statements.
- Generate inventories from source/OpenAPI where practical.
- Update the older security remediation plan to distinguish deployed,
  verified, and still-open work.
- Remove unsupported/inert settings or reject active values at startup until
  the implementation exists.

### Wave 1E gate

- Rebuilding twice from one lockfile resolves the same dependency set.
- Both services run non-root and complete upload/search/archive smokes.
- Slow upload writes do not stall an unrelated health/chat request in the
  concurrency test.
- Source-of-truth override tests cover environment versus YAML precedence.
- Generated inventories match code/OpenAPI.
- Full suite, changed-file ruff, lesson-link checks, and container builds pass;
  user completes the Unraid smoke.

---

## Wave 1F — bounded simplification after correctness

This wave is not permission for a broad rewrite. Each extraction must remove a
specific duplicate lifecycle or policy implementation proven by Waves 1A–1E.

1. Extract upload quota and lifecycle transitions from `routes/files.py` into
   the service built in Wave 1C; split thin routers only after route behavior is
   pinned.
2. Keep the shared stream owner from Wave 1B small and explicit. Do not merge
   panel and research policy merely because they both stream banners.
3. Split flat panel execution, staged research, and ledger policy only if the
   move reduces an active maintenance problem; cancellation tests must pass
   unchanged through the move.
4. Remove duplicated/inert configuration using the validation built in Waves
   1C and 1E.

Skills do not wait on an aesthetic `deep_panel.py` split. They do wait on the
behavioral and policy gates above.

---

## Finding-to-work traceability

| Finding | Wave | Primary acceptance evidence |
|---|---|---|
| H1 collection collision | 1A | two-colliding-user text/image isolation matrix |
| H2 GPU slot leak | 1A | deterministic granted-then-cancelled waiter test |
| H3 orphaned stream work | 1A | disconnect at every stage; no child work remains |
| H4 incomplete tool-loop messages | 1B | two-request stream/non-stream passthrough loop |
| H5 raceable quota | 1C | concurrent mixed reservation tests and crash cleanup |
| H6 archive repair/deletion | 1C | restart/failure-injection convergence |
| H7 Qdrant startup fan-out | 1D | degraded cold start and rediscovery recovery |
| M1 split fast streaming | 1B | one id/role frame and policy parity |
| M2 incorrect stream metrics | 1B | typed terminal outcome under injected failures |
| M3 archive latency/identity | 1C | bounded queue and growing-history stitch test |
| M4 stranded/resurrected data | 1C | retryable migration and deletion-state tests |
| M5 discovery contract | 1D | malformed/untagged multi-server tests |
| M6 shallow health | 1D | component readiness and matching metrics |
| M7 unlocked/root containers | 1E | frozen build plus non-root Unraid smoke |
| M8 blocking async I/O | 1E | concurrent slow-write responsiveness test |
| M9 misleading config/docs | 1E | precedence and generated-inventory checks |

## Dependency and release order

```text
1A isolation/cancellation
  -> 1B request + stream contract
  -> 1C durable data lifecycle
  -> 1D capability policy/readiness
  -> 1E packaging/I/O/docs
  -> 1F only the simplifications justified by the repaired behavior
  -> Campaign 3 Phase 2 skills
```

1C and 1D may be developed independently after 1A, but they should still ship
as separate deploy units. Phase 2 begins only after all Phase 1 findings are
closed, explicitly accepted as deferred with a reason, or proven already fixed
by re-audit.

## Phase 1 completion gate

Phase 1 is complete only when:

- every H/M row above has a test, deploy verification, or written accepted-risk
  disposition;
- no high finding remains open;
- full hermetic tests pass;
- changed-file ruff passes and the accepted global baseline is re-measured;
- lesson-link checks have no `DRIFT`;
- schema/outbox/retry paths converge under restart and failure injection;
- the user has completed the documented Unraid smoke for every deploy wave;
- `docs/PROJECT_STATE.md`, the audit report, older overlapping plans, and this
  campaign index agree on what is closed and what remains.

## Rollback posture

- Ship each numbered sub-wave independently when practical.
- Preserve old database columns/tables until the new lifecycle has completed a
  live verification window; destructive cleanup is a later explicit step.
- Put new background loops and readiness enforcement behind config switches
  for their first deploy, but do not leave switches that accept a value and do
  nothing.
- For stream refactors, rollback is the previous owner implementation, not a
  partial mix of old frame rendering and new terminal accounting.
- For non-root containers, resolve bind-mount ownership explicitly before the
  first recreate and keep the prior image available until upload, KB, archive,
  and model-cache smokes pass.
