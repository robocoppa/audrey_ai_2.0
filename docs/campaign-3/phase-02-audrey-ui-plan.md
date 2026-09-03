# Campaign 3 Phase 2 — Audrey application and web UI

**Status:** In progress. Slices 2A.1 and 2A.2, including the account-purge
lifecycle corrective, are complete and Unraid-verified. Slice 2A.3 canonical
application state is laptop-complete and awaits its Unraid gate.

## Goal

Make Audrey the product application, not an OpenAI-compatible backend hidden
behind another agent platform.

The native web client should make chat, tools, files, memory, modes, sources,
and later skills feel like one Audrey system. Open WebUI remains a migration
client while the native surface reaches parity; it is not the future source of
identity, conversation state, prompt behavior, or tool continuation.

## Current implementation slice

### 2A.1 — provider-neutral identity foundation

Laptop implementation and verification are complete. The deployed two-account
identity, restart-persistence, and existing-file compatibility checks all pass.
This slice adds:

- a frozen `Principal` with Audrey-owned user and storage identifiers;
- an Audrey application database with versioned SQLite/WAL migrations for users
  and external provider identities;
- an OWUI migration adapter keyed by OWUI subject rather than email similarity;
- exact legacy email namespaces for first-seen OWUI users, stable across later
  email changes, with implicit account merges refused;
- authenticated `GET /api/me` exposing the stable Audrey id and safe profile
  fields without provider subjects or storage namespaces; and
- fail-fast validation for the application database path.

Existing `/v1` routes remain email-keyed in this rollback-safe first slice. No
files, memories, collections, archives, or OpenAI-compatible behavior have been
migrated yet. The local gate is 2,613 passing tests, scoped ruff and compilation
clean, valid YAML, and a lesson-link scan with zero broken links.

The two-account Unraid smoke proves distinct identities, stable repeat identity,
restart persistence, active status, the safe `/api/me` projection, and unchanged
`/v1/files` behavior. Slice 2A.1 is complete.

### 2A.2 — Audrey personal access tokens

Laptop and deployed verification are complete. This slice adds:

- application schema v2 with owner-bound personal-token records;
- 256-bit random bearer secrets stored only as SHA-256 digests;
- explicit `account:read` and coarse `compat:full` scopes, mandatory expiry
  defaulting to 90 days and bounded to 1–365 days, last-use tracking, idempotent revocation, and one-time secret display;
- provider-authenticated create/list/revoke resources at `/api/tokens`;
- local token resolution through the same stable `Principal`, without sending
  Audrey tokens to OWUI or placing them in the OWUI auth cache; and
- compatibility access through the stable storage namespace while keeping
  admin routes and credential management provider-authenticated.

The additive v1-to-v2 migration preserves existing users and identifiers. The
local gate is 2,628 passing tests, scoped ruff and compilation clean, and a
lesson-link scan with zero broken links. The Unraid gate proved schema migration,
ordinary-user issuance, one-time secret handling, native and compatibility use,
restart persistence, owner-bound management, and immediate revocation. Slice
2A.2 passed its primary gate.

A post-gate lifecycle audit found that the existing account-wide purge neither
removed personal-token records nor excluded a `compat:full` token from starting
the destructive operation. The corrective requires provider authentication for
account-wide purge and erases every personal token owned by that account before
the existing durable purge begins. Its local gate is 2,630 passing tests with
the same clean static checks. On Unraid, a PAT received `403` before the purge
coordinator ran; provider-authenticated purge then completed without component
errors, erased all token records, and invalidated the bearer immediately.
Slice 2A.2 is complete.

### 2A.3 — canonical application state

Laptop implementation and verification are complete. This slice adds:

- additive schema v3 tables for user preferences, conversations, messages, and
  runs, kept separate from the existing sidecar archive projection;
- typed preference and conversation repositories keyed only by stable Audrey
  user ids, with validated IANA timezones and JSON response preferences;
- Audrey-issued `con_`, `run_`, and `msg_` identifiers plus transactionally
  assigned per-conversation message sequence numbers;
- one transaction that creates a running run, its completed user message, and
  its in-progress assistant message before streaming begins;
- one transaction that retains assistant output and records success,
  cancellation, or failure, with both repository-level race handling and
  schema-level terminal-outcome immutability;
- composite foreign keys that reject cross-user conversation/run/message
  linkage even under direct SQL; and
- owner-bound local purge that removes tokens and canonical conversation state
  and resets preferences before the existing durable remote purge starts.

No native conversation routes or compatibility dual-write are enabled in this
slice. The local gate is 2,642 passing tests, scoped ruff and compilation clean,
and a lesson-link scan with zero broken links. Deployed schema, restart,
repository, isolation, and purge verification remain pending.


## Decision

Build a first-party Audrey web application with these boundaries:

- Audrey owns authenticated principals, authorization, conversations, messages,
  runs, tool activity, attachments, preferences, and audit metadata.
- A React and TypeScript single-page application is compiled to static assets
  and served from the Audrey origin. Production does not require a Node server.
- The browser uses a native Audrey API and AG-UI event stream. It does not use
  Chat Completions as its application protocol.
- Audrey emits one typed, client-neutral run-event stream. Native AG-UI and
  OpenAI-compatible SSE are sibling adapters over those events.
- Cloudflare Access is the preferred browser identity provider. Audrey validates
  its signed token and maps the provider subject to a stable internal user.
- Audrey personal access tokens support evals, LAN clients, and automation.
- Open WebUI authentication remains a temporary provider during migration, then
  is removed from Audrey's runtime dependency graph.
- SQLite in WAL mode remains the initial transactional store for this
  single-node deployment. Repositories and migrations preserve a later
  PostgreSQL path if measured scale or availability needs justify it.
- Qdrant remains a derived semantic index, never the authoritative record for
  conversations or user ownership.

A stopped or broken Open WebUI instance must eventually have no effect on
native Audrey sign-in, chat history, uploads, tools, or administration.

## Target topology

~~~text
browser
  |
  | Cloudflare Access identity
  v
Audrey origin
  +-- static web application
  +-- /api/* native resources
  +-- /api/agent AG-UI event stream
  +-- /v1/* OpenAI compatibility adapters
  |
  +-- application database
  |     users, identities, tokens, preferences
  |     conversations, messages, runs, tools, attachments, sources
  |
  +-- orchestration and typed run events
  +-- Ollama / cloud model providers
  +-- Qdrant derived indexes
  +-- connector services for external systems

Open WebUI
  +-- temporary /v1 client and identity provider only
~~~

## Responsibility map

| Concern | Long-term owner | Browser responsibility |
|---|---|---|
| Authentication evidence | Cloudflare Access or Audrey token issuer | Present same-origin session |
| User identity and authorization | Audrey | Never supply an authoritative user id |
| Conversation and message history | Audrey database | Request and render pages |
| Pipeline mode and model policy | Audrey | Select a published product mode |
| Tool loop and permissions | Audrey | Render activity and approvals |
| Files and attachment ownership | Audrey | Upload once; reference durable ids |
| Timezone and persona | Audrey preferences | Capture and edit preferences |
| Run lifecycle and cancellation | Audrey | Start, observe, cancel, reconnect |
| Semantic search | Audrey policy plus Qdrant index | Display results and sources |
| OpenAI wire compatibility | Audrey adapter | External compatible clients only |
| Layout and interaction state | Native web client | Own ephemeral presentation state |

## Architectural invariants

- No UI-provided email, user id, role, or storage namespace is trusted.
- Email is a mutable attribute, not Audrey's durable user key.
- Provider identities and Audrey tokens resolve to one typed principal before
  product routes run.
- The server loads canonical conversation history. The browser sends the new
  action, not an editable transcript containing system, assistant, or tool
  messages.
- Conversation ids are Audrey-issued resource ids. Conversation creation
  returns the id, and every later run, mutation, archive projection, URL, and
  deletion carries that exact id; the native path never derives authoritative
  identity from first-turn text or a separately generated display URL.
- Files are uploaded once and referenced by durable attachment ids.
- Every run has stable run, message, and tool-call ids and exactly one terminal
  outcome.
- Tool authorization is enforced when tools are offered and again when they are
  dispatched.
- Compatibility adapters cannot become the source of internal business logic.
- Authoritative records are transactional; search indexes and analytics are
  repairable projections.
- Cross-user resources use the same external not-found behavior as missing
  resources.
- Native browser sessions do not keep long-lived bearer secrets in local
  storage.

## Client and protocol choices

### Web stack

Use React, TypeScript, and Vite for a small static application. Use
[assistant-ui](https://github.com/assistant-ui/assistant-ui) for accessible,
composable chat primitives and its
[AG-UI runtime](https://www.assistant-ui.com/docs/runtimes/ag-ui/overview) for
stream integration.

Adopt the library behind Audrey-owned interfaces. Audrey REST resources remain
authoritative for conversation listing, search, pagination, and mutations. If
the AG-UI adapter cannot express an Audrey event without loss, extend the
native transport rather than encode state into display text.

### Internal event spine

Campaign 3 Phase 1 Wave 1B introduces a typed event vocabulary at the
orchestration boundary:

- run started;
- stage started and stage finished;
- assistant message started;
- text delta;
- tool started, arguments available, and tool finished;
- source observed;
- usage reported;
- message finished;
- run succeeded, cancelled, or failed.

[AG-UI events](https://docs.ag-ui.com/concepts/events) are the native browser
adapter. Existing OpenAI chunks and Audrey's compatibility banners are derived
in a separate adapter. Neither adapter parses the other's serialized output.

The first release may use HTTP POST plus server-sent events. Durable resume is
not required for the first vertical slice, but event ids and run state must
allow a later reconnect endpoint without replacing the schema.

## Identity and authorization

### Principal model

Resolve every request to a provider-neutral principal containing:

- stable Audrey user id and private-storage namespace;
- provider and provider subject;
- current email and display name where available;
- Audrey role and account status;
- authentication method and token id where applicable.

Store provider bindings separately from users. A user can later replace an
identity provider without renaming private collections or losing history. Do
not normalize punctuation in email addresses or merge accounts on a
similar-looking address.

Existing users retain their exact current storage namespace during migration.
New users receive an opaque namespace. Any later namespace rewrite must be an
explicit, journaled migration with rollback evidence.

### Browser and API authentication

Prefer Cloudflare Access because it already protects the public tunnel. Audrey
validates token signature, issuer, audience, expiry, and provider subject as
described in Cloudflare's
[JWT validation guide](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/).
The assertion proves authentication; Audrey still decides status, roles,
quotas, and resource access.

If Access is unsuitable for LAN-only use, select a self-hosted OIDC provider
through the same adapter. Do not add a local password database as a shortcut.

Issue Audrey personal access tokens with hashed-at-rest secrets, explicit owner
and scopes, expiry, last-use, revocation, and one-time secret display. During
migration, OWUI validation maps to the same principal interface and can never
override an already validated identity.

Use same-origin secure sessions, narrow CORS, CSRF or strict-origin protection,
a restrictive content security policy, secure cookies, upload limits, and
server-side validation.

## Audrey-owned application state

Add migrations and repositories for:

- users, external identities, personal tokens, and preferences;
- conversations, messages, runs, and terminal run metadata;
- tool invocations and approvals;
- attachments and message-attachment links;
- sources and message-source links.

SQLite WAL is the default for Audrey's single-node, low-concurrency deployment.
Keep access behind repositories, use versioned migrations, define transaction
ownership, and measure lock time. Move to PostgreSQL only for multiple Audrey
replicas, high availability, or observed write contention.

The current custom-tools chat archive becomes an import source and derived
search projection. New authoritative chat writes belong to Audrey. Qdrant
stores embeddings keyed by Audrey record ids and can be rebuilt.

## Native API and context contract

The first stable resource contract includes:

- GET and PATCH /api/me;
- GET /api/capabilities, /api/modes, and later /api/skills;
- list, create, read, rename, archive, and delete conversations;
- paginated message retrieval;
- POST /api/agent and POST /api/runs/{run_id}/cancel;
- upload, inspect, and delete file resources;
- create, list, and revoke personal access tokens;
- account export and deletion before cutover.

A native run identifies the conversation, new user content parts, product mode,
attachment ids, and optional explicit skill. It does not accept a
browser-authored system prompt or prior assistant/tool transcript.

Publish modes as Auto, Fast, Deep, Research, Local, and Cloud rather than
exposing virtual model implementation names. The capabilities endpoint tells
the UI what is healthy and available.

Audrey creates durable user and assistant/run records before streaming.
Completion, cancellation, and failure update those records transactionally.
Partial content may be retained with an explicit status.

Timezone is a validated IANA preference and the server computes local time.
Persona and response preferences are stored in Audrey. OWUI utility prompts and
client-injected system text are not part of the native contract.

## Files, tools, memory, and skills

The browser uploads a file once, receives an owned attachment id, and references
that id in a message. Audrey validates quota and ownership and ties derived
text, images, and embeddings to the same deletion lifecycle.

The native client never continues Audrey's internal tool loop. Audrey chooses,
authorizes, executes, and records server tools; the UI renders structured tool
events. External OpenAI-compatible clients may keep protocol-defined
client-side continuation behavior.

Move durable memory and chat-history ownership into Audrey after the native
state foundation is proven. Keep the connector sidecar for independently
deployable external capabilities such as web search/fetch and future MCP
connectors.

Phase 2 exposes an empty or disabled skills capability contract so Phase 3 can
add selection without redesigning the client. Phase 3 owns registry, bundle,
enforcement, and evaluation behavior.

## Native UI scope

The first production UI includes:

- responsive authenticated shell and account menu;
- conversation list, search, create, rename, archive, and delete;
- message thread with Markdown, code, sources, images, and errors;
- composer with attachments, cancel, retry, and mode selection;
- structured planning/stage and tool activity;
- file, preference, and personal-token management;
- accessible keyboard navigation, focus handling, and screen-reader labels;
- honest offline, reconnect, degraded-capability, and terminal error states.

Voice, mobile apps, marketplaces, collaborative workspaces, branching threads,
and rich artifacts are non-goals for the first cutover.

## Delivery milestones

### Milestone 2A — identity and application-state foundation

- Add the principal/auth-provider boundary and stable user/storage identifiers.
- Add migrations and repositories for users, identities, tokens, preferences,
  conversations, messages, and runs.
- Map current OWUI users without renaming existing private storage.
- Add Cloudflare Access behind a disabled flag and personal access tokens.
- Prove two users cannot access or mutate each other's resources.
- Prove rollback restores the OWUI-only path without data loss.

Gate: focused tests, the full hermetic suite, disposable/copy migrations, and a
user-run two-account Unraid smoke.

### Milestone 2B — conversation, run, and event APIs

- Land the client-neutral event spine from Wave 1B.
- Implement native conversation/message/run resources and server-loaded history.
- Add the AG-UI adapter and cancellation endpoint.
- Persist success, cancellation, disconnect, and failure terminal state.
- Add archive import/dual-write and rebuildable chat-search projection.
- Keep existing OpenAI behavior unchanged.

Gate: fragmentation, ownership, cancellation, failure-injection, archive repair,
and compatibility regression tests pass.

### Milestone 2C — native vertical slice

- Lock the TypeScript/Vite web workspace and reproducible dependency graph.
- Serve built assets from Audrey with development proxy support.
- Implement session, conversations, thread, composer, modes, stages, tools,
  errors, and cancellation.
- Exercise every published mode through native resources.
- Keep the UI behind a feature flag and separate route during evaluation.

Gate: no browser-stored API secret; two-user browser isolation; tools need no
client continuation; accessibility checks; Playwright happy, failure, and
cancellation paths; user-confirmed Unraid smoke.

### Milestone 2D — files, preferences, and ownership operations

- Add native upload and attachment lifecycle.
- Move timezone, persona, and presentation preferences into Audrey.
- Add search, export, deletion, token management, and capability health.
- Expose the Phase 3 skills placeholder.

Gate: disposable-user upload/search/export/delete evidence covers transactional
and derived stores, including interrupted cleanup and restart.

### Milestone 2E — parallel migration and parity

- Run native Audrey UI and OWUI side by side.
- Import current archive records with stable provenance and idempotent reruns.
- Use a supported explicit OWUI export importer only if needed; do not depend on
  undocumented direct reads from its database.
- Compare answer, tool, source, image, history, and mode behavior.
- Publish deployment, backup, rollback, and recovery instructions.

Gate: the user's normal workflow completes in the native UI for a defined soak
period without returning to OWUI for a missing core capability.

### Milestone 2F — cutover and dependency removal

- Make the native application the public route.
- Disable OWUI authentication after browser/token migration.
- Remove native-path OWUI utility, chat-id, token-storage, datetime, and
  full-history image-resend special cases.
- Complete first-party chat-history and memory ownership.
- Preserve /v1 compatibility for external clients and evals.
- Retain a time-bounded rollback deployment, then stop OWUI.

Gate: stopping OWUI has no effect on native authentication, conversations,
files, memory, tools, or administration; rollback and restore work on Unraid.

## Verification and operations

Backend tests cover provider verification, principal resolution, namespaces,
repository ownership, migrations, transactions, export/deletion, event ordering
and fragmentation, cancellation, one terminal outcome, canonical history,
compatibility parity, upload quotas, and derived-data deletion.

Frontend and Playwright tests cover event rendering, conversations, composer,
attachments, modes, retry/cancel, auth expiry, degraded capability, reconnect,
keyboard access, two isolated users, a server-side tool call, refresh/history,
export, and deletion. Live evals continue to assess deployed structure and
human-reviewed answer quality.

Add bounded-cardinality metrics and structured logs for authentication outcome,
runs, stream disconnects, database lock time, uploads, tool duration, projection
lag, and repair. Never log token material or message content.

Back up the application database and attachment metadata with a restore test.
Qdrant projections must be rebuildable. Schema migration is an explicit
deployment step with a recorded prior version and rollback limit.

Production adds static assets but no always-on frontend service. During
migration, OWUI remains separately routable so rollback changes traffic rather
than rewriting data.

## Decision gates before implementation

1. Confirm Cloudflare Access supports public and practical LAN access; otherwise
   use the same principal boundary with a self-hosted OIDC provider.
2. Spike assistant-ui plus AG-UI against text, stage, source, and tool events.
   Retain it only if no core Audrey state is lost.
3. Benchmark SQLite WAL with representative run, archive, and upload writes.
   PostgreSQL requires evidence.
4. Choose terminal summaries or a bounded durable event log for MVP reconnect.
   Stable event ids are required either way.
5. Import only supported exports or Audrey's current archive; do not bind to
   undocumented OWUI internals.

## Likely repository shape

- src/audrey/identity/ for principals, providers, and personal tokens;
- src/audrey/app_state/ for migrations and repositories;
- src/audrey/events/ for typed events and protocol adapters;
- src/audrey/routes/app/ for native resources;
- web/ for the React and TypeScript client;
- focused backend and Playwright tests.

Existing orchestration, OpenAI routes, storage, archive, uploads, configuration,
metrics, Compose, and static serving change incrementally at milestone
boundaries. This is a map, not permission for a cross-cutting rewrite.

## Phase 2 completion gate

Phase 2 is complete when:

- Audrey owns stable identity, authorization, preferences, conversations,
  messages, runs, attachments, and personal tokens;
- the native UI covers the sole-owner daily workflow in supported modes;
- canonical history and the server-side tool loop do not depend on browser
  behavior;
- browser and Audrey-token auth pass two-user deployment evidence;
- stopping OWUI does not affect the native application;
- /v1 compatibility remains regression-tested;
- backups, migrations, export/deletion, rollback, and index rebuild are proven;
- backend, frontend, Playwright, live-eval, and user-run Unraid gates pass.

Only then does Campaign 3 Phase 3 make skills visible as a native Audrey
capability.
