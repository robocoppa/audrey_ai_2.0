# Campaign 3 Phase 2 — Audrey application and web UI

**Status:** In progress. Milestones 2A and 2B are complete and
Unraid-verified, including provider-neutral identity, canonical application
state, native conversation/run resources, typed and AG-UI events, real
pipeline observations, and rebuildable canonical archive projection.
Milestone 2C's native client and public route are live, with the broader browser
and soak gate still open. Milestone 2D slices 2D.1–2D.4 are Unraid-verified.
The no-landing startup corrective is user-verified; its
loader-transition and Research/Video artwork follow-up awaits redeployment.
The second-user Access handoff and isolated timeout controls also await
redeployment.

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
and a lesson-link scan with zero broken links. On Unraid, schema v3 migrated;
canonical state and preferences survived restart; cross-user reads returned no
resource; a second terminal transition was rejected; and owner purge removed
one conversation, two messages, and one run while resetting preferences. Slice
2A.3 is complete.

### 2A.4 — optional Cloudflare Access identity adapter

Laptop implementation and verification are complete. This slice adds:

- a strict RS256 verifier for Cloudflare Access's
  `Cf-Access-Jwt-Assertion` application token, including signature, issuer,
  application audience, expiry, not-before, subject, email, and token-type
  checks;
- a one-hour JWKS cache with refresh on an unknown signing-key id and a bounded
  refresh interval so attacker-controlled key ids cannot amplify requests to
  Cloudflare;
- fail-fast validation for the HTTPS `*.cloudflareaccess.com` team origin and
  application audience when the adapter is enabled;
- provider-neutral resolution keyed by the verified Cloudflare subject, with a
  new opaque storage namespace and no implicit merge with an OWUI account that
  happens to share the same email;
- Audrey-owned roles and display names: Cloudflare creates an ordinary user and
  cannot promote, demote, or overwrite an existing Audrey profile; and
- strict provider precedence: When enabled and present, an invalid Access
  assertion fails closed rather than falling through to a bearer token.

The adapter is off by default. With `CLOUDFLARE_ACCESS_ENABLED` unset or false,
the assertion header is ignored and the existing OWUI/PAT path is unchanged.
Audrey never accepts the Access cookie. Enabling the adapter also requires
`CLOUDFLARE_ACCESS_TEAM_DOMAIN` and `CLOUDFLARE_ACCESS_AUDIENCE`; no token or
private key is stored in Audrey configuration.

The local gate is 2,669 passing tests, including real generated RSA signatures,
HTTP header extraction, key rotation, invalid claims, service-token rejection,
provider outage behavior, role/profile ownership, email-match isolation, and
the disabled fallback. Scoped ruff, compilation, lockfile, and diff checks are
clean; the lesson-link scan has zero broken links. On Unraid, Audrey started
with the adapter disabled, schema v3 ready, and PyJWT 2.13.0 installed. Both
existing OWUI accounts retained their exact Audrey ids, roles, and provider;
an invalid Access assertion on one authenticated request was ignored as the
disabled contract requires. Slice 2A.4 and Milestone 2A are complete.

### 2B.1 — owner-bound conversation resources and history reads

Laptop implementation and verification are complete. This slice adds:

- native create, list, read, rename, mode-change, archive, unarchive, and
  delete resources under `/api/conversations`;
- cursor-paginated active/archived conversation views ordered by durable
  activity plus cursor-paginated, ascending message history;
- repository transactions that refuse archive or deletion while a run is
  active, then rely on the existing foreign-key cascade once deletion is safe;
- stable-Audrey-id ownership on every query and mutation, with missing and
  cross-owner resources returning the same `404`; and
- provider access plus `compat:full` personal-token access, while response
  models exclude internal owner and provider identifiers.

This is deliberately a read/history and conversation-lifecycle slice. It does
not expose native run creation, cancellation, or event streaming, and it does
not alter `/v1` OpenAI-compatible behavior. The event audit also found that
Wave 1B supplied reusable terminal/stage/channel primitives but not yet one
universal typed event vocabulary: Deep and research still produce ad-hoc event
dictionaries. The next slice must establish that shared spine before adding
the native run and AG-UI adapters.

The focused repository and HTTP gate is 27 passing tests. The full hermetic
suite is 2,677 passing tests; changed-file ruff, compilation, and diff checks
are clean, and the lesson-link scan has zero broken links. On Unraid, two
accounts created independent conversations and received identical `404`s for
cross-owner reads. Four persisted messages paginated in ascending sequence;
active-run archive and delete returned `409`; and rename, mode change,
archive/list, deletion, and post-delete `404` checks passed after the run was
terminalized. Only the two disposable smoke conversations were deleted. Slice
2B.1 is complete and Unraid-verified.

### 2B.2 — universal run events and native run lifecycle

Laptop implementation and verification are complete. This slice adds:

- one typed, client-neutral vocabulary for run, stage, assistant-message,
  answer-delta, tool, source, usage, and terminal events, with a sequenced
  emitter that rejects invalid lifecycle transitions;
- an OpenAI compatibility adapter over those events, while Fast, Deep, and
  Research now distinguish display-only progress from answer text and report
  final-model usage through the shared spine;
- owner-bound native run creation at
  `POST /api/conversations/{conversation_id}/runs`, persisted run reads,
  cancellation, and a typed SSE endpoint with event ids and
  `Last-Event-ID`/cursor replay;
- server-loaded canonical history: The client supplies only the new user text
  and optional product mode/sampling controls, never a prior transcript or
  browser-authored system prompt;
- atomic one-active-run and archived-conversation guards, with missing and
  cross-owner resources remaining indistinguishable `404`s; and
- durable success, cancellation, shutdown, and failure finalization. Startup
  marks rows interrupted by a previous process as failed, and bounded event
  retention cannot truncate the authoritative assistant message.

The first-release reconnect window is intentionally process-local. A completed
or active run can replay retained events while this Audrey process owns it; a
later request for an evicted stream receives `410` and reads the durable run
and messages instead. A restart preserves canonical state and explicitly
terminalizes interrupted rows, but it does not pretend the transient event
buffer survived. The event schema includes tool and source activity; mapping
the remaining internal tool/source observations into the browser protocol and
the AG-UI adapter remain subsequent Milestone 2B work.

The focused affected gate is 229 passing tests. The full hermetic suite is
2,690 passing tests; changed-file ruff, compilation, and diff checks are clean,
and the lesson-link scan has zero broken links. On Unraid, a test owner created,
streamed, resumed, read, cancelled, and deleted native runs and conversation
state while the admin owner received `404` for the test run. Event ids were
ordered, replay honored its cursor, exact answer content and usage persisted,
and cancellation left an explicit incomplete assistant message. A SIGKILL
proved startup recovery terminalizes an interrupted run as
`failed`/`server_restart`; its expired process-local stream returned `410`.
The OpenAI compatibility endpoint still returned valid SSE through `[DONE]`,
and the owner-scoped archive cleanup queue drained to ready. Slice 2B.2 is
complete and Unraid-verified.

### 2B.3 — AG-UI boundary adapter

Implementation is laptop-complete. This slice keeps Audrey's typed `RunEvent`
vocabulary authoritative and adds a narrow protocol adapter that emits the
current AG-UI lifecycle, step, text-message, tool-call, usage, error, and custom
event shapes. AG-UI field names and transport framing remain isolated at the
HTTP boundary so upstream protocol churn cannot leak into pipeline producers
or canonical storage.

The first endpoint is an owner-bound, read-only sibling of the native event
stream at `GET /api/runs/{run_id}/ag-ui-events`, advertised when the run is
created. It reuses the same live-run buffer and authorization rules, emits
standard `data:` SSE frames, and preserves Audrey's cursor/reconnect guarantee
with explicit composite SSE ids. Composite ids let a client reconnect between
the end and result frames produced from one tool event without duplication.
Stage progress and source attribution use documented AG-UI custom events
because neither is a core text-message event. Successful terminal events carry
numeric usage; failed and cancelled runs use sanitized, stable error codes.

The focused adapter/native gate is 24 passing tests, the broader affected gate
is 128 passing tests, and the full hermetic suite is 2,710 passing tests.
Changed-file ruff, compilation, and diff checks are clean. The lesson-link scan
has zero broken links; the existing citation-drift backlog remains deferred by
user direction. On Unraid, a Fast run produced standard data-only AG-UI frames,
the exact reconstructed answer, successful terminal usage, and ordered
composite ids. Reconnect after the answer returned only later events; an admin
identity received the owner-hiding `404`. Cancellation mapped to sanitized
`RUN_ERROR`/`cancelled_by_user`, `/v1` still returned its exact answer, and
three projection chunks plus the canonical conversation were deleted before
all repair queues returned to `ready`. Slice 2B.3 is complete and
Unraid-verified.

This slice does not accept browser-authored AG-UI transcripts or make an
external client the owner of Audrey's tool loop. Archive import/rebuild
behavior remains subsequent Milestone 2B work.

### 2B.4 — pipeline tool and source observations

Implementation is complete and Unraid-verified. One observer at the shared
ReAct dispatch seam now projects real Fast, Deep, and Research activity into
Audrey's existing typed run-event spine. Each dispatch emits a unique tool
lifecycle; successful web and KB retrieval emits deduplicated source records.
The native and AG-UI adapters therefore report what actually happened without
parsing display text, and tool execution remains server-owned.

The projection is deliberately narrower than the model's working context.
Allowlisted search/file arguments are bounded, identity and newly introduced
fields default to redacted, URL credentials/query/fragment data are removed,
raw result bodies never cross the event boundary, and failures expose only
stable error codes. The model still receives the complete dispatch result, so
the safety boundary does not weaken grounding. Compatibility requests create
no observer, and `/v1/chat/completions` retains its existing payload and banner
behavior.

The focused Fast/Deep/Research and projection gate is 96 passing tests; the
broader affected gate is 316 passing tests; and the full hermetic suite is
2,721 passing tests with the existing FastAPI deprecation warning. Changed-file
ruff, compilation, and diff checks are clean. The lesson-link scan has zero
broken links; the existing citation-drift backlog remains deferred by user
direction. On Unraid, Fast, Deep, and Research produced matched native and
AG-UI tool lifecycles and deduplicated sources; sanitized tool failure and
cursor reconnect behavior passed. The final live gate cancelled four active
tools with balanced lifecycle events, mapped the terminal to native
`cancelled_by_user` and AG-UI `RUN_ERROR`, measured one real `web_search`
through unchanged `/v1`, found no typed-event leakage, deleted its archive and
canonical test state, and returned every repair queue to `ready`.

The final live gate is packaged in `scripts/smoke_native_tool_events.py`
instead of a transient shell sequence. It creates one disposable test-owned
conversation, cancels a native Deep run after observing a real active tool,
checks balanced native and AG-UI cancellation events, proves an unchanged
`/v1` request still dispatches `web_search`, deletes its own canonical and
archive state, and waits for the repair queues to return to `ready`.

### 2B.5 — canonical archive projection and repair

Implementation and verification are complete and Unraid-verified. Schema v4
adds durable projection receipts owned by the canonical
application database. A native run's terminal transaction now commits its run,
assistant message, and search-projection receipt atomically. A lifecycle-owned
promoter hands that receipt to the existing local archive outbox with a stable
`native:{run_id}` identity, making retries and administrative replay
idempotent across either process crashing.

Native conversation deletion uses the same ownership rule. Its canonical
transaction records a durable deletion tombstone before removing the
conversation. Promotion serializes that delete behind any active archive write,
discards undelivered writes for the conversation, and hands the owner-scoped
delete to the sidecar's existing durable cleanup queue. Readiness, per-user
repair status, and global repair now include both sides of this handoff. An
admin-only rebuild endpoint resets canonical receipts for bounded replay.

The compatibility boundary is unchanged: Ordinary `/v1/chat/completions`
requests retain their existing archive hook, while native runs suppress that
hook because their canonical terminal transaction is authoritative. Schema-v3
terminal runs are deliberately not replayed during upgrade: Their pre-v4
archive writes used unrecorded random identities, so automatic replay would
duplicate history. Existing sidecar history remains intact and the explicit
historical import/migration in Milestone 2E owns that older-data boundary.

The full hermetic suite is 2,738 passing tests with the existing FastAPI
deprecation warning. Changed-file ruff, compilation, and diff checks are clean.
`scripts/smoke_native_chat_projection.py` packages the live gate: It creates
one disposable test-owned native turn, verifies the two projected messages,
replays all canonical receipts without duplication, deletes only that native
conversation, verifies its projection disappears, and waits for repair to
return to `ready`.

On Unraid, schema v4 started with zero pending canonical projections and the
worker enabled. The packaged gate projected exactly one canonical user and
assistant pair, reset one receipt and replayed it without duplication, then
deleted the canonical conversation with `204`; its read became `404`, its
projected message count became zero, and global repair returned to `ready`.
Slice 2B.5 and Milestone 2B are complete and Unraid-verified.


## Decision

Build a first-party Audrey web application with these boundaries:

- Audrey owns authenticated principals, authorization, conversations, messages,
  runs, tool activity, attachments, preferences, and audit metadata.
- A React and TypeScript single-page application is compiled to static assets
  and served by a dedicated non-root web container on the Audrey browser
  origin. Production does not require a Node server.
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
Audrey browser origin
  +-- audrey-ui static application
  +-- same-origin /api/* and /v1/* streaming proxy
        |
        v
      audrey
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

The deployment decision is to use Access on Audrey's public hostname, including
from LAN browsers when online. A direct LAN connection does not traverse
Cloudflare and therefore does not receive an Access assertion; OWUI and Audrey
personal tokens remain the migration paths for direct access. Do not treat the
mere presence of a forwarded header as proof—the origin always verifies the
signature and claims. A self-hosted OIDC adapter remains the fallback if a
future offline-LAN browser requirement makes the public hostname unsuitable.

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

Implementation status: Complete and Unraid-verified.

### Milestone 2B — conversation, run, and event APIs

- Land the client-neutral event spine from Wave 1B.
- Implement native conversation/message/run resources and server-loaded history.
- Add the AG-UI adapter and cancellation endpoint.
- Persist success, cancellation, disconnect, and failure terminal state.
- Add canonical archive dual-write and rebuildable chat-search projection;
  leave the explicit pre-native history import to Milestone 2E.
- Keep existing OpenAI behavior unchanged.

Gate: fragmentation, ownership, cancellation, failure-injection, archive repair,
and compatibility regression tests pass.

Implementation status: Complete and Unraid-verified.

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

Implementation status: The first 2C slice is complete and Unraid-verified. A pinned
React/TypeScript/Vite workspace builds in a Node stage and ships only static
assets in the Audrey image. `/` is disabled by default, applies restrictive
browser headers when enabled, and leaves OWUI untouched. The native shell lists
and creates owner-bound conversations, renders canonical history, selects and
persists a mode, and sends runs through the standard `/api/agent` AG-UI
endpoint. The browser transport removes prior history, state, identity, and
client tool declarations before sending the newest user action; Audrey reloads
canonical history and authorizes tools server-side.

Typed stages, progress, tools, sources, success, cancellation, and failures are
rendered without parsing compatibility banners. The login shell lazy-loads the
chat runtime as one self-contained production chunk. Three Vitest contracts and
three Chromium Playwright paths
cover same-origin identity, no browser-stored bearer, latest-action transport,
keyboard submission, typed activity, accessibility, cancellation, and session
expiry. All 2,745 hermetic backend tests pass; scoped ruff and compilation are
clean. Generated assets are explicit Hatch wheel artifacts despite their
gitignored build directory, production source maps are disabled, and the image
build asserts that the installed package contains the native shell.
`scripts/smoke_native_ui.py` packages the two-user Unraid gate. The deployed
gate served the 449-byte shell and hashed entry asset with CSP, resolved two
different Audrey users, returned `404` for both cross-owner read and run,
completed one real Fast AG-UI turn with two matching canonical messages, then
deleted both canonical and archive state and returned repair to `ready`.
Milestone 2C remains open for the public Cloudflare Access browser path.

Slice 2C.2 is complete and Unraid-verified. `/api/conversations` now provides owner-scoped,
case-insensitive literal title search and binds each pagination cursor to its
archive and search view. The browser adds debounced server search, active and
archived views, older-page loading, rename, archive, read-only archived
history, restore, and confirmed deletion. Destructive controls and mode changes
are unavailable during an active run. A mode change reloads canonical messages
before replacing its AG-UI agent, fixing the stale-initial-history path that
could hide a just-finished response until reload.

The expanded local gate passes all 2,746 backend tests, three Vitest contracts,
and six Chromium paths covering the original run/auth behavior plus complete
conversation lifecycle, pagination, and durable mode switching. Typecheck,
lint, scoped ruff, compilation, production build, wheel contents, lockfile, and
diff checks are clean; the lesson scan has zero broken links. The extended
`scripts/smoke_native_ui.py` also passed on Unraid: the deployed native asset
and CSP checks were clean, two identities remained isolated, a real Fast run
persisted its canonical messages, rename and literal `%_` title search worked,
archive/restore completed, and deletion cleanup returned repair to `ready`.

Slice 2C.3 is complete and Unraid-verified. It closes a parity gap found while
preparing the every-mode gate: `/v1/models` published `audrey_video`, but
canonical native state, native routes, and the browser accepted only six modes.
Schema v5 widens the two mode constraints with a transactional table rebuild.
The migration
preserves existing conversations, runs, messages, and projection receipts,
checks the rebuilt foreign-key graph before commit, restores foreign-key
enforcement, and recreates the immutable-terminal trigger. Native Video now
launches `audrey_video`, and the browser lists it alongside the other modes.

A permanent invariant requires the native mode map to cover every published
virtual model. `scripts/smoke_native_modes.py` is the repeatable Unraid gate: it
creates one disposable conversation per mode, runs all seven through the same
native AG-UI boundary used by the browser, verifies durable mode/model/message
state, then deletes canonical and archive records and drains repair once. The
local gate passes all 2,749 backend tests, three Vitest contracts, six Chromium
workflows, typecheck, lint, scoped ruff, compilation, production build, wheel
contents, lockfile, and diff checks. On Unraid, all seven modes completed with
the expected virtual model and successful AG-UI terminal event; all seven
canonical/archive cleanup operations returned `204`/`202`, and repair drained
to `ready`. The remaining Milestone 2C evidence is the interactive Cloudflare
Access browser path.
Take an online backup of `audrey_app.sqlite` before the first schema-v5 startup.
A pre-v5 image does not understand persisted `video` rows, so after real native
Video use the rollback choices are to roll forward or restore that snapshot;
the latter discards canonical changes made after the backup.

#### Public Cloudflare Access completion gate

The remaining 2C gate is deployment configuration and an interactive browser
check; it does not require another identity implementation. Create the Access
application before publishing the tunnel route so an ordering mistake never
leaves an unprotected Audrey hostname. In Cloudflare Zero Trust, create a
self-hosted public-hostname application for the whole Audrey hostname, with no
path restriction, and attach an Allow policy for the intended human identities.
Email one-time PIN is sufficient for this gate. Then add a published application
route for the same hostname. Use `http://audrey:8000` when `cloudflared`
shares `ollama-net`; use `http://127.0.0.1:8000` when its container uses host
networking. Cloudflare documents both the
[self-hosted application flow](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/)
and the [tunnel route](https://developers.cloudflare.com/tunnel/setup/).

Copy the application's Audience (AUD) tag and the account team domain into
`CLOUDFLARE_ACCESS_AUDIENCE` and `CLOUDFLARE_ACCESS_TEAM_DOMAIN`, enable
`CLOUDFLARE_ACCESS_ENABLED`, and recreate `audrey`. Do not store an Access
JWT in `.env`. Audrey validates the `Cf-Access-Jwt-Assertion` header Cloudflare
adds at the origin, following Cloudflare's
[JWT validation contract](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/).

The browser gate passes when an allowed user can open `/`, `/api/me`
reports `auth_provider: cloudflare_access`, a native Fast turn survives refresh,
and a second allowed identity cannot see the first identity's conversation.
No Access cookie or assertion should be copied into a terminal or test artifact.
A Cloudflare identity intentionally does not merge with an OWUI identity that
has the same email, and it starts as a normal Audrey user rather than an admin.

The first public session validated the Access assertion and returned a new
`cloudflare_access` Audrey principal, but exposed a production-only circular
chunk between AG-UI and ChatWorkspace. Playwright had previously used Vite's
development server, so the deployed chunk graph was outside the browser gate.
The first corrective removes the unsafe manual split, serves the dedicated
Audrey hostname at clean root `/`, and makes Playwright build and preview the
production artifact. After deployment, the allowed user opened the corrected
root application and a Fast turn returned the exact `2C-ACCESS-READY` response.

The second allowed Access identity subsequently produced a distinct active
Audrey principal and default preference row; direct identity and preference
reads both succeeded. Its first redirects nevertheless hit transient
authentication rejections while the Access session propagated. The client now
resolves identity before preferences and retries `401`/`403` seven times over
about 30 seconds. Callback URLs use a dedicated full-screen portrait handoff
with simple status text and no application chrome; success removes only
Cloudflare's stale message parameter while preserving the rest of the URL.
Only a persistent failure reaches the isolated Retry and Access-logout timeout
screen. Thirteen Vitest contracts and all 19 production-preview Chromium
workflows pass. Redeployment and conversation-isolation verification remain.

The subsequent refresh check exposed a second browser defect: Canonical
messages were assigned to `HttpAgent`, while assistant-ui renders a separate
runtime repository. Selecting another conversation unmounted that repository,
aborted its active stream, and recreated an empty rendered thread even though
Audrey still owned the canonical rows. The corrective now loads canonical rows
through assistant-ui's supported history adapter and keeps each visited thread
mounted for the browser session, hidden when inactive. Conversation selection
therefore neither destroys visible history nor cancels background work; server
canonical state remains authoritative and browser history writes remain no-ops.

All 2,749 backend tests, three Vitest contracts, and seven production-preview
Chromium workflows passed the navigation corrective. The added workflow loads
canonical history, switches away during an active run, returns to the same
progress state, receives the background answer, and repeats the switch after
completion. Public navigation verification is now user-confirmed. Hard-refresh
persistence and second-user isolation remain.

The next native-client refinement is laptop-complete: The supplied Builtryte
wordmark and standalone mark now brand the header and favicon. The application
uses a dark navy interpretation of the kit with accessible blue/violet accents,
restrained gradients, and soft illuminated focus states instead of hard outline
boxes. A larger portrait introduces every Audrey mode before the first prompt,
with one selector immediately beneath it and the selected mode's purpose below.
Once the conversation begins, the portrait and description disappear and that
same selector moves inside the composer beside the input. This removes the
redundant Audrey/mode label and preserves message space throughout the thread.
User and assistant text render safe CommonMark plus GitHub-flavored Markdown,
with raw HTML ignored.

The empty thread now collapses its introductory space so the complete composer
fits in the initial 720px browser viewport. The header reads `Ask Audrey`, the
redundant center heading is gone, the input uses the same wording, the mode
description and selector are larger, and the send control uses an unambiguous
arrow icon. The introductory portrait is 20 percent larger and rendered at 80
percent opacity with a short gap above the selector. The restored full
Builtryte wordmark is centered over the desktop conversation rail's exact
17rem column while `Ask Audrey` remains separately positioned in the main
header; the compact mobile header keeps its flex layout. A transparent
true-vector favicon in source blue `#3499fa` is available for the Cloudflare
page, whose matching Audrey background color is `#07101f`.

The authenticated shell now owns one viewport-height layout: conversation
messages are the scrolling region while the complete composer dock remains at
the bottom of the screen. Scrolling meaningfully above the newest message
reveals an accessible down-arrow immediately above the dock; activating it
returns the thread to the latest message, where the control disappears. This
uses assistant-ui's thread scroll state rather than a second competing browser
scroll listener.

New native conversations are created without a canonical title. Before the
first run is committed, the small configured title model labels the prompt with
a 3–7 word noun phrase describing its central topic or task. The full auxiliary
operation, including capability detection and both scheduler queues, is bounded
to ten seconds; any failure uses the previous deterministic cleaned-prompt
title, so a title outage cannot prevent the chat from starting. The final choice is capped
at 72 characters and committed atomically with the first messages. Later
prompts cannot replace it, and a title explicitly supplied or edited by the user
is never auto-renamed. The browser refreshes the canonical conversation when
the server confirms `RUN_STARTED`, so the summary appears in the header and
sidebar without waiting for a reload or the answer.

The header shows the stored profile's first name when one exists, falls back to
the account handle, and provides Cloudflare Access's same-origin logout link.
The account label also opens a self-service profile-name editor. Its provider-
authenticated `PATCH /api/me` changes only the current Audrey user's display
name, trims and bounds the value, rejects personal access tokens, evicts stale
authentication cache entries, and prevents later OWUI or Cloudflare logins from
overwriting the Audrey-owned value. Clean `npm ci`, typecheck, lint, five Vitest
contracts, ten production-preview Chromium workflows, the production build,
37 focused title/canonical-run contracts, 66 focused identity/authentication
contracts, and all 2,759 backend tests pass.
The required lesson-link scan reports zero broken links; its existing stale-line
backlog remains deferred by user direction. This refinement still needs its
Unraid browser smoke.

The frontend deployment boundary is now laptop-complete. `web/` owns its model
artwork, production output, Dockerfile, and proxy template without reading or
writing outside that directory. A pinned, non-root NGINX runtime serves the SPA
and forwards `/api/*` and `/v1/*` to `audrey` on `ollama-net`, including the
Cloudflare Access assertion, unbuffered SSE, and streamed upload bodies. Compose
publishes the UI only on loopback port 8090 for the host-network tunnel; 8088
remains assigned to SearXNG. The
backend image keeps an embedded copy for one transition release; after the live
soak, the frontend can move intact to its own GitHub repository before that
fallback is removed. See the
[standalone UI deployment runbook](phase-02-standalone-ui-deploy.md).

Rollback removes or disables the public tunnel route first, then disables the
Audrey Access flag and recreates `audrey`. Disabling the Access application
while leaving its tunnel route public is not a safe rollback.

### Milestone 2D — files, preferences, and ownership operations

- Add native upload and attachment lifecycle.
- Move timezone, persona, and presentation preferences into Audrey.
- Add search, export, deletion, token management, and capability health.
- Expose the Phase 3 skills placeholder.

The first 2D slice is complete and Unraid-verified. Owner-bound `/api/files` resources now
reuse the existing hardened upload lifecycle for listing, single-request and
chunked uploads, inspection, quota accounting, and durable deletion. The
browser adds a same-origin file manager with upload progress and confirmed
removal; it never stores or supplies a bearer token.

Schema v6 adds attachment snapshots keyed through the canonical message owner.
Native and AG-UI runs accept at most ten opaque file ids, resolve them against
the authenticated owner's ready-file listing, and commit their safe metadata in
the same transaction as the user message. Canonical text remains exactly what
the user typed. Audrey alone creates the model-facing manifest, which names the
attached files, requires authorized file/knowledge tools for their contents,
and treats filenames and retrieved content as data rather than instructions.
Message history renders the durable snapshots even if the source file is later
removed, while access to its contents correctly follows the live file state.

The native composer now selects ready owner files, sends only their ids through
the minimized AG-UI envelope, and renders saved attachment names on history
reload. The mode portrait is also twice its prior display size without pushing
the composer outside the initial browser viewport. All 2,767 backend tests,
eight Vitest contracts, and twelve production-preview Chromium workflows pass;
typecheck, lint, scoped ruff, compilation, production build, and diff checks are
clean. The lesson scan has zero broken links; its existing drift backlog remains
deferred by user direction. `scripts/smoke_native_files.py` packages the live
gate with disposable randomized data: It proves two-owner isolation, upload and
safe listing, a tool-grounded attached turn, exact canonical text and metadata,
snapshot survival after source deletion, and repair cleanup. Deployment and
the live lifecycle gate passed: a ready 61-byte owner file produced one durable
attachment, the Fast run made one complete tool call and recovered its private
marker, and the second owner received identical `404`s for the file and
conversation. Source deletion returned `200` without erasing the message
snapshot; canonical/archive cleanup returned `204`/`202`, and repair returned
to `ready`.

The 2D.2 slice moves account preferences behind owner-bound
`GET/PUT /api/me/preferences`. Timezone is validated as an IANA name; persona,
response detail, tone, and live-progress visibility have bounded typed
contracts. Preference replacement requires provider authentication, while a
scoped personal token may read but cannot mutate this account-level state. The
dark native settings surface edits profile and Audrey preferences without
storing a browser bearer token, and the saved progress choice is applied to the
chat workspace immediately.

Every native run now receives a server-built preference context containing its
server-computed local time and response guidance. It is never accepted as a
browser-authored system message or persisted as conversation content. A
separate routing transcript keeps even a long persona out of classification,
the Auto complexity gate, and Deep/Research planning thresholds while still
letting the selected model see it. All 2,773 backend tests, nine Vitest
contracts, and fourteen production-preview Chromium workflows pass; scoped
ruff, compilation, typecheck, lint, and production build are clean. The lesson
scan has zero broken links; its deferred 99 hard and 162 advisory drifts remain
out of scope by user direction. `scripts/smoke_native_preferences.py` packages
the two-owner live gate, including invalid-timezone atomicity, model-context
retrieval, canonical persistence, owner isolation, exact preference restoration,
and repair cleanup.

The Unraid gate passed through the standalone UI proxy. Invalid IANA input was
atomic; a 3,003-character persona and saved timezone reached the Fast model in
a successful 12-event native run; the second owner remained unchanged; and the
test owner's exact preferences were restored. Canonical/archive cleanup returned
`204`/`202`, and repair returned to `ready`. Slice 2D.2 is complete and
Unraid-verified. Milestone 2D remains open for the remaining ownership and
capability operations.

Slice 2D.3 brings the existing owner-bound personal-token lifecycle into native
Settings. The browser lazily lists active secret-free metadata, creates tokens
with explicit `account:read` and/or `compat:full` scopes and mandatory
1–365-day expiry, displays a new secret only until the user dismisses it, and
requires confirmation before revocation. No token is written to local or session
storage, and provider authentication remains mandatory for every management
operation.

The browser contract covers the real create/list/revoke shapes, one-time secret
handling, and storage isolation. Its production-preview workflow also caught and
removed a duplicate banner landmark from the Settings dialog. All 2,776 backend
tests, five native-UI packaging contracts, 11 Vitest contracts, and 16
production-preview Chromium workflows pass; typecheck, lint, production build,
and diff checks are clean. The lesson scan has zero broken links; its deferred
99 hard and 162 advisory drifts remain out of scope.

The Unraid gate passed through the standalone UI proxy. A disposable one-day
token created through native Settings authenticated its owner at `/api/me`;
after revocation, the same request returned `401`. Slice 2D.3 is complete and
Unraid-verified.

Slice 2D.4 exposes the hardened Phase 1 current-user data controls through
native Settings. Chat export follows every opaque pagination cursor, rejects a
repeated cursor or mid-export schema change, and downloads one timestamped JSON
artifact. The UI explicitly identifies the boundary: This is the current
chat-search archive, not uploaded file contents, memories, or canonical turns
still awaiting archive delivery.

Full Audrey-data deletion remains provider-authenticated on the server. The
browser requires the exact `DELETE ALL MY AUDREY DATA` phrase, supplies one
stable per-attempt idempotency key, and never accepts a user selector. After
the request settles, the native workspace remounts and refreshes account state
because the local transactional purge intentionally runs before downstream
durable cleanup and can therefore succeed even if the final request reports an
error. An accepted receipt replaces the old Settings forms and polls its exact
owner-bound status through pending, completed, or attention-required cleanup.
The Audrey identity and profile remain and preferences reset; conversations,
runs, tokens, uploads, memories, and derived chat history are deleted.

The full hermetic backend suite passes all 2,776 tests; its focused packaging/
user-data regression passes 32. All 12 Vitest contracts and 18 production-
preview Chromium workflows pass, including a real Blob download, pending-to-
complete status polling, workspace reset, no browser bearer storage/header,
and an axe audit. Typecheck, lint, production build, and diff checks are
clean.

The native conversation surface no longer contains the “What shall we work
through?” landing state. A first startup with no active conversations creates
and opens a blank Auto thread, while a hard refresh remains on the Audrey loader
until the saved canonical thread and messages return. Production-preview browser
coverage deliberately delays that reload response and proves the landing state
never appears. The deployed behavior is user-confirmed.

The next corrective keeps the centered full-screen loader but renders no portrait
during workspace and canonical-message loading, so Audrey appears again only in
her final composer position. Research uses `audrey8`; Video uses `audrey9`. All
19 production-preview Chromium workflows pass.

The disposable-account Unraid gate exported its seeded archive as a schema-v1
artifact containing one complete user/assistant turn with consistent model and
usage metadata and no partial rows. The exact-owner deletion flow passed its
remaining checks. Repair status briefly showed one pending chat deletion and
one pending account purge with four aggregate attempts but zero errors or
exhaustion, then reached `ready` on refresh as the asynchronous workers drained.
Slice 2D.4 is complete and Unraid-verified.

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

Production runs a dedicated static frontend service and keeps application state
in Audrey. During the transition, the backend retains the embedded shell and
OWUI remains separately routable, so rollback changes traffic rather than
rewriting data.

## Decision gates before implementation

1. Decided: Use Cloudflare Access on the public Audrey hostname. Direct LAN
   traffic does not traverse Access, so keep OWUI/PAT migration access and retain
   the same principal boundary for a future self-hosted OIDC adapter if offline
   LAN browser access becomes a requirement.
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
- self-contained web/ for the React and TypeScript client, production proxy,
  and container, prepared for extraction after its live gate;
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
