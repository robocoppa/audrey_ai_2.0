# Campaign 3 Phase 1 — platform hardening

**Status:** In progress. Waves 1A, 1B, and 1C are complete and
Unraid-verified. Wave 1D.1–1D.2 are Unraid-verified; Wave 1D.3 is
laptop-verified and awaits its Unraid smoke.

## Goal

Strengthen Audrey's privacy boundaries, request lifecycle, protocol
compatibility, durable storage behavior, degraded operation, and deployment
reproducibility before adding another extensibility layer.

The detailed engineering assessment and finding-level remediation notes are
laptop-local working documents. This public plan records product outcomes,
sequence, and verification gates without publishing internal security analysis.

## Delivery model

Phase 1 ships as independently verifiable waves, not one large rewrite. Each
wave has:

- focused regression and failure-injection tests;
- the full hermetic test gate;
- lint and lesson-link verification for changed source;
- a separate deployment boundary and rollback point;
- user-run Unraid smoke tests before it is marked complete.

## Product invariants

- Authenticated identity remains server-owned.
- Private data access is scoped below model instructions.
- The documented complexity-routing order remains unchanged.
- Client disconnects stop work owned by that request.
- Tool side effects are not blindly retried.
- Existing storage remains recoverable through migrations and rollback windows.
- Optional component failure does not unnecessarily disable unrelated features.
- Laptop tests do not stand in for deployed-stack verification.

## Wave 1A — data boundaries and request ownership

Make private search scoping consistent across every query mode and give every
spawned request task one explicit owner. Cover GPU scheduling, deep execution,
research execution, queue shutdown, and client cancellation with deterministic
tests.

Laptop verification passed on 2026-08-25: 163 focused lifecycle tests and all
2,413 hermetic tests pass; changed-file lint is clean. The user-run deployment
smoke then cancelled Deep during panel dispatch, and an immediately following
Fast request completed exactly, confirming the local slot was available.

Gate:

- private text, hybrid, filtered, and image query isolation tests pass;
- cancellation at every generation stage leaves no child work or held slot;
- ordinary requests still follow the existing routing behavior;
- laptop verification and the focused Unraid smoke pass.

## Wave 1B — protocol and streaming consistency

Complete the supported Chat Completions message lifecycle and consolidate
stream ownership so fast, deep, research, and passthrough responses share clear
terminal behavior, identity, metrics, and archive boundaries.

The durable endpoint is a typed, client-neutral run-event stream. OpenAI SSE
frames, progress banners, archive capture, metrics, and the future native UI
must adapt from those events rather than parsing or wrapping one another. This
keeps OpenAI compatibility important without making it Audrey's internal
application protocol.

First-slice status (2026-08-25): role-aware request validation and the
OpenAI-to-Ollama tool-call/result translation are Unraid-verified. The deployed
Open WebUI passthrough completed a client-provided timestamp tool call and
rendered its exact result instead of hanging. Streaming and non-streaming
two-request regressions pass; all 2,423 hermetic tests pass, changed-file lint
is clean, and the lesson check has no broken links. M2 and M1 status follow;
shared deep/research stage mechanics remain the final Wave 1B slice.

M2 status (2026-08-25): passthrough streaming now reports one typed `ok`,
`error`, `cancelled`, or `truncated` result from the inner generator to the
outer metric owner. Missing Ollama `done` produces one `length` finish frame
instead of a successful-looking bare terminator. Normal completion,
pre-token/mid-stream failure, missing terminal data, and cancellation pass 48
focused tests; all 2,429 hermetic tests pass, changed-file lint is clean, and
the lesson check has no broken links. On Unraid, a completed passthrough
incremented `outcome="ok"` while an interrupted live generation incremented
`outcome="cancelled"`, verifying the two terminal paths separately.

M1 status (2026-08-26): plain Fast streaming now has one
`OpenAIStreamSession` for response identity and OpenAI framing, backed by a
client-neutral attempt stream that owns model selection, bounded pre-token
fallback, thinking policy, GPU gating, health, metrics, and terminal outcome.
It never swaps models after answer text reaches the client. Archive capture
stays outside banners, missing terminal data is partial/truncated, and the
observed dangling `</think>` answer replay is filtered. The 99 focused stream
tests and all 2,440 hermetic tests pass; changed-file ruff and compilation are
clean, lesson conventions report zero findings, and the link check has zero
broken links. On Unraid, Open WebUI rendered `M1-OWUI-READY` exactly once
without a thinking-tag leak, and the Fast pipeline recorded `outcome="ok"`.
The streaming agent-client case passed on route Fast with 0.5-second TTFT,
1.5-second total latency, no truncation, and no reasoning leak. M1 is
Unraid-verified.

1B.4 status (2026-08-26): complete and Unraid-verified. `StreamStageRunner`
now owns the named deep/research child tasks, bounded
producer delivery, cancellation/drain, one terminal result, pipeline metrics,
and answer-only archive finalization. The deep panel and staged research state
machines still interpret their own events and banners. Both use one
`OpenAIStreamSession`; a producer that ends after visible text but without its
terminal event is recorded as truncated and archived partial. All 51 adjacent
streaming tests and all 2,446 hermetic tests pass; scoped ruff and compilation
are clean, lesson conventions report zero findings, and the link check has zero
broken links. On Unraid, the Deep CRISPR agent-client case passed on route Deep
with the Planning → Dispatching panel → Synthesizing sequence, 1.3-second TTFT,
58.8-second total latency, and no truncation or reasoning leak. The Research
Euclid case passed on route Research with all five banners, 6.6-second TTFT,
231.8-second total latency, eight quality-checked sources, and no truncation or
reasoning leak. Its completion log recorded `outcome=ok`. Wave 1B is complete.

Gate:

- supported multi-turn tool calling works in streaming and non-streaming modes;
- each stream has one response identity, role transition, and terminal outcome;
- fallback, thinking, health, and metrics policies are consistent where their
  contracts overlap;
- progress-banner ordering remains unchanged;
- typed run, stage, text, tool, usage, cancellation, and terminal events have
  one owner and adapter contract;
- Open WebUI and agent-client deployment smokes pass.

## Wave 1C — identity, storage accounting, and durable lifecycle

Introduce a provider-neutral authenticated principal with a stable Audrey user
id and storage namespace. Then centralize storage reservations and make
indexing, retention, migration, archive capture, and deletion retryable. Build
user-facing memory and chat inspection, correction, export, and deletion only
on top of those durable primitives.

1C.4 laptop status (2026-08-28): legacy memory migration retains its source
until every deterministic upsert succeeds. File deletion now writes a durable
tombstone first, hides pending records from private text/image search, retries
Qdrant and disk cleanup across restart, and prevents startup reconciliation
from resurrecting completed deletions. All 2,500 hermetic tests pass; changed-
file lint, compilation, YAML, and whitespace checks are clean.

On 2026-08-29 the normal delete recorded requested=1, completed=1, and
pending=0. During the failure smoke, the tombstone survived an Audrey restart
at pending=1 while Qdrant was unavailable, retried at the configured backoff,
then completed and returned pending to 0 after Qdrant recovered. 1C.4 is
Unraid-verified.

1C.5a status (2026-08-29): Audrey exposes authenticated, paginated
memory inventory and chat-source export under `/v1/me`. The browser cannot
select a user; Audrey injects the verified identity into service-token-gated
sidecar routes that are hidden from model discovery. Chat export omits pending
deletion tombstones. All 2,510 hermetic tests pass. The Unraid smoke returned
valid memory and chat pages with opaque continuation cursors, hid internal
memory scope tags, and rejected an unauthenticated request with HTTP 401.
1C.5a is Unraid-verified.

1C.5b status (2026-08-30): complete and Unraid-verified. Authenticated
current-user memory correction and deletion wait for Qdrant acknowledgement,
preserve ownership, and remove caller-supplied scope tags. Per-conversation
chat deletion hides the conversation immediately, durably retries
SQLite/Qdrant cleanup, and retains a cutoff tombstone so a late archive
delivery cannot resurrect deleted history while a new post-delete turn can
still be stored. Colliding client conversation ids remain user-scoped. All
2,521 hermetic tests pass; scoped ruff and compilation are clean. On Unraid,
memory correction removed an injected foreign scope tag and memory deletion
acknowledged the exact key. With Qdrant stopped, one conversation deletion
stayed durably pending after a failed repair attempt; after recovery it removed
two messages, one SQLite chunk, and one Qdrant point, returned both pending
counts to zero, and retained one completed conversation tombstone. OWUI did
not forward its displayed chat id on this request, so the archive used its
stable `derived-...` fallback; the native UI contract therefore requires one
Audrey-issued conversation id to be preserved end to end.

1C.5c status (2026-08-31): complete and Unraid-verified. Authenticated
`GET /v1/me/repair-status` composes exact-current-user counts for durable file
deletion, Audrey-to-sidecar archive delivery, chat indexing, chat deletion, and
conversation tombstones. It reports `ready`, `repairing`,
`attention_required`, or `degraded` without returning payloads, user
identifiers, raw backend errors, hostnames, or filesystem paths. The sidecar
status route is service-token protected and hidden from model discovery. Each
store filters the authenticated user before returning counts; global totals
are never reused as an owner view. The focused lifecycle/API matrix passes all
37 tests and the full hermetic suite passes all 2,534; scoped ruff,
compilation, and whitespace checks are clean, and the lesson scan has zero
broken links. On Unraid, the endpoint reported `ready` with zero pending or
exhausted work, changed to `degraded` only for the three sidecar-owned
components while custom-tools was stopped, and returned to `ready` after
restart. Local file-deletion and archive-delivery status remained available
throughout, and the completed conversation tombstone remained visible.
Admin-wide repair controls are implemented below; their Unraid smoke remains.

1C.5d status (2026-08-31): complete and Unraid-verified. Audrey exposes
an authenticated `POST /v1/me/data-purge` with an exact confirmation phrase,
optional scoped idempotency key, and an exact-owner progress receipt. One
durable cutoff coordinates upload records and disk artifacts, private
text/image points, local archive delivery, sidecar chat source/index rows,
and durable memory. Pre-cutoff records become logically unavailable before
physical cleanup, while post-cutoff activity is preserved. The coordinator
retries across dependency outages and process restarts; a durable
acknowledgement gate prevents memory or chat reads during sidecar handoff.
Maintenance-only archive storage remains purgeable even when ordinary
archive delivery is disabled. Cross-user, cutoff, idempotency, migration,
outage, and restart tests pass; the full hermetic suite passes all 2,549
tests. Scoped ruff and compilation are clean, and the lesson scan has zero
broken links. On Unraid, a normal purge deleted two pre-cutoff uploads and
converged with empty memory and chat owner views. A second purge, seeded with
a file, durable memory, and archived conversation, was requested while
Qdrant and custom-tools were unavailable. The receipt survived an Audrey
restart, and reads returned the purge-in-progress gate before sidecar
acknowledgement. Restoring dependencies completed the queued file deletion
and sidecar cleanup without manual replay. Final owner inventories were
empty, and repair status returned `ready` with two completed account purges
and no pending, errored, or exhausted work.

Admin-wide repair status and bounded retry (2026-08-31): implementation and
laptop gates are complete. Admin-only `GET /v1/admin/repair-status` returns
aggregate queue health, and `POST /v1/admin/repair` wakes each local durable
owner plus one bounded service-authenticated sidecar maintenance pass. Neither
route accepts a user selector or returns identities, content, paths, hostnames,
or raw errors. The sidecar controls require the service token and remain hidden
from model discovery; a sidecar outage produces a partial/degraded result while
local repair owners still run. The focused matrix passes 63 tests and the full
hermetic suite passes all 2,559; scoped ruff and compilation are clean, and the
lesson scan has zero broken links. On Unraid, aggregate status reported `ready`
and the trigger `accepted`; with custom-tools stopped they became `degraded`
and `partial` while every local repair owner stayed available. Restarting the
sidecar restored `ready` and `accepted`. Wave 1C is complete and
Unraid-verified.


Gate: **Passed on Unraid 2026-08-31.**

- concurrent mixed uploads honor configured storage limits;
- changing an email or authentication provider does not silently move or merge
  a user's private-data namespace;
- browser, personal-token, transitional OWUI, and internal-service identities
  cannot be confused or downgraded into one another;
- interrupted work and cleanup converge after restart;
- archive and deletion repair paths are idempotent;
- response delivery is not blocked on archive indexing;
- conversation identity remains stable across a growing thread;
- disposable-user export and deletion are verified across stores on Unraid.

## Wave 1D — capability policy and readiness

**1D.1–1D.2 status (2026-09-01): complete and Unraid-verified.**
One catalogue declares all ten model tools, including visibility, identity
binding, dependencies, purge gating, availability, and dispatch metadata. The
registry exposes only available model-visible records. Discovery requires an
explicit `tools` tag and known declaration, validates user-bearing schemas, and
isolates HTTP, JSON, schema, or unexpected coroutine failures per server.
Dispatch consumes the same identity and purge policy instead of duplicate name
sets. The discovery/dispatch matrix passes 44 tests and all 2,565 hermetic tests
pass; scoped ruff and compilation are clean, lesson conventions pass, the link
scan has zero broken cites, and the diff check is clean. On Unraid, the live
registry exposed exactly the ten declared tools with the expected visibility,
identity binding, dependencies, and availability. Rediscovery while
custom-tools was stopped returned an empty registry with HTTP 200 while Audrey
`/health` remained HTTP 200; restarting the sidecar and rediscovering restored
all ten tools without restarting Audrey.

**1D.3 status (2026-09-01): complete and Unraid-verified.** Custom-tools now
starts and remains live with Qdrant unavailable. The live degraded registry
contained exactly the four independent tools and reported 4 of 10 declarations;
a direct stateful call returned structured HTTP 503 with `Retry-After: 5` and
the unavailable `memory`/`qdrant` components, while direct web fetch and an
ordinary OWUI chat succeeded. After Qdrant returned, asynchronous supervision
and Audrey rediscovery converged from the initial 4 of 10 snapshot to all 10 of
10 without restarting Audrey. The affected 366-test matrix and all 2,575
hermetic tests pass.

**1D.4 laptop status (2026-09-01): implemented; Unraid smoke pending.**
Admin-only `GET /v1/admin/readiness` reports one sanitized snapshot spanning
required/optional component probes, tool policy/discovery/availability and
capability failures, archive delivery/index/delete backlogs, media-processing
and fetch queue age, watcher/reconciler activity, and aggregate GPU/in-flight
pressure. `/health` remains shallow process liveness. The same cached snapshot
publishes bounded-cardinality Prometheus gauges; only configured required
components produce `unready` and HTTP 503, while optional failures report
`degraded`. The default required component is Ollama and the policy is validated
at startup. All 2,586 hermetic tests pass; scoped ruff, compilation, YAML parsing,
and diff checks are clean. The lesson link scan has zero broken cites; existing
drift remains deferred.

Represent model-visible capabilities and their dependencies declaratively.
Make discovery resilient per source, allow independent capabilities to degrade
independently, and expose component-level readiness and repair status.

Gate:

- one unavailable optional dependency does not block unrelated chat/tools;
- capability recovery and rediscovery work without a full stack restart;
- only explicitly model-visible endpoints enter the tool registry;
- readiness, admin status, and metrics agree under injected component failures;
- degraded-start and recovery behavior is verified on Unraid.

## Wave 1E — reproducible and responsive runtime

Build Python services from the committed lockfile, finish non-root container
operation, remove blocking hot-path file work, reuse long-lived clients, and
bring configuration/documentation back in line with runtime behavior.

Gate:

- repeated builds from one lockfile resolve the same environment;
- Audrey and custom-tools operate non-root with deliberate bind-mount ownership;
- slow storage operations do not stall unrelated requests;
- environment/YAML precedence is tested and documented;
- generated capability/model inventories match their sources;
- container and Unraid smokes pass.

## Wave 1F — bounded simplification

After the behavior above is pinned, extract only the lifecycle and stream
mechanics whose duplication the earlier waves exposed. Keep request policy,
research policy, and storage ownership explicit rather than hiding them behind
an overly generic framework.

This wave is complete when the affected behavior tests pass unchanged and the
result reduces duplicate state transitions or policy declarations.

## Sequence

```text
1A data boundaries/request ownership
  -> 1B protocol/run-event consistency
  -> 1C identity/durable storage lifecycle
  -> 1D capability policy/readiness
  -> 1E runtime reproducibility/responsiveness
  -> 1F justified simplification
  -> Campaign 3 Phase 2 Audrey application and web UI
  -> Campaign 3 Phase 3 reusable skills
```

Storage lifecycle and capability readiness may be developed independently
after Wave 1A, but they remain separate deploy units. The native application
phase begins only after Phase 1's platform gates are closed or explicitly
accepted with a documented disposition in the private engineering record.

## Phase 1 completion gate

- All planned waves have laptop and user-confirmed deployment evidence.
- Full hermetic tests and changed-file lint pass.
- Lesson-link checks report no required drift fixes.
- Restart, cancellation, and failure-injection paths converge cleanly.
- Runtime status, configuration, and public documentation agree.
- No unresolved platform-boundary item is being deferred implicitly into the
  native application implementation.

## Deferred product work

OCR, broader audio/media support, ordinary-answer provenance, artifact
download, and `/v1/responses` remain candidates for later Campaign 3 phases.
They are not interleaved with the hardening or initial skills rollout.
