# Campaign 3 Phase 1 — platform hardening

**Status:** In progress. Waves 1A, 1B, and 1C.1–1C.2 are Unraid-verified; Wave
1C.3 archive capture is laptop-green and awaiting deployment.

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

Gate:

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
