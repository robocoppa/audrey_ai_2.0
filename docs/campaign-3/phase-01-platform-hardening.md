# Phase 01 — platform hardening

**Status:** Complete and deployed.

## Delivered

- Authenticated identity and private search scope are enforced below model
  instructions; collection naming does not substitute for user isolation.
- Request tasks and GPU queue grants have explicit owners. Cancellation drains
  owned work and releases slots; tool side effects are not blindly retried.
- Chat Completions validates supported message roles and translates client
  tool calls/results. Shared stream mechanics own identity and terminal state;
  Fast, Deep, and Research retain their separate orchestration policies.
- Storage reservations are atomic. Durable indexing/deletion outboxes retry
  across restarts; tombstones hide data immediately and prevent resurrection.
- User memory/chat inspection, correction, export, and deletion use those
  durable primitives. Repair status describes pending work without exposing
  another user's data.
- Declarative tool policy, bounded discovery/rediscovery, component readiness,
  and degraded startup keep optional outages from disabling ordinary chat.
- Locked dependencies, non-root services, bounded threaded disk/network work,
  and truthful deployment configuration complete the runtime foundation.

## Constraints

Authenticated user values override model-supplied identity. Preserve routing
order and tolerant structured-output parsing. Local/remote component readiness
does not prove answer quality. Media-worker remains network-isolated from
Ollama; media-fetcher retains its necessary download egress.

No remaining phase work. Later product capabilities are tracked in their own
phase documents; accepted hardening does not require another live test sweep.
