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

## Tool availability at runtime

Search clients are optional and belong to the sidecar. A configured Brave key
enables Brave; a configured SearXNG URL enables SearXNG. Either can serve alone.
With both configured, query hashing can choose the primary and the other is tried
after an error or empty result. With neither configured, only web_search is
unavailable; the process and unrelated tools still start.

The sidecar revalidates healthy memory/history collections and their separate
pinned embedding profiles every five minutes, alongside faster recovery polling.
An outage can withdraw the affected capabilities and recovery can restore them.
Qdrant reconnect validates collection identity before reuse. These are local
health checks, not user requests or content searches.

Audrey refreshes the same tool registry every minute after its bounded startup
recovery, and refreshes skill availability with it. Manual and background
rediscovery are serialized; shutdown cancels owned work before closing clients.
Explicit unavailable components hide dependent tools on the next successful
snapshot. An empty or failed discovery retains the previous view. Changes are
eventual, not instantaneous; request-time capability checks still return a
component-specific 503 while unavailable. Model visibility continues to require
both a model-facing route and its matching security declaration.

The October 24 maintenance followup is locally verified; rebuild audrey and
custom-tools after pull. No browser build, schema migration, model pull or index
run is required for this followup.

## Constraints

Authenticated user values override model-supplied identity. Preserve routing
order and tolerant structured-output parsing. Local/remote component readiness
does not prove answer quality. Media-worker remains network-isolated from
Ollama; media-fetcher retains its necessary download egress.

No remaining phase work. Later product capabilities are tracked in their own
phase documents; accepted hardening does not require another live test sweep.
