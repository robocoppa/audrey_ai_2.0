# Phase 17 — operations and API improvements

Useful Hermes requests are incorporated here. The working Chat Completions
connection remains in use; an optional Responses adapter must earn adoption.
Earlier backlog labels 11A–11D now correspond to 17A–17D; Phase 11 remains
the completed native file explorer.

## 17A — model telemetry and monitoring

**17A.1: Complete; user accepted October 7.** Provider calls produce exactly
one terminal outcome and latency observation, recorded before delivery.
Strict terminal confirmation distinguishes completion, failure, and
cancellation. The cloud error alert excludes cancellations.

`audrey_model_tokens_total` and `audrey_model_usage_observations_total` expose
provider-reported input, output, and cached-input fields. Valid zero is observed;
missing/invalid fields stay unknown. Output can include reasoning. Cache is a
subset observation, not extra input or a claim of billing savings.

**17A.2: Built — operator model dashboard; browser acceptance pending.**
The provisioned **Audrey — Models** view provides model filtering, call counters
and rate, terminal latency, error/cancellation behavior, reported token usage,
and observation coverage. JSON, layout, datasource, and parsed PromQL checks pass locally;
live Prometheus/Grafana execution remains the user check. One chat turn can make several
provider calls. Use normal traffic; no new telemetry database or chat UI.
Deployment/acceptance instructions live in [monitoring](../../monitoring/README.md).

A bot-readable JSON view is conditional on a real consumer. Before adding it,
define authorized model scope, time window, freshness, and monitoring-source
failure behavior. Do not expose global model traffic to arbitrary bot tokens.

## 17B — validated model reasoning controls

- Inspect installed Ollama/model thinking metadata and the actual controls
  Hermes sends, with credentials removed.
- Resolve supported controls consistently for completed/streamed Chat and
  Responses requests. Preserve omitted defaults and existing boolean `think`.
- Reject unsupported explicit effort values and conflicting settings clearly.
  Do not translate graded effort to a boolean and claim the same budget.
- Prove forwarding separately from a measured effect on provider behavior.

Current Chat passthrough supports boolean `think`, gated by model capability.
Chat currently ignores unknown `reasoning_effort`; Responses rejects unknown
reasoning fields. A rejection does not establish that a model stopped thinking.

## 17C — failure and retry diagnostics

Stream failures already expose `response.failed.response.id`; clients can log
that identity. Change a missing path only with an actual failure example.

Identify the source of a reported quota/session-limit 429 before adding
headers. Audrey's concurrency limiter waits; it does not own that quoted quota.
Preserve valid upstream retry timing when available, and emit remaining/reset
data only if the responsible service supplies it. Invent neither quotas nor
reset times. Close this item without code if current behavior suffices.

## 17D — optional tool contracts and catalog admission

Proceed for a caller wanting an actual Responses workflow. The reported Hermes
catalog has 20 tools / 35.4 KB: it fits 64 KiB but exceeds 16 definitions.

- Inspect the actual schemas before a minimal 16→24 definition expansion.
  Keep 64 KiB definitions, 8 calls/group, and 128 replay items until a measured
  request needs more. Prompt/storage/replay limits must remain coordinated.
- `tool_choice: required` can be a validated successful-output contract:
  require a schema-valid advertised call or return an explicit failure.
  Never fabricate a call or release streamed calls before the batch validates.
- Hermes retains execution, approvals, history, result matching, and retry
  deduplication. Test Kimi/GLM with the real catalog/subset only if adoption is
  useful; reported isolated calls do not prove installed-adapter recovery.

## Already delivered or deferred

Phase 13 provides owner-scoped stored responses, bounded continuation, and
failure identity. Rebuilding them or assuming token savings adds no value.
Background jobs and webhooks remain deferred until a caller needs them.
Keep the GPT bot outside this work and preserve Hermes's existing document
workspace at `cloud.builtryte.xyz`.
