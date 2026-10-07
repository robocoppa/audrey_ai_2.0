# Phase 17 — operations and API improvements

Potential Hermes improvements are recorded here for later. The user confirmed
Hermes is working and deferred further integration work October 7. Revisit
parked bot/API work only on an explicit future request with a concrete need.
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

**17A.2: Complete; user accepted October 7.**
The provisioned **Audrey — Models** view provides model filtering, call counters
and rate, terminal latency, error/cancellation behavior, reported token usage,
and observation coverage. One chat turn can make several provider calls.
Use normal traffic; no new telemetry database or chat UI. Maintenance instructions
live in [monitoring](../../monitoring/README.md).

A bot-readable JSON view is parked until a user request identifies a real consumer. Before adding it,
define authorized model scope, time window, freshness, and monitoring-source
failure behavior. Do not expose global model traffic to arbitrary bot tokens.

## 17B — validated model reasoning controls

**Built; live API acceptance unconfirmed. Further Hermes work deferred.**
The benefit to Hermes's existing adapter has not been established. Completed/streamed Chat accepts
`reasoning_effort`; Responses accepts `reasoning.effort`, including client-tool
generation. Controls apply to permitted `audrey_passthrough/<model>` requests.

Ollama `/api/show` supplies a typed thinking descriptor (`values` and `default`).
Named efforts forward exactly only when advertised; `none` maps to native
`false` only when that off control is advertised. A capability flag alone does
not establish graded effort support. Successful metadata is cached for the
backend process; restart after replacing a model to refresh its contract.

| Installed Ollama 0.35.1 metadata, user confirmed October 7 | Accepted efforts | Omitted provider default |
|---|---|---|
| Kimi K3 | `none`, `low`, `high`, `max` | `max` |
| GLM 5.3 | `low`, `high`, `max` | `max` |

No `medium` mapping is invented. Unsupported effort or explicit pipeline-model
effort returns JSON 400 before file hydration or SSE. Failed/malformed metadata
returns 503 and is not cached. Supplying both Chat `think` and effort returns
400. Omitted/null effort preserves existing defaults and the legacy boolean
`think` behavior, including its best-effort capability gate.

Responses echoes supplied reasoning configuration; saved continuation does not
inherit it. Provider-wire forwarding is verified independently of model quality,
reasoning-token counts, or billing effect. Native chat UI and Hermes's working
Chat connection retain their current configuration. Existing regression and
smoke code remain available; bot-specific testing is not a scheduled task.

## 17C — failure and retry diagnostics

**Conditional, inactive:** investigate an actual user-reported failure before
proposing a change. Working Hermes behavior does not need a new test sweep.

Stream failures already expose `response.failed.response.id`; clients can log
that identity. Change a missing path only with an actual failure example.

Identify the source of a reported quota/session-limit 429 before adding
headers. Audrey's concurrency limiter waits; it does not own that quoted quota.
Preserve valid upstream retry timing when available, and emit remaining/reset
data only if the responsible service supplies it. Invent neither quotas nor
reset times. Close this item without code if current behavior suffices.

## 17D — deferred tool contracts and catalog admission

**Deferred by the user October 7.** Hermes is working; no protocol migration,
tool-selection change, catalog expansion, or bot testing is queued. Resume only
on a future user request identifying a useful Responses workflow.

Notes for that future decision: the reported Hermes catalog has 20 tools /
35.4 KB, within 64 KiB but above the current 16-definition limit.

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
