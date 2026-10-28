# Phase 17 — operations and API improvements

**Status:** Closed by owner October 27, 2026. Monitoring and validated reasoning
controls are implemented. Unbuilt diagnostics and bot/API proposals are closed
out of scope. Hermes keeps its working adapter; a future change requires a new
user request with a concrete caller benefit.

## 17A — model telemetry and monitoring

Provider calls record one terminal outcome/latency before delivery, distinguishing
completion, failure and cancellation. Cloud-error alerts exclude cancellation.
`audrey_model_tokens_total` and `audrey_model_usage_observations_total` expose
provider-reported input/output/cached-input fields and their observation coverage.
Valid zero is observed; invalid/missing remains unknown. Output may include
reasoning; cached input is a subset, not additional tokens or proved savings.

The provisioned Audrey — Models Grafana view offers filters, counters/rates,
latency, failures/cancellations and token coverage. A chat can make several
provider calls. Maintenance belongs to [monitoring](../../monitoring/README.md).
A bot-readable global-traffic JSON surface was not implemented.

## 17B — validated model reasoning controls

Chat accepts `reasoning_effort`; Responses accepts `reasoning.effort` for
permitted `audrey_passthrough/<model>` calls, including client-tool generation.
Live API acceptance was not established by the recorded work; closure supplies
no model-quality, reasoning-token or billing claim.

Ollama's typed `/api/show` metadata supplies advertised values/default. Named
efforts forward only when advertised; `none` becomes native `false` only if an
off control exists. No `medium` mapping is invented. Successful metadata is
process-cached; a model replacement needs restart to refresh that contract.

Unsupported effort or explicit pipeline-model effort returns 400 before file
hydration/SSE. Invalid/unavailable metadata returns uncached 503. Chat `think`
and effort together return 400. Omitted/null preserves provider defaults and
legacy boolean behavior. Responses echoes supplied configuration; continuation
does not inherit it. Native chat and Hermes retain their existing configuration.

## 17C — failure and retry diagnostics

Closed without a new diagnostic implementation. Existing stream failures expose
`response.failed.response.id`. Any later quota/session-limit issue needs its
actual owning service identified first: Audrey's limiter waits and does not
invent upstream quotas/reset times. Existing valid retry timing is preserved.

## 17D — deferred tool contracts and catalog admission

Tool-choice expansion, larger catalogs, bot protocol migration/testing,
background generation and webhooks remain unimplemented. Hermes owns execution,
approvals, history, result matching and deduplication; model call validation does
not authorize execution. Current [Responses](phase-13-responses-multimodal-input.md)
limits and saved-continuation contracts remain authoritative. No GPT bot or
existing Hermes document-workspace change belongs to this closed campaign.
