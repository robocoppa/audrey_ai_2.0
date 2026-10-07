# Campaign 3 remaining-work review — October 6, 2026

**Decision:** Finish the small set of outstanding product checks, diagnose the
Tailscale route independently, and keep research experiments outside the
campaign completion gate. Useful Hermes API improvements go at the end of the
workflow. Existing Chat Completions connections remain supported. Conditional
Hermes extensions and research experiments do not become prerequisites for
closing the accepted Audrey product work.

## Scope correction

**User requests are English-only for this campaign.** Do not add non-English
prompt cases, spend live model calls evaluating them, or tune routing/selection
for their results. Supporting another input language requires an explicit
future scope change. This is a development and acceptance boundary, not a new
language-rejection feature in Audrey. Historical artifacts and existing
hermetic regressions remain evidence; they do not create new work.

The Japanese Slice 3D.5 was a scope error. Its reported Tower check at
`2026-10-07T03:09:11.145355+00:00` failed: 3/4 matches, two guarded
abstentions, two model calls, and one rejected ineligible document choice for a
video request. The [exact terminal receipt](../../evals/results/2026-10-06-skill-selection-v4-japanese-smoke-summary.json)
is retained. Close this branch as **out of scope**, without marking it passed
or rerunning it. No runtime selector was enabled. Automatic selection remains
disabled, and its further studies are parked until a concrete English workflow
justifies them. Mixed-language study aggregates are not English acceptance
scores; do not discard samples or relabel the old study to make it pass.

## Execution order

| Order | Work | Smallest useful finish |
|---|---|---|
| 1 | Deploy 11A.1; then monitoring view | Updated October 7: telemetry foundation built, 60 focused and 4,091 full tests pass. Phase 12/14 accepted; protocol/network evidence remains tracked separately. |
| 2 | Close the 13A inline-image API boundary | One targeted inline-image case on the laptop via `192.168.1.11:8000`; retain settled 13B–13F results. |
| 3 | Diagnose the Tailscale backend route | Identify the failing hop before a configuration change. LAN/WARP already supports the other checks. |
| 4 | Resolve narrow operational unknowns | Read existing evidence for Kimi K3 panel dispatch and DeepSeek capabilities; request only a missing observation needed for a decision. |
| 5 | Implement useful Hermes API additions | The bounded slices below, after the existing work; no forced adapter migration. |

**October 7 acceptance update:** Phase 12 and 14 are complete. The user
confirmed all checks passed, including the same-token `/api/models`
200 → 401 revocation proof. The temporary checklist was deleted. The next
real code slice is 11A.1 model telemetry; remaining 13A and network evidence
does not require repeating accepted UI/account checks. Phase 11/15 Files,
uploads, composer, Projects, grounding, and layouts stay accepted.

13A's inline data-URL admission remains distinct from 13C's uploaded-file and
13D's remote-URL proofs. It needs its own small protocol check. No dependency
requires Phase 14 to wait for further Responses features.

Tailscale diagnosis should inspect laptop reachability, Tower identity/routes,
firewall, and the published port separately. Do not change Audrey routing to
compensate for an unlocated VPN/network failure. Keep the working LAN address
in smoke handoffs until Tailscale is verified.

## Experiments that no longer block completion

- Automatic skill selection: explicit document/video skills are already
  accepted. Further selection rules, router candidates, prospective studies,
  and final-answer comparisons are parked until an English product need and
  decision criteria exist. `skills.auto_select: false` remains set.
- Compression and tool-result-size A-B-A studies: keep their unmeasured status,
  but run only to answer a named quality/cost question using a stable baseline.
  Do not bundle them into a broad sweep or vary both settings together.
- Media summaries: accepted PDF/audio/video upload and Summary/Transcript
  workflows stay accepted. Any unobserved no-thinking claim is a specific
  instrumentation/cost observation, not another ingest acceptance requirement.
- Fact-check fallback, ledger drops, escalation cost, and draft size: analyze
  accumulated traffic when there is enough; do not create traffic solely to
  satisfy an old checklist.
- Distinct Conversations API, background generation, completion webhooks,
  broader media analysis, historical chat import, and model swaps remain
  conditional on a demonstrated caller or product need.

## Assessment of the Hermes requests

| Bot request | Actual status and disposition |
|---|---|
| 1. `store: true` and `previous_response_id` | **Already built and live-passed in 13F.** Thirty-day retention, owner-scoped retrieval/deletion, and bounded text/function continuation exist. Explain adoption and limits; do not build it again. |
| 2. Per-model `/usage` view | **Useful and feasible.** Reuse existing Prometheus model histograms and Grafana; add missing bounded usage counters and a small protected read interface if bots need machine-readable data. |
| 3. Reasoning effort on Responses | **Useful, conditional on model support.** Validate installed model controls and translate supported meanings consistently with Chat Completions. Unsupported effort must not silently become a different setting. |
| 4. `tool_choice: required` | **Feasible as a validated response contract.** Successful output must include a valid advertised call; otherwise fail explicitly. This does not authorize execution or replace Hermes approvals. |
| 5. Raise all four limits | **Evidence supports only a smaller definition-count expansion.** A reported 20-tool / 35.4-KB catalog fits 64 KiB; 16→24 definitions may address the known blocker. Do not also double bytes, call groups, and replay limits without actual failing requests. |
| 6. Completion webhook | **Defer.** The bot reports no use for it. It requires durable background jobs and delivery/retry semantics, not just a URL field. |
| 7. 429 quota/reset headers | **Diagnose the source first.** The quoted session-limit text and a corresponding session-quota gate were not found in Audrey. Its concurrency limiter waits for a slot. Add or preserve truthful upstream timing only after identifying the responsible hop. |
| 8. Response ID in stream errors | **Already present.** Responses failure events contain `response.id` in the nested failed response, including client-function and storage failures. Help the client parse it; change a missing path only with an actual wire example. |

### Stored conversation state does not establish token savings

Audrey prepends retained history to the new input and sends that expanded
history to Ollama. This reduces Hermes-to-Audrey payload size and bookkeeping,
but does not remove that context from model input. Retention TTL does not change
this behavior. OpenAI also states that previous input tokens still count when
using `previous_response_id`. Its separate prompt cache reuses computed prefix
state; conversation storage and prompt caching are different capabilities.
[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state),
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

Claudette currently uses Chat Completions. Adding these fields to that request
does not enable Responses continuation; it needs the Responses endpoint and
adapter. Its full 20-tool catalog currently exceeds that optional interface's
definition limit. Keep its current connection. If actual inference-cost savings
are the goal, measure provider cache hits or evaluate client context compaction
against preserved answer/tool quality. No token-saving claim follows from 13F.

## Appended Hermes API slices

### 11A — model telemetry using the existing monitoring stack

**11A.1 build, October 7:** Provider termination is recorded once and before
terminal delivery. Cancellation and unconfirmed EOF cannot count as success.
Three fixed usage kinds expose token totals plus field-observation counts;
missing data stays unknown and real zero stays observable. The cloud alert
excludes cancelled calls. The 60 focused and 4,091 full backend tests pass,
along with Ruff and compilation (October 7). Normal deployment observation,
monitoring views, and any bot-readable API remain later work.

1. Define and repair terminal outcome accounting before building trustworthy
   success/error/cancellation rates. Before 11A.1, Ollama instrumentation
   could mark interrupted/unconfirmed streams `ok` or miss completed-call cancellation.
   Reuse `audrey_model_seconds` and its count after this boundary is corrected.
   Distinguish model calls from chat turns: one turn can invoke several models.
2. Add bounded per-model input/output-token observations where provider terminal
   usage is available. Count one terminal observation per call, report absent
   usage as unknown, and distinguish cached-input observations from billed cost.
   Responses currently emits cached/reasoning token details as compatibility
   zeros; those are not measured observations. Capture real cache-hit counts
   only where the installed provider supplies them.
3. Reuse Grafana provisioning. If Hermes needs JSON, expose a small protected
   API backed by a configured monitoring source and explicit time window/freshness
   metadata. Do not expose raw prompts, identities, or unapproved model traffic.
   Decide bot read authority before exposing global aggregates.
4. Verify completed, streamed, failed, and cancelled accounting with hermetic
   fixtures; observe normal post-deploy traffic for runtime metrics. Reuse
   those calls for the monitoring view.

The 11A.1 metrics supply generation latency/rate and per-model token
observations. Monitoring already scrapes every 15 seconds. The new interface is for
operations/bots; it does not require an Audrey chat UI feature or a new telemetry
database.

### 11B — validated model reasoning controls

1. Inspect the installed Ollama version and Kimi/GLM `/api/show` metadata plus
   the sanitized controls Hermes actually sends.
2. Share model-aware resolution between completed and streamed requests and
   across the current Chat interface and optional Responses interface.
3. Preserve omission/default behavior and existing supported `think` callers.
   Define conflicts and precedence explicitly. Admit an effort name only when
   its meaning is supported; reject explicit unsupported values clearly.
4. Prove the outgoing control on each model separately. Sending a flag is not
   proof of a changed reasoning budget; measure behavior before claiming it.

Current Ollama documentation describes boolean and model-defined named controls
in `thinking.values`; installed support remains unverified. Audrey currently
caches only the generic thinking capability and uses boolean controls. A blanket
`medium/high/max → true` conversion would misrepresent graded effort.
[Ollama thinking controls](https://docs.ollama.com/capabilities/thinking).

### 11C — bot failure and retry diagnostics

1. Reuse `response.failed.response.id` in Hermes logs. Obtain a sanitized example
   only if some actual response lacks correlation; avoid a duplicate ID feature.
2. Identify the origin of the reported 429 and session-quota text. Preserve a
   valid upstream `Retry-After` where it reaches the compatibility seam; expose
   remaining/reset information only when the upstream or Audrey owns that data.
3. Distinguish quota exhaustion from concurrency waiting and other failures.
   Do not invent a reset time or introduce a new quota system to manufacture headers.

This slice starts with existing artifacts. If they already establish correct
behavior, close it without code changes or another generation sweep.

### 11D — optional tool contracts and measured catalog admission

Proceed only when a bot actually wants this optional Responses workflow.

- Reuse the existing argument/batch validation and buffered stream calls. For
  `required`, validate at least one schema-valid advertised call before a
  successful terminal. If the provider does not comply, return HTTP 502 or typed
  `response.failed`; do not manufacture a call. Buffer executable stream items
  until the full batch satisfies the contract. Hermes still executes and approves.
- Review the actual sanitized 20-tool schemas and expand 16→24 definitions if
  their schema subset and aggregate prompt fit. Keep the 64-KiB definition,
  8-call group, and 128-item replay limits until another measured request needs more.
- A later replay/group increase must coordinate admission, raw generated calls,
  saved-history validation, request bytes, chain depth, storage quotas, and the
  combined 16,000-token prompt budget. Increasing one constant is insufficient.
- Assess Kimi primary, GLM fallback, tool results, interruption handling, and
  preserved history with the bot's actual subset. Keep Chat Completions as the
  working path until adoption has a concrete benefit and the measured flow fits.

Ollama's documented native chat request has tools but no `tool_choice` parameter.
Output validation can enforce what Audrey returns; it cannot guarantee model
compliance. [Ollama chat API](https://docs.ollama.com/api/chat).

## Source checks and evidence boundary

Audrey source inspected: [retained replay](../../src/audrey/routes/openai/response_storage.py:104),
[retention limits](../../src/audrey/app_state/responses.py:15),
[function admission](../../src/audrey/routes/openai/client_tools.py:30),
[model metrics](../../src/audrey/metrics.py:101),
[failure stream identity](../../src/audrey/routes/openai/streaming.py:288),
[tool stream identity](../../src/audrey/routes/openai/client_tool_generation.py:155),
[concurrency waiting](../../src/audrey/routes/inflight.py:103), and
[existing monitoring](../../monitoring/README.md).

Terminal accounting inspected in [Ollama calls](../../src/audrey/models/ollama.py:389)
and [stream cleanup](../../src/audrey/models/ollama.py:448); compatibility token
details are in [Responses objects](../../src/audrey/routes/openai/responses.py:159).

This review changes documentation and records supplied evidence. It does not
implement the proposed API additions, change deployed routing, or send a message
to Hermes. The existing 4,031-test backend baseline remains the latest code gate;
no new live calls or evaluator reruns were needed for this assessment.
