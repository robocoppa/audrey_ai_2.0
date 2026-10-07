# Hermes integration with Audrey Responses function tools

**Reviewed outcome, received 2026-10-06:** Keep Claudette's existing Chat
Completions connection with Kimi K3 primary and GLM 5.3 fallback. The optional
Responses API works for the reported bounded tests, but its 16-function limit
does not fit Claudette's reported full catalog of 20 tools. No production
provider configuration changed. The GPT Sol 6.1 bot remains outside this work.

## Scope and evidence

Claudette's supplied report, `audrey-responses-assessment-telegram.md`, identifies
Hermes v0.21.0 using Audrey as its inference provider at
`http://192.168.1.11:8000/v1`, in Chat Completions mode. It reports Kimi K3 as
primary, GLM 5.3 as fallback, streaming enabled, a 150-turn ceiling, and context
compression. Its measured request catalog contains 20 tools and 35.4 KB of
schemas. The catalog fits Audrey's 64 KiB definition budget but exceeds the
Responses limit of 16 functions. These are bot-reported observations; the raw
harnesses and wire records were not included with the report.

| Surface | Current evidence |
|---|---|
| `audrey_passthrough/qwen3.8:latest` Responses tools | Completed call, streamed function arguments, and client-result replay smoke passed 2026-10-05 |
| `audrey_passthrough/kimi-k3:cloud` | Bot reports completed functions, result loops, text SSE, and harmless local file execution passed |
| `audrey_passthrough/glm-5.3:cloud` | Bot reports completed functions, result loops, text SSE, and harmless local file execution passed |
| Streamed function arguments on Kimi/GLM | Not demonstrated by the supplied report; its listed lifecycle contains text events |
| Cross-model continuation | Bot reports isolated Kimi-to-GLM and reverse-direction replay passed before a call and after a result |
| Installed Hermes fallback and retry behavior | Not established by manual model swaps after an invalid-model HTTP 403; provider outages, stream interruption, and duplicate prevention were not demonstrated |
| Installed catalog and production connection | Bot reports 20 tools / 35.4 KB; production configuration untouched |
| Saved Responses / continuation | Slice 13F restart verification passed 2026-10-06; exact saved objects survived restart; descendant deletion passed afterward |
| GPT Sol 6.1 bot | Outside this assessment |

The report supports retaining the working connection. No new cloud generations,
limit increases, adapter changes, or production outages are needed to reach that
decision. The 8-call group, 128-item replay, and combined prompt limits remain
additional compatibility constraints. A configured maximum of 150 turns is not
itself a measurement of retained replay length after compression.

## Corrections to the bot report

### Approval and tool choice

Responses supports only `tool_choice: "auto"` or `"none"`; required and named
choices are rejected with HTTP 400. Audrey's current Chat Completions request
schema has no `tool_choice` field and ignores unknown top-level fields, so it
currently drops that parameter there too.

Forcing the model to choose a tool is not authorization to execute it. Hermes's
existing executor, security scanning, and approval rules must remain the
execution authority for either protocol. The claim that Responses would weaken
an approval gate therefore needs the actual executor path and outgoing request,
not just the presence of `tool_choice` in Hermes configuration.

### Reasoning controls

Responses does not accept `reasoning_effort` or `reasoning.effort`; unknown
fields receive HTTP 422 rather than being silently stripped. Current Chat
Completions also does not translate `reasoning_effort`: it ignores that unknown
field. Audrey supports the vendor `think: true | false` override on Chat
Completions passthrough requests, subject to the model's declared thinking
capability.

Responses retains the configured passthrough thinking policy and model defaults;
it does not currently expose that per-request override. A rejected effort field
does not establish that GLM stops reasoning. Establish which control Hermes
actually sends and which Audrey forwards before claiming a deep-thinking
regression or designing a new mapping.

### Prompt-limit observation

The client-function route enforces 16,000 admitted text tokens, measured with
`cl100k_base`, including history, function arguments/results, instructions, and
effective tool definitions. It checks before file admission and again after
adaptation. Stored/chained and file-input routes also apply that prompt budget.
An ordinary unretained text request with no client-tool controls/history or
file inputs follows a different path.

The reported accepted "20K-token" probe is not proof of a bypass without its
sanitized exact payload, counting method, HTTP result, and usage. Characters,
provider-native token counts, and Audrey's admission counter are different
measurements. Keep this observation unresolved until those artifacts establish
that the bounded route received more than 16,000 admitted tokens.

### Storage and model input cost

Slice 13F is now live-passed. Clients may opt in with `store: true`, retrieve a
saved response, and continue using `previous_response_id`. The assessment's
stateless tests did not exercise this capability because the original handoff
asked them to omit storage fields.

Storage reduces the request payload Hermes sends to Audrey and the history
bookkeeping Hermes must perform. Audrey reconstructs the complete retained
history and sends that context to the model on every continuation. It does not
reduce model input tokens or establish cheaper inference. Any provider prompt
cache benefit would need separate evidence. OpenAI also explains that chaining
with `previous_response_id` still accounts for prior input tokens in its
[conversation-state documentation](https://developers.openai.com/api/docs/guides/conversation-state).

## October 6 roadmap assessment

The bot requested conversation caching, model usage/latency visibility,
reasoning effort, required tools, larger limits, webhooks, quota headers,
and error correlation. The [review and appended API slices](../campaign-3/2026-10-06-remaining-work-review.md#assessment-of-the-hermes-requests)
distinguish already implemented features from useful future work.
13F already supplies storage/continuation without reducing model input
tokens. Streamed failures already include `response.id`. Keep the current
Chat connection; any optional adoption must fit the actual workload.

## Follow-up message to forward

> Keep your current Chat Completions connection, with Kimi K3 primary and GLM
> 5.3 fallback. The 20-tool catalog is enough to establish that Audrey's current
> bounded Responses interface is not a drop-in fit. Leave the GPT bot and all
> production provider settings unchanged.
>
> Please correct three points in the assessment using the existing artifacts
> and installed code, without running another model sweep:
>
> 1. Trace the actual outgoing reasoning control and approval execution path.
>    Audrey currently ignores `reasoning_effort` and `tool_choice` on Chat
>    Completions. Its supported reasoning override is `think: true | false`.
>    Responses rejects unknown reasoning fields with 422, supports auto/none
>    tool choice, and retains configured/model-default thinking. Show the
>    relevant request fields and executor checks, with credentials removed.
> 2. Share the sanitized exact 20K probe payload, token-count method, HTTP status,
>    response body, and usage already captured. We need to establish which
>    route and counter it exercised before calling the limit advisory.
> 3. Update storage status: Audrey's saved Responses and restart proof now pass.
>    Chaining reduces client request size but Audrey still supplies full model
>    history; it does not cut model input token cost.
>
> Keep the reported Kimi/GLM completed calls, result loops, text SSE, and isolated
> cross-model continuation results. Label streamed function-argument handling
> and installed Hermes interruption/retry deduplication as unproven by the
> supplied report. Recommend no adapter or limit changes unless a concrete
> workflow benefit appears. Do not send credentials in the follow-up.

## Historical assessment request

The message below was sent before Claudette's report and the 13F live pass.
Its pending/Qwen-only and storage-omission wording records that earlier scope;
it is not the current status or a request to repeat the completed assessment.

> Please assess Audrey's optional Responses function-tool interface for this
> Hermes installation. Keep Kimi K3 as primary, GLM 5.3 as fallback, and the
> working production connection unchanged during the assessment.
>
> First inspect your installed Hermes version, active adapter, actual Audrey
> endpoint, primary/fallback model ids, enabled tool catalog, context/compression,
> streaming mode, and tool execution/approval behavior. Establish whether Audrey
> is your inference provider or a separate tool you call. Do not assume a new
> endpoint requires changing your provider.
>
> Test `http://192.168.1.11:8000/v1/responses` separately with your existing
> Audrey PAT (`compat:full`) and both models independently:
> `audrey_passthrough/kimi-k3:cloud` and
> `audrey_passthrough/glm-5.3:cloud`. Both are already configured candidates.
> Only Qwen has passed this new protocol so far; neither cloud model is proven.
>
> For each model, verify a completed function call, a streamed function call,
> execution of a harmless local test function, and a final answer using its
> result. Then check representative existing Hermes tools and a multi-step turn
> against your current connection. Report correctness, argument/result handling,
> latency, token usage, and any errors.
>
> Check adapter compatibility before implementing changes. Responses uses flat
> function definitions, `function_call` items in `response.output`, typed SSE
> events, and matching `function_call_output` results. Changing the URL alone
> is insufficient. During this assessment omit storage/chaining fields and replay
> original input, every output item, and matching results. Execute streamed calls only
> after `response.completed`; failed/incomplete responses must not execute tools.
>
> Compare your real tools and context with Audrey's current limits: 16 function
> definitions / 64 KiB total, 1,024-character descriptions, the supported schema
> subset, 8 calls per group, 128 replay items, and 16,000 admitted prompt text
> tokens including definitions and results. Choices are `auto` or `none` only;
> `parallel_tool_calls:false` is supported. Missing `strict` means false; strict
> schemas require all properties and `additionalProperties:false`. Flag any
> incompatibility rather than silently removing tools, shortening context,
> rewriting schemas, or weakening approvals.
>
> In an isolated test, exercise Kimi-to-GLM fallback before a tool call and after
> a tool result. Preserve history, pending call ids, completed results, approvals,
> and execution records. Interruptions and retries must not duplicate execution;
> do not create an outage in the production connection to trigger fallback.
>
> Report the current setup, results for each model and fallback, compatibility
> gaps, and whether Responses offers a useful benefit over the existing path.
> Recommend keeping the current connection if the new path adds restrictions or
> no useful benefit. Complete the assessment before recommending any production
> adapter change. Keep credentials out of the report.

## Collecting additional protocol evidence if adoption is revisited

The existing `tests/smoke/smoke_responses_client_tools.py` accepts
`AUDREY_RESPONSES_TOOL_MODEL`; both cloud tags are already permitted candidates.
Its full case makes three generations and checks completed calls, streamed
argument identity, and stateless result replay. The targeted laptop commands
are in [the smoke runbook](../reference/live-smoke-testing.md#kimi-k3-and-glm-53-hermes-assessment).
Run a new cloud case only to answer a specific unresolved requirement for an
actual integration. The decision to retain Chat Completions needs no such run.
Do not repeat Qwen's passed proof or the accepted storage/restart gate.

The laptop harness uses the operator's Cloudflare Access assertion. A bot-side
reproduction must use its existing Bearer PAT with `compat:full`; do not put a
PAT in `AUDREY_USER_JWT`, which the harness sends as a Cloudflare assertion.
A protocol smoke does not establish the installed Hermes adapter, complete
catalog, compression, approval, or retry behavior.

## Contract details

- Functions execute in Hermes. Audrey requests calls and validates arguments.
- A call item has an `fc_` item id, a distinct `call_id`, the declared function
  `name`, and JSON-string `arguments`. Function-only output has no text message;
  empty `output_text` is valid when `response.output` contains calls.
- Streaming uses `response.output_item.added`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and `response.output_item.done`.
  Audrey currently emits one validated complete argument delta per call.
  Execute calls only after `response.completed`; failed or incomplete responses
  must not trigger execution.
- Submit every pending result before another message or generation. Parallel
  calls require distinct matching results. Never infer a result from a tool name.
- Limits: 16 definitions / 64 KiB total; 8 calls per group; 64 KiB arguments per
  call; 100,000 characters per result; 128 replay items; 16,000 admitted text
  tokens for the full prompt, including instructions and effective definitions.
- `strict: true` requires all object properties to be required and
  `additionalProperties: false`. Missing strict means false. Local validation
  does not guarantee the model chooses a function on every draw.
- Existing passthrough role, model allow-list, actual Ollama tools capability,
  concurrency, GPU fairness, and thinking policy apply.
- Built-in tools, required/forced choice, virtual models with client tools,
  combined skills/schema output, and background execution remain unsupported.

Both stateless replay and opt-in saved text/function Responses are live-passed.
`store: true` retains a successful response; a later request can send its
`previous_response_id` and only new inputs or matching call results. Omission,
null, or false leaves the new response unretained. Supply current instructions,
definitions, and desired generation settings on each request; these settings
are not inherited.

Retention lasts 30 days, with a 32-response chain limit, 128 combined replay
items, and the existing aggregate prompt budget. Deleting or expiring a parent
removes its descendants. Saved/chained requests support text and function
items, not files or images. The smaller client payload does not change the
full-context model input described above.

See the [Audrey implementation contract](../campaign-3/phase-13-responses-multimodal-input.md#slice-13e---client-executed-function-tools),
[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling),
and [Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events).
