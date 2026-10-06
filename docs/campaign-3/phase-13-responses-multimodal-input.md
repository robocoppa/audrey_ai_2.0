# Campaign 3 Phase 13 - Responses multimodal input

**Status:** Slices 13B, 13C, and 13D are live-settled. Slice 13A remains
laptop-complete with its targeted live gate pending. Slice 13D's streamed retry
passed on 2026-10-05 after the token-limit and progress-rendering correction.
Slice 13E client-executed function tools is laptop-complete; live proof is pending.

## Goal

Expand `POST /v1/responses` through one bounded protocol adapter while keeping
Audrey's existing generation, vision routing, authentication, policy,
streaming, usage, and archive paths authoritative.

## Slice 13A - typed text and inline image parts

A Responses easy-input message may keep its original string content or provide
a non-empty typed content array:

| Part | Accepted contract | Audrey adaptation |
|---|---|---|
| `input_text` | Non-blank `text`; accepted on system, developer, user, and assistant easy-input messages | Chat content part `type: text` |
| `input_image` | User messages only; inline base64 JPEG, PNG, or WEBP data URL; `detail` is `auto`, `low`, `high`, or `original` | Existing Chat `image_url` content part and vision pipeline |

Both completed and streaming Responses requests use this adapter before the
shared Chat Completions generation boundary. Response ids, typed SSE events,
output objects, usage accounting, explicit skills, model policy, and
authenticated identity behavior are unchanged.

Contract references:

- https://developers.openai.com/api/reference/resources/responses/methods/create
- https://developers.openai.com/api/docs/guides/images-vision

## Slice 13A initial boundary

At its initial release, Slice 13A rejected these before generation:

- HTTP or HTTPS image URLs;
- OpenAI `file_id` image references;
- GIF and unrecognized image MIME types;
- malformed or empty base64 data;
- image parts on non-user roles;
- unknown content-part types such as output-only `output_text`.

Slices 13C and 13D below subsequently add owned file ids and public URLs.
URLs use a separate bounded fetch contract that prevents server-side request
forgery (SSRF). The existing HTTP 400
`responses_feature_unsupported` boundary remains for stored/chained responses
and background execution. Slice 13E adds client function tools on permitted
passthrough models only.

## Automated contracts

- Completed input adapts typed text plus inline image parts to the established
  Chat multimodal shape without starting a second vision pipeline.
- Streaming input uses the same adapter and still creates a
  `ResponsesStreamSession`.
- Message roles and ordering survive typed text adaptation.
- Unsupported URL schemes, unsupported MIME types, malformed base64,
  wrong-role images, and unknown part types fail validation. Later slices
  cover owned file ids and bounded public URLs.
- The targeted smoke generates a valid 32 by 32 red PNG in memory, submits it
  through Responses, requires a completed typed response and sentinel, then
  proves a `file://` URL returns HTTP 422 without generation. Its output
  ceiling is now 4,096 tokens, matching the measured vision budget.

## Laptop result

**Passed, 2026-10-01.** The focused Responses module passes 25 tests.
Changed-file Ruff and Python compilation pass. The full hermetic suite passes
3,106 tests with one existing FastAPI deprecation warning, and the diff check
is clean.

## Targeted live gate

Run `tests/smoke/smoke_responses_multimodal.py` from the laptop against the
working LAN/WARP backend route after rebuilding Audrey. A pass reports HTTP
200, a `resp_` id, `output_text`, the `RESPONSES_IMAGE_OK` sentinel,
integer usage, and HTTP 422 for an unsupported `file://` image URL.

## Slice 13B - structured text output

Responses requests may now set text.format.type to json_schema with a name,
optional description and strict flag, and a bounded object-root JSON Schema.
Audrey sends the admitted schema to Ollama only for the final answer call and
validates the returned JSON again before reporting success.

Fast structured requests skip the ReAct tool loop and prose length escalation.
Deep workers and research gathering remain unconstrained; the schema applies to
the final synthesizer or writer. Research Sources appendices and all streamed
progress banners are suppressed so they cannot contaminate the JSON document.
Completed schema violations return HTTP 502. A streamed violation ends with
response.failed and error code structured_output_invalid.

Audrey accepts a bounded subset covering objects, arrays, scalar types,
properties, required fields, additionalProperties booleans, enums, constants,
local references, definitions, anyOf, and basic length and numeric bounds.
Schemas are capped by serialized size, depth, node count, property count, and
anyOf width. Unsupported keywords and non-object roots return HTTP 400 before
generation. Strict object schemas must require every declared property and set
additionalProperties to false.

The legacy json_object mode remains an explicit HTTP 400
responses_feature_unsupported response. Slice 13E below adds client functions
on permitted passthrough models. Stored responses, chaining, and background
work remain separate future slices.

Contract references:

- https://developers.openai.com/api/reference/resources/responses/methods/create
- https://developers.openai.com/api/docs/guides/structured-outputs

### Slice 13B automated contracts

- Typed text-format parsing preserves the schema key alias and rejects unknown
  format fields.
- Schema admission and local result validation cover nested objects, local
  references, anyOf, strictness, unsupported keywords, and result mismatches.
- Completed responses forward the schema, validate output, and echo the
  requested text format.
- Structured streams contain model JSON deltas without Audrey progress text,
  validate at the terminal boundary, and fail explicitly when invalid.
- Ollama completed and streaming calls have matching format support.
- Existing plain Responses, Chat Completions, fast, deep, research, and
  passthrough behavior remains covered.

### Slice 13B laptop result

**Passed, 2026-10-05.** The focused structured and regression suite passes 298
tests. The full hermetic backend suite passes 3,150 tests with the existing
FastAPI deprecation warning. Scoped Ruff and Python compilation pass.

### Slice 13B targeted live gate

Run tests/smoke/smoke_responses_structured.py from the laptop after rebuilding
Audrey. It makes one completed and one streamed Fast model call, requires the
same fixed JSON object from both, checks schema echo and token usage, confirms
progress text is absent from streamed JSON, and proves legacy json_object mode
still returns HTTP 400 before generation.

### Slice 13B live result

**Passed, 2026-10-05.** The completed request returned HTTP 200 with a resp_
identifier, output_text, the json_schema format, the sentinel object, and
329 input plus 15 output tokens. The streamed request returned HTTP 200 with
16 deltas across 24 typed events, matching schema-constrained output, hidden
progress text, and response.completed; usage reported 133 input plus 18 output
tokens. Legacy json_object mode returned HTTP 400 with
responses_feature_unsupported.

## Slice 13C - owner-scoped file references

Completed and streamed `POST /v1/responses` requests now accept Audrey upload
ids from the authenticated caller's library:

| Part | Accepted contract | Audrey adaptation |
|---|---|---|
| `input_image` | Exactly one inline `image_url` or owned `file_id`; user messages only | Existing bounded, metadata-free JPEG preview and shared vision path |
| `input_file` | Owned `file_id`; user messages only | Extracted document text, including existing OCR text, quoted as user evidence |

Use the `id` returned by `POST /api/files`, or the `file_id` returned by the
compatibility upload route. These are Audrey ids, not ids from OpenAI's hosted
Files service. Both existing Cloudflare Access authentication and Audrey PATs
with `compat:full` bind the caller to a durable principal. A client-provided
`user` field cannot choose a storage namespace.

The shared readers resolve ids only from the principal's owned listing.
Missing, foreign, and deleted ids receive the same HTTP 404 `File not found.`
Pending, failed, or wrong-kind files return HTTP 422. Reclaimed or missing
originals return HTTP 410; unreadable content returns HTTP 409. Every check
finishes before the shared generation or streaming boundary.

The document path uses the current PDF, DOCX, HTML, Markdown, CSV, RST, and
plain-text extractors. PDFs contribute extracted text, including OCR sidecars;
PDF page images and embedded document charts are not processed by this slice.
Document contents and filenames are serialized as quoted user evidence, so
attached text cannot create a system or developer message.

### Slice 13C admission limits

Requests containing file references have these explicit limits. Repeated
references count again, and image counts include inline images and history.
Oversized inputs fail with HTTP 413 instead of being silently truncated.

| Budget | Per file | Whole request |
|---|---|---|
| File reference parts | — | 10 |
| Images | — | Current `vision.max_images_per_turn` (default 4) |
| Document original bytes | 20 MiB | 32 MiB |
| Extracted document characters | 100,000 | 200,000 |
| Extracted document tokens | 8,000 | 12,000 |
| JPEG preview plus inline image bytes | — | 6 MiB |
| Adapted text prompt, including caller instructions and history | — | 250,000 characters and 16,000 tokens |

Token admission uses `cl100k_base`; provider usage still comes from the model.
These are protocol limits, not a claim about every provider's context window.
The native document viewer reuses the shared reader without these Responses
limits, preserving its existing paged reading flow.

### Slice 13C automated contracts

- Completed and streamed calls receive the same owned image and document
  adaptation before shared generation.
- Real JPEG preview conversion resizes images and removes original metadata.
- Real DOCX extraction and existing scanned-PDF OCR text reach the adapter.
- Missing, foreign, deleted, pending, failed, reclaimed, wrong-kind, missing
  source, and corrupt-source cases reject without a model call.
- Per-file, aggregate, repeated-reference, mixed-image, instruction, and final
  prompt budgets are enforced.
- HTTP route tests exercise parsing, the owner denial, and both answer formats.
- Completed Fast token limits return incomplete status, reason, partial text,
  and usage, including reasoning-only replies. Empty normal completions return
  HTTP 502; truncated JSON is incomplete rather than a schema error.
- A successful Deep answer ignores a previous Fast attempt's token-limit stop.
- Live harness tests prove stream consistency, terminal success, color checking,
  failure diagnostics, and cleanup on model, upload, or deletion failure.

### Slice 13C laptop result

**Passed, 2026-10-05.** Focused file-reference, Responses, structured-output,
Fast-path, vision, and harness tests pass 145 cases. The full hermetic backend
suite passes 3,224 tests with the existing FastAPI deprecation warning. Scoped Ruff, Python
compilation, and the diff check pass.

### Slice 13C targeted live gate

The settled gate uses `tests/smoke/smoke_responses_file_inputs.py` from
the laptop at
`http://192.168.1.11:8000`. The exact command and credential names are in
[the live smoke runbook](../reference/live-smoke-testing.md#run-the-13c-responses-file-reference-smoke-from-the-laptop).

No browser action, video, or pre-existing library file is needed. This API
proof uploads its own tiny text document and red PNG under the smoke user,
waits for Ready, and makes two short Fast model calls: one completed and one
streamed. Both must read a randomized code from the document and identify the
image as red. The script also proves both reference types deny a second owner
with the same HTTP 404 as a missing id, rejects the wrong file kind, deletes
both temporary uploads, and proves deleted references return HTTP 404.

**Live status: Passed, 2026-10-05**, from the user's reported result over
`http://192.168.1.11:8000`, after the output-budget and status follow-up.

| Check | Observed result |
|---|---|
| Uploads | Two files reached Ready |
| Completed answer | HTTP 200; document sentinel and RED; status completed; 25 characters |
| Completed usage | 1,376 input tokens and 205 output tokens; 4,096-token ceiling |
| Streamed answer | HTTP 200; document sentinel and RED; response.completed; 15 deltas across 23 events |
| Streamed usage | 1,178 input tokens and 179 output tokens |
| Owner and existence guards | Foreign, missing, and deleted references returned HTTP 404; same not-found response |
| Kind guard | Wrong file kind returned HTTP 422 |
| Cleanup | Both temporary uploads deleted; no cleanup errors |

Earlier attempts stopped on expired Access assertions, then on an empty
answer under a 64-token ceiling. The corrected smoke uses the measured
4,096-token vision ceiling. Both successful calls generated more than 64
tokens, supporting the budget change; the earlier upstream stop cause was
not captured. Token-limit and empty-answer error semantics remain covered
by the laptop regressions. This gate is settled; repeat only if a later
change touches these contracts.

Slice 13D below adds HTTP(S) image/document URLs. Inline `file_data`, PDF
visual detail, stored response chaining, and background execution remain
separate capabilities. Slice 13E below adds client-provided functions.

Contract references: [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs)
and [Responses request schema](https://developers.openai.com/api/reference/resources/responses/methods/create).

## Slice 13D - bounded public URL inputs

Both completed and streamed Responses requests accept these additional sources:

| Part | Source | Adaptation |
|---|---|---|
| `input_image` | Public HTTP(S) `image_url`, mutually exclusive with `file_id` | Sniffed JPEG/PNG/WEBP, first frame, metadata-free JPEG preview, at most 1600 by 1600 pixels |
| `input_file` | Public HTTP(S) `file_url`, mutually exclusive with `file_id` | Extracted PDF/DOCX/HTML/plain-text/Markdown/CSV/RST text quoted as user evidence |

Parts remain user-message evidence only. Existing authentication resolves a
principal before file admission; remote input does not grant a different account,
model, or skill policy. Caller headers, cookies, and access credentials are never
sent to the source host. Downloaded inputs are temporary and do not create My Files
entries, ingestion jobs, embeddings, or project references. Ordinary compatibility
archive behavior still applies to the generated turn.

For example:

```json
{
  "model": "audrey_fast",
  "max_output_tokens": 4096,
  "input": [{
    "role": "user",
    "content": [
      {"type": "input_text", "text": "Explain this document and its image."},
      {"type": "input_file", "file_url": "https://example.org/report.pdf"},
      {"type": "input_image", "image_url": "https://example.org/chart.png"}
    ]
  }]
}
```

### URL and connection contract

- Only HTTP port 80 and HTTPS port 443 are admitted. Embedded credentials,
  unsupported schemes, malformed URLs, control characters, backslashes, scoped
  IPv6 addresses, and URLs over 4,096 characters are rejected.
- All initial destinations are resolved and vetted before any HTTP request.
  Every returned address must be globally routable. Loopback, private, link-local,
  shared/Tailscale, multicast, reserved, site-local, special protocol, and
  Teredo/6to4/NAT64 addresses are blocked.
  DNS failure fails closed; container search domains are not used.
- Each socket connects to a vetted numeric address. The original hostname remains
  in Host and TLS SNI, and HTTPS certificates are verified against that name.
  Only connection failures try another address from the same vetted list.
- Redirects are manual, capped at three, and revalidate the next URL and every DNS
  answer before connecting. HTTPS cannot redirect to HTTP. Connections are not
  reused between hops, ensuring each hostname receives its own certificate check.
- The dedicated HTTP transport ignores proxy environment variables, has no cookie
  jar, and avoids HTTPX's full-URL INFO logging. Signed query parameters are used
  for the request but stripped from quoted source evidence and archives.
- HTTP compression is disabled and non-identity Content-Encoding is rejected.
  Raw bytes are streamed under their cap even without a Content-Length. Oversized
  declared lengths are rejected without reading the body; non-200 final responses
  and empty bodies fail explicitly.

### Download and parsing limits

The Slice 13C count, image, document byte/character/token, and final prompt limits
also apply to remote inputs and mixed owned/remote requests. Repeated inputs count
again. Additional remote bounds are:

| Limit | Value |
|---|---|
| Image source | 6 MiB per remote image |
| Total remote download bodies | 32 MiB per request |
| Remote admission | 45 seconds total, including queue wait, DNS, downloads, and parsing |
| Concurrent remote admissions | 4 per backend worker |
| Each fetch | 30 seconds total; 5-second connect and 10-second read/write/pool operations |
| Parser wall time | 15 seconds per asset |
| Parser CPU time / address space | 10 seconds / 512 MiB per Linux child process |
| Image pixels before decoding | 50 million |
| DOCX archive | 1,000 entries and 32 MiB total expanded contents |

Bytes are sniffed with libmagic; headers and filename extensions cannot make
an arbitrary blob into a supported input. Explicit Content-Type must match the
sniffed kind, with normal text-type aliases and octet-stream/no-header handling.
A disposable process applies the resource limits and reuses the existing document
loaders. The parent kills and reaps it on timeout or cancellation and removes the
private temporary directory. This is resource containment, not a general OS sandbox.

PDFs contribute their text layer only. Scanned or empty PDFs must use Audrey's
existing upload/OCR flow; URL input does not start OCR or process page images.
Unsupported, corrupt, encrypted, or empty content is rejected. Excess byte,
character, token, pixel, archive, or parser resource budgets return HTTP 413;
blocked destinations and invalid content return HTTP 422. Fetch failures return
HTTP 502, as do unexpected parser exits. Timeouts return HTTP 504 and parser
startup failure HTTP 503. All admission
failures are ordinary JSON responses before a model call or SSE stream opens.

### Slice 13D verification

**Initial laptop result, 2026-10-05:** 194 focused remote/file/Responses/structured-output
and harness cases pass. The full hermetic backend suite passes 3,309 tests with
the existing FastAPI deprecation warning. Scoped Ruff, compilation, and the diff
check pass. Tests cover mixed public/private DNS answers, rebinding after vetting,
IP-pinned Host/SNI routing, connection fallback, redirect denial and cookie
isolation, raw byte and deadline bounds, real image/PDF/DOCX parsing, quoted
evidence, shared budgets, JSON denials before SSE, and child cleanup after timeout
and cancellation. Signed query parameters are absent from HTTP logs and forwarded
evidence.

**First live result, 2026-10-05:** The completed answer used both remote inputs:
"Dummy PDF file" and "Python", 21 characters, 1,375 input tokens, and 692 output
tokens. Both private image/document probes returned JSON HTTP 422 before SSE.
No uploads were created. Streaming reached all 4,096 output tokens but returned
only the 36-character Thinking banner. It incorrectly reported completed, so
that first attempt did not pass the streaming gate.

**Streaming correction:** The plain Fast owner now maps Ollama's
`done_reason: length` to `StreamOutcome.TRUNCATED`, retaining token usage and
partial answer text without retrying past the caller's ceiling. It announces
answer start only after meaningful text; empty normal completion uses the
existing bounded pre-answer fallback, failing when exhausted. Responses exposes
`response.incomplete` with `reason: max_output_tokens` for the cap and has a
second guard against blank successful output. Incomplete/failed Responses keep
`completed_at: null`. Progress remains available in internal run events but is
excluded from both plain and structured `output_text`. Chat Completions and the
native browser retain their progress rendering.

The fixed smoke accepts `--case streamed --max-output-tokens 8192`; the retry
sent only one model request, preserving the normal answer and guards as passed.
It verifies both inputs, matching deltas, `response.completed`, and
`progress_hidden: true`. The larger request ceiling allows more vision reasoning
but does not change production defaults or bypass output bounds. An incomplete
response remains a failed answer gate, with usage and reason retained.
Exact steps are in
[the live runbook](../reference/live-smoke-testing.md#run-the-13d-responses-remote-input-smoke-from-the-laptop).
Slice 13C remains live-settled and is not repeated.

**Correction laptop result, 2026-10-05:** 106 focused streaming, Responses,
structured-output, and harness cases pass. The full backend suite passes 3,328
tests with the existing FastAPI deprecation warning. Scoped Ruff, compilation,
and diff checks pass. Regressions cover empty/partial token-limit stops,
whitespace-only completion, bounded fallback, internal progress observations,
literal answer text, the actual image/vl route, and the one-case smoke selector.

**Final live result, 2026-10-05: Passed.** The streamed-only retry returned
HTTP 200 and the 22-character answer "Dummy PDF file, Python". Five answer
deltas across 13 events ended with `response.completed`; `progress_hidden` was
true and `incomplete_details` was null. Usage was 1,177 input tokens and 1,670
output tokens under the explicit 8,192-token ceiling. No uploads were created.
Together with the already-passed normal answer and private-URL guards, this
closes Slice 13D. Do not repeat these settled checks without a relevant change.

Token-limit behavior follows the
[official OpenAI reasoning contract](https://developers.openai.com/api/docs/guides/reasoning).

Official URL field references: [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs)
and [OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision).

## Slice 13E - client-executed function tools

**Status:** Laptop implementation complete; targeted live gate pending.
This API capability lets a bot or other client advertise its own functions,
receive validated call requests, execute those functions itself, and submit
results for a final answer. It adds no Audrey browser controls or document
creation features.

### Models and execution policy

Use an existing permitted `audrey_passthrough/<concrete-model>` id. Audrey
checks `passthrough.enabled`, `require_role`, and `allowed_models`, then verifies
that Ollama reports the concrete model's `tools` capability. These checks finish
before file admission or SSE starts. Unavailable capability metadata fails
closed with HTTP 502; a model without tools returns HTTP 400.

Audrey's server tool registry and dispatcher never execute these functions.
Virtual pipeline models continue to use their existing server-managed tools;
client function controls on those models return HTTP 400. Native `direct/...`
model policy remains a separate catalog. Fair GPU scheduling, per-user limits,
configured thinking policy, sampling options, token usage, and authentication
reuse the existing passthrough path. Like existing passthrough requests, these
calls do not create compatibility chat history or stored Responses records.

### Request and replay contract

Definitions use the flat Responses function shape:

```json
{
  "model": "audrey_passthrough/qwen3.8:latest",
  "max_output_tokens": 8192,
  "tools": [{
    "type": "function",
    "name": "get_weather",
    "description": "Read current weather for a city from the client.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    }
  }],
  "tool_choice": "auto",
  "input": [{"role": "user", "content": "Check the weather in Denver."}]
}
```

A successful call item has `type: function_call`, an `fc_` item id, a distinct
`call_id`, the advertised `name`, completed status, and JSON-string `arguments`.
A function-only answer has no empty assistant message. Ordinary answer text
can accompany calls in a separate message item.

The client must validate and execute the requested function, then send another
request containing the original input, every item from `response.output`, and
one result per call:

```json
{
  "type": "function_call_output",
  "call_id": "call_id_from_the_response",
  "output": "{\"temperature_c\":18,\"conditions\":\"clear\"}"
}
```

Keep the advertised definitions when requesting further calls. `tool_choice:
"none"` withholds definitions from the provider for final synthesis. Historical
calls may also be replayed without current definitions. Every result must match
one unique preceding unanswered call; all pending calls require results before
a later message or generation. Adjacent parallel calls and their accompanying
assistant text become one provider assistant turn. Client input is stateless:
there is no lookup by response id.

Only `tool_choice: auto` or `none` is supported. Omission means auto.
`parallel_tool_calls: false` rejects a provider batch containing multiple calls
before exposing executable items. Required/forced choices, built-in tools,
combined skills, combined JSON Schema text output, stored/chained responses,
and background execution remain explicit HTTP 400 boundaries.

### Validation and limits

Audrey admits the documented [bounded JSON Schema subset](#slice-13b---structured-text-output)
and validates every generated call's name and arguments against the advertised
schema before returning any executable item. `strict: true` requires all object
properties to be required and `additionalProperties: false`. Missing `strict`
is treated and echoed as false; Audrey does not apply OpenAI's default strict
normalization. The provider receives native function definitions. Local final
validation does not guarantee constrained generation or that auto will call a
function on every model draw.

| Budget | Limit |
|---|---|
| Function definitions | 16; 64 KiB aggregate JSON |
| Function names | 1–64 letters, digits, underscores, or hyphens |
| Function description | 1,024 characters |
| Call arguments | JSON object; 64 KiB per call |
| Calls per pending/output group | 8 |
| Client replay input | 128 items |
| Result text | 100,000 characters per item |
| Adapted provider text, history arguments/results, instructions, and effective definitions | 250,000 characters and 16,000 `cl100k_base` tokens |

The aggregate prompt budget is checked before file access, after document
hydration, and after vision descriptions. Submitted definitions always undergo
their separate size and schema admission, including with `tool_choice: none`.
Existing owned-file and public-URL input limits and guards remain authoritative.

### Streaming and failure behavior

Function items use `response.output_item.added`,
`response.function_call_arguments.delta`,
`response.function_call_arguments.done`, and `response.output_item.done`, followed
by the typed terminal response. Ids and output indices are stable; sequence
numbers are contiguous from zero. Text streams through the ordinary output-text
events. Reasoning fields and native tool/progress events are not answer text.

Ollama sends parsed argument objects. Audrey collects calls from every chunk,
waits for confirmed successful termination, validates the entire batch, and
then emits one complete argument delta per call. Clients must wait for completed
call items before execution. Invalid batches expose no executable items;
completed requests return HTTP 502 and streamed requests end in response.failed.

A token-limit stop returns response.incomplete, partial text and usage, and
`incomplete_details.reason: max_output_tokens`; buffered calls are discarded.
An unconfirmed stream EOF returns response.failed with
`responses_stream_interrupted`, preserving partial text without attributing the
interruption to the token limit. Closing or cancelling a stream closes its
provider before releasing the GPU slot and releases the per-user slot.

### Verification and live gate

**Laptop result, 2026-10-05:** 3,519 full hermetic backend tests passed,
including 82 new tool-contract cases, 62 new route/provider integration cases,
45 new smoke-harness cases, and two shared provider-cleanup regressions. Scoped
Ruff, Python compilation, and diff checks passed. The existing FastAPI
deprecation warning remains.

Hermetic coverage includes real Ollama request/stream adaptation, call and
result links, parallel replay, malformed provider output, token truncation,
pre-fetch model/control denials, prompt budgets, owned evidence, and stream
cancellation. The separate smoke harness is also tested without live services.

Run [the targeted 13E protocol smoke](../reference/live-smoke-testing.md#run-the-13e-responses-client-tools-smoke-from-the-laptop)
after rebuilding Audrey. No browser upload is needed: the harness requests a
read-only local marker function, executes it on the laptop, and proves the
final answer uses that result. It makes three generation calls and two JSON
denial probes. Slice 13D remains passed and is not repeated.

Contract sources: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling),
[Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events),
and [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling).
