# Campaign 3 Phase 13 - Responses multimodal input

**Status:** Slice 13B is live-settled. Slice 13A remains laptop-complete with its targeted live gate pending. Slice 13C is laptop-complete with its targeted file-reference live gate pending.

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

Slice 13A rejects these before generation:

- HTTP or HTTPS image URLs;
- OpenAI `file_id` image references;
- GIF and unrecognized image MIME types;
- malformed or empty base64 data;
- image parts on non-user roles;
- unknown content-part types such as output-only `output_text`.

Remote fetching needs a separate SSRF-safe fetch contract. Slice 13C below
adds the explicit mapping from file ids to Audrey's owner-scoped file store
and document extraction. The existing HTTP 400
`responses_feature_unsupported` boundary remains unchanged for client tools, stored or chained responses,
and background execution.

## Automated contracts

- Completed input adapts typed text plus inline image parts to the established
  Chat multimodal shape without starting a second vision pipeline.
- Streaming input uses the same adapter and still creates a
  `ResponsesStreamSession`.
- Message roles and ordering survive typed text adaptation.
- Remote URLs, file ids, unsupported MIME types, malformed base64, wrong-role
  images, and unknown part types fail validation.
- The targeted smoke generates a valid 32 by 32 red PNG in memory, submits it
  through Responses, requires a completed typed response and sentinel, then
  proves a remote URL returns HTTP 422 without generation.

## Laptop result

**Passed, 2026-10-01.** The focused Responses module passes 25 tests.
Changed-file Ruff and Python compilation pass. The full hermetic suite passes
3,106 tests with one existing FastAPI deprecation warning, and the diff check
is clean.

## Targeted live gate

Run `tests/smoke/smoke_responses_multimodal.py` from the laptop against the
working LAN/WARP backend route after rebuilding Audrey. A pass reports HTTP
200, a `resp_` id, `output_text`, the `RESPONSES_IMAGE_OK` sentinel,
integer usage, and HTTP 422 for a remote image URL.

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
responses_feature_unsupported response. Client tools, stored responses,
chaining, and background work remain separate future slices.

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

After rebuilding Audrey, run
`tests/smoke/smoke_responses_file_inputs.py` from the laptop at
`http://192.168.1.11:8000`. The exact command and credential names are in
[the live smoke runbook](../reference/live-smoke-testing.md#run-the-13c-responses-file-reference-smoke-from-the-laptop).

No browser action, video, or pre-existing library file is needed. This API
proof uploads its own tiny text document and red PNG under the smoke user,
waits for Ready, and makes two short Fast model calls: one completed and one
streamed. Both must read a randomized code from the document and identify the
image as red. The script also proves both reference types deny a second owner
with the same HTTP 404 as a missing id, rejects the wrong file kind, deletes
both temporary uploads, and proves deleted references return HTTP 404.

**Live status:** Two attempts have not passed. Expired Access assertions
blocked the first attempt before uploads. After refreshing credentials, the
second attempt uploaded and indexed both files, received an empty completed
answer, and deleted both uploads. A follow-up raises the smoke's output ceiling
from 64 to 4,096 tokens, using the existing vision measurements, and preserves
Fast token-limit status instead of declaring success. The next failure report
will include response status, usage, answer length, and any incomplete reason.
The actual live stop cause remains unconfirmed; rebuild and rerun this gate.

HTTP and HTTPS image or document URLs, inline `file_data`, PDF visual detail,
client-provided tools, stored response chaining, and background execution remain
separate capabilities. Remote fetching requires an explicit SSRF-safe contract.

Contract references: [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs)
and [Responses request schema](https://developers.openai.com/api/reference/resources/responses/methods/create).
