# Campaign 3 Phase 13 - Responses multimodal input

**Status:** Slices 13A and 13B are laptop-complete; each targeted live gate is pending.

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

## Deliberate boundary

Slice 13A rejects these before generation:

- HTTP or HTTPS image URLs;
- OpenAI `file_id` image references;
- GIF and unrecognized image MIME types;
- malformed or empty base64 data;
- image parts on non-user roles;
- unknown content-part types such as output-only `output_text`.

Remote fetching needs a separate SSRF-safe fetch contract. File ids need an
explicit mapping to Audrey's owner-scoped file store. `input_file` document
parts remain a later slice. The existing HTTP 400
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

## Next slice

After Slice 13B passes live, choose the next bounded Responses capability.
Keep remote and file-id inputs, client tools, stored response chaining, and
background execution in separate slices.
