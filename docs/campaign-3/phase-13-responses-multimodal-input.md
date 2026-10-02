# Campaign 3 Phase 13 - Responses multimodal input

**Status:** Slice 13A is laptop-complete and awaiting the targeted live gate.

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
`responses_feature_unsupported` boundary remains unchanged for structured
output, client tools, stored or chained responses, and background execution.

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

## Next slice

After Slice 13A passes live, take structured text output as its own Slice 13B.
Keep client tools, stored responses, chaining, and background work separate.
