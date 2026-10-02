# Campaign 3 Phase 5 - Responses API compatibility

**Status:** Complete. Slice 5A was live-settled on 2026-09-29 and Slice
5B's typed streaming contract passed its targeted live gate on 2026-09-30.
Later typed multimodal expansion is tracked in
[Phase 13](phase-13-responses-multimodal-input.md).

## Goal

Let OpenAI-compatible clients use Audrey through `POST /v1/responses`
without creating a second generation pipeline. The adapter must preserve
Audrey authentication, virtual and passthrough model policy, explicit skills,
fair scheduling, tool restrictions, usage accounting, and compatibility chat
archiving.

The contract follows the official OpenAI Responses API shape: requests may
separate `instructions` from `input`, and completed text is returned
inside typed `output` items rather than Chat Completions `choices`.

Official references:

- https://developers.openai.com/api/reference/cli/resources/responses/methods/create
- https://developers.openai.com/api/docs/guides/migrate-to-responses

## Slice 5A - completed plain-text responses

The native backend exposes:

    POST /v1/responses

This first slice accepts:

| Field | Behavior |
|---|---|
| `model` | Required. Uses the same Audrey virtual and passthrough model policy as Chat Completions. |
| `input` | A non-empty string or a non-empty text-only message list using system, developer, user, and assistant roles. |
| `instructions` | Optional. Adapted to a developer message ahead of the input. |
| `temperature`, `top_p`, `max_output_tokens` | Reuse the existing Ollama option mapping; `max_output_tokens` maps to Audrey's current `max_tokens` boundary. |
| `metadata`, `user` | Retained at Audrey's compatibility boundary. Authenticated identity still wins over `user`. |
| `skill` | Audrey extension. Uses the same explicit skill resolution and tool narrowing as Chat Completions. |
| `stream: false`, `background: false` | Accepted explicitly. |

The response includes a `resp_` id, completed status, one assistant message
item containing `output_text`, a matching top-level `output_text`, and
Responses-style token usage. It does not return a Chat Completions
`choices` array.

## Slice 5B - typed streaming responses

A request with `stream: true` now returns `text/event-stream` using the
Responses API's typed event vocabulary. The successful plain-text lifecycle is:

1. `response.created` and `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. one or more `response.output_text.delta` events;
4. `response.output_text.done`, `response.content_part.done`, and
   `response.output_item.done`;
5. `response.completed` carrying the final response object and token usage.

Every event has a contiguous `sequence_number` beginning at zero. The response
and message ids stay stable for the whole stream. Responses streams terminate
with their typed terminal event and do not append Chat Completions' `[DONE]`
marker. Pipeline failures use `response.failed`; an upstream stream that ends
without its required completion marker uses `response.incomplete`.

Both virtual models and permitted passthrough models use the same authenticated
generation, policy, fair-scheduling, metrics, and token-accounting paths as
Chat Completions. A stream-session factory selects only the outer renderer:
Chat Completions retains its existing chunks, while Responses renders the same
client-neutral run events as typed Responses SSE. No adapter parses another
adapter's wire format.

## Deliberate boundary

Slice 5A rejects these fields with HTTP 400 and
`responses_feature_unsupported` before generation starts:

- background execution;
- persisted response retrieval or chaining through `store`,
  `previous_response_id`, or `conversation`;
- client-provided tools;
- structured text output configuration.

Phase 13 Slice 13A now accepts typed `input_text` parts and bounded inline
`input_image` data URLs. Remote image URLs, file-id images, `input_file`,
and unknown top-level fields still fail request validation rather than
disappearing silently. Later slices add each remaining capability with its own
storage, event, fetch, or tool-call contract.

## Shared behavior

The route adapts a validated Responses request to the existing
`ChatCompletionRequest` and calls the same internal generation function after
authentication. It therefore keeps model validation, passthrough role and
allow-list gates, skill resolution, server-managed tool policy, vision
fallbacks, inflight limits, GPU fairness, generation metrics, and archive
behavior in one implementation.

The targeted live smoke uses Audrey's existing `### Task:` compatibility
form so the one model call is excluded from chat history. It also submits one
unsupported background request and proves rejection happens before generation.

## Laptop verification

- 17 focused Responses contract cases pass, including successful, failed,
  incomplete, and shared-generation streaming boundaries.
- 175 broader fast, deep, research, native, archive, and passthrough stream
  regressions pass.
- The full hermetic backend suite passes: 2,988 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff and smoke-script compilation pass.

## Targeted live result

**Result:** Passed on 2026-09-30 over the working LAN/WARP route at
`http://192.168.1.11:8000`.

The deployed stream returned HTTP 200 with 19 typed events and 11 text deltas.
It kept stable `resp_` and `msg_` ids, reported input and output token usage,
included the expected sentinel, and terminated with `response.completed`.
The separate `background: true` request returned HTTP 400 with
`responses_feature_unsupported`, proving the deliberate boundary still rejects
before generation.

No upload or browser action was needed. This evidence is settled and is not
repeated unless a later change touches Responses streaming or its shared run
event renderer.
