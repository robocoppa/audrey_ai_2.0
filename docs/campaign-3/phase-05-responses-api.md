# Campaign 3 Phase 5 - Responses API compatibility

**Status:** Slice 5A is live-settled on 2026-09-29. Slice 5B is laptop
complete on 2026-09-30 and awaits its targeted Unraid smoke.

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

Multimodal content parts and unknown top-level fields fail request validation
rather than disappearing silently. Later slices can add each capability with
its own storage, event, or tool-call contract.

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

## Targeted live gate

**Result:** Pending a backend rebuild and one API-only smoke from the laptop.
Slice 5A's completed-response proof remains accepted and is not repeated.

Rebuild the Audrey backend, then run from the laptop checkout:

    cd /home/bart/Documents/github/audrey/audrey_ai_2.0
    (
      set -a
      source .env.test.local
      set +a
      AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_responses_api.py
    )

The smoke makes one short `audrey_fast` streaming generation. It expects:

- HTTP 200 with `text/event-stream`;
- the ordered typed start, text-delta, item-done, and `response.completed`
  lifecycle with contiguous sequence numbers;
- stable `resp_` and `msg_` ids across the stream;
- concatenated deltas containing `RESPONSES_STREAM_OK` and exactly matching
  the completed response's `output_text`;
- internally consistent token usage, no Chat Completions `choices`, and no
  `[DONE]` marker;
- HTTP 400 with `responses_feature_unsupported` for `background: true`.

Success is exit code zero and JSON ending in `"status": "passed"`. No upload,
prepared file, or manual browser check is needed.
