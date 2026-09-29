# Campaign 3 Phase 5 - Responses API compatibility

**Status:** Slice 5A is laptop-complete on 2026-09-29. Its targeted live
smoke remains open.

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

## Deliberate boundary

Slice 5A rejects these fields with HTTP 400 and
`responses_feature_unsupported` before generation starts:

- streaming and background execution;
- persisted response retrieval or chaining through `store`,
  `previous_response_id`, or `conversation`;
- client-provided tools;
- structured text output configuration.

Multimodal content parts and unknown top-level fields fail request validation
rather than disappearing silently. Later slices can add each capability with
its own storage, event, or tool-call contract.

## Shared behavior

The route adapts a validated Responses request to the existing
`ChatCompletionRequest` and calls the same route function after
authentication. It therefore keeps model validation, passthrough role and
allow-list gates, skill resolution, server-managed tool policy, vision
fallbacks, inflight limits, GPU fairness, generation metrics, and archive
behavior in one implementation.

The targeted live smoke uses Audrey's existing `### Task:` compatibility
form so the one model call is excluded from chat history. It also submits one
unsupported streaming request and proves rejection happens before generation.

## Laptop verification

- 14 focused Responses contract cases pass.
- The full hermetic backend suite passes: 2,984 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff and smoke-script compilation pass.

## Targeted live gate

Rebuild the Audrey backend, then run from the laptop checkout:

    cd /home/bart/Documents/github/audrey_ai_2.0
    (
      set -a
      source .env.test.local
      set +a
      AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python scripts/smoke_responses_api.py
    )

The smoke uses `AUDREY_EVAL_API_KEY` from `.env.test.local`. It makes
one short `audrey_fast` generation and expects:

- HTTP 200, `object: "response"`, and `status: "completed"`;
- a `resp_` id and one completed assistant `output_text` item;
- matching top-level and item text containing `RESPONSES_OK`;
- internally consistent token usage;
- no Chat Completions `choices`;
- HTTP 400 with `responses_feature_unsupported` for `stream: true`.

Success is exit code zero and JSON ending in `"status": "passed"`. No upload
or manual browser check is needed.
