# Campaign 4 Phase 05 — Responses API foundation

**Status:** Complete and accepted. Expanded contracts are in [Phase 13](phase-13-responses-multimodal-input.md).

## Shared generation

`POST /v1/responses` adapts validated requests to Audrey's existing generation and policy paths. It preserves authenticated identity, virtual/passthrough model policy, explicit skills, scheduling, tool restrictions, usage, vision fallback, and applicable compatibility archiving.

Supported foundation fields include `model`, nonempty text `input`, optional `instructions` as a leading developer message, sampling/output limits, `metadata`, `user`, and Audrey's explicit `skill` extension. Authenticated identity always wins over a claimed user. `max_output_tokens` uses the existing output-limit mapping.

## Output and streaming

Completed responses use a `resp_` id and typed assistant/output-text items plus usage. There is no Chat Completions `choices` array.

Streaming uses typed Responses events with contiguous sequence numbers from zero, stable response/message ids, text deltas and completed items, then `response.completed`. It does not append `[DONE]`. Pipeline failures use `response.failed`; confirmed output-limit stops use `response.incomplete` with `max_output_tokens`. Partial answer text and usage are preserved. Normal completions without an answer fail instead of returning progress as text.

Responses answer text excludes Audrey's internal progress. Chat Completions and native chat retain their existing progress renderers. Both adapters render shared run events; neither parses the other's wire format.

## Boundaries

Phase 13 adds typed images/files, bounded public URL inputs, JSON Schema answers, caller-executed functions, and opt-in response storage/chaining. Background execution, the separate Conversations API, inline file data, and unknown request fields remain rejected. Do not advertise full OpenAI parity.
