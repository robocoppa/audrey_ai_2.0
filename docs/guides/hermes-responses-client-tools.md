# Hermes integration with Audrey Responses function tools

Audrey's completed call, streamed call, and stateless result replay passed the
live protocol gate on 2026-10-05. This is a handoff for each Hermes installation;
its adapter and deployment still need to be inspected and tested.

## Message to forward

> Audrey's Responses function tools have passed the live test. Inspect our
> installed Hermes provider adapter and wire this capability into the Audrey
> connection where needed.
>
> Use `http://192.168.1.11:8000/v1/responses`, the existing Audrey PAT with
> `compat:full`, and permitted model `audrey_passthrough/qwen3.8:latest`.
> Keep credentials out of logs. Audrey uses port 8000; port 11434 is direct Ollama.
>
> Send flat function definitions:
> `{"type":"function","name":"…","description":"…","parameters":{…},"strict":true}`.
> Audrey supports `tool_choice: "auto"` or `"none"` and
> `parallel_tool_calls: false`. Select at most 16 relevant Hermes functions.
>
> Read `function_call` items from `response.output`. Validate their JSON arguments
> and dispatch through Hermes's existing tool executor and approval rules.
> Send the original input, every returned output item, and one matching
> `{"type":"function_call_output","call_id":"…","output":"…"}` per call.
> Repeat until the model returns its final answer. Keep definitions when asking
> for more calls; use `tool_choice: "none"` for final synthesis.
>
> For streaming, assemble the typed function argument events, preserve item
> and call ids, and execute only after `response.completed`. Failed or
> incomplete responses must not trigger execution. Track call ids so retrying
> cannot execute the same function twice.
>
> Inspect the installed Hermes version before changing its adapter.
> `/v1/chat/completions` and `/v1/responses` use different message formats;
> changing only the endpoint URL is insufficient. Verify a completed call,
> a streamed call, and a final answer using a harmless local tool's result.
> Report changed files, adapter selection, model, and test results.

## Contract details

- Functions execute in Hermes. Audrey requests calls and validates arguments.
- A call item has an `fc_` item id, a distinct `call_id`, the declared function
  `name`, and JSON-string `arguments`. Function-only output has no text message;
  empty `output_text` is valid when `response.output` contains calls.
- Streaming uses `response.output_item.added`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and `response.output_item.done`.
  Audrey currently emits one validated complete argument delta per call.
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

The passed Slice 13E path is stateless. Slice 13F adds opt-in text/function
storage, but it must pass its deployment gate before Hermes relies on it.
With verified storage, `store: true` retains a successful response; a later
request can send its `previous_response_id` and only new inputs or call results.
Supply current instructions and definitions on each request. Deleting a saved
parent removes its descendants; expiry can also make a chain unavailable.

See the [Audrey implementation contract](../campaign-3/phase-13-responses-multimodal-input.md#slice-13e---client-executed-function-tools),
[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling),
and [Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events).
