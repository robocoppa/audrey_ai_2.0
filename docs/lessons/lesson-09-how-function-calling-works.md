# Lesson 9 - How function calling actually works

**Estimated time:** 60-80 minutes if you read with one source file open
([`src/audrey/tools/discovery.py`](../../src/audrey/tools/discovery.py)) and the
Lesson 8 worked example handy.

**Goal:** by the end of this lesson, you can answer
*"where does that `tool_calls` JSON come from? What is the model actually being
told, what is it allowed to do, and what changes if I switch from Ollama to
Anthropic to OpenAI?"*

Lesson 8 used function calling. It showed Audrey building a `tools` list,
handing it to a model, receiving `tool_calls`, dispatching them, feeding
results back, and looping. The shape was treated as given.

This lesson opens the shape. We will look at the protocol from first
principles — what a "tool" actually is from the model's point of view, why
the wire format looks the way it does, how the model is taught to use it,
and what changes between provider dialects.

There are three ideas to keep separate:

```text
the protocol   - what the wire format is on the request and response side
the contract   - what the model is promising when it emits tool_calls
the loop       - why function calling is multi-turn, not single-shot
```

Lesson 8 was a code tour. This one is more conceptual, but every concept
lands on a citation in Audrey's source and that's what we will be diving into in more detail in this lesson.

## 1. Context

### The pre-function-calling world

Before mid-2023, "use a tool" meant prompt engineering. You would put
something like this in the system message:

```text
You can use the following tools. When you need one, respond with:

ACTION: web_search
INPUT: {"query": "..."}

Then wait. I will give you the output as OBSERVATION: {...}.
Then you can either call another tool or answer.
```

You would then parse the model's reply looking for `ACTION:` and
`INPUT:` lines, run the tool, paste the result back as
`OBSERVATION:`, and ask the model to continue. This is the original
ReAct prompt pattern — Yao et al., 2022 — and it works, but it is
fragile. The model decides the format. Small models forget the prefix.
Larger ones drift over many turns. Parsing is regex-shaped and breaks
on the first model that puts a colon inside a query string.

The pain was real enough that every major provider eventually shipped a
structured channel: a separate field in the request and response
specifically for tool calls. OpenAI shipped function calling in June
2023, then renamed it to tool use; Anthropic shipped tool use in 2024;
Ollama added compatibility for the OpenAI shape soon after. They are
not identical, but the conceptual model is the same:

```text
old way   - tools live in prose; you parse the model's output
new way   - tools live in a structured field; the model emits structured calls
```

The big win is not that the new way is more expressive, the old way
could call any tool you described, it is that the new way is **a
contract**. The model is trained to emit tool calls in a specific
JSON shape. The provider's API guarantees the shape on the response
side. You stop parsing free text and start reading a typed field.

### The conceptual model

Function calling is built around four things:

```text
1. A tools array          - sent on every request, describes what's available
2. A tool_calls array     - returned by the model when it wants to call something
3. A role:"tool" message  - your job: run the call, put the result in here
4. A loop                 - the model gets the result, decides what to do next
```

The model never runs the tool. The model never sees the tool's URL,
authentication, or implementation. The model sees a name, a
description, and a JSON Schema describing the arguments. It emits a
call. You — the caller — run the tool and send the result back as a
new message in the conversation. The next request includes that
message, and the model decides whether to call another tool or
answer.

That last point is what makes function calling **multi-turn**. A single
chat completion can produce a tool_calls response that does not end the
conversation. You have to round-trip: model → tool_calls → you →
tool result → model → tool_calls or content → ...

Lesson 8 traced this loop through Audrey's `run_react` function. This
lesson is about what happens on the wire at each step.


## 2. Read-along

### 2.1 A single round, byte-by-byte

Here is what Audrey actually sends to Ollama when a user asks "what's
the latest version of BTRFS?":

```json
{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "system", "content": "You are Audrey..."},
    {"role": "user",   "content": "what's the latest version of BTRFS?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Query the public web for current information...",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "search query"},
            "count": {"type": "integer", "minimum": 1, "maximum": 10}
          },
          "required": ["query"]
        }
      }
    },
    {"type": "function", "function": {"name": "kb_search", "...": "..."}},
    {"type": "function", "function": {"name": "memory_recall", "...": "..."}}
    // ... and so on for every discovered tool
  ],
  "stream": false
}
```

Three things to notice:

1. **The `tools` array is sent on every request.** The model has no
   memory of "tools that existed last turn." If Audrey did not include
   the array, the model would not know any tools exist. (Some
   providers cache it, but conceptually the model treats the array as
   fresh every time.)
2. **Each tool has three fields the model uses:** `name`,
   `description`, and `parameters`. The description is what the model
   reads when deciding *whether* to call this tool. The parameters
   schema is what it must conform to when deciding *what* to send.
3. **Audrey did not write this JSON by hand.** It was built by
   `to_ollama_tool` on `ToolSpec` —
   [`tools/discovery.py:46`](../../src/audrey/tools/discovery.py#L46):

```python
def to_ollama_tool(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        },
    }
```

Now the response. The model decides this question needs a search and
emits:

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_0",
        "function": {
          "name": "web_search",
          "arguments": {"query": "btrfs latest version 2026"}
        }
      }
    ]
  },
  "done_reason": "stop"
}
```

This is the same JSON Lesson 8 §2.5 used as its illustration. Now you
know where it came from: the model emitted it because it was trained
to, given a request whose `tools` array advertised `web_search` with a
description matching the user's query.

Audrey runs `dispatch_one` (Lesson 8 §2.6), gets the search results,
and packages them as a new message via `to_tool_message` —
[`tools/dispatch.py:200`](../../src/audrey/tools/dispatch.py#L200):

```python
def to_tool_message(result: ToolResult) -> dict[str, Any]:
    """Build the OpenAI-shaped `role=tool` message for the next ReAct round."""
    msg: dict[str, Any] = {
        "role": "tool",
        "name": result.name,
        "content": result.content,
    }
    if result.call_id:
        msg["tool_call_id"] = result.call_id
    return msg
```

The next chat request to Ollama includes that `role: "tool"` message
appended to the conversation. The model now has the search results in
context and produces round two — either another tool call or a final
answer. That round trip is the protocol's basic unit.

### 2.2 Concept spotlight — JSON Schema and what models actually read

The `parameters` field is JSON Schema. Audrey gets it from the
tools-server's `/openapi.json` (which FastAPI auto-generates from the
Pydantic models in the route signatures — Lesson 8 §2.1). That schema
contains everything FastAPI uses to validate incoming requests:
required fields, types, descriptions, optional constraints like
`minimum`, `maximum`, `pattern`, `format`.

The model reads this schema and tries to conform. But "tries" is the
load-bearing word. Two things are true at once:

- The protocol guarantees the **shape**: the model will emit a
  `tool_calls` array whose entries have `name` and `arguments` (or
  `function.name` and `function.arguments` in the OpenAI envelope).
- The protocol does **not** guarantee the **content**: the arguments
  object might not match your schema. Smaller models routinely emit
  arguments that are missing required fields, have wrong types, or
  invent fields that don't exist.

Provider tooling helps in two ways:

1. **Training.** Modern instruction-tuned models have seen huge
   quantities of "here is a schema, emit a conforming call" examples.
   They are biased toward valid output, but it is bias, not a
   guarantee.
2. **Constrained decoding.** Some inference stacks force the model's
   output through a grammar derived from the schema. Each token is
   sampled only from the set that keeps the output parseable. This is
   what powers OpenAI's "strict mode" and Anthropic's tool-use
   guardrails. Ollama supports constrained decoding for plain JSON
   output (the `format` parameter) but **not for the `tool_calls`
   path** as of this writing — Ollama's tool calling is unconstrained,
   which is part of why small local models hallucinate arguments more
   often than larger cloud models.

This is why Audrey strips schema features the protocol theoretically
supports but small models choke on. From
[`tools/discovery.py:99`](../../src/audrey/tools/discovery.py#L99),
`_strip_unsupported_keywords` removes things like `format: "email"`,
top-level `oneOf`, and unevaluated property constraints. The full
JSON Schema spec is rich; the subset that survives across model sizes
is small.

**The rule of thumb:** fewer schema features = more reliable tool
calls. Required fields, plain types (`string`, `integer`, `array`,
`object`), descriptions, and `enum` for closed sets are safe. Almost
everything else is a risk on smaller models.

### 2.3 How the model decides to call a tool

A common mental model: "the model reads the description and pattern-
matches it against the user's request." That is roughly right but
incomplete. What actually happens at inference time:

1. The model receives the full prompt — system message, conversation,
   tools array. The provider's API typically renders the tools array
   into the prompt as a structured prefix the model has been trained
   to recognize. (You can sometimes see it in raw model output if
   tool calling is misconfigured — the model will print the tool
   schema verbatim.)
2. The model generates tokens. For instruction-tuned tool-use models,
   the training distribution rewards "if the user's request maps to
   one of these tool descriptions, emit the structured call form
   rather than free text."
3. When the model produces the special tokens that indicate the start
   of a tool call, the inference stack switches into tool-call output
   mode: the assistant message has empty `content` and the
   `tool_calls` array is populated from the model's output.

Three consequences worth internalizing:

- **The model can ignore the tools entirely.** Tools are optional. The
  model is allowed to answer from training-time knowledge even when a
  perfectly applicable tool is in the array. Lesson 8's force-final-
  answer behavior on the last round (§2.11) leans on this: by
  dropping the `tools` array entirely from the final request, Audrey
  guarantees the model produces prose, not another tool call.
- **The description matters more than the name.** The model uses both,
  but "search the public web for current information" carries more
  signal than the literal string `web_search`. If you give two tools
  similar descriptions, the model will get confused about which to
  call.
- **Temperature, system prompt, and prior messages affect the
  decision.** A higher temperature makes the model more likely to
  emit unexpected tool calls (or skip them). A system message that
  says "use tools aggressively" shifts the bias; one that says
  "answer from your own knowledge first" shifts it the other way.
  Audrey's defaults are tuned for a middle ground in
  [`config.yaml`](../../config.yaml).

### 2.4 Dialect tour — OpenAI, Anthropic, Ollama

The conceptual model is shared. The wire shape differs. Here is the
same `web_search` call in all three envelopes.

**OpenAI (and Ollama, which mirrors OpenAI)**

Request:

```json
{
  "model": "gpt-4o-mini",
  "messages": [...],
  "tools": [
    {"type": "function", "function": {"name": "web_search", "parameters": {...}}}
  ],
  "tool_choice": "auto"
}
```

Response:

```json
{
  "choices": [{"message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "call_abc123",
      "type": "function",
      "function": {"name": "web_search", "arguments": "{\"query\":\"...\"}"}
    }]
  }}]
}
```

Note the OpenAI quirk: `arguments` is a **string** containing JSON,
not a JSON object. Audrey's dispatcher handles both shapes — Ollama
sometimes returns it pre-parsed as a dict, OpenAI returns it as a
string. Lesson 8 §2.6 lists this as one of `dispatch_one`'s five
explicit responsibilities ("Ollama sometimes returns `arguments` as a
JSON-encoded string instead of a dict").

The next request follows up with a `role: "tool"` message and a
`tool_call_id` that matches the call's `id`:

```json
{"role": "tool", "tool_call_id": "call_abc123", "content": "{...search results...}"}
```

The `tool_call_id` is load-bearing: it lets the model pair a result
with the call that produced it. This matters when the model emitted
multiple parallel calls in one response (see §2.6).

**Anthropic**

Request:

```json
{
  "model": "claude-opus-4-7",
  "messages": [...],
  "tools": [
    {
      "name": "web_search",
      "description": "...",
      "input_schema": {"type": "object", "properties": {...}, "required": ["query"]}
    }
  ]
}
```

Response (content blocks rather than a separate field):

```json
{
  "content": [
    {"type": "text", "text": "I'll search for that."},
    {"type": "tool_use",
     "id": "toolu_01ABCD",
     "name": "web_search",
     "input": {"query": "btrfs latest version 2026"}}
  ],
  "stop_reason": "tool_use"
}
```

Differences worth knowing:

- The tool definition uses `input_schema` instead of `parameters`, and
  the top-level fields are flat (no `type: "function"` wrapper).
- The response is structured as **content blocks** — an array where
  each entry is typed (`text`, `tool_use`, `thinking`, etc.) rather
  than a single message with side fields. A model can emit thinking,
  then text, then a tool call, then more text, all in one response.
- The arguments live in `input` (already a JSON object, not a string).
- The follow-up message uses a `tool_result` content block whose
  `tool_use_id` matches the call's `id`, embedded in a `user`-role
  message:

```json
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_01ABCD", "content": "..."}
]}
```

**Ollama**

Mirrors OpenAI's tools shape closely, but with quirks:

- Some smaller models advertised as tool-capable will emit malformed
  tool calls — for example, wrapping `arguments` in a markdown code
  fence, or including stray prose before the JSON. Audrey's
  `tool_capable_models` allow-list in
  [`config.yaml`](../../config.yaml) is the human-curated answer to
  "which Ollama models actually emit clean tool calls in practice"
  (Lesson 8 §2.12 covered this in passing). Ollama's own `capabilities`
  field on a model is sometimes wrong.
- Constrained decoding is not applied to tool calls. The model is free
  to emit arguments that violate the schema; you must validate at
  dispatch time.
- The wire shape is otherwise OpenAI-compatible, which is why Audrey
  uses `to_ollama_tool` for everything (the name is historical — it's
  really "to OpenAI-style tool", and the same JSON works against
  OpenAI's API if you pointed Audrey at it).

The three providers differ in their envelopes but agree on the
contract: the model emits a structured call with a name, an
identifier, and an arguments object; you run it and feed the result
back as a new message keyed to that identifier.

### 2.5 Concept spotlight — the multi-turn loop, protocol view

A single chat completion is not enough to use a tool. The minimum
sequence is two completions:

```text
round 1
  request: messages + tools array
  response: assistant message with tool_calls

  (you run each tool call, build a role:"tool" message per result)

round 2
  request: messages + tool result messages + tools array
  response: assistant message with content (final answer)
            OR another tool_calls round
```

The protocol shape (using OpenAI as the canonical):

```text
messages: [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant",
   "content": "",
   "tool_calls": [{"id": "call_0", ...}]},          ← from round 1's response
  {"role": "tool", "tool_call_id": "call_0", "content": "..."}, ← your work
  {"role": "assistant", "content": "Here's the answer..."}      ← round 2's response
]
```

Three things make this work:

1. **The assistant message from round 1 stays in the conversation.**
   The model's tool-call output is itself a message in the history.
   Removing it would leave the `tool_call_id` dangling.
2. **Each `role: "tool"` message must reference a `tool_call_id` from
   the immediately preceding assistant message.** Skipping this
   pairing — or sending tool results before the assistant message
   they answer — is a protocol error. Most providers will 400.
3. **The model can keep calling tools indefinitely.** There is no
   protocol-level round limit. The caller imposes one. Lesson 8
   §2.11 covered Audrey's: `agentic.react.max_rounds` (default 3).
   On the last round, the `tools` array is omitted, forcing prose.

This is why function calling feels heavier than plain chat completion:
a single user message can produce 2, 3, or more provider round-trips
before the user sees a reply.

### 2.6 Concept spotlight — parallel tool calls in one response

A modern model can emit multiple tool calls in a single round:

```json
"tool_calls": [
  {"id": "call_0", "function": {"name": "web_search", "arguments": {...}}},
  {"id": "call_1", "function": {"name": "kb_search",  "arguments": {...}}},
  {"id": "call_2", "function": {"name": "memory_recall", "arguments": {...}}}
]
```

The protocol allows this. The caller's job is to run all three and
return three `role: "tool"` messages (one per `tool_call_id`) in the
follow-up request. The order of results in the conversation does not
have to match the order of calls — providers match by id, not by
position.

Audrey dispatches them concurrently — Lesson 8 §2.9 walked the
`asyncio.gather` block in `react.py`. The protocol does not require
this; you could run them serially and the model would still get the
right results. Concurrency is an optimization, not a correctness
requirement.

When does the model emit parallel calls? Roughly: when the user's
request can be decomposed into independent sub-questions, **and** the
model has been trained to do this. Larger frontier models do it
spontaneously; smaller local models almost never do, even when it
would help. This is one of the gaps that closes with model size.

### 2.7 Concept spotlight — `tool_choice` and forced calls

Most providers expose a `tool_choice` field on the request. Common
values:

- `"auto"` — the model decides whether to call any tool. Default.
- `"none"` — the model must not call a tool this round. Forces prose.
- `"required"` (OpenAI) / `tool_choice.type = "any"` (Anthropic) — the
  model must call *some* tool. Useful when you know a tool is needed
  but want the model to pick which.
- `{"type": "function", "function": {"name": "web_search"}}` — the
  model must call *this specific tool*. Useful for routing.

Audrey does not use `tool_choice` explicitly. Instead, it controls
behavior at a coarser level:

- **Force prose on the last round** (Lesson 8 §2.11): drop the `tools`
  array entirely from the final request. This is equivalent to
  `tool_choice: "none"` but works across providers that don't
  implement that field. The model literally cannot call a tool it
  doesn't know exists.
- **Per-model gating** (Lesson 8 §2.12): if the configured model is
  not in `tool_capable_models`, Audrey skips ReAct entirely and does
  a plain chat completion. There is no `tools` array on the request,
  no tool-use code path active.

Both of these are protocol-level: by controlling what enters the
request, Audrey controls what can come out.

### 2.8 Failure modes the protocol exposes

Function calling adds five failure modes that plain chat completion
does not have. Each is real; each shows up in production.

**1. Model emits invalid JSON in `arguments`.**

Protocol cause: the model is generating tokens; nothing forces the
output to be valid JSON. Common on smaller Ollama models. Audrey
handles it in `dispatch_one` — the `JSONDecodeError` branch in
[`tools/dispatch.py`](../../src/audrey/tools/dispatch.py) returns a
structured `ToolResult` with `is_error=true` and the raw arguments
echoed back so the model can see what it sent and try again. The
loop then continues. Mitigation at the protocol level: use a model
with constrained decoding, or upgrade to a larger model.

**2. Model calls a tool that doesn't exist.**

Protocol cause: nothing prevents the model from emitting
`function.name = "search_the_web"` when the tool was registered as
`web_search`. Even if the model has the schema in front of it, an
extra token can produce a hallucinated name. Audrey's dispatcher
detects unknown names and returns an error result naming the missing
tool. The model usually self-corrects on the next round once it sees
the error. Mitigation: keep tool names short, distinct, and easy to
emit token-by-token.

**3. Model loops on the same tool forever.**

Protocol cause: each round, the model sees the previous tool result
and decides whether to act. If the result wasn't useful and the
model thinks "try again with different arguments," it will. Without
a round budget, this can run for as many rounds as you allow.
Mitigation: a strict round limit. Audrey's `max_rounds` (default 3)
plus the force-prose-on-last-round trick handle this.

**4. Tool returns enormous output.**

Protocol cause: there is no provider-side limit on `role: "tool"`
message content size. A `kb_search` that returns 50 KB of matched
chunks will all go into the next request — eating context budget,
slowing the model, and potentially exceeding the model's context
window. Audrey truncates at `max_tool_result_chars` before building
the message (Lesson 8 §2.8). Mitigation is always caller-side; the
protocol won't help.

**5. Model ignores the tool entirely and answers from training data.**

Protocol cause: tools are optional. The model is allowed to skip the
`tool_calls` path and produce a `content` answer even when calling a
tool would clearly be better. This is **not a bug**. If the model is
confident the answer is in its training data, it has no obligation
to call `web_search`. Mitigation: better system prompts ("for any
question about current events, prefer web_search over your own
knowledge"), or use `tool_choice: "required"` if your provider
supports it.

These are the protocol's failure modes. Audrey's loop design — Lesson
8 — is largely the catalogue of how it handles them.


## 3. Comprehension questions

These are scenarios you might hit. Try to answer from the protocol
first, then check against Audrey's actual code path.

**1. "A new local model emits `tool_calls` but the arguments are
wrapped in markdown code fences (```json ... ```). Where in the
protocol does that break, and what would fix it?"**

The protocol expects `arguments` to be either a JSON object or a JSON
string (depending on dialect). Markdown is neither — it's a string
that contains JSON wrapped in formatting tokens. The model is
emitting text where the inference stack should have switched into
tool-call output mode. The break is at step 2 of §2.3: the model
produced "I should call a tool" but didn't emit the special tokens
that switch decoding modes. Fix options: (a) drop the model from
`tool_capable_models` in
[`config.yaml`](../../config.yaml); (b) upgrade Ollama to a version
with better tool-call detection; (c) write a pre-parse step in
`dispatch_one` that strips code fences before `json.loads`. Audrey
does (a) today.

**2. "Why does the response from a tool round include `tool_call_id`
instead of just sending the result alone?"**

Because the model can emit multiple tool calls in a single round
(§2.6). Without the id, there is no way to pair a result back with
the call that produced it. The provider would have to guess based on
ordering, and ordering isn't guaranteed when the caller dispatches
concurrently. The id makes the pairing explicit and unambiguous.

**3. "You add a tool whose schema has `oneOf` at the top level.
Audrey's small Ollama models start ignoring it. What's the protocol-
level explanation?"**

Two layers. First, `_strip_unsupported_keywords` in
[`tools/discovery.py:99`](../../src/audrey/tools/discovery.py#L99)
will remove the `oneOf` before the schema is ever sent — so the model
sees a schema that no longer matches the underlying endpoint, and the
"required" hints get dropped. Second, even if the strip didn't happen,
small models trained on simpler schemas have not seen many `oneOf`
examples and will emit calls that match neither branch. The fix is
schema design: flatten `oneOf` into a single object with an optional
discriminator field, or split into two tools with distinct names.

**4. "An Anthropic-only feature lets the model emit `<thinking>`
blocks alongside `tool_use`. Why is that not part of OpenAI's
protocol, and what would you change in a hypothetical Anthropic
adapter for Audrey?"**

OpenAI's response shape has one assistant message with `content` and
optional `tool_calls` as side fields. Anthropic's response is an
array of typed content blocks, so an assistant message can contain a
thinking block, then a text block, then a tool_use block. The data
model is different. A hypothetical Anthropic adapter would have to
(a) translate `content` blocks into a single string or drop thinking
blocks; (b) extract `tool_use` blocks into Audrey's expected
`tool_calls` array; (c) build follow-up messages using
`tool_result` content blocks in a `user`-role message instead of
`role: "tool"`. The conceptual loop is the same; the envelopes
differ.

**5. "A user reports 'the model said it called web_search but my logs
show no dispatch.' Walk the protocol — at what step would the call
have been dropped?"**

The model is permitted to say "I'll search for that" in prose without
emitting an actual `tool_calls` array. This is step 3 of §2.3 — the
model generated text describing what it might do but never switched
into tool-call output mode. The protocol does not consider this an
error. Audrey will see a normal assistant message with `content` and
no `tool_calls`, so the ReAct loop exits and dispatch never runs.
The user sees a response that "talks about searching" but no search
happened. Fix: either tune the system prompt to discourage describing
intended tool calls, or detect this pattern and force a retry with
`tool_choice: "required"` if available. Most often this is a
prompt-engineering fix, not a code fix.


## When you're ready for the next lesson

You have now walked the function-calling protocol end-to-end — request
shape, response shape, multi-turn loop, dialect differences, and
failure modes — and you have the citations to find the matching code
in Audrey when you need them.

The next lesson opens up the knowledge base: how documents and images
get extracted, chunked, embedded, indexed in Qdrant, and served back
through the `kb_search` and `kb_image_search` tools whose call shapes
you just learned to read.
