# Lesson 9 - Tool use and the ReAct loop

**Estimated time:** 60-80 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer
*"when a model needs to call a tool, how does Audrey find the right server,
send the call, return the result, and decide when to stop?"*

Lesson 7 traced how a request becomes a `task_type` and a `mode`, and Lesson 8
walked the four-stage deep pipeline (planner → panel → synthesize → reflect)
that runs when `mode = deep`. This lesson picks up one step lower: after the
fast path or a deep worker has chosen a model. If that model is tool-capable
and tools exist, Audrey hands the request to the **ReAct loop**: a small
driver that lets the model call tools, feeds the results back, and forces a
final answer when the round budget runs out.

There are four ideas to keep separate:

```text
discovery   - learn what tools exist
dispatch    - run one tool call against the right server
react       - loop: chat -> dispatch -> chat -> ... -> answer
budget      - stop the loop and force prose at max_rounds
```

The whole lesson is built around `web_search` as the worked example because
the round trip is short and the schema is intuitive.


## 1. Context

Audrey is a self-hosted orchestrator that wants its models to ground answers
in fresh information. "What's the latest version of BTRFS?" should not be
answered from training cutoff guesses. Tool use is how the model reaches
outside the prompt: search the web, query a knowledge base, look up a stored
memory.

The trade is:

```text
without tools  - one chat call, one answer, fast, possibly stale
with tools     - many chat calls, fresher answer, more latency
```

Audrey leans toward "use a tool when it helps, then stop." Tool use is opt-in
per model (some local models can't emit `tool_calls`), opt-in per request
(the model decides each turn), and time-bounded (the loop has a fixed round
budget). When the model finally returns prose with no tool calls, the loop
exits.

The end-to-end shape:

```text
startup
  -> discover_all() reads /openapi.json from each tool server
  -> ToolRegistry maps name -> ToolSpec

request arrives
  -> classify + complexity pick task and mode
  -> fast_path or deep_panel picks a model
  -> if model in tool_capable_models AND registry is non-empty
       -> run_react(...)
       -> loop:
            chat(messages, tools=registry.to_ollama_tools())
            if tool_calls: dispatch each in parallel, append results
            else: return content
       -> at max_rounds: force a final answer without tools

return answer
```

This lesson focuses on the boxed middle section. Lesson 6 covered model
selection and gate scheduling; Lesson 7 covered classify/complexity/mode;
Lesson 8 covered how the panel dispatches deep workers in the first place.
Here we open the *inside* of one of those workers (or of a fast-path
single-model call) when that model decides to use a tool.

### Why "ReAct"?

ReAct is shorthand for "reason and act" — a 2022 paper by Yao et al.
describing a prompt pattern where the model alternates between **thinking**
(reasoning aloud about what to do next) and **acting** (calling a tool, then
seeing the result). Modern function-calling models build the same loop into
their tool-use protocol: each round, the model either emits a `tool_calls`
array (act) or returns prose (reason and answer).

Audrey's loop is a faithful implementation of that pattern. Nothing exotic —
the cleverness is in the budget and the failure handling, not in the loop
shape.

### Two ReAct contexts in Audrey

The same loop runs in two places with slightly different scheduling rules.
Hold both in your head:

| Context | Where | Round budget | Gate policy |
| --- | --- | --- | --- |
| Fast path | one model answering directly | `agentic.react.max_rounds` (default 3) | Gate passed *into* ReAct; released between rounds during tool dispatch |
| Deep worker | each worker in the panel | `agentic.react.deep_worker.max_rounds` (default 2) | Gate held *outside* the loop by `_run_one_worker`; ReAct gets `gate=None` |

The reason for the gate difference comes from Lessons 6 and 8: the GPU fits
one local model at a time, and the deep panel holds the gate for the
*whole* worker so the per-round release doesn't let another local worker
slip in mid-loop. Fast-path tool dispatch *does* release the gate between
rounds, because there's only one model running and the dispatch window is
a good moment to let another user's request slip onto the GPU.


## 2. Read-along

These are the files we'll reference in this lesson:

- [`src/audrey/tools/discovery.py:77`](../../src/audrey/tools/discovery.py#L77)
  - turns OpenAPI specs into Ollama tool schemas.
- [`src/audrey/tools/dispatch.py:79`](../../src/audrey/tools/dispatch.py#L79)
  - executes one tool call, returns a `ToolResult`, never raises.
- [`src/audrey/pipeline/react.py:101`](../../src/audrey/pipeline/react.py#L101)
  - the loop: chat, dispatch, repeat.
- [`config.yaml:142`](../../config.yaml#L142) - `agentic.react.*` knobs.
- [`tools-server/app.py`](../../tools-server/app.py) - the FastAPI service
  that exposes the actual tools.

Open them as we go. The useful shape is the sequence of decisions a tool
call makes from registry to result.

### 2.1 Discovery happens once, at startup

Open [`tools/discovery.py:178`](../../src/audrey/tools/discovery.py#L178).
At startup, `lifespan` in `main.py` calls `discover_all(...)` once. That
walks each tool server's `/openapi.json`, picks out the operations tagged
`tools`, and builds a `ToolRegistry`:

```python
async def discover_one(client, server_url, *, timeout_s=10.0):
    r = await client.get(f"{base}/openapi.json")
    ...
    for path, methods in paths.items():
        post = methods.get("post")
        ...
        tool = _build_tool_from_operation(...)
```

Two practical consequences:

- **Discovery does not retry by default.** If `custom-tools` was still
  starting when Audrey came up, the registry is empty and the live tool
  count says `tools=0`. The boot-order retry task added later (see
  [`main.py`](../../src/audrey/main.py)) re-runs discovery every few seconds
  when the registry is empty; `POST /v1/tools/rediscover` is the manual
  escape hatch.
- **Models never see HTTP routes.** They see a JSON object per tool:

```python
{"type": "function",
 "function": {"name": "web_search",
              "description": "...",
              "parameters": {<inlined JSON Schema>}}}
```

The `name` is the FastAPI `operation_id`, the `description` is the route's
summary or docstring, and `parameters` is the request-body schema with all
`$ref`s inlined. That inlining lives in
[`discovery.py:77`](../../src/audrey/tools/discovery.py#L77) (`_resolve_refs`).
Ollama's tool-calling implementation does not follow refs at runtime, so
Audrey resolves them ahead of time.

### 2.2 Schemas get scrubbed before they reach the model

Models trip over JSON Schema keywords that aren't in their training-time
tool format. Open
[`tools/discovery.py:100`](../../src/audrey/tools/discovery.py#L100):

```python
def _strip_unsupported_keywords(schema):
    allowed = {"type", "properties", "required", "enum", "description",
               "items", "default", "minLength", "maxLength",
               "minimum", "maximum", "title"}
    ...
```

Everything else (`format`, `additionalProperties`, `examples`, custom
extensions) gets dropped. The small models route around clean schemas; a
stray `format: "email"` can silently break tool calls on a 4B-class model.

There is one wrinkle worth slowing down for. `properties` and `$defs` are
**name-keyed maps** — the keys are user-chosen field names, not JSON Schema
keywords. So the scrubber recurses into them but does *not* filter their
keys against the allow-list. That's the `name_keyed` branch a few lines
down. Getting that wrong would silently drop every property in your schema.

### 2.3 `ToolSpec` carries enough to dispatch later

A discovered tool is held in this dataclass at
[`tools/discovery.py:40`](../../src/audrey/tools/discovery.py#L40):

```python
@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    server_url: str
    path: str
```

The `server_url` and `path` are what the dispatcher will join into a POST
URL later. The model never sees them. This split keeps the surface the
model gets (name + description + parameters) clean while the routing
metadata stays internal.

`ToolRegistry` is a thin dict around `name -> ToolSpec`. Its only
behavioral helper is `to_ollama_tools()`, which produces the list of
schemas the model gets per request.

### 2.4 The ReAct loop, top-down

Open [`pipeline/react.py:101`](../../src/audrey/pipeline/react.py#L101).
The function signature is long but the body is short:

```python
async def run_react(
    ollama, health, registry,
    *,
    model, messages, options,
    timeout_s, max_rounds,
    compress_after_round, max_tool_result_chars, tool_dispatch_timeout_s,
    ...
) -> ReactResult:
    tools = registry.to_ollama_tools() if registry.by_name else None
    convo = list(messages)
    ...
    async with httpx.AsyncClient() as http:
        for round_idx in range(max_rounds):
            ...
```

The shape:

```text
build tools list (or None)
copy messages into a local `convo`
open one httpx.AsyncClient for the loop
for round_idx in range(max_rounds):
    optional history compression
    ollama.chat(convo, tools=tools)
    if no tool_calls: return content
    append the assistant's tool-call turn to convo
    dispatch all tool_calls in parallel
    append each tool result to convo
on budget exhausted: force a final answer
```

The most important sentence in that flow: **if there are no tool calls, the
loop returns immediately**. The "loop" runs at most `max_rounds` times, but
in practice short prompts often complete in one round (the model answered
straight away) or two rounds (one tool call, then the answer).

### 2.5 What "tool_calls" actually looks like

The protocol-level details of this shape — where it comes from, why it
looks the way it does, and how it differs between OpenAI, Anthropic,
and Ollama — are covered in the next lesson
([`lesson-10-how-function-calling-works.md`](lesson-10-how-function-calling-works.md)).
Here we focus on what Audrey does with it.

When the model decides to call a tool, the chat response looks like:

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {"id": "call_0",
       "function": {"name": "web_search",
                    "arguments": {"query": "btrfs latest version"}}}
    ]
  }
}
```

`content` is often empty when the model is calling a tool. `tool_calls` is
an array — a single turn may emit multiple tool calls (e.g. "search the
web AND check my notes"). Audrey dispatches them in parallel via
`asyncio.gather` so a slow `web_search` doesn't block a fast
`memory_recall`.

A few lines later in `react.py`:

```python
results = await asyncio.gather(*[
    dispatch_one(http, registry, tc,
                 max_result_chars=max_tool_result_chars,
                 timeout_s=tool_dispatch_timeout_s,
                 user_id=user_id)
    for tc in tool_calls
])
```

`dispatch_one` is where the dispatch happens — let's look at it next.

### 2.6 The dispatcher: turn one tool_call into a ToolResult

Open [`tools/dispatch.py:79`](../../src/audrey/tools/dispatch.py#L79). The
function signature is small but the body has several explicit failure
paths:

```python
async def dispatch_one(client, registry, tool_call,
                      *, max_result_chars, timeout_s, user_id=None):
    ...
    return ToolResult(name=..., content=..., is_error=...)
```

There are exactly five things this function does:

1. **Parse the tool_call.** Extract `name` and `arguments`. Ollama
   sometimes returns `arguments` as a JSON-encoded string instead of a
   dict; the dispatcher handles both shapes.
2. **Apply user-scope.** When the tool is in `_USER_SCOPED_TOOLS`,
   overwrite `args["user"]` (or `tags`, for `memory_store`) with the
   real pipeline user. The model's value is ignored.
3. **Look the tool up in the registry.** If it's not there, return an
   error tool result naming the available tools.
4. **POST the arguments** to `spec.server_url + spec.path`, honoring
   `timeout_s`. Both timeouts and other network errors become tool
   results with `is_error=True`.
5. **Truncate the response** at `max_result_chars` and return.

The slogan worth remembering:

> The dispatcher never raises — failures come back as data.

That's the whole point of the `ToolResult` shape. Every failure path
returns a `ToolResult` with `is_error=True` and a JSON-string `content`
explaining what happened. The ReAct loop then includes that as a
`role=tool` message; the model sees the error and decides what to do
(retry with different arguments, apologize, try a different tool).

### 2.7 Concept spotlight — the user-overwrite invariant

This is the most important part of the dispatcher, and it's three lines.
At [`tools/dispatch.py:130`](../../src/audrey/tools/dispatch.py#L130):

```python
if user_id and name in _USER_SCOPED_TOOLS:
    if name == "memory_store":
        args["tags"] = _force_user_tag(str(args.get("tags") or ""), user_id)
    else:
        args["user"] = user_id
```

Why this matters: imagine a model has been told its identity is
`alice@example.com`. It emits a tool call like

```json
{"function": {"name": "memory_search",
              "arguments": {"user": "bob@example.com", "query": "..."}}}
```

Without the overwrite, that call would search Bob's memories from Alice's
session. The model can be tricked (a malicious user prompt, a prompt
injection from a tool result, a confused model) into asking for someone
else's data.

The dispatcher refuses. It replaces `args["user"]` with whatever
`require_user` resolved on the request — the authenticated email. The
model's value is ignored entirely.

For `memory_store` the user identity lives inside the free-form `tags`
string as `user:<id>`. `_force_user_tag` at line 72 strips any existing
`user:` token from the tag string and appends the real one. Same
invariant, different field.

This is the single load-bearing security move in the tool path. Lesson 1's
"trust the boundary, not the data" applies: the model is *data*, its
output is *not authority*. The pipeline user comes from auth (Lesson 4),
and the dispatcher enforces it on every user-scoped call.

### 2.8 Concept spotlight — truncation and what the model sees

Tool results can be large. A `kb_search` query that returns five chunks
of ingested documentation can easily run to 8-15 KB. A `web_search` for a
busy topic can return paragraphs of snippets.

`max_tool_result_chars` (default 2000, from
[`config.yaml:145`](../../config.yaml#L145)) is the single-shot cap. The
dispatcher truncates to that length and appends `…[truncated]` so the
model knows it didn't see everything. The helper lives at
[`tools/dispatch.py:66`](../../src/audrey/tools/dispatch.py#L66):

```python
def _truncate(s, limit):
    if len(s) <= limit:
        return s
    return s[: limit - len("\n…[truncated]")] + "\n…[truncated]"
```

The mental model: every tool message is a budget item. The model only has
a finite context window; multiplying long tool results across multiple
rounds blows that fast. Audrey trades "model sees less of each result"
for "model can keep reasoning across more rounds." When a result is too
big to be useful in 2000 chars, the right answer is usually "narrow your
query," not "raise the cap."

### 2.9 Concept spotlight — concurrent dispatch

Tool dispatch within one ReAct round is parallel. From
[`react.py:171`](../../src/audrey/pipeline/react.py#L171):

```python
results = await asyncio.gather(*[
    dispatch_one(http, registry, tc, ...)
    for tc in tool_calls
])
```

This matters when the model emits multiple tool calls in one assistant
turn. A common pattern: "I need to check the web AND look up my notes."
Sequential dispatch would add latency for no reason — these tools talk to
different services and don't share state. Parallel dispatch keeps the
round latency at `max(t_web, t_memory)` rather than `sum(...)`.

The downside is that all in-flight tools share the `tool_dispatch_timeout_s`
budget independently. A slow tool can't block another tool, but it also
can't share its time budget. That's fine in practice — the timeout is
per-call anyway.

### 2.10 Compression keeps the convo small

After `compress_after_round` rounds (default 2), Audrey replaces older
`role=tool` messages with a one-line stub:

```python
def _summarize_tool_message(msg):
    name = msg.get("name", "?")
    content = msg.get("content", "") or ""
    return f"[earlier tool call: {name} -> {len(content)} chars elided]"
```

So after round 3, the convo holds: original messages, recent assistant
turn with tool_calls, recent tool results verbatim, and one-line stubs
for everything older. The model can still see *that* an earlier call
happened (so it doesn't redo work), but the 8 KB of `kb_search` result
from round 1 is gone.

Compression is a hard call. The model loses access to old evidence; in
return, the prompt stays small enough to fit several more rounds of
work. Audrey errs on the side of compression because the alternative is
a 30 KB context that the model thinks slower against. Real tool work in
Audrey rarely spans more than 2-3 rounds, so this is usually a no-op.

### 2.11 Concept spotlight — forcing the final answer

The trickiest part of ReAct is what happens when the model *keeps* asking
for tools after the budget runs out. Just removing the tools list isn't
strong enough: small models can stall (no bytes for minutes), or invent a
"pseudo tool-call" in plain text ("I would search for ...").

Audrey forces the mode change explicitly. From
[`react.py:195`](../../src/audrey/pipeline/react.py#L195):

```python
log.warning("react: max_rounds=%d reached for %s; forcing final answer without tools", ...)
convo = _compress_history(convo, keep_last_round=1)
final_answer_text = prompt_from_config(cfg, "react_final_answer", REACT_FINAL_ANSWER_USER)
convo.append({"role": "user", "content": final_answer_text})
...
final = await ollama.chat(model=model, messages=convo,
                          options=options or None, tools=None, ...)
```

Three things happen together:

1. Older tool messages collapse to one-line stubs.
2. A new `role=user` turn appends with the override-aware "wrap up now"
   instruction (default text in `prompts.py`, can be tuned through
   `agentic.prompts.react_final_answer`).
3. The follow-up `chat` call passes `tools=None`. The schema is gone; the
   model has nothing to call.

The user turn does the heavy lifting. A `role=system` reminder is too
soft; a `role=user` turn looks to the model like a fresh user request,
and "the user just asked me to wrap up" is a very clear mode signal.

### 2.12 Concept spotlight — health, gates, and errors

Each `ollama.chat` call in ReAct is wrapped in `health.record_success`
and `health.record_failure`. From
[`react.py:144`](../../src/audrey/pipeline/react.py#L144):

```python
try:
    async with _gate_ctx(gate, model, location, user_id):
        last_resp = await ollama.chat(...)
    health.record_success(model)
except OllamaError as e:
    health.record_failure(model, str(e))
    raise
```

The same model is used across all rounds, so the health record per round
is mostly informational — it's used to decide whether *future* requests
can pick the same model. But because each round is an independent chat
call, a flaky cloud model can fail in round 2 even though it succeeded
in round 1, and the health record will reflect that.

`OllamaError` escapes the loop and reaches the graph node. Tool-call
errors do not — they live inside `ToolResult` and the loop continues.
The split is intentional:

- **Tool errors are recoverable.** The model can choose a different tool,
  retry with different args, or apologize. Bubbling them up would skip
  that decision.
- **Chat errors are not recoverable here.** If Ollama is down or the
  model is broken, the loop can't continue. The graph node turns the
  exception into a 502.

### 2.13 Metrics on every dispatch

The Prometheus side is small but useful:

```text
audrey_tool_calls_total{tool, outcome}    # counter
audrey_tool_call_seconds{tool}            # histogram
```

The outcomes are exactly three: `ok`, `error`, `timeout`. The two error
buckets matter operationally: "this tool is slow" (timeouts climbing) and
"this tool is broken" (errors climbing) are different problems with
different fixes.

A real example: if you see `audrey_tool_calls_total{tool="web_search",
outcome="timeout"}` climbing while `outcome="ok"` is flat, the cause is
almost always the Brave API rate-limiting. Other tools' metrics will
look fine. The split lets you point at the cause without inspecting logs.

### 2.14 Walking one request end to end

A request that uses `web_search`:

1. User: *"What is the latest stable BTRFS release as of this week?"*
2. Classify picks `general`. Fast path picks `qwen3.6:35b`. That model is
   in `tool_capable_models`, so the tool-capable branch fires.
3. `run_fast_path` calls `run_react`. The registry has 7 tools including
   `web_search`. ReAct converts each into Ollama's tool schema.
4. Round 0: `ollama.chat(messages, tools=[...])`. The model emits a
   `tool_calls` array:
   `[{"function": {"name": "web_search", "arguments": {"query": "btrfs latest stable release"}}}]`
5. The assistant's tool-call turn is appended to the convo.
6. `dispatch_one` looks up `web_search` in the registry. Spec says
   `server_url=http://custom-tools:8001`, `path=/web_search`. `_USER_SCOPED_TOOLS`
   does not contain `web_search` (web search is anonymous to Brave), so no
   user overwrite. The dispatcher POSTs `{"query": "btrfs latest stable release"}`.
7. Brave returns 5 results. Custom-tools wraps them and returns
   `{"query": "...", "results": [...]}`. The JSON is ~3 KB.
8. The dispatcher truncates to 2000 chars, appends `…[truncated]`,
   wraps as a `role=tool` message, and ReAct appends it.
9. Round 1: `ollama.chat(convo, tools=[...])`. The model has the search
   results now. It returns prose: `"As of today, BTRFS releases are
   tracked alongside the kernel; the current stable kernel ships with..."`
   No `tool_calls`. The loop returns.
10. `run_fast_path` wraps the answer in a response-like dict and
    returns to the graph.
11. The graph emits the answer to the route. The user sees it.

Total: 2 chat calls, 1 dispatch, 1 truncation. The whole tour fits in
about half a second on warm models.

### 2.15 Where to look in the source

Once you have the story above, the source files read in this order:

```text
tools/discovery.py
  - discover_one / discover_all                 (startup)
  - ToolSpec / ToolRegistry                     (data shape)
  - _resolve_refs / _strip_unsupported_keywords (schema scrub)

tools/dispatch.py
  - dispatch_one                                (the workhorse)
  - _USER_SCOPED_TOOLS / _force_user_tag        (the invariant)
  - to_tool_message                             (back into convo)
  - ToolResult                                  (data shape)

pipeline/react.py
  - run_react                                   (the loop)
  - _compress_history                           (older tool messages → stubs)
  - _gate_ctx                                   (fast vs deep scheduling)
  - REACT_FINAL_ANSWER_USER                     (mode-change at budget)
```

The dependency line is one-way: react imports dispatch imports discovery.
There is no back-edge.


## 3. Comprehension questions

**1. The custom-tools service was restarted. The next request fails with
"unknown tool: web_search". What happened, and how do you fix it?**

Discovery runs once at startup. When custom-tools came back, Audrey's
registry still holds the old `ToolSpec` entries, but the live tools-server
process may have changed routes or operation IDs. (And if Audrey itself
restarted first while custom-tools was down, the registry is empty.)

Two fixes: `POST /v1/tools/rediscover` re-hits each tool server's
`/openapi.json` and rebuilds the registry in place, or just restart
`audrey-ai` so `lifespan` runs discovery again. The graph keeps its
closure over the same registry instance, so live mutation works
without a graph rebuild.

**2. A model emits a tool_call like
`{"function": {"name": "memory_search", "arguments": {"user": "evil@example.com", "query": "..."}}}`.
What does the dispatcher do?**

It overwrites `args["user"]` with the authenticated pipeline user before
making the network call. The model's value is ignored entirely. The
allow-list at `_USER_SCOPED_TOOLS` decides which tools get this
treatment; new user-scoped tools added to the tools-server side need to
be added to that set or they bypass the invariant.

**3. A tool returns a 100 KB JSON response. What does the model see?**

The dispatcher truncates to `max_tool_result_chars` (default 2000),
appends `…[truncated]`, and the tool message ends there. The model
knows it didn't see everything because the marker is explicit. The
right response from the model is usually to narrow the next query, not
to ask for more.

**4. The ReAct loop hits `max_rounds`. The model wants another tool call.
What does Audrey do?**

It compresses older `role=tool` messages into one-line stubs, appends a
`role=user` turn carrying the `REACT_FINAL_ANSWER_USER` text ("wrap up
now"), and re-calls the model with `tools=None`. The user turn is the
load-bearing signal — without it, small models can stall or invent a
pseudo tool-call in plain text. The `tools=None` removes any temptation
to comply.

**5. A fast tool-using request and a deep tool-using request differ in one
important scheduling detail. Which one, and why?**

The GPU gate. Fast path passes a real `FairLocalGate` into ReAct, so the
gate is released between rounds during tool dispatch — another user's
request can run on the GPU while a slow web search resolves. Deep workers
hold the gate at the whole-worker level (one acquire across all rounds)
and pass `gate=None` into ReAct, because local workers run serialized
under `as_completed` and can't share the GPU mid-round without thrashing.

**6. The Prometheus counter
`audrey_tool_calls_total{tool="web_search", outcome="timeout"}` keeps
climbing. What's the likely cause?**

The Brave API is rate-limiting or the network path is slow. The `error`
bucket would climb instead if Brave were returning 4xx/5xx, and the
`ok` bucket would also be flat. Timeouts specifically mean the request
exceeded `tool_dispatch_timeout_s` — that's a wall-clock measurement,
not a server response.

**7. Why does `dispatch_one` return errors as `ToolResult` instead of
raising?**

Because the ReAct loop is the right place to decide what to do about a
tool failure. The model has more context: maybe the same query phrased
differently works, maybe it should fall back to a different tool, maybe
it should apologize to the user with a partial answer. Raising would
strip that decision from the loop. The slogan: *the dispatcher never
raises — failures come back as data.*

**8. What happens to the message history between rounds?**

Each round adds two kinds of messages to the convo: the assistant's
tool-call turn (`role=assistant` with the `tool_calls` array), and
one `role=tool` message per dispatched call. After
`compress_after_round` rounds, the loop calls `_compress_history` to
replace older `role=tool` messages with a one-line summary, keeping
only the most recent round verbatim. The assistant turns and the
original user turns stay intact.

**9. Why does `discover_one` skip endpoints not tagged `tools`?**

Tools-server has system endpoints (`/health`, the internal
`/chat_history/archive`) that are not callable as tools. The
tag-based filter is how the FastAPI side declares intent: tagging a
route `tools` says "discover this as a callable tool." Untagged or
differently-tagged routes are part of the service surface but
invisible to the model.

**10. The model is "tool-capable" — what does that actually mean?**

In Ollama's API, certain models accept a `tools=[...]` parameter and
will emit `tool_calls` in the assistant message when they want to call
one. Models that don't support function calling either ignore the
parameter or produce garbled output. Audrey maintains a small
allow-list in `fast_path.tool_capable_models` (config.yaml) because
Ollama's own "has tools" flag is sometimes wrong — the allow-list is
the human-curated answer to "which of our actual models can I trust
with function calling?"


## When you're ready for the next lesson

We have walked classification, routing, model selection, gate scheduling,
and now tool dispatch. This lesson treated the function-calling protocol
as given — Audrey builds a `tools` array, hands it to a model, receives
`tool_calls`, runs them, loops. The next lesson opens that black box:
where the JSON shape comes from, how the model is taught to emit it, and
what changes between OpenAI, Anthropic, and Ollama dialects. After that,
we open the knowledge base — how documents and images get ingested,
embedded, indexed in Qdrant, and served back through the `kb_search` and
`kb_image_search` tools whose dispatch you just learned to follow.
