# Lesson 10 — Tool use and the ReAct loop

**Estimated time:** 45-60 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"when a model needs to
call a tool, how does Audrey find the right server, send the call, return the
result, and decide when to stop, and how does it make sure a tool only ever
touches the caller's own data?"*

Lesson 8 traced how a request becomes a `task_type` and a `mode`, and Lesson 9
walked the deep and research pipelines that run when `mode = deep`. This lesson picks up one
step lower: after the fast path or a deep worker has chosen a model. If that
model can call tools and tools are available, Audrey hands the request to the
**ReAct loop**: a small driver that lets the model call tools, feeds the
results back, and forces a final answer when the round limit runs out.

Five ideas to keep separate:

```text
declare   - what each tool may touch, and what it needs to work
discover  - which declared tools the tool servers offer right now
dispatch  - run one tool call against the right server
loop      - chat -> dispatch -> chat -> ... -> answer
stop      - at the round limit, force an answer in prose
```

`web_search` is the worked example throughout: its round trip is short and its
schema is easy to read.


## 1. Context

### 1.1 Why tools, and the trade

Audrey wants answers grounded in current information. "What's the latest
version of BTRFS?" shouldn't be answered from whatever a model remembers from
its training. A **tool** is how a model reaches outside its prompt: search the
web, search the knowledge base, look up a saved memory, read one of your files.

The trade:

```text
without tools  - one chat call, one answer, fast, possibly stale
with tools     - several chat calls, a better-grounded answer, more latency
```

Audrey leans toward "use a tool when it helps, then stop." The model decides
each turn whether to call a tool, and the loop has a fixed round limit. When
the model replies in prose with no tool calls, the loop ends.

Not every model can take part. A model that supports function calling accepts
a `tools=[...]` list and replies with `tool_calls` when it wants one; other
models ignore the list or produce garbled output. Ollama's own capability flag
isn't always right, so Audrey keeps a hand-maintained list,
[`tool_capable_models`](../../config.yaml#L262), and runs the loop only for
models on it.

### 1.2 Why "ReAct"?

ReAct is short for "reason and act": a prompt pattern where the model
alternates between reasoning about what to do next and acting (calling a tool,
then reading the result). Function-calling models build the same alternation
into their protocol: Each round, the model either emits a `tool_calls` array
(act) or returns prose (answer). Audrey's loop is a plain implementation of
that pattern. The care goes into the budgets and the failure handling, not the
loop's shape.

### 1.3 Where the loop runs

The same loop serves three callers, each with its own settings:

| Caller | Settings | Rounds and searches |
|---|---|---|
| The fast path, one model answering | [`agentic.react`](../../config.yaml#L685) | the defaults |
| Each deep worker | [`deep_worker`](../../config.yaml#L773) | room for a few more tool messages |
| Each research worker | [`research_worker`](../../config.yaml#L798) | more rounds and a larger search budget |

In all three, the loop takes the GPU gate from Lesson 6 only for each model
call, and gives it back while tools run. A slow web search doesn't hold the
GPU, and a tool that needs the GPU itself, such as one that looks at a video
frame, can take its turn instead of waiting on a lock the loop is holding. A
deep worker that doesn't use tools holds the gate for its one generation.


## 2. Read-along

These are the files this lesson reads:

- [`tools/discovery.py`](../../src/audrey/tools/discovery.py): the declaration
  catalogue, discovery, and the registry of tools.
- [`tools/dispatch.py`](../../src/audrey/tools/dispatch.py): one tool call
  in, one result out, never an exception.
- [`pipeline/react.py`](../../src/audrey/pipeline/react.py): the loop.
- [`tools-server/app.py`](../../tools-server/app.py): the sidecar service that
  runs most of the tools.

### 2.1 The declaration catalogue

Before Audrey offers a model any tool, the tool must be declared in
[`TOOL_DECLARATIONS`](../../src/audrey/tools/discovery.py#L100). A declaration
answers the questions that matter for safety, which are none of the model's
business:

- **Whose data does it touch?** `user_scope` is `NONE` for a public tool like
  `web_search`, `ARGUMENT` when the request carries a `user` field (searching
  your files), or `TAGS` when the identity rides in a `user:` tag (saving a
  memory).
- **What does it need to work?** `dependencies` names the components behind
  it, such as Qdrant or the text embedder. When the tool server reports one of
  them down, the tool is hidden from models until it recovers.
- **Does a purge block it?** `purge_gated` marks tools that read stored
  memories or chat history. While a person's data purge is being installed,
  those reads are refused (§2.7).
- **Does it need longer than usual?** `timeout_s` overrides the dispatcher's
  default, for the two tools that inspect a document page or a video frame.

A few entries, from
[`discovery.py:100`](../../src/audrey/tools/discovery.py#L100):

```python
TOOL_DECLARATIONS: dict[str, ToolDeclaration] = {
    declaration.name: declaration
    for declaration in (
        ...
        _declare("web_search", dependencies=("web_search",)),
        ...
        _declare(
            "kb_search",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("audrey_kb", "qdrant", "text_embedding"),
        ),
        ...
        _declare(
            "memory_store",
            user_scope=ToolUserScope.TAGS,
            dependencies=("memory", "qdrant", "text_embedding"),
        ),
        ...
    )
}
```

> **Concept spotlight — a route isn't a tool until it's declared.** The tool
> server decides which routes exist; Audrey decides which of them a model may
> call, and on whose behalf. Tagging a route `tools` on the server isn't
> enough. If the server offers a `tools` route with no declaration, or one
> whose request fields don't match its declaration (a `user` field the
> declaration doesn't bind, say), Audrey rejects **every** tool from that
> server and logs an error. Failing closed is deliberate: A server whose
> contract has drifted from Audrey's catalogue isn't trusted with any of its
> tools. The rule that follows: A new tool's declaration ships with its route
> or before it, never after.

### 2.2 Discovery

Discovery runs during startup.
[`discover_all`](../../src/audrey/tools/discovery.py#L491) asks every
configured tool server for its `/openapi.json` at the same time, each request
with its own timeout, so one slow or broken server can't hold up or spoil the
others. For each server,
[`discover_one`](../../src/audrey/tools/discovery.py#L409) keeps only POST
routes tagged `tools` and checks each against its declaration, at
[`discovery.py:456`](../../src/audrey/tools/discovery.py#L456):

```python
if "tools" not in tags:
    # Only an explicit tools tag grants model visibility. Untagged
    # POSTs and internal/system operations are not capabilities.
    continue
op_id = post.get("operationId")
...
declaration = TOOL_DECLARATIONS.get(op_id)
if declaration is None:
    raise ToolPolicyError(
        f"{op_id}: model-visible tool has no policy declaration"
    )
```

That `raise` is what rejects the whole server: The error is caught once,
around the server's entire catalogue. Routes without the tag, like `/health` or
the sidecar's internal archive route, are part of the service but never
offered to a model.

Each route that passes is converted for the model in two steps:

- **References are inlined.** A route's request schema can point at shared
  definitions (`$ref`). Ollama doesn't follow those, so
  [`_resolve_refs`](../../src/audrey/tools/discovery.py#L254) copies each one
  into place.
- **Unfamiliar keywords are scrubbed.**
  [`_strip_unsupported_keywords`](../../src/audrey/tools/discovery.py#L277)
  keeps a short list of JSON Schema keywords (`type`, `properties`,
  `required`, `enum`, `description` and a few more) and drops the rest,
  because small models can fail silently on keywords like `format`. One
  subtlety: `properties` maps field names, chosen by whoever wrote the tool,
  to their schemas. The scrubber walks into it without filtering the names;
  filtering them against the keyword list would delete every field.

The result is a [`ToolSpec`](../../src/audrey/tools/discovery.py#L179): the
name, description and parameters the model will see, plus what only Audrey
sees, which is the server URL and path to POST to and the policy copied from
the declaration. The tool server also reports which of its components are
working. A tool whose dependency is down stays in the registry, marked
unavailable.

If discovery comes back empty at startup, because custom-tools is still
starting, a background task retries for a while, and an admin can ask again
with `POST /v1/tools/rediscover`, which refuses an empty answer rather than
wiping the live tools. Lesson 5 walked both.

### 2.3 What the model is offered

The registry hands the loop only what a model may use right now, at
[`discovery.py:216`](../../src/audrey/tools/discovery.py#L216):

```python
def specs(self) -> list[ToolSpec]:
    return [
        spec
        for spec in self.by_name.values()
        if spec.visibility is ToolVisibility.MODEL and spec.available
    ]
```

Each of those becomes one entry in the request's `tools` list. The model never
sees a URL or a policy, only this:

```text
{"type": "function",
 "function": {"name": "web_search",
              "description": "...",
              "parameters": {<the inlined JSON Schema>}}}
```

A skill can narrow the list further. When a run uses a skill, the route hands
the loop a registry restricted to the tools that skill allows, through
[`restrict`](../../src/audrey/tools/discovery.py#L240).

### 2.4 The loop, top-down

[`run_react`](../../src/audrey/pipeline/react.py#L558) has a long signature
and a short body. Its shape:

```text
offer the registry's tools (or none)
copy the messages into a local `convo`
for each round, up to max_rounds:
    after the first few rounds, compact older tool results
    chat(convo, tools)              the GPU gate is held for this call only
    no tool_calls? return the answer
    append the assistant's tool-call turn
    answer over-budget web searches with a note
    dispatch the rest in parallel, append each result
round limit reached: force a final answer without tools
```

The most important line in that flow: **if the model makes no tool calls, the
loop returns at once.** A simple question takes one round; a question that
needs one search takes two.

### 2.5 What `tool_calls` looks like

The next lesson covers the protocol behind this shape and how it differs
between providers. Here is what Audrey receives when the model wants a tool:

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

`content` is often empty on a tool-calling turn, and `tool_calls` is a list:
One turn can ask for several tools at once ("search the web and check my
files"). The loop runs them in parallel (§2.9).

### 2.6 The dispatcher: one call in, one result out

[`dispatch_one`](../../src/audrey/tools/dispatch.py#L170) turns one entry of
`tool_calls` into a `ToolResult`. In order, it:

1. **Parses the arguments.** Some models send them as a JSON string instead of
   an object; both work. Arguments that won't parse come back as an error
   result.
2. **Looks the tool up.** An unknown or unavailable tool comes back as an
   error result that lists the tools that are available.
3. **Applies the identity rule** (§2.7): a tool that touches personal data
   gets the caller's identity, never the model's.
4. **Checks the purge block** (§2.7).
5. **POSTs the arguments** to the tool's server, with the tool's own timeout
   when its declaration sets one. Timeouts, network errors and HTTP errors all
   come back as error results.
6. **Shapes the result** to fit the size cap (§2.8).

> **The dispatcher never raises — failures come back as data.**

Every failure returns a `ToolResult` with `is_error=True` and a JSON body
saying what went wrong. The loop passes it to the model as an ordinary tool
message, and the model decides what to do: try other arguments, try another
tool, or answer with what it has. A broken tool costs one round, not the whole
request.

### 2.7 Spotlight — the caller's identity, not the model's

This is the security center of the tool path, at
[`dispatch.py:263`](../../src/audrey/tools/dispatch.py#L263):

```python
if (spec.user_scoped or spec.purge_gated) and (
    not isinstance(user_id, str) or not user_id.strip()
):
    ...  # refused with "authenticated_user_required"

# Bind the authenticated identity exactly as the declared policy says.
# The model never chooses another user for a scoped capability.
if user_id and spec.user_scope is ToolUserScope.TAGS:
    args["tags"] = _force_user_tag(str(args.get("tags") or ""), user_id)
elif user_id and spec.user_scope is ToolUserScope.ARGUMENT:
    args["user"] = user_id
```

Why it matters: In a session signed in as `alice@example.com`, a model emits

```json
{"function": {"name": "memory_search",
              "arguments": {"user": "<another person's id>", "query": "..."}}}
```

A model can be steered into this by a hostile prompt, by instructions hidden in
a web page a tool returned, or by plain confusion. The dispatcher doesn't argue
with it. For a tool declared `ARGUMENT`, it overwrites `user` with the caller's
identity; for `TAGS`,
[`_force_user_tag`](../../src/audrey/tools/dispatch.py#L163) removes any
`user:` tag and appends the real one. The identity comes from authentication
(Lesson 4): the caller's private storage id, which a later lesson on identity
explains in full.

The first check covers the opposite mistake. If a call reaches the dispatcher
with no caller identity at all, a personal tool is refused before any request
is made. A code path that forgets to pass the identity fails closed, instead of
quietly letting the model's argument decide whose data is read.

**The purge block.** A person can ask Audrey to purge their stored memories and
chat history. Between the moment Audrey records that request and the moment
the tool server confirms it has applied the purge, any purge-gated tool for
that person is refused with `personal_data_purge_in_progress`. The memory
recall that runs before each answer goes through the same dispatcher, with the
same identity, so it is blocked too.

### 2.8 Spotlight — truncation that says what was cut

Tool results can be large: a knowledge-base search returning several long
passages, or a page of search snippets. Every result is capped at
[`max_tool_result_chars`](../../config.yaml#L723) characters, because every
tool message spends the model's context, and long results repeated across
rounds crowd out everything else.

How the cut is made matters as much as where.
[`_truncate_payload`](../../src/audrey/tools/dispatch.py#L118) first drops
whole items from the result's largest list, so the model still receives valid
JSON, plus a note, at
[`dispatch.py:148`](../../src/audrey/tools/dispatch.py#L148):

```python
for keep in range(len(items) - 1, 0, -1):
    trial = dict(payload)
    trial[key] = items[:keep]
    trial["_truncated"] = (
        f"showing {keep} of {len(items)} {key} — {len(items) - keep} omitted "
        f"to fit a {limit:,}-char cap. {_RETRY_IS_POINTLESS}"
    )
```

"Showing 2 of 12 results" is something the model can repeat truthfully. When
there's no list to shorten, or one item alone is too big, the dispatcher cuts
characters instead, and its marker still gives the characters shown out of the
total. Both forms end with the same sentence: The cap applies to this result,
not to the query, so asking again for more returns the same amount of text or
less. Without that sentence, a model that sees a cut tends to invent a remedy,
such as re-running the search for more results, which spends a round and gets
nothing new. The principle: **Tell the model how much is missing, and that
asking again won't fix it.**

### 2.9 Spotlight — parallel calls and the search budget

A round's tool calls run in parallel: The loop gathers them with
`asyncio.gather`, so a round takes as long as its slowest call, not the sum of
all of them. Before that, each worker's web searches are counted against a
budget, [`max_web_searches`](../../config.yaml#L769), because searches spend a
search provider's quota and a research request runs several workers at once.
At [`react.py:749`](../../src/audrey/pipeline/react.py#L749):

```python
for tc in tool_calls:
    name = _tool_name(tc)
    if name == "web_search" and max_web_searches > 0:
        if web_searches_used >= max_web_searches:
            stubs.append(ToolResult(
                name=name, call_id=tc.get("id"),
                content=_WEB_SEARCH_BUDGET_STUB,
                elapsed_s=0.0, is_error=False,
            ))
            continue
        web_searches_used += 1
    to_dispatch.append(tc)
```

A call over budget isn't sent. It still gets an answering tool message, because
every tool call needs one: a note saying the search limit is reached and the
model should answer from the evidence it has. The note is worded so it can't be
mistaken for a failed search. Once the budget is spent, `web_search` is also
removed from the tools offered in later rounds, so the model turns to writing.
Other tools aren't budgeted.

Each dispatched call also passes through `_dispatch_observed`, which reports it
to the run's event stream as it starts and finishes. A later lesson follows
those events to the browser.

### 2.10 Compaction keeps the conversation small

After [`compress_after_round`](../../config.yaml#L703) rounds, and once more
before a forced final answer,
[`_compress_history`](../../src/audrey/pipeline/react.py#L214) replaces older
tool results with a short placeholder. It counts tool **messages**, not rounds,
because one round can produce several. The setting
[`compress_keep_last`](../../config.yaml#L762) says how many keep their full
text, and [`_tool_message_priority`](../../src/audrey/pipeline/react.py#L204)
decides which:

```python
def _tool_message_priority(msg: dict[str, Any]) -> int:
    """Keep usable results before notices/failures, and prior stubs last."""
    content = msg.get("content")
    if isinstance(content, str) and content.startswith(_COMPACTED_MARKER):
        return 2
    if _is_error_tool_message(msg) or content == _WEB_SEARCH_BUDGET_STUB:
        return 1
    return 0
```

Real results come first, then failures and budget notes, and placeholders from
an earlier pass last. Within each group, the newest wins. The newest messages
aren't always the most valuable ones: two failed page fetches at the end of a
round shouldn't push out the search results a worker needs to write its
answer.

A placeholder keeps the tool role and its place in the conversation. Its
wording, from
[`_summarize_tool_message`](../../src/audrey/pipeline/react.py#L156), is chosen
so a model quoting it can't present it as a failed or empty search:

```text
[history compacted: an earlier `web_search` result is omitted here to save context]
```

### 2.11 Spotlight — forcing the final answer

When the round limit runs out and the model still wants tools, removing the
tool list isn't enough on its own: A small model can stall, or write a pretend
tool call as plain text. So the loop changes mode explicitly, at
[`react.py:802`](../../src/audrey/pipeline/react.py#L802):

```python
convo = _compress_history(convo, keep_last_round=compress_keep_last)
...
final_answer_text = prompt_from_config(cfg, "react_final_answer", REACT_FINAL_ANSWER_USER)
convo.append({"role": "user", "content": final_answer_text})
try:
    async with _gate_ctx(gate, model, location, user_id):
        final = await ollama.chat(
            model=model, messages=convo, options=options or None,
            tools=None, timeout_s=timeout_s, think=think,
        )
```

Three things happen together. Older results are compacted once more. A new
user turn tells the model to write its final answer now; the default text is in
`prompts.py`, and `agentic.prompts.react_final_answer` in config overrides it.
And the call goes out with `tools=None`. The user turn does most of the work: a
system-level reminder is easy for a model to ignore, while a fresh request from
the user to wrap up is a clear signal.

### 2.12 Health, gates and errors

Each model call in the loop records the model's health and holds the GPU gate
for that call only, at
[`react.py:650`](../../src/audrey/pipeline/react.py#L650):

```python
try:
    async with _gate_ctx(gate, model, location, user_id):
        last_resp = await ollama.chat(
            model=model, messages=convo, options=options or None,
            tools=tools, timeout_s=timeout_s, think=think,
        )
    health.record_success(model)
except OllamaError as e:
    health.record_failure(model, str(e))
    raise
```

The two kinds of failure are handled differently on purpose. A **tool** failure
is data: The model sees it and can recover. A **model** failure can't be
recovered inside the loop, because Ollama is down or the model is broken, so
`OllamaError` propagates to the caller, which reports it the way Lesson 4
described.

Every dispatch is also counted, in `audrey_tool_calls_total{tool, outcome}`
with an outcome of `ok`, `error` or `timeout`, and timed, in
`audrey_tool_call_seconds{tool}`. Climbing timeouts and climbing errors point
at different problems: a slow tool, or a broken one.

### 2.13 One request, end to end

1. You ask: *"What is the latest stable BTRFS release?"*
2. Classification says `general`, and the fast path picks the best healthy
   `general` model. It is on the tool-capable list, so the fast path calls
   `run_react`.
3. The registry offers its available tools, `web_search` among them.
4. Round 0: the model replies with one `tool_calls` entry, `web_search` with
   `{"query": "btrfs latest stable release"}`.
5. The loop appends that assistant turn, counts the search against the budget,
   and dispatches it.
6. `dispatch_one` finds `web_search` in the registry. Its declaration has no
   user scope, so no identity is bound and no purge block applies. It POSTs
   the arguments to the tool server's `/web_search` route.
7. The tool server returns a JSON body of results, a few kilobytes and well
   under the cap, so it goes into the conversation whole, as a tool message.
8. Round 1: the model has the results in front of it and replies in prose,
   with no tool calls. The loop returns the answer.

Two model calls, one tool call, nothing cut.

### 2.14 Where to look in the source

```text
tools/discovery.py
  TOOL_DECLARATIONS / ToolDeclaration           what each tool may touch
  discover_all / discover_one                   startup; fail closed per server
  _resolve_refs / _strip_unsupported_keywords   schema clean-up
  ToolSpec / ToolRegistry                       routing, policy, availability

tools/dispatch.py
  dispatch_one                                  the workhorse
  _force_user_tag                               identity inside tags
  _truncate_payload / _truncate                 cuts that say what was cut
  to_tool_message                               back into the conversation

pipeline/react.py
  run_react                                     the loop
  _compress_history / _tool_message_priority    what keeps its full text
  _gate_ctx                                     the GPU gate per model call
```

The imports run one way: `react` uses `dispatch`, which uses `discovery`.


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. A new tool route is deployed to custom-tools before Audrey's matching
declaration, and Audrey then restarts. What do people notice, and what is the
fix?**

Every tool from custom-tools disappears, not just the new one. Discovery finds
a `tools` route with no declaration and rejects that server's whole catalogue,
logging an error that names the route. Models get no web search, memory or file
tools. Because discovery came back empty, the background retry tries again, and
gets the same refusal each time. The fix is to deploy Audrey with the
declaration; its startup discovery then loads every tool again. The lasting
rule: A tool's declaration ships with its route or before it, never after.

**2. In a session signed in as `alice@example.com`, a model calls
`memory_search` with another person's id in `user`. Separately, a code path
calls `dispatch_one` for `memory_search` with no `user_id` at all. What happens
in each case?**

In the first case, the declaration says `memory_search` binds a `user`
argument, so the dispatcher overwrites it with the caller's identity before
sending. The model's value never reaches the tool server, and the search
returns the caller's own memories, or a refusal while the caller's purge is
being installed. In the second case, the tool touches personal data and there
is no caller identity to bind, so the dispatcher refuses it with
`authenticated_user_required` before any request is made. Neither path lets the
model choose whose data is read.

**3. A tool returns a 100 KB result. What does the model see?**

Never the whole thing. If the body is JSON with a list in it, the dispatcher
drops whole items from the largest list until it fits under
`max_tool_result_chars`, and adds a `_truncated` note: how many items are
shown, how many were left out, and that asking again won't return more. If
there's no list to shorten, it cuts characters and appends a marker giving the
characters shown out of the total, with the same advice. Either way the model
knows how much is missing. The useful response is a narrower query, not the
same one again.

**4. A worker's search budget is three. It asks for three searches in its first
round and two more in the next. What reaches the model, and what survives
compaction?**

The first three are dispatched, and their results are appended. The next two
aren't sent: each gets a note saying the search limit is reached, and
`web_search` is no longer offered after that. When compaction runs, it keeps
real results ahead of notes and failures, so the three search results keep
their full text, and if anything has to become a placeholder, the notes go
first. The notes answer the model's calls without pushing out its evidence.

**5. The loop reaches its round limit and the model still wants a tool. What
does Audrey do, and why isn't removing the tools enough?**

It compacts older results once more, appends a user turn telling the model to
write its final answer now, and calls the model with `tools=None`. Taking the
tools away alone gives a model that has been calling tools for several rounds
no clear signal to stop: A small model can stall or write a pretend tool call
as plain text. The user turn is the signal; `tools=None` removes the option.


## When you're ready for the next lesson

This lesson treated the function-calling protocol as given: Audrey builds a
`tools` list, hands it to a model, receives `tool_calls`, runs them and loops.
The next lesson opens that box: where the JSON shape comes from, how a model is
taught to emit it, and how it differs between providers. After that, the
course turns to the knowledge base: how documents and images are ingested,
embedded and searched, and served back through the `kb_search` and
`kb_image_search` tools whose dispatch you just followed.
