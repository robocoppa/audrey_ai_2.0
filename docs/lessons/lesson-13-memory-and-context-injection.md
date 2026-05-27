# Lesson 13 — Memory recall and context injection

**Estimated time:** 45-60 minutes if you keep
[`pipeline/memory.py`](../../src/audrey/pipeline/memory.py),
[`pipeline/context.py`](../../src/audrey/pipeline/context.py),
[`pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py), and
[`pipeline/graph.py`](../../src/audrey/pipeline/graph.py) open.

**Goal:** by the end of this lesson, you can answer
*"before classification sees a single word of the user's prompt, the
graph has already prepended two system messages — what are they, why,
and where do they come from?"*

Earlier lessons traced the request from FastAPI route to model call.
Throughout that walk, the graph's first two nodes — `datetime` and
`memory_recall` — kept showing up as the front of the pipeline without
ever getting a full read. This lesson opens them.

The two nodes do related but separate jobs:

```text
datetime       - "today is ..." system message, prepended every request
memory_recall  - durable per-user facts, recalled and prepended when present
```

Both run before classify. Neither calls a model. Their whole job is to
give every downstream node — classify, fast path, deep workers,
synthesizer — the same context to reason from.


## 1. Context

### 1.1 What problem does context injection solve?

A model's reply depends on its prompt. If you ask "what's the latest
version of BTRFS?" with nothing else in context, the answer comes from
training cutoff — useful, but stale. If you ask "remind me what city
I'm in?" with no prior conversation in the prompt, the model has no
way to answer; it didn't see your last visit.

Audrey solves this not by training new models, but by **prepending
small system messages before the user's turn**. The model never knows
which messages came from the user and which Audrey wrote — that's
fine, system messages are exactly the right tool for "context the user
didn't think to provide."

Two pieces of context matter for almost every request:

1. **Time.** Without it, the model hedges about "today" and "recent"
   against whatever its training data implied. With a real ISO
   timestamp in the prompt, the model just uses it.
2. **Durable user facts.** Things the user told Audrey in some past
   conversation — preferences, goals, projects, constraints. The
   model needs them when relevant; the user shouldn't have to retype
   them.

These two are different enough to warrant separate modules:

- `pipeline/context.py` is the trivial one — two functions, no I/O,
  no failure modes. It generates the datetime system message.
- `pipeline/memory.py` is the substantial one — talks to the
  custom-tools server over HTTP, has skip paths for every failure
  mode, formats hits into a human-readable system message, and
  conditionally injects a "how to write back" hint for tool-capable
  models.

### 1.2 Where the two nodes sit in the graph

Open [`graph.py:410-413`](../../src/audrey/pipeline/graph.py#L410).
The graph adds nodes in one block:

```python
g.add_node("datetime", node_datetime)
g.add_node("memory_recall", node_memory_recall)
g.add_node("classify", node_classify)
g.add_node("complexity", node_complexity)
```

Then the edges fix the order:

```python
g.set_entry_point("datetime")
g.add_edge("datetime", "memory_recall")
g.add_edge("memory_recall", "classify")
g.add_edge("classify", "complexity")
```

`datetime` is the entry point — literally the first thing that runs
for every request. `memory_recall` is next, then classify and the
complexity gate (covered in the classification-and-routing lesson).
By the time the classifier reads `state["messages"]`, the front of
that list already has the date and (maybe) some recalled memories.

There's a second entry point hidden in this picture, and we'll come
back to it: the streaming deep route bypasses the graph and calls the
same building blocks directly. For now, hold the graph in your head.


## 2. Read-along: datetime injection

### 2.1 `iso_now` and `datetime_system_message`

Open [`pipeline/context.py`](../../src/audrey/pipeline/context.py).
The whole module is two functions and a module docstring. That's all.

[`iso_now`](../../src/audrey/pipeline/context.py#L37) returns the
current time as a timezone-aware ISO-8601 string:

```python
def iso_now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")
```

Three details worth pointing out:

- `datetime.now()` returns *naive* local time (no tzinfo attached).
  Calling `.astimezone()` with no argument attaches the system's
  local timezone — whatever `TZ=` resolves to inside the process,
  falling back to the host's tzdata.
- `timespec="seconds"` truncates sub-second precision. Without it
  you get microseconds — wasted tokens, wasted log space, no
  practical benefit.
- The output looks like `2026-05-27T14:32:00-07:00`. Models parse
  ISO-8601 fluently; "the afternoon of Tuesday, May twenty-seventh"
  is not an improvement.

[`datetime_system_message`](../../src/audrey/pipeline/context.py#L48)
wraps that string in the OpenAI message shape:

```python
def datetime_system_message() -> dict[str, Any]:
    return {
        "role": "system",
        "content": (
            f"Current server date and time: {iso_now()}. "
            "Treat this as the present moment when reasoning about "
            "dates, recency, or time-sensitive facts."
        ),
    }
```

The phrasing matters. Without "treat this as the present moment,"
some models try to *reason about* the timestamp as data ("interesting,
that's about a year past my training cutoff…"). The instruction
defuses that — the model is told what the timestamp *means*, so it
applies it instead of analyzing it.

### 2.2 Where the datetime node runs

In the graph, the node is a thin wrapper at
[`graph.py:139`](../../src/audrey/pipeline/graph.py#L139):

```python
async def node_datetime(state: PipelineState) -> dict[str, Any]:
    sys_msg = datetime_system_message()
    return {"messages": [sys_msg, *state["messages"]]}
```

The `return {"messages": [...]}` shape comes from LangGraph: the
returned dict gets merged into the pipeline state. We're saying
"replace `messages` with this new list — datetime first, then
everything that was already there." Net result: every downstream node
sees the timestamp at the top of `state["messages"]`.

No network call. No try/except. No skip path. Datetime injection is
cheap enough — one `datetime.now()`, one f-string, one list prepend —
that it just always runs.

### 2.3 The streaming bypass

There's a parallel call site at
[`routes/openai.py:946`](../../src/audrey/routes/openai.py#L946),
inside `_phase_thinking`:

```python
# Mirrors what node_datetime does for the non-streaming graph.
msgs = [datetime_system_message(), *messages]
```

Why the duplication? The streaming deep path doesn't run the graph
node-by-node — it orchestrates synthesis and SSE framing directly,
because it needs to emit progress banners as work happens (covered in
the lesson on classification and routing, §2.9). To keep context
identical between streaming and non-streaming requests, the streaming
route imports the same building block and calls it directly.

This is a load-bearing fact: **two entry points, one building block.**
A future change that adds a third kind of context injection has to
touch both call sites or one path silently drops it. (Audit note
covers this — it's a known shape, not a forgotten one.)


## 3. Read-along: memory recall

Memory recall is where this lesson earns its time. The module has six
ideas to keep straight:

```text
1. when to skip the search entirely  (recall_for_request guards)
2. how the query gets to custom-tools (dispatch_one reuse)
3. how the result becomes a system message (memory_system_message)
4. when to add the "how to write back" hint (memory_store)
5. how the composer pins ordering (compose_system_messages)
6. how the streaming route stays in step (parallel call site)
```

### 3.1 The skip list

Open [`pipeline/memory.py:59`](../../src/audrey/pipeline/memory.py#L59).
`recall_for_request` opens with four guards before any network call:

```python
if not user_id:
    return []
if registry is None or MEMORY_SEARCH_TOOL not in registry.by_name:
    return []
query = last_user_text(messages).strip()
if not query:
    return []
if len(query) > MAX_QUERY_CHARS:
    query = query[:MAX_QUERY_CHARS]
```

Each of these encodes a real failure mode:

- **`not user_id`** — anonymous request. The pipeline runs even
  without authentication for direct-curl callers. Without a user
  identity, there's nothing to recall, so don't ask. (Memory is
  per-user; recalling somebody else's data would be a security bug.)
- **`registry is None or MEMORY_SEARCH_TOOL not in registry.by_name`**
  — custom-tools is offline or hasn't published `memory_search` yet.
  The tool discovery lesson covered how the registry gets populated;
  the relevant guarantee here is "if the tool isn't there, dispatch
  would fail anyway." Skip cheaply.
- **`not query`** — the last user turn is empty. Happens with the
  OWUI utility-task prefix-only messages, with malformed payloads,
  and with edge cases like a `role=user` turn that's all whitespace.
- **`len(query) > MAX_QUERY_CHARS`** — long prompts dilute the
  embedding's signal. A 50-paragraph essay embeds to roughly the
  centroid of *everything*; a 500-character query embeds to
  something more recognisable. The clamp is at
  [`memory.py:34`](../../src/audrey/pipeline/memory.py#L34) — the
  comment used to mention SQL LIKE, but the backend is semantic
  search now (more on that in a moment).

All four guards return without raising. **A best-effort feature
should not break the pipeline.** If recall can't run, the rest of
the request runs without recall — fine.

### 3.2 The dispatcher-reuse trick

The recall path could open `httpx.AsyncClient()` and POST to
`/memory_search` directly. It doesn't. Instead, it builds an Ollama
tool-call shape and reuses `dispatch_one` from the ReAct lesson:

```python
call = {
    "function": {
        "name": MEMORY_SEARCH_TOOL,
        "arguments": {"user": user_id, "query": query, "top_k": top_k},
    }
}
async with httpx.AsyncClient() as http:
    result = await dispatch_one(
        http, registry, call,
        max_result_chars=10_000,
        timeout_s=timeout_s,
    )
```

Why route through the dispatcher when the recall code already knows
exactly which endpoint it wants?

**Errors come back as data, not exceptions.** The dispatcher catches
network errors, timeouts, 4xx, 5xx, and unknown-tool failures, and
returns each as a `ToolResult` with `is_error=True`. The recall
function just checks `result.is_error` and degrades to `[]` — no
try/except wrapper, no `httpx.RequestError` handling, no
`asyncio.TimeoutError` handling. The dispatcher already did that work
for the ReAct loop; reusing it here means there's one piece of code
that knows how to turn "tool call went wrong" into "tool message we
can show the model" or, in this case, "skip the recall."

**`max_result_chars=10_000`** is higher than the ReAct default (2000)
because we want the full JSON body — the dispatcher's truncation is
designed for model context, not for our own parsing.

### 3.3 Parsing the result

If the dispatcher returns success, we still need to handle the body.
The tools-server endpoint returns JSON with a `results` array. Two
things can still go wrong:

```python
try:
    body = json.loads(result.content)
except json.JSONDecodeError:
    log.warning("memory: search returned non-JSON body")
    return []
hits = body.get("results") or []
if not isinstance(hits, list):
    return []
return hits
```

- A 200 response with a non-JSON body (HTML error page, partial
  text) — happens with proxies, misconfigured upstreams. Treat as
  zero hits.
- A 200 response with valid JSON but a malformed `results` field
  (string instead of list, missing key). Also zero hits.

Same posture: don't crash the pipeline. Log enough that an operator
can debug. Move on.

### 3.4 Why the backend changed under the comment

Worth flagging because it shows up in two places. The recall path
*used* to talk to SQLite directly with `LIKE` queries — that's why
there's a 5-second timeout (cheap to query a local file). At some
point the backend moved to the custom-tools `memory_search` endpoint,
which uses Qdrant + embeddings under the hood (covered in the lesson
on KB ingest and search — same machinery, different collection).

The 5-second timeout stayed because the new backend is also fast.
The 500-character query clamp got a new rationale (embedding signal
dilution, not LIKE performance). The
[`config.yaml`](../../config.yaml#L178) comment lagged for a while
and still said "SQL LIKE"; it now reflects the semantic-search
reality.

The lesson here: **comments age faster than code.** When the backend
moves, the timeout might still be correct but the *reason* changes.
A comment that explained the old reason becomes a quiet lie.


## 4. The system message body

Once `recall_for_request` returns hits, somebody has to turn them into
a system message. That's
[`memory_system_message`](../../src/audrey/pipeline/memory.py#L105):

```python
def memory_system_message(
    hits: list[dict[str, Any]],
    *,
    user_id: str = "",
    include_store_hint: bool = False,
    cfg: Any = None,
) -> dict[str, Any] | None:
```

It returns `None` when there's nothing to inject — no hits and no
store hint. Otherwise, it composes one of three shapes: hits only,
store hint only, or hits + store hint.

### 4.1 Formatting hits

`_format_memory_hint` is at
[`memory.py:43`](../../src/audrey/pipeline/memory.py#L43). It takes
the hit list (each a dict with `key` and `value`) and produces
something like:

```text
[Relevant memories from previous conversations with this user:]
1. (favorite_color) blue
2. (city) Portland
Use these facts if they're relevant to the user's question. Ignore
irrelevant ones without mentioning them.
```

The trailing instruction is load-bearing — without it, models tend
to greet the user with "I see you're in Portland!" even when the
question is about Python. Telling the model to *use* facts when
relevant and *ignore* them silently when not relevant cuts that
behaviour.

There's a per-value truncation:

```python
value = (h.get("value") or "").strip()
if len(value) > 400:
    value = value[:400].rstrip() + "…"
```

A user can `memory_store` arbitrarily long values; 400 characters
per hit, times the configured `top_k` (default 3), caps the
total recall block at roughly a kilobyte. Worth more than that and
you're pushing the model's actual prompt off the front of context.

### 4.2 The store-hint and `{user_id}` substitution

Memory writes happen via the model itself calling the `memory_store`
tool. For the model to know *when* to call it, it needs a hint —
something like "if the user states a durable fact, write it to
memory." That hint lives in
[`pipeline/prompts.py:105`](../../src/audrey/pipeline/prompts.py#L105)
as `MEMORY_STORE_HINT`:

```python
MEMORY_STORE_HINT = (
    "If the user states a durable fact about themselves (preferences, "
    "goals, projects, constraints) or explicitly asks you to remember "
    "something, call the `memory_store` tool with: a short descriptive "
    "`key`, the fact as `value`, and `tags=\"user:{user_id}\"` "
    "(use exactly that user tag). Do this silently — do not narrate "
    "the tool call in your reply."
)
```

The `{user_id}` placeholder is significant. Memory is keyed by user,
and the tags field is how `memory_search` will later find this entry.
At injection time, `memory_system_message` substitutes the real user
identity:

```python
hint_template = prompt_from_config(cfg, "memory_store_hint", _MEMORY_STORE_HINT)
parts.append(hint_template.replace("{user_id}", user_id))
```

If the substitution didn't happen, the model would write back with a
literal `{user_id}` tag — every entry would land under the same
literal-string tag, and recall would never find anything for any
real user.

The store hint is **only included when there's a real user identity**
(`include_store_hint and user_id`). Anonymous requests can't write to
memory anyway — telling the model to call a tool that will reject the
call is wasted tokens.

### 4.3 When the store-hint actually fires

Back in [`graph.py:166-169`](../../src/audrey/pipeline/graph.py#L166)
the decision is one line:

```python
include_store_hint = tools is not None and MEMORY_STORE_TOOL in tools.by_name
```

Read it as: "we'll tell the model how to write back only if the write
tool actually exists in the registry right now." If `memory_store`
isn't a registered tool (custom-tools offline, or that endpoint was
removed), don't ask the model to call something that won't work.

The same registry-driven conditional applies to a second hint we
haven't met yet — chat-history search.


## 5. The composer

After memory hits and the store-hint are formatted, somebody has to
prepend them to `state["messages"]` in a defined order. Without a
rule, the order drifts every time a new system message gets added.
With a rule, the order lives in one function.

The rule is
[`compose_system_messages`](../../src/audrey/pipeline/prompts.py#L189):

```python
def compose_system_messages(
    *,
    incoming: list[dict[str, Any]] | None = None,
    task_role: str | None = None,
    memory_hint: dict[str, Any] | None = None,
    chat_history_guidance: bool = False,
    chat_history_text: str | None = None,
) -> list[dict[str, Any]]:
```

Four slots, fixed order:

1. **Incoming system messages.** Anything the user or OWUI sent with
   `role=system` lands first. The user's persona wins on tone.
2. **Task-role prompt.** A slot reserved for a per-task system
   prompt (e.g. "you are a fast answerer"). Currently always `None`
   — the slot exists so a future change can wire one in without
   touching the order rule.
3. **Memory recall + memory_store hint.** Passed in as one pre-built
   system message because `memory.py` already composed the body.
4. **Chat-history search guidance.** Only included when
   `chat_history_guidance=True` — same registry-driven gate as the
   store hint.

### 5.1 The chat-history-search conditional

[`CHAT_HISTORY_SEARCH_SYSTEM`](../../src/audrey/pipeline/prompts.py#L116)
is a system message that teaches the model when to call the
`chat_history_search` tool:

```python
CHAT_HISTORY_SEARCH_SYSTEM = (
    "Use `chat_history_search` only when the user references something "
    "they previously discussed with you, or when answering requires a "
    "specific prior decision. Do not call it for ordinary "
    "personalization or to repeat back recent context — it returns "
    "short snippets per call and burns context every time."
)
```

Why teach this in a system message instead of relying on the tool
description? Because the tool description tells the model *how* to
call it; the system message tells the model *when not to*. Without
the discipline, tool-capable models call `chat_history_search` on
nearly every request — burning latency and context for no signal.

The composer adds this block only when `chat_history_guidance=True`,
which the graph and the streaming route both compute from registry
membership:

```python
chat_history_available = tools is not None and "chat_history_search" in tools.by_name
```

Same pattern as the store-hint gate: don't teach the model about a
tool it can't call.


## 6. The two entry points

We've alluded to it twice; now it's explicit. Memory recall and
context injection have **two parallel call sites** that have to stay
in sync:

| | Non-streaming | Streaming-deep |
|---|---|---|
| Lives in | [`graph.py:139, 149`](../../src/audrey/pipeline/graph.py#L139) | [`routes/openai.py:937`](../../src/audrey/routes/openai.py#L937) |
| Datetime | `node_datetime` | direct `datetime_system_message()` call |
| Recall | `node_memory_recall` | direct `recall_for_request()` call |
| Composer | `compose_system_messages(...)` | `compose_system_messages(...)` |

The streaming route bypasses the graph because it needs control over
when to emit SSE progress frames (`> _Planning_`, `> _Dispatching
panel_`, `> _Synthesizing_`). Running the graph node-by-node and then
also emitting SSE turned out to be too rigid; the streaming route
calls the same building blocks directly so it can interleave them
with banner emission.

That's why the building blocks are in `pipeline/context.py` and
`pipeline/memory.py` (not just hidden inside the graph nodes) — they
have to be importable by both call sites. If a future change adds a
third kind of context injection, **both call sites need the update**.
The pattern works today; the audit queue tracks "consider extracting
a shared helper" as a future move if a third injection lands.


## 7. Comprehension questions

For each scenario below, sketch your answer before reading the
discussion. Operational judgment, not trivia.

**1. A user reports "I told Audrey my birthday is in July, but the
next day it didn't remember." What's the most likely cause? Trace the
write path and the read path.**

Memory writes happen via the model calling `memory_store` (a tool),
not via Audrey writing directly. Two failure modes are possible:

- **The model never called `memory_store`.** Check the request logs
  for `tool_calls`. Likely cause: the model wasn't tool-capable, or
  `memory_store` wasn't in the registry, or the model decided the
  fact wasn't "durable enough." The store-hint at injection time
  improves the success rate but doesn't force it.
- **The model called it but with a different tag.** If `{user_id}`
  substitution failed (e.g. the user_id was empty at hint-injection
  time), the entry got stored under a literal `user:{user_id}` tag
  and `memory_search` can't find it for any real user. Check the
  `tags` field in `chat_history_search` against the entry's stored
  payload.

**2. Memory recall returns three hits, but the model still asks
"what's your favorite color?" Why might that be?**

The hits arrived as a system message at the top of the prompt, but
the model decided they weren't relevant. The instruction at the end
of the memory block — "use these facts if they're relevant…ignore
irrelevant ones without mentioning them" — gives the model permission
to ignore. If the model is ignoring *relevant* hits, the likely
causes are: (a) the hit's `value` was truncated at 400 chars and the
relevant part was cut off, (b) the model is small enough that it
doesn't reliably read system messages above the user turn, (c) the
`top_k` is set too low and the relevant memory isn't in the
returned hits at all.

**3. Recall is timing out on every request. The timeout is 5 seconds.
Where do you look first?**

The custom-tools server, not Audrey. `dispatch_one` is the layer that
enforces the timeout, and a 5-second ceiling on `memory_search` means
the tool itself is slow — likely Qdrant is degraded, the embedder is
slow, or the query is being clamped at 500 chars but still embedding
slowly. From Audrey's side, the timeout is the right knob if you
want recall to give up faster; from the tools-server side, the right
move is to find what's slow. Either way, recall failures degrade
silently to "no recall" — the user gets an answer without memory,
which is the intended posture for a best-effort feature.

**4. You add a new system-message type, say "user's current OS." Where
do you add it, and what's the risk?**

Three changes are needed:

- A function somewhere (probably a new module) that builds the
  message — same shape as `datetime_system_message`.
- A new keyword arg on `compose_system_messages` and a slot
  decision (between memory and chat-history? Before incoming?).
- **Two call sites** to update: `node_memory_recall` in
  [`graph.py:149`](../../src/audrey/pipeline/graph.py#L149) (which
  currently owns the composer call) and `_phase_thinking` in
  [`routes/openai.py:937`](../../src/audrey/routes/openai.py#L937).

The risk: if you only update one, the new context is silently
missing from either streaming-deep requests or non-streaming
requests — a regression that's invisible until a user complains.
That's why the audit queue tracks "consider extracting a shared
helper" as a future cleanup.

**5. The store-hint includes `tags="user:{user_id}"`. If a model
called `memory_store` without that tag, what would break?**

`memory_search` filters by tag to scope recall to one user. Without
the `user:<id>` tag, the entry would be unfilterable — `memory_search`
would either miss it (if the search filters strictly) or surface it
across all users (if the search is permissive). The dispatcher's
user-overwrite invariant (`_USER_SCOPED_TOOLS` in
[`tools/dispatch.py`](../../src/audrey/tools/dispatch.py)) catches
some of this — it forces the `user` argument to the real pipeline
user — but the `tags` field is free-form; the model has to pick the
right shape itself. The store-hint is what teaches it to.

**6. Anonymous request (no `user_id`) hits the pipeline. Walk the
context injection.**

- `node_datetime` runs as usual. The datetime message has nothing
  to do with user identity.
- `node_memory_recall` checks `if not user_id: return {}` very
  early — no recall, no store hint, no system message added.
- `compose_system_messages` gets called with `memory_hint=None`
  and `chat_history_guidance=False` (the chat-history hint is
  also gated on registry membership, but the no-user path won't
  reach it). The composer returns `[]` and the messages list is
  unchanged.

Net result: anonymous requests get just the datetime injection.
That's the intended design — best-effort memory is per-user, and a
user without an identity has no memory to recall and no permission
to write any.


## When you're ready for the next lesson

This lesson opened the two earliest pipeline nodes — the ones that
prepend context before the classifier reads a single user word. The
next lesson in line backs up one layer: when multiple users are
hammering the same instance, what keeps one user's deep request from
monopolizing the GPU and starving everybody else? Fair scheduling, the
in-flight slot cap, and the round-robin guard are the upcoming
subject. Check `docs/lessons/` for what's landed since.
