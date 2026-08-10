# Lesson 15 — the OpenAI routes, end-to-end

**Estimated time:** 60-90 minutes. Keep the
[`routes/openai/`](../../src/audrey/routes/openai/) package open.

**Goal:** by the end of this lesson, you can answer
*"when a chat request lands, how does the route decide which pipeline
runs, how does it stream a deep panel without buffering the whole
answer, and what happens when the client hangs up?"*

Lesson 4 walked one request through Audrey at high altitude — enough
to see where the OpenAI route fits in the request lifecycle, not
enough to read any single function in it. This lesson lands on the
code. It was once a single ~1500-line module, but it owns three
responsibilities that don't share much code — schema validation, the
five virtual models, and the streaming SSE machinery — so it's now a
**package** split by responsibility:

| Module                         | What it holds                                  |
| ------------------------------ | ---------------------------------------------- |
| `schemas.py`                   | `ChatMessage`, `ChatCompletionRequest`         |
| `responses.py`                 | OpenAI response formatting (pure helpers)      |
| `passthrough.py`               | the `audrey_passthrough/<concrete>` family     |
| `pipeline.py`                  | the streaming + non-streaming engine           |
| `routes.py`                    | the `@router` endpoints (the thin dispatch layer) |

`__init__.py` re-exports the public surface, so `from
audrey.routes.openai import …` still works for `main.py` and the
tests. Read the modules in roughly the order above — each depends only
on the ones before it.


## 1. Context

### 1.1 What the route owns

The OpenAI route is Audrey's only public chat surface. Every request
a client sends to `POST /v1/chat/completions` lands here, gets
validated against `ChatCompletionRequest`, gets dispatched to one of
three entry points (passthrough, generate, stream), and eventually
returns either an OpenAI-shaped JSON object or an SSE stream of
OpenAI-shaped delta frames.

What it *doesn't* own: the pipeline itself (that's
[`pipeline/`](../../src/audrey/pipeline/)), classification
([`pipeline/classify.py`](../../src/audrey/pipeline/classify.py)),
the model registry
([`models/registry.py`](../../src/audrey/models/registry.py)), or
the chat archive
([`pipeline/chat_archive.py`](../../src/audrey/pipeline/chat_archive.py)).
The route is the *seam* between the OpenAI contract and Audrey's
internals; everything past dispatch is somebody else's responsibility.

### 1.2 Why five virtual models, not one

The OpenAI `chat/completions` request schema has exactly one
per-request control surface that survives every client SDK: the
`model` string. Audrey overloads it as a routing signal:

| `model` value                   | Behavior                                    |
| ------------------------------- | ------------------------------------------- |
| `audrey_deep`                   | Always deep panel (mixed local+cloud pool)  |
| `audrey_cloud`                  | Always deep panel (cloud-only pool)         |
| `audrey_local`                  | Always deep panel (local-only pool)         |
| `audrey_auto`                   | Adaptive — fast for ordinary short prompts, deep for long prompts or explicit depth cues |
| `audrey_fast`                   | Always fast (no escalation, ever)           |
| `audrey_passthrough/<concrete>` | Forward straight to Ollama, no pipeline     |

The principle is "the schema is the contract" (Lesson 2 §4): if a
client picks a model, Audrey can read that field and trust that
choice without asking elsewhere. Compare to the alternative — a
custom `X-Audrey-Mode` header — which would fail through any client
that doesn't expose custom header configuration (most of them).

The five non-passthrough names are listed once in `VIRTUAL_MODELS`
at [`routes/openai/routes.py:41`](../../src/audrey/routes/openai/routes.py#L41).
Passthrough uses a *prefix* (`audrey_passthrough/`) so one virtual
model can route to any concrete model in
`passthrough.allowed_models`; the deploy notes for the passthrough
family
([`docs/campaign-2/phase-13-passthrough-virtual-model.md`](../campaign-2/phase-13-passthrough-virtual-model.md))
cover the wiring, and we'll only touch the route-side branch here.

### 1.3 Streaming vs. non-streaming as different code paths

The same client sends the same OpenAI request twice, only flipping
`stream: true` ↔ `stream: false`. The response contract is the same
shape on both sides. The implementations are wildly different.

Non-streaming is a coroutine returning a dict: classify → fast or
deep → shape the response → return. The HTTP layer serializes it,
writes it once, closes the connection.

Streaming returns an `async generator` of strings, wrapped by
FastAPI's `StreamingResponse`. Each `yield` becomes one chunk on the
wire, *as it happens*. Deep mode produces a 30-second answer; the
client sees "Thinking…", then "Planning…", then per-worker progress,
then the synthesizer streaming tokens. If the route tried to build
the whole answer first and then stream it, the user would stare at
a blank screen for half a minute and probably disconnect — defeating
the point of streaming.

Those two paths live in `_generate_via_pipeline` and
`_stream_via_pipeline` respectively. They share `inflight.slot()`,
classification, and the model-shape contract; they don't share much
else.


## 2. Read-along

### 2.1 The schema

[`routes/openai/schemas.py:16-57`](../../src/audrey/routes/openai/schemas.py#L16)
defines two Pydantic models:

```python
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = ...
    user: str | None = ...
```

Three fields earn special mention:

- **`messages: list[ChatMessage] = Field(min_length=1)`** — at least
  one message is required. An empty list is a 422 before any of the
  route body runs. Pydantic enforces this in FastAPI's dependency
  resolution phase, *before* `Depends(require_user)` even has a
  chance to check the auth header. (More on the order in §2.2.)
- **`tools`** is OpenAI-spec passthrough but **only honored on the
  passthrough path**. Pipeline modes ignore it — Audrey's tools come
  from the server-side registry at
  [`tools/discovery.py`](../../src/audrey/tools/discovery.py), not
  from per-request client claims. The docstring on the field calls
  this out explicitly because the silent-ignore is a real surprise
  for someone debugging "I sent a `tools` array, why aren't they
  showing up?"
- **`user`** is also OpenAI-spec passthrough but **not trusted for
  identity**. Audrey's user ID comes from the bearer token
  (`require_user → AuthedUser.email`); `payload.user` is logged for
  drift-debugging at
  [`routes/openai/routes.py:112`](../../src/audrey/routes/openai/routes.py#L112)
  and otherwise ignored. The field is in the schema purely for
  client compat.

### 2.2 The dispatch decision tree

The route entry is
[`routes/openai/routes.py:87`](../../src/audrey/routes/openai/routes.py#L87).
The ordering of checks is load-bearing — getting it wrong would
either leak identity surface or let invalid passthrough requests
escape into the pipeline.

```
POST /v1/chat/completions
  ↓
Pydantic validation (Field min_length, type checks)        → 422 on fail
  ↓
Depends(require_user)                                       → 401 on fail
  ↓
_is_passthrough(payload.model)?  → _handle_passthrough  ─→ Ollama
  ↓ no
payload.model in VIRTUAL_MODELS?                            → 400 on fail
  ↓ yes
payload.stream?
  ├─ true  → StreamingResponse(_stream_via_pipeline(...))
  └─ false → await _generate_via_pipeline(...)
```

Three things to notice in
[`routes/openai/routes.py:87`](../../src/audrey/routes/openai/routes.py#L87):

  - **Passthrough is checked first** because it owns its own model-string
    space (`audrey_passthrough/<x>`) and isn't in `VIRTUAL_MODELS`. If
    the order were reversed, passthrough requests would 400 with
    "Unknown model" before reaching the passthrough handler.
  - **`VIRTUAL_MODELS` is validated in the route, not the schema.** A
    Pydantic `Literal[...]` would push this to the 422 layer with
    less-helpful error text. The route check at
    [`routes/openai/routes.py:104`](../../src/audrey/routes/openai/routes.py#L104)
    emits `"Unknown model 'X'. Supported virtual models: [...]"` —
    actionable enough that a developer trying `audrey_deeo` (typo)
    can fix it without grepping the source.
  - **`conversation_id` is resolved once, before the branch** at
    [`routes/openai/routes.py:151`](../../src/audrey/routes/openai/routes.py#L151).
    Both pipeline paths receive the same id, so a stream and a
    non-stream completion of the same conversation thread into the
    archive correctly. Lesson 13 §2.5 covered how
    `resolve_conversation_id` derives the id from OWUI's
    `chat_id` or falls back to a deterministic hash.

### 2.3 Concept spotlight: SSE and `StreamingResponse`

Server-Sent Events (SSE) is HTTP's standard one-way streaming
protocol. The server keeps a connection open and writes successive
`data: <payload>\n\n` frames; the client (a browser's
`EventSource`, OWUI's chat UI, an OpenAI SDK in streaming mode)
parses them as they arrive.

In FastAPI, you return a `StreamingResponse` with `media_type=
"text/event-stream"` and pass it an async generator. Every `yield`
in the generator becomes one chunk on the wire:

```python
async def my_stream():
    yield "data: hello\n\n"
    await asyncio.sleep(1)
    yield "data: world\n\n"

return StreamingResponse(my_stream(), media_type="text/event-stream")
```

OpenAI's chat-completion streaming spec is more structured: each
frame is a JSON object describing a *delta* (a piece of the response
being built), with a final `data: [DONE]\n\n` marker. Audrey emits
that shape directly. You can see the helpers at
[`routes/openai/pipeline.py:541`](../../src/audrey/routes/openai/pipeline.py#L541)
inside `_stream_deep_with_banners`:

```python
def _delta_frame(text: str) -> str:
    frame = {
        "id": cid, "object": "chat.completion.chunk", ...,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(frame)}\n\n"

def _stop_frame() -> str:
    frame = {
        "id": cid, "object": "chat.completion.chunk", ...,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(frame)}\n\n"
```

`_delta_frame` adds content; `_stop_frame` says "the answer is
complete." Every streaming response ends with one `_stop_frame()`
plus the literal `data: [DONE]\n\n` sentinel.

The crucial property: there's no buffer between the generator and
the client. When you `yield _delta_frame("hello")`, the HTTP layer
flushes "hello" out immediately. That's why streaming actually
streams — and why the next subsection's architecture matters.

### 2.4 The non-streaming path

`_generate_via_pipeline` at
[`routes/openai/pipeline.py:108`](../../src/audrey/routes/openai/pipeline.py#L108)
is the simpler half. The shape:

```
build initial PipelineState
  ↓
inflight.slot(user_id) acquired
  ↓
_run_graph_with_metrics(...) awaits the compiled graph
  ↓
graph nodes classify, check complexity/depth intent, and choose fast or deep
  ↓
_to_openai_response(...) shapes the answer
  ↓
archive_turn(...) records the finished turn, best-effort
  ↓
inflight.slot released (async with exit)
```

This is the path Lesson 4 previewed. The graph itself
([`pipeline/graph.py`](../../src/audrey/pipeline/graph.py)) is
the orchestration; the route just hands it a state dict, awaits a
result, and shapes the result into an OpenAI response. One coroutine,
one return.

The interesting wrinkle is `is_owui_task_request(messages)`. For the
non-streaming path you're reading here, that check runs *inside the
graph* — the complexity node calls it while choosing the mode at
[`pipeline/graph.py:291`](../../src/audrey/pipeline/graph.py#L291),
so `_generate_via_pipeline` itself never touches it (it just hands the
graph a state dict and awaits the result). OWUI fires "generate title"
and "generate tags" utility prompts that bundle the whole conversation
as one user message. These fire in the *background* — OWUI issues them
itself, after a turn, to populate the sidebar. No human picks a mode
for them; the request just inherits whatever virtual model the
conversation is pinned to (`audrey_deep`, say). So when the check
forces fast mode for these, it isn't overriding a choice the user made
— it's declining to act on a choice nobody made, refusing to burn a
full deep-panel run on a string OWUI will truncate to one line. The
streaming path runs the *same* check inline at
[`routes/openai/pipeline.py:254`](../../src/audrey/routes/openai/pipeline.py#L254)
(because it bypasses the graph — §2.5); this is a deliberate two-gate
behavior, not a streaming-path one-off.

### 2.5 The streaming deep path — `_stream_deep_with_banners`

Open
[`routes/openai/pipeline.py:492`](../../src/audrey/routes/openai/pipeline.py#L492).
This function is the single longest in the file (~330 lines) and
the most complicated. It's a streaming deep panel: a 30-second
response built from multiple parallel workers, with banner text
("Thinking…", "Planning…", "Dispatching panel…", "Synthesizing…")
emitted to keep the client engaged while the real work happens.

Why is it complicated? Because two things have to happen at once:
banner text needs to flow to the client *immediately*, while the
graph's worker tasks run in parallel in the background. If the
route awaited the workers and then started streaming, the user would
see nothing for 25 seconds. If the route ran the workers
inline-with-yielding, it couldn't yield banner text during the
gaps. So the route splits producer and consumer roles using a
queue.

The shape:

```
                ┌─────────────────────────────────────┐
                │ _stream_deep_with_banners           │
                │ (the route generator)               │
                │                                     │
   client ◄──── │ yields _delta_frame(text) chunks    │
                │                                     │
                │   pulls from:                       │
                │   ┌────────────┐  ┌──────────────┐ │
                │   │ banner_q   │  │ events_q     │ │
                │   │ asyncio.   │  │ asyncio.     │ │
                │   │ Queue      │  │ Queue        │ │
                │   └────────────┘  └──────────────┘ │
                │        ▲                ▲          │
                │        │                │          │
                │   pushes from           pushes from│
                │   ┌────────────┐  ┌──────────────┐ │
                │   │ PhaseTicker│  │ synth_task   │ │
                │   │ background │  │ background   │ │
                │   │ task       │  │ task         │ │
                │   └────────────┘  └──────────────┘ │
                └─────────────────────────────────────┘
```

`PhaseTicker` (defined in
[`pipeline/banners.py`](../../src/audrey/pipeline/banners.py))
spins a small background task that posts dots to the banner queue
every few hundred ms so the user sees motion. `synth_task` is the
synthesizer streaming tokens through `synthesize_stream`. The route
generator's job is to interleave both queues into one SSE stream.

### 2.6 Concept spotlight: `asyncio.Queue` as backpressure

Lesson 14 introduced `asyncio.Future` as a one-shot signal. The
streaming route uses a related primitive: `asyncio.Queue`, a
bounded FIFO that decouples producer and consumer.

```python
banner_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)
```

The producer (`PhaseTicker`) does `await banner_q.put(text)`. The
consumer (the route generator) does `banner_q.get()` (or `.get_nowait()`
if it doesn't want to block). They never see each other; they share
the queue.

Three properties matter:

  - **Bounded.** `maxsize=128` means the producer blocks if the
    queue is full. This is *backpressure*: if the route is slow at
    consuming (network is slow, client is slow), the banner task
    can't pile up unread frames in memory. It waits.
  - **`None` is the sentinel.** Both queues in
    `_stream_deep_with_banners` use `Optional[T]`; pushing `None`
    means "I'm done producing." The consumer treats `None` as
    end-of-stream. This is a common pattern — Python's `Queue`
    has no built-in "done" signal, so you reserve a sentinel value
    and document it.
  - **`async with` doesn't apply.** Queues don't have a lifecycle to
    manage; they're just buffers. The producer/consumer tasks have
    `try/finally` blocks for cleanup, but the queue itself just
    exists for as long as both ends hold a reference.

Why a Queue and not a Future? Futures resolve once and stop. The
banner ticker emits dozens of fragments over a single deep call;
each fragment needs to flow through and be consumed independently.
A Queue is the right shape when *many* signals flow one direction.

### 2.6.1 What `PhaseTicker` actually is

§2.6 named the producer — `PhaseTicker` — but treated it as a box that
"posts dots." It's worth a few lines on what's inside that box, because
it's the piece that turns "work is happening" into the growing
`> _Dispatching panel..._  ✅ kimi-k2.6:cloud  ❌ qwen3.6:35b` line you
watch during a deep request. It lives in
[`pipeline/banners.py`](../../src/audrey/pipeline/banners.py), and it is
deliberately small: it knows nothing about synthesis, drafts, or models.
It receives a header string and an *emitter*, and it manages one thing —
the lifecycle of a single phase's progress line.

**It's an async context manager.** You met those in Lesson 1; here's one
doing real work. The route uses it like this (the actual call site for the
panel phase is
[routes/openai/pipeline.py:629](../../src/audrey/routes/openai/pipeline.py#L629)):

```python
async with PhaseTicker(BANNER_DISPATCHING, emit) as ticker:
    drafts = await run_the_panel(..., ticker)
```

The context-manager shape *is* the design. `__aenter__` emits the header
(`> _Dispatching panel_`) and starts a background task. `__aexit__` stops
that task and emits the closing mark — and crucially, it emits ` ❌` if the
block raised and ` ✅` if it didn't. So phase failure is visible to the
user for free: any exception inside the `async with` body produces a
red-cross banner on the way out, and then propagates (the context manager
does not swallow it). You don't write error-banner code at each call site;
the lifecycle handles it.

**It runs two background tasks, not one,** and the reason is a small race.
One task is the *tick loop*: every five seconds it appends a `.` to the
header so the line looks alive during a long phase. The other is the *tail
drainer*: it emits fragments pushed via `ticker.append_tail(...)` — the
per-worker `✅ model` / `❌ model` marks that arrive as workers finish.
These are kept on separate tasks so a worker-result mark and a dot can't
interleave incorrectly with the closing checkmark. On exit, the tick task
is cancelled *first* (so no stray dot lands after the ✅), then the tail
queue is drained, then the closing mark is emitted — a fixed order that
keeps the rendered line coherent.

**Two queues, two backpressure policies — don't conflate them.** This is
the easy thing to get wrong. There are two queues in play:

  - The **route's** `banner_q` (the one §2.6 showed, `maxsize=128`). The
    emitter the route hands to `PhaseTicker` does `await banner_q.put(text)`.
    So when the ticker emits a *dot* or a *header*, and the client is slow,
    the `await put` blocks — but it blocks the **tick task only**, never the
    coroutine running the actual model work. That's why a slow network
    stutters the dots without ever stalling synthesis. (The tick loop's
    comment in `banners.py` now says exactly this.)
  - The **ticker's internal** `_tail_q` (`maxsize=64`), used only for
    `append_tail` fragments. This one uses `put_nowait` with **drop-on-full**:
    if per-worker marks arrive faster than a congested client drains them,
    the ticker drops a mark rather than block. The reasoning is a deliberate
    tradeoff — for progress decoration, a missing checkmark under transient
    congestion is better than stalling the panel. Compare this with the dot
    path, which *does* block (on the tick task): dots are the heartbeat, so
    blocking the tick task is acceptable; worker marks are nice-to-have, so
    they're droppable.

**The pure formatters.** The rest of `banners.py` is plain string
functions, no async: `worker_ok(model)` / `worker_fail(model)` produce the
inline `  ✅ model` tail fragments, and `tool_summary_block(...)` renders
the per-worker "Tools used" footer that appears below the answer. These are
shared identically by the deep panel and the fast path — the same footer
shape regardless of which path ran. They're independent formatters by
design (the inline-tail producers and the footer producer render into
different contexts), not a single shared convention; don't read a common
format contract into them that isn't there.

The takeaway: banners are a thin UX layer wrapped around the real work.
The route owns delivery (the `banner_q`, the SSE framing); `PhaseTicker`
owns one phase's progress lifecycle; and synthesis itself — the part that
makes the answer — happens inside the `async with` body and is covered in
the deep-mode lesson, not here.

### 2.7 Cancellation — what actually happens when the client hangs up

The deferred Lesson 4 question: *when the browser tab closes
mid-stream, does anything get cleaned up?*

Trace the cancel through:

  1. **The HTTP transport sees the client disconnect.** Starlette
     (FastAPI's underlying ASGI framework) raises
     `asyncio.CancelledError` into the generator that's producing
     SSE frames.
  2. **The generator's `try` block catches it** at
     [`routes/openai/pipeline.py:799`](../../src/audrey/routes/openai/pipeline.py#L799).
     The route records `pipeline_outcome = "cancelled"` (so the
     metric reflects "user left," not "ok") and re-raises.
  3. **The inner `try/finally` at
     [`routes/openai/pipeline.py:791`](../../src/audrey/routes/openai/pipeline.py#L791)**
     cancels the synth producer task explicitly:

     ```python
     finally:
         if not synth_task.done():
             synth_task.cancel()
             try:
                 await synth_task
             except (asyncio.CancelledError, Exception):
                 pass
     ```

     `await synth_task` after the cancel makes sure the cancel
     actually lands (and any cleanup inside the synth coroutine
     finishes) before we move on. Without it, the synth task could
     keep running in the background after the route returned —
     a slow leak.
  4. **The synth task's own cancellation** propagates through
     `synthesize_stream`'s `async for` into the `httpx.AsyncClient`
     making the actual upstream HTTP call. httpx documents
     CancelledError as closing the underlying socket — so the
     local-Ollama call stops generating tokens promptly. Cloud
     calls close the socket the same way, *but the upstream
     provider may keep generating and billing* until they notice
     the client is gone. That's outside Audrey's control.

  5. **The `async with inflight.slot(user_id)` and
     `gate.acquire()` blocks** unwind via the `try/finally` that
     `@asynccontextmanager` adds. The inflight semaphore is
     released; the gate's release path runs (Lesson 14 §2.5
     covered the gate's cancellation handling at
     `fair_gate.py:120`). Other parked waiters wake up cleanly.

Net result: the route releases the in-flight slot, unwinds any active gate
contexts, cancels the synth producer when cancellation reaches that stage, and
closes the Ollama socket. The archive write is still attempted from the deep
stream `finally` block with `partial=True` when the route knows the stream was
cancelled. One remaining cleanup gap is earlier background phase tasks:
`think_task` and `panel_task` are awaited in the normal drain path, but unlike
`synth_task` they do not have their own explicit cancel-and-await `finally` if
the client disconnects while that phase task is running. Cloud-billing for
already-dispatched cloud workers is outside Audrey's control.

### 2.8 The passthrough fork

[`_handle_passthrough` at routes/openai/routes.py:95](../../src/audrey/routes/openai/routes.py#L98)
is a sibling of the pipeline dispatch — same `inflight.slot()` wrap,
same fair-gate acquisition (inside the helper), but no classifier,
no complexity gate, no banners. It exists for one specific use case:
LAN clients that already know what model they want and just need
the GPU-fairness layer in front of Ollama. The passthrough deploy
notes cover the design rationale and the per-client wiring.

The route-side fork is small. It splits into streaming
(`_passthrough_stream_sse` at
[`routes/openai/passthrough.py:187`](../../src/audrey/routes/openai/passthrough.py#L187))
and non-streaming. The streaming variant *doesn't share*
`_stream_deep_with_banners` because there are no banners — Ollama's
own chunks get reshaped to OpenAI SSE format and forwarded
verbatim. The `_ollama_to_openai_tool_calls` helper at
[`routes/openai/responses.py:73`](../../src/audrey/routes/openai/responses.py#L73)
handles one specific format mismatch: Ollama returns tool-call
arguments as a dict, OpenAI clients expect a JSON string. Audrey
serializes it before forwarding.

For purposes of this lesson, treat the
passthrough fork as "a near-clone of the pipeline path with most of
the smarts stripped out, for callers who already made every routing
decision themselves."

### 2.9 Why `_options_from_request` exists

[`routes/openai/responses.py:20`](../../src/audrey/routes/openai/responses.py#L20):

```python
def _options_from_request(req: ChatCompletionRequest) -> dict[str, Any]:
    """Map OpenAI-shape sampling knobs onto Ollama's options dict.

    Sibling: `pipeline.graph._options_from_state` does the same conceptual
    mapping from the LangGraph state dict. ...
    """
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    return opts
```

Small helper that maps OpenAI-shape sampling knobs onto Ollama's
options dict. (That `Sibling:` docstring note — shipped during this
lesson's own audit — points straight at the near-twin.) There's a
near-twin in
[`pipeline/graph.py:656`](../../src/audrey/pipeline/graph.py#L656)
called `_options_from_state` that does the same conceptual mapping
from the LangGraph state dict instead of a Pydantic object. The
two functions look like they want to be one, but their input shapes
genuinely differ: one has typed attribute access on a Pydantic model,
the other reaches into a dict with `.get()`. Folding them together
would mean either inventing a third "view" type or weakening typing
to `Any`-with-`getattr` — neither is worth it for three lines of
logic each.

The takeaway: two small parallel helpers can be the right answer
when their *only* commonality is the shape of the output and the
inputs are structurally different.


## 3. Comprehension questions

**1. A client POSTs to `/v1/chat/completions` with `messages: []`
and a valid bearer token. What HTTP status do they get, and at
what point in the request lifecycle does it fire?**

422, before `require_user` even runs. FastAPI resolves request
validation as part of dependency injection, and Pydantic's
`Field(min_length=1)` on `ChatCompletionRequest.messages`
([`routes/openai/schemas.py:31`](../../src/audrey/routes/openai/schemas.py#L31))
rejects the empty list during schema validation — which happens
*before* the route's body and *before* its declared dependencies
get awaited. The user gets a structured 422 with a path-based error
("messages: List should have at least 1 item"). The fact that the
bearer token was valid is irrelevant; the request never got that
far. A practical implication: hitting `/v1/chat/completions` with
bad JSON or a malformed schema doesn't even touch the auth-cache
state, so a client spamming malformed requests can't cost you
authentication round-trips.

**2. A client sends `model: "audrey_fast"` with a 3,000-token
prompt. What runs?**

Fast path, unconditionally. The `audrey_fast` virtual model is the
"always fast" override: at
[`routes/openai/pipeline.py:253`](../../src/audrey/routes/openai/pipeline.py#L253)
the route sets `forced_fast = payload.model == "audrey_fast"`, and
the subsequent branch picks fast regardless of `is_complex()`'s
verdict. The complexity gate that normally escalates long prompts
to deep mode (in `audrey_auto`) is bypassed. The user explicitly
chose fast; the route respects that.

This is the inverse of the OWUI utility-prompt case in §2.4: there,
a request *nobody* chose a mode for gets forced cheap; here, a mode
the user *explicitly* chose overrides the complexity heuristic.
Both are deliberate, and they share one rule of thumb: invisible
behaviors (OWUI title-gen) get forced into the cheap path; explicit
user choices (picking `audrey_fast`) get honored even when a heuristic would
disagree.

**3. A streaming deep request's synth model dies mid-stream. What
does the client see? What does the archive write look like?**

The `_stream_deep_with_banners` exception handler at
[`routes/openai/pipeline.py:791`](../../src/audrey/routes/openai/pipeline.py#L791)
catches the failure, sets `pipeline_outcome = "error"`, and yields
two final frames: a delta containing `"\n\n[ollama error: ...]"`
(or `"\n\n[internal error]"` for non-Ollama exceptions) and a stop
frame, plus the `data: [DONE]\n\n` sentinel. The client sees their
partial answer with an error banner appended — not a 500, because
the HTTP response already started streaming and the status was
committed to `200 OK` the moment the first frame went out.

The archive write at
[`routes/openai/pipeline.py:818`](../../src/audrey/routes/openai/pipeline.py#L818)
runs in the `finally` block, which fires *after* the error
handling. It captures whatever `final_content` accumulated up to
the failure (which may be partial or empty if the failure came
before first_token). Lesson 13 §2.6 documented the archive as
best-effort: this partial-write case is exactly what that
best-effort posture is designed for. The user keeps what they
got; the archive records what actually streamed.

**4. A passthrough request comes in with `tools: [...]`. Trace
what gets forwarded and what gets reshaped.**

The route hits
[`routes/openai/routes.py:98`](../../src/audrey/routes/openai/routes.py#L98),
recognizes the `audrey_passthrough/` prefix, and dispatches to
`_handle_passthrough`. There the `tools` array is forwarded
*verbatim* to Ollama — Audrey doesn't filter, validate, or
substitute. This is the only path where `payload.tools` does
anything; in pipeline modes, the field is dropped on the floor
(see §2.1 and the field's docstring at
[`routes/openai/schemas.py:36`](../../src/audrey/routes/openai/schemas.py#L36)).

Reshaping happens *on the way back*. Ollama returns tool-call
arguments as a Python dict, but the OpenAI streaming spec expects
arguments as a JSON-encoded *string*. The reshape lives at
[`routes/openai/responses.py:73`](../../src/audrey/routes/openai/responses.py#L73)
(`_ollama_to_openai_tool_calls`) and serializes each call's
arguments via `json.dumps`, plus generating a synthetic `id`
(Ollama doesn't supply one). Agent clients like Hermes and
OpenClaw call `json.loads` on the field and would crash on a
raw dict — that's why this conversion is load-bearing rather than
cosmetic.

**5. The browser tab closes after the planner has dispatched four
workers but before any has finished. Walk the cleanup, naming the
specific seam each layer relies on.**

Five things have to land cleanly:

- **The route generator** receives `asyncio.CancelledError` from
  Starlette. It catches at
  [`routes/openai/pipeline.py:799`](../../src/audrey/routes/openai/pipeline.py#L799),
  records `outcome="cancelled"`, and re-raises.
- **The inner `try/finally` at [`routes/openai/pipeline.py:791`](../../src/audrey/routes/openai/pipeline.py#L791)** cancels
  `synth_task` and awaits it — making sure the synth producer
  doesn't keep streaming into a queue nobody reads.
- **The panel phase task** is the current weak point. In the normal path,
  `_phase_dispatch` runs inside `panel_task` at
  [`routes/openai/pipeline.py:590`](../../src/audrey/routes/openai/pipeline.py#L590)
  and is awaited before synthesis begins. If cancellation arrives while the
  route is already in synthesis, the panel workers have already finished and
  their gate contexts have exited. If cancellation arrives while `panel_task`
  itself is still running, the route does not currently have a dedicated
  cancel-and-await `finally` for that task the way it does for `synth_task`.
- **The `async with inflight.slot(user_id)`** at the top of
  `_stream_via_pipeline` releases the per-user semaphore on its way
  out, so the user's next request is not blocked by the orphaned slot.
- **Cloud workers' upstream sockets** close via httpx's cancel
  handling. The cloud *provider* may keep billing for already-dispatched
  tokens, since the cancel doesn't reach them as a "stop now"
  signal — it just closes the receiving socket. Audrey doesn't pay
  for tokens already generated; the cloud provider does whatever
  they do.

The load-bearing pattern is still `async with` cleanup, plus the explicit
`synth_task.cancel()` block. The practical audit note is that synth has the
manual cleanup path; planning and panel phase tasks currently rely on normal
await/cancellation propagation and are not cancelled explicitly by the route.

**6. OWUI fires a "generate title" utility request with the user's
whole conversation bundled as one user message. The user happens to
have selected `audrey_deep`. What runs and why?**

Fast mode runs, not deep. The `is_owui_task_request(messages)`
check at
[`routes/openai/pipeline.py:254`](../../src/audrey/routes/openai/pipeline.py#L254)
detects OWUI's utility prompts by a single tell: the latest user
message opens with the `### Task:` header OWUI stamps on its
internal prompts (the conversation is bundled into the body, but
it's the header — not the size — that the check keys on). When it
fires, `use_deep` is forced to `False` regardless of the virtual
model.

The reasoning: OWUI's title-gen, tag-gen, and follow-up-suggestion
prompts are *invisible* behaviors the user didn't actively pick.
They fire every time the user sends a chat. If a user has
`audrey_deep` selected because they want deep mode for *their*
prompts, they don't want title-gen also burning 30 seconds and
panel-of-workers GPU time on every turn. Forcing fast for OWUI's
internal prompts keeps deep mode reserved for what the user
actually asked.

It works because OWUI's utility prompts are structurally distinct
(the `### Task:` header isn't something a real user types) —
pattern-matching has a low false-positive rate. The fragile part
isn't the forcing-fast decision, which is plainly right; it's the
detection. The prefix belongs to OWUI, not to Audrey, so an OWUI
upgrade that renames it would silently break the match — no error,
just title-gen quietly starting to route by token count again. If a
real user *did* type that header, the check would misfire the other
way, but that's a corner case worth accepting for the common-case
win.


## When you're ready for the next lesson

This lesson opened Audrey's only public chat surface end-to-end —
schema, dispatch, streaming machinery, cancellation cleanup, and
(in §2.6.1) the `PhaseTicker` progress-banner layer that rides
along those streams. That's the route surface fully covered.

The remaining subsystem isn't a route at all — it's the *other side*
of every tool call. Each time the model invokes `web_search`,
`kb_search`, or `memory_store`, the request crosses to a separate
service: the custom-tools sidecar. The next lesson opens it — how a
tool route becomes a model-callable tool, and what lives behind each
one.

(The admin surface — `routes/admin.py` plus the `require_admin`-gated
`/v1/tools/rediscover` in `main.py` — isn't covered by its own lesson;
those endpoints are operator levers documented in their own
docstrings.)
