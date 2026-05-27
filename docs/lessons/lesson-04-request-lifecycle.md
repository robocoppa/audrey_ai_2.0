# Lesson 4 — The request lifecycle, end-to-end

**Estimated time:** 45-60 minutes (read 25, walk through code 20)

**Goal:** by the end of this lesson, you can answer the question
*"what happens between the user clicking 'send' in Open WebUI and the
answer appearing on screen?"* — at a level where every term has a
referent in your head, even if you don't yet know how each piece is
implemented.

We're not going deep on any single component today. The point is to
build a **map**. Subsequent lessons fill in each region in detail.


## 1. Context

Audrey sits between Open WebUI (a chat frontend) and Ollama (the
inference engine that actually runs the language models). When a user
types a question and hits send, OWUI doesn't talk to Ollama directly —
it sends an HTTP request to Audrey, and Audrey decides:

- Which model(s) should answer this?
- Should I use one model and answer fast, or several in parallel and
  pick the best?
- Should the model use any tools (web search, KB lookup, memory)?
- How do I stream the answer back as it's being generated, with
  progress banners, so the user sees something happening?

That decision-making is the **pipeline**. The thing that *runs* the
pipeline is **LangGraph**. The thing that *exposes the pipeline to the
network* is **FastAPI**. The thing that *makes everything concurrent*
(so we can wait for the model to think without blocking other requests)
is Python's built-in `asyncio`.

If you read Lessons 1, 2, and 3 those names should be familiar in the
abstract — this lesson is where you see them touch each other in real
code.


## 2. Read-along

Open these files in your editor as we go. Don't try to understand
every line yet — just *locate* the things we point at.

### 2.1 The HTTP entry point

Open [`src/audrey/routes/openai.py`](../../src/audrey/routes/openai.py).
Scroll to **line 124**:

```python
@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
):
```

This is where every request lands. Read the function (lines 124-158)
top to bottom. Notice:

- **`@router.post("/chat/completions")`** — a FastAPI decorator. It
  registers this function as the handler for `POST /v1/chat/completions`
  HTTP requests. The `/v1` prefix is set on line 63
  (`router = APIRouter(prefix="/v1")`).
- **`payload: ChatCompletionRequest`** — FastAPI parses the incoming
  JSON body into a `ChatCompletionRequest` object (defined at line
  84). If the JSON doesn't match, FastAPI returns 422 automatically
  before your code runs.
- **`me: AuthedUser = Depends(require_user)`** — FastAPI's dependency
  injection. Before `chat_completions` runs, FastAPI calls
  `require_user()` (in `auth.py`), which checks the `Authorization:
  Bearer <token>` header against Open WebUI. If it fails, the user
  gets 401 and `chat_completions` never runs. If it succeeds, `me`
  contains their email + role.
- **The `if payload.stream:` branch (lines 152-156)** — OpenAI's API
  supports two modes: Streaming (server pushes tokens as they're
  generated) and non-streaming (server waits, returns full answer in
  one JSON response). Audrey supports both. Streaming is what OWUI
  uses; non-streaming is what programmatic clients use.

So the request, after auth and JSON parsing, splits into one of two
paths: **`_stream_via_pipeline`** or **`_generate_via_pipeline`**.
Both eventually call the same LangGraph pipeline; they differ in *how
the result is returned to the client*.

### 2.2 The pipeline graph

Open [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py).
Scroll to **line 376**:

```python
g: StateGraph = StateGraph(PipelineState)
g.add_node("datetime", node_datetime)
g.add_node("memory_recall", node_memory_recall)
g.add_node("classify", node_classify)
g.add_node("complexity", node_complexity)
g.add_node("fast_path", node_fast_path)
g.add_node("escalate_bridge", node_mark_escalated)
g.add_node("planner", node_planner)
g.add_node("deep_panel", node_deep_panel)
g.add_node("synthesize", node_synthesize)
g.add_node("reflect", node_reflect)
```

This is the whole pipeline laid out as a graph. **Ten nodes.** Each
one is a Python `async def` function defined earlier in the same file
(grep `def node_` — you'll see them at lines 128, 138, 171, 188, 208, 235, 252, 284, 303, 369).

The graph topology — which node runs after which — is on **lines
388-407**:

```python
g.set_entry_point("datetime")
g.add_edge("datetime", "memory_recall")
g.add_edge("memory_recall", "classify")
g.add_edge("classify", "complexity")
g.add_conditional_edges("complexity", route_after_complexity,
    {"fast": "fast_path", "deep": "planner"})
g.add_conditional_edges("fast_path", route_after_fast_path,
    {"end": END, "escalate": "escalate_bridge"})
g.add_edge("escalate_bridge", "planner")
g.add_edge("planner", "deep_panel")
g.add_edge("deep_panel", "synthesize")
g.add_edge("synthesize", "reflect")
g.add_conditional_edges("reflect", route_after_reflect,
    {"end": END, "retry": "deep_panel"})
```

Read these top to bottom. The pipeline always starts at `datetime` and
flows through `memory_recall` → `classify` → `complexity`. At
`complexity`, it splits — short prompts go through `fast_path` (one
model answers); long prompts go through `planner` → `deep_panel` (many
models answer in parallel) → `synthesize` (one model reconciles them).

Here's the same thing as a picture (read top to bottom, branches
right):

```
   datetime
      │
      ▼
  memory_recall   ← inject any per-user memories matching the prompt
      │
      ▼
   classify       ← code? reasoning? general? vl?
      │
      ▼
  complexity      ← short prompt or long prompt?
      │
   ┌──┴──────────────────────────────┐
   │                                 │
short                               long
   │                                 │
   ▼                                 ▼
 fast_path                        planner    ← decompose into subtasks
   │                                 │
   ├─ ok? ──► END                    ▼
   │                              deep_panel  ← N workers run in parallel
   └─ too short? ──► planner          │
       (escalate)                     ▼
                                  synthesize  ← one model reconciles drafts
                                      │
                                      ▼
                                   reflect   ← was the answer good enough?
                                      │
                                      ├─ ok? ──► END
                                      └─ too short? ──► back to deep_panel (retry once)
```

Don't memorize this — you'll see it many more times. The point right
now is just: **the pipeline is a graph, and each node is a function.**

### 2.3 What a node looks like

Look at [`graph.py:139`](../../src/audrey/pipeline/graph.py#L139) —
the `node_datetime` function:

```python
async def node_datetime(state: PipelineState) -> dict[str, Any]:
    msg = datetime_system_message()
    new_messages = [msg, *state["messages"]]
    return {"messages": new_messages}
```

That's it. A node is a function that:

1. Reads from `state` (a dictionary of everything the pipeline knows
   so far).
2. Does some work.
3. Returns a dict of fields to update in the state.

LangGraph merges the returned dict into the state and runs the next
node. This is the whole pattern. Every node — even the most complex
ones like `node_deep_panel` — is structurally just this.

`PipelineState` is defined in
[`pipeline/state.py`](../../src/audrey/pipeline/state.py) as a `TypedDict`
— see [Lesson 1 §4](lesson-01-foundations.md#4-typed-dictionaries-typeddict)
if that's unfamiliar. The short version: It's a regular dict where the
allowed keys and their types are declared up front, so the type checker
can catch typos and editors can autocomplete.

### 2.4 The request flow, end to end

Now we can trace one request all the way through. I'll narrate; you
follow along in `openai.py`.

**A user types "what is BTRFS?" in Open WebUI and hits send.**

1. **OWUI sends `POST /v1/chat/completions`** with a JSON body
   containing the model name (e.g. `audrey_auto`), the message
   history, and `"stream": true`. The Authorization header carries
   the user's bearer token from their OWUI login.

2. **Cloudflared routes it to the audrey-ai container.** (We won't
   touch Cloudflare in this course; that's container infra, see
   `docs/campaign-1/`.) Inside the container, FastAPI receives the
   request.

3. **FastAPI runs `require_user`** (the `Depends(...)` we saw in
   line 124). This calls Open WebUI's `/api/v1/auths/` endpoint to
   verify the token. On success, `me: AuthedUser` gets populated
   with the user's email + role.

4. **`chat_completions` runs** (line 124). It validates the model
   name against `VIRTUAL_MODELS` (line 67), splits messages out of
   the payload, and — because `stream=true` — calls
   `_stream_via_pipeline()` wrapped in a `StreamingResponse` (a
   FastAPI primitive that holds the HTTP connection open and pushes
   bytes as the inner generator yields them — we'll see the frame
   format at the end of this section).

5. **`_stream_via_pipeline()`** (line 240) does the routing:
   `audrey_deep` / `audrey_cloud` / `audrey_local` always go through
   the deep panel; `audrey_fast` always uses the fast path;
   `audrey_auto` decides based on prompt length. For our "what is
   BTRFS?" example with `audrey_auto`, the prompt is short — fast path
   wins.

6. **The fast path runs through the LangGraph nodes:**
   - `datetime` adds an ISO-8601 timestamp to the message stack so the
     model knows what year it is.
   - `memory_recall` checks if this user has stored memories matching
     the prompt (e.g. "user's hardware") — none for this query.
   - `classify` decides this is a `general` task (not code, reasoning,
     or vision).
   - `complexity` measures the prompt at a small token count —
     well under the configured threshold — and routes to `fast_path`.
   - `fast_path` picks the best healthy local or cloud model for
     `general` tasks (e.g. `qwen3.6:35b`) and asks it.
   - The model is "tool-capable" so it could call `web_search` or
     `kb_search` if it wanted. For this question it doesn't bother —
     it knows what BTRFS is.

7. **The model's tokens stream back** through `_stream_openai` (line
   809), which converts Ollama's chunks into OpenAI-format SSE frames
   and yields them. FastAPI passes each frame through to OWUI as it's
   produced. The user sees the answer typing itself out.

8. **The `reflect` node runs** at the end to check the answer met a
   minimum length. If not, it might retry with the deep panel. For our
   case, the answer is fine.

9. **`pipeline_total` metric increments** (line 181), the request
   completes, the connection closes.

That's it. Every Audrey request is some variation of those nine steps.

The streaming protocol Audrey speaks is **SSE (Server-Sent Events)** —
the server keeps the HTTP connection open and pushes text-formatted
"frames" (each starts with `data: ` and ends with a blank line) until
done. OpenAI's chat-completion streaming is SSE; FastAPI's
`StreamingResponse` is how Audrey produces it. Lesson 14 walks the
exact frame format.

### 2.5 Where the heavy lifting lives

To fix our map: When something goes wrong or you want to understand
something deeply, here's where to look first.

| If you're asking… | Look in… |
|---|---|
| "Where does a request enter Audrey?" | [`routes/openai.py:124`](../../src/audrey/routes/openai.py#L124) |
| "How does the pipeline decide what to do?" | [`pipeline/graph.py:376`](../../src/audrey/pipeline/graph.py#L376) (the graph topology) |
| "Why did it pick model X?" | [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py) + [`models/registry.py`](../../src/audrey/models/registry.py) |
| "Why did the request hang?" | [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) (GPU queue) + Ollama logs |
| "How did the answer get streamed?" | [`routes/openai.py:401`](../../src/audrey/routes/openai.py#L401) (`_stream_deep_with_banners`) |
| "Where do tools get called?" | [`pipeline/react.py`](../../src/audrey/pipeline/react.py) |

Bookmark this. You'll come back.


## 3. Comprehension Q&A

Try answering each question yourself before reading the answer.

**1. What's the difference between `async def` and `def` in this
codebase? Why is every route handler `async`?**

`def` declares a regular synchronous function: When Python calls it,
it runs to completion before returning, and during any waiting (a
network call, a disk read) the event loop is frozen and no other
in-flight request can make progress. `async def` declares a
coroutine: It can use `await` to pause itself while waiting on I/O,
hand control back to the event loop, and resume when the I/O is
done.

Every route handler is `async` because Audrey is a gateway — most of
its time is spent waiting for Ollama, OWUI, Qdrant, or custom-tools
to reply. If a route were synchronous, one user's 30-second deep-
panel call would freeze every other user's request behind it. With
`async def` and `await`, those waits yield control, and other users'
requests make progress on the same Python process at the same time.
See Lesson 1 §1.

**2. Walk me through what `Depends(require_user)` does before
`chat_completions` runs. Where does the user's identity come from?**

When FastAPI dispatches a request to `chat_completions`, it sees the
`me: AuthedUser = Depends(require_user)` parameter and runs
`require_user()` *first* — the route function itself doesn't run
yet. `require_user` reads the `Authorization: Bearer <token>` header
from the incoming HTTP request, then makes an httpx call out to
Open WebUI's `/api/v1/auths/` endpoint asking "who owns this token?"

If the token is missing, malformed, or rejected by OWUI, the
dependency raises `HTTPException(401)` and FastAPI returns 401 to
the client without `chat_completions` ever being called. If OWUI
responds with the user's record, `require_user` builds an
`AuthedUser` dataclass (email + role + owui_id) and FastAPI passes
it in as `me`.

So the user's identity is sourced from OWUI's session database,
keyed by the bearer token the frontend includes on every request.
Audrey itself doesn't store credentials.

**3. In the LangGraph topology, what's the difference between
`add_edge` and `add_conditional_edges`?**

`add_edge("a", "b")` is *unconditional*: After node `a` finishes,
node `b` runs next, every time. It's the "go straight here" arrow
in the graph.

`add_conditional_edges("a", router_fn, {...})` lets the next node
depend on the current state. LangGraph runs `router_fn(state)`,
which returns a string; that string is looked up in the dictionary
to find the next node's name. It's the branching arrow.

In Lesson 4's graph, **`g.add_edge("datetime", "memory_recall")`**
is unconditional — after the datetime stamp gets injected,
memory_recall always runs next. **`g.add_conditional_edges("complexity",
route_after_complexity, {"fast": "fast_path", "deep": "planner"})`**
is conditional — the `route_after_complexity` function inspects
state to decide whether the request takes the fast path or the deep
panel.

**4. A request comes in for `audrey_fast` with a short prompt. List
the nodes it visits, in order.**

`datetime` → `memory_recall` → `classify` → `complexity` →
`fast_path` → END.

The deep-panel branch (`planner` → `deep_panel` → `synthesize` →
`reflect`) doesn't run because `complexity` routed to "fast", and
the escalate branch doesn't run because `fast_path` returned "ok"
rather than "escalate". Six function calls total, all chained by
LangGraph based on the topology in `graph.py`.

**5. `_stream_via_pipeline` and `_generate_via_pipeline` both end up
running the same pipeline. What's the difference between them?**

They differ in how the answer is delivered to the client, not in
how it's computed.

`_generate_via_pipeline` runs the whole pipeline, waits for the
final answer, packages it into one OpenAI-shaped JSON response
(`{"choices": [{"message": {...}}]}`), and returns that as the
HTTP response body. The client gets a single response only after
generation is fully done.

`_stream_via_pipeline` runs the same pipeline but returns a
`StreamingResponse` — instead of buffering the answer, it pushes
each token (and progress banners like "thinking…") to the client
as a Server-Sent Events stream of `data: {...}\n\n` frames. The
client sees text appear character-by-character, the way OWUI's
chat UI does.

Same pipeline graph, same nodes, same tools — different *response
shape*. The split happens in `chat_completions` based on
`payload.stream`.

**6. What happens if Ollama is down when a request arrives? Where
does the failure surface, and what HTTP status does the user see?**

For the non-streaming path: `_generate_via_pipeline` runs the
pipeline, one of the model-calling nodes tries to reach Ollama,
and the underlying httpx call fails. Audrey's Ollama client wrapper
catches the httpx error and re-raises it as the project's own typed
`OllamaError`. That exception propagates up through the pipeline
to the route handler, which catches it explicitly (see
[`routes/openai.py:218-222`](../../src/audrey/routes/openai.py#L218))
and converts it to **HTTP 502 Bad Gateway** with a JSON error body
explaining the upstream failure:

```python
try:
    final = await _run_graph_with_metrics(graph, state)
except OllamaError as e:
    raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e
```

502 is the right code: From the client's point of view, Audrey is
the gateway, and the *next* server up the chain (Ollama) is the
one that failed.

The streaming path is messier because the response headers (`200
OK`) have already been sent to the client by the time generation
starts — you can't change a status code mid-stream. The streaming
path has to encode the failure as an in-stream error frame instead.
We'll trace that in detail when we cover the streaming route.

## When you're ready for the next lesson

The next lesson covers `main.py` + `config.py` + `compose.yaml`:
[Lesson 5 - Configuration and startup](lesson-05-configuration-and-startup.md).
We'll learn how the app boots, what `app.state.*` is, how config flows
from a YAML file into running code, and what the lifespan context
manager does. That lesson is where we properly meet FastAPI dependency
injection + async lifecycle.
