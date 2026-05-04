# Lesson 2 — The request lifecycle, end-to-end

**Estimated time:** 45-60 minutes (read 25, walk through code 20, audit
+ questions 15).

**Prerequisite:** [Lesson 1](lesson-01-foundations.md). This lesson
assumes you've already met `async`/`await`, FastAPI, Pydantic, and
LangGraph in the abstract. If `async def foo(): await bar()` doesn't
look meaningful to you, go back to Lesson 1 first.

**Goal:** by the end of this lesson, you can answer the question
*"what happens between the user clicking 'send' in Open WebUI and the
answer appearing on screen?"* — at a level where every term has a
referent in your head, even if you don't yet know how each piece is
implemented.

We're not going deep on any single component today. The point is to
build a **map**. Subsequent lessons fill in each region in detail.

---

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

If you read Lesson 1 those names should be familiar in the abstract —
this lesson is where you see them touch each other in real code.

---

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

This is where every request lands. Read the function (lines 124-159)
top to bottom. Notice:

- **`@router.post("/chat/completions")`** — a FastAPI decorator. It
  registers this function as the handler for `POST /v1/chat/completions`
  HTTP requests. The `/v1` prefix is set on line 63 (`router = APIRouter(prefix="/v1")`).
- **`payload: ChatCompletionRequest`** — FastAPI parses the incoming
  JSON body into a `ChatCompletionRequest` object (defined at line 84).
  If the JSON doesn't match, FastAPI returns 422 automatically before
  your code runs.
- **`me: AuthedUser = Depends(require_user)`** — FastAPI's dependency
  injection. Before `chat_completions` runs, FastAPI calls
  `require_user()` (in `auth.py`), which checks the `Authorization:
  Bearer <token>` header against Open WebUI. If it fails, the user
  gets 401 and `chat_completions` never runs. If it succeeds, `me`
  contains their email + role.
- **The `if payload.stream:` branch (lines 153-157)** — OpenAI's API
  supports two modes: streaming (server pushes tokens as they're
  generated) and non-streaming (server waits, returns full answer in
  one JSON response). Audrey supports both. Streaming is what OWUI
  uses; non-streaming is what programmatic clients use.

So the request, after auth and JSON parsing, splits into one of two
paths: **`_stream_via_pipeline`** or **`_generate_via_pipeline`**.
Both eventually call the same LangGraph pipeline; they differ in *how
the result is returned to the client*.

### 2.2 The pipeline graph

Open [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py).
Scroll to **line 371**:

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
(grep `def node_` — you'll see them at lines 123, 133, 166, 183, 203,
230, 247, 279, 298, 364).

The graph topology — which node runs after which — is on **lines
383-402**:

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

Look at [`graph.py:123`](../../src/audrey/pipeline/graph.py#L123) —
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
— see [Lesson 1 §6](lesson-01-foundations.md#6-typed-dictionaries-typeddict)
if that's unfamiliar. The short version: it's a regular dict where the
allowed keys and their types are declared up front, so the type checker
can catch typos and editors can autocomplete.

### 2.4 The request flow, end to end

Now we can trace one request all the way through. I'll narrate; you
follow along in the files.

**A user types "what is BTRFS?" in Open WebUI and hits send.**

1. **OWUI sends `POST /v1/chat/completions`** with a JSON body
   containing the model name (e.g. `audrey_auto`), the message
   history, and `"stream": true`. The Authorization header carries
   the user's bearer token from their OWUI login.

2. **Cloudflared routes it to the audrey-ai container.** (We won't
   touch Cloudflare in this course; that's container infra, see
   `docs/campaign-1/`.) Inside the container, FastAPI receives the
   request.

3. **FastAPI runs `require_user`** (the `Depends(...)` we saw on line
   128). This calls Open WebUI's `/api/v1/auths/` endpoint to verify
   the token. On success, `me: AuthedUser` gets populated with the
   user's email + role.

4. **`chat_completions` runs** (line 124). It validates the model name
   against `VIRTUAL_MODELS` (line 67), splits messages out of the
   payload, and — because `stream=true` — calls
   `_stream_via_pipeline()` wrapped in a `StreamingResponse`.

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
   - `complexity` measures the prompt at maybe 50 tokens — well under
     the 500-token threshold — and routes to `fast_path`.
   - `fast_path` picks the best healthy local or cloud model for
     `general` tasks (e.g. `qwen3.6:35b`) and asks it.
   - The model is "tool-capable" so it could call `web_search` or
     `kb_search` if it wanted. For this question it doesn't bother —
     it knows what BTRFS is.

7. **The model's tokens stream back** through `_stream_openai` (line
   791), which converts Ollama's chunks into OpenAI-format SSE frames
   and yields them. FastAPI passes each frame through to OWUI as it's
   produced. The user sees the answer typing itself out.

8. **The `reflect` node runs** at the end to check the answer met a
   minimum length. If not, it might retry with the deep panel. For our
   case, the answer is fine.

9. **`pipeline_total` metric increments** (line 182), the request
   completes, the connection closes.

That's it. Every Audrey request is some variation of those nine steps.

The streaming protocol Audrey speaks is **SSE (Server-Sent Events)** —
the server keeps the HTTP connection open and pushes text-formatted
"frames" (each starts with `data: ` and ends with a blank line) until
done. OpenAI's chat-completion streaming is SSE; FastAPI's
`StreamingResponse` is how Audrey produces it. Lesson 12 walks the
exact frame format.

### 2.5 Where the heavy lifting lives

To fix our map: when something goes wrong or you want to understand
something deeply, here's where to look first.

| If you're asking… | Look in… |
|---|---|
| "Where does a request enter Audrey?" | [`routes/openai.py:124`](../../src/audrey/routes/openai.py#L124) |
| "How does the pipeline decide what to do?" | [`pipeline/graph.py:371`](../../src/audrey/pipeline/graph.py#L371) (the graph topology) |
| "Why did it pick model X?" | [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py) + [`models/registry.py`](../../src/audrey/models/registry.py) |
| "Why did the request hang?" | [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) (GPU queue) + Ollama logs |
| "How did the answer get streamed?" | [`routes/openai.py:392`](../../src/audrey/routes/openai.py#L392) (`_stream_deep_with_banners`) |
| "Where do tools get called?" | [`pipeline/react.py`](../../src/audrey/pipeline/react.py) |

Bookmark this. You'll come back.

---

## 3. Audit notes

This is the first lesson, so the audit is mostly "does this codebase's
front door make sense?" Detailed audits start in Lesson 2.

### `nit` — the `_options_from_request` helper is duplicated logic

[`routes/openai.py:352-360`](../../src/audrey/routes/openai.py#L352)
defines `_options_from_request`, and
[`pipeline/graph.py:408-416`](../../src/audrey/pipeline/graph.py#L408)
defines a near-identical `_options_from_state`. They're not literally
the same function (one reads from a Pydantic object, the other from a
dict), but they do the same conceptual mapping. Worth knowing they
exist as a pair; not worth fixing. **No action proposed.**

### `consider` — the "VIRTUAL_MODELS validation" lives in the route, not the schema

The route checks `if payload.model not in VIRTUAL_MODELS:` on line 131.
This could be expressed in the Pydantic schema with `Literal[...]`,
which would push the 400 to FastAPI's automatic validation layer. It
would also make the OpenAPI spec self-document the supported models.

**Tradeoff:** the current approach lets us emit a more descriptive
error message ("Unknown model X. Supported: [...]") than Pydantic's
default. Probably worth keeping the manual check for that reason.
**No action proposed; flagging because you'll see this pattern again.**

### `consider` — what if a streaming client disconnects mid-pipeline?

We don't yet know how cancellation propagates from the HTTP layer down
to the LangGraph nodes (or to the Ollama call inside a node). Will the
deep-panel workers keep running and burn cloud-time after the user
hits stop? Phase 18 mentions handling `asyncio.CancelledError` in the
streaming route, but it's not obvious from this lesson whether
mid-stream cancellation actually frees the resources downstream.

**This is a Lesson 12 question** when we cover the streaming route in
detail. Filing it now.

---

## 4. Comprehension questions

Answer these out loud (or in writing) before moving to Lesson 2. If
any feel hard, that's the signal — re-read the relevant section.
Don't grade yourself; just notice the gaps.

1. **What's the difference between `async def` and `def` in this
   codebase?** Why is every route handler `async`?

2. **Walk me through what `Depends(require_user)` does** before
   `chat_completions` runs. Where does the user's identity come from?

3. **In the LangGraph topology, what's the difference between
   `add_edge` and `add_conditional_edges`?** Give an example of each.

4. **A request comes in for `audrey_fast` with a 50-token prompt.
   List the nodes it visits, in order.** (You don't need to know what
   each node does in detail — just trace the graph from
   `set_entry_point` through to `END`.)

5. **`_stream_via_pipeline` and `_generate_via_pipeline` both end up
   running the same pipeline.** What's the difference between them?

6. **(Stretch)** **What happens if Ollama is down when a request
   arrives?** Where does the failure surface to the user, and what
   HTTP status do they see? Hint: scan
   [`routes/openai.py:200-204`](../../src/audrey/routes/openai.py#L200)
   for the non-streaming path. The streaming path is more subtle —
   we'll cover it later.

---

## When you're ready for Lesson 3

Reply with anything from "ready" to "I'm stuck on question N" to
"actually back up, what does X mean?" — all are useful signals.

Lesson 3 covers `main.py` + `config.py` + `compose.yaml`. We'll learn
how the app boots, what `app.state.*` is, how config flows from a YAML
file into running code, and what the lifespan context manager does.
That lesson is where we properly meet FastAPI dependency injection +
async lifecycle.
