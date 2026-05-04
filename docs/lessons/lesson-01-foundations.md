# Lesson 1 — Foundations: the tools you'll meet in this codebase

**Estimated time:** 90-120 minutes if you read carefully and try the
examples. You can split it across sessions — sections are
self-contained.

**Goal:** before we touch the Audrey codebase proper, you should know
what each major tool *is*, what problem it solves, and recognize its
shape when you see it. This is **not** a Python tutorial — you already
know the language basics. It's a tour of the libraries and patterns
this project uses.

You don't need to memorize anything. Skim what's already familiar,
read carefully what's new. Each section ends with a "where you'll see
this in Audrey" pointer so the abstract idea has a concrete home.

**What this lesson covers:**

1. [`async`/`await` and `asyncio`](#1-asyncawait-and-asyncio) — the
   model for everything else
2. [FastAPI](#2-fastapi) — the web framework
3. [Pydantic](#3-pydantic) — data validation + typed models
4. [LangGraph](#4-langgraph) — pipeline-as-graph orchestration
5. [Type hints, dataclasses, TypedDict](#5-type-hints-dataclasses-typeddict)
6. [Typed dictionaries (TypedDict)](#6-typed-dictionaries-typeddict) — the
   pipeline-state pattern
7. [Context managers](#7-context-managers) — the `with` statement
8. [httpx](#8-httpx) — the async HTTP client
9. [Vector search: Qdrant + embedding models](#9-vector-search-qdrant--embedding-models)
10. [Prometheus metrics](#10-prometheus-metrics)
11. [pytest](#11-pytest)

---

## 1. `async`/`await` and `asyncio`

This is the most important section. Everything else builds on it.

### The problem

Imagine a function that fetches a web page:

```python
def fetch(url: str) -> str:
    response = requests.get(url)   # ← waits for network
    return response.text
```

When Python hits `requests.get(url)`, it stops. The CPU sits there
doing nothing while the bytes travel from the server. If three users
hit your app at the same time, requests two and three wait in line
behind request one — even though the CPU is mostly idle the whole
time. This is "blocking I/O."

For Audrey, this would be catastrophic. A single chat completion can
take 30 seconds (or three minutes for a deep-panel cloud call). If we
blocked, we'd handle one request at a time. With 5 users, average wait
time would balloon.

### The solution

`async`/`await` is Python's way to **pause one function while it's
waiting**, hand control back to a scheduler, run something else, and
resume the first function when its wait is done. The scheduler is
called the **event loop**.

The same fetch function, async-style:

```python
async def fetch(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)   # ← yields control while waiting
    return response.text
```

Two new keywords:

- **`async def`** — declares an async function (also called a
  *coroutine*). Calling `fetch(url)` doesn't run it; it returns a
  coroutine object. To actually run it, something has to `await` it
  or schedule it on the event loop.
- **`await`** — used inside async functions. `await x` says: "if `x`
  is not done yet, pause me, let other tasks run, and resume me when
  `x` finishes." You can only `await` things that are async-aware
  (coroutines, futures, async generators).

### What it does NOT do

- **It doesn't make code faster on the CPU.** Async helps when you're
  waiting on I/O (network, disk, another process). A pure-CPU loop
  (`for i in range(10**9)`) is no faster async — it just blocks the
  event loop.
- **It's not parallelism in the multi-CPU sense.** A single Python
  event loop runs on one thread, one task at a time. The trick is
  that "running" includes "waiting" — and we just don't wait.
- **You can't mix sync and async carelessly.** Calling a regular
  blocking function from an `async def` blocks the whole event loop —
  every other task stalls until it returns. Audrey hits this with
  `Path.exists()` in the KB reconciler (the `ASYNC240` ruff warnings
  you may have seen).

### Tiny working example

If you want to see this for yourself, run this in a shell:

```python
import asyncio

async def slow_task(name: str, seconds: float) -> str:
    print(f"{name}: starting")
    await asyncio.sleep(seconds)   # like a network call
    print(f"{name}: done")
    return name

async def main():
    # Run three slow tasks concurrently. Total time = ~3s, not 6s.
    results = await asyncio.gather(
        slow_task("A", 1.0),
        slow_task("B", 2.0),
        slow_task("C", 3.0),
    )
    print(results)

asyncio.run(main())
```

Output (timestamps elided):

```
A: starting
B: starting
C: starting
A: done
B: done
C: done
['A', 'B', 'C']
```

All three started immediately. Because each `await asyncio.sleep(...)`
yields control back to the event loop, the loop ran the others while
the first was waiting. Total runtime is ~3 seconds (the longest), not
6 (the sum).

### Where you'll see it in Audrey

**Everywhere.** Every route handler in
[`src/audrey/routes/`](../../src/audrey/routes/), every pipeline node
in [`src/audrey/pipeline/`](../../src/audrey/pipeline/), every Ollama
call. The whole app runs on one event loop. When request A is waiting
for Ollama to generate tokens, request B is making progress in
parallel.

**One pattern worth recognizing right now:**

```python
async with gate.acquire(model, location="local"):
    response = await ollama.chat(...)
```

That's an *async context manager* (the `async with`) wrapping an
*async function call* (the `await`). We'll meet `with` properly in
[§7](#7-context-managers); for now know that `async with` is the
async-aware version.

### What to ignore for now

- `asyncio.Task` vs coroutines vs futures — there's a hierarchy here
  but day-to-day you write `await thing()` and don't think about it.
- `loop.run_in_executor` — used for "I have to call a blocking
  function from async code." We use the simpler `asyncio.to_thread`
  in Audrey when needed.
- `async generators` — functions with `async def` + `yield` instead
  of `return`. You'll meet them in `pipeline/synthesize.py`
  (`synthesize_stream`); same idea as a normal generator but with
  awaits in between.

---

## 2. FastAPI

A web framework for Python. "Web framework" = a library that makes it
easy to write a program that responds to HTTP requests.

### The shape

You write functions, decorate them, FastAPI turns them into HTTP
endpoints:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

That's it. Run that file with `uvicorn` (an ASGI server) and `GET
/health` returns `{"status": "ok"}`.

### What FastAPI gives you

Three things, all valuable:

**(1) Decorator routing.** `@app.get(...)`, `@app.post(...)`,
`@app.delete(...)` — decorators that map URL paths to functions. You
also get path parameters and query parameters automatically:

```python
@app.get("/users/{user_id}")
async def get_user(user_id: str, include_email: bool = False) -> dict:
    return {"id": user_id, "include_email": include_email}
```

`{user_id}` in the URL becomes the `user_id` parameter.
`include_email` comes from the query string (`/users/123?include_email=true`).
FastAPI parses, type-converts, and validates all of these.

**(2) Automatic JSON parsing + validation.** Declare a Pydantic model
(see [§3](#3-pydantic)) as a parameter and FastAPI parses the request
body into it, returning 422 automatically if it's malformed:

```python
class CreateUser(BaseModel):
    email: str
    role: str

@app.post("/users")
async def create_user(payload: CreateUser) -> dict:
    return {"created": payload.email}
```

You never write JSON-parsing code. Or schema-validation code.

**(3) Dependency injection.** Functions can declare *dependencies* via
the `Depends(...)` mechanism. FastAPI runs the dependency first and
passes the result to your function:

```python
def get_db():
    return DatabaseConnection()

@app.get("/items")
async def list_items(db = Depends(get_db)) -> list:
    return db.query("SELECT ...")
```

This is how Audrey enforces auth on every protected route. You write:

```python
@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    me: AuthedUser = Depends(require_user),
):
    ...
```

`require_user` is a function that validates the `Authorization`
header. If it fails it raises `HTTPException(401)` and the route never
runs. If it succeeds, `me` contains the user's identity. Concise +
hard to forget.

### Request/response lifecycle (mental model)

```
incoming HTTP
    │
    ▼
parse path/query/body → validate against type hints + Pydantic models
    │
    ▼
run dependencies (Depends(...) chain)
    │
    ▼
call the route function with all parameters
    │
    ▼
serialize the return value to JSON (or stream it)
    │
    ▼
outgoing HTTP
```

If anything raises `HTTPException(status, detail)`, FastAPI turns it
into the appropriate HTTP response and short-circuits the rest.

### Where you'll see it in Audrey

- Route definitions live in [`src/audrey/routes/`](../../src/audrey/routes/) —
  one file per concern (chat completions, file uploads, admin, KB,
  etc.).
- The app object itself is built in
  [`src/audrey/main.py`](../../src/audrey/main.py).
- Auth uses `Depends(require_user)` and `Depends(require_admin)` —
  defined in [`src/audrey/auth.py`](../../src/audrey/auth.py).

### What to ignore for now

- **Middleware** — FastAPI supports it; Audrey deliberately doesn't
  use much. The `Depends` pattern covers what we need.
- **Background tasks** — there's a `BackgroundTasks` parameter you can
  inject. We don't use it — we use long-lived asyncio tasks started in
  the lifespan instead (the watcher, the reconciler).
- **WebSockets** — supported, unused. We use SSE (one-way streaming)
  instead.

---

## 3. Pydantic

A library for **data validation using Python type hints**. You declare
what data should look like; Pydantic checks that incoming data
matches, with clear error messages when it doesn't.

### The shape

```python
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
```

Now `ChatMessage(role="user", content="hello")` returns a validated
instance. `ChatMessage(role=123, content="hello")` raises a
`ValidationError` because `role` should be a string.

When this class is used as a FastAPI route parameter, Pydantic does
the validation for you and FastAPI returns 422 to the client on
failure — you never write validation code yourself.

### Why it matters

The boundary between "stuff arriving from the network" and "stuff your
code can trust" is one of the highest-bug-density places in any
program. Pydantic moves that boundary into a single declarative
schema, so you can read the schema and know exactly what you accept.

It's also runtime-checked, not just type-checked — so even if a client
sends garbage that confuses the type checker, Pydantic catches it.

### Slightly fuller example

```python
from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
```

That schema enforces:

- `model` is a string (required).
- `messages` is a list with at least one item, each a `ChatMessage`.
- `stream` is a bool, defaulting to false.
- `temperature` is either missing/null, or a float between 0 and 2.

A client sending `{"model": "x", "messages": []}` gets back a 422 with
a precise error: `messages: List should have at least 1 item`.

### Where you'll see it in Audrey

- [`src/audrey/routes/openai.py:78-100`](../../src/audrey/routes/openai.py#L78) —
  `ChatMessage` and `ChatCompletionRequest` schemas.
- [`src/audrey/routes/admin.py`](../../src/audrey/routes/admin.py) —
  response models like `AuthClearResponse`.
- [`src/audrey/config.py`](../../src/audrey/config.py) — Pydantic
  Settings models for env-var loading.
- [`tools-server/app.py`](../../tools-server/app.py) — every endpoint
  has a request schema.

### What to ignore for now

- **Validators** (`@field_validator`, `@model_validator`) — custom
  per-field checks. Audrey rarely uses them; the field-type hints +
  `Field(ge=, le=, min_length=, ...)` constraints cover what we need.
- **Config** (`model_config = ConfigDict(...)`) — fine-tuning
  serialization, mutation, etc. Unused in Audrey.
- **Pydantic v1 vs v2 syntax differences** — Audrey uses v2. If you
  Google an example and it uses `class Config:` instead of
  `model_config`, that's v1; same idea, different spelling.

---

## 4. LangGraph

A library for **structuring an LLM application as a state machine**.
You declare nodes (functions), edges (which node runs next), and a
shared state object that flows through. LangGraph runs the graph for
you.

### The problem it solves

When you orchestrate multiple LLM calls — classify a request, plan
subtasks, dispatch parallel workers, synthesize the results, retry on
failure — the control flow gets messy fast. You end up with
deeply-nested if/else branches, retry counters scattered across
functions, and no obvious place to look when something goes wrong.

LangGraph's pitch: extract the control flow into a **graph** you can
look at. Each node does one thing. Each edge says "after node A, run
node B" or "after node A, decide based on state which of B/C/D to
run." The state object accumulates results as the graph runs.

### The shape

A minimal LangGraph:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    user_input: str
    classification: str
    answer: str

async def classify(state: State) -> dict:
    is_question = "?" in state["user_input"]
    return {"classification": "question" if is_question else "statement"}

async def respond(state: State) -> dict:
    if state["classification"] == "question":
        return {"answer": "good question!"}
    return {"answer": "interesting!"}

g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("respond", respond)
g.set_entry_point("classify")
g.add_edge("classify", "respond")
g.add_edge("respond", END)

graph = g.compile()
result = await graph.ainvoke({"user_input": "what time is it?"})
# result == {"user_input": "what time is it?", "classification": "question", "answer": "good question!"}
```

Three concepts:

- **State** — a `TypedDict` (see [§6](#6-typed-dictionaries-typeddict)).
  Each node reads from it and returns a dict of fields to update. The
  graph merges the returned dict into the state.
- **Nodes** — async functions. One responsibility each.
- **Edges** — the topology. `add_edge("a", "b")` means "after a runs,
  run b." `add_conditional_edges` lets you branch based on state.

### Conditional edges (the interesting part)

```python
def route_after_classify(state: State) -> str:
    return "answer_question" if state["classification"] == "question" else "make_statement"

g.add_conditional_edges("classify", route_after_classify, {
    "answer_question": "question_node",
    "make_statement": "statement_node",
})
```

The router function returns a string; the dict maps that string to a
node name. This is how Audrey routes between fast path and deep
panel based on prompt complexity.

### Where you'll see it in Audrey

- [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)
  is the entire graph definition. **One file. ~430 lines. The whole
  pipeline lives there.**
- The state schema is
  [`src/audrey/pipeline/state.py`](../../src/audrey/pipeline/state.py).
- Each node implementation lives next to the graph definition (in the
  same file) — they're small wrappers around the real logic in
  `classify.py`, `fast_path.py`, `deep_panel.py`, etc.

### What to ignore for now

- **Checkpointing** — LangGraph supports persisting state to a DB so
  you can resume a graph mid-run. Audrey doesn't use it; pipelines
  run start-to-finish in one process.
- **Subgraphs** — graphs nested inside graphs. Audrey is a single
  top-level graph.
- **The streaming API** — LangGraph has its own streaming protocol;
  Audrey's streaming bypasses LangGraph entirely (we re-implement the
  pipeline imperatively in `_stream_deep_with_banners` because we
  need finer-grained control over when banners flush). We'll cover
  this fully in Lesson 12.

---

## 5. Type hints, dataclasses, TypedDict

Modern Python encourages annotating types. Audrey leans into this
heavily because it makes the codebase easier to read and lets editors
+ `mypy` catch mistakes before runtime.

### Type hints

Just annotations on parameters and return values:

```python
def greet(name: str, repeat: int = 1) -> list[str]:
    return [f"hello {name}"] * repeat
```

Python doesn't enforce these at runtime — they're hints. But:

- Editors use them for autocomplete + warnings.
- `mypy` (a separate tool) checks them statically.
- FastAPI + Pydantic use them at runtime to validate.
- Future readers use them to understand intent.

You'll see these everywhere in Audrey:

- `dict[str, Any]` — a dictionary with string keys and arbitrary
  values.
- `list[str]` — a list of strings.
- `str | None` — a string or `None` (the `|` is "or"; new in 3.10+).
- `Literal["a", "b"]` — must be exactly one of those strings.

### Dataclasses

A normal Python class is verbose:

```python
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
```

A dataclass writes the boilerplate for you:

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

Same thing, less typing. You also get `__repr__`, `__eq__`, and
optionally `__hash__` for free.

`@dataclass(slots=True)` is what Audrey uses — adds `__slots__` so
instances use less memory and attribute access is slightly faster.
Mostly cosmetic; included because the codebase has many small data
holders.

`@dataclass(frozen=True)` makes instances immutable. Used in places
like `KeywordSignal` in `pipeline/classify.py`.

### TypedDict (briefly — full treatment in §6)

A dict where the keys are declared in advance. Acts like a regular
dict at runtime (`state["x"]`, not `state.x`) but the type checker
knows what keys exist and what their types are. We'll meet this
properly in the next section.

### Where you'll see it in Audrey

Dataclasses:
- `AuthedUser` in [`src/audrey/auth.py`](../../src/audrey/auth.py)
- `WorkerDraft` in [`src/audrey/pipeline/state.py`](../../src/audrey/pipeline/state.py)
- `ReactResult` in [`src/audrey/pipeline/react.py`](../../src/audrey/pipeline/react.py)
- `KBHit` in [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py)

Type hints: every function in the codebase.

### What to ignore for now

- **Generics** (`TypeVar`, `Generic[T]`) — you'll see `list[T]`,
  `dict[K, V]` syntax everywhere; declaring your *own* generic types
  is an advanced topic Audrey rarely needs.
- **Protocols** (structural typing) — Audrey doesn't use them.
- **`@dataclass(eq=False, order=True, ...)`** options — you can tweak
  what dataclass generates. Almost always the defaults are fine.

---

## 6. Typed dictionaries (TypedDict)

LangGraph uses `TypedDict` for state. Worth its own section because
it's slightly weird if you haven't seen it.

### The shape

```python
from typing import TypedDict

class PipelineState(TypedDict, total=False):
    virtual_model: str
    messages: list[dict]
    task_type: str
    # ... more fields
```

That looks like a class, but at runtime it's just a regular `dict`.
There's no `PipelineState(...)` constructor. You build instances as
`{"virtual_model": "...", ...}` and access them as `state["x"]`.

### Why use it

You get the **type checker's** view of a dict's structure without
changing the runtime. So if you write `state["task_type"]`, the
editor knows the value is a string. If you write `state["taks_type"]`
(typo), the type checker can flag it before you run the code.

The `total=False` part says "all fields are optional" — useful for
LangGraph because nodes only set the fields they care about.

### Why not just use a dataclass?

LangGraph specifically wants dicts (it merges node return values into
the state with `dict.update(...)` semantics). A dataclass is a class,
not a dict, and would need translation. TypedDict gives you the type
checking of a dataclass with the runtime behavior of a dict.

### Where you'll see it in Audrey

- [`src/audrey/pipeline/state.py`](../../src/audrey/pipeline/state.py)
  defines `PipelineState` and `WorkerDraft` (yes, `WorkerDraft` is
  TypedDict, not dataclass — because it lives inside the LangGraph
  state).

### What to ignore for now

- **Required vs NotRequired** (3.11+) — you can mark individual
  fields as required while leaving the class `total=False`. Audrey
  doesn't use this.
- **Inheritance** — TypedDicts can inherit from each other. We don't.

---

## 7. Context managers

The `with` statement and its async variant `async with`. Used for
**resource management** — opening something, doing work with it,
guaranteeing cleanup happens even on errors.

### The shape

```python
with open("file.txt") as f:
    content = f.read()
# `f` is closed here, even if read() raised
```

The `with` block guarantees that whatever cleanup logic the resource
needs runs when you leave the block — by normal exit, by exception,
or even by `return`/`break`/`continue`.

### Why this matters

Without `with`, you'd write:

```python
f = open("file.txt")
try:
    content = f.read()
finally:
    f.close()
```

Same effect, three times the typing, easy to forget.

### Async context managers

Same idea, async-aware. Used when "open" or "close" need to await:

```python
async with httpx.AsyncClient() as client:
    response = await client.get(url)
# client is gracefully closed here
```

### Custom context managers

You can write your own:

```python
from contextlib import contextmanager

@contextmanager
def timing(label: str):
    start = time.monotonic()
    try:
        yield
    finally:
        print(f"{label}: {time.monotonic() - start:.2f}s")

with timing("query"):
    result = expensive_thing()
```

The `yield` is where the `with` block runs. Code before the `yield`
is the setup; code after is the teardown.

For async, use `@asynccontextmanager` from the same module.

### Where you'll see it in Audrey

- `async with httpx.AsyncClient(...) as client:` — every place we
  make an outbound HTTP call.
- `async with gate.acquire(...)` in
  [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) —
  custom async context manager. Acquiring the GPU slot on enter,
  releasing on exit.
- `async with PhaseTicker(...)` in
  [`pipeline/banners.py`](../../src/audrey/pipeline/banners.py) —
  custom async context manager. Starts a "tick the dots" task on
  enter, cancels it on exit.
- `@asynccontextmanager async def lifespan(app: FastAPI):` in
  [`main.py`](../../src/audrey/main.py) — FastAPI's lifecycle hook.
  Code before `yield` runs at startup; after `yield` runs at
  shutdown.

### What to ignore for now

- `contextlib.ExitStack` — for managing many context managers
  dynamically. Audrey doesn't use it.
- The full protocol (`__enter__` / `__exit__` /
  `__aenter__` / `__aexit__`) — you can write context managers as
  classes too, but `@contextmanager` is enough for almost everything.

---

## 8. httpx

A modern async-aware HTTP client for Python. Same idea as the
classic `requests` library, but async-native (and with a
sync API if you want it).

### The shape

```python
import httpx

# Sync (rare in Audrey)
response = httpx.get("https://example.com")
print(response.status_code, response.text)

# Async (the common one)
async def fetch():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get("https://example.com")
        return response.json()
```

Why `AsyncClient` and not just `httpx.get`? The client object holds
connection-pool state — reusing a client across multiple requests is
faster than making a fresh connection each time. Use the client
inside an `async with` so the connection pool gets cleaned up.

### Useful methods

- `await client.get(url, headers=..., params=..., timeout=...)`
- `await client.post(url, json={...})`
- `await client.stream("GET", url)` — for streaming response bodies
  in chunks. Used in Audrey's image fetcher to enforce the size cap.

### Errors

- `response.raise_for_status()` raises `httpx.HTTPStatusError` on
  4xx/5xx.
- `httpx.TimeoutException` for timeouts.
- `httpx.ConnectError` / `httpx.NetworkError` / etc. — all subclass
  `httpx.HTTPError`.

### Where you'll see it in Audrey

- [`src/audrey/models/ollama.py`](../../src/audrey/models/ollama.py) —
  the entire Ollama client.
- [`src/audrey/auth.py`](../../src/audrey/auth.py) — OWUI probe.
- [`src/audrey/tools/dispatch.py`](../../src/audrey/tools/dispatch.py) —
  dispatching tool calls to custom-tools.
- [`src/audrey/kb/embed.py`](../../src/audrey/kb/embed.py) — image
  fetch (with the SSRF guards).

### What to ignore for now

- **Event hooks**, **transport customization**, **proxies**, **HTTP/2
  tuning** — all real features, none used.

---

## 9. Vector search: Qdrant + embedding models

Audrey's knowledge base stores **embeddings** (numeric vectors) of
text chunks and images, and answers queries via **vector similarity
search**. Three pieces:

### What's an embedding?

A neural network that turns a piece of text (or an image) into a
fixed-length array of floats. The geometry of these vectors encodes
semantic meaning: similar concepts produce vectors near each other in
space; unrelated concepts produce vectors far apart.

Concretely:

- `nomic-embed-text` produces 768-dimensional float vectors from
  text. Run via Ollama.
- `clip-ViT-B-32` (CLIP) produces 512-dimensional vectors from
  images **or** from text — both end up in the same vector space, so
  you can search images using a text query.

You don't need to understand how the network works. Treat it as: text
in → vector out. Two vectors are "close" if their cosine similarity
is near 1; "far" if near 0 or negative.

### What's Qdrant?

A vector database. Stores points (each = a vector + a payload
dictionary), indexes them so you can find the N nearest vectors to
a query vector quickly, and supports metadata filters (e.g. "find the
nearest 5 vectors *that also have payload.user = 'bart@proton.me'*").

You speak to Qdrant over HTTP via the `qdrant-client` Python library.
Audrey wraps it in [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py).

### How they fit together in Audrey

```
text content ──► nomic-embed-text ──► 768-d vector ──► Qdrant.upsert(point)
                                                             │
                                                             ▼
                                                       kb_text collection

text query ──► nomic-embed-text ──► 768-d query vector ──► Qdrant.search(top_k=5)
                                                                  │
                                                                  ▼
                                                          5 nearest hits + payloads
```

Same flow for images, but with CLIP and the `kb_images` collection.

### Where you'll see it in Audrey

- [`src/audrey/kb/embed.py`](../../src/audrey/kb/embed.py) —
  `TextEmbedder`, `ImageEmbedder`.
- [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py) — the
  Qdrant client wrapper.
- [`src/audrey/kb/ingest.py`](../../src/audrey/kb/ingest.py) — the
  full ingest flow.

### What to ignore for now

- The internal architecture of the embedding networks.
- Qdrant's quantization, sharding, and payload-index tuning.
- The cosine-vs-Euclidean-vs-dot-product debate — Audrey uses cosine
  for both collections and that's fine.

---

## 10. Prometheus metrics

A standard for instrumenting an application with **counters**,
**gauges**, and **histograms** so an external scraper (also called
Prometheus) can collect them periodically and store them as time
series.

### The three metric types

- **Counter** — only goes up (or resets to 0 on restart). "Number of
  requests served." `requests_total.inc()`.
- **Gauge** — goes up and down. "Current cache size." `gauge.set(5)`
  / `.inc()` / `.dec()`.
- **Histogram** — records a distribution of values. "Request latency
  in seconds." `latency_seconds.observe(0.42)`. Internally it's a
  bunch of counters bucketed by value range.

Each metric has a name (`audrey_requests_total`) and optional
**labels** (`method="POST"`, `path="/v1/chat/completions"`). Labels
let you slice the data — but each unique label combination becomes a
separate time series, so high-cardinality labels (like user emails)
explode storage. Audrey is careful about label cardinality
([`metrics.py`](../../src/audrey/metrics.py) has notes).

### How exposition works

Your app exposes a `/metrics` endpoint that returns the current values
in a specific text format:

```
# HELP audrey_requests_total Total requests served
# TYPE audrey_requests_total counter
audrey_requests_total{method="POST",path="/v1/chat/completions"} 142
```

The Prometheus server scrapes this endpoint every N seconds and
stores each line as a data point. Grafana then queries Prometheus to
draw graphs.

### Where you'll see it in Audrey

- [`src/audrey/metrics.py`](../../src/audrey/metrics.py) — every
  metric definition. ~12 metrics total.
- [`src/audrey/main.py`](../../src/audrey/main.py) — the `/metrics`
  endpoint.
- Throughout the codebase: `pipeline_total.labels(...).inc()`,
  `model_seconds.observe(elapsed)`, etc. — instrumentation sites.
- [`monitoring/`](../../monitoring/) — Prometheus + Grafana compose
  files (separate from the audrey-ai container).

### What to ignore for now

- **Summary** metric type (`prometheus_client.Summary`) — like
  histogram but computes quantiles client-side. Audrey only uses
  histograms.
- **Multiprocess mode** — needed when you have multiple Python
  processes serving the same app. Audrey runs on a single uvicorn
  process so this doesn't apply.
- **Push gateway** — for short-lived jobs that can't be scraped.
  Audrey is long-lived; we use the scrape model.

---

## 11. pytest

The standard Python test framework. Discovers files matching
`test_*.py`, finds functions matching `test_*`, runs them, reports
results.

### The shape

```python
# tests/test_math.py
def test_addition():
    assert 2 + 2 == 4

def test_division_by_zero():
    import pytest
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

Run with `pytest tests/`. Output tells you what passed, what failed,
and shows the offending line for failures.

### Useful features

- **`assert` is the gospel.** No special methods like
  `self.assertEqual` (that's `unittest`, the older framework).
  `assert x == y` and pytest gives you a helpful diff on failure.
- **Fixtures** — reusable test setup. `@pytest.fixture` decorates a
  function; tests that take that function's name as a parameter
  receive its return value:
  ```python
  @pytest.fixture
  def db():
      conn = make_test_db()
      yield conn
      conn.close()

  def test_query(db):
      assert db.query("SELECT 1") == [(1,)]
  ```
- **Parametrization** — run the same test against many inputs:
  ```python
  @pytest.mark.parametrize("input,expected", [(2, 4), (3, 9), (4, 16)])
  def test_square(input, expected):
      assert input ** 2 == expected
  ```
  Each input becomes its own reported test case.

### Async tests

Audrey uses `pytest-asyncio` (a plugin) so async tests just work:

```python
async def test_fetch():
    result = await some_async_function()
    assert result == "expected"
```

The plugin's `asyncio_mode = "auto"` config (in `pyproject.toml`)
means any `async def test_*` is auto-detected — you don't need
`@pytest.mark.asyncio`.

### Where you'll see it in Audrey

- [`tests/`](../../tests/) — 6 test files, ~110 tests, all hermetic
  (no live Ollama / Qdrant / OWUI required). Run with
  `.venv/bin/pytest tests/ -q`.
- The test suite itself is what Lesson 12+ uses to demonstrate
  testing patterns.

### What to ignore for now

- **`unittest`** (the stdlib alternative) — Audrey doesn't use it.
- **`tox`** — for testing against multiple Python versions. We pin
  Python 3.12 and don't bother.
- **Mocking with `unittest.mock`** — Audrey uses small hand-rolled
  fakes (see `tests/test_auth.py`) instead, because the surface area
  needed is small.

---

## You're done

That's the toolbox. None of those sections is the whole story for any
single library — each one has a documentation site you could spend a
day on. The point of this lesson was to get you to the place where
when you see `async def chat_completions(payload: ChatCompletionRequest, me: AuthedUser = Depends(require_user)):`
you can read it as:

> "An async function (so it doesn't block other requests). Takes a
> request body parsed and validated by Pydantic into a
> `ChatCompletionRequest`. Also gets `me`, the result of running the
> `require_user` function as a dependency — which validates the auth
> header and returns a typed `AuthedUser` dataclass."

If that sentence makes sense, you're ready for Lesson 2 — where we
walk a real request all the way through Audrey's code.

If parts didn't land, that's expected. Re-read the section that felt
fuzzy, try a quick example in a Python REPL, then come back.

---

## When you're ready for Lesson 2

Reply with anything — "ready," "I want to dig into LangGraph more
first," "what does `await` actually do under the hood," whatever. All
useful signals.

[Lesson 2 — The request lifecycle, end-to-end](lesson-02-request-lifecycle.md)
covers tracing one request from "OWUI sends POST" to "user sees the
answer," touching every major component without going deep on any.
