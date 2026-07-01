# Lesson 2 — Foundations II: the orchestration stack

**Estimated time:** 45-60 minutes if you read carefully and try the
examples. You can split it across sessions — sections are
self-contained.

**Goal:** know what the three libraries that *shape* Audrey's HTTP
surface and pipeline structure each do, what problem they solve, and
recognize their shape when you see them. Lesson 3 covers the
satellite libraries Audrey calls *out to* (httpx, vector search,
Prometheus, pytest).

**What this lesson covers:**

1. [FastAPI](#1-fastapi) — the web framework
2. [Pydantic](#2-pydantic) — data validation + typed models
3. [LangGraph](#3-langgraph) — pipeline-as-graph orchestration

---

## 1. FastAPI

A web framework for Python. ("Web framework" = a library that makes
it easy to write a program that responds to HTTP requests.)

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

### Why Audrey needs FastAPI

Audrey is one HTTP surface that has to do three different things:

1. **Authenticate every protected route.** Bearer tokens validated
   against OWUI on every request to chat completions, file uploads,
   admin routes, memory recall.
2. **Validate the request schemas for every endpoint.**
   `ChatCompletionRequest`, `KBIngestRequest`, file-upload metadata,
   admin actions — all coming from arbitrary clients (OWUI, direct
   `curl`s, anything reaching our public URL).
3. **Stream Server-Sent Events back.** Chat completions return tokens
   as they arrive from Ollama, not in a single response. The shape
   has to be OpenAI-compatible (`data: {...}\n\n` framing) so OWUI
   and other clients consume it correctly.

Without a framework, each of those becomes hand-rolled work in every
route. With FastAPI, you get all three for free:

- Auth becomes one parameter declaration: `me: AuthedUser =
  Depends(require_user)`. The function never runs if auth fails.
- Validation becomes one parameter: `payload: ChatCompletionRequest`.
  Bad input gets a structured 422 response automatically; your code
  never sees malformed data.
- Streaming becomes a return type: `StreamingResponse(generator)`.
  FastAPI handles the chunked transfer encoding.

The dependency-injection model in particular pays off in this
codebase. Audrey has multiple auth tiers (`require_user`,
`require_admin`) and multiple per-request derived values (decoded
user identity, request-scoped metrics labels). With `Depends(...)`,
each one is a small declarative function used as a parameter. The
alternative — a middleware that mutates a `request.state` dict that
every handler has to read from carefully — is what every other
framework's tutorials apologize about.

### What FastAPI gives you

Three things, all valuable.

**(1) Decorator routing.** `@app.get(...)`, `@app.post(...)`,
`@app.delete(...)` — decorators that map URL paths to functions.

Quick primer on the verbs themselves, since FastAPI just exposes them
and the choice of which one to use is yours. HTTP defines a small set
of "methods" (also called verbs) that signal the *intent* of a
request:

| Verb     | Use it for                                                          | Should it change data? |
|----------|---------------------------------------------------------------------|------------------------|
| `GET`    | "Read this thing." Listing, fetching, searching.                    | No                     |
| `POST`   | "Create a new thing" or "perform an action that has side effects."  | Yes                    |
| `PUT`    | "Replace this whole thing with what I'm sending."                   | Yes                    |
| `PATCH`  | "Modify part of this thing."                                        | Yes                    |
| `DELETE` | "Remove this thing."                                                | Yes                    |

Audrey uses `GET`, `POST`, and `DELETE`. `PUT` and `PATCH` exist but
aren't needed yet. The rules are conventions, not enforcement —
nothing stops you writing `@app.get("/delete-everything")` — but
violating them confuses every other tool that talks HTTP (browsers,
proxies, monitoring, your own future self).

**Examples of each, and when you'd reach for them:**

`GET` — fetch without changing anything:

```python
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.get("/users")
async def list_users(role: str | None = None) -> list[dict]:
    # `role` comes from the query string: /users?role=admin
    return query_users(role=role)

@app.get("/users/{user_id}")
async def get_user(user_id: str) -> dict:
    # `{user_id}` in the URL becomes the parameter.
    # GET /users/abc123  →  user_id="abc123"
    return load_user(user_id)
```

`GET` requests typically have no body — input comes from the URL path
or the query string. They're safe to retry, safe to cache, and safe
for a browser to follow on a link click.

`POST` — create something new, or do something with side effects:

```python
@app.post("/users")
async def create_user(payload: CreateUserRequest) -> dict:
    # Body comes in as JSON; FastAPI parses + validates it.
    return {"id": create(payload.email, payload.role)}

@app.post("/users/{user_id}/reset-password")
async def reset_password(user_id: str) -> dict:
    # Side-effecting action with no "thing" being created.
    # POST is the catch-all for "do this, please."
    send_reset_email(user_id)
    return {"sent": True}
```

`POST` is also what Audrey uses for chat completions — even though
you might think of "ask the model a question" as a read, the act of
generating an answer consumes resources and produces a billable side
effect, so `POST` is the right verb.

`DELETE` — remove something:

```python
@app.delete("/users/{user_id}")
async def delete_user(user_id: str) -> dict:
    remove_user(user_id)
    return {"deleted": user_id}
```

`DELETE` requests usually carry no body (the URL identifies what to
delete). Returning the deleted resource's id is a friendly
convention; some APIs return 204 No Content with no body at all.

In every case, FastAPI takes the path string and turns the
`{placeholders}` into function parameters with the type you declared.
Query string values come in as parameters that aren't in the path.
Request bodies come in as Pydantic models. You write the function;
FastAPI does the wiring.

**(2) Automatic JSON parsing + validation.** This one's easier to
appreciate after you've seen the alternative. In a barebones web
framework, handling a `POST /users` request looks something like:

```python
# What you'd write WITHOUT FastAPI — for contrast only.
def create_user_handler(request):
    raw_body = request.body    # bytes from the network
    try:
        data = json.loads(raw_body)   # parse JSON or fail
    except json.JSONDecodeError:
        return Response(status=400, body="invalid JSON")
    if "email" not in data or not isinstance(data["email"], str):
        return Response(status=422, body="email is required and must be a string")
    if "role" not in data or not isinstance(data["role"], str):
        return Response(status=422, body="role is required and must be a string")
    # finally, the actual logic:
    create_user(email=data["email"], role=data["role"])
    return Response(status=200, body=json.dumps({"created": data["email"]}))
```

That's a lot of boilerplate, and you'd write it for every endpoint
that takes a body. With FastAPI, you write the schema once as a
Pydantic model and declare it as a parameter:

```python
# What you actually write WITH FastAPI:
class CreateUser(BaseModel):
    email: str
    role: str

@app.post("/users")
async def create_user(payload: CreateUser) -> dict:
    return {"created": payload.email}
```

When a request hits this endpoint, FastAPI automatically:

- Reads the raw bytes from the request body.
- Parses them as JSON.
- Constructs a `CreateUser` instance from the parsed dict.
- Validates that `email` and `role` are present and are strings.
- If anything fails, returns a structured 422 response describing
  exactly what was wrong — your function never runs.
- If everything passes, hands you `payload` as a typed `CreateUser`
  object (so `payload.email` works in your editor with autocomplete).

The validation rules are derived from the Pydantic model's type hints
— that's why declaring `email: str` is enough to mean "required, must
be a string." More elaborate constraints (max length, regex pattern,
numeric ranges) come from `Field(...)` annotations covered in §2.

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

---

## 2. Pydantic

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

### Why Audrey needs Pydantic

The boundary between "stuff arriving from the network" and "stuff
your code can trust" is one of the highest-bug-density places in any
program. For Audrey, that boundary is a public-internet-reachable
chat completion endpoint that:

- Accepts JSON from arbitrary clients (OWUI, but also direct
  `curl`s, and eventually anything with our tunnel hostname).
- Forwards parts of that JSON to Ollama, where bad input can either
  silently produce nonsense or, worse, send odd parameters to a
  cloud-billed model.
- Threads the user's `messages` array through every pipeline stage,
  where one rogue field could break a downstream node.

Concrete value Pydantic adds, with examples drawn from the real
`ChatCompletionRequest` schema:

- **`messages: list[ChatMessage] = Field(min_length=1)`** rejects
  empty-conversation requests at the boundary, before they reach the
  pipeline. Without it, the empty list silently flows to the
  classifier and produces a confusing "no user message found" error
  ten frames deeper.
- **Typed optional fields like `temperature: float | None = None`**
  validate for free from the annotation alone: a client that sends
  `"temperature": "hot"` gets a clean 422 ("Input should be a valid
  number") instead of that string reaching Ollama. Pydantic enforces
  the *type* without any extra code. Note Audrey deliberately does
  *not* bound the *range* here — you could add
  `Field(ge=0.0, le=2.0)` to reject out-of-range values at the
  boundary, but Audrey leaves that off because the field exists mainly
  for OpenAI-client compatibility and Ollama clamps out-of-range
  sampling values itself. The lesson here is the split: Pydantic gives
  you type-validation for nothing, and range-validation is a
  deliberate choice you opt into per field.
- **The schema is the contract.** When OWUI's request format changes
  in some future version, the Pydantic model is the single place
  that has to be updated. Without it, the contract lives implicitly
  across however many `payload.get(...)` calls exist in the codebase.

It's also **runtime-checked**, which is different from being just
**type-checked**.

> **Aside: type-checked vs runtime-checked.** Both are ways to catch
> "this code expected an X but got a Y" mistakes. The difference is
> *when* the check happens.
>
> - **Type-checked.** A separate tool (mypy, or the type checker
>   built into your editor) reads your code as text without running
>   it, follows the type hints, and flags inconsistencies. Happens
>   *before* the code runs. Limitation: It can only see the code
>   itself; it has no idea what an HTTP client will actually send.
>   ```python
>   def greet(name: str) -> str: return f"hello {name}"
>   greet(123)  # mypy flags this; Python itself runs it fine
>   ```
> - **Runtime-checked.** The validation happens *as the code
>   executes* against real values. Pydantic does this — when you
>   construct `CreateUser(email=123, ...)`, it actually inspects the
>   value `123` and raises `ValidationError`. No separate tool; the
>   validation IS the code.
>   ```python
>   CreateUser(email=123, role="admin")  # raises immediately
>   ```
>
> They cover different mistake categories. Type checks catch your own
> programming errors before you run the program; runtime checks catch
> bad data from outside the program (request bodies, config files,
> database rows). Audrey uses both: type hints + mypy for the code we
> control, Pydantic for the data we don't.
>
> Mental model: The type checker is a proofreader who reads your
> manuscript before publication. The runtime checker is a bouncer
> inspecting each guest at the door.

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

(This is an *illustrative* schema showing what a `Field(...)` range
constraint looks like — Audrey's real `ChatCompletionRequest` keeps
`messages = Field(min_length=1)` but leaves `temperature` unbounded,
as §2.2 explains. The `ge`/`le` here is the example, not Audrey's
actual rule.)

A client sending `{"model": "x", "messages": []}` gets back a 422 with
a precise error: `messages: List should have at least 1 item`.

### Where you'll see it in Audrey

- [`src/audrey/routes/openai/schemas.py:16-57`](../../src/audrey/routes/openai/schemas.py#L16) —
  `ChatMessage` and `ChatCompletionRequest` schemas.
- [`src/audrey/routes/admin.py`](../../src/audrey/routes/admin.py) —
  response models like `AuthClearResponse`.
- [`src/audrey/config.py`](../../src/audrey/config.py) — Pydantic
  Settings models for env-var loading.
- [`tools-server/app.py`](../../tools-server/app.py) — every endpoint
  has a request schema.

---

## 3. LangGraph

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

LangGraph's pitch: Extract the control flow into a **graph** you can
look at. Each node does one thing. Each edge says "after node A, run
node B" or "after node A, decide based on state which of B/C/D to
run." The state object accumulates results as the graph runs.

### Why Audrey needs LangGraph

Audrey's pipeline is a real state machine, not a linear chain. Within
one chat completion, control flows through:

1. **classify** — figure out the task type from the prompt.
2. **complexity** — fast-path or deep-panel?
3. **memory recall** — fetch any relevant remembered facts.
4. **datetime injection** — add ISO-8601 timestamp context.
5. **fair-gate** — wait for a per-user concurrency slot.
6. **either fast-path (single model) OR deep panel (N parallel
   workers + planner + synth)**.
7. **reflect** — does the answer pass quality checks?
8. **on failure: retry the panel; on success: stream out**.

There are conditional edges everywhere — fast vs deep, reflect-pass
vs reflect-retry, brevity-cue bypass, escalation. Without a graph
library, this becomes ~15 functions passing a giant dict by kwarg
through nested `if/else` blocks. The flow lives implicitly in the
call stack and cannot be read top-to-bottom from any single file.

With LangGraph, the topology lives in
[`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py).
**The whole pipeline is in one file.** Adding a stage is one
`add_node` call and one or two `add_edge` calls. Concrete past pain:
the brevity-cue bypass that fixed the reflection step burning cloud
time on legitimately short answers (e.g. "what year is it? answer in
one sentence") was a *one new conditional edge*, not a refactor —
the existing topology absorbed it cleanly.

A second thing LangGraph buys you: Each node is a small async
function with one responsibility. They're individually testable
(tests/test_classify.py, tests/test_reflect.py, tests/test_fair_gate.py)
because they take state in and return state out, no hidden globals.

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

- **State** — a `TypedDict` (see [Lesson 1 §4](lesson-01-foundations.md#4-typed-dictionaries-typeddict)).
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
  is the entire graph definition. **One file.**
- The state schema is
  [`src/audrey/pipeline/state.py`](../../src/audrey/pipeline/state.py).
- Each node implementation lives next to the graph definition (in the
  same file) — they're small wrappers around the real logic in
  `classify.py`, `fast_path.py`, `deep_panel.py`, etc.


> **Aside: LangGraph vs LangChain.** You'll see both names if you
> Google around. They're separate libraries from the same team:
>
> - **LangChain** is the older, broader toolkit — model wrappers,
>   document loaders, retrievers, prompt templates, agents, vector
>   store integrations, plus a hundred more building blocks. Pitch:
>   "any LLM thing you might want, here's a class for it."
> - **LangGraph** is the newer, narrower library focused only on
>   orchestration — the directed-graph-of-nodes pattern we just
>   walked through. Pitch: "your app's control flow is a graph; let
>   me run it."
>
> They're designed to play together, but you can use either alone.
> **Audrey uses LangGraph and not LangChain.** Audrey talks to Ollama
> through its own small client (`models/ollama.py`) rather than
> LangChain's `ChatOllama`, because the abstraction tax wasn't worth
> the dependency for the limited surface we use.


## You're done with Lesson 2

That's the orchestration stack — the three libraries that shape
Audrey's HTTP surface and pipeline structure. As you read this
signature:

```python
async def chat_completions(
    payload: ChatCompletionRequest,
    me: AuthedUser = Depends(require_user),
):
```

…know that it parses as "an async function (Lesson 1 §1, so it
doesn't block other requests). Takes a request body parsed and
validated by Pydantic (§2) into a `ChatCompletionRequest`. Also gets
`me`, the result of running the `require_user` function as a FastAPI
dependency (§1) — which validates the auth header and returns a
typed `AuthedUser` dataclass (Lesson 1 §3)." If that lands, you're
ready to move on.

[Lesson 3 — Foundations III: the satellite libraries](lesson-03-foundations-satellites.md)
covers the libraries Audrey *calls out to*: httpx for outbound HTTP,
Qdrant + embeddings for vector search, Prometheus for metrics,
pytest for tests. These sit underneath the orchestration stack and
make the system actually work in production.
