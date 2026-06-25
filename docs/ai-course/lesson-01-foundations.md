# Lesson 1 — Foundations I: Python language features

**Estimated time:** 60-90 minutes if you read carefully and try the
examples. You can split it across sessions — sections are
self-contained.

**Goal:** before we touch the Audrey codebase proper, you should know
what each major *language feature* the project relies on actually is,
what problem it solves for Audrey specifically, and recognize its
shape when you see it. This is **not** a Python tutorial — you already
know the language basics. It's a tour of the modern-Python features
this project leans on heavily.

This is split out from a former combined foundations lesson into two
halves. Lesson 1 (this one) is the **language layer** — async,
context managers, types, TypedDict. [Lesson 2](lesson-02-foundations-libraries.md)
is the **library layer** — FastAPI, Pydantic, LangGraph, httpx,
Qdrant, Prometheus, pytest. Read this one first; the libraries assume
you've internalized async + types.

You don't need to memorize anything. Skim what's already familiar,
read carefully what's new. Each section has a "**Why Audrey needs
this**" subsection that anchors the abstract feature to the specific
problem it solves in this codebase, and ends with a "where you'll see
this in Audrey" pointer so the idea has a concrete home.

**What this lesson covers:**

1. [`async`/`await` and `asyncio`](#1-asyncawait-and-asyncio) — the
   model for everything else
2. [Context managers](#2-context-managers) — the `with` statement
3. [Type hints, dataclasses](#3-type-hints-dataclasses) — annotating
   the shapes data takes
4. [Typed dictionaries (TypedDict)](#4-typed-dictionaries-typeddict) — the
   pipeline-state pattern

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

### Why Audrey needs async

Audrey is a *gateway*. It rarely does meaningful CPU work itself;
mostly it forwards requests to other services and waits for replies.
A single chat completion can take 30 seconds (fast path, local
quantized model) or three minutes (deep panel cloud call burning paid
provider time). During those waits, Audrey is doing essentially
nothing except holding a TCP socket open and waiting for tokens to
arrive.

If those waits *blocked* — the way a sync HTTP library would — the
whole Python process would stall. With 5 users on the system, average
wait time would balloon from "actual model latency" to "actual model
latency × queue depth." A single deep-panel cloud call would freeze
every other user behind it for three minutes.

With async, while user A's request is sitting on a 30-second cloud
call, user B's short fast-path request runs in parallel on the same
event loop, on the same thread, in the same process. The savings
aren't theoretical — Audrey's deep panel runs **N cloud workers
concurrently** (typically 3-5), and the only reason that works is
that all of them are awaiting tokens at once instead of taking turns.

There's a flip side worth flagging now, because it explains some
quirks you'll meet later in the codebase. Calling a synchronous
function from inside an `async def` is *legal* — Python won't stop
you — but it silently breaks the concurrency story. While the sync
call runs, the event loop is frozen; every other in-flight request
is paused until the sync call returns. On fast operations (a memory
lookup, a millisecond of math) this is invisible. On slow operations
(network I/O, disk I/O on slow storage) it's a stall.

Audrey has two known cases of this in `kb/ingest.py` and
`kb/reconcile.py`, where the code calls `Path.exists()` (a sync
filesystem syscall) from inside `async def` functions. They're
flagged by a linting tool — basically an automated code reviewer —
under a rule called `ASYNC240`. Audrey accepts these warnings rather
than fixing them: The KB lives on local SSD where `exists()` is
sub-millisecond, and the calls happen on a periodic background sweep
rather than the request hot path, so the stall is too short to
matter. If the KB ever moved to slow storage, that decision would
need to be revisited. Don't worry about the specifics now — just
know that "sync call inside async function" is a real hazard and
Audrey is aware of its instances.

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
  every other task stalls until it returns.

### Tiny working example

The code below uses async features, so it has to live in a `.py` file
and be run with `python3` against that file. The top-level
`asyncio.run(main())` line is what kicks the whole thing off.

**Step 1.** Open your terminal.

- **Linux:** any terminal app, running its default shell (usually
  bash). The shell brand doesn't matter for these examples.
- **macOS:** open the **Terminal** app (Applications → Utilities →
  Terminal). Default shell is zsh; that's fine.
- **Windows:** the Linux/macOS commands below use Unix shell syntax
  (heredocs, `/tmp/...` paths, `$PATH`). The clean way to use them on
  Windows is **WSL2** (Windows Subsystem for Linux). If you don't
  already have it: Open PowerShell *as administrator* and run
  `wsl --install`, reboot when prompted, then launch the **Ubuntu**
  app from your Start menu. From inside Ubuntu/WSL you have a real
  Linux environment with bash — every command in this lesson works
  unchanged. (Windows-native PowerShell or `cmd` would need different
  commands for file creation; not worth fighting.)

The rest of this lesson assumes you're at a bash/zsh prompt with
`python3` available. Verify:

```
$ python3 --version
Python 3.12.3
```

If that command says "command not found" or shows a version below
3.7, install Python 3 first (Audrey targets 3.12). On WSL Ubuntu:
`sudo apt update && sudo apt install python3 python3-pip`.

**Step 2.** Create a file at `/tmp/async_demo.py` with the example
code. The easiest copy-pasteable way is a heredoc — paste this entire
block into your terminal and press Enter:

```bash
cat > /tmp/async_demo.py <<'EOF'
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
EOF
```

(`cat > file <<'EOF' … EOF` is a shell shortcut for "take everything
between the EOFs as input and write it to `file`." Using the file
manager + a text editor would work the same way; the heredoc is just
faster.)

**Step 3.** Run the file:

```
$ python3 /tmp/async_demo.py
```

You should see:

```
A: starting
B: starting
C: starting
A: done
B: done
C: done
['A', 'B', 'C']
```

If you see an error mentioning `SyntaxError: 'await' outside async
function`, you're running an old Python (pre-3.7). Check
`python3 --version` again.

All three tasks started immediately. Because each
`await asyncio.sleep(...)` yields control back to the event loop, the
loop ran the others while the first was waiting. Total runtime is
~3 seconds (the longest task), not 6 seconds (the sum). That's the
whole shape of async — you'll see the same pattern at much larger
scale when Audrey's deep panel runs N cloud workers concurrently.

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
*async function call* (the `await`). The next section is exactly
about that `with` keyword.

---

## 2. Context managers

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

### Why this matters in general

Without `with`, you'd write:

```python
f = open("file.txt")
try:
    content = f.read()
finally:
    f.close()
```

Same effect, three times the typing, easy to forget — especially in a
function with multiple return paths or several exceptions to consider.

### Why Audrey needs context managers

In a normal request, Audrey acquires several resources that *must*
be released, even when things go wrong:

- **httpx clients with connection pools.** Every Ollama call, OWUI
  auth probe, custom-tools dispatch, and Qdrant query goes through
  one. If the underlying connection isn't returned to the pool after
  an exception, the pool eventually exhausts and new requests stall.
- **Fair-gate slots.** Audrey caps how many concurrent generations a
  single user can have running at once — without a cap, one user
  could fire 50 requests and starve everyone else off the GPU. The
  cap is enforced by handing out a fixed number of "slots" per user;
  a request has to claim a slot before it can start generating, and
  give the slot back when it finishes. The bookkeeping lives in
  [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py).

  The thing that makes this fragile: A request might fail
  *partway through* — the model call could time out, the network
  could drop, the user could disconnect, an unexpected exception
  could fire. If the failure path skips the "give the slot back"
  step, that slot is leaked: The bookkeeping still thinks it's in
  use, even though nothing is actually using it. Leak enough slots
  for one user and they hit their cap permanently — every subsequent
  request from them queues forever waiting for a slot that will
  never come back. The user-visible symptom is "Audrey hangs forever
  for me," with no error message, hours after the leak happened.
- **Phase tickers.** Streaming responses spin up a background task
  that emits "thinking…" progress dots; that task has to be cancelled
  when generation finishes, even if generation finished by raising.
- **The whole FastAPI app's startup/shutdown.** Models, watchers,
  scheduler tasks all need to start cleanly and tear down cleanly.

Cloud calls timing out mid-stream is *normal*, not exceptional. So
the exception path isn't a rare edge case to defend against — it's
the daily traffic. `try/finally` works, but every site that needs it
becomes seven lines of boilerplate that all have to be correct.
`with`/`async with` collapses that to one line and makes the cleanup
visually colocated with the resource acquisition.

The cost of getting this wrong is not "the program crashes." It's
worse: The program *keeps running* with a slowly leaking pool of
connections or slots, and you find out hours later when a user
reports "Audrey hangs forever for me."

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

The `yield` here might look strange — `yield` is normally a Python
keyword that turns a function into a *generator* (something you can
ask for values one at a time, where execution pauses between values).
The `@contextmanager` decorator hijacks that behavior: It lets the
function pause at the `yield`, run the body of the `with` block, then
resume the function so it can clean up.

So in this idiom:

- Code **before** the `yield` runs when the `with` block is entered
  (setup).
- The `yield` itself **suspends** the function.
- The body of the `with` block runs.
- When the `with` block exits — by normal flow or by exception — the
  function resumes after the `yield` and runs to completion (cleanup,
  inside the `finally:`).

In the `timing` example above, `yield` had nothing after it — the
`with` block didn't need a handle to anything; it just needed the
setup-then-cleanup behavior. That's why the calling code was
`with timing("query"):` and not `with timing("query") as something:`.

But sometimes the `with` block *does* need a handle — something the
setup step produced that the body wants to use. The classic case is
opening a file: The setup opens it, the body wants to read from it,
the cleanup closes it. To pass that handle to the body, put a value
*after* `yield`. Whatever you yield gets assigned to whatever the
caller wrote after `as`:

```python
@contextmanager
def opened(path: str):
    f = open(path)               # setup: open the file
    try:
        yield f                  # the `f` here lands in `as f` below
    finally:
        f.close()                # cleanup: always runs

with opened("config.yaml") as f: # `f` is the file handle yielded above
    data = f.read()              # use it inside the block
# file is closed here
```

So `yield x` is doing two things at once: It's the suspension point
where the body of the `with` block runs, *and* it's how you hand a
value out to that body. `yield` with nothing after it (as in
`timing`) means "suspend, but the body doesn't get a handle from
me." `yield x` means "suspend, and bind `x` to whatever the caller
wrote after `as`."

For async, use `@asynccontextmanager` from the same module — exactly
the same pattern, just with `async def` and an awaitable body.

### A real custom one from Audrey

The fair-gate slot-acquisition is a custom async context manager —
because the "release on exit" guarantee is exactly what you need to
prevent slot leaks under exceptions. Simplified down to the
context-manager skeleton:

```python
@asynccontextmanager
async def acquire(self, model: str, *, location: str):
    """Wait for a free slot for this user+model, hold it for the body,
    release it on exit even if the body raises."""
    fut = self._enqueue(model, location)
    await fut                                    # wait for our turn
    try:
        yield                                    # caller runs their model call here
    finally:
        self._release(model, location)           # always runs, even on exception
```

Read it like this:

- `await fut` runs at `async with` *entry* — blocks until the gate
  hands this request a slot.
- `yield` pauses the function and lets the body of the `async with`
  block run.
- `self._release(...)` in the `finally` runs at `async with` *exit*,
  whether the body completed normally or raised.

The real version in
[`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py)
also takes a `user_id` (so each user gets their own queue bucket)
and short-circuits to a no-op for cloud calls (those don't compete
for local GPU slots). The structural shape — `try / yield /
finally(_release)` — is identical.

If you wrote that without the context-manager pattern, every caller
would be responsible for the `try/finally` themselves — and one
forgotten `finally` somewhere in the codebase becomes a slot leak
that bites in production months later.

### Where you'll see it in Audrey

- `async with httpx.AsyncClient(...) as client:` — every place we
  make an outbound HTTP call (`auth.py`, `models/ollama.py`,
  `tools/dispatch.py`, `kb/embed.py`).
- `async with gate.acquire(...)` in
  [`pipeline/fair_gate.py`](../../src/audrey/pipeline/fair_gate.py) —
  the slot-leak guard described above.
- `async with PhaseTicker(...)` in
  [`pipeline/banners.py`](../../src/audrey/pipeline/banners.py) —
  starts a "tick the dots" background task on enter, cancels it on
  exit.
- `@asynccontextmanager async def lifespan(app: FastAPI):` in
  [`main.py`](../../src/audrey/main.py) — FastAPI's lifecycle hook.
  Code before `yield` runs at startup; after `yield` runs at
  shutdown. We'll meet this properly in a later lesson when we open
  `main.py`.

---

## 3. Type hints, dataclasses

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
- FastAPI + Pydantic use them at runtime to validate (we'll cover
  these in Lesson 2).
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

### Why Audrey needs dataclasses (and type hints)

Audrey shuffles small, structured records between layers constantly:

- `AuthedUser` (email + role + owui_id) — produced by `auth.py`,
  consumed by every protected route.
- `WorkerDraft` (a single deep-panel worker's output: model, content,
  tool calls, errors) — produced by `deep_panel.py`, consumed by
  `synthesize.py`.
- `KBHit` (qdrant search result: source, text, score) — produced by
  `kb/qdrant.py`, consumed by tools and routes.
- `ReactResult` — the per-iteration state of a ReAct loop.
- `KeywordSignal` — `pipeline/classify.py`'s frozen rule registry.

If those were untyped dicts, every call site would have to *remember*
the keys and types. One typo (`me["emial"]` instead of `me["email"]`)
is a `KeyError` at runtime — found only when that code path actually
runs, which on a low-traffic route might be days after deploy. With
a dataclass, the editor knows the field is `email`, autocompletes
correctly, and a static type checker catches the typo before you save
the file.

There's a second benefit specific to dataclasses: Free `__repr__`
gives you readable log output. Logging an `AuthedUser` shows
`AuthedUser(email='alice@example.com', role='admin', owui_id='abc123')`
instead of `<auth.AuthedUser object at 0x7f4c...>`. That's the
difference between "I can debug from logs" and "I have to add print
statements and redeploy."

Concrete past pain: An early version of the auth wiring crashed
with `AttributeError: 'AuthedUser' object has no attribute 'id'`
because the field is `email`, not `id`. The fix was a one-line
change, but the diagnostic was instant precisely because the
dataclass refused to invent a missing attribute. An untyped-dict
version would have returned `None` from `state.get("id")` and produced
a much weirder downstream symptom.

### Where you'll see it in Audrey

Dataclasses:
- `AuthedUser` in [`src/audrey/auth.py`](../../src/audrey/auth.py)
- `ReactResult` in [`src/audrey/pipeline/react.py`](../../src/audrey/pipeline/react.py)
- `KBHit` in [`src/audrey/kb/qdrant.py`](../../src/audrey/kb/qdrant.py)

Type hints: every function in the codebase.

---

## 4. Typed dictionaries (TypedDict)

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

### Why use TypedDict at all

You get the **type checker's** view of a dict's structure without
changing the runtime. So if you write `state["task_type"]`, the
editor knows the value is a string. If you write `state["taks_type"]`
(typo), the type checker can flag it before you run the code.

The `total=False` part says "all fields are optional" — useful for
LangGraph because nodes only set the fields they care about.

### Why Audrey needs TypedDict (and not a dataclass)

LangGraph's contract is: **state is a dict, and nodes return dicts
that get merged into it.** Concretely, when a node returns
`{"classification": "deep"}`, LangGraph does the equivalent of
`state.update({"classification": "deep"})` on the running state.

A dataclass is a class, not a dict. Merging into one would require
either:

- Translating between dict and dataclass at every node boundary
  (extra code, easy to drift), or
- Subclassing in some `dict`-compatible way (then you've reinvented
  TypedDict the hard way).

TypedDict is the answer that fits LangGraph's merge model without
glue: It's *literally* a dict at runtime — `state["x"]`, `state.update(...)`,
`state.get("x", default)` all work as you'd expect — but `mypy` and
your editor treat it as if it had declared fields.

You can think of it as: dataclass-style type checking, dict-style
runtime. Two mental models in one tool.

A second smaller reason: LangGraph state often passes through
serialization boundaries (logging, checkpointing, debugging output).
Dicts serialize trivially with `json.dumps`. Dataclasses need
`asdict(...)` first. For a state object that's flying through a
pipeline ten times per request, the dict-by-default form is
genuinely lower friction.

### Where you'll see it in Audrey

- [`src/audrey/pipeline/state.py`](../../src/audrey/pipeline/state.py)
  defines `PipelineState` and `WorkerDraft` (yes, `WorkerDraft` is
  TypedDict, not dataclass — because it lives inside the LangGraph
  state and rides the merge semantics).

---

## You're done with Lesson 1

That's the language layer. Four features, all mandatory background
for reading the rest of the codebase:

- **async/await** — why Audrey can serve concurrent users with one
  Python process.
- **context managers** — why resources don't leak when cloud calls
  time out mid-stream.
- **type hints + dataclasses** — why structured records stay readable
  and typos get caught at edit time.
- **TypedDict** — the shape LangGraph state takes.

As you read this line:

```python
async with gate.acquire("qwen3.6:35b", location="local"):
    drafts: list[WorkerDraft] = await deep_panel(state)
```

…know that it parses as "this is an async context manager guarding
a slot acquisition, and the body runs `deep_panel` which produces a
list of typed `WorkerDraft` records." If that lands, you're ready
to move on.

[Lesson 2 — Foundations II: the orchestration stack](lesson-02-foundations-libraries.md)
covers the three libraries that *shape* Audrey — FastAPI for HTTP,
Pydantic for validation, LangGraph for pipeline orchestration. Now
that the language pieces are in place, those sit on top cleanly.
The lesson after that covers the satellite libraries (httpx, Qdrant,
Prometheus, pytest).
