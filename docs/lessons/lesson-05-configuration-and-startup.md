# Lesson 5 — Configuration and startup

**Estimated time:** 45-60 minutes (read 25, walk through code 20)

**Goal:** by the end of this lesson, you can answer
*"what happens between the Python process starting and the first
request being served?"* — including where settings come from, in
what order things get wired together, and what `app.state.*` is for.

In Lesson 4 we traced one request through the running app. This lesson
zooms out to the moment *before* any request: How the process boots,
how config flows from a YAML file through env vars into Python objects,
and how the long-lived dependencies (Ollama client, model registry, KB
stack, tool registry, compiled pipeline graph) get attached to the app
so every route can reach them.


## 1. Context

A FastAPI app is, in principle, just a Python process running uvicorn
— uvicorn being the **ASGI server** that actually opens the TCP
socket, accepts HTTP connections, and hands each one to FastAPI as a
request. ("ASGI" is the asynchronous successor to WSGI: A Python
calling-convention for web frameworks. FastAPI defines the routes;
uvicorn does the network plumbing.) The two are separate programs by
design — FastAPI is the application framework, uvicorn is the server
running it, and you'd swap uvicorn for hypercorn or daphne without
touching FastAPI code. In Audrey we launch uvicorn explicitly via the
`CMD` line at the bottom of `docker/audrey.Dockerfile`:

```
CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

That command imports `audrey.main`, finds the `app` object, and
starts serving it on port 8000.

But Audrey isn't *just* a FastAPI app — by the time it serves its
first request it has:

- Read a YAML config file and merged env-var overrides on top.
- Opened an httpx client to Ollama (kept open for the process lifetime).
- Built a model registry and a per-task health tracker.
- Built two queueing layers — a global GPU semaphore (`FairLocalGate`)
  and a per-user in-flight cap (`UserInflightRegistry`).
- Walked every configured tool server's `/openapi.json` and registered
  the tools it found.
- Compiled a LangGraph pipeline that closes over all of the above.
- Connected to Qdrant, ensured collections exist, and reconciled an
  on-disk SQLite index against them.
- Started two background tasks (a filesystem watcher for the dataset
  directories, and a periodic reconciler).

All of that is **boot work** — it happens once, before the first
request lands, and most of it gets **stashed on `app.state`** so route
handlers can pull it back out. Once you unpack `main.py`, you will understand that it is essentially a large and complex checklist.

The same goes for config: There is one source of truth (`config.yaml`),
one override layer (env vars from `.env` and the shell), and one access
function (`get_config()`). Every other module in Audrey ultimately gets
its tunables from there. Knowing how that pipeline works is what lets
you change a setting and predict where it'll show up.


## 2. Read-along

Open these files in your editor as we go:

- [`src/audrey/main.py`](../../src/audrey/main.py) — the FastAPI
  entrypoint and lifespan.
- [`src/audrey/config.py`](../../src/audrey/config.py) — the YAML +
  env-var loader.
- [`config.yaml`](../../config.yaml) — the actual settings, at the
  repo root.

### 2.1 The config stack: yaml → env → Python

Config has three layers, from "most static" to "most overridable":

1. **`config.yaml`** at the repo root — the source of truth for the
   model registry, deep-panel pools, timeouts, KB knobs, fairness
   limits, complexity threshold, etc.
2. **`.env` file + shell environment** — overrides for things that
   change per deployment (URLs, ports, secrets). The `.env` file is
   gitignored; whatever launches the Python process is responsible
   for getting these into `os.environ`.
3. **`EnvOverrides`** in [`config.py:23`](../../src/audrey/config.py#L23)
   — a `pydantic_settings.BaseSettings` subclass that reads those env
   vars (with optional fallback defaults) into typed Python fields.

The merge happens in `Config._apply_env_overrides` at
[`config.py:71`](../../src/audrey/config.py#L71). Read it top to bottom
— it's just a list of "if env var X was set, write its value into the
right slot of the YAML dict." The YAML dict is the merged result;
`Config.raw` exposes it.

#### A concrete walk-through

You set `GPU_CONCURRENCY=2` in the environment Python sees at boot.
Here's what happens:

1. **Python boots.** uvicorn imports `audrey.main:app`, which
   triggers FastAPI's startup, which calls `lifespan()`, which on its
   first line calls [`get_config()`](../../src/audrey/config.py#L130).
2. **`get_config()` instantiates `EnvOverrides()`.** Pydantic
   Settings reads the process environment, sees
   `GPU_CONCURRENCY=2`, matches the `gpu_concurrency` field's `alias`
   on [`config.py:53`](../../src/audrey/config.py#L53), and stores
   `2` on the object.
3. **`Config.__init__` calls `_apply_env_overrides`,** which on
   [`config.py:74-75`](../../src/audrey/config.py#L74) does
   `self._yaml.setdefault("gpu", {})["concurrency"] = 2`. The YAML
   originally said `gpu.concurrency: 1`; now the merged dict says `2`.
4. **`lifespan()` reads it back** at [`main.py:58`](../../src/audrey/main.py#L58):
   `int(cfg.raw.get("gpu", {}).get("concurrency", 1))`. The result is
   `2`, which it passes to `FairLocalGate(concurrency=2)`.

That's the whole pipeline. Memorize the shape: **process env →
Pydantic Settings reads env → merge into YAML dict → lifespan reads
merged dict.**

#### What's a `BaseSettings`?

`pydantic_settings.BaseSettings` is Pydantic's "model whose fields
auto-populate from environment variables" subclass. You declare fields
with type annotations and (optionally) aliases; instantiating the
class reads the environment and validates each value against the
declared type — same machinery as a regular Pydantic model
([Lesson 2 §2](lesson-02-foundations-libraries.md#2-pydantic)) but
sourced from `os.environ` instead of an HTTP body.

`SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
extra="ignore")` on [`config.py:26`](../../src/audrey/config.py#L26)
configures three things: also read variables from `.env` if the file
is present, decode it as UTF-8, and silently drop env vars that don't
match any declared field (instead of erroring).

`alias="OLLAMA_HOST"` on [`config.py:31`](../../src/audrey/config.py#L31)
lets us write `cfg.env.ollama_host` in Python while the env var is
spelled `OLLAMA_HOST` — Python's snake_case meets shell convention's
SHOUTY_CASE.

#### Why this shape, not just env vars everywhere?

Two reasons.

**(1) The model registry is too big to live in env vars.** It's
roughly thirty entries, each with a name, priority, speed score,
quality score, and location. That kind of structured data wants to be
YAML. Same for the deep-panel pools, the tool-capable model list,
the dataset paths.

**(2) The YAML is checked into git; the env vars aren't.** A new
deployment can `git clone` and immediately have a sensible default
config. The deployment-specific bits (URLs of Ollama / Qdrant / OWUI,
the Brave API key) go in `.env`, which is gitignored — they're
deployment secrets and shouldn't follow the repo around.

So: YAML for shape (rarely changes), env for site-specific (often
changes, often secret).

### 2.2 `lifespan` — the boot script

Open [`main.py:48`](../../src/audrey/main.py#L48):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    log.info("audrey starting; version=%s", __version__)
    ...
```

That `@asynccontextmanager` decorator is the same one we covered in
[Lesson 1 §2](lesson-01-foundations.md#async-context-managers). The
function uses `yield` exactly once; everything *before* the yield
runs on startup, everything *after* runs on shutdown.

FastAPI calls this when uvicorn starts the app. The `lifespan=lifespan`
keyword on [`main.py:186`](../../src/audrey/main.py#L186) is what
hooks it in. Here's the script the function executes, top to bottom:

**Lines 50-52 — load config.** `get_config()` returns the merged
`Config` object we built in §2.1. The `log.info(...)` is the first
sign-of-life line in the logs.

**Lines 54-55 — open the Ollama client.** `OllamaClient` wraps an
httpx `AsyncClient` and is reused for the rest of the process —
exactly the use case we called out in
[Lesson 3 §1](lesson-03-foundations-satellites.md). One client,
process-lifetime, closed during shutdown.

**Lines 56-64 — build the model layer.** A `ModelRegistry` (knows
which models exist for which task type), a `HealthTracker` (knows
which ones are currently failing), a `FairLocalGate` (the GPU
semaphore — capacity 1 by default), a `UserInflightRegistry` (the
per-user request cap). All four are plain Python objects with no I/O
in their constructors; we'll meet them in detail in later lessons.

**Lines 66-73 — discover tools.** `discover_all(tool_servers)` walks
each configured tool-server URL, hits its `/openapi.json`, parses the
tool definitions, and returns a populated `ToolRegistry`. If
`tools.enabled` is false or no servers are configured, we build an
empty registry and log it. We'll cover this in detail later.

> **The discovery step is one-shot.** If a tool server isn't ready
> when this runs, `discover_all` returns whatever it found (possibly
> zero tools) and never auto-retries. The registry stays in that
> state until someone calls `POST /v1/tools/rediscover`, exposed at
> [`main.py:231`](../../src/audrey/main.py#L231). The deploy
> environment is responsible for not starting Audrey until its tool
> servers are reachable; the Python side has no retry loop.

**Line 75 — build the LangGraph pipeline.** `build_graph(...)` is
the function we glanced at in [Lesson 4 §2.2](lesson-04-request-lifecycle.md#22-the-pipeline-graph).
Crucially, `cfg`, `ollama`, `registry`, `health`, `gate`, and
`tool_registry` get passed in here, and the resulting compiled graph
*closes over* those instances. That means the graph nodes don't read
from `app.state` to find these things — they captured them at
build time. Live mutations to `app.state.tools` (e.g. via
`/v1/tools/rediscover`) work only because we keep handing back the
*same `ToolRegistry` instance* and mutating its insides
([`main.py:245-246`](../../src/audrey/main.py#L245)).

**Lines 78-97 — KB stack.** `QdrantKB` opens a connection;
`ensure_collections()` is idempotent (creates the `kb_text` /
`kb_images` collections if they don't exist, no-op otherwise).
`UploadsDB` is the SQLite index over per-user uploads. The
`reconcile_with_qdrant` call cross-checks the SQLite rows against
Qdrant's reality and prunes ghosts / backfills missing rows.

Note the two `try/except`s on [`main.py:85-88`](../../src/audrey/main.py#L85)
and [`94-97`](../../src/audrey/main.py#L94): If Qdrant is unreachable
at boot, we log a warning and keep going. **The KB endpoints will
return 503 later**, but the rest of Audrey (chat, tools, fast path,
deep panel without KB tool calls) still works. That's a deliberate
choice — Qdrant being down shouldn't kill the whole orchestrator.

**Lines 99-106 — build embedders.** A text embedder (delegates to
Ollama's `/api/embeddings` endpoint) and an image embedder (uses
sentence-transformers locally with a CLIP model). Both are reused
across requests.

**Lines 108-120 — KB watcher (conditional).** If `KB_WATCHER_ENABLED`
is set, instantiate a watcher that listens for filesystem changes
under the configured dataset paths and re-ingests files as they
change. `watcher.start()` spawns a background task. We'll cover the
watcher in detail when we reach the KB-ingest lesson.

**Lines 122-131 — KB reconciler (conditional).** Periodic background
sweep that catches drift the watcher misses. Also background-tasked.

**Lines 133-146 — `app.state.*` assignments.** This is the punchline.
Every long-lived dependency we built above gets stashed on
`app.state`, which is the per-app key-value store FastAPI provides
exactly for this purpose. Routes pull these back out via
`request.app.state.foo`. We'll see consumers in later lessons — every
route handler is, structurally, "look up some app.state things, do
work with them, return."

**Line 159 — readiness log.** A single multi-line `log.info` that
prints the boot configuration in one place. When you're tailing the
running process's logs and want to know *what config it actually
booted with*, this is the line to grep for.

**Line 160 — `yield`.** Control returns to FastAPI; uvicorn starts
accepting requests. This line stays "paused" for the entire lifetime
of the process.

**Lines 162-168 — shutdown.** When uvicorn receives SIGTERM, it
cancels the lifespan task; control resumes after the `yield`, and
we tear down in reverse order — stop the reconciler, stop the
watcher, close the SQLite handle, close the Qdrant client, close
the Ollama httpx client. **Order matters here.** We stop the
background tasks first so they're not mid-flight when their backing
clients close.

#### Why a context manager and not separate `startup` / `shutdown`
hooks?

FastAPI used to expose `@app.on_event("startup")` and
`@app.on_event("shutdown")` decorators. They're now deprecated in
favour of `lifespan`. The reason is exactly the one
[Lesson 1 §2](lesson-01-foundations.md#why-audrey-needs-context-managers)
points at: With a context manager, a single block of code owns *both*
sides of the resource lifecycle. The local variables you create on
the way up are still in scope on the way down. With separate startup
/ shutdown handlers you have to stash everything in a module-level
global to get it from one to the other, which is exactly the kind of
state that gets out of sync.

### 2.3 `app.state` as a registry

`app.state` is just a `types.SimpleNamespace`-ish object FastAPI
attaches to the `FastAPI()` instance. You can write any attribute to
it; you can read it back from `request.app.state` inside a route, or
from `app.state` directly if you have the app object in scope.

Audrey uses it as a service registry — the lifespan populates it once
on boot, routes read from it during requests, and the
`/v1/tools/rediscover` endpoint mutates one of its entries in place.

A representative reader is [`main.py:241-247`](../../src/audrey/main.py#L241):

```python
async def rediscover_tools(...) -> dict[str, list[str] | int]:
    cfg = app.state.cfg
    reg = app.state.tools
    tool_servers = list(cfg.tools.get("servers", []) or [])
    fresh = await discover_all(tool_servers)
    reg.by_name.clear()
    reg.by_name.update(fresh.by_name)
```

Look at this carefully. The handler doesn't replace
`app.state.tools` with `fresh`; it mutates the *existing* registry's
internal dict in place. That's because the compiled LangGraph
captured a reference to that exact `ToolRegistry` object back in
`build_graph` — replacing `app.state.tools` would leave the graph
holding the old empty one. Mutating in place keeps the closure
pointed at the right thing.

That's a deliberate design choice. It's the kind of subtle thing
you only notice once you've seen the bug it prevents.

## 3. Comprehension Q&A

Try answering each yourself before reading the answer.

**1. You change `gpu.concurrency` in `config.yaml` from `1` to `3`.
What do you have to do for the running app to pick up the change?**

You have to restart the process. `get_config()` is
`@lru_cache(maxsize=1)` on
[`config.py:129`](../../src/audrey/config.py#L129) — config is read
once per process. The `FairLocalGate` is constructed once during
`lifespan()` and stashed on `app.state.gate`; nothing re-reads
`gpu.concurrency` after that. So the change in YAML lives on disk
but has no effect until uvicorn is restarted, at which point
lifespan runs again and builds a fresh gate from the new value.

If you wanted live reload, you'd have to plumb in a
`reload_config()` call somewhere and rebuild the gate from the new
value — and you'd also need to migrate any in-flight requests off
the old gate. Audrey deliberately doesn't do this; restart is
simpler and fast enough.

**2. Two engineers debate setting `OLLAMA_HOST`. One says "set it in
`config.yaml`," the other says "set it in `.env`." Who's right and
why?**

`.env` is right. `config.yaml` doesn't have an `ollama_host` key at
all — Ollama's URL is purely an `EnvOverrides` field
([`config.py:31`](../../src/audrey/config.py#L31)) with no YAML
counterpart. The reason: It's a deployment-specific URL. Whoever runs
Audrey on a different network needs a different value, and we don't
want everyone editing the YAML (which is checked into git) to pin
their personal host. Same for `QDRANT_HOST`, `OWUI_URL`,
`BRAVE_API_KEY`, etc. — anything that varies per deployment goes in
`.env`, anything that describes the *application*'s shape (model
registry, deep-panel pools, KB topics, fairness limits) goes in
`config.yaml`.

The shape we settled on, restated: YAML for what Audrey *is*, env
for *where* it's running.

**3. The lifespan does `await qdrant.ensure_collections()` but
catches the exception. Why is this `try/except` justified, when most
of the boot is fail-fast?**

Qdrant outage is recoverable in a way config-file corruption isn't.
If `config.yaml` is malformed, there's no sensible default and Audrey
can't function as an orchestrator at all — fast-fail is the right
behaviour and the operator gets a Python traceback in the process
logs. But if Qdrant is temporarily unreachable, Audrey still does
useful work — chat completions, tool calls, fast path, deep panel
without the KB tool — and the user benefit of "service stays up" beats
the cost of "KB endpoints return 503 for a few minutes."

The same logic applies to the `reconcile_with_qdrant` call directly
below ([`main.py:94-97`](../../src/audrey/main.py#L94)): The
SQLite index is still readable without reconciliation; it might just
be slightly stale until next boot.

The pattern is: Catch when degraded operation is meaningful; let it
bubble when degraded operation is a lie.

**4. What's `app.state` actually for, and why does the request
handler in `_stream_via_pipeline` reach for `request.app.state.graph`
instead of importing the graph at module load?**

`app.state` is FastAPI's per-app key-value bag — the canonical place
to stash long-lived dependencies that the lifespan builds and routes
read. The two reasons routes pull from `request.app.state` rather
than module-level imports:

**(1) The objects can't exist at import time.** The `OllamaClient`,
the `QdrantKB`, the compiled LangGraph all need a running event loop
or live network connections to instantiate. Module-level code runs at
import, before the loop starts. They have to be built inside the
async lifespan and stashed somewhere routes can find them later.

**(2) It keeps the dependencies in one place.** Everything the app
needs at runtime is on `app.state`; reading `main.py` once tells you
exactly what services exist. If routes individually `import` their
own clients, you get the same proliferation of half-shared half-not
state that
[Lesson 1's resource-leak example](lesson-01-foundations.md#why-audrey-needs-context-managers)
calls out.

In Lesson 4's request walk, every node-level dependency — Ollama
client, model registry, fair gate, tool registry — was passed into
`build_graph(...)` at boot and *captured by closure* in the node
functions. The route handlers themselves only need `app.state.graph`
(the compiled pipeline) and `app.state.cfg` (for tunables); the rest
is reachable through the graph's closure.

**5. The `ChatCompletionRequest` Pydantic model and the
`EnvOverrides` Pydantic-Settings model are both Pydantic. Why are
they separate base classes?**

They solve different shapes of the same problem.

`BaseModel` (what `ChatCompletionRequest` extends) is for
*request-time* data — JSON bodies that arrive over HTTP, get parsed
once per request, validated, and discarded after the handler
returns. The values are different for every request; Pydantic's job
is "validate this incoming payload."

`BaseSettings` (what `EnvOverrides` extends) is for *boot-time*
data — environment variables that get read once at process start, get
turned into typed Python attributes, and stay constant for the
process's lifetime. The values are the same for every request;
Pydantic Settings' job is "validate the environment Audrey was
launched in."

They share the same validation engine — type coercion, default
handling, error messages — but `BaseSettings` adds the
"read from environment + optional dotenv file + alias support"
machinery on top. Same family, different role.


## When you're ready for the next lesson

The next lesson covers `models/` — `OllamaClient`, `ModelRegistry`,
and `HealthTracker`. We'll trace what happens when a node says "I
want a `general` model": How the registry ranks candidates, how the
health tracker filters out failing ones, how a request gets shaped
into an HTTP call, and what happens when the call fails (retries,
circuit breaker, escalation). It's also where we look at the
distinction between `:cloud`-suffixed model names and local ones,
and why the same `OllamaClient` handles both.
