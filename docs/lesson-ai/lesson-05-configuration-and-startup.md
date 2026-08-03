# Lesson 5 — Configuration and startup

**Estimated time:** 50-70 minutes if you read slowly and keep the files open.

**Goal:** by the end of this lesson, you can answer
*"what happens between starting the container and Audrey serving its first
request?"* You should understand where settings come from, why startup lives in
FastAPI's `lifespan`, what gets stored on `app.state`, and why some boot
failures crash Audrey while others only degrade part of the system.

Lesson 4 traced a request through a running Audrey process. This lesson walks
the moment before that: the container starts, config gets loaded, long-lived
clients and registries are built, the LangGraph pipeline is compiled, and only
then does FastAPI begin accepting traffic.


## 1. Context

Audrey has two different kinds of code:

- **Request-time code** runs every time a user sends a prompt. That was Lesson
  4: route handler, auth, pipeline nodes, model call, streaming response.
- **Boot-time code** runs once when the process starts. That is this lesson:
  config loading, client construction, tool discovery, graph compilation,
  background task startup, and cleanup on shutdown.

If request-time code is "what happens when someone asks Audrey a question,"
boot-time code is "what Audrey has to prepare before any question can be
answered."

Here is the whole startup story in one pass:

```text
docker compose starts audrey-ai
  -> compose supplies env vars and mounts config.yaml
  -> Docker CMD runs uvicorn
  -> uvicorn imports audrey.main:app
  -> FastAPI creates the app object
  -> uvicorn starts FastAPI's lifespan
  -> lifespan loads config
  -> lifespan builds long-lived services
  -> lifespan compiles the LangGraph pipeline
  -> lifespan stores shared objects on app.state
  -> lifespan reaches yield
  -> Audrey is ready for requests
```

That `yield` is the hinge of the whole file. Everything before it is startup.
Everything after it is shutdown.


## 2. Read-along

Open these files in your editor as we go:

- [`compose.yaml`](../../compose.yaml) — how the audrey-ai container gets its
  environment, volumes, healthcheck, and tool-server startup ordering.
- [`docker/audrey.Dockerfile`](../../docker/audrey.Dockerfile) — the command
  that starts uvicorn inside the container.
- [`config.yaml`](../../config.yaml) — Audrey's checked-in runtime settings.
- [`src/audrey/config.py`](../../src/audrey/config.py) — the YAML +
  env-var loader.
- [`src/audrey/main.py`](../../src/audrey/main.py) — the FastAPI
  entrypoint and lifespan.

### 2.1 From compose to Python

Start in [`compose.yaml`](../../compose.yaml). The `audrey-ai` service does
four startup-relevant things before Audrey's Python code runs.

**It builds the right image.** The service points at
[`docker/audrey.Dockerfile`](../../docker/audrey.Dockerfile), which installs
Audrey and ends with:

```dockerfile
CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

That command says: "Run uvicorn, import the object named `app` from the Python
module `audrey.main`, listen on all interfaces inside the container, and serve
HTTP on port 8000."

`uvicorn` is the ASGI server. ASGI is the async Python calling convention that
lets a web server talk to a Python web app. FastAPI defines the application;
uvicorn opens the socket, accepts HTTP connections, and hands each request to
FastAPI. You can think of FastAPI as the route table and uvicorn as the process
that puts that route table on the network.

**It supplies environment variables.** The service has both `env_file: .env`
and an explicit `environment:` block. Those values become process environment
variables inside the container. When Python later instantiates `EnvOverrides`,
Pydantic Settings reads from exactly that environment.

**It mounts config.** Compose binds the repo's `./config.yaml` into the
container at `/app/config.yaml` read-only. The image also contains a copy from
build time, but the runtime bind mount is the important one during deploy.
The Dockerfile sets `AUDREY_CONFIG=/app/config.yaml`, so `get_config()` reads
that mounted file. When the repo is pulled on Unraid and the container is
restarted, the current repo `config.yaml` is what Audrey sees.

**It waits for custom-tools.** Audrey discovers tools once during startup by
asking custom-tools for `/openapi.json`. Compose's healthcheck and
`depends_on` ordering keep Audrey from starting until custom-tools is healthy.
Without that, Audrey can boot with an empty tool registry and stay that way
until the admin rediscover endpoint is called.

Now open [`src/audrey/main.py`](../../src/audrey/main.py). When uvicorn imports
`audrey.main:app`, Python executes the module top to bottom and creates the
FastAPI app object near the middle of the file. Importing the module does not
mean Audrey is ready yet. The app exists, but the expensive shared objects are
built later by `lifespan`.

### 2.2 The config stack: YAML → env → Python

Config has three layers:

1. **`config.yaml`** describes the application's shape: model registry,
   routing pools, timeouts, tool settings, knowledge-base settings, fairness
   limits, and agentic behavior.
2. **Environment variables** describe the deployment: where Ollama is, where
   Qdrant is, whether the KB watcher is enabled, secrets, and site-specific
   overrides.
3. **`Config`** is the Python object Audrey passes around after YAML and env
   have been merged.

The rule of thumb:

```text
YAML says what Audrey is.
Env says where this copy of Audrey is running.
```

Open [`src/audrey/config.py`](../../src/audrey/config.py). The first important
class is [`EnvOverrides`](../../src/audrey/config.py#L23), which inherits from
`pydantic_settings.BaseSettings`.

The merge happens in `Config._apply_env_overrides` at
[`config.py:73`](../../src/audrey/config.py#L73). Read it top to bottom
— it's just a list of "if env var X was set, write its value into the
right slot of the YAML dict." The YAML dict is the merged result;
`Config.raw` exposes it.

#### A concrete value trace

You set `GPU_CONCURRENCY=2` in the environment Python sees at boot.
Here's what happens:

1. `lifespan()` calls [`get_config()`](../../src/audrey/config.py#L242).
2. `get_config()` creates `EnvOverrides()`.
3. Pydantic Settings sees `GPU_CONCURRENCY=2`, matches it to
   `gpu_concurrency`, and converts it to an integer.
4. `Config.__init__` stores the YAML dict on `self._merged`, then calls
   `_apply_env_overrides()`.
5. `_apply_env_overrides()` runs:

   ```python
   if (v := self.env.gpu_concurrency) is not None:
       self._merged.setdefault("gpu", {})["concurrency"] = v
   ```

   `setdefault("gpu", {})` means "if the merged config does not already have a
   `gpu` section, create an empty one." Then it writes `concurrency = 2` into
   that section.

6. `lifespan()` later reads `cfg.raw["gpu"]["concurrency"]` and passes the
   value into `FairLocalGate`.

So the path is:

```text
process env
  -> EnvOverrides.gpu_concurrency
  -> Config._merged["gpu"]["concurrency"]
  -> FairLocalGate(concurrency=...)
```

That is the mental model for every env override in this file.

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

**(1) The model registry is too structured to live in env vars.** A
model registry entry has a model name, task type, priority, speed hint,
quality hint, and local/cloud location. A deep-panel pool is a list of
worker models plus a synthesizer and fallback. A KB config has
collections, dataset paths, chunking knobs, watcher settings, and
reconcile settings. YAML is good at this kind of shape.

**(2) The YAML is checked into git; the env vars aren't.** A new
deployment can `git clone` and immediately have a sensible default
config. The deployment-specific bits (URLs of Ollama / Qdrant / OWUI,
the Brave API key) go in `.env`, which is gitignored — they're
deployment secrets and shouldn't follow the repo around.

This is also why `TOOL_SERVERS` and `KB_DATASET_PATHS` default to
`None` in `EnvOverrides`. If those env vars are unset, the YAML lists
should win. If an operator explicitly sets the env var, the env var
wins. Silent env defaults that replace YAML lists are dangerous because
Audrey can boot with a config that looks correct in git but is not the
config actually running.

#### Why config changes require restart

`get_config()` is decorated with `@lru_cache(maxsize=1)`. In plain
English: "run this function once per process, remember the result, and
return the same object on future calls."

That is intentional. Config is a boot-time concern, not something every
route reloads from disk. If you change `config.yaml`, restart the
process. Restarting runs the whole boot sequence again and builds fresh
long-lived objects from the new merged config.

### 2.3 `lifespan`: the startup and shutdown owner

Open [`main.py:51`](../../src/audrey/main.py#L51):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    log.info("audrey starting; version=%s", __version__)
    ...
    try:
        yield
    finally:
        ...
```

This is an async context manager, the same pattern from Lesson 1's context
manager section. FastAPI runs the code before `yield` during startup. It keeps
the function suspended at `yield` while the app serves requests. On shutdown,
control resumes after `yield`, inside the `finally` block, so cleanup runs even
when the process is stopping because of a signal or an exception.

The important thing is ownership. The same function that creates long-lived
resources also closes them. That keeps lifecycle code in one place.

Here is what `lifespan` builds, grouped by role rather than by exact line
number:

| Boot step | What gets built | Why Audrey needs it | Where it goes |
|---|---|---|---|
| Load config | `Config` | One merged view of YAML + env | `app.state.cfg`; also passed into graph |
| Open model client | `OllamaClient` | Shared async HTTP client for model calls and embeddings | `app.state.ollama`; graph closure |
| Build model state | `ModelRegistry`, `HealthTracker` | Pick models by task and avoid failing ones | `app.state.registry`, `app.state.health`; graph closure |
| Build fairness controls | `FairLocalGate`, `UserInflightRegistry` | Protect local GPU and cap per-user concurrency | `app.state.gate`, `app.state.inflight`; gate also in graph closure |
| Discover tools | `ToolRegistry` | Let ReAct-capable models call custom-tools | `app.state.tools`; graph closure |
| Compile pipeline | compiled LangGraph | The request pipeline from Lesson 4 | `app.state.graph` |
| Connect KB | `QdrantKB`, `UploadsDB` | Global KB and per-user upload metadata | `app.state.qdrant`, `app.state.uploads_db` |
| Build embedders | `TextEmbedder`, `ImageEmbedder` | Convert text/images into vectors | `app.state.text_embedder`, `app.state.image_embedder` |
| Start background work | `KBWatcher`, `KBReconciler` | Re-ingest changed files and clean stale KB vectors | `app.state.kb_watcher`, `app.state.kb_reconciler` |

Two concepts in that table need a little extra care: graceful degradation and
background tasks.

#### Graceful degradation

Not every startup failure means the same thing.

If `config.yaml` is missing or malformed, Audrey cannot know which models,
tools, timeouts, and routing rules to use. There is no honest degraded mode.
[`_load_yaml()`](../../src/audrey/config.py#L137) lets that exception crash
startup.

If Qdrant is unreachable, Audrey loses some KB functionality, but it can still
serve ordinary chat completions, route prompts, call non-KB tools, and answer
without uploaded-file search. So the Qdrant boot calls are wrapped in
`try/except`: log the problem, keep the rest of the service alive, and let KB
routes fail later if they need Qdrant.

The pattern is:

```text
Fail fast when running would be a lie.
Degrade when a useful subset of Audrey still works.
```

#### Background tasks

The KB watcher and reconciler keep doing work after startup. They are not
ordinary "build this object and forget it" services.

The watcher reacts to filesystem changes under configured dataset paths and
queues re-ingest/delete work. The reconciler periodically checks for global KB
vectors whose source files disappeared. Both need Qdrant and embedding clients
to stay alive while they run.

That explains the shutdown order in `finally`: stop the background tasks first,
then close the objects they depend on. Closing Qdrant first would leave the
watcher or reconciler mid-flight with a dead client.

### 2.4 `app.state`: the app's service shelf

`app.state` is FastAPI's per-application storage object. You can attach
attributes to it during startup:

```python
app.state.cfg = cfg
app.state.graph = graph
app.state.tools = tool_registry
```

Then route handlers can retrieve those same objects from `request.app.state`.

Why not just create clients inside each route? Because these are long-lived
resources. Reopening an Ollama HTTP client, rebuilding the model registry, or
rediscovering tools on every request would be slower, noisier, and harder to
reason about. Startup builds them once; routes reuse them.

Why not create them as module-level globals? Because most of them need runtime
config, async startup, or cleanup. Module-level code runs during import, before
FastAPI's lifespan starts. The lifespan is the right place because it has an
event loop, can `await`, and owns shutdown.

So the pattern is:

```text
lifespan builds services once
  -> stores shared services on app.state
  -> routes read app.state during requests
```

### 2.5 Graph closures: why rediscover mutates in place

One subtle part of `main.py` is easy to miss if you are new to Python
closures.

During startup, Audrey does this:

```python
graph = build_graph(cfg, ollama, registry, health, gate, tool_registry)
app.state.tools = tool_registry
app.state.graph = graph
```

`build_graph(...)` defines node functions that use the objects passed into it.
Those node functions keep references to those objects. That is a **closure**:
an inner function remembers values from the environment where it was created.

A tiny example:

```python
def make_reader(registry):
    def read_names():
        return registry.names()
    return read_names
```

`read_names()` can still use `registry` even after `make_reader()` has
returned, because it captured that object.

Audrey's graph does the same thing with `tool_registry`. The snippet below is
the **correct pattern Audrey uses** in
[`rediscover_tools` (main.py:323)](../../src/audrey/main.py#L323). It refreshes
the existing registry object in place:

```python
fresh = await discover_all(tool_servers)
reg.by_name.clear()
reg.by_name.update(fresh.by_name)
```

The thing **not** to do is this:

```python
app.state.tools = fresh
```

That would make the FastAPI app point at the new registry, but the
already-compiled graph would still hold the old registry in its closure.
Mutating `reg.by_name` keeps the object identity the same while changing its
contents, so the graph sees the refreshed tools on the next request.

This is one of those details that feels small until it breaks tool use. Once
you know the closure exists, the in-place mutation is not  just clever; it is the
correct shape.

### 2.6 The readiness log

Near the end of startup, `lifespan` logs a single "ready" line with the
important boot facts: Ollama URL, task types, GPU concurrency, per-user
in-flight limit, discovered tools, Qdrant location, watcher state, reconciler
state, and whether the pipeline compiled.

When you are operating Audrey and asking "what did this process actually boot
with?", that readiness line is the first thing to find. It is the runtime truth
after YAML, env vars, discovery, and startup conditionals have all done their
work.

## 3. Comprehension Q&A

Try answering each yourself before reading the answer.

**1. You change `gpu.concurrency` in `config.yaml`. What has to happen before
the running app uses the new value?**

Restart Audrey. Config is loaded once per process by `get_config()`, and the
`FairLocalGate` is built once during `lifespan`. Changing YAML on disk does not
modify the already-created gate. Restarting the container runs the boot path
again: YAML is loaded, env overrides are merged, and a new gate is built from
the new value.

Live reload would be possible, but it would not be just "read the file again."
You would need to decide what happens to in-flight requests waiting on the old
gate, rebuild affected services, and make sure route handlers and graph nodes
all see the same new objects. Audrey deliberately keeps config reload simple:
change config, restart process.

**2. Why does `OLLAMA_HOST` belong in `.env` or compose environment instead of
`config.yaml`?**

Because it describes where this deployment is running, not Audrey's application
shape. On the Unraid network, Ollama may be reachable as `http://ollama:11434`.
In a laptop dev run, it may be somewhere else. That should not require editing
the checked-in application config.

By contrast, model pools, routing thresholds, timeouts, KB settings, and
tool-capable model lists describe Audrey itself. Those belong in YAML.

The short version: YAML for Audrey's design, env for deployment location and
secrets.

**3. What would happen if custom-tools were not healthy when Audrey discovers
tools?**

Tool discovery runs once during startup. Audrey asks each configured tool
server for `/openapi.json` and builds a `ToolRegistry` from whatever it finds.
If custom-tools is unreachable at that moment, the registry can be empty.

Audrey does not automatically retry discovery in the background. The live fix
is to call the admin `/v1/tools/rediscover` route after custom-tools is
healthy. The deployment fix is what compose already does: make `audrey-ai`
wait for `custom-tools` to pass its healthcheck before Audrey starts.

**4. Why does Audrey crash on a bad config file but keep booting when Qdrant is
temporarily unavailable?**

A bad config file means Audrey does not know its own model registry, routing
rules, timeouts, tools config, or KB config. Serving requests from that state
would be pretending the app is valid when it is not, so startup should fail.

Qdrant being down is different. Audrey can still authenticate users, route
requests, call models, and answer prompts that do not need KB search. The KB
parts degrade, but the whole orchestrator is not useless. That is why Qdrant
startup checks are caught and logged instead of crashing the process.

**5. What is `app.state` for?**

`app.state` is FastAPI's app-level storage for objects that should live for the
whole process and be reused by routes. Audrey uses it for config, model client,
model registry, health tracker, fairness controls, tools, compiled graph,
Qdrant, upload DB, embedders, watcher, and reconciler.

The key idea is lifetime. These objects are too expensive or too stateful to
rebuild on every request, and most need cleanup on shutdown. `lifespan` owns
their lifecycle; `app.state` makes them reachable during requests.

**6. Why does `/v1/tools/rediscover` mutate `reg.by_name` instead of assigning
`app.state.tools = fresh`?**

Because the compiled graph captured the original `ToolRegistry` object when
`build_graph(...)` ran. Replacing `app.state.tools` would only change the
attribute on the FastAPI app; the graph would still have the old object.

Mutating `reg.by_name` changes the contents of the same object. Both
`app.state.tools` and the graph closure still point at that object, so the next
request sees the refreshed tools.

**7. What does the `yield` inside `lifespan` mean?**

It marks the boundary between startup and shutdown.

Everything before `yield` runs before Audrey accepts requests. FastAPI waits
for that code to finish. When execution reaches `yield`, the app is considered
started and uvicorn can serve traffic. When the server shuts down, execution
resumes after `yield`, where Audrey stops background tasks and closes clients.

That is why lifespan is a good fit: the code that creates resources and the
code that cleans them up live in one function.


## When you're ready for the next lesson

The next lesson covers the model layer:
[Lesson 6 - The model layer](lesson-06-the-model-layer.md). We will trace what
happens when a pipeline node says "I need a general-purpose model": how the
registry ranks candidates, how the health tracker filters failing ones, how the
request becomes an HTTP call to Ollama, and how local and `:cloud`-suffixed
model names travel through the same client.
