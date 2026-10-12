# Lesson 5 — Configuration and startup

**Estimated time:** 60-80 minutes if you read slowly and keep the files open.

**Goal:** by the end of this lesson, you can answer
*"what happens between starting the containers and Audrey serving its first
request?"* You should understand what each container may reach and why, where
settings come from, why startup lives in FastAPI's `lifespan`, what gets stored
on `app.state`, and why some boot failures crash Audrey while others only
degrade part of the system.

Lesson 4 traced a message through a running Audrey. This lesson walks the
moment before that: the containers start, config gets loaded, the application
database is opened, long-lived clients, registries and background workers are
built, the LangGraph pipeline is compiled, and only then does FastAPI begin
accepting traffic.


## 1. Context

Audrey has two different kinds of code:

- **Request-time code** runs every time someone sends a message. That was
  Lesson 4: route handler, identity check, run, pipeline, model call,
  streaming response.
- **Boot-time code** runs once when the process starts. That is this lesson:
  config loading, opening the database, client construction, tool discovery,
  graph compilation, background workers, and cleanup on shutdown.

If request-time code is "what happens when someone asks Audrey a question,"
boot-time code is "what Audrey has to prepare before any question can be
answered."

Here is the whole startup story in one pass:

```text
docker compose starts audrey
  -> compose supplies env vars and mounts config.yaml
  -> Docker CMD runs uvicorn
  -> uvicorn imports audrey.main:app
  -> FastAPI creates the app object
  -> uvicorn starts FastAPI's lifespan
  -> lifespan loads config
  -> lifespan opens the application database and settles interrupted runs
  -> lifespan builds long-lived services: models, tools, skills, the KB stack
  -> lifespan compiles the LangGraph pipeline
  -> lifespan starts background workers
  -> lifespan stores shared objects on app.state
  -> lifespan takes a first readiness snapshot
  -> lifespan reaches yield
  -> Audrey is ready for requests
```

That `yield` is the hinge of the whole file. Everything before it is startup.
Everything after it is shutdown.


## 2. Read-along

Open these files in your editor as we go:

- [`compose.yaml`](../../compose.yaml): the five containers, with their
  networks, ports, mounts and environment.
- [`docker/audrey.Dockerfile`](../../docker/audrey.Dockerfile): the command
  that starts uvicorn inside the `audrey` container.
- [`config.yaml`](../../config.yaml): Audrey's checked-in runtime settings.
- [`src/audrey/config.py`](../../src/audrey/config.py): the YAML and
  env-var loader.
- [`src/audrey/main.py`](../../src/audrey/main.py): the FastAPI entrypoint
  and lifespan.

### 2.1 The containers as a trust map

Lesson 4 drew the containers as boxes and arrows. `compose.yaml` is where those
arrows become enforceable. Each service declares which networks it joins,
which ports it publishes, what it mounts and which environment it receives.
Read together, those four declarations answer one question per container:
*what can this reach, and what can reach it?*

| Container | Networks | Reachable from outside the server? | Mounts |
|---|---|---|---|
| `audrey` | `ollama-net`, `media-net`, `fetch-net` | yes: its API port, on the home network | `/data` read-write; config, skills and the KB folders read-only |
| `audrey-ui` | `ollama-net` | only through the server's loopback address, by cloudflared | none |
| `custom-tools` | `ollama-net` | no: no port published | its own data folder |
| `media-worker` | `media-net` only | no | `/data`, read-only |
| `media-fetcher` | `fetch-net` only | no | one staging folder for downloads, read-write |

Ollama and Qdrant sit on `ollama-net` too; they're managed outside this file.
Each row has a reason.

**`audrey` publishes its API port to the home network**
([`compose.yaml:70`](../../compose.yaml#L70)), because bots and Prometheus on
the home network call it directly. That's safe only because every route that
touches user data checks identity itself (Lesson 4 §2.2). Being reachable is not
the same as being allowed.

**`audrey-ui` publishes only on the loopback address**
([`compose.yaml:119`](../../compose.yaml#L119)), so the only way to the browser
app is through cloudflared, and therefore through Cloudflare Access.

**`custom-tools` publishes nothing.** None of the tools the models call check
identity: they trust the `user` value Audrey sends with each call, and Audrey
sets that value from the authenticated account, never from the model. A
service that trusts its caller that much must be reachable only by that
caller. The comments in
[the `custom-tools` stanza](../../compose.yaml#L135) record what went wrong
when its port was published.

**The media worker can reach Audrey and nothing else.** It joins only
[`media-net`](../../compose.yaml#L356), which is declared `internal: true`. The
worker parses video, audio and scanned PDFs that someone uploaded, the riskiest
input Audrey handles, so a compromised worker should have nowhere to go. It
must not reach Ollama either: its model work (describing video frames) has to
go through Audrey, where fair scheduling applies.

**The media fetcher can reach the internet, because downloading is its whole
job.** It joins only [`fetch-net`](../../compose.yaml#L363), an ordinary network
with a route out.

> **Concept spotlight — Docker networks, and why a missing name is not a
> missing route.** A Docker network is a virtual switch for containers.
> Containers on the same network reach each other by service name: Docker runs
> a small DNS server that answers `audrey` or `ollama` with the right
> container's address. A container on a *different* network can't resolve
> those names. That looks like isolation, but it isn't, quite. An ordinary
> Docker network has a route out through the host, so a container on it can
> still reach any port the *host* publishes on the home network, by IP
> address. Ollama publishes its port on the host, so a container that can't
> resolve `ollama` can still connect to the server's address on Ollama's port.
> The media worker's first version did exactly that. `internal: true` removes
> the route: a container that is only on an internal network can reach the
> other containers on that network, and nothing else. The comment above
> `media-net` in `compose.yaml` tells the story.
> **A missing name is not a missing route.**

The fetcher can't have `internal: true`, so its isolation is weaker, and the
comment on its stanza says so plainly: nothing in its configuration tells it
Ollama's address, but it does have a route there.

**Even environment variables are part of the map.** `audrey` and
`custom-tools` read the whole `.env` file (`env_file: .env`). The worker and the
fetcher deliberately don't; each lists only the variables it needs. `.env`
holds `OLLAMA_HOST` among other things, and handing the file to the fetcher
would give it the one address its design keeps from it, along with every other
secret in the file.

**Audrey is the only writer of `/data`.** The application database, the
uploads database and the uploaded files all live there. `audrey` mounts it
read-write (its [`volumes`](../../compose.yaml#L88) list); the worker mounts it
read-only; the fetcher writes only into a staging folder, and Audrey checks each
download before moving it into its owner's folder. With one writer, Audrey's
SQLite databases never have to coordinate writes from two processes, and no
other container can alter a stored file.

The rule behind the whole table: **each container gets the routes its job
needs, and nothing more.**

#### From compose to Python

Now narrow to the `audrey` service. It does four startup-relevant things before
Audrey's Python code runs.

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
container at `/app/config.yaml`, read-only. The image also contains a copy from
build time, but the runtime bind mount is the one that counts. The Dockerfile
sets `AUDREY_CONFIG=/app/config.yaml`, so `get_config()` reads that mounted
file. After a `git pull` that changes `config.yaml` on the server, recreate the
container (`docker compose up -d --force-recreate audrey`) rather than only
restarting it, so it starts from the pulled file.

**It waits for custom-tools.** Audrey discovers tools during startup by asking
custom-tools for `/openapi.json`. Its [`depends_on`](../../compose.yaml#L48)
entry, with `condition: service_healthy`, keeps Audrey from starting until
custom-tools passes its healthcheck. If discovery still comes back empty or
partial, Audrey retries in the background (§2.5).

Now open [`src/audrey/main.py`](../../src/audrey/main.py). When uvicorn imports
`audrey.main:app`, Python executes the module top to bottom and creates the
FastAPI [`app`](../../src/audrey/main.py#L479) object near the end of the file.
Importing the module does not mean Audrey is ready yet. The app exists, but
the expensive shared objects are built later, by `lifespan`.

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
class is [`EnvOverrides`](../../src/audrey/config.py#L24), which inherits from
`pydantic_settings.BaseSettings`.

The merge happens in `Config._apply_env_overrides` at
[`config.py:166`](../../src/audrey/config.py#L166). Read it top to bottom
— it's just a list of "if env var X was set, write its value into the
right slot of the YAML dict." The YAML dict is the merged result;
`Config.raw` exposes it.

#### A concrete value trace

You set `GPU_CONCURRENCY=2` in the environment Python sees at boot.
Here's what happens:

1. `lifespan()` calls [`get_config()`](../../src/audrey/config.py#L745).
2. `get_config()` creates `EnvOverrides()`.
3. Pydantic Settings sees `GPU_CONCURRENCY=2`, matches it to
   `gpu_concurrency`, and converts it to an integer.
4. `Config.__init__` stores the YAML dict on `self._merged`, then calls
   `_apply_env_overrides()`.
5. `_apply_env_overrides()` runs:

   ```python
   if (v := self.env.gpu_concurrency) is not None:
       self._set("GPU_CONCURRENCY", v, "gpu", "concurrency")
   ```

   `_set` walks the path `gpu` → `concurrency`, creating the `gpu` section if
   the YAML doesn't have one, and writes `2` there. It also records
   `GPU_CONCURRENCY=2` as an override that is active on this boot.

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

The record from step 5 is what startup prints. The first thing `lifespan` does
after loading config is log every active override, or say plainly that there
are none, at [`main.py:75`](../../src/audrey/main.py#L75):

```python
if overrides := cfg.active_env_overrides:
    log.warning("config: %d ENV OVERRIDE(S) active, config.yaml is not the "
                "whole truth: %s", len(overrides),
                ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items())))
else:
    log.info("config: no env overrides; config.yaml is authoritative")
```

The reason is in the docstring of
[`active_env_overrides`](../../src/audrey/config.py#L214): an override silently
displaces a value committed in `config.yaml`, so the file stops describing what
is running and nothing in it admits that. One log line makes "what is this box
actually running?" a search instead of a deduction.

#### What's a `BaseSettings`?

`pydantic_settings.BaseSettings` is Pydantic's "model whose fields
auto-populate from environment variables" subclass. You declare fields
with type annotations and (optionally) aliases; instantiating the
class reads the environment and validates each value against the
declared type — same machinery as a regular Pydantic model
([Lesson 2 §2](lesson-02-foundations-libraries.md#2-pydantic)) but
sourced from `os.environ` instead of an HTTP body.

`SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
extra="ignore")` on [`config.py:27`](../../src/audrey/config.py#L27)
configures three things: also read variables from `.env` if the file
is present, decode it as UTF-8, and silently drop env vars that don't
match any declared field (instead of erroring).

`alias="OLLAMA_HOST"` on the [`ollama_host`](../../src/audrey/config.py#L32)
field lets us write `cfg.env.ollama_host` in Python while the env var is
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
config. The deployment-specific bits (the addresses of Ollama and Qdrant,
the Brave API key, the Cloudflare Access settings, the internal service
token) go in `.env`, which is gitignored: they're deployment secrets and
shouldn't follow the repo around.

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
route reloads from disk. If you change `config.yaml`, start a new process
(on the server: recreate the container, as §2.1 explained). That runs the
whole boot sequence again and builds fresh long-lived objects from the new
merged config.

### 2.3 `lifespan`: the startup and shutdown owner

Open [`main.py:65`](../../src/audrey/main.py#L65):

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

The lifespan has grown with the application. Here is what it builds, in order,
grouped by role rather than by exact line number:

| Role | What gets built | Why Audrey needs it | Where it goes |
|---|---|---|---|
| Config | `Config` | One merged view of YAML and env | `app.state.cfg`; also passed into the graph |
| Identity | `CloudflareAccessVerifier`, when Access is enabled | Verify the browser's identity header | `app.state.cloudflare_access_verifier` |
| Canonical state | `ApplicationStore`; then interrupted runs are settled | Accounts, tokens, conversations, messages, runs | `app.state.application_store` |
| Model layer | `OllamaClient`, `ModelRegistry`, `HealthTracker` | Call models, pick them by task, avoid failing ones | `app.state.ollama`, `.registry`, `.health`; graph closure |
| Fairness | `FairLocalGate`, `UserInflightRegistry` | Share the local GPU; cap each person's concurrent work | `app.state.gate`, `.inflight` |
| Tools and skills | `ToolRegistry` (discovered), `SkillRegistry` | Tools the models may call; packaged instruction sets | `app.state.tools`, `.skills`; tools also in the graph closure |
| Pipeline | the compiled LangGraph graph | The pipeline from Lesson 4 | `app.state.graph` |
| KB stack | `QdrantKB`, `UploadsDB` (reconciled), `StorageLifecycle`, embedders, `KBWatcher`, `KBReconciler` | Search, upload records, file quotas, re-ingest and cleanup | `app.state.qdrant`, `.uploads_db`, and so on |
| Durable delivery | `FileDeletionWorker`, `ChatArchiveQueue`, `ChatProjectionPromoter`, `UserDataPurgeCoordinator` | Work that must finish even across a restart: deleting files, archiving turns, purging an account | `app.state.file_deletions`, `.archive_queue`, `.archive_projector`, `.user_data_purges` |
| Native runs | `ConversationTitleGenerator`, `NativeRunManager` | Name new conversations; own live runs | `app.state.conversation_titles`, `.native_runs` |
| Readiness | `ReadinessCollector`, plus a first snapshot | One report on what's up and what's degraded | `app.state.readiness` |

You don't need to know these classes yet; later lessons take them one at a
time. What matters here is the shape: one function builds all of them, stores
them on `app.state`, and closes them again. Three things in that table need
extra care: settling interrupted runs, graceful degradation, and background
workers.

#### Settling interrupted runs

Opening the application database does real work. The
[`ApplicationStore`](../../src/audrey/app_state/store.py#L83) constructor
connects to SQLite, switches on the settings Audrey relies on, and applies any
pending schema migrations before anything else can touch the data. Then
`lifespan` settles runs a previous process left behind, at
[`main.py:108`](../../src/audrey/main.py#L108):

```python
recovered_runs = await application_store.conversations.recover_interrupted_runs()
```

Lesson 4 showed that a run has two halves: a durable record in the database and
a live task in memory. A crash or a power cut destroys the task and leaves the
record saying `running`, which nothing would ever change.
[`recover_interrupted_runs`](../../src/audrey/app_state/repositories.py#L1100)
marks every such run as failed with the error code `server_restart`, and its
half-written answer as incomplete, before Audrey serves its first request. A
reloaded page then shows an honest ending instead of an answer that never
finishes.

#### Graceful degradation

Not every startup failure means the same thing.

If `config.yaml` is missing or malformed, Audrey cannot know which models,
tools, timeouts, and routing rules to use. There is no honest degraded mode.
[`_load_yaml()`](../../src/audrey/config.py#L262) lets that exception crash
startup, and so do the checks `get_config()` runs on several config sections.
The same goes for Cloudflare Access switched on without its settings, and for
an application database that can't be opened or migrated: without them, Audrey
couldn't tell who anyone is or what they own.

If Qdrant is unreachable, Audrey loses some KB functionality, but it can still
serve ordinary chat, route prompts, call non-KB tools, and answer without
uploaded-file search. So the Qdrant boot calls are wrapped in `try/except`:
log the problem, keep the rest of the service alive, and let KB routes fail
later if they need Qdrant. Tool discovery degrades the same way (§2.5), and so
do the skill registry, one bundle at a time, and archive delivery, whose
outbox simply holds its rows until the archive is back.

The pattern is:

```text
Fail fast when running would be a lie.
Degrade when a useful subset of Audrey still works.
```

#### Background workers, and the shutdown order

Several of the objects in the table keep working after startup. The KB watcher
reacts to file changes under the dataset folders; the reconciler periodically
removes vectors whose source files disappeared. The durable-delivery workers
retry their queues until each item is done: file deletions, archive
deliveries, account purges. `NativeRunManager` holds every live run.

Ordering matters at startup too. The uploads reconcile,
[`reconcile_with_qdrant`](../../src/audrey/main.py#L206), must finish before
uvicorn accepts traffic, because it isn't safe alongside uploads that are in
progress. It restores missing legacy catalog rows, preserves existing
ownership and lifecycle metadata, and reports missing search content. Empty
or partially lost search collections never delete upload records. The
lifespan runs this before `yield`, so that ordering is structural.

At shutdown, the `finally` block at
[`main.py:398`](../../src/audrey/main.py#L398) stops things in a deliberate
order:

```python
finally:
    await native_runs.stop()
    if tools_retry_task is not None and not tools_retry_task.done():
        tools_retry_task.cancel()
        try:
            await tools_retry_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 — shutdown path
            pass
    if reconciler is not None:
        await reconciler.stop()
    if watcher is not None:
        await watcher.stop()
    await user_data_purges.stop()
    await archive_projector.stop()
    await archive_queue.stop()
    await file_deletions.stop()
    application_store.close()
    uploads_db.close()
    qdrant.close()
    await ollama.aclose()
    await archive_http.aclose()
```

Read it as **"stop producers before the queues they feed, and close the stores
last."** Live runs go first: they're cancelled with the error code
`server_shutdown`, and each one records its ending while the database is still
open. A finished run wakes the archive projector, the projector feeds the
archive queue, and a purge feeds both file deletions and the archive queue, so
those stop in that order. Only then are the databases and clients closed.
Closing Qdrant first would leave the watcher mid-flight with a dead client;
closing the application database first would leave a run nowhere to record that
it was cancelled.

### 2.4 `app.state`: the app's service shelf

`app.state` is FastAPI's per-application storage object. You can attach
attributes to it during startup:

```python
app.state.cfg = cfg
app.state.application_store = application_store
app.state.graph = graph
app.state.tools = tool_registry
```

Then route handlers can retrieve those same objects from `request.app.state`.
Routes usually do it through small helpers that fail clearly when a service
isn't there, like [`_store`](../../src/audrey/routes/app/runs.py#L835) in the
runs module:

```python
def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Audrey application state is unavailable.")
    return store
```

A missing service becomes an honest `503 Service Unavailable`, not an
`AttributeError` deep inside a route.

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

During startup, Audrey compiles the graph at
[`main.py:180`](../../src/audrey/main.py#L180):

```python
graph = build_graph(
    cfg, ollama, registry, health, gate, tool_registry, archive_http,
)
```

and later stores both the graph and the registry on `app.state`.

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

Audrey's graph does the same thing with `tool_registry`. So when tools are
rediscovered, the existing registry object has to be refreshed in place. Here
is the admin route that does it,
[`rediscover_tools`](../../src/audrey/main.py#L557):

```python
fresh = await discover_all(tool_servers)
if tool_servers and not fresh.by_name:
    log.warning("tools: rediscover returned no records; keeping the live registry")
    raise HTTPException(
        status_code=503,
        detail="Tool discovery returned no records; existing tools are unchanged.",
    )
reg.by_name.clear()
reg.by_name.update(fresh.by_name)
```

Look at the last two lines first. The thing **not** to do is this:

```python
app.state.tools = fresh
```

That would make the FastAPI app point at the new registry, but the
already-compiled graph would still hold the old registry in its closure.
Mutating `reg.by_name` keeps the object identity the same while changing its
contents, so the graph sees the refreshed tools on the next request.

Now the guard above them. `clear()` followed by `update()` replaces whatever
the registry held with whatever discovery returned. If custom-tools happened to
be restarting, discovery would return nothing, and an admin trying to *fix*
the tools would wipe the working ones. So an empty answer is refused with a 503
and the live registry stays as it was.

The same in-place refresh runs without anyone asking, too. If the first
discovery at startup comes back empty, or with some tools unavailable,
[`_retry_tool_discovery`](../../src/audrey/main.py#L421) retries in the
background for about two minutes, mutating the same registry when it succeeds.
After that window, the admin route is the way to retry.

This is one of those details that feels small until it breaks tool use. Once
you know the closure exists, the in-place mutation is not just clever; it is
the correct shape.

### 2.6 Readiness: the ready line and the snapshot

Near the end of startup, `lifespan` still logs a single "ready" line with the
important boot facts: Ollama URL, task types, GPU concurrency, per-user in-flight
limit, discovered tools, Qdrant location, watcher state, reconciler state, and
whether the pipeline compiled. When you're asking "what did this process boot
with?", that line is the first thing to find.

But a log line describes one moment, and the question an operator usually has
is "what's wrong *now*?" So startup also builds a
[`ReadinessCollector`](../../src/audrey/readiness.py#L130) and takes a first
snapshot before `yield`. A snapshot reports each component (Ollama, Qdrant,
custom-tools, skills, the chat archive, the KB watcher and reconciler) as
available, unavailable or disabled. It also covers which tools are missing, how
deep the work queues are, whether the background workers are running, and how
busy the GPU gate is. It boils all of that down to one status:

- **ready**: everything is up;
- **degraded**: something optional is down, but chat works;
- **unready**: a *required* component is down. By default only Ollama is
  required ([`readiness`](../../config.yaml#L1387) in `config.yaml`), because
  without it Audrey can't answer at all.

The snapshot can be read three ways: admins call `GET /v1/admin/readiness`
(which answers 503 when the status is unready), the browser asks
`/api/capabilities` for a coarse version that names no hosts or tools, and
Prometheus scrapes it as metrics from `/metrics`. Snapshots are cached for a
few seconds, so a dashboard and a person refreshing at once don't double the
probes. The course's closing lesson, on operating Audrey, reads a snapshot in
detail.

That's degraded startup made visible. Qdrant down at boot is no longer a log
line you have to go looking for: it's a `degraded` status anyone with access can
see, until Qdrant comes back.


## 3. Comprehension Q&A

Try answering each yourself before reading the answer.

**1. Someone adds `env_file: .env` to the `media-fetcher` stanza so it can read
one more setting. Why does `compose.yaml` warn against exactly that?**

Because `.env` holds `OLLAMA_HOST`, and the fetcher is the one container that
has a real route to Ollama. It sits on an ordinary network with internet
access, so it can reach any port the host publishes. Its only protection was
that nothing told it Ollama's address. `env_file: .env` hands it that address,
plus every other secret in the file. The right fix is what the stanza already
does: name the one extra variable under `environment:`, so compose reads it
from `.env` without passing the whole file in.

**2. You change `gpu.concurrency` in `config.yaml`. What has to happen before
the running app uses the new value?**

Recreate the `audrey` container. Config is loaded once per process by
`get_config()`, and the `FairLocalGate` is built once during `lifespan`.
Changing YAML on disk does not modify the already-created gate. A new process
runs the boot path again: YAML is loaded, env overrides are merged, and a new
gate is built from the new value.

Live reload would be possible, but it would not be just "read the file again."
You would need to decide what happens to in-flight requests waiting on the old
gate, rebuild affected services, and make sure route handlers and graph nodes
all see the same new objects. Audrey deliberately keeps config reload simple:
change config, start a new process.

**3. Why does `OLLAMA_HOST` belong in `.env` or compose environment instead of
`config.yaml`?**

Because it describes where this deployment is running, not Audrey's application
shape. On the server's Docker network, Ollama may be reachable as
`http://ollama:11434`. In a laptop dev run, it may be somewhere else. That
should not require editing the checked-in application config.

By contrast, model pools, routing thresholds, timeouts, KB settings, and
tool-capable model lists describe Audrey itself. Those belong in YAML.

The short version: YAML for Audrey's design, env for deployment location and
secrets.

**4. custom-tools is still starting when Audrey discovers tools, and discovery
comes back empty. What happens next?**

Audrey starts anyway, with an empty tool registry, and readiness reports the
missing tools. Because the first discovery came back empty, `lifespan` starts
`_retry_tool_discovery`, which tries again every few seconds for about two
minutes. When custom-tools answers, it refreshes the same registry object in
place, so the compiled graph picks the tools up on its next request without a
restart. If custom-tools stays down past that window, an admin can call
`/v1/tools/rediscover` once it's healthy. That route refuses an empty answer
with a 503 rather than replacing the live registry with nothing. The deployment
side already does its part: compose makes `audrey` wait for custom-tools'
healthcheck before starting, so this path is for the unlucky case.

**5. Why does Audrey crash on a bad config file but keep booting when Qdrant is
temporarily unavailable?**

A bad config file means Audrey does not know its own model registry, routing
rules, timeouts, tools config, or KB config. Serving requests from that state
would be pretending the app is valid when it is not, so startup should fail.

Qdrant being down is different. Audrey can still authenticate people, route
requests, call models, and answer prompts that do not need KB search. The KB
parts degrade, readiness reports `degraded`, and the rest of the application
works. That is why Qdrant startup checks are caught and logged instead of
crashing the process. The application database is on the other side of the
line: if it can't be opened, Audrey can't tell who anyone is or what they own,
so startup fails.

**6. The server loses power in the middle of an answer. What does that
conversation show after Audrey comes back, and which code decides it?**

The answer shows as interrupted, not as still running. Before serving any
request, `lifespan` calls `recover_interrupted_runs`, which finds every run the
database still calls `running`. No task can be running for them in a process
that just started, so it marks them failed with the error code `server_restart`
and marks their half-written answers incomplete. Whatever had already been
saved is still there; the live task, and anything it hadn't yet written, is
gone.

**7. What is `app.state` for?**

`app.state` is FastAPI's app-level storage for objects that should live for the
whole process and be reused by routes. Audrey uses it for the config, the
application database, the identity verifier, the model client, registry and
health tracker, the fairness controls, tools and skills, the compiled graph,
the KB stack, the durable-delivery workers, the run manager and the readiness
collector.

The key idea is lifetime. These objects are too expensive or too stateful to
rebuild on every request, and most need cleanup on shutdown. `lifespan` owns
their lifecycle; `app.state` makes them reachable during requests.

**8. Why does `/v1/tools/rediscover` mutate `reg.by_name` instead of assigning
`app.state.tools = fresh`, and why does it check for an empty result first?**

Because the compiled graph captured the original `ToolRegistry` object when
`build_graph(...)` ran. Replacing `app.state.tools` would only change the
attribute on the FastAPI app; the graph would still have the old object.
Mutating `reg.by_name` changes the contents of the same object. Both
`app.state.tools` and the graph closure still point at that object, so the next
request sees the refreshed tools.

The empty check exists because clearing and refilling replaces everything. A
discovery that returned nothing (custom-tools restarting, say) would turn a
working registry into an empty one, so the route refuses with a 503 and leaves
the live tools alone.

**9. What does the `yield` inside `lifespan` mean?**

It marks the boundary between startup and shutdown.

Everything before `yield` runs before Audrey accepts requests. FastAPI waits
for that code to finish. When execution reaches `yield`, the app is considered
started and uvicorn can serve traffic. When the server shuts down, execution
resumes after `yield`, where Audrey stops its live runs and background workers
and closes its stores and clients.

That is why lifespan is a good fit: the code that creates resources and the
code that cleans them up live in one function.


## When you're ready for the next lesson

The next lesson covers the model layer:
[Lesson 6 - The model layer](lesson-06-the-model-layer.md). We will trace what
happens when a pipeline node says "I need a general-purpose model": how the
registry ranks candidates, how the health tracker filters failing ones, how the
request becomes an HTTP call to Ollama, and how local and `:cloud`-suffixed
model names travel through the same client.
