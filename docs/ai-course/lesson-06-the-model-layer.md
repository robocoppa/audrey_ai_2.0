# Lesson 6 - The model layer

**Estimated time:** 55-75 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer
*"when Audrey decides it needs a general, code, reasoning, or vision model, how
does it choose one, call it, and recover if that model fails?"*

Lesson 5 showed that startup builds three long-lived model-layer objects:
`OllamaClient`, `ModelRegistry`, and `HealthTracker`. This lesson follows those
objects at request time. We will trace how a virtual model like `audrey_fast` or
`audrey_deep` turns into one or more concrete Ollama model names, why local and
cloud models go through the same client, and why a failing model is cooled down
instead of permanently removed.


## 1. Context

Audrey exposes **virtual models** to Open WebUI and API clients:

- `audrey_auto`
- `audrey_fast`
- `audrey_deep`
- `audrey_cloud`
- `audrey_local`

Those are not Ollama model names. They are Audrey routing choices. A user asks
for `audrey_fast`, but Audrey may actually call a concrete model such as
`qwen3.6:35b` or `deepseek-v4-pro:cloud`, depending on the task type, model
health, and config.

The model layer is the part of Audrey that answers:

```text
The pipeline says: "I need a general model."

Which concrete model should that mean right now?
Is the preferred model healthy?
Is it local or cloud?
Should it wait for the local GPU gate?
How do we call Ollama?
What happens if the call fails?
```

Here is the whole model-layer path in one pass:

```text
config.yaml model_registry
  -> ModelRegistry builds ranked ModelSpec lists
  -> HealthTracker says which models are temporarily cooling down
  -> pipeline chooses fast path, deep workers, or synthesizer
  -> FairLocalGate serializes local model calls
  -> OllamaClient sends HTTP to Ollama
  -> success clears model health
  -> failure records cooldown and lets fallbacks try
```

Four ideas are doing most of the work:

| Piece | What it knows | What it does not know |
| --- | --- | --- |
| `ModelRegistry` | Which models exist for each task, their priority, and their local/cloud location. | Whether Ollama is currently reachable or whether a model just failed. |
| `HealthTracker` | Which model names are temporarily cooling down after failures. | Which models are best for code, reasoning, or general work. |
| `FairLocalGate` | Whether a local generation may enter the GPU slot right now. | How to choose models or call Ollama. |
| `OllamaClient` | How to speak Ollama's HTTP API. | Which model should be chosen for a task. |

That separation is the point. If model choice, health, GPU fairness, and HTTP
transport all lived in one function, Audrey would be harder to reason about and
harder to test.


## 2. Read-along

These are the files we'll reference in this lesson, open each one as we go:

- [`config.yaml`](../../config.yaml#L15) - the model registry and
  deep-panel pools.
- [`src/audrey/main.py`](../../src/audrey/main.py#L53) - where the
  model-layer objects are built at startup.
- [`src/audrey/models/registry.py`](../../src/audrey/models/registry.py#L16)
  - the ranked model registry.
- [`src/audrey/models/health.py`](../../src/audrey/models/health.py#L18)
  - temporary model cooldowns.
- [`src/audrey/models/ollama.py`](../../src/audrey/models/ollama.py#L29)
  - Audrey's async HTTP client for Ollama.
- [`src/audrey/pipeline/fast_path.py`](../../src/audrey/pipeline/fast_path.py#L31)
  - single-model selection.
- [`src/audrey/pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py#L52)
  - worker-pool selection.
- [`src/audrey/pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py#L82)
  - synthesizer primary/fallback selection.

Don't get too bogged down with how much code is contained within these files. We're going to focus on specific sections that are pertinent to what we're learning. By the end of the course you may be surprised how differently you see the code compared to when you started.

### 2.1 Startup builds the shared model objects

Start in [`src/audrey/main.py`](../../src/audrey/main.py#L53), inside
`lifespan`. Lesson 5 covered this function as startup code. This time, focus
only on the model-layer objects:

```python
default_timeout = float(cfg.timeouts.get("medium", 180))
ollama = OllamaClient(cfg.env.ollama_host, default_timeout_s=default_timeout)
registry = ModelRegistry(cfg)
health = HealthTracker()
gpu_concurrency = int(cfg.raw.get("gpu", {}).get("concurrency", 1))
gate = FairLocalGate(concurrency=gpu_concurrency)
```

Then `lifespan` passes those objects into `build_graph(...)` at
[`main.py:89`](../../src/audrey/main.py#L89):

```python
graph = build_graph(cfg, ollama, registry, health, gate, tool_registry)
```

This is the same closure idea from Lesson 5. The compiled graph keeps
references to these objects. Every request uses the same `OllamaClient`, the
same `ModelRegistry`, the same `HealthTracker`, and the same `FairLocalGate`.
The same objects are also stored on `app.state` starting at
[`main.py:132`](../../src/audrey/main.py#L132), so routes can reach them too.

That matters because health is process-local memory. If a model times out on
one request, the next request should know to avoid it for a little while. That
only works because the health tracker is shared across requests instead of
rebuilt every time.

The mental model:

```text
startup creates the model-layer objects once
  -> request handlers and graph nodes reuse them many times
```

### 2.2 `config.yaml` is the source of model choices

Open [`config.yaml`](../../config.yaml#L15) and find `model_registry`.

The registry is grouped by **task type**:

- `code`
- `reasoning`
- `general`
- `vl`

Those task types come from the classification step in the pipeline. The user
does not choose them directly. Audrey classifies the prompt and then asks the
model layer for a model that fits the task.

A registry entry looks like this at
[`config.yaml:38`](../../config.yaml#L38):

```yaml
- { name: "qwen3.6:35b", priority: 100, speed: 75, quality: 92, location: local }
```

Read that as:

- `name` is the concrete Ollama model name Audrey will send to `/api/chat`.
- `priority` is the current ranking. Higher priority wins when the model is
  healthy.
- `speed` and `quality` are hints carried with the model. The registry stores
  them, but the simple first-choice path sorts by priority.
- `location` tells Audrey whether the model should count as `local` or `cloud`
  for scheduling.

The important distinction:

```text
Task type chooses a registry list.
Priority orders models inside that list.
Health temporarily removes models from consideration.
Location decides whether local GPU fairness applies.
```

Below `model_registry`, `config.yaml` also defines deep-panel pools:

- [`deep_panel`](../../config.yaml#L76)
- [`deep_panel_cloud`](../../config.yaml#L96)
- [`deep_panel_local`](../../config.yaml#L116)

Those pools are different from the registry. The registry is a ranked menu of
possible models by task. A deep-panel pool is a more explicit recipe: "for this
virtual model and task type, run these workers, then use this synthesizer."

That gives Audrey three model-selection places to understand:

| Path | Selection style |
| --- | --- |
| Fast path | Pick the highest-priority healthy model from `model_registry`. |
| Deep panel | Start from the configured worker pool, then filter by health and scheduling limits. |
| Synthesizer | Use the pool's configured primary synthesizer, then fallback synthesizer. |

### 2.3 `ModelSpec`: one model entry as Python data

Open [`src/audrey/models/registry.py`](../../src/audrey/models/registry.py#L16).

Near the top are two `Literal` types at
[`registry.py:16`](../../src/audrey/models/registry.py#L16):

```python
TaskType = Literal["code", "reasoning", "general", "vl"]
Location = Literal["local", "cloud"]
```

`Literal` means "this string is not just any string; it must be one of these
specific values." Type checkers and editors can use that to catch mistakes.
But Python type hints do not automatically validate YAML at runtime. YAML is
just data loaded from a file.

That is why Audrey also has `_parse_location(...)` at
[`registry.py:84`](../../src/audrey/models/registry.py#L84):

```python
def _parse_location(raw: object, *, task: str, model: str) -> Location:
    if isinstance(raw, str) and raw in _VALID_LOCATIONS:
        return cast(Location, raw)
    ...
    raise ValueError(...)
```

This is a small but important pattern:

```text
Literal helps the type checker.
Runtime validation protects the running program.
```

If someone mistypes `cloud` as `clodu` in `config.yaml`, Audrey should fail at
startup. It should not silently treat a local model as "not local" and bypass
the GPU gate.

Now look at `ModelSpec` at
[`registry.py:22`](../../src/audrey/models/registry.py#L22):

```python
@dataclass(slots=True, frozen=True)
class ModelSpec:
    name: str
    priority: int
    speed: int
    quality: int
    location: Location
```

This is the Python shape of one registry entry.

`@dataclass` means Python writes the boring constructor for you. Instead of
manually writing an `__init__` that assigns `self.name`, `self.priority`, and
so on, you declare the fields and Python builds the method.

`frozen=True` means a `ModelSpec` should not be changed after it is created. If
the config changes, Audrey should restart and build a new registry; request-time
code should not mutate one model's priority in place.

`slots=True` means instances have a fixed set of attributes. It saves a little
memory and prevents accidental new attributes like `spec.priorty` from being
attached by typo.

The short version: `ModelSpec` is a small, immutable record for one concrete
model.

### 2.4 `ModelRegistry`: from YAML lists to ranked candidates

Stay in [`registry.py`](../../src/audrey/models/registry.py#L36) and read
`ModelRegistry.__init__`.

For each task in `cfg.model_registry`, Audrey turns the raw YAML dictionaries
into `ModelSpec` objects:

```python
ModelSpec(
    name=entry["name"],
    priority=int(entry.get("priority", 0)),
    speed=int(entry.get("speed", 50)),
    quality=int(entry.get("quality", 50)),
    location=_parse_location(...),
)
```

Then it sorts the list at
[`registry.py:53`](../../src/audrey/models/registry.py#L53):

```python
specs.sort(key=lambda s: s.priority, reverse=True)
```

`lambda s: s.priority` is a tiny function used only for sorting. It tells
Python, "when comparing two `ModelSpec` objects, use their `priority` field."

`reverse=True` means highest priority first.

After startup, the registry has an internal dictionary shaped like this:

```text
{
  "general": [highest-priority ModelSpec, next ModelSpec, ...],
  "code": [...],
  "reasoning": [...],
  "vl": [...]
}
```

There are two request-time methods to understand.

First, `candidates(task)` at
[`registry.py:56`](../../src/audrey/models/registry.py#L56):

```python
def candidates(self, task: TaskType) -> list[ModelSpec]:
    return list(self._by_task.get(task, ()))
```

This returns a copy of the list for that task. The copy matters. If a caller
does `candidates("general").pop()`, it should not mutate the registry's stored
list.

Second, `first_healthy(task, is_healthy)` at
[`registry.py:59`](../../src/audrey/models/registry.py#L59):

```python
def first_healthy(self, task: TaskType, is_healthy) -> ModelSpec | None:
    for spec in self._by_task.get(task, ()):
        if is_healthy(spec.name):
            return spec
    return None
```

This is the fast path's main selection helper. It walks the already-sorted list
and returns the first model whose name passes the `is_healthy` check.

The call site in the fast path looks like this at
[`fast_path.py:37`](../../src/audrey/pipeline/fast_path.py#L37):

```python
spec = registry.first_healthy(task, health.is_healthy)
```

Notice the subtle Python idea: Audrey passes `health.is_healthy`, not
`health.is_healthy(...)`.

That means "here is a function you can call later." `first_healthy` calls it
once per candidate model name.

Why not make the registry import `HealthTracker` directly? Because the registry
does not need to know what "healthy" means. It only needs a yes/no predicate.
That keeps `ModelRegistry` reusable and easy to test.

### 2.5 `HealthTracker`: temporary cooldown, not deletion

Open [`src/audrey/models/health.py`](../../src/audrey/models/health.py#L18).

The health tracker answers one question:

```text
Should Audrey avoid this model right now?
```

It does **not** mean "is this model installed?" or "is this model good?" It only
tracks recent failures in this Python process.

The private `_HealthState` dataclass stores this state at
[`health.py:18`](../../src/audrey/models/health.py#L18):

```python
consecutive_failures: int
cooldown_until: float
last_error: str
history: list[tuple[float, str]]
```

Each model gets one `_HealthState` only after it fails. A model with no state is
considered healthy by `is_healthy(...)` at
[`health.py:44`](../../src/audrey/models/health.py#L44):

```python
def is_healthy(self, model: str) -> bool:
    state = self._by_model.get(model)
    if state is None:
        return True
    return time.monotonic() >= state.cooldown_until
```

`time.monotonic()` is Python's "time that only moves forward" clock. It is good
for durations and deadlines because it is not affected if the system clock is
adjusted. Wall-clock time answers "what time is it?" Monotonic time answers
"has enough time passed?"

When a model fails, `record_failure(...)` increments the failure count and sets
a cooldown at [`health.py:53`](../../src/audrey/models/health.py#L53):

```python
backoff = min(self._base * (2 ** (state.consecutive_failures - 1)), self._max)
state.cooldown_until = time.monotonic() + backoff
```

This is **exponential backoff**. The first failure cools down briefly. Repeated
failures cool down for longer, up to a cap.

Why do that? Because failures are often temporary:

- Ollama is busy.
- A cloud model times out.
- The network hiccups.
- A large local model is slow to load.

Audrey should not delete that model from the registry. It should step around it
for a bit, try another model if possible, and let the failed model become
eligible again later.

When a model succeeds, `record_success(...)` clears its failure state at
[`health.py:50`](../../src/audrey/models/health.py#L50):

```python
def record_success(self, model: str) -> None:
    self._by_model.pop(model, None)
```

That is a nice little line. It means "remove the model from the failure map if
it is there; do nothing if it is not." A successful call restores the model to
normal.

The mental model:

```text
failure -> cool down temporarily
cooldown expires -> model can be tried again
success -> forget past failures
```

### 2.6 Fast path: one model answers

Now open
[`src/audrey/pipeline/fast_path.py:31`](../../src/audrey/pipeline/fast_path.py#L31).

This file is used when Audrey is in fast mode: one concrete model should answer
the request. Fast mode can happen because the user selected `audrey_fast`, or
because `audrey_auto` decided the prompt was small enough to answer directly.

```python
def pick_fast_model(
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
) -> ModelSpec:
    spec = registry.first_healthy(task, health.is_healthy)
    if spec is None:
        raise OllamaError(f"No healthy model available for task={task}")
    return spec
```

That is almost the whole story:

```text
task type -> ranked registry list -> first model not cooling down
```

Then `run_fast_path(...)` decides whether to use tools at
[`fast_path.py:71`](../../src/audrey/pipeline/fast_path.py#L71):

```python
use_tools = bool(
    tools and tools.by_name
    and tool_capable_models is not None
    and spec.name in tool_capable_models
)
```

If the selected model is tool-capable and tools are registered, Audrey runs the
ReAct loop. ReAct gets its own lesson later. For now, know that ReAct still uses
the same selected model; it just gives the model a chance to call tools before
writing the final answer.

If tools are not used, the fast path is a single Ollama chat call at
[`fast_path.py:85`](../../src/audrey/pipeline/fast_path.py#L85):

```python
async with gate.acquire(spec.name, location=spec.location, user_id=user_id):
    resp = await ollama.chat(...)
health.record_success(spec.name)
```

This line is where `location` matters.

If `spec.location == "local"`, `FairLocalGate` may make the request wait for a
GPU slot. If `spec.location == "cloud"`, the gate is a no-op and the call
continues immediately.

Both local and cloud models still go through `OllamaClient`. In Audrey's setup,
`:cloud` models are exposed through the same Ollama-compatible API surface.
"Cloud" is not a different Python client here; it is a scheduling and
concurrency hint.

If the Ollama call raises `OllamaError`, the fast path records a failure at
[`fast_path.py:92`](../../src/audrey/pipeline/fast_path.py#L92):

```python
except OllamaError as e:
    health.record_failure(spec.name, str(e))
    raise
```

That is the model layer's central contract:

```text
success -> record_success(model)
OllamaError -> record_failure(model, error)
```

Once a model has recorded a failure, the next request's
`registry.first_healthy(...)` call can skip it until the cooldown expires.

### 2.7 Deep panel: several workers answer

Open
[`src/audrey/pipeline/deep_panel.py:52`](../../src/audrey/pipeline/deep_panel.py#L52).

Deep mode is different from fast mode. Instead of asking the registry for one
best model, Audrey starts from a configured worker pool.

The virtual model chooses the pool through `_POOL_KEYS`:

```python
_POOL_KEYS = {
    "audrey_deep": "deep_panel",
    "audrey_cloud": "deep_panel_cloud",
    "audrey_local": "deep_panel_local",
}
```

Then `select_workers(...)` reads the pool for the current task at
[`deep_panel.py:103`](../../src/audrey/pipeline/deep_panel.py#L103):

```python
pool = cfg.raw.get(pool_key, {}).get(task, {})
raw_workers: list[str] = list(pool.get("workers", []) or [])
```

The workers in the pool are model names, not full `ModelSpec` objects. So
Audrey looks up each worker's location with `registry.location_of(...)` at
[`registry.py:69`](../../src/audrey/models/registry.py#L69):

```python
def location_of(self, model: str) -> Location:
    for specs in self._by_task.values():
        for spec in specs:
            if spec.name == model:
                return spec.location
    return "local"
```

This method scans the registry and returns the model's declared location. If
it cannot find the model, it defaults to `local`. That default is conservative
for scheduling: an unknown model should not bypass the local gate by accident.

`select_workers(...)` also filters by health at
[`deep_panel.py:109`](../../src/audrey/pipeline/deep_panel.py#L109):

```python
if not health.is_healthy(name):
    log.info("deep_panel: skipping unhealthy worker %s", name)
    continue
```

And it caps cloud workers at
[`deep_panel.py:114`](../../src/audrey/pipeline/deep_panel.py#L114):

```python
if loc == "cloud":
    if cloud_count >= max_workers_cloud:
        continue
    cloud_count += 1
```

So deep-panel worker selection is:

```text
virtual model -> pool key
pool key + task type -> configured worker names
worker name -> registry location
health tracker -> skip cooling-down workers
cloud cap -> avoid too many cloud workers at once
```

If the configured pool has no healthy workers, Audrey falls back to healthy
registry candidates for that task. The non-streaming fallback path starts at
[`deep_panel.py:285`](../../src/audrey/pipeline/deep_panel.py#L285), and the
streaming version does the same kind of fallback at
[`deep_panel.py:382`](../../src/audrey/pipeline/deep_panel.py#L382). That keeps
deep mode from becoming brittle when a pool entry is temporarily unavailable.

Each worker runs through `_run_one_worker(...)` (defined at
[`deep_panel.py:121`](../../src/audrey/pipeline/deep_panel.py#L121)). The most
important behavior is in its docstring at
[`deep_panel.py:141`](../../src/audrey/pipeline/deep_panel.py#L141):

```python
"""Execute one worker. Always returns a WorkerDraft — never raises."""
```

A deep worker failure becomes a draft with an `error` field at
[`deep_panel.py:207`](../../src/audrey/pipeline/deep_panel.py#L207):

```python
except OllamaError as e:
    health.record_failure(model, str(e))
    return WorkerDraft(model=model, content="", error=str(e)[:300], ...)
```

That is different from the fast path. In fast mode, one model is the whole
answer path, so a failure bubbles up. In deep mode, one failed worker does not
mean the whole request is useless. Another worker may still produce a good
draft, and the synthesizer can still answer.

The mental model:

```text
fast path: one model fails -> the fast attempt fails
deep panel: one worker fails -> keep the error as a draft and continue
```

### 2.8 Synthesis: primary, fallback, then graceful degradation

Open
[`src/audrey/pipeline/synthesize.py:82`](../../src/audrey/pipeline/synthesize.py#L82).

After the deep panel runs, Audrey may have several worker drafts. The
synthesizer turns those drafts into one final answer.

The synthesizer is not chosen by `first_healthy(...)`. It comes from the same
deep-panel pool config, starting in `pick_synthesizer(...)`:

```python
def pick_synthesizer(cfg: Config, *, pool_key: str, task: TaskType) -> tuple[str, str]:
    pool = cfg.raw.get(pool_key, {}).get(task, {})
    primary = pool.get("synthesizer")
    fallback = pool.get("fallback_synth")
    ...
    return primary, fallback
```

Then `synthesize(...)` tries the primary and fallback in order at
[`synthesize.py:201`](../../src/audrey/pipeline/synthesize.py#L201):

```python
candidates = [primary] if primary == fallback else [primary, fallback]
for attempt, model in enumerate(candidates, start=1):
    if not health.is_healthy(model):
        continue
    ...
```

Notice the repeated pattern:

- Check health before choosing a model.
- Use `registry.location_of(...)` so the gate knows local vs cloud.
- Call `ollama.chat(...)`.
- On success, `record_success(...)`.
- On `OllamaError`, `record_failure(...)`.

If both synthesizers fail, Audrey degrades to the longest worker draft at
[`synthesize.py:240`](../../src/audrey/pipeline/synthesize.py#L240):

```python
best = max(drafts, key=lambda d: len(d.get("content") or ""))
```

That is not as good as a real synthesis pass, but it is often better than
returning nothing. The user gets a usable answer from one worker instead of a
total failure after the expensive deep-panel work already happened.

The design principle:

```text
When there is no answer material, say so.
When there is answer material but synthesis fails, return the best available material.
```

### 2.9 `OllamaClient`: the HTTP boundary

Open [`src/audrey/models/ollama.py`](../../src/audrey/models/ollama.py#L29).

This is the only file in the model layer that actually speaks HTTP to Ollama.
Everything else chooses model names and handles success/failure policy.

The client owns an `httpx.AsyncClient` at
[`ollama.py:44`](../../src/audrey/models/ollama.py#L44):

```python
self._client = httpx.AsyncClient(
    base_url=self._base_url,
    timeout=httpx.Timeout(default_timeout_s),
    headers={"Accept": "application/json"},
    transport=transport,
)
```

`base_url` means callers can use paths like `/api/chat` instead of building the
full URL every time. `timeout` gives every call a default deadline. Individual
calls can override that default with `timeout_s`.

The optional `transport` argument is mainly for tests. It lets tests provide an
`httpx.MockTransport` so they can simulate Ollama responses without a real
network connection.

All public methods are `async`:

- [`tags()`](../../src/audrey/models/ollama.py#L56)
- [`chat()`](../../src/audrey/models/ollama.py#L71)
- [`chat_stream()`](../../src/audrey/models/ollama.py#L114)
- [`embed()`](../../src/audrey/models/ollama.py#L161)

That means callers must use `await` or `async for`. Audrey is an async web app;
while one request waits on Ollama, the event loop can keep serving other work.

#### Non-streaming chat

`chat(...)` builds an Ollama `/api/chat` payload at
[`ollama.py:87`](../../src/audrey/models/ollama.py#L87):

```python
payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
if options:
    payload["options"] = options
if tools:
    payload["tools"] = tools
```

Then it sends the request at
[`ollama.py:94`](../../src/audrey/models/ollama.py#L94):

```python
r = await self._client.post("/api/chat", json=payload, timeout=...)
```

There are three broad failure types:

| Failure | Example | Audrey behavior |
| --- | --- | --- |
| Transport failure | timeout, connection error | Wrap in `OllamaError`. |
| HTTP error status | Ollama returns 500 or 503 | Raise `OllamaError`. |
| Bad response body | status 200 but invalid JSON or wrong shape | Raise `OllamaError`. |

That last one is easy to miss. A "successful" HTTP status is not enough. Audrey
expects a JSON object from Ollama. `_json_object(...)` enforces that at
[`ollama.py:208`](../../src/audrey/models/ollama.py#L208):

```python
def _json_object(r: httpx.Response, op: str) -> dict[str, Any]:
    try:
        body = r.json()
    except ValueError as e:
        raise OllamaError(...)
    if not isinstance(body, dict):
        raise OllamaError(...)
    return body
```

Why normalize all of those failures into `OllamaError`? Because callers already
know what to do with `OllamaError`: record model failure, cool the model down,
try fallback behavior if the path has one, or surface a controlled error to the
client.

One exception type keeps the failure contract simple.

#### Streaming chat

`chat_stream(...)` is different at
[`ollama.py:114`](../../src/audrey/models/ollama.py#L114):

```python
async def chat_stream(...) -> AsyncIterator[dict[str, Any]]:
```

It does not return one final dictionary. It yields many smaller dictionaries as
Ollama sends token chunks.

That is why callers use `async for`:

```python
async for chunk in ollama.chat_stream(...):
    ...
```

Think of non-streaming `chat()` as "wait until the model is done, then hand me
the whole answer." Think of `chat_stream()` as "hand me each piece as soon as
it arrives."

Streaming changes fallback behavior. Before any text has been sent to the user,
Audrey may still be able to try a fallback. After text has already gone out on
the wire, Audrey cannot pretend that partial answer never happened. In those
cases, the streaming route or streaming synthesizer surfaces a controlled
truncation/error message instead of silently retrying from scratch.

#### Embeddings

`embed(...)` calls `/api/embed` and expects one vector per input text, starting
at [`ollama.py:161`](../../src/audrey/models/ollama.py#L161).

This is not used for normal chat answers. It supports the knowledge-base path:
text needs to become embedding vectors before Audrey can store or search it in
Qdrant. We will revisit embeddings when we get to the KB lessons. For now, just
notice that `embed(...)` follows the same client pattern:

```text
build payload
POST to Ollama
raise OllamaError on transport/status/body problems
return typed Python data
```

### 2.10 One concrete request path

Now put the pieces together.

Imagine a user sends a short general question through `audrey_fast`:

```text
"What is BTRFS?"
```

Here is what happens after the request reaches the graph:

1. The classifier labels the request as `general`.
2. Complexity routing keeps it in fast mode because the prompt is short and the
   user chose `audrey_fast`; the forced-fast check lives at
   [`graph.py:217`](../../src/audrey/pipeline/graph.py#L217).
3. `node_fast_path` calls `run_fast_path(...)` at
   [`graph.py:244`](../../src/audrey/pipeline/graph.py#L244).
4. `run_fast_path(...)` calls `pick_fast_model(...)` at
   [`fast_path.py:70`](../../src/audrey/pipeline/fast_path.py#L70).
5. `pick_fast_model(...)` asks the registry for the first healthy `general`
   candidate:

   ```python
   registry.first_healthy("general", health.is_healthy)
   ```

6. The registry walks the `general` list in priority order.
7. The health tracker says whether each candidate is cooling down.
8. The first healthy candidate becomes `spec`.
9. If `spec.location == "local"`, the request enters `FairLocalGate`; if it is
   `cloud`, the gate immediately passes through.
10. `OllamaClient.chat(...)` sends `POST /api/chat` with `stream=False`.
11. If Ollama returns a good response, the fast path records
    `health.record_success(spec.name)` and returns the answer.
12. If Ollama raises `OllamaError`, the fast path records
    `health.record_failure(spec.name, error)` and the model cools down.

That path is the model layer in miniature.

The key is that "best model" means **best currently healthy candidate**, not
"top priority no matter what."


## 3. Comprehension Q&A

Try answering each yourself before reading the answer.

**1. A `general` request comes in, and the top-priority general model is cooling
down. What happens?**

`ModelRegistry.first_healthy("general", health.is_healthy)` walks the general
candidate list in priority order. The cooling-down model fails the health check,
so the registry keeps walking. The first candidate whose name returns `True`
from `health.is_healthy(...)` is selected.

The registry order does not change. Health temporarily filters the list.

**2. Why not remove a failed model from the registry?**

Because a model failure is often temporary. The model may be busy, slow to load,
or affected by a network hiccup. Removing it from the registry would turn a
temporary operational problem into a permanent configuration change.

Audrey keeps the registry stable and records transient trouble in
`HealthTracker`. After the cooldown expires, the model can be tried again. A
successful call clears the failure record.

**3. Why does `location` matter if local and cloud models both use
`OllamaClient`?**

Because `location` is about scheduling, not HTTP shape.

Audrey calls both local and `:cloud` models through the Ollama-compatible API.
But local models consume the host GPU and must respect `FairLocalGate`. Cloud
models do not consume the local GPU, so the gate lets them pass through.

The same client sends the HTTP request; `location` decides whether local
fairness applies before that request runs.

**4. What happens if someone writes `location: clodu` in `config.yaml`?**

`ModelRegistry` validates location during startup. `clodu` is not one of the
allowed values, so registry construction raises `ValueError`, and Audrey fails
startup instead of serving with a broken scheduling assumption.

That is intentional. A bad model registry is not a graceful-degradation case;
Audrey should not pretend it knows how to route models when config is invalid.

**5. Why does `first_healthy` accept `health.is_healthy` as an argument instead
of importing `HealthTracker` inside `registry.py`?**

Because the registry only needs a yes/no test. It does not need to know how
health is stored, how long cooldowns last, or what counts as a failure.

Passing a predicate keeps the registry focused on ordering candidates. Health
policy stays in `HealthTracker`.

**6. What is the difference between `chat()` and `chat_stream()`?**

`chat()` is non-streaming. It waits for Ollama to finish and returns one full
response dictionary.

`chat_stream()` is streaming. It returns an async iterator, and callers use
`async for` to receive chunks as Ollama emits them.

That difference matters for fallback. Before tokens are sent, Audrey can still
try another model in some paths. After partial text is already streamed to the
client, Audrey has to surface the failure as part of that stream.

**7. Why does `OllamaClient` wrap transport errors, bad HTTP statuses, and bad
JSON bodies in `OllamaError`?**

Because callers need one failure contract. The fast path, deep panel, synth,
and streaming route all know how to respond to `OllamaError`: record a health
failure, try a fallback where possible, or return a controlled error.

If raw `httpx` errors or JSON parsing errors leaked out, some paths could miss
`health.record_failure(...)` and treat a model as healthy even after a bad
response.

**8. Why does a deep-panel worker return an error draft instead of raising?**

Because deep mode can still succeed with partial worker results. If one worker
fails but another produces useful content, the synthesizer may still be able to
answer.

The error is kept as data in a `WorkerDraft`. That lets the panel finish and
lets the synthesizer decide whether there is enough material.

**9. What happens if both synthesizers fail after worker drafts exist?**

Audrey degrades to the longest available worker draft. That is not as strong as
a real synthesis pass, but it gives the user the best material Audrey already
paid to generate.

If there are no usable drafts at all, Audrey returns a clear "no usable drafts"
style message instead.

**10. Why use `time.monotonic()` for cooldowns instead of `time.time()`?**

Cooldowns are durations. Audrey needs to know whether five seconds, ten
seconds, or several minutes have elapsed.

`time.time()` is wall-clock time and can jump if the system clock is adjusted.
`time.monotonic()` only moves forward, so it is the right clock for deadlines
and elapsed-time calculations.


## When you're ready for the next lesson

The next lesson moves one step earlier in the request path:
[Lesson 7 - Classification and routing](lesson-07-classification-and-routing.md).
We have now seen what happens once Audrey knows the task type. The next useful
question is how Audrey decides whether a prompt is `code`, `reasoning`,
`general`, or `vl`, and how that classification combines with `audrey_auto`,
`audrey_fast`, and the deep virtual models.
