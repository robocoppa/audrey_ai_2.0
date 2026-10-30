# Lesson 6 — The model layer I: choosing a model

**Estimated time:** 40-55 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"when a person picks a
model, which concrete model actually answers, and who decides whether they
may pick it?"*

The model layer is the part of Audrey that turns a choice into a model and
calls it. It is big enough for two lessons. This one covers choosing: the
choices a person sees, the configuration behind them, how Audrey ranks
candidates, and how it steps around a model that just failed. The next lesson
covers calling: the client that talks to Ollama, the fast path, the deep
panel, synthesis, images, and Private.


## 1. Context

### 1.1 Two kinds of model name

A person picks a **model choice**: Auto, Fast, Deep, Research, Private, or a
specific model by name. Most of those aren't models at all. They are routing
choices that Audrey's pipeline turns into a **concrete model**, a name Ollama
knows: a model installed on this server, or a cloud model, whose name ends in
`:cloud`. Which concrete model it becomes depends on the task, on which models
are healthy right now, and on the configuration. The models themselves change
often, which is why the lessons describe roles and rules rather than names.
Only a **direct model** skips all that: It names one concrete model, which
answers with no pipeline around it.

Behind every choice, the model layer answers the same few questions:

```text
The pipeline says: "I need a general model."

Which concrete model should that mean right now?
Is the preferred model healthy?
Is it local or cloud?
Should it wait for the local GPU gate?
How do we call Ollama?
What happens if the call fails?
```

This lesson answers the first three. The next lesson answers the rest.

### 1.2 Four pieces, four jobs

| Piece | What it knows | What it doesn't know |
| --- | --- | --- |
| `ModelRegistry` | Which models exist for each task, their priority, and whether each is local or cloud | Whether a model just failed |
| `HealthTracker` | Which models are cooling down after recent failures | Which models suit code, reasoning or general work |
| `FairLocalGate` | Whether a local generation may use the GPU right now | How to choose or call a model |
| `OllamaClient` | How to speak Ollama's HTTP API | Which model should be chosen |

That separation is the point. If choosing, health, GPU fairness and HTTP all
lived in one function, Audrey would be much harder to change safely. This
lesson reads the first two; the next lesson puts the other two to work.


## 2. Read-along

These are the files this lesson reads:

- [`model_catalog.py`](../../src/audrey/model_catalog.py): the choices a
  person sees, and who may use each.
- [`config.yaml`](../../config.yaml): the model pools.
- [`models/registry.py`](../../src/audrey/models/registry.py): ranked
  candidates per task.
- [`models/health.py`](../../src/audrey/models/health.py): temporary
  cooldowns.

### 2.1 The choices a person sees

Each entry in the model picker is a
[`ServedModel`](../../src/audrey/model_catalog.py#L32): the id the browser
sends, a label, and the target that runs it. Audrey's own workflows come
first, from [`_WORKFLOWS`](../../src/audrey/model_catalog.py#L62), and each
names a `protocol_model`, the routing choice the pipeline understands:

```python
_WORKFLOWS: tuple[ServedModel, ...] = (
    ServedModel(
        id="auto",
        label="Auto",
        description="Audrey chooses the right path for the request.",
        kind="workflow",
        mode="auto",
        protocol_model="audrey_auto",
        ...
    ),
    ...
)
```

[`configured_models`](../../src/audrey/model_catalog.py#L157) adds two more
kinds of entry. Private appears as one more workflow when
`private_processing` is enabled in config; the next lesson explains what it
changes. Direct models come from Ollama itself: For the browser app,
[`discover_models`](../../src/audrey/model_catalog.py#L193) asks Ollama which
models are installed and offers each one as a direct entry, with defaults
from [`native_models`](../../config.yaml#L1538) (by default, only admins see
them). If Ollama can't be reached, the configured list stands in, so Audrey's
own workflows stay usable. The application caches the inventory and the
admin's model settings for a short while, so a message doesn't have to ask
Ollama for its model list and reread every setting. An admin's change to a
model clears that cache at once, and the account's own status and groups are
still checked on every request.

Then access.
[`catalog_for_principal`](../../src/audrey/model_catalog.py#L238) applies the
admin's settings for each entry (enabled or not, and which groups may use it)
and keeps only what this account may use, through
[`_model_allows`](../../src/audrey/model_catalog.py#L446):

```python
def _model_allows(model: ServedModel, principal: Principal) -> bool:
    if principal.status != "active":
        return False
    if principal.is_admin:
        return True
    if model.visibility == "restricted":
        return False
    return bool(set(model.roles) & principal.groups)
```

An inactive account gets nothing and an admin gets everything. An entry
marked `restricted` is admin-only and hidden from everyone else; it has nothing
to do with the Private choice. Any other entry is offered to accounts in one
of its groups.

> **Concept spotlight — hiding is not enforcement.** The picker shows only
> what an account may use, but the browser can send any id. So every run
> checks again:
> [`resolve_model`](../../src/audrey/model_catalog.py#L329) filters the
> catalog for this account again and looks the requested id up in it, and a
> model the account may not use gets 404, exactly as if it didn't exist
> (Lesson 4 §2.5). Showing and allowing are separate steps, and only the
> second one is security.

### 2.2 Startup builds the shared model objects

Start in `lifespan`, at
[`main.py:126`](../../src/audrey/main.py#L126). Lesson 5 read this function
as startup code; this time, look only at the model layer:

```python
default_timeout = float(cfg.timeouts.get("medium", 180))
ollama = OllamaClient(cfg.env.ollama_host, default_timeout_s=default_timeout)
registry = ModelRegistry(cfg)
health = HealthTracker()
gpu_concurrency = int(cfg.raw.get("gpu", {}).get("concurrency", 1))
gate = FairLocalGate(concurrency=gpu_concurrency)
```

`lifespan` then hands those objects to the graph at
[`main.py:170`](../../src/audrey/main.py#L170) and stores them on
`app.state` for the routes. Every request uses the same client, the same
registry, the same health tracker and the same gate.

That matters most for health. If a model times out on one request, the next
request should know to avoid it for a little while, and that works only
because the tracker is shared instead of rebuilt per request:

```text
startup creates the model-layer objects once
  -> request handlers and graph nodes reuse them many times
```

### 2.3 `config.yaml` is the source of model pools

Open [`model_registry`](../../config.yaml#L80). It is grouped by **task
type**: `code`, `reasoning`, `general` and `vl` (vision). Those come from the
classification step a later lesson walks; nobody picks them directly.
Audrey classifies the prompt, then asks the model layer for a model that fits
the task.

Each entry in a task's list has the same three fields. The shape (with
placeholder names, since the models themselves change):

```yaml
general:
  - { name: "<a local model>",  priority: 100, location: local }
  - { name: "<a model>:cloud",  priority: 90,  location: cloud }
```

- `name` is the concrete Ollama model name.
- `priority` ranks the entry: The higher, the earlier it is tried.
- `location` says whether the model runs on this server's GPU (`local`) or
  in Ollama's cloud (`cloud`), which decides whether the GPU gate applies.

The rule of thumb:

```text
Task type chooses a list.
Priority orders the list.
Health temporarily skips entries.
Location decides whether local GPU fairness applies.
```

Further down, `config.yaml` defines the deep-panel pools:
[`deep_panel`](../../config.yaml#L288),
[`deep_panel_cloud`](../../config.yaml#L354),
[`deep_panel_local`](../../config.yaml#L387) and
[`deep_panel_research`](../../config.yaml#L521). A pool is a recipe rather
than a ranked menu: "for this choice and task type, run these workers, then
this synthesizer." So there are three ways a model gets chosen:

| Path | How the model is chosen |
| --- | --- |
| Fast path | The highest-priority healthy model in the task's `model_registry` list |
| Deep panel | The pool's configured workers, minus unhealthy ones and those over the cloud cap |
| Synthesizer | The pool's configured synthesizer, then its fallback |

The next lesson follows all three. The rest of this one is the machinery they
share.

### 2.4 `ModelSpec`: one entry as Python data

Open [`registry.py:15`](../../src/audrey/models/registry.py#L15):

```python
TaskType = Literal["code", "reasoning", "general", "vl"]
Location = Literal["local", "cloud"]
```

`Literal` means "not just any string: one of these exact values." Type
checkers and editors use it to catch mistakes, but type hints don't check
YAML at runtime; YAML is just data read from a file. So the registry also
validates the location itself, in
[`_parse_location`](../../src/audrey/models/registry.py#L113), and a typo such
as `clodu` stops Audrey at startup instead of quietly treating a local model
as cloud and letting it skip the GPU gate.

One entry becomes a [`ModelSpec`](../../src/audrey/models/registry.py#L20):

```python
@dataclass(slots=True, frozen=True)
class ModelSpec:
    name: str
    priority: int
    location: Location
```

`@dataclass` writes the constructor for you. `frozen=True` means a spec
can't change after it's built: If the config changes, Audrey restarts and
builds a new registry, rather than request code editing a priority in place.
`slots=True` fixes the set of attributes, so a typo like `spec.priorty`
fails instead of quietly creating a new one.

### 2.5 `ModelRegistry`: ranked candidates

[`ModelRegistry`](../../src/audrey/models/registry.py#L27) turns each task's
YAML list into `ModelSpec`s and sorts it once, at startup, at
[`registry.py:48`](../../src/audrey/models/registry.py#L48):

```python
specs.sort(key=lambda s: s.priority, reverse=True)
```

`lambda s: s.priority` is a tiny function that tells `sort` what to compare;
`reverse=True` puts the highest priority first. Request time then needs two
methods.

[`candidates(task)`](../../src/audrey/models/registry.py#L51) returns a copy
of the task's ranked list, so a caller can't change the registry by changing
the list it got back.
[`first_healthy`](../../src/audrey/models/registry.py#L54) walks that list and
returns the first model that passes a health test (the image describer of the
next lesson uses it to pick a vision model):

```python
def first_healthy(self, task: TaskType, is_healthy) -> ModelSpec | None:
    """Return the highest-priority candidate for which `is_healthy(name)` is True."""
    for spec in self._by_task.get(task, ()):
        if is_healthy(spec.name):
            return spec
    return None
```

The fast path uses the same priority-and-health rule, with one recovery
exception: If every candidate is cooling down, it tries the one whose
cooldown ends first. Notice `health.is_healthy`, not `health.is_healthy(...)`:
The caller passes the function itself, for the registry to call once per
candidate. The registry doesn't need to know what "healthy" means, only how to
ask.

One more method decides scheduling.
[`location_of`](../../src/audrey/models/registry.py#L64) returns a model's
declared location from whichever list it appears in. A model with no entry
anywhere (a pool worker missing from `model_registry`, say) falls through to
[`_location_from_tag`](../../src/audrey/models/registry.py#L82), which treats
it as local unless Ollama's own tag says `cloud`. Local is the safe default:
an unknown model goes through the GPU gate rather than skipping it.

### 2.6 `HealthTracker`: cooldown, not deletion

The [`HealthTracker`](../../src/audrey/models/health.py#L24) answers one
question: *should Audrey avoid this model right now?* It doesn't know
whether a model is installed or any good, only whether it failed recently in
this process.

A model has no record until it fails, and no record means healthy, at
[`health.py:43`](../../src/audrey/models/health.py#L43):

```python
def is_healthy(self, model: str) -> bool:
    state = self._by_model.get(model)
    if state is None:
        return True
    return time.monotonic() >= state.cooldown_until
```

`time.monotonic()` is a clock that only moves forward, so it suits "has
enough time passed?" even if the system clock is adjusted. A failure outside
an existing cooldown window lengthens the next cooldown, at
[`health.py:65`](../../src/audrey/models/health.py#L65):

```python
backoff = min(self._base * (2 ** (state.consecutive_failures - 1)), self._max)
```

This is **exponential backoff**: The first failure benches a model for a few
seconds, and each failure after a cooldown has run out doubles the next one,
up to a cap of minutes. Failures from calls that were already in flight when
the cooldown started don't extend it, so one outage counts once, not once per
waiting request. A single success clears the record, in
[`record_success`](../../src/audrey/models/health.py#L54).

Why cool a model down instead of removing it? Most failures pass: Ollama is
busy, a cloud model times out, the network hiccups, a large local model is
slow to load. Removing the model would turn a passing problem into a
configuration change. Stepping around it for a while, and trying it again
later, doesn't.

```text
failure -> cool down for a while
cooldown expires -> the model can be tried again
success -> forget past failures
```

### 2.7 From choice to model, end to end

Someone picks Fast and asks *"What is BTRFS?"*

1. The browser sends the id `fast`. `resolve_model` finds it in the account's
   catalog, so the run may proceed with the routing choice `audrey_fast`.
2. Classification labels the question `general`.
3. `audrey_fast` means one model answers, so the fast path reads the
   registry's ranked `general` candidates.
4. It asks the health tracker about each name and chooses the first healthy
   one; if every one is cooling down, it tries the one that recovers first.
5. That model's `location` decides whether the call waits for the GPU gate.

The preferred model is the best **currently healthy** candidate, with one
recovery probe when none are healthy. What happens next, the call itself and
what a failure does, is the next lesson.


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. A `general` request arrives, and the top-priority general model is
cooling down. What happens?**

The fast path walks the `general` list in priority order. The cooling-down
model fails the health test, so the walk continues, and the first model that
passes answers instead. The list's order never changes; health only filters
it for a while. Once the cooldown expires, the top model is first in line
again. If every candidate is cooling down, the fast path tries the one whose
cooldown ends first, so a recovered Ollama doesn't have to wait out every
timer.

**2. Local and cloud models are called through the same client. Why does
`location` matter at all?**

Because it's about scheduling, not about how the call is made. A local model
uses this server's GPU, so its call waits its turn at the GPU gate. A cloud
model runs elsewhere and touches no local GPU, so the gate lets it straight
through. The same client sends both requests; `location` decides whether
local fairness applies first.

**3. An admin restricts a model while someone still has it selected in an
open tab. They send their next message. What happens?**

The run is refused. The browser still sends the old id, but run creation
calls `resolve_model`, which rebuilds the account's catalog with the
admin's current settings. The model is no longer in it, so the request gets
404, the same answer as for an id that never existed. What the picker showed
earlier doesn't matter; only the check at run time does.

**4. A deep-panel pool names a worker that has no entry in `model_registry`.
Does its call go through the GPU gate?**

That depends on its name. `location_of` finds no declared location, so it
falls through to `_location_from_tag`. If the model's Ollama tag says
`cloud` (`some-model:cloud`, say), the model counts as cloud and skips the
gate. Otherwise it counts as local and waits for the gate. Local is the
deliberate default: An unknown model costs a little waiting rather than
risking two models on a GPU that fits one.

**5. A new model is installed in Ollama. Does it appear in the browser's model
picker?**

As a direct model, yes, for admins. For the browser app, `discover_models`
asks Ollama which models are installed and offers each one as a direct
entry, and `native_models` sets the defaults (by default only admins see
them). An admin can then open it to other groups. Making it a candidate for
Audrey's own workflows is a different change: It needs an entry in
`model_registry` or a pool, with a priority and a location.


## When you're ready for the next lesson

[Lesson 7 — The model layer II: calling models](lesson-07-calling-models.md)
picks up where this one stops. A model has been chosen; the next lesson
follows the call itself: the client that speaks to Ollama, thinking
controls, what the fast path, the deep panel and the synthesizer do when a
call fails, how a model that can't see images still answers about one, and
how the Private choice closes every route to the cloud.
