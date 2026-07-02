# Lesson 8 - Deep mode: planner, panel, synthesizer, reflect

**Estimated time:** 60-80 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer
*"when Audrey routes a request down the deep path instead of the fast one,
what actually happens to that request — and why does the answer take four
stages instead of one?"*

[Lesson 7](lesson-07-classification-and-routing.md) left off with two
values pinned to the request: a `task_type` (`code`, `reasoning`,
`general`, or `vl`) from the classifier, and a `mode` (`fast` or `deep`)
from the complexity gate. This lesson picks up the instant `mode = deep`
and opens the door that mode goes through — both of those Lesson-7 values
ride along into deep mode and shape what happens next. We will follow one
request through four nodes:

```text
planner -> deep_panel -> synthesize -> reflect
```

Each node has a different job, a different failure mode, and a different
reason to exist. By the end you should be able to glance at a deep-mode log
line and name the node that wrote it.


## 1. Context

Fast mode is one chat call — one model, one answer. Deep mode is the
opposite bet: spend more wall-clock to ask several workers in parallel, then
merge what they wrote. The contrast is straightforward:

- **Fast mode:** one chat call, one perspective, low latency, cheap.
- **Deep mode:** several chat calls in parallel, merged into one voice, higher latency, more expensive.

The user never sees the four stages — only the one answer streaming back.
Inside Audrey, four distinct things happen between "request arrived" and
"answer ready."

### 1.1 Why four stages and not one

The obvious way to "answer harder" is to throw a bigger model at the prompt.
Audrey takes another path: a small **panel** of workers answers in
parallel, and a synthesizer stitches their drafts into one voice. Two
reasons it's shaped this way:

1. **Different models miss different things.** A code-tuned worker and a
   reasoning-tuned worker, writing the same prompt at the same time, cover
   each other's blind spots more reliably than either one writing alone.
2. **The merge is cheap relative to the work.** The panel owns the
   wall-clock budget; the synthesis pass costs one extra chat call and turns
   a stitched quilt into a single answer.

The four stages map onto four questions Audrey has to answer:

| Stage | Question it answers |
| --- | --- |
| Planner | "Is the request one question or several?" |
| Deep panel | "What do the workers each think the answer is?" |
| Synthesizer | "What is the single best answer to ship the user?" |
| Reflect | "Is the answer we generated actually good enough?" |

If you remember the questions, the code falls into place.

### 1.2 The four-stage timeline

Here is the path in one pass, with the dispatcher choices that affect each
stage:

1. **`planner`** *(optional)*
   - Skip if the prompt is short (`planning.min_prompt_tokens`).
   - Skip if the planner returned 0 or 1 subtask.
   - Otherwise: 2-3 sub-questions assigned round-robin to workers.
2. **`deep_panel`**
   - Pick the pool from the virtual model (`audrey_deep` / `audrey_cloud` / `audrey_local`).
   - Filter the pool's workers by health.
   - If no healthy workers, fall back to the model registry (cap 2).
   - Run all workers in parallel:
     - Local workers serialize through the GPU gate.
     - Cloud workers run concurrently (capped at `max_deep_workers_cloud`).
     - Tool-capable workers run a ReAct loop instead of one chat call.
3. **`synthesize`**
   - Look up the pool's `synthesizer` + `fallback_synth`.
   - Run primary; on failure, run fallback.
   - If both fail, ship the longest non-empty draft verbatim.
4. **`reflect`**
   - Check the synth output meets `min_answer_chars`.
   - Skip the floor if the user explicitly asked for brevity.
   - If it failed and we haven't already retried, loop back to `deep_panel`.

Two routing rules outside the four nodes matter for understanding the trace:

- **Virtual model picks the pool.** `audrey_deep` uses the mixed
  local+cloud pool, `audrey_cloud` uses the cloud-only pool, `audrey_local`
  uses the local-only pool. `audrey_auto` lands here when the gate
  chooses deep — a long prompt or an explicit depth cue — and its pool is
  the same as `audrey_deep`.
- **Reflection is best-effort, bounded.** At most one retry of
  panel+synth. If the second attempt still fails the quality check,
  Audrey ships what it has rather than 502 the request.


## 2. Read-along

These are the files we'll reference. Open them as we go.

- [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py) — the LangGraph nodes that enter each stage (`node_planner`, `node_deep_panel`, etc.).
- [`src/audrey/pipeline/planner.py`](../../src/audrey/pipeline/planner.py) — the planner LLM call and its forgiving JSON parser.
- [`src/audrey/pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py) — pool selection, worker filtering, the registry fallback, per-worker dispatch, and the subtask round-robin.
- [`src/audrey/pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py) — draft bundling, primary/fallback synth selection, and the three-tier failure handling.
- [`src/audrey/pipeline/reflect.py`](../../src/audrey/pipeline/reflect.py) — the deterministic quality gate and brevity-cue escape hatch.
- [`src/audrey/pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py) — the `PLANNER_SYSTEM` and `SYNTH_SYSTEM` instructions.
- [`config.yaml`](../../config.yaml) — the three `deep_panel*` pools (mixed, cloud-only, local-only) keyed by task type.


### 2.1 Stage 1 — Planner

The planner asks one question: *"Is this user request really one ask, or
several asks bundled together?"*

If it's one ask, the panel runs the prompt verbatim. If it's several, the
panel splits them — one worker per sub-question — so each worker focuses on
a tighter slice instead of trying to cover everything at once.

#### When the planner runs

The graph node is at
[`graph.py:293`](../../src/audrey/pipeline/graph.py#L293):

```python
async def node_planner(state: PipelineState) -> dict[str, Any]:
    if not planning_enabled or state.get("prompt_tokens", 0) < planning_min_tokens:
        return {"subtasks": []}
```

Two early exits. If `agentic.planning.enabled` is off, return no subtasks.
If the prompt is shorter than `planning.min_prompt_tokens` (default 40),
also return no subtasks — on short prompts the planner round-trip costs
more than it saves.

#### What the planner does

When it does run, it makes one LLM call. The system prompt is fixed at
[`prompts.py:62`](../../src/audrey/pipeline/prompts.py#L62):

```text
You decompose a user request into 2 or 3 focused sub-questions that, if
answered separately, would together cover the original request. Output a
JSON object with exactly this shape:
  {"subtasks": ["...", "...", "..."]}
Rules:
- 2 to 3 entries, each a complete question or instruction (≤ 200 chars).
- Sub-questions must be independent — no 'first do X then Y' chaining.
- If the request is already atomic (one clear ask), return {"subtasks": []}.
Output ONLY the JSON. No prose, no markdown.
```

Notice the prompt explicitly invites `{"subtasks": []}` as a valid output.
The planner is allowed — encouraged, even — to say "this isn't decomposable."

The call itself is at
[`planner.py:60`](../../src/audrey/pipeline/planner.py#L60):

```python
resp = await ollama.chat(
    model=planner_model,
    messages=messages,
    options={"temperature": 0.0},
    timeout_s=timeout_s,
)
```

`temperature=0.0` because we want deterministic decomposition. The planner
isn't being creative; it's reading structure.

#### The parser is intentionally forgiving

LLM JSON output in practice is messy — leading prose, trailing markdown,
mismatched braces. The parser shrugs all of that off at
[`planner.py:79`](../../src/audrey/pipeline/planner.py#L79):

```python
def _parse_planner_output(raw: str) -> list[str]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = obj.get("subtasks", [])
    ...
```

`find('{')` and `rfind('}')` grab the outermost brace pair, dropping
preamble and trailing commentary in one move. The slice can still be
malformed — a stray `}` in the prose, two JSON objects in one response — in
which case `json.loads` raises and the parser quietly returns `[]`. The
permissiveness is by design: every planner failure mode degrades to "no
decomposition," and the panel handles that case fine.

#### Three exits, all benign

The planner has three exits, and all of them produce `[]`:

1. The model returned zero subtasks — recognized the prompt as atomic, by
   design.
2. The model returned one subtask — probably failed to decompose; logged at
   debug, treated as no decomposition.
3. The model returned malformed JSON — parsing failed; degrades to `[]`.

Every path lands in the same place. `subtasks = []` flows into the deep
panel, the panel runs every worker against the original prompt, and life
goes on. The planner is opt-in routing, not a hard requirement — when it
works, it sharpens the panel; when it fails, the panel doesn't notice.

When the planner *does* return subtasks, the log line at
[`graph.py:306`](../../src/audrey/pipeline/graph.py#L306) shows the count
and the first 60 chars of each:

```python
log.info("planner: %d subtasks: %s", len(subs), [s[:60] for s in subs])
```


### 2.2 Concept spotlight — `asyncio.gather` and the parallel-worker idea

Before we open the deep panel, a quick aside on what "parallel" actually
means here.

Python is single-threaded for CPU work, but `asyncio` lets one thread juggle
many in-flight I/O operations at once. `await ollama.chat(...)` is mostly
spent waiting on the network — the model generates tokens on the GPU (or in
the cloud), the bytes trickle back over HTTP, and Python's event loop is
free to do other things during the wait.

`asyncio.gather(*coros)` takes a list of those coroutines and runs them
**concurrently**. From the caller's perspective:

```python
drafts = await asyncio.gather(coro_worker_a, coro_worker_b, coro_worker_c)
```

…blocks until *all* of them finish, then returns a list of results in
submission order. While it blocks, the event loop interleaves the three
workers' I/O: if worker A is waiting on the network, the loop dispatches
B's next request, then C's, then comes back to A the moment bytes arrive.

For deep-panel workers, this is the entire reason multiple cloud workers
finish in roughly the time of one. Three workers each waiting 12 seconds on
the cloud take roughly 12 seconds total, not 36, because their waits
overlap rather than queue.

Local workers don't enjoy the same speedup — they all want the same GPU.
That constraint is what the next concept is built around.


### 2.3 Stage 2 — Deep panel

The deep panel is where the actual work happens. Three knobs decide which
workers run:

1. Which **pool** to draw from — set by the virtual model.
2. Which workers in that pool are **healthy** — set by `HealthTracker`.
3. Whether each worker is **tool-capable** — set by
   `fast_path.tool_capable_models`.

#### Picking the pool

`pool_key_for` at
[`deep_panel.py:81`](../../src/audrey/pipeline/deep_panel.py#L81) does the
mapping:

```python
def pool_key_for(virtual_model: str) -> str:
    pool = _POOL_KEYS.get(virtual_model)
    if pool is None:
        log.warning(
            "deep_panel: unknown virtual_model %r, falling back to default pool 'deep_panel'",
            virtual_model,
        )
        return "deep_panel"
    return pool
```

The fallback is defensive. The public route validates virtual model names
before this helper runs, so a normal client typo should be rejected earlier.
Keeping the fallback here protects internal and future call sites from turning
an unexpected model string into a mid-request `KeyError`; the operator still
gets a warning in the logs.

The pools themselves live in `config.yaml`. Open
[`config.yaml:98`](../../config.yaml#L98):

```yaml
deep_panel:
  code:
    workers: ["qwen3-coder-next:latest", "kimi-k2.7-code:cloud", "deepseek-v4-pro:cloud"]
    synthesizer: "qwen3.6:35b"
    fallback_synth: "glm-5.2:cloud"
  reasoning:
    workers: ["qwen3.6:35b", "kimi-k2.6:cloud", "deepseek-v4-pro:cloud"]
    synthesizer: "qwen3.6:35b"
    fallback_synth: "glm-5.2:cloud"
  ...
```

Each pool is keyed by **task type** (code, reasoning, general, vl), so
"which workers run" is a two-axis lookup: the virtual model picks the pool,
and the classifier output from Lesson 7 picks the task entry within it.

#### Selecting healthy workers

`select_workers` at
[`deep_panel.py:130`](../../src/audrey/pipeline/deep_panel.py#L130) walks the
configured worker list and filters:

```python
for name in raw_workers:
    if not health.is_healthy(name):
        log.info("deep_panel: skipping unhealthy worker %s", name)
        continue
    loc = registry.location_of(name)
    if loc == "cloud":
        if cloud_count >= max_workers_cloud:
            continue
        cloud_count += 1
    out.append((name, loc))
```

Three filters:
- Unhealthy workers (recent failures) are skipped.
- Cloud workers are capped at `max_workers_cloud` (default 3 — Ollama Pro's
  concurrent limit).
- Local workers aren't capped here; the GPU gate handles that.

`registry.location_of(name)` answers a single question — "is this a local
or a cloud model?" — by walking the registry rather than trusting whatever
the pool config declared. A model that gets renamed still ends up on the
right code path.

#### The registry fallback

If `select_workers` returns nothing — every pool worker is unhealthy —
the panel falls back to the model registry itself. This selection logic lives
in a shared helper, `_prepare_panel`, that both the non-streaming `run_panel`
and the streaming `run_panel_streaming` call, at
[`deep_panel.py:310`](../../src/audrey/pipeline/deep_panel.py#L310):

```python
if not workers:
    log.warning("deep_panel: no healthy pool workers for %s/%s; falling back to registry", pool_key, task)
    for spec in registry.candidates(task):
        if not health.is_healthy(spec.name):
            continue
        workers.append((spec.name, spec.location))
        if len(workers) >= 2:
            break
```

The cap of 2 keeps the emergency path bounded — enough drafts to merge,
without flooding the GPU gate or burning cloud quota when the configured pool
has already failed. The log line is a `warning` precisely so operators notice
when a pool has lost its workers; if you see this in production, those models
need attention now.

#### Concept spotlight — the GPU gate and per-worker accounting

Local workers all squeeze through a `FairLocalGate` semaphore (Lesson 6
introduced it). With `GPU_CONCURRENCY=1` — the production default — only
one local worker can run at a time, because they're all competing for the
same GPU's VRAM.

Per-worker dispatch *looks* parallel on the dispatcher side
([`deep_panel.py:309-328`](../../src/audrey/pipeline/deep_panel.py#L309)),
but execution serializes through the gate. A deep panel with two local
workers runs them back to back, not side by side. Two cloud workers in the
same panel run concurrently because they never touch the gate.

This is also why tool-capable local workers hold the gate for the *entire*
ReAct loop, not just one chat call. Look at
[`deep_panel.py:151-172`](../../src/audrey/pipeline/deep_panel.py#L173):

```python
async with gate.acquire(model, location=location, user_id=user_id):
    if use_tools:
        react: ReactResult = await run_react(
            ollama, health, tools,
            ...
            gate=None,  # we already hold it
            ...
        )
```

`gate=None` is the load-bearing detail. The ReAct loop is told *not* to
acquire its own gate, because the panel already holds one. If ReAct
acquired again mid-loop, a local worker would briefly release the GPU
between rounds — handing another local worker (or another user's request)
a window to slip in. Holding for the whole loop keeps tool-using workers
atomic from the gate's perspective.

One subtlety worth knowing before you read deep-mode logs: when a worker
runs ReAct, `WorkerDraft.prompt_eval_count` and `eval_count` carry only the
**final** chat call's tokens, not the sum across rounds. A tool-grounded worker
that ran three rounds reports just the last round's count. It is not a metrics
bug — current Audrey dashboards treat these as per-final-call counters — but
if you compare a tool-grounded worker's token count against a one-shot worker
count, the tool-grounded one looks artificially light. That is the accounting
working as designed, not a worker cheating its way to a smaller number.

#### Sub-question round-robin

If the planner produced subtasks, each worker gets one. `_prepare_panel`
distributes them round-robin at
[`deep_panel.py:299-304`](../../src/audrey/pipeline/deep_panel.py#L321):

```python
if subtasks:
    per_worker_messages = [
        _messages_for_subtask(messages, subtasks[i % len(subtasks)])
        for i in range(len(workers))
    ]
else:
    per_worker_messages = [messages] * len(workers)
```

Two scenarios make the math concrete. Three subtasks and two workers:
worker 0 gets subtask 0, worker 1 gets subtask 1, subtask 2 is dropped
(the modulo would wrap, but with `len(workers) < len(subtasks)` the loop
never reaches it). Two subtasks and three workers: worker 0 gets subtask 0,
worker 1 gets subtask 1, worker 2 circles back to subtask 0 — two workers
answer the same question with different perspectives, and the synthesizer
reconciles them.

`_messages_for_subtask` at
[`deep_panel.py:254`](../../src/audrey/pipeline/deep_panel.py#L254) builds
the per-worker message list by replacing the **last** user message with
the subtask:

```python
for m in reversed(base_messages):
    if not replaced and m.get("role") == "user":
        out.append({"role": "user", "content": subtask})
        replaced = True
    else:
        out.append(m)
out.reverse()
```

The "last user message" detail matters in multi-turn conversations. It's
*this turn's* question — the one the planner just decomposed — not some
arbitrary earlier turn. Earlier user/assistant pairs stay put as context,
so the worker still sees the full thread; only the focal question gets
swapped for the subtask.

#### Worker output: `WorkerDraft`

Every worker returns a `WorkerDraft` regardless of outcome — success,
empty answer, or outright error. The shape lives in
[`pipeline/state.py`](../../src/audrey/pipeline/state.py), but for our
purposes the fields that matter are:

- `content` — the model's text output, possibly empty on failure.
- `error` — populated only on `OllamaError`; truncated to 300 chars.
- `tool_rounds` — `> 0` if the worker ran ReAct.
- `model`, `elapsed_s`, `tool_calls` — observability fields.

The never-raise rule is a load-bearing contract. The synthesizer needs the
full draft list in front of it, even if some entries are empty error rows;
otherwise one misbehaving worker could 502 the entire request.


### 2.4 Stage 3 — Synthesizer

The synthesizer takes the worker drafts and merges them into a single
final answer. One LLM call carries it most of the time; two on retry.

#### Picking the synthesizer

The pool config has two slots:

```yaml
synthesizer: "qwen3.6:35b"
fallback_synth: "glm-5.2:cloud"
```

`pick_synthesizer` at
[`synthesize.py:82`](../../src/audrey/pipeline/synthesize.py#L82) reads
them:

```python
def pick_synthesizer(cfg: Config, *, pool_key: str, task: TaskType) -> tuple[str, str]:
    pool = cfg.raw.get(pool_key, {}).get(task, {})
    primary = pool.get("synthesizer")
    fallback = pool.get("fallback_synth")
    if not primary:
        raise KeyError(f"No synthesizer configured for {pool_key}/{task}")
    if not fallback:
        fallback = primary
    return primary, fallback
```

The `KeyError` should never fire in production — startup validation in
[`config.py`](../../src/audrey/config.py) walks every `deep_panel*` pool/task
at boot and crashes the process if any synthesizer is missing. The runtime
guard is there anyway, in case someone wires Audrey up in a way that
bypasses the validator. Defense in depth, costing nothing.

#### Bundling the drafts

`_format_drafts_for_synth` at
[`synthesize.py:42`](../../src/audrey/pipeline/synthesize.py#L42) lays out
the user message the synthesizer reads. The shape is:

```text
USER REQUEST:
<original prompt>

PLANNED SUB-QUESTIONS:           (only if planner produced any)
  1. ...
  2. ...

DRAFTS:

--- draft 1 (model=qwen3-coder-next, elapsed=8.3s) ---
<worker A's content>

--- draft 2 (model=kimi-k2.6, elapsed=11.2s) [tool-grounded: 2 rounds] ---
<worker B's content>
```

The `[tool-grounded: N rounds]` tag isn't just labeling — it's a signal.
The synthesizer prompt (read it at
[`prompts.py:81`](../../src/audrey/pipeline/prompts.py#L81)) tells the
model to treat a tool-grounded draft as the factual spine of the answer:

> FACTUAL ANCHORING: when one or more drafts are `[tool-grounded]`, treat
> them as the factual spine of the answer. [...] A specific, checkable
> claim [...] that appears ONLY in tool-free drafts and is absent from
> every tool-grounded draft is unverified — soften it [...] or drop it,
> even if several tool-free drafts assert it confidently.

This does two things at once. On a direct conflict ("the model thinks the
answer is X" vs. "the search tool says Y"), Y wins. But it also closes the
subtler hole: a confident-sounding claim that only the *non*-grounded
drafts make — and the grounded draft simply never mentions — gets softened
or dropped rather than promoted just because two workers happened to agree
on it. Agreement between models isn't corroboration; they share the same
training blind spots. The rule deliberately goes quiet when *no* draft is
grounded, so a strong all-from-memory panel on a well-known topic isn't
needlessly hedged.

#### Forwarding original system context

The synthesizer runs against the same system messages the workers saw. Open
[`synthesize.py:94`](../../src/audrey/pipeline/synthesize.py#L94):

```python
def _build_synth_messages(
    prior_messages: list[dict[str, Any]],
    drafts_block: str,
    *,
    draft_count: int,
    cfg: Config | None = None,
) -> list[dict[str, Any]]:
    synth_system = prompt_from_config(cfg, "synthesizer", _SYNTH_SYSTEM)
    system_msgs = [m for m in prior_messages if m.get("role") == "system"]
    return [
        *system_msgs,
        {"role": "system", "content": synth_system},
        {"role": "user", "content": (
            f"Original user request and {draft_count} drafts follow."
            f" Produce the final answer now.\n\n{drafts_block}"
        )},
    ]
```

The original system messages carry the datetime context (set by
`node_datetime`, the first graph node — its module is walked in the
[memory + context injection lesson](lesson-13-memory-and-context-injection.md)),
memory recall hits, and any OWUI-supplied templates. Strip them and
the synthesizer starts hedging about "today" against its training
cutoff — the merged answer suddenly feels months out of date.

`draft_count` is threaded through explicitly so that a future change to the
draft separator string in `_format_drafts_for_synth` can't silently break
the count the synthesizer was told to expect.

#### Three-tier failure handling

The synthesizer can fail three different ways, and each gets its own
handling — read [`synthesize.py:201`](../../src/audrey/pipeline/synthesize.py#L201):

```python
candidates = [primary] if primary == fallback else [primary, fallback]
for attempt, model in enumerate(candidates, start=1):
    if not health.is_healthy(model):
        log.warning("synth: %s unhealthy, skipping (attempt %d)", model, attempt)
        continue
    ...
    try:
        content, ptok, etok = await _try_synth(...)
        ...
        if content.strip():
            return {"content": content, ...}
        log.warning("synth: %s returned empty content (attempt %d)", model, attempt)
    except OllamaError as e:
        health.record_failure(model, str(e))
        ...
```

The tiers:

1. **Empty drafts list**: short-circuits before any LLM call —
   [`synthesize.py:188-193`](../../src/audrey/pipeline/synthesize.py#L188)
   returns `synth_error="no_drafts"` with a placeholder message. Reflect
   will see this and pass it through (it's a deterministic failure, not a
   retryable one).
2. **Primary synth fails**: try the fallback. If the fallback equals the
   primary (no fallback configured), the list collapses to one entry and
   this tier is skipped.
3. **Both synths fail**: degrade to the longest non-empty worker draft
   verbatim, tagged `synthesizer_model="fallback:longest_draft"`. The
   user gets an answer (probably less polished than a real synth would
   produce), and the failure is visible in the metric labels.

The rule across all three tiers: never 502 the request because the synth
failed. The worker drafts are the evidence base — at least one of them
produced something a user can read, and Audrey ships that something rather
than handing back an error.


### 2.5 Stage 4 — Reflect

Reflect is the cheapest stage in the pipeline by a wide margin. No LLM
calls; it only inspects what the synthesizer produced. Open
[`reflect.py:69`](../../src/audrey/pipeline/reflect.py#L69):

```python
def reflect(
    *,
    content: str,
    synth_error: str,
    min_chars: int,
    user_text: str = "",
) -> ReflectionResult:
    if synth_error == "no_drafts":
        return ReflectionResult(False, "no_drafts")

    text = (content or "").strip()
    if len(text) < min_chars:
        if _user_wants_brevity(user_text):
            return ReflectionResult(True, "ok_brevity_requested")
        return ReflectionResult(False, "too_short")

    return ReflectionResult(True, "ok")
```

Three outcomes:

- **`ok`** — content meets the length floor (default `min_answer_chars=80`).
  The graph routes to END.
- **`ok_brevity_requested`** — content is short, but the user asked for
  brevity ("in one sentence", "tldr", etc.) Routes to END.
- **`too_short`** or **`no_drafts`** — fail. The graph router decides
  whether to retry.

#### Why the brevity escape hatch exists

The synth prompt is permissive about length on purpose — it's told to
write the best answer for the user, not to hit a word count. That's fine
for ordinary requests, where the answer turns out substantive on its own.
It falls apart on "What year is it? Answer in one sentence" — the correct
answer is roughly fifteen characters, fails the floor, and gets retried
wastefully as "too short."

The `_BREVITY_CUES` tuple at
[`reflect.py:35`](../../src/audrey/pipeline/reflect.py#L35) is the
hardcoded list of phrases that trip the escape hatch. It's English-only
because Audrey is English-only — a non-English user asking for brevity in
their own language would trigger the same wasteful retry the cues exist to
prevent.

#### Retry routing

The retry decision doesn't live in `reflect()` — `reflect()` only
inspects. The retry lives one level up, in `route_after_reflect` (the
graph router), and it gates on three things:

1. `reflect_passed` must be False.
2. `reflect_attempts` must be ≤ `reflection_max_retries` (default 1).
3. `synth_error` must not be `"no_drafts"` (no point retrying with no
   workers).

When the retry fires, the panel and synth run again with a single nudge:
`compose_system_messages` injects a brief "be more substantive" hint into
the second pass. If that pass still trips the length check, Audrey ships
what it has and marks `reflect_passed=False` in the state — a quiet
breadcrumb for whoever's reading the logs later.


### 2.6 The deep loop in one trace

Here's what an info-level log of a typical deep request looks like, with
each line annotated by the stage that produced it:

```text
classify: task=reasoning conf=0.72                       # Lesson 7
complexity: 640 tokens -> deep (tokens>=500)             # Lesson 7
planner: 2 subtasks: ['How does ...', 'What about ...']  # node_planner
deep_panel: pool=deep_panel task=reasoning workers=2     # node_deep_panel
synth: qwen3.6:35b ok in 6.42s (attempt 1)               # node_synthesize
reflect: attempt=1 passed=True reason=ok                 # node_reflect
```

Each prefix maps to a stage. A line that doesn't match this pattern is
either coming from an error-suppressing path (a worker that logged a
warning mid-panel) or from the streaming variant, which uses the same
prefixes but also emits extra `worker_done` / `first_token` events.

When something goes wrong, the trace tells you a story:

```text
deep_panel: skipping unhealthy worker qwen3-coder-next
deep_panel: no healthy pool workers for deep_panel/code; falling back to registry
deep_panel: worker qwen3.6:35b failed in 4.12s: timeout
synth: qwen3.6:35b unhealthy, skipping (attempt 1)
synth: glm-5.2:cloud ok in 8.21s (attempt 2)
reflect: attempt=1 passed=False reason=too_short
reflect: attempt=2 passed=True reason=ok
```

You can read the cascade right off the page: the pool's primary worker was
unhealthy, the fallback to registry kicked in, that worker timed out, the
primary synth was also unhealthy, the fallback synth worked, but its
output was too short — so the panel re-ran and the second pass cleared the
gate. Audrey didn't fail the request; it just had to work harder.


## 3. Comprehension questions

**1. A user asks "What's the capital of France? Answer briefly." with
`audrey_deep`. Walk through what happens stage by stage. Does the panel
actually run all its workers, and does reflect catch anything?**

Classification routes the request to `general` — "capital of a country"
isn't code, reasoning, or VL. Complexity counts the prompt and finds it
well under the deep threshold, but `audrey_deep` forces deep regardless,
so mode = deep anyway. The planner is gated on
`planning.min_prompt_tokens` (default 40); a six-word prompt is nowhere
near it, so the planner returns `[]` without ever calling an LLM. The
deep panel runs both workers from the `deep_panel/general` pool against
the original prompt; both come back with short answers ("Paris."). The
synthesizer merges them — short input in, short output out. Reflect sees
the short output and would normally fail the `min_answer_chars` check —
except "briefly" is in `_BREVITY_CUES`, so it passes with
`ok_brevity_requested`. End result: roughly a second to "Paris."

**2. You're reading the logs and see
`deep_panel: pool=deep_panel task=code workers=1 ok=1 tool_grounded=1`.
Only one worker ran. The `tool_grounded=1` field is set. What does this
combination tell you about the pool's health and the worker's behavior?**

`workers=1` says one of two things happened: either the pool had exactly
one healthy worker out of N configured, or the registry fallback kicked
in and stopped at one (possible — the cap is 2, so one match could be
found, the next candidate unhealthy, and the loop ends). Either way, the
pool's other workers are either unhealthy (recent failures cooling down
in `HealthTracker`) or were removed from the pool list. The
`tool_grounded=1` field means the surviving worker is in
`fast_path.tool_capable_models` and ran a ReAct loop — at least one tool
call fired before the worker answered. None of this is alarming on its
own, but it's a nudge to check `HealthTracker.snapshot()` (admin
endpoint) and confirm the missing workers aren't permanently broken
rather than just cooling off.

**3. The synthesizer pool config has `synthesizer: "qwen3.6:35b"` but no
`fallback_synth` entry. What happens at boot? What happens if
`qwen3.6:35b` is unhealthy when a deep request arrives?**

At boot, `_validate_deep_panel_pools` walks every `deep_panel*`
pool/task and checks for `synthesizer`. `fallback_synth` is optional, so
a missing fallback slides past the validator without complaint. At
request time, `pick_synthesizer` returns `(primary, primary)` when
`fallback_synth` is missing — that's the `if not fallback: fallback =
primary` line doing its work. The candidates list collapses to one
entry, and the synth side loses its retry. If `qwen3.6:35b` is unhealthy
in that state, the synthesize loop logs `synth: qwen3.6:35b unhealthy,
skipping (attempt 1)`, runs out of candidates, and falls through to the
longest-draft degrade. The synth pass ships the worker's longest draft
verbatim — less polished than a real synthesis, but the user gets an
answer rather than an error.

**4. The planner returned `["explain X", "compare X to Y", "give an
example of X"]` but the pool has 2 healthy workers. How are the
subtasks distributed? What does the synthesizer see in its DRAFTS
block?**

Round-robin assigns by `i % len(subtasks)`: worker 0 gets subtask 0
("explain X"), worker 1 gets subtask 1 ("compare X to Y"). Subtask 2
("give an example of X") falls off the end — there aren't enough workers
to cover it. The synthesizer sees a DRAFTS block with two entries, plus
a PLANNED SUB-QUESTIONS block listing all three (the planner output
isn't filtered by what actually ran). This mismatch is deliberate: the
synthesizer sees what was *planned* even when the panel couldn't cover
all of it, so it can flag the gap explicitly in its synthesis. The
dropped subtask isn't a bug — it's the planner asking for more
parallelism than the pool offered.

**5. A user reports that `audrey_local` (the local-only deep pool) feels
slower than `audrey_deep` (mixed local+cloud) on the same prompt. The
pools have the same number of workers configured. Why might that be,
even with GPU concurrency unchanged?**

Both pools have two workers, but the *shape* of those two is what
differs. `audrey_deep` is one local plus one cloud — they run
concurrently, so the wall-clock is roughly the time of the slower one
(usually the local worker). `audrey_local` is two locals, both squeezing
through the GPU gate at `GPU_CONCURRENCY=1`. They run back to back, so
the wall-clock is roughly the sum of both. Same worker count, very
different concurrency profile — and the gate is the reason.

**6. Reflect returns `reflect_passed=False reason=no_drafts`. Should the
graph retry, and what would change on the retry? Trace the routing
logic.**

`reason=no_drafts` is special — the `route_after_reflect` router checks
`synth_error` and doesn't retry if it's `"no_drafts"`. Retrying would be
pointless: the panel produced zero usable drafts on the first pass, the
second pass would hit the same unhealthy workers and produce zero
drafts again. The graph routes straight to END with
`reflect_passed=False` in state, and the user gets the placeholder
message ("[deep panel produced no usable drafts — all workers failed]").
Operationally, this is a "your model layer is having a bad day" signal —
check `HealthTracker.snapshot()` and the Ollama / Ollama-Pro health.


## When you're ready for the next lesson

We have walked the four deep stages — planner, panel, synthesize, reflect —
and seen how the graph routes between them. This lesson treated each
worker's model call as a single round trip. The next lesson opens that
black box: when a model wants to invoke a tool (fast path or a single
deep worker), how does Audrey discover what tools exist, dispatch the
call, feed the result back, and stop the loop when the round budget
runs out? It lives in
[`lesson-09-tool-use-and-react.md`](lesson-09-tool-use-and-react.md).
