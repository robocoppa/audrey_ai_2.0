# Lesson 8 - Deep mode: planner, panel, synthesizer, reflect

**Estimated time:** 60-80 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer
*"when Audrey routes a request down the deep path instead of the fast one,
what actually happens to that request — and why does the answer take four
stages instead of one?"*

Lesson 7 left off with the complexity gate choosing `mode = deep`. This lesson
opens the door that mode goes through. We will follow one request through
four nodes:

```text
planner -> deep_panel -> synthesize -> reflect
```

Each node has a different job, a different failure mode, and a different
reason to exist. By the end you should be able to look at a deep-mode log
line and predict which node produced it.


## 1. Context

Fast mode is a single chat call: one model writes the answer. Deep mode is
deliberately not that. Deep mode runs several worker models in parallel,
then merges their drafts. The trade-off is straightforward:

- **Fast mode**: low latency, one perspective, cheap.
- **Deep mode**: higher latency, multiple perspectives merged, more expensive.

The user does not see the four stages — they see one answer streaming back.
But inside Audrey, four distinct things happen between "request arrived" and
"answer ready."

### 1.1 Why four stages and not one

The naive way to "answer harder" is to send the prompt to a bigger model.
Audrey does not do that. Instead it sends the prompt to a small **panel** of
workers and then merges them. Two reasons:

1. **Different models miss different things.** A code-tuned model and a
   reasoning-tuned model writing in parallel will catch each other's blind
   spots more reliably than either one writing alone.
2. **The merge step is cheap relative to the work.** Running the panel takes
   the bulk of the wall-clock time. Adding a synthesis pass on top adds one
   more chat call but produces an answer that reads as one voice instead of
   a stitched-together quilt.

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

```text
1. planner          (optional)
     - skip if prompt is short (`planning.min_prompt_tokens`)
     - skip if planner returned 0 or 1 subtask
     - otherwise: 2-3 sub-questions assigned round-robin to workers

2. deep_panel
     - pick the pool from the virtual model
       (audrey_deep / audrey_cloud / audrey_local)
     - filter the pool's workers by health
     - if no healthy workers, fall back to the model registry (cap 2)
     - run all workers in parallel
       - local workers serialize through the GPU gate
       - cloud workers run concurrently (capped at `max_deep_workers_cloud`)
       - tool-capable workers run a ReAct loop instead of one chat call

3. synthesize
     - look up the pool's synthesizer + fallback_synth
     - run primary; on failure, run fallback
     - if both fail, ship the longest non-empty draft verbatim

4. reflect
     - check the synth output meets `min_answer_chars`
     - skip the floor if the user explicitly asked for brevity
     - if it failed and we haven't already retried, loop back to deep_panel
```

Two routing rules outside the four nodes matter for understanding the trace:

- **Virtual model picks the pool.** `audrey_deep` uses the mixed
  local+cloud pool, `audrey_cloud` uses the cloud-only pool, `audrey_local`
  uses the local-only pool. `audrey_auto` lands here only when the
  complexity gate said deep — its pool is the same as `audrey_deep`.
- **Reflection is best-effort, bounded.** At most one retry of
  panel+synth. If the second attempt still fails the quality check,
  Audrey ships what it has rather than 502 the request.


## 2. Read-along

These are the files we'll reference:

- [`src/audrey/pipeline/graph.py:265`](../../src/audrey/pipeline/graph.py#L265)
  — `node_planner`, the LangGraph entry to planning.
- [`src/audrey/pipeline/planner.py:32`](../../src/audrey/pipeline/planner.py#L32)
  — `plan()`, the planner LLM call and its parser.
- [`src/audrey/pipeline/graph.py:283`](../../src/audrey/pipeline/graph.py#L283)
  — `node_deep_panel`, the LangGraph entry to dispatch.
- [`src/audrey/pipeline/deep_panel.py:59`](../../src/audrey/pipeline/deep_panel.py#L59)
  — `pool_key_for`, the virtual-model → pool mapping.
- [`src/audrey/pipeline/deep_panel.py:68`](../../src/audrey/pipeline/deep_panel.py#L68)
  — `select_workers`, healthy-worker selection plus the registry fallback.
- [`src/audrey/pipeline/deep_panel.py:105`](../../src/audrey/pipeline/deep_panel.py#L105)
  — `_run_one_worker`, the per-worker chat or ReAct loop.
- [`src/audrey/pipeline/deep_panel.py:200`](../../src/audrey/pipeline/deep_panel.py#L200)
  — `_messages_for_subtask`, how each worker gets its slice of the prompt.
- [`src/audrey/pipeline/deep_panel.py:220`](../../src/audrey/pipeline/deep_panel.py#L220)
  — `run_panel`, the panel dispatcher.
- [`src/audrey/pipeline/synthesize.py:41`](../../src/audrey/pipeline/synthesize.py#L41)
  — `_format_drafts_for_synth`, the draft-bundling for the synth prompt.
- [`src/audrey/pipeline/synthesize.py:81`](../../src/audrey/pipeline/synthesize.py#L81)
  — `pick_synthesizer`, the primary/fallback selector.
- [`src/audrey/pipeline/synthesize.py:167`](../../src/audrey/pipeline/synthesize.py#L167)
  — `synthesize`, the merge orchestrator.
- [`src/audrey/pipeline/reflect.py:69`](../../src/audrey/pipeline/reflect.py#L69)
  — `reflect`, the deterministic quality gate.
- [`config.yaml:79`](../../config.yaml#L79) — the `deep_panel` mixed pool.
- [`config.yaml:99`](../../config.yaml#L99) — the `deep_panel_cloud` pool.
- [`config.yaml:119`](../../config.yaml#L119) — the `deep_panel_local` pool.
- [`src/audrey/pipeline/prompts.py:62`](../../src/audrey/pipeline/prompts.py#L62)
  — `PLANNER_SYSTEM`, the planner instruction.
- [`src/audrey/pipeline/prompts.py:74`](../../src/audrey/pipeline/prompts.py#L74)
  — `SYNTH_SYSTEM`, the synthesizer instruction.

Open them as we go.


### 2.1 Stage 1 — Planner

The planner asks one question: *"Is this user request really one ask, or
several asks bundled together?"*

If it's one ask, the panel runs the prompt as-is. If it's several asks, the
panel splits them — one worker per sub-question — so each worker focuses on
a tighter slice.

#### When the planner runs

The graph node is at
[`graph.py:265`](../../src/audrey/pipeline/graph.py#L265):

```python
async def node_planner(state: PipelineState) -> dict[str, Any]:
    if not planning_enabled or state.get("prompt_tokens", 0) < planning_min_tokens:
        return {"subtasks": []}
```

Two early exits. If `agentic.planning.enabled` is off, return no subtasks.
If the prompt is shorter than `planning.min_prompt_tokens` (default 40), also
return no subtasks — the planner round-trip costs more than it saves on
short prompts.

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
[`planner.py:54`](../../src/audrey/pipeline/planner.py#L54):

```python
resp = await ollama.chat(
    model=planner_model,
    messages=messages,
    options={"temperature": 0.0},
    timeout_s=timeout_s,
)
```

`temperature=0.0` because we want deterministic decomposition. The planner
isn't being creative — it's reading structure.

#### The parser is intentionally forgiving

LLM JSON output is unreliable in practice — leading prose, trailing
markdown, mismatched braces. The parser handles that at
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

`find('{')` + `rfind('}')` grabs the outermost brace pair, dropping any
preamble or trailing commentary. The slice can still be malformed — a stray
`}` in the prose, two JSON objects in one response — and in that case
`json.loads` raises and the parser returns `[]`. That permissiveness is
intentional: every planner failure mode degrades to "no decomposition," and
the panel handles that fine.

#### Three exits, all benign

The planner has three ways to return `[]` (no decomposition):

1. The model returned 0 subtasks (recognized atomic prompt — by design).
2. The model returned 1 subtask (probably failed to decompose; logged at
   debug, treated as no decomposition).
3. The model returned malformed JSON (parsing failed; degrades to `[]`).

All three paths produce the same downstream behavior: `subtasks = []` flows
into the deep panel, and the panel runs every worker against the original
prompt. The planner is opt-in routing, not a hard requirement.

When the planner *does* return subtasks, the log line at
[`graph.py:278`](../../src/audrey/pipeline/graph.py#L278) shows the count
and the first 60 chars of each:

```python
log.info("planner: %d subtasks: %s", len(subs), [s[:60] for s in subs])
```


### 2.2 Concept spotlight — `asyncio.gather` and the parallel-worker idea

Before we open the deep panel, a quick aside on what "parallel" means here.

Python is single-threaded for CPU work, but `asyncio` lets one thread juggle
many in-flight I/O operations. `await ollama.chat(...)` is mostly waiting
for the network — the model generates tokens on the GPU (or in the cloud),
the bytes come back over HTTP, and Python's event loop is free to do other
things during the wait.

`asyncio.gather(*coros)` takes a list of those coroutines and runs them
**concurrently**. From the caller's perspective:

```python
drafts = await asyncio.gather(coro_worker_a, coro_worker_b, coro_worker_c)
```

…blocks until *all* of them finish, then returns a list of results in
submission order. While it's blocking, the event loop interleaves the
three workers' I/O. If worker A is waiting on the network, the loop dispatches
B's next request, then C's, then comes back to A when bytes arrive.

For deep-panel workers, this is the entire reason multiple cloud workers
finish in roughly the time of one. Three workers each waiting 12 seconds on
the cloud take roughly 12 seconds total, not 36 — because their waits
overlap.

Local workers don't get the same speedup, because they all want the same
GPU. That's where the next concept comes in.


### 2.3 Stage 2 — Deep panel

The deep panel runs the actual workers. Three knobs decide who runs:

1. Which **pool** to draw from (set by the virtual model).
2. Which workers in the pool are **healthy** (set by `HealthTracker`).
3. Whether each worker is **tool-capable** (set by `fast_path.tool_capable_models`).

#### Picking the pool

`pool_key_for` at
[`deep_panel.py:59`](../../src/audrey/pipeline/deep_panel.py#L59) does the
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

The fallback exists because the function is also called from the streaming
route, where a misspelled virtual model would otherwise cause a `KeyError` at
request time. Instead it warns and uses the default pool — the answer still
ships, but the operator sees the typo in the logs.

The pools themselves live in `config.yaml`. Open
[`config.yaml:79`](../../config.yaml#L79):

```yaml
deep_panel:
  code:
    workers: ["qwen3-coder-next:latest", "kimi-k2.6:cloud"]
    synthesizer: "qwen3.6:35b"
    fallback_synth: "glm-5.1:cloud"
  reasoning:
    workers: [...]
    synthesizer: "qwen3.6:35b"
    fallback_synth: "glm-5.1:cloud"
  ...
```

Each pool is keyed by **task type** (code, reasoning, general, vl). So
"which workers run" depends on both the virtual model (picks the pool) and
the classifier output from Lesson 7 (picks the task entry within the pool).

#### Selecting healthy workers

`select_workers` at
[`deep_panel.py:92`](../../src/audrey/pipeline/deep_panel.py#L92) walks the
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

`registry.location_of(name)` is the model-layer method that says "where
does this model run?" — local or cloud. It walks the registry rather than
trusting the pool's declared shape, so a renamed model still gets the right
treatment.

#### The registry fallback

If `select_workers` returns nothing — every pool worker is unhealthy —
`run_panel` falls back to the model registry itself at
[`deep_panel.py:264`](../../src/audrey/pipeline/deep_panel.py#L264):

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

The cap of 2 mirrors the typical pool size — we want enough drafts to merge
without flooding the GPU gate or burning cloud quota on what is already
an emergency path. The log line is a `warning` so operators notice when the
pool has lost its workers; if you see this in production, the pool's models
need attention.

#### Concept spotlight — the GPU gate and per-worker accounting

Local workers go through a `FairLocalGate` semaphore (Lesson 6 introduced
this). With `GPU_CONCURRENCY=1` — the production default — only one local
worker can run at a time, because they all want the same GPU's VRAM.

Per-worker dispatch looks parallel on the dispatcher side
([`deep_panel.py:294-313`](../../src/audrey/pipeline/deep_panel.py#L294))
but execution serializes through the gate. A deep panel with two local
workers will run them one after the other, not at the same time. Two cloud
workers in the same panel run concurrently because they don't hold the gate.

This is also why tool-capable local workers hold the gate for the *entire*
ReAct loop, not just one chat call. Look at
[`deep_panel.py:134-155`](../../src/audrey/pipeline/deep_panel.py#L134):

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

The `gate=None` is the key detail. The ReAct loop is told to *not* acquire
its own gate, because the panel already holds it. If ReAct acquired again
mid-loop, a local worker would briefly release the GPU between rounds —
giving another local worker (or another user's request) a window to grab
it. Holding for the whole loop keeps tool-using workers atomic from the GPU
gate's perspective.

One subtle consequence worth knowing for log-reading: when a worker runs
ReAct, the `WorkerDraft.prompt_eval_count` and `eval_count` carry only the
**final** chat call's token counts, not the sum across rounds. A
tool-grounded worker that ran 3 rounds reports only the last round's
tokens. This isn't a metrics bug — per-worker totals work for the
dashboards Audrey actually uses — but if you compare a tool-grounded
worker's token count to a one-shot worker's count, the tool-grounded one
looks artificially light. That's the accounting working as designed.

#### Sub-question round-robin

If the planner produced subtasks, each worker gets one. `run_panel`
distributes them round-robin at
[`deep_panel.py:281-285`](../../src/audrey/pipeline/deep_panel.py#L281):

```python
if subtasks:
    per_worker_messages = [
        _messages_for_subtask(messages, subtasks[i % len(subtasks)])
        for i in range(len(workers))
    ]
else:
    per_worker_messages = [messages] * len(workers)
```

Three subtasks and two workers: worker 0 gets subtask 0, worker 1 gets
subtask 1, subtask 2 is dropped (the modulo would wrap, but with
`len(workers) < len(subtasks)` the loop just doesn't reach it). Two
subtasks and three workers: worker 0 gets subtask 0, worker 1 gets
subtask 1, worker 2 gets subtask 0 again — two workers answer the same
question with different perspectives, which the synthesizer reconciles.

`_messages_for_subtask` at
[`deep_panel.py:216`](../../src/audrey/pipeline/deep_panel.py#L216) builds
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

The semantic matters in multi-turn conversations. "The last user message" is
this turn's question — the one the planner just decomposed — not an
arbitrary earlier turn. Earlier user/assistant pairs stay in place as
context, so the worker still sees the full thread; only the focal question
becomes the subtask.

#### Worker output: `WorkerDraft`

Each worker returns a `WorkerDraft` regardless of outcome — even on error.
The shape is at [`pipeline/state.py`](../../src/audrey/pipeline/state.py),
but for now the fields that matter are:

- `content` — the model's text output, possibly empty on failure.
- `error` — populated only on `OllamaError`; truncated to 300 chars.
- `tool_rounds` — `> 0` if the worker ran ReAct.
- `model`, `elapsed_s`, `tool_calls` — observability fields.

Never-raising is a load-bearing contract. The synthesizer needs the full
draft list to be present, even if some entries are empty error rows —
otherwise a single misbehaving worker could 502 the entire request.


### 2.4 Stage 3 — Synthesizer

The synthesizer takes the worker drafts and merges them into one final
answer. It runs one LLM call (or two, on retry).

#### Picking the synthesizer

The pool config has two slots:

```yaml
synthesizer: "qwen3.6:35b"
fallback_synth: "glm-5.1:cloud"
```

`pick_synthesizer` at
[`synthesize.py:92`](../../src/audrey/pipeline/synthesize.py#L92) reads
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

The `KeyError` would normally never trigger in production — startup
validation in [`config.py`](../../src/audrey/config.py) walks every
`deep_panel*` pool/task at boot and crashes the process if any synthesizer
is missing. The function still raises defensively in case someone bypasses
the validator.

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

The `[tool-grounded: N rounds]` tag is informational. The synthesizer
prompt (read it at
[`prompts.py:74`](../../src/audrey/pipeline/prompts.py#L74)) explicitly
tells the model to prefer tool-grounded drafts on factual disagreements:

> When a tool-grounded draft and a tool-free draft disagree on a factual
> point, prefer the tool-grounded one.

This is how Audrey reconciles "the model thinks the answer is X" with "the
search tool says the answer is Y." Y wins, but only because the synthesizer
is told to read the tag.

#### Forwarding original system context

The synthesizer runs against the same system messages the workers saw. Open
[`synthesize.py:104`](../../src/audrey/pipeline/synthesize.py#L104):

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

The original system messages include the datetime context (set by
`node_datetime` — Lesson 7 covered the graph entry), memory recall hits,
and any OWUI-supplied templates. Without them the synthesizer would
"hedge" about "today" using its training cutoff, which makes the merged
answer feel stale.

`draft_count` is passed explicitly so a future change to the draft
separator string in `_format_drafts_for_synth` can't silently break the
count the synthesizer is told to expect.

#### Three-tier failure handling

The synthesizer can fail in three ways. Each one is handled differently —
read [`synthesize.py:211`](../../src/audrey/pipeline/synthesize.py#L211):

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
   [`synthesize.py:186-193`](../../src/audrey/pipeline/synthesize.py#L186)
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

Never 502 the request because the synth failed. The worker drafts are
the evidence base — at least one of them produced something the user
can read, and Audrey ships that something rather than nothing.


### 2.5 Stage 4 — Reflect

Reflect is the cheapest stage in the pipeline. It runs no LLM calls — it
only inspects the synthesized content. Open
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

The synth prompt is permissive about length — it's told to write the best
answer for the user, not to hit a word count. That works for ordinary
requests, where the answer is naturally substantive. It breaks on
"What year is it? Answer in one sentence" because the correct answer is
~15 chars and gets retried as "too short" wastefully.

The `_BREVITY_CUES` tuple at
[`reflect.py:35`](../../src/audrey/pipeline/reflect.py#L35) is the
hardcoded list of phrases that trip the escape hatch. It's English-only
because Audrey is English-only; a non-English user asking for brevity
in their own language would trigger the same wasteful retry the cues
were added to prevent.

#### Retry routing

The retry isn't in `reflect()` — `reflect()` only inspects. The retry
decision is in `route_after_reflect` (graph router) and it gates on three
things:

1. `reflect_passed` must be False.
2. `reflect_attempts` must be ≤ `reflection_max_retries` (default 1).
3. `synth_error` must not be `"no_drafts"` (no point retrying with no
   workers).

When the retry fires, the panel and synth run again with one nudge:
`compose_system_messages` injects a brief "be more substantive" hint into
the second pass. If the second pass still fails the length check, Audrey
ships what it has and marks `reflect_passed=False` in the state — visible
to whoever's reading logs.


### 2.6 The deep loop in one trace

Here is what an info-level log for a typical deep request looks like, with
the stage producing each line annotated on the right:

```text
classify: task=reasoning conf=0.72                       # Lesson 7
complexity: tokens=287 mode=deep reason=token_count      # Lesson 7
planner: 2 subtasks: ['How does ...', 'What about ...']  # node_planner
deep_panel: pool=deep_panel task=reasoning workers=2     # node_deep_panel
synth: qwen3.6:35b ok in 6.42s (attempt 1)               # node_synthesize
reflect: attempt=1 passed=True reason=ok                 # node_reflect
```

Each prefix maps to a stage. If a line in your trace doesn't match this
pattern, the stage that produced it is either error-suppressing (e.g.
a worker logged a warning during the panel call) or running on the
streaming path (the streaming variant uses similar prefixes but emits
extra `worker_done`/`first_token` events).

When something goes wrong, the trace looks like:

```text
deep_panel: skipping unhealthy worker qwen3-coder-next
deep_panel: no healthy pool workers for deep_panel/code; falling back to registry
deep_panel: worker qwen3.6:35b failed in 4.12s: timeout
synth: qwen3.6:35b unhealthy, skipping (attempt 1)
synth: glm-5.1:cloud ok in 8.21s (attempt 2)
reflect: attempt=1 passed=False reason=too_short
reflect: attempt=2 passed=True reason=ok
```

You can read off the cascade: the pool's primary worker was unhealthy, the
fallback to registry kicked in, that worker timed out, the primary synth
was also unhealthy, the fallback synth worked, but its output was too
short, so the panel re-ran and the second pass cleared the gate.


## 3. Comprehension questions

These are scenario-based — work through them as if you were diagnosing a
live system. Suggested answers follow.

**Q1.** A user asks "What's the capital of France? Answer briefly." with
`audrey_deep`. Walk through what happens stage by stage. Does the panel
actually run all its workers, and does reflect catch anything?

**Q2.** You're reading the logs and see:

```text
deep_panel: pool=deep_panel task=code workers=1 ok=1 tool_grounded=1
```

Only one worker ran. The `tool_grounded=1` field is set. What does this
combination tell you about the pool's health and the worker's behavior?

**Q3.** The synthesizer pool config has `synthesizer: "qwen3.6:35b"` but
no `fallback_synth` entry. What happens at boot? What happens if
`qwen3.6:35b` is unhealthy when a deep request arrives?

**Q4.** The planner returned `["explain X", "compare X to Y", "give an
example of X"]` but the pool has 2 healthy workers. How are the subtasks
distributed? What does the synthesizer see in its DRAFTS block?

**Q5.** A user reports that `audrey_local` (the local-only deep pool)
feels slower than `audrey_deep` (mixed local+cloud) on the same prompt.
The pools have the same number of workers configured. Why might that
be, even with GPU concurrency unchanged?

**Q6.** Reflect returns `reflect_passed=False reason=no_drafts`. Should
the graph retry, and what would change on the retry? Trace the routing
logic.

---

### Suggested answers

**A1.** Classification probably routes the request to `general` (capital
of a country isn't code, reasoning, or VL). Complexity counts the prompt —
it's short, well under the deep threshold, but `audrey_deep` is a forced-
deep virtual model, so mode = deep anyway. The planner is gated on
`planning.min_prompt_tokens` (default 40); a six-word prompt is well
under that, so the planner returns `[]` without making an LLM call. The
deep panel runs both workers from the `deep_panel/general` pool with the
original prompt; both workers return short answers ("Paris."). The
synthesizer merges them — short input, short output. Reflect sees the
short output and would normally fail the `min_answer_chars` check, but
"briefly" is in `_BREVITY_CUES` so it passes with `ok_brevity_requested`.
End result: ~one second to "Paris."

**A2.** `workers=1` means either the pool had one healthy worker out of
N configured, or the registry fallback kicked in and stopped at one
(possible since the cap is 2 — one match was found, then loop ended
when the next candidate was unhealthy). The pool's other workers are
either unhealthy (recent failures cooling down in `HealthTracker`) or
were removed from the pool list. `tool_grounded=1` means the surviving
worker is in `fast_path.tool_capable_models` and ran a ReAct loop —
it made at least one tool call before answering. Mostly this is fine,
but you should check `HealthTracker.snapshot()` (admin endpoint) to
confirm the missing workers aren't permanently broken.

**A3.** At boot, `_validate_deep_panel_pools` walks every `deep_panel*`
pool/task and checks for `synthesizer`. `fallback_synth` is optional,
so a missing fallback doesn't block boot. At request time,
`pick_synthesizer` returns `(primary, primary)` when `fallback_synth`
is missing — `if not fallback: fallback = primary`. The candidates list
collapses to one entry, so there's effectively no retry on the synth
side. If `qwen3.6:35b` is unhealthy, the synthesize loop logs `synth:
qwen3.6:35b unhealthy, skipping (attempt 1)`, skips the only candidate,
and falls through to the longest-draft degrade. The synth pass produces
the worker's longest draft verbatim — less polished than a real synth
would write, but the user still gets an answer.

**A4.** Round-robin assigns by `i % len(subtasks)`: worker 0 gets subtask 0
("explain X"), worker 1 gets subtask 1 ("compare X to Y"). Subtask 2
("give an example of X") is dropped — there aren't enough workers to
cover it. The synthesizer sees a DRAFTS block with two entries, plus a
PLANNED SUB-QUESTIONS block listing all three subtasks (the planner output
isn't filtered by what actually ran). This is intentional: the
synthesizer sees what was *planned* even when the panel couldn't cover
all of it, so it can flag gaps in its synthesis. The dropped subtask
isn't a bug — it's the planner asking for more parallelism than the
pool offered.

**A5.** Even though `audrey_local` and `audrey_deep` both have two
workers in most pool/task entries, the `audrey_deep` pool is one local +
one cloud, so they run concurrently — the wall-clock is roughly the time
of the slower one (usually the local). `audrey_local` is two locals,
serialized through the GPU gate at `GPU_CONCURRENCY=1`. The two workers
run sequentially, so the wall-clock is roughly the sum of both. Same
worker count, different concurrency profile because of the gate.

**A6.** `reason=no_drafts` is special — the route_after_reflect router
checks `synth_error` and doesn't retry if it's `"no_drafts"`. Retrying
would be pointless: the panel produced zero usable drafts on the first
pass, the second pass would hit the same unhealthy workers and produce
zero drafts again. The graph routes straight to END with
`reflect_passed=False` in state, and the user gets the placeholder
message ("[deep panel produced no usable drafts — all workers failed]").
Operationally, this is a "your model layer is having a bad day" signal —
check `HealthTracker.snapshot()` and the Ollama / Ollama-Pro health.
