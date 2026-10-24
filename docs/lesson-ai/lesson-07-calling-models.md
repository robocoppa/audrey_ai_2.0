# Lesson 7 — The model layer II: calling models

**Estimated time:** 45-60 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"once a model is chosen,
how does Audrey call it, what happens when the call fails, and what changes
when someone picks Private?"*

Lesson 6 ended with a chosen model: the best currently healthy candidate for
the task. This lesson follows the call. It reads the one client that speaks
to Ollama, the thinking controls, the three callers that use it (the fast
path, the deep panel and the synthesizer) and what each does when a call
fails, how a model that can't see images still answers about one, and how
the Private choice closes every route to the cloud.


## 1. Context

Every model call in Audrey goes through one object, `OllamaClient`, whether
the model runs on this server's GPU or in Ollama's cloud. Around each call,
two other pieces from Lesson 6 do their part:

```text
choose a model          registry + health tracker      (Lesson 6)
wait for the GPU        FairLocalGate, local models only
call the model          OllamaClient
record the outcome      success clears health; failure starts a cooldown
fall back if possible   another candidate, another worker, or the best draft
```

The last step is where the three callers differ. The fast path tries one
more candidate. The deep panel treats a failed worker as a missing draft and
carries on. The synthesizer tries its fallback, then hands over the best
draft it has. The common thread: **a failed model call becomes data the
caller can act on, not a crash.**


## 2. Read-along

These are the files this lesson reads:

- [`models/ollama.py`](../../src/audrey/models/ollama.py): the client.
- [`pipeline/fast_path.py`](../../src/audrey/pipeline/fast_path.py): one
  model answers.
- [`pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py):
  several workers answer.
- [`pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py): one
  answer from several drafts.
- [`pipeline/vision.py`](../../src/audrey/pipeline/vision.py): images for
  models that can't see them.
- [`private_processing.py`](../../src/audrey/private_processing.py): the
  Private runtime.

### 2.1 `OllamaClient`: the one HTTP boundary

[`OllamaClient`](../../src/audrey/models/ollama.py#L410) is the only part of
the model layer that speaks HTTP. Everything else chooses names and decides
what to do with success or failure. It wraps one `httpx.AsyncClient`, built
at [`ollama.py:426`](../../src/audrey/models/ollama.py#L426):

```python
self._client = httpx.AsyncClient(
    base_url=self._base_url,
    timeout=httpx.Timeout(default_timeout_s),
    headers={"Accept": "application/json"},
    transport=transport,
    trust_env=trust_env,
)
```

`base_url` lets callers write `/api/chat` instead of a full URL, and
`timeout` gives every call a default deadline that a call can override.
Cloud models need no second client: Ollama serves `:cloud` models through the
same API, so "cloud" is a scheduling fact, not a different transport.

The client's methods are all `async`: [`tags`](../../src/audrey/models/ollama.py#L444)
lists the installed models, [`show`](../../src/audrey/models/ollama.py#L459)
reads one model's details, [`chat`](../../src/audrey/models/ollama.py#L596)
and [`chat_stream`](../../src/audrey/models/ollama.py#L683) generate, and
[`embed`](../../src/audrey/models/ollama.py#L793) turns text into vectors for
the knowledge base. Before a chat goes out,
[`_to_ollama_messages`](../../src/audrey/models/ollama.py#L276) reshapes the
messages: An OpenAI-style message can carry a list of text and image parts,
while Ollama wants plain text plus an `images` list.

Three kinds of failure can happen, and the client turns all three into one
exception, `OllamaError`:

| Failure | Example |
| --- | --- |
| Transport | a timeout, a refused connection |
| HTTP status | Ollama answers 500 or 503 |
| Response body | status 200, but not the JSON object Audrey expects |

The third is easy to miss: A successful status isn't enough. At
[`ollama.py:858`](../../src/audrey/models/ollama.py#L858):

```python
def _json_object(r: httpx.Response, op: str) -> dict[str, Any]:
    try:
        body = r.json()
    except ValueError as e:
        ...
        raise OllamaError(f"{op}: invalid JSON response: {e}") from e
    if not isinstance(body, dict):
        raise OllamaError(f"{op}: expected JSON object, got {type(body).__name__}")
    return body
```

One exception type gives every caller one contract: Catch `OllamaError`,
record the failure, and fall back if the path has a fallback.

### 2.2 Two ways to read an answer

`chat` waits for the whole answer and returns one dictionary. `chat_stream`
is an async iterator: It yields a small dictionary per chunk as Ollama
produces the words, so callers read it with `async for`. The browser's
word-by-word answers come from the second.

The difference matters for fallback. Before any words have gone out, a
caller can still try another model and nobody notices. Once words have
reached the person, Audrey can't pretend they didn't: A failure after that
point has to be reported inside the stream, as a cut-off answer with a
note, never as a silent restart.

### 2.3 Thinking controls

Some models **think** before they answer: They write out their reasoning
first, as tokens that cost time but never appear in the answer. That helps a
hard question and wastes time on an easy one, so Audrey can switch it off
per model or per role.

The switch has three states, and the difference between two of them is
load-bearing. Sending `think: false` turns thinking off, but Ollama rejects
the field outright for a model without the thinking capability. Leaving the
field out is safe for every model. So the default is to leave it out, and
[`thinking_flag`](../../src/audrey/models/ollama.py#L547) sends a value only
after checking the model's capabilities:

```python
async def thinking_flag(self, model: str, want: bool) -> bool | None:
    ...
    cached = self._thinking_caps.get(model)
    if cached is None:
        try:
            cached = "thinking" in await self.capabilities(model)
        except OllamaError as e:
            log.info("ollama: capability probe for %s failed (%s)", model, e)
            return None
        self._thinking_caps[model] = cached
    return want if cached else None
```

It asks once per model and remembers the answer for the life of the
process. If it can't find out, it returns `None`, so a passing Ollama
problem leaves the field out instead of breaking every later call.

Which calls should skip thinking is configuration, in the
[`thinking`](../../config.yaml#L581) section, read by
[`think_for`](../../src/audrey/pipeline/deep_panel.py#L249) at
[`deep_panel.py:282`](../../src/audrey/pipeline/deep_panel.py#L282):

```python
section = getattr(cfg, "thinking", None) or {}
...
named = {str(m) for m in (section.get("no_thinking_models") or [])}
role_off = bool(section.get(role, False))
if model not in named and not role_off:
    return None
return await ollama.thinking_flag(model, False)
```

A model can be named on its own, or a whole role (every deep worker, say)
switched off. Per model is preferred: The evidence for turning thinking off
usually comes from one model on one kind of task. The same policy applies to
streamed and completed synthesis. Research verification and writing use
`deep_synth`; researchers and the prose factchecker use `deep_worker`. A named
model override therefore reaches every deep role it holds.

### 2.4 The fast path: one model answers

The fast path starts from
[`pick_fast_model`](../../src/audrey/pipeline/fast_path.py#L81), which is
Lesson 6's priority-and-health rule, including the recovery probe when every
candidate is cooling down; only a task with no candidates at all raises an
error. Then it decides whether the model may use tools. If the model is on the
tool-capable list and tools are available, the turn goes to the tool loop,
which a later lesson reads. Otherwise it makes a plain call, with a small
fallback, at
[`fast_path.py:586`](../../src/audrey/pipeline/fast_path.py#L586):

```python
async with gate.acquire(cand.name, location=cand.location, user_id=user_id):
    ...
    resp = await ollama.chat(**chat_kwargs)
health.record_success(cand.name)
return cand.name, resp
```

and, when the call fails:

```python
except OllamaError as e:
    health.record_failure(cand.name, str(e))
    last_err = e
```

`cand` is the candidate being tried, and the loop tries at most
[`_FAST_FALLBACK_LIMIT`](../../src/audrey/pipeline/fast_path.py#L97) healthy
candidates in priority order: the best one, then one fallback. If both fail,
the last error goes back to the caller. If all candidates are cooling, only
one recovery probe is tried. A tool-using turn isn't retried on
another model, because a tool call may already have had effects.

This is also where `location` from Lesson 6 does its job. For a local model,
`gate.acquire` waits its turn for the GPU; for a cloud model, it lets the call
straight through. And it is the model layer's central contract in two lines:
success clears the model's health record; `OllamaError` starts a cooldown.

### 2.5 The deep panel: several workers answer

Deep mode starts from a pool, not a ranked list. The choice picks the pool,
through [`_POOL_KEYS`](../../src/audrey/pipeline/deep_panel.py#L106):

```python
_POOL_KEYS = {
    "audrey_auto": "deep_panel",
    "audrey_video": "deep_panel",
    "audrey_deep": "deep_panel",
    "audrey_cloud": "deep_panel_cloud",
    "audrey_local": "deep_panel_local",
    "audrey_research": "deep_panel_research",
}
```

[`select_workers`](../../src/audrey/pipeline/deep_panel.py#L146) reads the
pool's worker names for the task, skips any that are cooling down, and caps
how many cloud workers run at once, at
[`deep_panel.py:165`](../../src/audrey/pipeline/deep_panel.py#L165):

```python
for name in raw_workers:
    if not health.is_healthy(name):
        log.info("deep_panel: skipping unhealthy worker %s", name)
        continue
    loc = registry.location_of(name)
    if loc == "cloud":
        if cloud_count >= max_workers_cloud:
            ...
            continue
        cloud_count += 1
    out.append((name, loc))
```

The cap protects the cloud account's concurrency limit; local workers need no
cap here, because the GPU gate already makes them take turns. A cloud worker
over the cap is dropped with a warning in the log that names it and the cap.
Without that warning, a pool configured with more cloud workers than the cap
would quietly run fewer, and nothing would say why. If no configured worker
is healthy, the panel falls back to healthy candidates from the registry
instead, in
[`_prepare_panel`](../../src/audrey/pipeline/deep_panel.py#L549).

Each worker runs in
[`_run_one_worker`](../../src/audrey/pipeline/deep_panel.py#L345), whose
promise is in its first line, at
[`deep_panel.py:367`](../../src/audrey/pipeline/deep_panel.py#L367):

```python
"""Execute one worker. Always returns a WorkerDraft — never raises.
```

A worker that fails records the failure and returns a draft with an `error`
field and no content. That's the difference from the fast path. The fast path
needs one good answer; the deep panel needs enough material, and one failed
worker out of three still leaves two drafts to work from.

### 2.6 Synthesis: primary, fallback, then the best draft

The synthesizer turns the drafts into one answer. It isn't picked by health
ranking: [`pick_synthesizer`](../../src/audrey/pipeline/synthesize.py#L106)
reads the pool's `synthesizer` and `fallback_synth`, and
[`synthesize`](../../src/audrey/pipeline/synthesize.py#L206) tries them in
order, at
[`synthesize.py:239`](../../src/audrey/pipeline/synthesize.py#L239):

```python
candidates = [primary] if primary == fallback else [primary, fallback]
```

Each attempt follows the same pattern as everywhere else: Skip a model that's
cooling down, use its location for the gate, call, record the outcome. If
both fail, Audrey returns the longest draft, at
[`synthesize.py:279`](../../src/audrey/pipeline/synthesize.py#L279):

```python
best = max(drafts, key=lambda d: len(d.get("content") or ""))
```

That's weaker than a real synthesis, but the panel's work is already paid
for, and one worker's full answer beats an error. Only when no worker
produced anything does Audrey say so outright: "deep panel produced no usable
drafts".

```text
No answer material: say so.
Answer material but no synthesis: return the best material.
```

### 2.7 Images for models that can't see them

Only the models in the `vl` pool can see images. Every other model is text in,
text out: Hand one an image, and it either refuses or answers blind. So when a
model that can't see is the one the person asked for, Audrey splits seeing from
reasoning. A local vision model describes the image once, in text, and the
description replaces the image in the conversation. The chosen model then
answers from that text. The entry point is
[`describe_for_text_model`](../../src/audrey/pipeline/vision.py#L621), and
[`describe_for_text_models`](../../config.yaml#L1431) in config switches it
off.

A description isn't the pixels, so the instruction to the vision model leans
hard on copying text exactly, which is what screenshots, diagrams and error
messages need. Descriptions are cached per image, because a conversation
that keeps an image in its history would otherwise pay for the same
description on every turn. The next lesson shows when an image turn goes to
a vision model directly and when it's described first.

### 2.8 Private: the same layer with every route to the cloud closed

Private is a choice in the picker, like Auto or Deep. What it changes is where
anything may run. When someone picks it:

- **The server attaches a policy to the run.** It comes from the server's own
  settings, never from the browser, and nothing later in the run can weaken
  it. The conversation is marked Private for good.
- **Every model is local and verified.** The run gets its own copy of the
  model configuration with only approved local models in it
  ([`private_configuration`](../../src/audrey/private_processing.py#L133)), and
  each call checks that the model is the exact local model expected. Routing,
  answers, embeddings and file processing all stay on this server.
- **A failure says "unavailable".** Where an ordinary request would fall back
  to a cloud model, a Private request stops and reports that the local model
  is unavailable.
- **Tools are local only.** Your files, the calculator and the reference
  library work; web search and shared memory are refused, and a Private run
  only ever sees Private files.
- **Shared services are an explicit list.** The run may reuse a declared set
  of local services, such as storage and the GPU gate. Anything else is
  refused rather than borrowed from the ordinary application, so a service
  added later is closed to Private until someone decides otherwise.
- **The restriction sticks.** It stays with the conversation, the Project and
  the files added through it, so protected material can't reach a cloud model
  later, and Private turns never enter the shared chat history.

"Private" describes where the processing happens: on this server. It isn't a
private network path. The browser still reaches Audrey through Cloudflare, and
whoever administers the server can still reach the server.

### 2.9 One request, end to end

Lesson 6 followed Fast and *"What is BTRFS?"* as far as a chosen model. From
there:

1. The fast path has its candidate. The model isn't on the tool-capable list,
   so the turn is a plain call.
2. `think` is left out unless config switches thinking off for this model and
   the model says it supports thinking.
3. The model is local, so `gate.acquire` waits for the GPU.
4. `OllamaClient.chat` posts to `/api/chat` and checks the response is a JSON
   object.
5. Success: `health.record_success`, and the answer goes back up the
   pipeline.
6. Failure: `health.record_failure` starts a cooldown, and the fast path tries
   the next healthy candidate. If that fails too, the error goes back to the
   caller.


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. Audrey can sometimes try another model after a failed call, and sometimes
can't. What decides which?**

Whether any words have reached the person. With `chat`, nothing is sent until
the whole answer exists, so a caller can quietly try another model. With
`chat_stream`, words go out as they arrive; once some have, a failure has to
be reported inside the stream as a cut-off answer, because retrying from
scratch would contradict what the person already saw.

**2. Why does the client turn transport errors, error statuses and malformed
bodies all into `OllamaError`?**

Because every caller handles one contract: Catch `OllamaError`, record a
health failure, and fall back if the path can. If a raw `httpx` error or a
JSON parsing error escaped instead, some paths would miss it, and a model
that had just failed would still count as healthy for the next request.

**3. The deep panel produced three drafts, and both synthesizers fail. What
does the person get?**

The longest draft, returned as the answer. It isn't as good as a synthesis,
but the workers' answers are already paid for, and a complete answer from one
worker beats an error. Only if no worker produced anything does Audrey say,
in so many words, that the panel produced no usable drafts.

**4. A deep pool lists three cloud workers, but the cloud cap is two. What
happens, and how would you notice?**

The first two healthy cloud workers run, and the third is dropped. Every
time, `select_workers` logs a warning naming the dropped worker and the cap.
That warning is the only sign: The answer still arrives, built from fewer
drafts than the configuration suggests. Raise the cap or trim the pool so the
configuration says what actually runs.

**5. In a Private conversation, someone asks Audrey to search the web, just
as Ollama is restarting. What do they get?**

No web search, and no cloud fallback. Web search isn't one of Private's tools,
so it's never offered and a call to it is refused. When the local model call
fails because Ollama is restarting, an ordinary request might fall back to a
cloud model; a Private request instead reports that the local model is
unavailable. They can try again once Ollama is back.


## When you're ready for the next lesson

[Lesson 8 — Classification and routing](lesson-08-classification-and-routing.md)
moves one step earlier in the request. These two lessons took the task type
as given; the next one shows how Audrey decides whether a prompt is `code`,
`reasoning`, `general` or `vl`, how that combines with the choice the person
made, and when a turn goes fast or deep.
