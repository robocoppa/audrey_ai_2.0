# Lesson 7 - Classification and routing

**Estimated time:** 55-75 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer
*"how does Audrey decide whether a request is code, reasoning, general, or
vision work, and how does that combine with `audrey_auto`, `audrey_fast`, and
the deep virtual models?"*

Lesson 6 started after Audrey already knew the task type. This lesson moves
one step earlier. We will trace how a request becomes:

```text
task_type = code | reasoning | general | vl
mode = fast | deep
```

Those two values are different. `task_type` chooses the kind of model Audrey
needs. `mode` chooses the shape of the answer path.

The reason this routing layer matters: `mode = deep` is the gateway to
Audrey's most interesting machinery. When the gate flips to `deep`, the
request leaves the one-model-one-answer world and enters a four-stage
pipeline — planner, panel, synthesizer, reflect — where several worker
models run in parallel and a synthesizer merges their drafts into one
voice. The next lesson opens that pipeline up; this one is about what
decides which requests get sent through it. Get classification + routing
right and the deep machinery does its best work on the right prompts; get
it wrong and Audrey either burns wall-clock on simple requests or
flat-lines short on hard ones.


## 1. Context

Audrey has to answer two routing questions before the model layer can do its
job:

```text
What kind of work is this?
Should Audrey answer with one model, or with the deep panel?
```

The first question is **classification**. It produces a task type:

- `code`
- `reasoning`
- `general`
- `vl`

The second question is **routing mode**. It produces a path:

- `fast`
- `deep`

The two are related, but they are not the same thing.

A short code request can be fast:

```text
task_type = code
mode = fast
```

A long general request can be deep:

```text
task_type = general
mode = deep
```

And a user-selected virtual model can override the normal length-based choice:

```text
audrey_fast -> fast
audrey_deep / audrey_cloud / audrey_local -> deep
audrey_auto -> decide from images, utility prompts, length, and depth cues
```

Here is the routing path in one pass:

```text
OpenAI request
  -> route builds initial PipelineState
  -> graph prepends datetime context
  -> graph recalls durable user memory
  -> classify chooses task_type
  -> complexity counts prompt tokens and checks explicit depth cues
  -> image/utility/virtual-model/complexity rules choose mode
  -> fast_path or planner -> deep_panel -> synthesize
  -> fast answer may escalate to deep unless guarded
```

Think of it like a train switchyard. Classification decides which cargo track
the request belongs on. Complexity and virtual model decide whether it takes
the short local spur or the longer panel route.


## 2. Read-along

These are the files we'll reference in this lesson:

- [`src/audrey/routes/openai/pipeline.py:115`](../../src/audrey/routes/openai/pipeline.py#L115)
  - where the non-streaming route builds the initial graph state.
- [`src/audrey/pipeline/state.py:27`](../../src/audrey/pipeline/state.py#L27)
  - the shared request state that graph nodes read and update.
- [`src/audrey/pipeline/graph.py:215`](../../src/audrey/pipeline/graph.py#L215)
  - the LangGraph node functions and routing edges.
- [`src/audrey/pipeline/classify.py:33`](../../src/audrey/pipeline/classify.py#L33)
  - keyword signals, router model, and fallback classification.
- [`src/audrey/pipeline/complexity.py:72`](../../src/audrey/pipeline/complexity.py#L72)
  - token counting and depth-intent helpers for the fast/deep gate.
- [`config.yaml:7`](../../config.yaml#L7) - router-model config.
- [`config.yaml:450`](../../config.yaml#L450) - complexity threshold and depth-intent config.

Open them as we go, but do not try to memorize all the code at once. The useful
shape is the sequence of decisions.

### 2.1 The route gives the graph its starting state

The non-streaming route handler hands off to a helper that builds a
plain dictionary named `state`. Open
[`routes/openai/pipeline.py:112`](../../src/audrey/routes/openai/pipeline.py#L112),
which is the start of `_generate_via_pipeline`:

```python
state = {
    "virtual_model": payload.model,
    "messages": messages,
    "temperature": payload.temperature,
    "top_p": payload.top_p,
    "max_tokens": payload.max_tokens,
    "user_id": user_id,
}
```

That dictionary is built at
[`routes/openai/pipeline.py:114`](../../src/audrey/routes/openai/pipeline.py#L114).
This dictionary is the graph's starting memory for one request. The user asked
for a virtual model, sent messages, maybe set generation options, and was
authenticated as a particular user.

Now open [`state.py:27`](../../src/audrey/pipeline/state.py#L27). `PipelineState`
is a `TypedDict`. That means it documents the keys Audrey expects to pass
between graph nodes.

At the beginning of the request, only a few keys exist:

```text
virtual_model
messages
temperature 
top_p 
max_tokens
user_id
```

Later nodes add more keys:

```text
task_type
classify_reason
classify_confidence
prompt_tokens
mode
content
```

That is the basic LangGraph pattern:

```text
node reads state
  -> node returns a small dict
  -> LangGraph merges that dict into state
  -> next node sees the updated state
```

### 2.2 The graph order puts context before classification

Open [`graph.py:515`](../../src/audrey/pipeline/graph.py#L515). The graph adds
nodes in one block:

```python
g.add_node("datetime", node_datetime)
g.add_node("memory_recall", node_memory_recall)
g.add_node("classify", node_classify)
g.add_node("complexity", node_complexity)
```

Then the edges at [`graph.py:527`](../../src/audrey/pipeline/graph.py#L527)
make the order explicit:

```python
g.set_entry_point("datetime")
g.add_edge("datetime", "memory_recall")
g.add_edge("memory_recall", "classify")
g.add_edge("classify", "complexity")
```

So classification does not run on the raw user prompt alone. It runs after two
small context steps:

- `datetime` prepends the current server date/time.
- `memory_recall` may prepend relevant durable user memories.

The classifier still looks for the last user message, not the system messages.
That helper lives in
[`pipeline/messages.py:17`](../../src/audrey/pipeline/messages.py#L17):

```python
def last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        ...
```

This is a good example of "context is available, but routing still centers the
user's actual ask."

### 2.3 `node_classify`: the graph asks for a task type

The graph node lives at [`graph.py:219`](../../src/audrey/pipeline/graph.py#L219):

```python
async def node_classify(state: PipelineState) -> dict[str, Any]:
    task, reason, conf = await classify_with_registry(
        ollama,
        user_text=last_user_text(state["messages"]),
        router_cfg=router_cfg,
        cfg=cfg,
        registry=tools,
    )
    log.info("classify: %s (%s, conf=%.2f)", task, reason, conf)
    return {"task_type": task, "classify_reason": reason, "classify_confidence": conf}
```

The returned keys matter:

- `task_type` is the answer Audrey will later pass to the model layer.
- `classify_reason` is a human-readable breadcrumb for logs/debugging.
- `classify_confidence` is used later by the fast-path escalation guard.

The node does almost no work itself. It delegates to a shared helper,
`classify_with_registry`, at
[`classify.py:267`](../../src/audrey/pipeline/classify.py#L267). That
helper exists because the streaming route in `routes/openai.py` needs to
do the same thing — read the router config, extract `tool_names` from the
live tool registry, and call `classify(...)`. When that setup was inlined
in two places it silently diverged: the streaming path forgot to pass
`tool_names` and lost the explicit-tool-mention routing override. The
helper kills the duplication and pins both paths to the same call shape.

The interesting line inside the helper is at
[`classify.py:287`](../../src/audrey/pipeline/classify.py#L287):

```python
tool_names = set(registry.names()) if registry is not None else set()
```

Why this matters: if the user explicitly says "use `kb_search`" or "use
`kb_image_search`," Audrey should route toward tool-capable behavior
instead of accidentally treating the word "image" as a vision-only
request. Without `tool_names`, the keyword classifier's `_VL_STRONG`
regex matches first and the prompt gets sent to a tool-blind VL model.
The tool-mention signal lives inside `keyword_classify(...)` (next
section), but it only fires when the caller hands it the current
tool-name set.

### 2.4 The classifier has a cheap first pass

Now open [`classify.py:33`](../../src/audrey/pipeline/classify.py#L33). The top
of the file defines regex signals:

- code signals start at [`classify.py:33`](../../src/audrey/pipeline/classify.py#L33)
- reasoning signals start at [`classify.py:48`](../../src/audrey/pipeline/classify.py#L48)
- vision-language signals start at [`classify.py:54`](../../src/audrey/pipeline/classify.py#L54)

Regex means "pattern match against text." It is much cheaper than asking a
model. Audrey uses it for strong obvious cases.

The helper `keyword_classify(...)` starts at
[`classify.py:93`](../../src/audrey/pipeline/classify.py#L93). Its order is
important:

```python
if tool_names:
    sig = _tool_mention_signal(text, tool_names)
    if sig is not None:
        return sig

if _REVIEW_OVERRIDE.search(text):
    return KeywordSignal("reasoning", "strong", "review_override")

if _VL_STRONG.search(text):
    return KeywordSignal("vl", "strong", "vl_strong")
...
```

Two special cases are worth slowing down for.

First, tool mentions win. `_tool_mention_signal(...)` starts at
[`classify.py:75`](../../src/audrey/pipeline/classify.py#L75); when the
user explicitly names a registered tool it returns `general`
([`classify.py:89`](../../src/audrey/pipeline/classify.py#L89)):

```python
return KeywordSignal("general", "strong", f"tool_mention:{name}")
```

Why `general`? Because explicit tool use should go through the ordinary
tool-capable answer path. A prompt like:

```text
use kb_image_search to find this rock image
```

contains the word "image," but the user's real instruction is "use this tool."
Routing it as `general` keeps it on the tool-capable answer path rather
than letting the `_VL_STRONG` regex sweep it toward a vision-only model.

Second, code review is reasoning. `_REVIEW_OVERRIDE` starts at
[`classify.py:61`](../../src/audrey/pipeline/classify.py#L61), and the check is
at [`classify.py:103`](../../src/audrey/pipeline/classify.py#L103). If the user
says:

```text
review this code for bugs
```

Audrey treats that as `reasoning`, not `code`. Writing code and reviewing code
are different jobs. A review needs analysis, tradeoffs, and judgment.

### 2.5 The router model handles less obvious prompts

Regexes cannot classify everything. For plain prose like:

```text
Could you help me think through whether this backup plan is sensible?
```

the classifier asks a small router model.

The router prompt is `_ROUTER_SYSTEM` at
[`classify.py:122`](../../src/audrey/pipeline/classify.py#L122) — which is
just an alias (`_ROUTER_SYSTEM = CLASSIFIER_SYSTEM`); the actual prompt
text lives in
[`prompts.py:50`](../../src/audrey/pipeline/prompts.py#L50). It tells the
model to return only JSON:

```json
{"task": "code|reasoning|general|vl", "confidence": 0.0}
```

`router_classify(...)` starts at
[`classify.py:127`](../../src/audrey/pipeline/classify.py#L127). It sends only
the first part of the user text
([`classify.py:143`](../../src/audrey/pipeline/classify.py#L143)):

```python
{"role": "user", "content": user_text[:2000]}
```

That cap is at [`classify.py:143`](../../src/audrey/pipeline/classify.py#L143).
Routing should be cheap. The router does not need the whole long paste to
decide "this is code" or "this is reasoning."

The parser starts at [`classify.py:162`](../../src/audrey/pipeline/classify.py#L162).
It is forgiving: if the model wraps JSON in extra text, Audrey extracts the
first `{...}` block. It also clamps confidence into the `0.0` to `1.0` range at
[`classify.py:178`](../../src/audrey/pipeline/classify.py#L178).

The top-level `classify(...)` function starts at
[`classify.py:201`](../../src/audrey/pipeline/classify.py#L201). Its decision
order is:

```text
strong keyword signal -> use immediately
router model -> use if it returns a valid task
weak keyword signal -> fallback if router failed
general -> final fallback
```

Router settings come from [`config.yaml:7`](../../config.yaml#L7):

```yaml
router:
  model: "qwen3:4b"
  timeout_s: 20
  max_failures_before_fallback: 2
```

That last value matters. The loop at
[`classify.py:249`](../../src/audrey/pipeline/classify.py#L249) lets Audrey try
the router more than once before falling back.

### 2.6 Complexity is a separate gate

After classification, the graph runs
[`graph.py:214`](../../src/audrey/pipeline/graph.py#L214).

This node asks a different question:

```text
Is this prompt large enough that one fast answer is probably the wrong shape?
```

The token counter lives in
[`complexity.py:72`](../../src/audrey/pipeline/complexity.py#L72). It walks all
message content and counts text tokens. Multimodal messages get special care:
`_iter_text_parts` at
[`complexity.py:43`](../../src/audrey/pipeline/complexity.py#L43) yields text
parts and ignores image parts, so the gate counts words rather than bytes.

The actual complexity decision starts at
[`complexity.py:122`](../../src/audrey/pipeline/complexity.py#L122):

```python
def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    n = count_tokens(messages)
    return n >= threshold, n
```

That function starts at
[`complexity.py:122`](../../src/audrey/pipeline/complexity.py#L122). The
threshold and explicit depth cues come from
[`config.yaml:475`](../../config.yaml#L690):

```yaml
complexity:
  token_threshold: 500
  deep_intent_phrases:
    - deep dive
    - think hard
```

Important distinction:

```text
classification decides what kind of model Audrey needs
complexity decides whether one answer pass is enough, including short prompts that explicitly ask for depth
```

They are deliberately separate because long and short prompts exist in every
task family.

### 2.7 Virtual models can force the route

Now read the middle of `node_complexity`, starting at
[`graph.py:236`](../../src/audrey/pipeline/graph.py#L236):

```python
complex_, n = is_complex(...)
deep_intent = has_deep_intent(...)
vm = state.get("virtual_model")
forced_deep = vm in ("audrey_deep", "audrey_cloud", "audrey_local")
forced_fast = vm == "audrey_fast"
owui_task = is_owui_task_request(...)
image_turn = has_image_part(...)
```

Then the mode decision is:

```python
if image_turn:
    mode = "fast"
    task_override = "vl"
elif owui_task:
    mode = "fast"
elif forced_deep:
    mode = "deep"
elif forced_fast:
    mode = "fast"
elif complex_:
    mode = "deep"
elif deep_intent:
    mode = "deep"
else:
    mode = "fast"
```

That code starts at [`graph.py:223`](../../src/audrey/pipeline/graph.py#L223).

So the virtual model lineup means:

| Virtual model | Mode decision |
| --- | --- |
| `audrey_fast` | Always fast. |
| `audrey_deep` | Always deep, mixed local/cloud pool. |
| `audrey_cloud` | Always deep, cloud-only pool. |
| `audrey_local` | Always deep, local-only pool. |
| `audrey_auto` | Fast for ordinary short prompts, deep for large prompts or explicit depth cues. |

Two higher-priority cases sit above that table: attached images force fast mode
with the vision task, and OWUI background utility prompts force fast mode even
if the conversation is pinned to a deep virtual model.

The graph returns `prompt_tokens`, `complex`, and `mode` at
[`graph.py:259`](../../src/audrey/pipeline/graph.py#L259). Later nodes do not
need to repeat the complexity calculation.

### 2.8 LangGraph chooses the next branch

The routing function after complexity is intentionally boring:

```python
def route_after_complexity(state: PipelineState) -> str:
    return "fast" if state.get("mode") == "fast" else "deep"
```

That is at [`graph.py:384`](../../src/audrey/pipeline/graph.py#L384).

The wiring at [`graph.py:531`](../../src/audrey/pipeline/graph.py#L531) tells
LangGraph what those return strings mean:

```python
g.add_conditional_edges(
    "complexity", route_after_complexity,
    {"fast": "fast_path", "deep": "planner"},
)
```

So:

```text
mode=fast -> node_fast_path
mode=deep -> node_planner -> node_deep_panel -> node_synthesize
```

The graph does not call `if` statements spread all over the route handler.
It keeps the branch decision in one router function and the wiring in one graph
block.

### 2.9 Fast answers can escalate, but not always

After `fast_path` returns, Audrey may still decide the answer was not good
enough. The router for that is
[`graph.py:454`](../../src/audrey/pipeline/graph.py#L454).

The first guard is simple: if escalation is disabled, stop.

The second guard is more important:

```python
if state.get("virtual_model") == "audrey_fast":
    return "end"
```

That is at [`graph.py:390`](../../src/audrey/pipeline/graph.py#L390).
`audrey_fast` means "do the fast thing." It should not secretly become deep
because the answer was short.

Two other guards stop escalation:

- tool-grounded fast answers at [`graph.py:398`](../../src/audrey/pipeline/graph.py#L398)
- memory-grounded fast answers at [`graph.py:402`](../../src/audrey/pipeline/graph.py#L402)

Those guards exist because a short answer grounded in tools or recalled memory
may be exactly right. Re-running it through deep workers can wash out the
specific fact Audrey already found.

Only after those guards does Audrey check answer length and classifier
confidence:

```python
too_short = len(content) < escalation_min_chars
low_confidence = conf < escalation_conf_ceiling and conf > 0
```

That starts at [`graph.py:413`](../../src/audrey/pipeline/graph.py#L413). If
either condition trips, the graph routes to `escalate_bridge`, then into the
deep branch at [`graph.py:461`](../../src/audrey/pipeline/graph.py#L461).

The mental model:

```text
audrey_auto fast answer
  -> if weak-looking and not tool/memory-grounded
  -> mark escalated
  -> run planner/deep_panel/synthesize
```

### 2.10 Streaming uses a separate driver

Non-streaming requests run the compiled graph through
[`_generate_via_pipeline` at routes/openai/pipeline.py:108](../../src/audrey/routes/openai/pipeline.py#L108).

Streaming has to interleave progress banners and token chunks, so it has a
separate route driver beginning at
[`_stream_via_pipeline` at routes/openai/pipeline.py:205](../../src/audrey/routes/openai/pipeline.py#L205).

The streaming route still performs the same major decisions:

- count complexity and depth intent at [`routes/openai/pipeline.py:250`](../../src/audrey/routes/openai/pipeline.py#L250)
- force image, OWUI utility, or virtual-model decisions at [`routes/openai/pipeline.py:253`](../../src/audrey/routes/openai/pipeline.py#L253)
- choose deep banners or fast streaming at [`routes/openai/pipeline.py:240`](../../src/audrey/routes/openai/pipeline.py#L240)
- classify (to pick the concrete model) at [`routes/openai/pipeline.py:242`](../../src/audrey/routes/openai/pipeline.py#L242) for deep, [`pipeline.py:334`](../../src/audrey/routes/openai/pipeline.py#L334) for fast

But it is not literally the graph. It mirrors the same ideas so it can stream
the right user experience. We will revisit the streaming route in a later
lesson, because it has its own practical concerns: banners, cancellation,
partial output, and when a tool-using fast request has to finish before it can
emit a chunk.

For this lesson, the important point is:

#### Graph path and streaming path share the same routing concepts, but streaming has a route-level driver so it can control the event stream

### 2.11 One concrete routing path

Imagine the user sends this through `audrey_auto`:

```text
Review this Python function for bugs:

def total(xs):
    return xs[0] + xs[1]
```

Here is the path:

1. The route builds initial state with `virtual_model="audrey_auto"` and the
   OpenAI-shaped messages.
2. `datetime` prepends current time context.
3. `memory_recall` may prepend relevant durable memories.
4. `node_classify` extracts the last user text.
5. `keyword_classify(...)` sees the review override.
6. The task becomes `reasoning`, with reason `keyword:review_override`.
7. `node_complexity` counts the prompt tokens and checks for explicit depth cues.
8. Because the virtual model is `audrey_auto`, no image is attached, and this is not an OWUI utility task, token count plus depth intent decides fast vs deep.
9. If the prompt is under the threshold, `mode="fast"`.
10. LangGraph routes to `fast_path`.
11. The model layer receives `task="reasoning"` and picks a reasoning model.
12. If the fast answer is too short or low-confidence, Audrey may escalate to
    deep, unless one of the escalation guards applies.

That is the whole lesson in miniature:

```text
words in the prompt
  -> task type
  -> token count + virtual model
  -> fast/deep mode
  -> model-layer selection
```


## 3. Comprehension Q&A

Try answering each yourself before reading the answer.

**1. Is `task_type` the same thing as `mode`?**

No. `task_type` says what kind of work Audrey thinks the request is:
`code`, `reasoning`, `general`, or `vl`.

`mode` says which answer path Audrey will use: `fast` or `deep`.

A request can be `code` and `fast`, or `code` and `deep`. Same task family,
different route shape.

**2. Why does Audrey use keyword checks before asking the router model?**

Because some cases are obvious and cheap. A fenced code block, a tool name, or
"review this code" should not require a model call just to notice the pattern.

The keyword pass also protects special cases where the router or ordinary
regexes might make the wrong kind of confident decision.

**3. Why does an explicit tool mention classify as `general`?**

Because the user is asking for tool dispatch. If the prompt says
`kb_image_search`, the word "image" should not route straight to a vision model
that may not call tools. Classifying as `general` keeps the request on the
tool-capable answer path.

**4. Why is "review this code" classified as `reasoning` instead of `code`?**

Because the user is asking for judgment, not generation. A code review needs
analysis: what might break, what tradeoffs exist, what assumptions are risky.
That is why the review override wins before the code regexes.

**5. What does router confidence do?**

It is recorded in `classify_confidence`. Later, if Audrey used a fast path and
the answer looks weak, low classifier confidence can help trigger escalation to
deep.

It is not the only factor. Virtual model, answer length, tool rounds, and
memory hits also matter.

**6. What happens if the router model times out or returns bad JSON?**

`classify(...)` records a router strike and may try again, up to
`max_failures_before_fallback`. If the router keeps failing, Audrey falls back
to a weak keyword signal if one exists. If nothing exists, it defaults to
`general` with low confidence.

That is graceful degradation: routing gets less smart, but the request can
still continue.

**7. What does `audrey_auto` do that `audrey_fast` does not?**

`audrey_auto` lets Audrey choose fast or deep based on prompt length and explicit depth cues. It can
also escalate from fast to deep if the fast answer looks inadequate.

`audrey_fast` forces fast mode and suppresses escalation. It means "I really
want the fast path."

**8. Why does Audrey count tokens instead of characters for complexity?**

Models consume tokens, not characters. Token count is a better rough measure
of how much context the model has to process.

The count is still approximate because Ollama models may tokenize differently,
but it is good enough for a routing gate.

**9. Why does a tool-grounded fast answer skip escalation?**

Because tool use means the answer may be grounded in fresh or user-specific
data. A short answer can be correct if it came from a tool result.

Escalating that answer through deep workers can remove the specific evidence
and produce a more generic answer.

**10. Why does streaming have its own route driver?**

Because streaming is not just "run graph, then return JSON." Audrey has to send
SSE frames, progress banners, partial tokens, final stop frames, and controlled
error messages while work is still happening.

So the streaming route mirrors the routing concepts, but it has route-level
control over how the client sees progress.


## When you're ready for the next lesson

Lesson 8 picks up where this lesson leaves off. We have seen how Audrey
decides a request's task type and routes between fast and deep modes;
the next lesson opens what *happens* on the deep side. Planner, deep
panel workers, synthesizer, and reflect — the four-stage pipeline that
runs when the complexity gate sends a request down the deep path —
all live in [`lesson-08-deep-mode.md`](lesson-08-deep-mode.md).
