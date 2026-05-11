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
audrey_auto -> decide from prompt length
```

Here is the routing path in one pass:

```text
OpenAI request
  -> route builds initial PipelineState
  -> graph prepends datetime context
  -> graph recalls durable user memory
  -> classify chooses task_type
  -> complexity counts prompt tokens
  -> virtual model + complexity choose mode
  -> fast_path or planner -> deep_panel -> synthesize
  -> fast answer may escalate to deep unless guarded
```

Think of it like a train switchyard. Classification decides which cargo track
the request belongs on. Complexity and virtual model decide whether it takes
the short local spur or the longer panel route.


## 2. Read-along

These are the files we'll reference in this lesson:

- [`src/audrey/routes/openai.py:224`](../../src/audrey/routes/openai.py#L224)
  - where the non-streaming route builds the initial graph state.
- [`src/audrey/pipeline/state.py:27`](../../src/audrey/pipeline/state.py#L27)
  - the shared request state that graph nodes read and update.
- [`src/audrey/pipeline/graph.py:73`](../../src/audrey/pipeline/graph.py#L73)
  - the LangGraph builder and routing nodes.
- [`src/audrey/pipeline/classify.py:32`](../../src/audrey/pipeline/classify.py#L32)
  - keyword signals, router model, and fallback classification.
- [`src/audrey/pipeline/complexity.py:24`](../../src/audrey/pipeline/complexity.py#L24)
  - token counting for the fast/deep gate.
- [`config.yaml:7`](../../config.yaml#L7) - router-model config.
- [`config.yaml:244`](../../config.yaml#L244) - complexity threshold config.
- [`tests/test_classify.py:61`](../../tests/test_classify.py#L61) - tests that
  pin down one important classifier ordering rule.

Open them as we go, but do not try to memorize all the code at once. The useful
shape is the sequence of decisions.

### 2.1 The route gives the graph its starting state

The non-streaming route handler hands off to a helper that builds a
plain dictionary named `state`. Open
[`routes/openai.py:217`](../../src/audrey/routes/openai.py#L217),
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
[`routes/openai.py:224`](../../src/audrey/routes/openai.py#L224).
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

Open [`graph.py:391`](../../src/audrey/pipeline/graph.py#L391). The graph adds
nodes in one block:

```python
g.add_node("datetime", node_datetime)
g.add_node("memory_recall", node_memory_recall)
g.add_node("classify", node_classify)
g.add_node("complexity", node_complexity)
```

Then the edges at [`graph.py:403`](../../src/audrey/pipeline/graph.py#L403)
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
That helper is at [`graph.py:438`](../../src/audrey/pipeline/graph.py#L438):

```python
def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            ...
```

This is a good example of "context is available, but routing still centers the
user's actual ask."

### 2.3 `node_classify`: the graph asks for a task type

The graph node lives at [`graph.py:182`](../../src/audrey/pipeline/graph.py#L182):

```python
async def node_classify(state: PipelineState) -> dict[str, Any]:
    user_text = _last_user_text(state["messages"])
    tool_names = set(tools.names()) if tools is not None else set()
    task, reason, conf = await classify_fn(...)
    return {"task_type": task, "classify_reason": reason, "classify_confidence": conf}
```

The returned keys matter:

- `task_type` is the answer Audrey will later pass to the model layer.
- `classify_reason` is a human-readable breadcrumb for logs/debugging.
- `classify_confidence` is used later by the fast-path escalation guard.

Notice that the graph passes `tool_names` into the classifier at
[`graph.py:187`](../../src/audrey/pipeline/graph.py#L187). That is not just
metadata. If the user explicitly says "use `kb_search`" or "use
`kb_image_search`," Audrey should route toward tool-capable behavior instead
of accidentally treating the word "image" as a vision-only request.

### 2.4 The classifier has a cheap first pass

Now open [`classify.py:32`](../../src/audrey/pipeline/classify.py#L32). The top
of the file defines regex signals:

- code signals start at [`classify.py:32`](../../src/audrey/pipeline/classify.py#L32)
- reasoning signals start at [`classify.py:46`](../../src/audrey/pipeline/classify.py#L46)
- vision-language signals start at [`classify.py:52`](../../src/audrey/pipeline/classify.py#L52)

Regex means "pattern match against text." It is much cheaper than asking a
model. Audrey uses it for strong obvious cases.

The helper `keyword_classify(...)` starts at
[`classify.py:91`](../../src/audrey/pipeline/classify.py#L91). Its order is
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
[`classify.py:73`](../../src/audrey/pipeline/classify.py#L73). It returns
`general` when the user explicitly names a registered tool:

```python
return KeywordSignal("general", "strong", f"tool_mention:{name}")
```

Why `general`? Because explicit tool use should go through the ordinary
tool-capable answer path. A prompt like:

```text
use kb_image_search to find this rock image
```

contains the word "image," but the user's real instruction is "use this tool."
The tests at [`tests/test_classify.py:61`](../../tests/test_classify.py#L61)
pin that down.

Second, code review is reasoning. `_REVIEW_OVERRIDE` starts at
[`classify.py:59`](../../src/audrey/pipeline/classify.py#L59), and the check is
at [`classify.py:101`](../../src/audrey/pipeline/classify.py#L101). If the user
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
[`classify.py:120`](../../src/audrey/pipeline/classify.py#L120). It tells the
model to return only JSON:

```json
{"task": "code|reasoning|general|vl", "confidence": 0.0}
```

`router_classify(...)` starts at
[`classify.py:125`](../../src/audrey/pipeline/classify.py#L125). It sends only
the first part of the user text:

```python
{"role": "user", "content": user_text[:2000]}
```

That cap is at [`classify.py:141`](../../src/audrey/pipeline/classify.py#L141).
Routing should be cheap. The router does not need the whole long paste to
decide "this is code" or "this is reasoning."

The parser starts at [`classify.py:160`](../../src/audrey/pipeline/classify.py#L160).
It is forgiving: if the model wraps JSON in extra text, Audrey extracts the
first `{...}` block. It also clamps confidence into the `0.0` to `1.0` range at
[`classify.py:176`](../../src/audrey/pipeline/classify.py#L176).

The top-level `classify(...)` function starts at
[`classify.py:184`](../../src/audrey/pipeline/classify.py#L184). Its decision
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
[`classify.py:214`](../../src/audrey/pipeline/classify.py#L214) lets Audrey try
the router more than once before falling back.

### 2.6 Complexity is a separate gate

After classification, the graph runs
[`graph.py:200`](../../src/audrey/pipeline/graph.py#L200).

This node asks a different question:

```text
Is this prompt large enough that one fast answer is probably the wrong shape?
```

The token counter lives in
[`complexity.py:24`](../../src/audrey/pipeline/complexity.py#L24). It walks all
message content and counts text tokens. Multimodal messages get special care:
for list-shaped content, Audrey counts only text parts at
[`complexity.py:34`](../../src/audrey/pipeline/complexity.py#L34).

`is_complex(...)` is tiny:

```python
def is_complex(messages: list[dict], *, threshold: int) -> tuple[bool, int]:
    n = count_tokens(messages)
    return n >= threshold, n
```

That function starts at
[`complexity.py:41`](../../src/audrey/pipeline/complexity.py#L41). The
threshold comes from [`config.yaml:244`](../../config.yaml#L244):

```yaml
complexity:
  token_threshold: 500
```

Important distinction:

```text
classification decides what kind of model Audrey needs
complexity decides whether one answer pass is enough
```

They are deliberately separate because long and short prompts exist in every
task family.

### 2.7 Virtual models can force the route

Now read the middle of `node_complexity`, starting at
[`graph.py:202`](../../src/audrey/pipeline/graph.py#L202):

```python
vm = state.get("virtual_model")
forced_deep = vm in ("audrey_deep", "audrey_cloud", "audrey_local")
forced_fast = vm == "audrey_fast"
```

Then the mode decision is:

```python
if forced_deep:
    mode = "deep"
elif forced_fast:
    mode = "fast"
elif complex_:
    mode = "deep"
else:
    mode = "fast"
```

That code is at [`graph.py:205`](../../src/audrey/pipeline/graph.py#L205).

So the virtual model lineup means:

| Virtual model | Mode decision |
| --- | --- |
| `audrey_fast` | Always fast. |
| `audrey_deep` | Always deep, mixed local/cloud pool. |
| `audrey_cloud` | Always deep, cloud-only pool. |
| `audrey_local` | Always deep, local-only pool. |
| `audrey_auto` | Fast for short prompts, deep for large prompts. |

The graph returns `prompt_tokens`, `complex`, and `mode` at
[`graph.py:218`](../../src/audrey/pipeline/graph.py#L218). Later nodes do not
need to repeat the complexity calculation.

### 2.8 LangGraph chooses the next branch

The routing function after complexity is intentionally boring:

```python
def route_after_complexity(state: PipelineState) -> str:
    return "fast" if state.get("mode") == "fast" else "deep"
```

That is at [`graph.py:337`](../../src/audrey/pipeline/graph.py#L337).

The wiring at [`graph.py:406`](../../src/audrey/pipeline/graph.py#L406) tells
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
[`graph.py:340`](../../src/audrey/pipeline/graph.py#L340).

The first guard is simple: if escalation is disabled, stop.

The second guard is more important:

```python
if state.get("virtual_model") == "audrey_fast":
    return "end"
```

That is at [`graph.py:343`](../../src/audrey/pipeline/graph.py#L343).
`audrey_fast` means "do the fast thing." It should not secretly become deep
because the answer was short.

Two other guards stop escalation:

- tool-grounded fast answers at [`graph.py:351`](../../src/audrey/pipeline/graph.py#L351)
- memory-grounded fast answers at [`graph.py:355`](../../src/audrey/pipeline/graph.py#L355)

Those guards exist because a short answer grounded in tools or recalled memory
may be exactly right. Re-running it through deep workers can wash out the
specific fact Audrey already found.

Only after those guards does Audrey check answer length and classifier
confidence:

```python
too_short = len(content) < escalation_min_chars
low_confidence = conf < escalation_conf_ceiling and conf > 0
```

That starts at [`graph.py:364`](../../src/audrey/pipeline/graph.py#L364). If
either condition trips, the graph routes to `escalate_bridge`, then into the
deep branch at [`graph.py:414`](../../src/audrey/pipeline/graph.py#L414).

The mental model:

```text
audrey_auto fast answer
  -> if weak-looking and not tool/memory-grounded
  -> mark escalated
  -> run planner/deep_panel/synthesize
```

### 2.10 Streaming uses a separate driver

Non-streaming requests run the compiled graph through
[`_generate_via_pipeline` at routes/openai.py:217](../../src/audrey/routes/openai.py#L217).

Streaming has to interleave progress banners and token chunks, so it has a
separate route driver beginning at
[`_stream_via_pipeline` at routes/openai.py:291](../../src/audrey/routes/openai.py#L291).

The streaming route still performs the same major decisions:

- classify at [`routes/openai.py:329`](../../src/audrey/routes/openai.py#L329)
- count complexity at [`routes/openai.py:337`](../../src/audrey/routes/openai.py#L337)
- force deep/fast from the virtual model at [`routes/openai.py:338`](../../src/audrey/routes/openai.py#L338)
- choose deep banners or fast streaming at [`routes/openai.py:352`](../../src/audrey/routes/openai.py#L352)

But it is not literally the graph. It mirrors the same ideas so it can stream
the right user experience. We will revisit the streaming route in a later
lesson, because it has its own practical concerns: banners, cancellation,
partial output, and when a tool-using fast request has to finish before it can
emit a chunk.

For this lesson, the important point is:

```text
graph path and streaming path share the same routing concepts
but streaming has a route-level driver so it can control the event stream
```

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
7. `node_complexity` counts the prompt tokens.
8. Because the virtual model is `audrey_auto`, length decides fast vs deep.
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

`audrey_auto` lets Audrey choose fast or deep based on prompt length. It can
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

The next lesson can follow the most dynamic part of the request path: tool use.
We have now seen how Audrey decides a request's task and route. The next useful
question is what happens when a selected model can call tools: how ReAct loops
work, how Audrey dispatches custom-tools safely, and how tool results come back
into the model's conversation.
