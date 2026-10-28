# Lesson 8 — Classification and routing

**Estimated time:** 45-60 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"how does Audrey decide
whether a request is code, reasoning, general or vision work, and how does
that combine with the choice a person made to pick a fast or a deep answer?"*

Lessons 6 and 7 started after Audrey already knew the task type. This lesson
moves one step earlier and traces how a request becomes two values:

```text
task_type = code | reasoning | general | vl
mode      = fast | deep
```

`task_type` chooses the kind of model Audrey needs. `mode` chooses the shape of
the answer: one model, or the deep path the next lesson opens. Get these right
and the expensive machinery runs on the prompts that need it; get them wrong
and Audrey either spends minutes on a simple question or gives a hard one a
thin answer.


## 1. Context

### 1.1 Two questions, answered separately

Before the model layer can do anything, Audrey answers two questions:

```text
What kind of work is this?                       -> classification -> task_type
Should one model answer, or the deep path?       -> the mode gate  -> mode
```

They're related, but different. A short code question can be fast; a long
general one can be deep. Both values matter for every request, so both are
computed for every request.

The person's choice can override the mode:

| Choice | Mode |
| --- | --- |
| Fast (`audrey_fast`) | Always fast, and never escalated to deep. |
| Deep, Cloud or Private (`audrey_deep`, `audrey_cloud`, `audrey_local`) | Always deep, with a mixed, cloud-only or local-only pool. |
| Research (`audrey_research`) | Always deep, through the research branch. |
| Auto (`audrey_auto`) | Audrey decides, from an attached image, the prompt's length and explicit requests for depth. |

### 1.2 The route in one pass

```text
request
  -> datetime and memory recall add context
  -> classify picks task_type
  -> complexity measures the prompt and checks for depth cues
  -> image, choice and length rules pick mode
  -> fast path, or the planner and a deep branch
  -> a thin fast answer may escalate to deep, unless something says it's right
```

One principle runs through all of it: **routing judges the person's own
words, not what Audrey added.** The date, recalled memories, a skill's
instructions and the person's saved preferences are all added to the
conversation before the model sees it. None of them should make a request
look longer, more technical or more visual than the person made it. So
routing reads the person's messages, and, on the native app, a copy of the
conversation without Audrey's own additions (`routing_messages`).


## 2. Read-along

These are the files this lesson reads:

- [`pipeline/graph.py`](../../src/audrey/pipeline/graph.py): the classify
  and complexity nodes, and the escalation rule.
- [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py): keyword
  signals, the router model, and the fallbacks.
- [`pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py): token
  counting and depth cues.
- [`config.yaml`](../../config.yaml): the `router`, `complexity` and
  `agentic.escalation` settings.

### 2.1 What the graph starts with

A request enters the graph as a plain dictionary, the
[`PipelineState`](../../src/audrey/pipeline/state.py#L47) of Lesson 4. The
route fills in what it knows: the choice (`virtual_model`), the messages,
sampling options, the authenticated user, the tools this turn may use, the
skill chosen for the turn and, when a client asks for it, a structured output
format. Each node then adds what it decides: `task_type`,
`classify_reason` and `classify_confidence` from classification;
`prompt_tokens` and `mode` from the gate; the answer itself later.

The graph runs `datetime` and `memory_recall` before `classify` (Lesson 4
§2.3 shows the whole graph). Classification still centers the person's ask:
it reads the **last user message**, through
[`last_user_text`](../../src/audrey/pipeline/messages.py#L17), never the
system messages that came before it.

### 2.2 `node_classify` and the shared helper

The graph node, at
[`graph.py:312`](../../src/audrey/pipeline/graph.py#L312), does almost
nothing itself:

```python
async def node_classify(state: PipelineState) -> dict[str, Any]:
    ...
    routing_messages = state.get("routing_messages") or state["messages"]
    task, reason, conf = await classify_with_registry(
        ollama,
        user_text=last_user_text(routing_messages),
        messages=routing_messages,
        router_cfg=router_cfg,
        cfg=cfg,
        registry=_model_tools(state),
    )
    log.info("classify: %s (%s, conf=%.2f)", task, reason, conf)
    return {"task_type": task, "classify_reason": reason, "classify_confidence": conf}
```

The three keys matter later. `task_type` goes to the model layer.
`classify_reason` is a breadcrumb for the logs (`keyword:code_strong`,
`router:general`, `fallback:general`). `classify_confidence` feeds the
escalation rule in §2.7.

The work happens in
[`classify_with_registry`](../../src/audrey/pipeline/classify.py#L335), which
both drivers call, so the graph and the streaming driver classify the same
way. It reads the `router` settings, passes along the names of the tools this
turn may use (§2.3 shows why), and calls `classify`. Then it applies one last
rule, at
[`classify.py:397`](../../src/audrey/pipeline/classify.py#L397):

```python
if task == "vl" and not conversation_has_image(messages):
```

A `vl` verdict with no image anywhere in the conversation is demoted to
`general`. The vision models can't call tools, so a vision model handed a
question about, say, a video file someone uploaded has nothing to look at and
no way to fetch the transcript. As `general`, the same question takes one tool
call. The check covers the whole conversation, not just the latest turn, so a
follow-up about an image attached earlier stays `vl`.

### 2.3 The cheap first pass: keyword signals

Asking a model costs time, and many prompts announce their kind plainly. So
[`keyword_classify`](../../src/audrey/pipeline/classify.py#L109) checks a few
regular expressions first, in a deliberate order:

```python
def keyword_classify(text: str, *, tool_names: set[str] | None = None) -> KeywordSignal | None:
    ...
    if tool_names:
        sig = _tool_mention_signal(text, tool_names)
        if sig is not None:
            return sig
    if _REVIEW_OVERRIDE.search(text):
        return KeywordSignal("reasoning", "strong", "review_override")
    if _VL_STRONG.search(text):
        return KeywordSignal("vl", "strong", "vl_strong")
    if _CODE_STRONG.search(text):
        return KeywordSignal("code", "strong", "code_strong")
    if _REASONING_STRONG.search(text):
        return KeywordSignal("reasoning", "strong", "reasoning_strong")
    if _CODE_WEAK.search(text):
        return KeywordSignal("code", "weak", "code_weak")
    return None
```

A **strong** signal decides the task outright. A **weak** one is kept in
reserve, for when nothing better is available. Two of the checks are worth
slowing down for.

**Naming a tool wins.** "Use kb_image_search to find this rock photo" contains
the word "photo", which would otherwise look like vision work. But the person
asked for a tool, the general path always has a model that can call one, and
the vision models can't call tools at all. So
[`_tool_mention_signal`](../../src/audrey/pipeline/classify.py#L91) returns
`general` when the prompt names a tool this turn may use. That's why the
helper passes the tool names in.

**Reviewing code is reasoning.** "Review this code for bugs" contains code,
but the job is analysis and judgment, not writing code, so
[`_REVIEW_OVERRIDE`](../../src/audrey/pipeline/classify.py#L77) sends it to
`reasoning` before the code checks can claim it.

### 2.4 Short prompts skip the router

Before asking the router model,
[`classify`](../../src/audrey/pipeline/classify.py#L265) takes one more
shortcut, at
[`classify.py:302`](../../src/audrey/pipeline/classify.py#L302):

```python
signal = keyword_classify(user_text, tool_names=tool_names)
if signal is not None and signal.strength == "strong":
    return signal.task, f"keyword:{signal.reason}", 0.95

if skip_llm_under_tokens > 0 and _short_prompt_tokens(user_text) <= skip_llm_under_tokens:
    if signal is not None:  # weak signal
        return signal.task, f"short_skip_keyword:{signal.reason}", 0.6
    return "general", "short_skip:general", 0.5
```

A prompt of only a few tokens with no strong keyword
([`skip_llm_under_tokens`](../../config.yaml#L69) sets "a few") goes straight
to `general`, or to its weak keyword's task. Short plain prompts are almost
always general questions, and on a busy machine the router call would be most
of the wait before the answer starts.

### 2.5 The router model handles the rest

Plain prose like *"Could you help me think through whether this backup plan
is sensible?"* matches no keyword, so
[`router_classify`](../../src/audrey/pipeline/classify.py#L159) asks a small
model. Its instructions,
[`CLASSIFIER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L50), ask for one
JSON object:

```text
{"task": "code|reasoning|general|vl", "confidence": 0.0}
```

The call is built to be cheap and predictable:

- **Only the start of the prompt.** The router sees at most the first 2,000
  characters ([`classify.py:204`](../../src/audrey/pipeline/classify.py#L204 '{"role": "user", "content": user_text[:2000]}')).
  Deciding "this is code" doesn't need the whole pasted file.
- **Temperature 0, the answer pinned to a schema, thinking off.** The label is
  the whole product, so the output is constrained to the JSON shape above,
  and a model that would reason first is told not to, because reasoning was
  measured to cost seconds per turn for a four-way choice.
- **A forgiving parser.**
  [`_parse_router_output`](../../src/audrey/pipeline/classify.py#L226) takes
  the first `{...}` it finds and clamps the confidence between 0 and 1.

The [`router`](../../config.yaml#L12) settings name the model, its timeout and
how many failed attempts to allow. The model is kept deliberately small. The
router runs on nearly every request and doesn't wait for the GPU gate, so a
large router model wouldn't queue behind a deep worker on the GPU: It would
push that worker's model out of memory.

When the router fails (a timeout, unparseable output, an unknown label), the
attempt counts as a strike, and `classify` tries again up to the configured
limit. The whole decision order:

```text
strong keyword              -> that task                          (confidence 0.95)
short prompt, no strong one -> weak keyword's task, or general    (0.6 or 0.5)
router answers              -> its task                           (its confidence)
router keeps failing        -> weak keyword's task, or general    (0.6 or 0.25)
```

Routing gets less clever when the router is down, but the request goes on.

### 2.6 Complexity: a separate gate

The second question, fast or deep, has nothing to do with the kind of work.
[`count_tokens`](../../src/audrey/pipeline/complexity.py#L70) counts the text
tokens in the conversation, using
[`_iter_text_parts`](../../src/audrey/pipeline/complexity.py#L43) so an
attached image counts as nothing rather than as megabytes of data, and
[`is_complex`](../../src/audrey/pipeline/complexity.py#L99) compares the count
with a threshold from [`complexity`](../../config.yaml#L1330) in config.
A long paste is the clearest sign that one quick answer is the wrong shape.

Length misses one case: a short prompt that asks for a lot ("compare the top
three backup tools, think hard"). So
[`has_deep_intent`](../../src/audrey/pipeline/complexity.py#L105) looks for
explicit depth cues ("think hard", "in depth", "step by step" and others listed
in config) in the **latest** user turn only, so a cue three turns ago doesn't
make every later follow-up deep.

### 2.7 The mode decision

[`node_complexity`](../../src/audrey/pipeline/graph.py#L330) gathers its
inputs first, at
[`graph.py:337`](../../src/audrey/pipeline/graph.py#L337):

```python
gate_messages = without_skill_instruction(
    routing_messages, state.get("skill_instruction")
)
complex_, n = is_complex(gate_messages, threshold=complexity_threshold)
deep_intent = has_deep_intent(routing_messages, deep_intent_phrases)
forced_deep = vm in ("audrey_deep", "audrey_cloud", "audrey_local", "audrey_research")
forced_fast = vm == "audrey_fast"
image_turn = has_image_part(routing_messages)
```

`without_skill_instruction` is the principle from §1.2 in code: The length
gate measures the person's words with the skill's instructions taken out.
Then the decision, at
[`graph.py:347-373`](../../src/audrey/pipeline/graph.py#L347-L373):

```python
if image_turn and not describe_first:
    mode = "fast"
    reason = "image"
    task_override = "vl"
elif forced_deep:
    mode = "deep"
    reason = "forced_by_virtual_model"
elif forced_fast:
    mode = "fast"
    reason = "forced_by_virtual_model"
elif complex_:
    mode = "deep"
    reason = f"tokens>={complexity_threshold}"
elif deep_intent:
    mode = "deep"
    reason = "deep_intent"
else:
    mode = "fast"
    reason = f"tokens<{complexity_threshold}"
```

Read it top to bottom; the first rule that applies wins.

1. **An attached image goes fast, as `vl`.** A vision model has to see it,
   and the deep path rebuilds prompts as text. The exception is a person who
   explicitly picked a deep choice while image description is on
   (`describe_first`): Then a vision model describes the image in text, and the
   chosen deep path reasons over the description (Lesson 7 §2.7).
2. **An explicit choice decides.** The deep choices are always deep, Fast is
   always fast.
3. **Auto decides by length, then by depth cues.**

The `reason` lands in the log line, so `complexity: 640 tokens -> deep
(tokens>=500)` tells you exactly which rule fired. LangGraph then follows the
mode: [`route_after_complexity`](../../src/audrey/pipeline/graph.py#L584)
returns `"fast"` or `"deep"`, and the graph's wiring maps those to the fast
path or the planner.

### 2.8 Fast answers can escalate, but not always

After the fast path answers, Audrey can still decide the answer came out too
thin and re-run the request through the deep path.
[`escalation_decision`](../../src/audrey/pipeline/graph.py#L102) makes that
call, and most of it is reasons *not* to, at
[`graph.py:120-140`](../../src/audrey/pipeline/graph.py#L120-L140):

```python
if not enabled or state.get("response_format") is not None:
    return "end"
if state.get("virtual_model") == "audrey_fast":
    return "end"
if state.get("escalated_from_fast"):
    return "end"  # already came from an escalation; don't loop
if int(state.get("tool_rounds", 0)) > 0:
    return "end"
if state.get("memory_hits"):
    return "end"
```

Each guard describes an answer that may be short because it's right. Fast
means fast, and never quietly becomes deep. A turn that already escalated
doesn't loop. An answer grounded in a tool result or in a recalled memory
holds a specific fact, and deep workers who don't know the person would wash
it out ("You're running a Threadripper" turning into "I can't see your
computer").

Only then come the two triggers: an answer shorter than
[`min_chars`](../../config.yaml#L871), or a classification confidence under
the ceiling. Two kinds of confidence don't count, because they're
placeholders rather than measurements: the one `general` gets when the router
never answered and no keyword matched, and the ones a prompt gets when it was
short enough to skip the router (§2.4). Neither says anything about how hard
the question is. A weak keyword the router fell back to still counts; it's a
real signal, if a weak one.

Length is judged more gently for the shortest requests. When the whole
request, earlier turns included, fits under the router-skip threshold, a short
answer is the expected one: "hi" doesn't need a panel. So it doesn't count as
too short, unless it's empty, came with an error, or was cut off at the length
limit. An escalated request goes through a small bridge node that marks it,
and then into the planner like any deep request.

Escalation happens inside the graph. A fast turn the streaming driver streams
word by word can't escalate, because the words are already on screen; a fast
turn whose model can call tools runs through the graph (Lesson 4 §2.4), so it
can.

### 2.9 The streaming driver makes the same decisions

[`_stream_via_pipeline`](../../src/audrey/routes/openai/pipeline.py#L276)
can't run `node_complexity`, because it wants a banner on screen before
classification finishes. So it makes the mode decision itself, from the
choice, an attached image, the prompt's length and depth cues, in the same
order, and only then classifies: A deep turn needs the task to pick its pool,
and a fast turn needs it to pick its model. Lesson 4 §2.4 showed that
decision; the rule to remember is that **the two drivers must agree**, which is
why both call the same helpers (`classify_with_registry`, `is_complex`,
`has_deep_intent`, `without_skill_instruction`).

### 2.10 One routing path

Someone picks Auto and sends:

```text
Review this Python function for bugs:

def total(xs):
    return xs[0] + xs[1]
```

1. `datetime` and `memory_recall` add their context.
2. `node_classify` reads the last user message. The review override matches
   before the code checks, so the task is `reasoning`, reason
   `keyword:review_override`, with no router call.
3. `node_complexity` counts the person's words: far under the threshold, no
   depth cue, no image, and Auto lets the gate decide, so the mode is `fast`.
4. The fast path asks the model layer for the best healthy `reasoning` model.
5. If the answer comes back thin, with no tool use and no recalled memory,
   it may escalate to deep.

The whole lesson in miniature:

```text
words in the prompt -> task type
choice + length + depth cues + image -> mode
task type + mode -> which models run
```


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. Is `task_type` the same thing as `mode`?**

No. `task_type` says what kind of work the request is: `code`, `reasoning`,
`general` or `vl`. `mode` says which answer path runs: `fast` or `deep`. A
request can be `code` and fast, or `code` and deep; the first picks the kind of
model, the second the shape of the answer.

**2. Someone writes "use kb_image_search to find a photo of my old car". Why
doesn't the word "photo" send this to a vision model?**

Because naming a tool wins over every other keyword. The helper passes the
names of the tools this turn may use into `keyword_classify`, and
`_tool_mention_signal` returns `general` when the prompt names one. The
general path has a model that can call tools; a vision model would have no
way to run the search.

**3. A person asks "what did they say about retirement in the interview.mp4
I uploaded?", and the router answers `vl`. What happens next?**

`classify_with_registry` demotes the verdict to `general`, because there's no
image anywhere in the conversation. A vision model given that question would
have nothing to look at and no tools to fetch the transcript; as `general`,
the fast path can call a tool to read the file's text and answer from it.

**4. The router model times out on every attempt. What task does the request
get, and can that make a fast answer escalate?**

`classify` counts each timeout as a strike and tries again up to the
configured limit, then falls back, and the request goes on. If a weak keyword
matched, the request gets that keyword's task and confidence, which sits under
the escalation ceiling, so a fast answer escalates unless a guard stops it
first: The keyword is a real signal, if a weak one. If nothing matched, it
gets `general` with a placeholder confidence that `escalation_decision`
ignores, since a router that never answered says nothing about the question.
A thin answer can still escalate on length alone.

**5. Someone picks Auto and asks a short question that recalls one of their
saved memories. The fast answer is a single sentence. Does it escalate?**

No. The answer is short, but it's grounded in a recalled memory, and the
memory guard stops escalation before the length check runs. Deep workers
don't know the person, so re-running the question through them would likely
replace the specific, correct fact with a generic answer.


## When you're ready for the next lesson

[Lesson 9 — Deep mode and research mode](lesson-09-deep-mode.md) opens the
deep path this lesson routes into: the planner, the panel of workers, the
synthesizer and reflect, and research mode, which checks claims before one
writer answers.
