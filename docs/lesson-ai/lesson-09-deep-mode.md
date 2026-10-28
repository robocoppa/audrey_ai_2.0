# Lesson 9 — Deep mode and research mode

**Estimated time:** 50-70 minutes if you read with the source files open.

**Goal:** by the end of this lesson, you can answer *"when a request takes the
deep path, what happens to it on the way to an answer, and what does research
mode do differently to earn its Sources list?"*

[Lesson 8](lesson-08-classification-and-routing.md) ended with two values
pinned to the request: a `task_type` (`code`, `reasoning`, `general` or `vl`)
and a `mode` (`fast` or `deep`). This lesson starts where `mode` is `deep`.
[Lesson 7](lesson-07-calling-models.md) showed the deep pieces from the model
layer's side: how a panel picks its workers, why a worker never raises, and
how synthesis falls back when a model fails. Here those pieces become a
pipeline, in one of two shapes:

```text
deep mode       planner -> deep_panel -> synthesize -> reflect
research mode   context -> research -> reflect
                           (research, verify, fact-check, write)
```


## 1. Context

### 1.1 Why a panel instead of one bigger model

The obvious way to answer a hard question better is a bigger model. Deep mode
takes another route. Several **workers**, each a different model, answer the
same request at the same time, and a **synthesizer** writes one answer from
their drafts. Two reasons:

1. **Different models miss different things.** Models from different vendors
   were trained differently, so their blind spots differ, and one draft often
   covers what another left out. The pools mix vendors on purpose.
2. **The waiting overlaps.** Cloud workers answer at the same time, so three
   drafts take about as long as the slowest of them. The synthesizer adds one
   more call.

The price is cost. A deep answer is several model calls where the fast path
makes one, which is why only the deep choices, and Auto on a prompt that needs
it, come this way.

### 1.2 Two deep shapes

Research mode is the second shape. Only the Research choice
(`audrey_research`) takes it; nothing routes there automatically. It's for
questions where the person wants to know where the facts came from.

| | Deep mode | Research mode |
|---|---|---|
| Who takes it | the deep choices, and Auto when a prompt needs depth | only the Research choice |
| What the panel produces | drafts of the answer | research notes with their sources |
| What checks the work | reflect, a length check | a verifier and a fact-checker, claim by claim |
| Who writes the answer | a synthesizer, merging the drafts | one writer, from the checked findings |
| Sources list | none | built by code from the claims that survived |

Both keep the promise from Lesson 7: **a failed model call becomes data, and
the person still gets an answer.** A worker that fails is a missing draft, and
a stage that fails is a skipped stage.

### 1.3 The deep path in one pass

1. **Planner** (optional). Skipped for a short prompt. Otherwise one call
   splits the request into two or three sub-questions, or decides it's one
   question.
2. **Panel.** Each worker gets a sub-question, or the whole request. A worker
   whose model can call tools runs the tool loop of the next lesson; the rest
   make one call.
3. **Synthesizer.** One call reads the request, the planned sub-questions and
   every draft, and writes the answer.
4. **Reflect.** A length check with no model call. A too-short answer sends
   the request back through the panel once.

Research mode skips sub-question planning and replaces the panel and synthesis
with its own four stages (§2.5).

Hold onto one fact before you read any logs: **reflect runs only on the graph
driver.** [Lesson 4](lesson-04-request-lifecycle.md#24-two-drivers-over-the-same-building-blocks)
showed that Audrey has two drivers. The streaming driver serves every native
run and every streamed `/v1` request. It runs the planner, the panel and the
synthesizer (or research's stages) with progress banners, and it has no
reflect step. A streamed answer reaches the screen as it's written, so nothing
is left to check before it ships. Reflection protects the answers you wait
for.


## 2. Read-along

These are the files this lesson reads:

- [`pipeline/graph.py`](../../src/audrey/pipeline/graph.py): the deep nodes
  and the routers between them.
- [`pipeline/planner.py`](../../src/audrey/pipeline/planner.py): the
  planner's call and its forgiving parser.
- [`pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py): how
  each worker is prepared, and in its second half, the research pipeline.
- [`pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py): what
  the synthesizer reads.
- [`pipeline/reflect.py`](../../src/audrey/pipeline/reflect.py): the length
  check.
- [`pipeline/ledger.py`](../../src/audrey/pipeline/ledger.py): research's
  records of claims and sources.
- [`pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py): each role's
  instructions.
- [`config.yaml`](../../config.yaml): the pools, the research pool, and the
  planning and reflection settings.

### 2.1 The planner: one question or several?

The planner asks whether a request is one question or several bundled
together. If several, each worker can take one and focus on it.

The graph node skips research, disabled planning and short requests, at
[`graph.py:436`](../../src/audrey/pipeline/graph.py#L436):

```python
async def node_planner(state: PipelineState) -> dict[str, Any]:
    if (state.get("virtual_model") == "audrey_research"
            or not planning_enabled or state.get("prompt_tokens", 0) < planning_min_tokens):
        return {"subtasks": []}
```

The threshold is [`min_prompt_tokens`](../../config.yaml#L859), 120 tokens in
the shipped config. Its comment explains why it isn't lower. A short prompt
split into pieces hands each worker a fragment that can lose its subject ("the
tradeoffs between the leading ones?", with no "Rust runtimes"), and the worker
spends its turn trying to work out what the question was about.

When the planner does run, it makes one call to the router's small model at
temperature 0. It's reading the request's structure, not writing anything
creative. Like any local model call, it waits its turn at the GPU gate, and the
wait counts against the planner's deadline: On a busy GPU, planning gives up
rather than push another person's model off the GPU. Its instructions,
[`PLANNER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L67), ask for JSON
(abridged here):

```text
{"subtasks": ["...", "...", "..."]}
Rules:
- 2 to 3 entries, each a complete question or instruction (≤ 200 chars).
- Sub-questions must be independent — no 'first do X then Y' chaining.
- Each sub-question must be SELF-CONTAINED: a worker will see ONLY that
  one sub-question, never the original request. Restate the concrete
  subject in every entry.
- If the request is already atomic (one clear ask), return {"subtasks": []}.
```

The self-contained rule exists because of what the panel does with a
sub-question: It *replaces* the person's question (§2.2). A sub-question that
says "the leading ones" means nothing to a worker that never saw the original.

The reply goes through a deliberately forgiving parser,
[`_parse_planner_output`](../../src/audrey/pipeline/planner.py#L103), which
looks for the outermost `{...}` and returns `[]` for anything it can't read,
and [`plan`](../../src/audrey/pipeline/planner.py#L36) treats a single
sub-question as none. So every way the planner can fail ends in the same
place. A network error, a missed deadline, a reply that isn't valid JSON, one
sub-question: Each becomes `[]`, and `[]` means "every worker answers the whole
request". The planner sharpens the panel when it works, and the panel doesn't
notice when it doesn't.

### 2.2 The panel: giving each worker its question

Lesson 7 covered how the panel chooses its workers. Before any of them runs,
[`_prepare_panel`](../../src/audrey/pipeline/deep_panel.py#L549) builds each
worker's messages, at
[`deep_panel.py:623`](../../src/audrey/pipeline/deep_panel.py#L623):

```python
for i, (name, _loc) in enumerate(workers):
    subtask = subtasks[i % len(subtasks)] if subtasks else ""
    msgs = _messages_for_subtask(messages, subtask) if subtasks else messages
    if have_tools and name in capable:
        msgs = _with_worker_system(msgs, worker_role)
    per_worker_messages.append(msgs)
    per_worker_subtask.append(subtask)
```

Two jobs happen in this loop.

**Sub-questions are dealt out in turn.** Worker `i` gets sub-question
`i % len(subtasks)`, so with two sub-questions and three workers, the third
worker circles back to the first sub-question and the synthesizer gets two
takes on it. With three sub-questions and two workers, the third is never
asked. That's why [`max_subtasks`](../../config.yaml#L860) matches the
three-worker pools.
[`_messages_for_subtask`](../../src/audrey/pipeline/deep_panel.py#L499)
replaces the *last* user message, this turn's question, with the
sub-question. Earlier turns stay in place, so the worker still sees the
conversation. But it never sees the person's own wording for this turn, and
an instruction in it ("reply with a single code block") is gone unless the
planner carried it into the sub-question. Each draft records the sub-question
its worker was asked, so a strange draft can be traced to what was asked.

**Workers that can use tools get a role.** A worker whose model can call
tools, when tools are available, gets
[`DEEP_WORKER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L102) (abridged):

```text
You are one of several workers answering the same request in parallel.
Your output is a DRAFT — a later pass merges the drafts into the user's
final answer. Write the draft as a direct answer to the request.
- Do NOT withhold the whole answer because one part is unverified. ...
- Do NOT write about your own search process. ...
- Do NOT invent facts, sources, dates, or URLs to fill a gap. ...
```

Without it, a worker had no idea that it was one of several. Some workers
wrote a note about how thin their search results were instead of a draft,
which gives the synthesizer nothing to merge. The role goes in *after* the
conversation's own system messages, so a persona the person set still
decides the tone. A worker without tools gets no role, because it has no
evidence to reason about and the role would only cost tokens.

> **Concept spotlight — the GPU gate during a worker's tool loop.** Local
> models take turns on the one GPU through `FairLocalGate` (Lessons 6 and 7).
> A worker without tools holds the gate for its one generation. A worker that
> uses tools doesn't take the gate itself, as
> [`deep_panel.py:383`](../../src/audrey/pipeline/deep_panel.py#L383) shows:
>
> ```python
> async with (nullcontext() if use_tools else
>             gate.acquire(model, location=location, user_id=user_id)):
> ```
>
> The tool loop takes it around each model call instead
> ([`react.py:651`](../../src/audrey/pipeline/react.py#L651)) and gives it
> back while tools run. Some tools need the GPU too. The ones that look at a
> PDF page or a video frame ask a vision model, through the same gate. If the
> worker held the gate while such a tool waited for it, neither could move.
> The cost is that another job can use the GPU between two of a worker's
> rounds, and the worker's model may have to load again.

Two consequences show up in logs. Two local workers in one panel run one after
the other, while cloud workers run alongside them. And a worker that used
tools reports the token counts of its *last* model call only, so it looks
lighter than a one-shot worker that did less work.

### 2.3 The synthesizer: what it reads

The synthesizer gets the request and every draft in one message, laid out by
[`_format_drafts_for_synth`](../../src/audrey/pipeline/synthesize.py#L44):

```text
USER REQUEST:
<this turn's question>

PLANNED SUB-QUESTIONS:
  1. ...
  2. ...

DRAFTS:

--- draft 1 (model=..., elapsed=9.8s) ---
OBSERVED TOOL ACTIVITY (not claim verification):
Successful calls: web_search × 2
Failed calls: none recorded
Recorded public source identities: 4 (does not describe private file or other tool results).
<the worker's draft>

--- draft 2 (model=...) — empty (timeout)
```

Three details matter. The planned sub-questions are listed even if no worker
answered one, so the synthesizer can see a gap. A failed worker still gets a
line, saying it came back empty and why. And a worker that used tools carries
a factual record of what it called. The heading says plainly that this record
is not verification.

That heading matches the synthesizer's instructions,
[`SYNTH_SYSTEM`](../../src/audrey/pipeline/prompts.py#L129). Its central rule
is about evidence:

> FACTUAL ANCHORING: prefer claims tied to relevant retrieved passages or
> actual tool results over unsupported assertions. Tool-call counts,
> successful status, source counts and confident prose do not establish
> claim support.

It adds two more checks against false confidence. Two drafts agreeing doesn't
make a claim true, because models share the same training blind spots. And a
well-phrased claim isn't more reliable for being well-phrased. The
synthesizer also has to acknowledge a planned sub-question that no draft
covers, and adds a `## Caveats` section only when the drafts disagree on a
fact or the evidence is incomplete.

The synthesizer also sees the conversation's system messages,
[`_build_synth_messages`](../../src/audrey/pipeline/synthesize.py#L118) at
[`synthesize.py:146`](../../src/audrey/pipeline/synthesize.py#L146):

```python
synth_system = prompt_from_config(cfg, "synthesizer", _SYNTH_SYSTEM)
system_msgs = [m for m in prior_messages if m.get("role") == "system"]
```

Those carry the current date, the person's recalled memories and any system
prompt they set. Without the date, a synthesizer reasons about "today" from
its training data, and the merged answer sounds months out of date. The
conversation's other turns aren't replayed, because the drafts already
answer them.

If the synthesizer fails, Lesson 7 showed what happens next: the fallback
synthesizer, then the longest draft.

### 2.4 Reflect: a cheap check, and one retry

[`reflect`](../../src/audrey/pipeline/reflect.py#L71) makes no model call. It
looks at the answer's length:

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

The floor is [`min_answer_chars`](../../config.yaml#L868), 40 characters in
the shipped config, about one short sentence. The escape hatch is
[`_BREVITY_CUES`](../../src/audrey/pipeline/reflect.py#L37), phrases such as
"in one sentence", "tldr" and "briefly". "What year is it? Answer in one
sentence" has a correct answer shorter than the floor, and retrying it would
waste a whole panel. The cues are English, like Audrey. A request for
structured output skips the check entirely.

`reflect` only judges. The decision to retry belongs to the graph's router,
[`route_after_reflect`](../../src/audrey/pipeline/graph.py#L596):

```python
def route_after_reflect(state: PipelineState) -> str:
    if state.get("reflect_passed"):
        return "end"
    if (state.get("reflect_reason") == "no_drafts"
            or state.get("synth_error") in {"no_writer", "write_failed", "write_truncated"}):
        ...
        return "end"
    if int(state.get("reflect_attempts", 0)) > reflection_max_retries:
        ...
        return "end"
    ...
    return "retry_research" if state.get("virtual_model") == "audrey_research" else "retry"
```

A pass ends the request. So does `no_drafts`, since every worker failed and a
retry would ask the same failing workers again. An unavailable or truncated
research writer also ends the run, rather than repeating all of its research.
Otherwise the request goes back once
([`max_retries`](../../config.yaml#L863) is 1) through the branch it came
from. The retry runs with the same inputs, so it's a second sample, and a
second answer is often long enough. If it isn't, Audrey ships what it has and
leaves `reflect_passed=False` in the state for whoever reads the logs.

### 2.5 Research mode: check the claims, then write

#### How a request gets here

After context preparation, the graph's
[`route_deep_kind`](../../src/audrey/pipeline/graph.py#L614) sends
`audrey_research` one way and every other deep choice the other.
[`node_research`](../../src/audrey/pipeline/graph.py#L513) runs the whole
staged pipeline and writes the answer itself, skipping `synthesize`. The
streaming driver runs the same pipeline under the banners Planning,
Researching, Verifying, Fact-checking and Writing. Both drivers share one
implementation,
[`run_research_pipeline_streaming`](../../src/audrey/pipeline/deep_panel.py#L1846),
an async generator that reports each stage as an event. The graph's version
just drains it and keeps the last event. Neither driver calls the sub-question
planner for research: each researcher works on the whole question. The Planning
banner covers context preparation, including optional memory recall.

The roles come from the pool,
[`deep_panel_research`](../../config.yaml#L521), one entry per task type. Its
shape, with roles in place of the models (which change often):

```text
researchers:         usually two cloud models from different vendors and a local one
verifier:            one model
factchecker:         a model that can call tools
fallback_factcheck:  another tool-capable model, never one of the researchers
writer:              one model
fallback_synth:      a model from a different vendor than the writer
```

The fallbacks follow one rule: **a fallback must not fail for the same reason
as the model it replaces.** A writer's fallback from the same vendor would go
down in the same outage, and a fact-checker's fallback that is also a
researcher could be put out of action by research itself (Stage 3 below).

The four stages:

| Stage | Job | Produces |
|---|---|---|
| Research | Researchers search and take notes, in parallel. | Notes with sources, then a ledger of claims and sources |
| Verify | One model audits the notes for overclaiming. | A critique the writer must apply |
| Fact-check | One model with tools confirms the risky claims. | Per-claim verdicts, as corrections |
| Write | One model writes the answer from all of the above. | The answer, then a Sources list built by code |

#### Stage 1: researchers take notes

[`select_researchers`](../../src/audrey/pipeline/deep_panel.py#L816) works
like the panel's worker selection: healthy models only, with its own cloud
cap, [`max_research_workers_cloud`](../../config.yaml#L626). A researcher omitted
by this cap is logged rather than silently dropped. Each researcher
then runs through `_run_one_worker`, the same never-raising worker as the
deep panel, with two differences. The researcher role goes first in the
messages, and the tool budget is bigger:
[`research_worker`](../../config.yaml#L788) allows more rounds and more
searches than a deep worker gets. With the deep worker's tighter budget,
researchers rushed their notes from truncated search snippets.

The role,
[`RESEARCHER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L201), asks for
the factual backbone, not the prose. It asks the researcher to prefer the
official page (release notes, the project's own site, the original paper)
over secondary write-ups, because for a date or a version one official page
settles what a handful of blog posts can't. It asks for notes that mark what
couldn't be verified, ending with a `SOURCES:` section that is always present,
even as `SOURCES: none`, because a section that may be left out gets left
out. The researchers' notes are merged into one findings block, and if no
researcher produced anything, the pipeline knows it has no grounding.

#### The ledger: notes become claims and sources

Prose is hard for the later stages to reason about precisely, so each
researcher's notes go through a second, mechanical call that turns them into a
**ledger**: a list of claims, each linked to the sources behind it. The ledger
is internal. The person never sees it; it exists so the system can tell where
each claim came from before the writer makes it read well.

[`_structure_one_draft`](../../src/audrey/pipeline/deep_panel.py#L1070) makes
that call with the researcher's own model, no tools and thinking switched off
(a model that thinks here can spend its whole budget and return no JSON). The
call passes a schema, at
[`deep_panel.py:1120`](../../src/audrey/pipeline/deep_panel.py#L1120):

```python
format=inlined_schema(ResearchResult),
```

The call adds no facts. It re-expresses the notes as records like
[`Claim`](../../src/audrey/pipeline/ledger.py#L144):

```python
class Claim(BaseModel):
    id: StrId = ""          # optional + backfilled — see Source.id
    text: str = ""
    source_ids: StrIdList = []
    risk: Annotated[Risk, BeforeValidator(_norm_risk)] = "medium"
    needs_hedge: bool = False
    hedge_reason: str | None = None
```

A [`Source`](../../src/audrey/pipeline/ledger.py#L129) has a title, a URL and
a `source_type`: `official`, `primary_paper`, `scholarly`, `reference`,
`news`, `company_claim` (a vendor's own assertion), `blog` or `unknown`.

Code, not the model, decides which sources exist. The call is handed a
numbered catalogue (`s1`, `s2`, ...) of the pages the researcher actually
retrieved, and
[`_reconcile_with_catalogue`](../../src/audrey/pipeline/deep_panel.py#L948)
drops any claim-to-source link that doesn't point into it. When the model
minted its own numbering, claims ended up attached to the wrong sources, and a
wrong link looks exactly like a right one. A missing link is at least honest.

The researchers' ledgers are then merged by
[`_merge_ledgers`](../../src/audrey/pipeline/deep_panel.py#L1211). Sources
with the same URL become one. Claims are *not* merged: If two researchers
disagree, both claims go forward, and the fact-check stage gets to see the
disagreement.

> **Concept spotlight — structured output, strict going out and tolerant
> coming in.** Every model call before this one returned free text. `format=`
> hands Ollama a JSON schema, and Ollama constrains the model's output, token
> by token, to text that fits it. Two adjustments make that work across
> models. [`inlined_schema`](../../src/audrey/pipeline/ledger.py#L292) copies
> every nested definition in place, because some models handle the
> cross-referenced form badly. And
> [`_REQUIRED_FOR_DECODE`](../../src/audrey/pipeline/ledger.py#L238) marks the
> fields a model must write, such as each claim's `source_ids`, so the model
> has to decide about linking every claim instead of quietly skipping the
> field.
>
> Reading the reply is the opposite. Every field has a default, and many have
> a validator that repairs instead of rejecting. Models still write
> `"url": null`, ids as numbers and risks outside the allowed values, so
> [`_to_str_or_empty`](../../src/audrey/pipeline/ledger.py#L44) turns a null
> URL into an empty string and
> [`_norm_risk`](../../src/audrey/pipeline/ledger.py#L116) maps an unknown
> risk to `medium`. Without the repairs, one bad field would discard a whole
> researcher's ledger. A blank URL is harmless here; whether a URL is usable
> is checked later, when deciding what to show the person.

#### Stage 2: verify

One call, made only when there are findings. The verifier,
[`VERIFIER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L307), doesn't write
the answer. It flags claims stated more strongly than the evidence supports: an
exact date presented as certain, "invented" or "the first to" without a
source that says so, a sweeping "every". Its critique goes to the writer, who
must apply it. Asking one prompt both to research and to doubt itself does
neither well, so the jobs are separate.

#### Stage 3: fact-check

The fact-checker is a tool-using loop with its own budget,
[`factcheck_worker`](../../config.yaml#L821). It searches to confirm the
high-risk claims (exact dates, versions, firsts), not to research anew. Its
model is chosen at
[`deep_panel.py:2048`](../../src/audrey/pipeline/deep_panel.py#L2048):

```python
fc_candidates = [
    m for m in (pool.get("factchecker"), pool.get("fallback_factcheck")) if m
]
...
factchecker = next(
    (m for m in fc_candidates if health.is_healthy(m) and m in capable), None
)
```

The fallback exists because of **role doubling**: one model holding two
roles. In some shipped pools the fact-checker is also a researcher, so a
single failed call during research once put that model into a health
cooldown, and the fact-check stage silently disappeared. The fallback works
only if research can't disqualify it too, so it must not be one of the pool's
researchers. A fallback that fires is logged as a warning, because the answer
looks the same either way.

With a ledger, the fact-checker's notes go through one more schema-pinned
call, in batches of
[`_FACTCHECK_STRUCTURE_BATCH`](../../src/audrey/pipeline/deep_panel.py#L1359)
claims (a single call over a long ledger tended to come back nearly empty).
The result is a verdict per claim, and
[`_factcheck_result_to_corrections`](../../src/audrey/pipeline/deep_panel.py#L1494)
turns the verdicts into lines the writer follows: `DROP` for an unsupported
claim, `CORRECT` with the corrected text, `UNVERIFIED` for a claim to hedge.
If that structuring fails, the writer gets the fact-checker's notes as they
are. If the stage itself fails, it contributes nothing and the pipeline moves
on.

#### Stage 4: write, and what code adds around it

Before the writer runs, code decides how confidently each surviving claim may
be stated. [`hedge_policy`](../../src/audrey/pipeline/ledger.py#L630) is a
plain function: no model call and no I/O.

```python
def hedge_policy(
    claim: Claim,
    source_types: set[SourceType],
    verdict: Verdict | None = None,
) -> HedgeDisposition:
    ...
    if "company_claim" in source_types:
        return "attribute_to_company"
    if claim.needs_hedge:
        return "hedge"
    if claim.risk == "high" and verdict != "supported":
        return "hedge_or_cite_strongly"
    if source_types & _AUTHORITATIVE_SOURCES:
        return "state_plainly"
    return "hedge"
```

The first rule that matches wins. A vendor's own claim is attributed to the
vendor ("Meta reports...") and never stated as fact. A claim flagged for
hedging is softened. A high-risk claim the fact-checker didn't confirm is
softened unless a strong source backs it. A claim backed by an authoritative
source is stated plainly, and anything else is hedged.
[`_render_dispositions_block`](../../src/audrey/pipeline/deep_panel.py#L1705)
lists only the claims that need special handling, under one line saying
everything else may be stated plainly. It drops the block entirely when every
claim would be hedged, because "hedge everything" is no guidance at all, and
it turns a confident explanation into mush.

The writer then gets one message, built by
[`_write_user_block`](../../src/audrey/pipeline/deep_panel.py#L1797):

```text
ORIGINAL REQUEST: ...
VERIFIED FINDINGS: ...   (or: [No grounding could be retrieved ...])
VERIFIER FLAGS (apply these): ...
FACT-CHECK CORRECTIONS (apply these): ...
CLAIM DISPOSITIONS (apply these): ...
Write the final answer for the user now.
```

Its instructions, [`WRITER_SYSTEM`](../../src/audrey/pipeline/prompts.py#L429),
set two hard rules: no facts beyond the findings, and every flag applied.
When there was little or no grounding, the writer opens with an honest
caveat, states plainly what it knows with confidence, and declines what it
can't confirm, instead of writing a wall of hedged maybes. The answer streams
to the person as it's written. If the writer fails before its first word, the
pool's `fallback_synth`, from another vendor, gets a turn; once words have gone
out, there's no fallback, and the answer is marked as cut short.

Last, code appends the Sources list, at
[`deep_panel.py:2245`](../../src/audrey/pipeline/deep_panel.py#L2245):

```python
if (
    response_format is None
    and write_error == ""
    and accumulated.strip()
):
    sources_block = _render_sources_block(ledger, fc_result)
```

[`_render_sources_block`](../../src/audrey/pipeline/deep_panel.py#L1630)
keeps the sources behind claims the fact-checker didn't drop, ranks them by
authority ([`_SOURCE_RANK`](../../src/audrey/pipeline/deep_panel.py#L1576):
official pages and encyclopedias above news, news above blogs), skips any
without a usable URL, and stops at
[`_MAX_SOURCES_RENDERED`](../../src/audrey/pipeline/deep_panel.py#L1568). An
answer with no ledger, a creative one for instance, gets no Sources heading
at all.

> **Concept spotlight — compute it, don't ask for it.** The writer could have
> been told "list your sources" and "hedge what's uncertain". Research mode
> doesn't, because asking a model to keep that bookkeeping makes its prose
> worse: It spends effort tracking citations and pads the list with weak
> sources to satisfy the instruction. Anything that can be decided from data
> (which sources survived, how confidently a claim may be stated) is decided
> by a plain function the writer never sees. A function can also be tested,
> which a prompt can't.

### 2.6 Reading the logs

A deep request on the graph driver, shortened:

```text
planner: 2 subtasks: ['What are the main tradeoffs ...', 'How does ...']
deep_panel: draft local-model:latest done_reason=stop ... elapsed=21.42s subtask='How does ...'
deep_panel: pool=deep_panel task=reasoning workers=3 ok=3 tool_grounded=2 attempted=[...]
synth: synth-model:cloud ok in 6.42s (attempt 1)
reflect: attempt=1 passed=True reason=ok
```

A research request adds its own lines:

```text
research: ledger built — 41 claims, 14 sources from 3/3 workers
research: factcheck ledger — 12 checks (2 drop [0 unlinked], 3 hedge), 0 fatal
research: Sources block appended (8 entries)
```

When something degrades, the log names what caught it: `skipping unhealthy
worker`, `falling back to registry`, `fact-check SKIPPED` with the reason,
`fact-check FELL BACK to`, `all synthesizers failed; returning longest draft`.
A streamed request logs the lines that come from shared code, but never a
`reflect:` line.


## 3. Comprehension questions

Try answering each question yourself before reading the answer.

**1. A `/v1` client sends "What's the capital of France? Answer briefly." to
`audrey_deep` and waits for the whole answer. What happens at each step, and
what would change if the client streamed?**

The deep choice forces deep mode however short the prompt. The prompt is far
below the planner's threshold, so the planner returns `[]` without a model
call, and every worker answers the whole question with something like
"Paris." The synthesizer merges the drafts into a short answer. Reflect finds
it under the length floor, but "briefly" is a brevity cue, so it passes as
`ok_brevity_requested` and the request ends. If the client streamed, the
same planner, panel and synthesizer would run, and nothing would check the
length: Reflect exists only on the graph driver.

**2. The planner returns three sub-questions, but only two workers in the
pool are healthy. How is the work divided, and what does the synthesizer
see?**

Worker 0 gets sub-question 0 and worker 1 gets sub-question 1; sub-question 2
is never asked. The synthesizer still sees all three under PLANNED
SUB-QUESTIONS, because the list shows what was planned, not what ran. Its
instructions tell it to acknowledge a planned sub-question no draft covers, so
the gap shows up in the answer instead of disappearing.

**3. Private's deep panel has fewer workers than Deep's, yet Private is
often the slower of the two on the same prompt. Why?**

Private keeps everything on local models, so its panel is the local-only
pool, while most of the mixed panel's workers are cloud models. The mixed
panel's workers all run at once, so the wait is roughly that of the slowest
worker, usually the local one. The local panel's workers all need the one
GPU, and the gate makes them take turns, so the wait is roughly their sum,
plus the time to load each model once the one before it is unloaded. Tools
don't change that: A local worker takes the gate for each model call, and
other local work runs in the gaps.

**4. In research mode, the fact-checker's model also works as a researcher.
During research it fails one call, and the health tracker starts a cooldown.
What happens to the fact-check stage?**

The stage picks the first candidate that is healthy and can call tools. The
primary is cooling down, so the `fallback_factcheck` model checks the claims,
and a warning records that the fallback fired. That's why the fallback must
not also be a researcher: The same failure could disqualify both. With no
usable candidate at all, the stage is skipped and logged with the reason, and
the writer works from the findings and the critique, with no corrections.

**5. Someone picks Research and asks for a birthday toast. Why does the
answer have no Sources list, and why doesn't research mode ask the writer to
list sources itself?**

The Sources list is built from the ledger. A toast has nothing to look up, so
the researchers come back without usable sources, and `_render_sources_block`
returns nothing. Asking the writer for citations would make the prose worse,
because the model would spend effort on bookkeeping and pad the list. So the
list is computed from data after the writer finishes, by a function that can
be tested.


## When you're ready for the next lesson

[Lesson 10 — Tool use and the ReAct loop](lesson-10-tool-use-and-react.md)
opens the box this lesson kept closed. Deep workers, researchers and the
fact-checker all hand their model to the same tool loop. The next lesson
shows how Audrey finds the tools that exist, runs a call, feeds the result
back, makes sure a tool only touches the caller's own data, and stops when
the budget runs out.
