# Lesson 17 — Research mode

When you pick `audrey_research` instead of `audrey_deep`, the answer comes
back slower, more careful, and ends with a short Sources list. This lesson is
about what it does differently to earn that — and it's the natural capstone,
because it composes nearly everything the course has already taught. The deep
panel ([Lesson 8](lesson-08-deep-mode.md)), the ReAct tool loop
([Lesson 9](lesson-09-tool-use-and-react.md)), KB and web search
([Lessons 11–12](lesson-11-kb-ingest-and-search.md)), fair scheduling
([Lesson 14](lesson-14-fair-scheduling.md)), the OpenAI route and streaming
([Lesson 15](lesson-15-openai-routes.md)), the tools sidecar
([Lesson 16](lesson-16-custom-tools-sidecar.md)) — research mode uses all of
them, which is why it can only come last.

It also adds three ideas the course has not covered: **structured model
output** (pinning a model's reply to a JSON schema), an **internal
claim/source ledger** the pipeline reasons over, and **deterministic shaping**
of the final answer — where *code*, not the model, decides the Sources list and
how confidently each claim is stated. By the end you should be able to explain
why those three exist, and reason about how the mode degrades when grounding is
thin.

## 1. Context

### 1.1 What research mode is, and the question it answers

Deep mode and research mode both fan a question out to a panel of models. The
difference is what happens to the panel's work. Deep mode runs
planner → panel → synth → reflect: several workers draft, and a synthesizer
*merges their drafts* into the answer. Research mode runs a different shape —
**research → verify → fact-check → write** — and the answer is one model's
prose (the *writer's*), built from findings the other stages have already
checked. The whole entry point is
[`run_research_pipeline_streaming`](../../src/audrey/pipeline/deep_panel.py#L1105);
its docstring lists the four stages and the events each one emits.

The learner question to hold onto: *what makes the answer more trustworthy?*
The one-line version is — **research mode builds a ledger of who said what,
checks it, and then lets code (not the model) decide the confidence and the
sources.** Everything below is that sentence in detail.

| Stage | Job | Output |
|---|---|---|
| **Research** | Fan out to a panel; each worker is a ReAct loop that searches and grounds claims. | Prose findings, plus a structured ledger. |
| **Verify** | Audit the merged findings for overclaiming. | A critique the writer must apply. |
| **Fact-check** | Confirm the high-risk claims against sources (a tool-using loop). | Per-claim verdicts → corrections. |
| **Write** | Turn verified findings into one answer. | The user's prose, then a code-built Sources list. |

> **Every stage degrades; the mode never dead-ends.** The pipeline
> [never raises](../../src/audrey/pipeline/deep_panel.py#L1148) — each stage
> catches its own failures and falls back to the prior prose behaviour, and the
> write stage runs *even with no findings at all*, so the user gets a flagged,
> honest answer rather than an error. This is the same "always answer
> something" posture as deep mode ([Lesson 8](lesson-08-deep-mode.md)), carried
> through four stages instead of one.

### 1.2 The files

| File | Role |
|---|---|
| [`deep_panel.py`](../../src/audrey/pipeline/deep_panel.py) | The pipeline itself — `run_research_pipeline_streaming` and all the stage helpers. (Shared with deep mode; research mode is the lower half.) |
| [`ledger.py`](../../src/audrey/pipeline/ledger.py) | The claim/source ledger: the Pydantic models, the tolerant parsers, and the `hedge_policy` function. |
| [`prompts.py`](../../src/audrey/pipeline/prompts.py) | The four role prompts — researcher, verifier, fact-checker, writer. |
| [`config.yaml`](../../config.yaml) | The research pool and the two feature flags that gate the ledger and the hedging. |

## 2. Read-along

We'll follow one request through the four stages, then look at the two things
the code does *after* the writer finishes.

### 2.1 Research fan-out

Stage 1 selects a panel of researchers and runs each as its own worker
([`deep_panel.py:1156`](../../src/audrey/pipeline/deep_panel.py#L1156)). Every
worker is a full ReAct loop — the same search-and-reason machinery from
[Lesson 9](lesson-09-tool-use-and-react.md) — and they run concurrently, collected
as each finishes. Each returns prose notes ending in a `SOURCES:` section.

The researcher prompt
([`prompts.py:118`](../../src/audrey/pipeline/prompts.py#L118)) is worth
reading closely, because it encodes a real lesson about grounding: it tells the
worker that when a search surfaces an authoritative or primary source — an
official project page, release notes, the original paper — to read *that*
directly rather than running more broad queries. For a date or a spec, one
official page beats a handful of secondary write-ups. A panel left to its own
devices tends to fan out across weak, SEO-flavoured results; this nudge points
it at the source that actually settles the question.

### 2.2 Structuring the findings into a ledger

The prose findings are the answer's raw material, but prose is hard for the
*next* stages to reason over precisely. So — when the ledger feature is on
([`config.yaml:231`](../../config.yaml#L231)) — a second, mechanical pass runs
per worker, converting each one's prose into a structured `ResearchResult`
([`deep_panel.py:1208`](../../src/audrey/pipeline/deep_panel.py#L1208)). This is
a *separate* model call with a separate prompt
([`prompts.py:148`](../../src/audrey/pipeline/prompts.py#L148)) that adds no new
facts — it only re-expresses what the researcher already found as claims and
sources.

The ledger types live in [`ledger.py`](../../src/audrey/pipeline/ledger.py): a
[`Claim`](../../src/audrey/pipeline/ledger.py#L134) carries its text, the
`source_ids` that back it, a `risk` rating, and a `needs_hedge` flag; a
[`Source`](../../src/audrey/pipeline/ledger.py#L119) carries a title, URL, and a
`source_type` (official, primary_paper, scholarly, reference, news,
company_claim, …). The controlling idea, stated at the top of the file: **the
ledger is internal scaffolding the models reason over — it is not user-facing.**
The user sees clean prose plus a short Sources list; the structure exists so the
system can prove to *itself* where each load-bearing claim came from before the
writer makes it read well.

> **Concept spotlight — structured (constrained) model output.** Every model
> call earlier in the course returned free text. This one is *pinned to a
> schema*: the call passes the `ResearchResult` JSON schema to the model's
> decoder, which constrains it to emit JSON matching that shape. There's a
> sharp edge — schema `$ref`s (the references Pydantic generates for nested
> models) aren't reliably handled across different models, so
> [`inlined_schema`](../../src/audrey/pipeline/ledger.py#L193) flattens every
> `$ref` into an inline copy first. Some models choke on the nested form and
> handle the flat one; inlining makes the call portable across the panel.

> **Concept spotlight — tolerant validation.** A model pinned to a schema will
> *still* emit off-spec values — a URL as `null`, an id as the integer `2`
> instead of `"2"`, a `risk` outside the enum. The naïve result is a
> `ValidationError` that discards the **whole** worker's ledger over one bad
> field — losing everything that worker found because one URL was null. The
> ledger refuses to pay that price: nearly every field has a `BeforeValidator`
> that *coerces* instead of rejecting — see
> [`_to_str_or_empty`](../../src/audrey/pipeline/ledger.py#L43) (null/int URL →
> usable value) and [`_norm_risk`](../../src/audrey/pipeline/ledger.py#L106)
> (off-enum risk → a sane default). The rule it encodes: **one malformed field
> must never throw away a whole worker's work.** A blank URL is harmless — we
> sanity-check URL shape later, when we decide what to show the user, not here.

The structured ledgers from all workers are then merged
([`deep_panel.py:1227`](../../src/audrey/pipeline/deep_panel.py#L1227)): sources
are deduplicated by URL, but claims are **not** deduplicated. That's deliberate
— if two workers disagree, the pipeline *wants* both claims to survive into the
fact-check stage, where the disagreement can be resolved against sources. Merging
away the conflict would hide exactly what the next stage exists to catch.

### 2.3 Verify

With a grounded ledger in hand, the verifier
([`deep_panel.py:1240`](../../src/audrey/pipeline/deep_panel.py#L1240)) audits
the merged findings for overclaiming. Its prompt
([`prompts.py:174`](../../src/audrey/pipeline/prompts.py#L174)) is narrow and
specific: flag claims stated more precisely than the evidence supports — an
exact date presented as certain when the sources hedge, a superlative
("the first to", "invented"), an over-specific attribution. It does **not**
write the answer; it produces a critique the writer is later required to apply.
Being a single, focused job is the point — a prompt asked to both research *and*
self-censor does neither well.

### 2.4 Fact-check

The fact-check stage
([`deep_panel.py:1262`](../../src/audrey/pipeline/deep_panel.py#L1262)) is itself
a tool-using ReAct loop: it `web_search`-confirms the high-risk and dated claims
and returns corrections the writer applies. It only runs when a fact-checker is
configured, healthy, and tool-capable — and when it can't run, the pipeline
[logs *which* precondition failed](../../src/audrey/pipeline/deep_panel.py#L1265)
and proceeds, exactly as the verify → write flow did before.

When a ledger exists, the prose corrections are structured back against it
([`deep_panel.py:1315`](../../src/audrey/pipeline/deep_panel.py#L1315)) into a
[`FactCheckResult`](../../src/audrey/pipeline/ledger.py#L167) — per-claim
verdicts like `supported`, `unsupported`, `needs_hedge`. Those verdicts are what
let the next steps drop an unsupported claim from the Sources list and soften a
shaky one. As everywhere, this is fail-soft: any problem keeps the plain prose
corrections and moves on.

### 2.5 Write — and the two things code does after it

The writer ([`deep_panel.py:1347`](../../src/audrey/pipeline/deep_panel.py#L1347))
turns the verified findings, the critique, and the corrections into one answer,
streamed live to the user. Its prompt
([`prompts.py:249`](../../src/audrey/pipeline/prompts.py#L249)) binds it to two
hard rules — introduce no new facts, and apply every flag the verifier raised —
and tells it how to behave when grounding was thin: open with an honest caveat,
then state what it knows plainly and decline on what it can't confirm, rather
than padding with vague maybes.

Now the part that makes research mode distinctive. Two pieces of the final
answer are **not asked of the writer at all** — they're computed by code from
the ledger, after the prose is written.

**The Sources list.** Once the writer finishes cleanly, the pipeline appends a
`## Sources` block built by
[`_render_sources_block`](../../src/audrey/pipeline/deep_panel.py#L933). It takes
the sources backing claims the fact-checker did *not* drop, ranks them by
authority ([`_source_rank`](../../src/audrey/pipeline/deep_panel.py#L897) — an
official page or encyclopedia outranks a blog), deduplicates by URL, and caps the
list. An ungrounded or creative answer produces **nothing** here — no ledger, no
surviving sourced claim, so no empty Sources header sprouts on a birthday toast.

**The hedging.** When the hedging feature is on
([`config.yaml:233`](../../config.yaml#L233)), each surviving claim gets a
*disposition* — state it plainly, attribute it to its source, or hedge it —
from the pure function
[`hedge_policy`](../../src/audrey/pipeline/ledger.py#L323). The rules are ordered
and small: a vendor's own claim is *attributed*, never endorsed; a claim flagged
`needs_hedge` is softened; a high-risk claim hedges unless a strong source
carries it; an authoritative, non-high-risk claim is stated plainly; everything
else hedges as the conservative default. The dispositions are rendered into a
short block of writer guidance by
[`_render_dispositions_block`](../../src/audrey/pipeline/deep_panel.py#L996).

> **Concept spotlight — the pipeline shapes the answer after the model, not by
> asking the model.** Why compute these instead of telling the writer "list your
> sources" and "hedge uncertain claims"? Because asking a model to track and
> emit citations inline *degrades its prose* — it spends effort on bookkeeping
> and pads with weak sources to satisfy the instruction. Anything deterministic
> — which sources survived, how to hedge a claim given its source types — belongs
> in a pure function the writer never has to think about. `hedge_policy` is a
> plain function over a claim and a set of source types: no I/O, no model, fully
> unit-testable. That testability is the payoff of pushing the decision out of
> the prompt and into code.

There's one nuance in the hedging block worth its own mention, because it's the
difference between helpful and useless. The block renders **only** the few
claims that need special handling, against a one-line "state everything else
plainly" backdrop — and
[`_render_dispositions_block`](../../src/audrey/pipeline/deep_panel.py#L996)
suppresses the block entirely if *every* surviving claim would be hedged. A
disposition list that says "hedge everything" carries no more signal than a
blanket "be careful," and blanket caution is exactly what turns a confident
explanation into timid mush. The block earns its place only when it's
*selective*: state the well-grounded facts plainly, flag the specific few the
evidence doesn't earn.

### 2.6 Why the prompts and the pipeline are shaped this way

Pull back from the individual stages and four design principles are visible —
the transferable part of this lesson, worth more than any one function.

**Compute what you can; don't ask the model to do bookkeeping.** The Sources
list and the hedge dispositions are *code* (§2.5), not prompt instructions,
because deterministic work pushed into a prompt both degrades the prose and
loses its testability. If you can decide it from data, decide it in a function.

**One job per role prompt.** Researcher grounds, verifier audits, fact-checker
confirms, writer writes (§2.1–2.5). The pipeline is staged precisely so each
prompt can be single-purpose — a prompt asked to do two things does both worse.

**Hedge the uncertain, not everything.** Selective dispositions against a
plain-by-default backdrop, with the all-hedge block suppressed (§2.5). An
instruction that applies to *everything* effectively applies to nothing.

**An empty result is not an error — and not proof that nothing exists.** A tool
call can succeed and still return nothing. Three states must stay distinct:
*failed* (the call errored), *empty* (it succeeded but returned no usable
content), and *grounded* (it returned content). Collapse "empty" into "success"
and a worker silently concludes "there are no sources" when the truth is "the
lookup came back thin this time" — and the answer degrades with nothing looking
wrong. It's the same tolerant-handling instinct as the ledger's
`BeforeValidator`s (§2.2) and the fail-soft stages (§1.1), applied to retrieval
instead of parsing.

A closing note on *how you'd know* whether a change to any of this helped. The
hermetic test suite proves the plumbing — that `hedge_policy` returns the right
disposition, that a null URL doesn't crash a worker — but it cannot tell you
whether an answer reads better, because it never calls a real model. That
judgement needs a live evaluation over saved answers, diffed against a prior
baseline. The discipline that follows: tune against that baseline, and when a
direction trends *down* across several edits, revert to the known-good version
rather than stacking more rules to patch the symptoms.

## 3. Comprehension questions

**1. A worker's structured JSON comes back with one field off-spec — a source
URL is `null`. What happens to that worker's claims, and why isn't the whole
worker dropped?**

Nothing is dropped. The `Source.url` field has a `BeforeValidator`
([`_to_str_or_empty`](../../src/audrey/pipeline/ledger.py#L43)) that coerces a
`null` (or an integer) URL into an empty string instead of letting Pydantic
raise. Without it, one null URL would `ValidationError` the entire
`ResearchResult` and discard everything that worker found — see the §2.2
tolerant-validation spotlight. A blank URL is harmless because URL shape is
checked later, when deciding what to show the user
([`_usable_url`](../../src/audrey/pipeline/deep_panel.py#L903)), not at parse
time. The rule: one malformed field must never throw away a whole worker's work.

**2. The answer is a creative, ungrounded one (say, "write me a toast"). Why is
there no Sources list and no hedging block?**

Because both are built from the ledger, and a creative answer has no grounded
ledger to build from.
[`_render_sources_block`](../../src/audrey/pipeline/deep_panel.py#L933) returns
`""` when there's no ledger or no surviving source with a usable URL, so no
`## Sources` header appears; the append step only runs on a clean answer at
[`deep_panel.py:1403`](../../src/audrey/pipeline/deep_panel.py#L1403). The hedging
block is likewise empty with no claims to disposition. This is the §2.5 point:
the deterministic shaping is *conditional on grounding*, so an ungrounded answer
stays clean prose rather than sprouting empty scaffolding.

**3. You want to A/B the hedging behaviour — see whether selective dispositions
actually read better — without rebuilding the image. What's the lever?**

The `hedge_policy` flag in the research-ledger config
([`config.yaml:233`](../../config.yaml#L233)). It's a *separate* flag from the
ledger's own `enabled` flag precisely so the hedging can be toggled against a
ledger-on baseline: flip it, re-run a live eval over saved answers, and diff the
two. It changes only the writer's wording (plain statement becomes explicitly
allowed), so it's the clean variable to test in isolation — which is the §2.6
"tune against a baseline" discipline made operational.

**4. Research mode and deep mode both fan out to a panel. What's the difference
in how the final answer is produced?**

Deep mode *merges* the panel's drafts: a synthesizer reads all the workers'
drafts and composes the answer from them. Research mode does **not** merge — the
panel produces checked *findings* (research → verify → fact-check), and a single
**writer** turns those findings into the answer
([`deep_panel.py:1347`](../../src/audrey/pipeline/deep_panel.py#L1347)). The
answer is one model's prose constrained by what the earlier stages verified, not
a blend of several drafts. That's why research mode can bind the writer to "apply
every verifier flag" and "introduce no new facts" — there's a single authoring
step to bind.

**5. The Sources list and the per-claim hedging could both have been asked of
the writer in its prompt. Why are they computed by code instead?**

Because asking a model to do that bookkeeping degrades the very prose you want
from it — it spends effort tracking citations and pads with weak sources to
satisfy the instruction, instead of writing well (the §2.5 spotlight). Pushing
the deterministic decisions into pure functions
([`_render_sources_block`](../../src/audrey/pipeline/deep_panel.py#L933),
[`hedge_policy`](../../src/audrey/pipeline/ledger.py#L323)) keeps the writer
focused on prose *and* makes those decisions unit-testable, which a prompt never
is. The principle: if you can compute it from data, compute it — don't ask the
model.

**6. A search call succeeds but comes back empty, and the answer quietly turns
vague. Why is treating that empty as a plain success a trap, and what are the
three states it must be told apart from?**

Because "succeeded" and "found something" are not the same thing. The three
states that must stay distinct are *failed* (the call errored), *empty* (it
succeeded but returned nothing usable), and *grounded* (it returned content) —
§2.6. If "empty" is folded into "success", a worker reads the empty result as
"there are no sources" and grounds nothing, while every status indicator says
the call was fine — so the answer degrades with nothing looking wrong. An empty
that's actually a transient thin lookup deserves a careful retry before it's
believed, and must never be cached as if it were the real answer. It's the same
tolerant-handling instinct as the ledger's coercing validators, applied to
retrieval.

## That's it for the course

Research mode is the last subsystem — and a fitting one to end on, because it
ties the whole course together. The path you've traced runs from the public
route ([Lesson 15](lesson-15-openai-routes.md)), through classification and
routing ([Lesson 7](lesson-07-classification-and-routing.md)), deep mode and the
ReAct/tool loop ([Lessons 8–9](lesson-08-deep-mode.md)), the model and KB layers
([Lessons 6, 11–12](lesson-06-the-model-layer.md)), per-user context
([Lesson 13](lesson-13-memory-and-context-injection.md)), fair scheduling
([Lesson 14](lesson-14-fair-scheduling.md)), and the tools sidecar on the far
side of the wire ([Lesson 16](lesson-16-custom-tools-sidecar.md)) — and here,
in research mode, nearly all of it composes into one pipeline that grounds,
checks, and shapes a trustworthy answer.

You have now seen every load-bearing file in Audrey and the sidecar that serves
it. The remaining way to deepen this knowledge isn't another lesson — it's
maintenance: when a bug surfaces or a feature is needed, you have the map to find
the right file and the *why* behind its shape.
