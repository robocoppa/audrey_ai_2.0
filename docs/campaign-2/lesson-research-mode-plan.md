# Plan — new course lesson: Research mode (`audrey_research`)

A plan for adding a maintainer-course lesson on Audrey's research mode — the
staged claim/source-ledger pipeline behind `audrey_research`. This is the **plan**,
not the lesson; per the lesson workflow, the actual write is gated on (a) user
go-ahead on this outline and (b) draining the in-scope audit findings first.

## Why this lesson exists

Research mode is the single most sophisticated thing in the codebase and the course
**does not mention it at all** (`grep` for `audrey_research` across
`docs/ai-course/` returns nothing). It composes nearly every subsystem the course
already taught — the deep panel, the ReAct/tool loop, KB + web search, streaming
banners, fail-soft degradation — into one pipeline, and it adds genuinely new ideas
the course has never covered: **structured model output** (Ollama `format=` +
Pydantic), an **internal claim/source ledger** the pipeline reasons over, and
**deterministic post-hoc shaping** of an LLM answer (the Sources list and the
hedging policy are computed by code, not asked of the model). It is the natural
capstone.

## Placement — new Lesson 17, NO renumbering

**Decision: append as `lesson-17-research-mode.md`, after L16 (custom-tools
sidecar). Do not renumber any existing lesson.**

Why this works without renumbering:
- By the end of L16 the reader has every prerequisite: deep panel (L8), ReAct +
  function calling (L9–L10), KB ingest/search (L11–L12), `OllamaClient` (L6),
  fair scheduling (L14), the OpenAI route + streaming (L15), and `web_search` in
  the sidecar (L16). Research mode *uses all of them*, so it can only come last.
- The course is a linear "next-pointer" chain with no index table, and L17 sits at
  the very end — so the only wiring is: change L16's closer from "that's it for the
  course" into a pointer to L17, and move the course-wrap-up prose into L17's
  footer.
- **Renumbering is the most disruptive operation in this course** (every "Lesson N"
  prose cross-reference + every `file:line` cite anchor would need re-sweeping, and
  the cite checker can't catch prose-level lesson-number drift — see the manual
  accuracy passes that fixed exactly this). Appending avoids all of it. The user
  authorized renumbering "if we have to" — we don't.

**L16 edit needed:** its final section is currently "## That's it for the course"
with a whole-course recap + "the remaining way to deepen this is maintenance." That
recap should move to L17's footer (and gain research mode as the final subsystem).
L16 gets a short "one more subsystem composes everything you've seen — research
mode" next-pointer instead. Sweep L16's intro/footer for the now-stale "last
subsystem" framing.

## Scope — what the lesson covers

The learner question to open with: *"When I pick `audrey_research` instead of
`audrey_deep`, the answer is slower but more careful and ends with a Sources list —
what is it doing differently, and how does it make itself more trustworthy?"*

In scope (the `run_research_pipeline_streaming` path in `pipeline/deep_panel.py`,
`pipeline/ledger.py`, the research role prompts in `prompts.py`, and the research
pool config in `config.yaml`):

1. **The staged pipeline shape** — research (parallel fan-out) → verify →
   fact-check → write, vs. deep mode's planner→panel→synth→reflect. Why staged:
   each stage has one job, and the answer is the *writer's* output, not a synth
   merge.
2. **Structured model output** — `OllamaClient.chat(format=…)` pins a JSON schema
   on a model's decoder; `inlined_schema()` flattens `$ref`s because cloud models
   choke on them. This is a NEW concept for the course (every prior model call
   returned free text).
3. **The ledger as internal scaffolding** — `Source`/`Claim`/`ResearchResult`/
   `ClaimCheck`/`FactCheckResult`. The controlling principle: the ledger is
   something the *models reason over and the pipeline computes from*, never
   user-facing citation bookkeeping. (Teach the principle, not the campaign
   history — see "hard rules" below.)
4. **Tolerant parsing** — why every ledger field has a `BeforeValidator` that
   coerces instead of rejects, and the rule it encodes: a structured-output model
   WILL emit off-spec values, and one bad field must not discard a whole worker's
   work. Anchor: `_to_str_or_empty`, `_norm_risk`, the optional+backfilled ids.
5. **Deterministic shaping of the answer** — two things the *code* does after the
   model writes, not the model:
   - the **Sources list** (`_render_sources_block`): surviving sources only,
     ranked by domain authority, deduped, capped — appended by the pipeline.
   - the **hedge policy** (`hedge_policy` pure function + `_render_dispositions_block`):
     each claim gets a state-plainly / attribute / hedge disposition from its
     source types + risk, and the writer applies it. The over-hedge suppression
     guard is the teachable nuance (an all-hedge instruction is just blanket
     caution).
6. **Fail-soft everywhere** — every stage degrades to the prior prose behaviour on
   any error; the mode never dead-ends. Tie back to the deep-mode "always answer
   something" posture from L8.

Out of scope (mention, don't dwell): the SearXNG fallback infra (that's a sidecar/
ops detail — L16 territory at most, and infra is out of course scope per the
project conventions); the eval harness; the deploy/operational saga.

## Proposed outline (Lesson 5 style — Context / Read-along / Comprehension)

Match the latest published lesson's shape exactly before drafting (re-read it).
Draft outline:

- **Context** — the learner question; research vs. deep at a glance (a small
  table); the one-line "what makes it more trustworthy: it builds a ledger of who
  said what, checks it, and lets code — not the model — decide the confidence and
  the sources."
- **Whole-system map** — the four-stage diagram before any code; where each stage's
  models come from (the research pool in config).
- **Read-along, one request through the stages:**
  1. Research fan-out — parallel workers, each a ReAct loop (link back to L9), each
     returning prose.
  2. Structuring — the prose→`ResearchResult` pass; concept spotlight on
     **structured output** (`format=`, the schema module, why `$ref` inlining).
  3. Merge — claims kept, sources URL-deduped, ids worker-prefixed.
  4. Fact-check — claims→verdicts→corrections; DROP/HEDGE/CORRECT.
  5. Write — the source-bound writer; then the two **deterministic** post-steps
     (Sources list ranking; hedge dispositions). Concept spotlight on "the pipeline
     shapes the answer after the model, not by asking the model."
- **Concept spotlights** (pause before relying on the terms): structured/constrained
  output; tolerant validation (`BeforeValidator`); pure-function policy
  (`hedge_policy`) as the testable core.
- **Comprehension questions** — operational/scenario: "a worker's structured JSON
  has one off-enum field — what happens to that worker's claims, and why isn't the
  whole worker dropped?"; "the answer is creative/ungrounded — why is there no
  Sources list and no hedging block?"; "you want to A/B the hedging behaviour
  without a rebuild — what's the lever?"; "research mode and deep mode both fan out
  to a panel — what's the difference in how the final answer is produced?"
- **Footer** — the moved whole-course recap (now ending on research mode as the
  capstone) + the "from here it's maintenance" close.

## Hard rules for the writer (course conventions — easy to trip on here)

- **No "Phase N" / campaign vocabulary.** Describe research mode by substance.
  This is the highest-risk rule for this lesson because all the source material
  (this plan, the deploy doc, PROJECT_STATE) is phase-tagged. Strip it.
- **No specific counts/sizes** — no "601 tests", "~640s", "8 sources cap as a
  number in prose" (the cap is fine to name as a behaviour; avoid baking eval
  timings/test counts). `file:line` cites are the documented exception.
- **No forward references by number; backward refs encouraged** — this lesson
  links back heavily (L6/L8/L9/L10/L11/L15). Use `see [Lesson 8](…)` form with the
  section anchor where possible.
- **Teach the principle, not the war story.** The over-hedge fix, the worker-drop
  saga, the SearXNG outage are deploy lessons (they belong in the deploy doc, which
  now has them). The *lesson* teaches WHY the code is shaped this way (tolerant
  validation, deterministic shaping, the all-hedge floor) — the design rationale,
  not the incident that surfaced it.
- **Explain new concepts on first use** — structured output and constrained
  decoding are new to the course; define before relying.

## Process (gated — do not skip)

Per the lesson workflow:
1. **Audit the in-scope files first** (`run_research_pipeline_streaming` +
   `ledger.py` + the research prompts + the research pool config), filing findings
   in `docs/ai-course/AUDIT.md` under a new heading.
2. **Drain those findings WITH THE USER** (fix-with-approval / accept / defer)
   before writing prose — the lesson should describe already-clean code.
3. **User approves this outline** before any drafting.
4. Write, matching the latest published lesson's style.
5. After source-adjacent edits (none expected — this is doc-only), run the cite +
   convention checkers; after the prose lands, run `check-lesson-conventions.py`
   on it. Add L17 to any course navigation, update L16's footer, update
   `PROJECT_STATE.md`'s lesson catalog.

## Open questions for the user

- **Split or single?** Research mode is large. It may want to be two lessons
  (L17 "the staged pipeline + structured output + the ledger" / L18 "deterministic
  shaping: Sources ranking + hedge policy"). If split, L18 appends after L17 —
  still no renumbering of existing lessons. Recommend deciding at outline-approval
  time based on draft length (the course targets ~1,400–1,800-word lessons).
- **Confirm placement** — append-as-capstone vs. anywhere earlier. (Strong
  recommendation: capstone; it depends on everything.)
