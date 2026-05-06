# Lessons CONTINUITY

_Last updated: 2026-05-05 (Audit posture changed end-of-day —
audit findings no longer live in the lesson docs themselves.
`docs/lessons/AUDIT.md` (new, gitignored) holds the structured
findings log + already-known issues + posture/severity/process
rules. CONTINUITY no longer carries the findings log; lesson
template for Lessons 4+ dropped the Audit-notes section (now 3
sections: Context / Read-along / Comprehension questions).
Lesson 4's three findings (`_options_from_request` near-duplicate,
`VIRTUAL_MODELS` validation in route, streaming cancellation) moved
to AUDIT.md verbatim. Earlier today: heavy iteration day on Lessons
0-3.
Lesson 0 (`lesson-00-introduction.md`) added as a public-facing
project intro. Lesson 1 was split into 1 (language features) and 2
(libraries); Lesson 2 was then split *again* into 2 (orchestration
stack: FastAPI/Pydantic/LangGraph) and 3 (satellite libraries:
httpx/Qdrant/Prometheus/pytest). Request-lifecycle is now Lesson 4.
Course total is 15 lessons. Every foundations section gained a "Why
Audrey needs this" subsection. Several pedagogy passes on
individual paragraphs based on cold-reader feedback: ASYNC240
hazard explained from first principles instead of citing the
warning code, fair-gate slot leak rewritten so "slot" is defined
before being assumed, contextmanager `yield`-with-value idiom
explained as "pass a handle to the body" instead of "bind a value
with `as`", closing examples reframed from quiz ("If you can read
this line:") to direct exposition ("As you read this line, know
that it parses as:"), pytest's `assert`-vs-unittest tradeoff
expanded so unittest jargon doesn't drop in cold. Three new memory
rules captured today (see Memory): no real emails in lessons (use
alice@example.com), no Phase-N references in public lessons
(describe by substance), no specific counts/sizes from KB or
codebase in lessons (use ballpark phrasing — file:line citations
are the documented exception). Pre-existing lesson docs scrubbed
to comply with all three. Earlier 2026-05-04: Audit posture
promoted to first-class; CONTINUITY now requires every lesson to
include audit notes (even if "no issues found") and tracks findings
in a structured log section with status — open / proposed /
resolved / accepted. Pre-seeded the log with 4 already-known items
inherited from HISTORY.md to revisit during their relevant lessons.
Earlier still: phase references stripped from 32 source files in a
separate sweep — code now reads as present-tense.)_

This file tracks the **current state of the lesson plan**. Sister to
`docs/campaign-1/HISTORY.md` (which tracks the build campaign — now
historical, all 31 phases verified).

A new session should read both files and the relevant lesson before
continuing.

---

## Course goal

Teach Bart, the codebase author, the entire Audrey codebase end-to-end
so he can maintain and extend it solo. Bart's Python is "basic" —
async/FastAPI/LangGraph need to be explained, not assumed. Lesson 1 is
the explicit foundations primer; subsequent lessons reference back to
its sections rather than re-explaining.

15 lessons total. Each ~30-90 min (Lessons 1, 2, and 3 are the
foundations and run ~45-90 each; subsequent lessons are 30-60).
Audit-as-we-go: anything that smells wrong gets flagged, but no code
changes without explicit approval.

## Course state

**Lessons complete:**
- Lesson 0 — Introduction (public-facing project orientation: what
  Audrey is, the pipeline shape, the operational layer, the phased
  build process + AI pair-programming culture, what the course is)
  ✅ written 2026-05-05
- Lesson 1 — Foundations I: Python language features (async, context
  managers, types, TypedDict) ✅ written 2026-05-04, split + augmented
  with Why-Audrey-needs sections 2026-05-05, pedagogy passes 2026-05-05
- Lesson 2 — Foundations II: the orchestration stack (FastAPI,
  Pydantic, LangGraph) ✅ written 2026-05-05 (split a second time on
  2026-05-05 — formerly held all 7 libraries; satellite libraries
  promoted to Lesson 3)
- Lesson 3 — Foundations III: the satellite libraries (httpx,
  Qdrant + embeddings, Prometheus, pytest) ✅ written 2026-05-05
  (split out of Lesson 2), pedagogy passes 2026-05-05
- Lesson 4 — The request lifecycle, end-to-end ✅ written 2026-05-03,
  adapted 2026-05-04 (Concept callouts moved to Lesson 1), renumbered
  to Lesson 4 on 2026-05-05

**In progress:**
- *(none — waiting for Bart to finish Lessons 1-4 and signal ready for Lesson 5)*

**Queued (in order):**
5. Configuration + startup (`main.py`, `config.py`, `compose.yaml`)
6. The model layer (`models/`)
7. Classify + complexity (`pipeline/classify.py`, `pipeline/complexity.py`)
8. Memory recall + datetime injection (`pipeline/memory.py`, `pipeline/context.py`)
9. Fast path + ReAct (`pipeline/fast_path.py`, `pipeline/react.py`)
10. Deep panel + synthesis (`pipeline/deep_panel.py`, `pipeline/planner.py`, `pipeline/synthesize.py`)
11. Reflection + retry (`pipeline/reflect.py`, escalation in `graph.py`)
12. Fair scheduling + in-flight cap (`pipeline/fair_gate.py`, `routes/inflight.py`)
13. KB ingest + storage (`kb/`)
14. Tools dispatch + custom-tools server (`tools/`, `tools-server/`)
15. Routes + auth + metrics (`routes/openai.py`, `auth.py`, `metrics.py`)

## Behavioral facts about teaching this learner

- **Python depth: basic.** Knows what `def` and classes are, can read
  control flow. Does NOT know async/await, FastAPI, LangGraph, or context
  managers in depth. Explain these as they appear, with concrete
  examples from the actual code (not toy examples).
- **Prefers seeing the why, not the what.** "Well-named identifiers
  tell you what the code does. The lesson should tell you why it's
  shaped this way."
- **Wants to maintain solo.** Lessons should give him enough context to
  fix bugs and extend features without external help. Don't hide
  complexity; if something is hard, slow down.
- **Goal is depth, not breadth.** Better one lesson where he understands
  every line than three lessons where he glosses each one.

## Audit posture

The course has a dual purpose: teach Bart, AND give the codebase a
fresh-eyes pass. Both halves are first-class — when writing a lesson,
do an audit pass on the files in scope.

**Audit findings do NOT go in the lesson doc.** They go in
`docs/lessons/AUDIT.md` (gitignored), which is the queue Bart drains
on his own schedule. Lesson docs ship to GitHub as teaching material;
audit findings are internal opinion-laden code commentary that has no
business in published lessons.

When you find something during a lesson's audit pass, file it in
AUDIT.md under the appropriate lesson heading with severity tag,
file:line citation, and a one-paragraph explanation. AUDIT.md
documents the severity tags, status values, and process; read it
before filing.

**No code changes without explicit approval**, ever — even for
obvious-looking nit fixes. AUDIT.md is the queue; Bart drains it.

## Format conventions for lesson docs

- Lesson files: `lesson-NN-short-slug.md` (zero-padded number).
- Lessons 1, 2, and 3 are the foundations primers (language features,
  orchestration stack, satellite libraries respectively) — different
  shape from the others. Each has section-by-section "Why Audrey
  needs this" subsections that anchor abstract concepts to concrete
  Audrey reality.
- Lessons 4+ have 3 sections: Context, Read-along, Comprehension
  questions. Audit findings raised during writing go in AUDIT.md
  (gitignored), not in the lesson itself.
- Code references use markdown links with line numbers when possible:
  `[main.py:48](src/audrey/main.py#L48)`.
- Foundations explanations live in Lessons 1, 2, and 3. Later lessons
  that touch a new-to-the-reader concept link back to the relevant
  foundations section (e.g. `see [Lesson 1 §4](lesson-01-foundations.md#4-typed-dictionaries-typeddict)`)
  rather than re-explaining.

## Path note (2026-05-03 reorg)

Build campaign was originally at:
- `CONTINUITY.md` (repo root) — now `docs/campaign-1/HISTORY.md`
- `docs/phase-N-deploy.md` — now `docs/campaign-1/phase-N-deploy.md`

Reorg done because Phases 1-31 are complete; the project transitioned to
"learn-the-codebase" mode, and the build docs are historical now.
`docs/lessons/` (this directory) is the active workspace.

`.gitignore` updated: HISTORY.md still gitignored (laptop-local), but
lessons go to GitHub.

## Next session should

Wait for Bart to finish Lessons 0-4 and signal ready for Lesson 5.
When he does:

1. **Read this file first** — know where we are. Then read
   `AUDIT.md` for the open code-review queue + already-known issues
   to revisit when you reach the relevant file.
2. **Skim Lessons 0, 1, 2, 3, and 4** (project intro, foundations-
   language, orchestration stack, satellite libraries, request-
   lifecycle) to know what's been covered. Lesson 5 should
   *cite back* to the foundations rather than re-explain async,
   FastAPI, dependency injection, context managers, Pydantic
   Settings, lifespan, etc.
3. **Audit pass first, then write.** Open every file in scope
   (`src/audrey/main.py`, `src/audrey/config.py`, `compose.yaml`,
   `monitoring/compose.yaml`) and read top-to-bottom looking for
   the things listed in `AUDIT.md`'s "What to look for" section.
   Note findings as you go.
4. **File audit findings in `AUDIT.md`, NOT in the lesson.** Each
   finding gets a bullet under the lesson's heading in AUDIT.md's
   "Open" section, with severity tag, file:line citation, and a
   one-paragraph explanation. The lesson itself stays focused on
   teaching.
5. **Write `lesson-05-configuration-and-startup.md`** following the
   Lesson 4 format: 3 sections — Context / Read-along /
   Comprehension questions. No Audit-notes section.
6. **Update this file's "Lessons complete" + "Queued" sections** and
   bump the `_Last updated:` line.
7. **Cite back, don't re-explain.** Topics the foundations lessons
   cover that Lesson 5 should reference rather than redefine: async
   context managers (lifespan — Lesson 1 §2), Pydantic Settings
   (config.py — Lesson 2 §2), TypedDict-vs-dataclass distinction
   (Lesson 1 §§3-4).
8. **Write for the cold reader.** Recurring pedagogy issue across
   the foundations lessons: a paragraph that sounded fine to the
   author landed as opaque jargon to a fresh reader. Before
   shipping, sanity-check every paragraph for: (a) terms used
   without definition (e.g. "ASYNC240", "fair-gate slot", "the
   gospel"), (b) cross-references to concepts that *haven't been
   built yet* in the reader's head, (c) sentences that test the
   reader instead of teaching them ("If you can read this line:" →
   "As you read this line, know that it parses as:"), (d) examples
   that assume familiarity with libraries the reader hasn't been
   introduced to (`self.assertEqual` without saying what unittest
   is). When in doubt, define the thing the first time it appears,
   then use it freely afterward.
9. **Apply the three baked-in lesson rules** (memory has full text):
   never use real emails (`bart@proton.me`, `robocoppa@proton.me`)
   in lesson content — use `alice@example.com`. Never reference
   "Phase N" — describe the bug/feature by substance. Never bake in
   specific counts or sizes from the codebase or KB ("110 tests",
   "~16k chunks", "~430 lines") — use ballpark phrasing. file:line
   citations are the documented exception.
10. **No code changes without explicit approval.** Findings go in
    AUDIT.md; Bart drains the queue on his schedule.
