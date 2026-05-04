# Lessons CONTINUITY

_Last updated: 2026-05-04 (Lesson plan restructured to 13 lessons. Lesson 1
became a Python-tools foundations primer (async/FastAPI/Pydantic/LangGraph/
type hints/context managers/httpx/vector search/Prometheus/pytest). The
former Lesson 1 — request lifecycle end-to-end — became Lesson 2 with
Concept callouts removed (they live in Lesson 1 now) and back-references
added. README + CONTINUITY index updated to 13 lessons. Also: phase
references stripped from 32 source files in a separate sweep — code now
reads as present-tense without dated build-history references.)_

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

13 lessons total. Each ~30-90 min (Lesson 1 is the longest at ~90-120;
subsequent lessons are 30-60). Audit-as-we-go: anything that smells
wrong gets flagged, but no code changes without explicit approval.

## Course state

**Lessons complete:**
- Lesson 1 — Foundations: the tools you'll meet in this codebase
  ✅ written 2026-05-04
- Lesson 2 — The request lifecycle, end-to-end ✅ written 2026-05-03,
  adapted 2026-05-04 (Concept callouts moved to Lesson 1)

**In progress:**
- *(none — waiting for Bart to finish Lessons 1+2 and signal ready for Lesson 3)*

**Queued (in order):**
3. Configuration + startup (`main.py`, `config.py`, `compose.yaml`)
4. The model layer (`models/`)
5. Classify + complexity (`pipeline/classify.py`, `pipeline/complexity.py`)
6. Memory recall + datetime injection (`pipeline/memory.py`, `pipeline/context.py`)
7. Fast path + ReAct (`pipeline/fast_path.py`, `pipeline/react.py`)
8. Deep panel + synthesis (`pipeline/deep_panel.py`, `pipeline/planner.py`, `pipeline/synthesize.py`)
9. Reflection + retry (`pipeline/reflect.py`, escalation in `graph.py`)
10. Fair scheduling + in-flight cap (`pipeline/fair_gate.py`, `routes/inflight.py`)
11. KB ingest + storage (`kb/`)
12. Tools dispatch + custom-tools server (`tools/`, `tools-server/`)
13. Routes + auth + metrics (`routes/openai.py`, `auth.py`, `metrics.py`)

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

- Every lesson includes an **Audit notes** section.
- Severity tags: `nit` (cosmetic), `consider` (worth thinking about),
  `should-fix` (real problem, low urgency), `bug` (real problem, fix soon).
- Bart approves any change before code is touched. Default is "flag and
  discuss."
- Audit findings that the user accepts but defers go into followup memory
  notes (`/home/bart/.claude/projects/.../memory/`).

## Format conventions for lesson docs

- Lesson files: `lesson-NN-short-slug.md` (zero-padded number).
- Lesson 1 is the foundations primer (Python tools tour) — different
  shape from the others.
- Lessons 2+ have 4 sections: Context, Read-along, Audit notes,
  Comprehension questions.
- Code references use markdown links with line numbers when possible:
  `[main.py:48](src/audrey/main.py#L48)`.
- Foundations explanations live in Lesson 1. Later lessons that touch
  a new-to-the-reader concept link back to the relevant Lesson 1
  section (e.g. `see [Lesson 1 §6](lesson-01-foundations.md#6-typed-dictionaries-typeddict)`)
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

Wait for Bart to finish Lessons 1+2 and signal ready for Lesson 3. When
he does:

1. Read this file to know where we are.
2. Skim `lesson-01-foundations.md` and `lesson-02-request-lifecycle.md`
   to know what he's already covered (so Lesson 3 doesn't re-explain
   async, FastAPI, dependency injection, etc.).
3. Write `lesson-03-configuration-and-startup.md` following the
   Lesson 2 format (Context / Read-along / Audit notes / Comprehension
   questions). Files in scope: `src/audrey/main.py`,
   `src/audrey/config.py`, `compose.yaml`, `monitoring/compose.yaml`.
   ~570 LOC total. Topics that Lesson 1's foundations cover and
   Lesson 3 should cite back rather than re-explain: async context
   managers (lifespan), Pydantic Settings (config.py), TypedDict-vs-
   dataclass distinction.
4. Update this file's "Lessons complete" + "Queued" sections.
5. If audit findings come up that Bart wants to act on, propose first;
   only edit code after explicit approval.
