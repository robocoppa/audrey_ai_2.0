# Lessons CONTINUITY

_Last updated: 2026-05-03 (Lesson 1 written. Course structure created at
`docs/lessons/`. Build campaign moved to `docs/campaign-1/`; CONTINUITY.md
renamed to HISTORY.md and lives there now. Lessons go up to GitHub;
HISTORY stays gitignored/laptop-local.)_

This file tracks the **current state of the lesson plan**. Sister to
`docs/campaign-1/HISTORY.md` (which tracks the build campaign — now
historical, all 31 phases verified).

A new session should read both files and the relevant lesson before
continuing.

---

## Course goal

Teach Bart, the codebase author, the entire Audrey codebase end-to-end
so he can maintain and extend it solo. Bart's Python is "basic" —
async/FastAPI/LangGraph need to be explained as they appear, not
assumed. Concrete-first; concepts arrive when they're needed.

12 lessons total. Each ~30-60 min. Audit-as-we-go: anything that smells
wrong gets flagged, but no code changes without explicit approval.

## Course state

**Lessons complete:**
- Lesson 1 — The request lifecycle, end-to-end ✅ written 2026-05-03

**In progress:**
- *(none — waiting for Bart to read Lesson 1 and signal ready for Lesson 2)*

**Queued (in order):**
2. Configuration + startup (`main.py`, `config.py`, `compose.yaml`)
3. The model layer (`models/`)
4. Classify + complexity (`pipeline/classify.py`, `pipeline/complexity.py`)
5. Memory recall + datetime injection (`pipeline/memory.py`, `pipeline/context.py`)
6. Fast path + ReAct (`pipeline/fast_path.py`, `pipeline/react.py`)
7. Deep panel + synthesis (`pipeline/deep_panel.py`, `pipeline/planner.py`, `pipeline/synthesize.py`)
8. Reflection + retry (`pipeline/reflect.py`, escalation in `graph.py`)
9. Fair scheduling + in-flight cap (`pipeline/fair_gate.py`, `routes/inflight.py`)
10. KB ingest + storage (`kb/`)
11. Tools dispatch + custom-tools server (`tools/`, `tools-server/`)
12. Routes + auth + metrics (`routes/openai.py`, `auth.py`, `metrics.py`)

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
- Each lesson has 4 sections: Context, Read-along, Audit notes,
  Comprehension questions.
- Code references use markdown links with line numbers when possible:
  `[main.py:48](src/audrey/main.py#L48)`.
- "Concept boxes" (a `> Concept:` blockquote) explain Python/framework
  concepts the first time they appear. Subsequent lessons reference back
  to them rather than re-explain.

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

Wait for Bart to finish Lesson 1 and signal ready for Lesson 2. When he
does:

1. Read this file to know where we are.
2. Read `lesson-01-request-lifecycle.md` to know what he's already
   covered (so Lesson 2 doesn't re-explain).
3. Write `lesson-02-configuration-and-startup.md` following the same
   format. Files in scope: `src/audrey/main.py`, `src/audrey/config.py`,
   `compose.yaml`, `monitoring/compose.yaml`. ~570 LOC total.
4. Update this file's "Lessons complete" + "Queued" sections.
5. If audit findings come up that Bart wants to act on, propose first;
   only edit code after explicit approval.
