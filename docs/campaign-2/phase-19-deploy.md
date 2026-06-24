# Campaign 2 Phase 19 — Split `routes/openai.py` into a package

Breaks the largest module in the codebase (1464 lines) into a package with
one responsibility per file. Pure structural refactor — the route behavior,
the SSE wire format, and the public import surface are all unchanged.

**Two commits: a mechanical move, then the lesson-cite re-anchor.** This phase
carries a real documentation cost (82 lesson cites point into this file), so
it's sequenced last in the optimization pass, after Phases 16–18 land on an
otherwise-stable tree.

## What it does

`routes/openai.py` mixes five responsibilities. This phase turns it into a
package `routes/openai/` (keeping the import path `audrey.routes.openai`
intact), split along the existing section seams:

| Module                | Holds                                                       |
|-----------------------|------------------------------------------------------------|
| `__init__.py`         | re-exports the public surface (see contract below)         |
| `schemas.py`          | `ChatMessage`, `ChatCompletionRequest`                     |
| `passthrough.py`      | `_is_passthrough`, `_resolve_passthrough_model`, `_handle_passthrough`, `_passthrough_stream_sse` |
| `responses.py`        | `_options_from_request`, `_to_openai_response`, `_ollama_to_openai_tool_calls`, SSE frame helpers |
| `streaming.py`        | `_stream_via_pipeline`, `_stream_deep_with_banners`, `_phase_*`, `_drain_q_*`, `_stream_openai` (the ~540-line subsystem) |
| `routes.py`           | the `@router` endpoints + `_run_graph_with_metrics` + `_generate_via_pipeline` |

`router = APIRouter(...)` lives in `routes.py`; `__init__.py` re-exports it.

## Why this exists

1464 lines in one route module is a genuine navigation and comprehension
cost. The streaming machinery (~540 lines) is a self-contained subsystem with
a documented ordering contract (`first_token` precedes every `delta`) that is
far easier to reason about in isolation. `PROJECT_STATE.md` had deferred this
split specifically because of the lesson-cite churn — this phase pays that
cost deliberately rather than letting the file keep growing.

## Sequencing (do NOT collapse into one commit)

1. **Mechanical move — commit 1, zero logic edits.** Move blocks verbatim
   into the new modules, fix imports, add the `__init__.py` re-exports. No
   renames, no reordering, no "while I'm here" cleanups — a pure move keeps
   the diff reviewable and makes the cite re-anchor a deterministic
   line-delta.
2. **Resolve the import graph.** Expected shape: `routes` imports from
   `streaming`/`responses`/`passthrough`/`schemas`; `streaming` imports from
   `responses` (frame helpers) + `schemas`; `passthrough` imports from
   `schemas`. If a cycle appears between `streaming` and `routes`, push the
   shared piece down a layer (into `responses.py` or a dedicated `_sse.py`).
3. **Verify the public surface** (below) before touching any docs.
4. **Re-anchor lesson cites — commit 2.** Run the cite checker (below) and
   apply the corrected paths/lines. Update Lesson 15's prose (its whole-system
   map references "the 1464-line file"). Update `AUDIT.md`'s cites.

## Public surface that MUST stay importable from `audrey.routes.openai`

Verified consumers (`grep "from audrey.routes.openai"`):

- `main.py` → `router` (via `app.include_router`)
- `tests/test_inline_image.py` → `ChatCompletionRequest`, `ChatMessage`
- `tests/test_passthrough_route.py` → passthrough names
- `tests/test_passthrough_dispatch.py` → passthrough names
- module `__all__` today → `["router", "VIRTUAL_MODELS"]`

`__init__.py` must re-export **all** of these so **no consumer import line
changes**. This is the contract that makes the test suite a green checkpoint
after commit 1: `test_passthrough_route.py`, `test_passthrough_dispatch.py`,
and `test_inline_image.py` must pass **untouched**. If one fails, the
re-export is incomplete — fix `__init__.py`, not the test.

## Behavior invariant

Pure structural refactor. Same routes, same status codes, same SSE frames,
same streaming ordering contract, same passthrough fork. The full suite passes
with no test edits — that is the equivalence proof.

## Lesson-cite blast radius (the documentation cost)

**82 line-cites** point into `routes/openai.py` across the maintainer course:

| File                                       | Cites |
|--------------------------------------------|-------|
| `lesson-15-openai-routes.md`               | 28    |
| `lesson-07-classification-and-routing.md`  | 9     |
| `lesson-13-memory-and-context-injection.md`| 8     |
| `lesson-04-request-lifecycle.md`           | 3     |
| `lesson-14-fair-scheduling.md`             | 2     |
| `lesson-02-foundations-libraries.md`       | 1     |
| `AUDIT.md`                                  | 31    |

Every cite's **path** changes (e.g. `routes/openai.py#L920` →
`routes/openai/streaming.py#L40`), not just its line — budget for a path
rewrite. Run, in commit 2:

```
DOCS_GLOB="docs/ai-course/lesson-*.md" \
  .venv/bin/python scripts/check-lesson-links.py
```

Apply the corrected lines for identifier-labelled cites (the checker prints
them); eyeball the bare `file.py:NN` cites. Then reframe Lesson 15's prose to
the package layout (a lesson edit — match the course style per `AGENTS.md`).

> Note: the lessons live in `docs/ai-course/`, and the cite checker defaults
> to `docs/lessons/`. Always pass the `DOCS_GLOB` override above, or it
> reports "no docs found."

## Risk / rollback

- **Import cycle** between `routes` and `streaming` → move shared SSE helpers
  to the lowest layer so both import *down*, never sideways.
- **Missed re-export** silently breaks a test import → the suite is the gate;
  don't proceed to commit 2 until green.
- **Rollback:** commit 1 (the move) is self-contained — revert it and the
  cites/prose are untouched, because the cite work is a separate commit.

## Deploy on Unraid

No config or custom-tools change. From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): full suite green with the three openai-importing test
files **unmodified**; ruff clean across `routes/openai/`. Cite checker reports
zero **confident** DRIFT on the six lesson files (the path rewrites applied);
remaining `DRIFT?` advisories are bare-cite shape flags, not new drift.

Live, on the box:

1. The five virtual models still list and answer:

   ```
   docker logs audrey-ai 2>&1 | grep -E "pipeline compiled|tools discovered"
   ```

2. A streaming **audrey_deep** request still shows
   `Planning → Dispatching panel → Synthesizing` banners then the answer — the
   ordering contract held across the move.
3. A passthrough request (`audrey_passthrough/<model>`) still streams,
   including tool_calls accumulation.

## What this unblocks

A navigable route layer: each subsystem (schemas, passthrough, streaming,
response formatting, the thin endpoint orchestration) is its own file, and the
streaming machinery is isolated for the next time it needs work. Closes Item 4
of the 2026-06-23 optimization review (`optimization-pass-plan.md`) and the
long-standing deferred `routes/openai.py` split in `PROJECT_STATE.md`.
