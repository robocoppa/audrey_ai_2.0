# Campaign 2 — Optimization pass

A focused plan for four items surfaced by a fresh-eyes codebase review
(2026-06-23). **All four are in scope, done in stages** (user decision
2026-06-23) — one phase each, with its own deploy doc:

| Item | Phase | Deploy doc            | Status                    |
|------|-------|-----------------------|---------------------------|
| 1    | 16    | `phase-16-deploy.md`  | ✅ deployed + verified (2026-06-24) |
| 2    | 17    | `phase-17-deploy.md`  | ✅ deployed + verified (2026-06-24) |
| 3    | 18    | `phase-18-deploy.md`  | ✅ deployed + verified (2026-06-24) |
| 4    | 19    | `phase-19-deploy.md`  | ✅ deployed + verified (2026-06-24) |
| —    | 20    | `phase-20-deploy.md`  | ✅ shipped (laptop); awaiting smoke test |
| —    | 21    | `phase-21-deploy.md`  | ✅ shipped (laptop); awaiting smoke test |

Phase 16 also absorbed the fast-path banner-latency fix (early `> _Thinking_`
emit + `router.skip_llm_under_tokens`), reported and verified after the plan
was first written.

**Phases 20–21 are follow-ons from live observation, not the original review:**
20 = `audrey_deep` → 1 local + 2 cloud workers (more draft diversity at ~0
wall-clock); 21 = fix the `audrey_local` worker timeout (reorder the local pool
to lead with `deepseek-r1:32b`, drop the timing-out `glm-4.7-flash:q8_0`, bump
`timeouts.deep_worker` 240 → 360 for cold-load headroom).

Ordered by value. Each item names the problem, the fix, the test coverage it
needs, and what "done" looks like. Items 1–3 are self-contained and shippable
independently. Item 4 (the `routes/openai.py` split) is the largest and
carries a lesson-cite cost — it is sequenced last and the cite re-anchoring is
part of the work, not an afterthought.

Read `AGENTS.md` "How to work" first (verify-before-claiming, edit-before-
write, no git writes). Run `.venv/bin/pytest tests/ -q` and
`.venv/bin/ruff check .` before reporting any item done. After any source
edit under `src/audrey/`, run `scripts/check-lesson-links.py <changed file>`
— the maintainer course (`docs/lesson-ai/`) cites these files by line.

---

## Item 1 — Fast-path model fallback (resilience) ⭐ highest value

**Problem.** `run_fast_path` picks the single highest-priority healthy model
(`pick_fast_model` → `registry.first_healthy`). If that one `ollama.chat`
fails, it records the failure and **re-raises** — it never tries the next
healthy candidate. The non-streaming caller `node_fast_path`
(`pipeline/graph.py:257`) has no try/except, so a transient blip on the top
model 502s the *current* request even though healthy fallbacks exist in the
registry. The cooldown means the *next* request routes around it — but the
user already saw the error.

Contrast: the deep panel tolerates per-worker failure and has a registry
fallback (`deep_panel.py:284-291`). The fast path — the most common path —
is the one that's single-shot.

**Files.**
- `src/audrey/pipeline/fast_path.py` — `run_fast_path`, `pick_fast_model`
- (no graph change needed — the fallback lives inside `run_fast_path`)

**Fix.** Turn the single-model attempt into a bounded loop over the
registry's healthy candidates for the task:

1. Replace the one-shot `pick_fast_model` + `ollama.chat` with iteration over
   `registry.candidates(task)`, skipping models where
   `health.is_healthy(name)` is False (mirrors `select_workers`).
2. Cap attempts at **2** (top model + one fallback) to match the deep
   panel's emergency-fallback cap and avoid burning the GPU gate / cloud
   quota on a long cascade. The cap is the right call: if the top two
   healthy models both fail mid-request, a third is unlikely to help and the
   latency cost is real.
3. On `OllamaError`: `health.record_failure(name, str(e))` (cools it down so
   the loop's next iteration and future requests skip it), log a warning,
   continue to the next candidate.
4. Only re-raise `OllamaError` after **all** candidates are exhausted — same
   final-failure contract `node_fast_path` already expects, so the 502 path
   is unchanged when nothing healthy can answer.
5. **Tools branch caveat.** The ReAct branch (`run_react`) is harder to
   retry safely — a failure can land after tool side effects (e.g.
   `memory_store`). Scope this item to the **non-tools branch only** for the
   first cut; the tools branch keeps its current single-shot + raise
   behavior. Document the asymmetry in the docstring. (Revisit retrying the
   tools branch only if production shows fast-react failures are a real
   source of 502s — likely a separate, smaller item.)

**Behavior invariant.** When the top model is healthy and succeeds (the
common case), behavior is byte-identical: one attempt, one model, same
`dispatch_total` increment, same return shape. The fallback only engages on
an actual `OllamaError`.

**Tests** (`tests/test_fast_path.py` or wherever fast-path tests live — grep
first):
- Top model healthy + succeeds → one attempt, returns its response (no
  behavior change). *Pin the no-regression case.*
- Top model raises `OllamaError`, second healthy → returns second model's
  response; assert `record_failure` fired for the first, `record_success`
  for the second.
- Top two both raise → `OllamaError` propagates (the 502 contract holds).
- Only one healthy candidate that fails → raises (no phantom third attempt).
- Tools branch unchanged: tool-capable model failing still raises on first
  attempt (no fallback), proving the asymmetry is intentional.

**Done when.** New tests pass, full suite green, ruff clean, docstring
updated to describe the bounded fallback + the tools-branch asymmetry.
Estimated ~15–25 lines of production code.

---

## Item 2 — De-duplicate `run_panel` / `run_panel_streaming` (maintainability)

**Problem.** `run_panel` (`deep_panel.py:244-331`) and
`run_panel_streaming` (`deep_panel.py:334-442`) share ~40 lines that are
byte-for-byte identical: worker selection, the registry-fallback block,
subtask assignment, the `dispatch_total` metric loop, and the `coros = [...]`
construction. The streaming one even comments *"Same registry-fallback shape
as run_panel; see comment there."* The two genuinely differ only in **how
they await** — `asyncio.gather` (collect all, return) vs
`asyncio.as_completed` (yield per completion). Today a fix to the fallback
cap or the dispatch metric has to be made in two places or it drifts.

**Files.**
- `src/audrey/pipeline/deep_panel.py`

**Fix.** Extract the shared prep into one helper:

```python
def _prepare_panel(
    cfg, registry, health, *, pool_key, task, messages, subtasks,
    options, timeout_s, max_workers_cloud, tools, tool_capable_models,
    react_*..., user_id, ollama, gate,
) -> tuple[list[tuple[str, str]], list]:
    """Select workers (+ registry fallback), assign subtasks, emit dispatch
    metrics, and build the coroutine list. Returns (workers, coros).
    Returns ([], []) when no workers are available."""
```

Then:
- `run_panel` calls `_prepare_panel`, and if `coros` is empty returns
  `([], [])`; else `drafts = await asyncio.gather(*coros)` and returns
  `(list(drafts), attempted)`.
- `run_panel_streaming` calls `_prepare_panel`, and if `coros` is empty
  yields the `final` sentinel and returns; else loops
  `asyncio.as_completed(coros)` yielding `worker_done` events, then the
  `final` event.

Keep the two public function signatures **exactly** as they are (same kwargs,
same defaults) — only the bodies change. `attempted = [name for name, _ in
workers]` can come back from the helper or be recomputed by each caller; pick
whichever keeps the diff smaller.

**Behavior invariant.** Pure refactor. No scheduling change — `gather` vs
`as_completed` still only differ in *reception* order, exactly as the current
docstrings promise. Worker execution order, gate semantics, metric
increments, and event shapes are all unchanged.

**Tests.** The existing deep-panel tests should pass unmodified — that's the
proof of equivalence. If coverage is thin on the registry-fallback path
(no healthy pool workers → top-2 registry candidates), add one test that
hits it through **both** entry points to lock the shared helper. Confirm the
streaming path still emits exactly one `final` event with `drafts=[],
attempted=[]` when zero workers are available.

**Done when.** Existing tests pass with no edits, ruff clean, the duplicated
block exists in exactly one place. Net LOC should drop ~30.

---

## Item 3 — `discover_all` sequential discovery (minor efficiency)

**Problem.** `discover_all` (`tools/discovery.py:219-237`) discovers servers
in a `for` loop with `await discover_one` inside — each server blocks the
next. Low impact today (single `custom-tools` server, startup-only), but
trivially parallelizable and future-proofs multi-server setups. **In scope
this pass.**

**Files.** `src/audrey/tools/discovery.py`

**Fix.** Replace the sequential loop with `asyncio.gather(*[discover_one(
client, url, ...) for url in server_urls])`, then fold the results into the
registry **in `server_urls` order** so the "later names win on collision"
contract is preserved (`gather` preserves input order in its result list, so
iterate the zipped `(url, tools)` pairs in order). Keep the per-server log
line and the final total log line. `discover_one` already returns `[]` on
error and never raises, so `gather` needs no `return_exceptions=True`.

**Behavior invariant.** Single-server case is byte-identical (one coro, same
result). Multi-server: collision-resolution order unchanged because we still
fold in `server_urls` order; only the *fetch* now overlaps.

**Tests.** Existing discovery tests should pass. Add a two-server collision
test asserting the later-listed server still wins after the reorder — that's
the only behavior the change could threaten.

**Done when.** New collision test passes, full suite green, ruff clean,
single-server behavior unchanged.

---

## Item 4 — Split `routes/openai.py` (1464 lines) ⚠️ gated on cite re-anchoring

**Problem.** `routes/openai.py` is the largest module in the codebase (1464
lines) and mixes five distinct responsibilities. `PROJECT_STATE.md` has
deferred this split specifically because it churns lesson cites — and that
cost is now measured: **82 line-cites point into this file** across the
maintainer course (28 in `lesson-15-openai-routes.md`, 23 across lessons
02/04/07/13/14, 31 in `AUDIT.md`). Every line that moves invalidates the
cites below it. So this is sequenced **last** and the cite re-anchoring is
part of the item, not an afterthought.

**Why split anyway.** 1464 lines in one route module is a genuine
navigation/comprehension cost, and the streaming machinery (~540 lines) is a
self-contained subsystem with a documented ordering contract that would be
far easier to reason about in isolation. The split also makes the eventual
Lesson 15 re-read cleaner (each part maps to a course section).

### 4a. Target structure

Current structure (verified by `grep`):

| Lines       | Block                                              |
|-------------|----------------------------------------------------|
| 103–152     | passthrough helpers (`_is_passthrough`, `_resolve_passthrough_model`) |
| 156–200     | schemas (`ChatMessage`, `ChatCompletionRequest`)   |
| 201–313     | route handlers (`/v1/models`, `/v1/chat/completions`) |
| 315–487     | passthrough handlers (`_handle_passthrough`, `_passthrough_stream_sse`) |
| 489–586     | non-streaming pipeline (`_run_graph_with_metrics`, `_generate_via_pipeline`) |
| 587–921     | streaming pipeline + response helpers (`_stream_via_pipeline`, `_options_from_request`, `_to_openai_response`, `_ollama_to_openai_tool_calls`) |
| 923–1453    | deep-stream banner machinery (`_stream_deep_with_banners`, `_phase_*`, `_drain_q_*`, `_stream_openai`, SSE frame helpers) |

Proposed package layout — turn `routes/openai.py` into a small package
`routes/openai/` (keeps the import path `audrey.routes.openai` intact):

```
routes/openai/
  __init__.py        # re-exports: router, VIRTUAL_MODELS, ChatMessage,
                     #   ChatCompletionRequest, and the passthrough names
                     #   the tests import. THE PUBLIC SURFACE — see 4c.
  schemas.py         # ChatMessage, ChatCompletionRequest
  passthrough.py     # _is_passthrough, _resolve_passthrough_model,
                     #   _handle_passthrough, _passthrough_stream_sse
  responses.py       # _options_from_request, _to_openai_response,
                     #   _ollama_to_openai_tool_calls (pure formatters)
  streaming.py       # _stream_via_pipeline, _stream_deep_with_banners,
                     #   _phase_thinking, _phase_dispatch, _drain_q_*,
                     #   _stream_openai, SSE frame helpers (_delta_frame,
                     #   _stop_frame, etc.) — the ~540-line subsystem
  routes.py          # the @router endpoints + _run_graph_with_metrics +
                     #   _generate_via_pipeline (the thin orchestration layer)
```

`router = APIRouter(...)` lives in `routes.py`; `__init__.py` re-exports it.

### 4b. Sequencing (do this AFTER items 1–2 land)

1. **Mechanical move only — one commit, zero logic edits.** Move blocks
   verbatim into the new modules. Fix imports. Do **not** rename, reorder, or
   "improve" anything in the same pass — a pure move keeps the diff
   reviewable and makes the cite re-anchoring a deterministic line-delta.
2. Resolve the import graph. Likely shape: `routes.py` imports from
   `streaming`, `responses`, `passthrough`, `schemas`; `streaming` imports
   from `responses` (frame helpers) and `schemas`; `passthrough` imports
   from `schemas`. Watch for cycles — if `streaming` and `routes` cross-
   reference, the shared piece moves down a layer (probably into
   `responses.py` or a new `_sse.py`).
3. `__init__.py` re-exports the public surface (4c). Verify
   `app.include_router(openai_router)` in `main.py:268` still resolves and
   the three test imports still work **without touching their import lines**.
4. Run the full suite. The route behavior is unchanged, so
   `tests/test_passthrough_route.py`, `test_passthrough_dispatch.py`, and
   `test_inline_image.py` must pass untouched. If they don't, the public
   surface in `__init__.py` is incomplete — fix the re-export, not the test.
5. **Re-anchor the 82 lesson cites.** Run
   `scripts/check-lesson-links.py docs/lesson-ai/lesson-15-openai-routes.md`
   (and lessons 02/04/07/13/14). The checker prints the corrected line for
   identifier-labelled cites — apply those mechanically. For bare
   `file.py:NN` cites it can only flag shape, so eyeball those. **The cite
   paths also change**: `routes/openai.py#L920` becomes e.g.
   `routes/openai/streaming.py#L40`. This is a path rewrite, not just a line
   shift — budget for it. Update `AUDIT.md`'s 31 cites too (gitignored, but
   the queue should stay accurate).
6. Sweep Lesson 15's prose: its whole-system map and section headers
   reference "the 1464-line file" / single-file framing. Reframe as the new
   package layout. This is a **lesson edit** — follow the lesson workflow in
   `AGENTS.md` (it's not an audit finding, it's a mechanical follow-on to a
   source move, so no drain gate; but match the course's style).

### 4c. Public surface that MUST stay importable from `audrey.routes.openai`

Verified consumers (grep `from audrey.routes.openai`):
- `main.py` → `router`
- `tests/test_inline_image.py` → `ChatCompletionRequest`, `ChatMessage`
- `tests/test_passthrough_route.py` → (passthrough names — open the file)
- `tests/test_passthrough_dispatch.py` → (passthrough names — open the file)
- module `__all__` today: `["router", "VIRTUAL_MODELS"]`

`__init__.py` must re-export **all** of these so no consumer import line
changes. This is the contract that makes 4b step 4 a green-test checkpoint.

### 4d. Risk / rollback

- **Risk:** import cycle between `routes` and `streaming`. Mitigation: move
  the shared SSE-frame helpers to the lowest layer (`responses.py` or a
  dedicated `_sse.py`) so both import *down*, never sideways.
- **Risk:** a missed re-export silently breaks a test import. Mitigation:
  the test suite is the gate — don't proceed past 4b.4 until green.
- **Rollback:** the mechanical-move commit is self-contained; revert it and
  the cites/prose are untouched (do the cite work in a *separate* commit
  after the move is confirmed green).

### Done when

Package split landed, `audrey.routes.openai` public surface unchanged, full
suite green untouched, ruff clean, all 82 cites re-anchored to the new paths
(`check-lesson-links.py` reports zero confident DRIFT on the six lesson
files), Lesson 15 prose reframed to the package layout, `PROJECT_STATE.md`
"deferred `routes/openai.py` split" followup marked resolved.

---

## Suggested order & commits

All four ship this pass. Suggested commit slicing:

1. **Item 1** (fast-path fallback) — independent, highest value.
   `feat(fast-path): fall back to next healthy model on OllamaError`
2. **Item 2** (deep-panel de-dup) — independent, pure refactor.
   `refactor(deep-panel): extract shared worker-prep from run_panel variants`
3. **Item 3** (discovery parallelism) — independent, small.
   `perf(discovery): fetch tool servers concurrently`
4. **Item 4** (openai split) — last, two commits:
   `refactor(routes): split routes/openai.py into a package` (mechanical move),
   then `docs(lesson-ai): re-anchor cites after openai.py split`.

Items 1–3 are order-independent (no shared files). Item 4 goes last so its
large mechanical diff doesn't collide with the smaller logic changes, and so
the cite re-anchoring happens against an otherwise-stable tree.
