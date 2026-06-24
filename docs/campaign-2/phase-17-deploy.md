# Campaign 2 Phase 17 — De-duplicate the deep-panel run functions

A pure refactor: `run_panel` and `run_panel_streaming` shared ~40 lines of
identical setup. This phase extracts that into one helper so a future change
to worker selection or the registry-fallback cap can't drift between the two
entry points.

**No behavior change. No deploy-time risk beyond a normal rebuild.**

## What it does

`run_panel` (non-streaming, `asyncio.gather`) and `run_panel_streaming`
(streaming, `asyncio.as_completed`) duplicated, byte-for-byte:

- healthy-worker selection (`select_workers`)
- the registry-fallback block (no healthy pool workers → top-2 registry
  candidates)
- subtask → per-worker message assignment
- the `dispatch_total` metric loop
- the `coros = [...]` construction

The two functions genuinely differ only in **how they await** the coroutines.
This phase moves the shared setup into a `_prepare_panel(...)` helper that
returns `(workers, coros)` (or `([], [])` when nothing is available); each
public function keeps its own await strategy.

## Why this exists

The streaming function's own comment said *"Same registry-fallback shape as
run_panel; see comment there."* That's a standing invitation to drift: a fix
to the fallback cap, the dispatch metric, or subtask assignment has to be
made in two places today, or the streaming and non-streaming deep paths
silently diverge. One source of truth removes that risk.

## What's in scope

- **[`src/audrey/pipeline/deep_panel.py`](../../src/audrey/pipeline/deep_panel.py)** —
  new private `_prepare_panel(...)` holding the shared selection / fallback /
  subtask / metric / coro-build logic. `run_panel` and `run_panel_streaming`
  call it, then apply `asyncio.gather` vs `asyncio.as_completed` respectively.
  **Public signatures unchanged** — same kwargs, same defaults, same return
  shapes and event shapes.

## Behavior invariant

Pure refactor. `gather` vs `as_completed` still differ only in *reception*
order, exactly as the current docstrings promise — worker execution order,
gate semantics, metric increments, and the streaming event shapes
(`worker_done`, `final`) are all unchanged. The streaming path still emits
exactly one `final` event (with `drafts=[], attempted=[]` when zero workers
are available).

The proof of equivalence is that the **existing deep-panel tests pass with no
edits**. If a test needs changing, the refactor changed behavior and is wrong.

## What's NOT in scope

- No scheduling change. No new fallback policy. No metric change.
- The per-worker execution (`_run_one_worker`) is untouched.

## Deploy on Unraid

No config or custom-tools change. From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): full suite green with the deep-panel tests **unmodified**;
ruff clean on `deep_panel.py`. Net LOC drops ~30. If coverage on the
registry-fallback path is thin, this phase adds one test that hits it through
**both** `run_panel` and `run_panel_streaming` to lock the shared helper.

Live, on the box (deep mode unchanged, so this is a regression check):

1. A deep-mode prompt to **audrey_deep** still fans out to the same workers
   and synthesizes the same way:

   ```
   docker logs audrey-ai 2>&1 | grep -E "deep_panel:|dispatch|synth:"
   ```

2. A streaming deep request still emits per-worker progress then the answer —
   no missing or duplicated `worker_done` banners.

## What this unblocks

One place to change deep-panel worker setup. Closes Item 2 of the 2026-06-23
optimization review (`optimization-pass-plan.md`).
