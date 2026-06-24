# Campaign 2 Phase 16 — Fast-path model fallback

Makes the most common request path resilient to a transient failure on the
top-ranked model. Before this phase, if the highest-priority healthy model
errored mid-request, the **current** request failed (a 502 on the
non-streaming path) even though a healthy fallback existed in the registry.

## What it does

The non-tools fast path used to pick one model and re-raise on any
`OllamaError`. Now it tries the highest-priority healthy model first, and on
an `OllamaError` it cools that model down and tries the next healthy
candidate — up to two models total. Only when every candidate fails does the
error propagate (the graph node still turns that into a 502).

```text
before:  pick top healthy -> chat -> error? -> record_failure + raise (502)
after:   for cand in top-2 healthy:
             chat -> ok?    -> record_success, return
                  -> error? -> record_failure, try next
         all failed -> raise (502)
```

This mirrors the deep panel, which already tolerates per-worker failure and
falls back to registry candidates. The fast path — the path almost every
request takes — was the one that was single-shot.

## Why this exists

A model can blip transiently (cold load, a momentary cloud 5xx, a timeout
under contention). The health tracker already cooled the model down on
failure, so the *next* request routed around it — but the request that hit
the blip still 502'd. The fix closes that one-request window by retrying
within the same request against a healthy fallback.

The cap is **two** models (top + one fallback), matching the deep panel's
emergency-fallback cap. If the two highest-priority healthy models both fail
mid-request, a third is unlikely to help and the added latency is real.

## What's in scope

- **[`src/audrey/pipeline/fast_path.py`](../../src/audrey/pipeline/fast_path.py)** —
  new `_FAST_FALLBACK_LIMIT = 2` and `_healthy_fast_candidates(...)` helper;
  `run_fast_path`'s non-tools branch rewritten as a bounded loop over healthy
  candidates (`fast_path.py:113`+). `pick_fast_model` is unchanged and still
  used to choose the model the *tools* branch runs.
- **[`tests/test_fast_path.py`](../../tests/test_fast_path.py)** — new file, 8
  tests: happy-path single attempt (byte-identical to before), fallback to the
  next healthy model, exhaustion → raise, the cap stops at two, no-healthy →
  raise, and the tools branch staying single-shot.
- **[`docs/ai-course/lesson-06-the-model-layer.md`](../../docs/ai-course/lesson-06-the-model-layer.md)** —
  re-anchored the four `fast_path.py` cites that shifted, and updated the
  now-stale prose (the "single Ollama chat call" passage) to describe the
  bounded fallback. The model-layer contract (`success → record_success`,
  `OllamaError → record_failure`) is unchanged.

## What's NOT in scope — the tools branch stays single-shot

The ReAct (tools) branch is **not** retried across models. A failure can land
*after* a tool side effect (e.g. `memory_store` already wrote), so a blind
model-swap could double-apply it. The tool-capable model chosen first is the
one that answers; an `OllamaError` from it propagates as before (and still
cools the model down for the next request). The asymmetry is intentional and
pinned by a test. Retrying the tools branch is a possible later item only if
production shows fast-react failures are a real 502 source.

## Behavior invariant

When the top model is healthy and succeeds — the overwhelmingly common case —
behavior is **byte-identical**: one model, one chat call, one `dispatch_total`
increment, same return shape. The fallback engages only on an actual
`OllamaError`.

## Deploy on Unraid

No config or custom-tools change. From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop, already green): **478 pytests pass** (+8), ruff clean on
`fast_path.py` + `test_fast_path.py`.

Live, on the box (the fallback only shows itself under a real failure, so this
is best-effort):

1. Normal traffic to **audrey_fast** / **audrey_auto** answers as before — no
   latency change on the happy path. Confirm the single-attempt log line:

   ```
   docker logs audrey-ai 2>&1 | grep -E "fast_path .* tools=off, attempt 1/"
   ```

2. If you can force a top-model failure (e.g. stop the local model so the
   first chat errors), the next request should log a fallback attempt and
   still answer:

   ```
   docker logs audrey-ai 2>&1 | grep -E "fast_path: .* failed \(attempt 1/.*\)"
   docker logs audrey-ai 2>&1 | grep -E "fast_path .* attempt 2/"
   ```

   Without an induced failure, attempt-2 lines simply won't appear — that's
   correct.

## What this unblocks

A transient blip on the highest-priority model no longer fails the user's
current fast-path request when a healthy fallback exists. Closes the
fast-path resilience gap surfaced in the 2026-06-23 optimization review
(`optimization-pass-plan.md`, Item 1).
