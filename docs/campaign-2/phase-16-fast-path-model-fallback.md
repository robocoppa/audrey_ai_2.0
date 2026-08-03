# Campaign 2 Phase 16 — Fast-path model fallback + faster Thinking banner

Two fast-path improvements that ship together:

1. **Model fallback** — makes the most common request path resilient to a
   transient failure on the top-ranked model. Before this phase, if the
   highest-priority healthy model errored mid-request, the **current** request
   failed (a 502 on the non-streaming path) even though a healthy fallback
   existed in the registry.
2. **Faster Thinking banner** — the streaming fast path used to show
   `> _Thinking_` only *after* the classifier LLM call finished (router model,
   up to 20s, worse under GPU contention). The banner now appears immediately,
   and short keyword-free prompts skip the classifier LLM entirely.

The two are independent but both live in the fast path, so they deploy as one
rebuild.

---

# Part 1 — Model fallback

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
- **[`docs/lesson-ai/lesson-06-the-model-layer.md`](../../docs/lesson-ai/lesson-06-the-model-layer.md)** —
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

---

# Part 2 — Faster Thinking banner

## The problem

On the **streaming** fast path, the `> _Thinking_` banner used to be emitted
*after* `classify_with_registry` — a blocking LLM call to the router model
([classify.py](../../src/audrey/pipeline/classify.py), `router.timeout_s`
default **20s**). The banner couldn't render earlier because its
`system_fingerprint` embedded the chosen model name, which depended on the
classification result. So on a busy GPU (router model cold, or another turn
holding the GPU), the user stared at an empty response while classification
ran.

The deep path never had this lag — it emits `Planning` immediately using a
fixed `deep_panel` fingerprint, before any model work.

## The fix — two parts

**(a) Emit the banner before classifying.** The deep-vs-fast routing decision
uses only cheap *local* signals (`is_complex`, the forced-model checks,
`owui_task`, `image_turn`) — it never needed the classifier LLM. Only model
*selection* needs the task type. So the fast branch now:

1. Decides fast-vs-deep from local signals.
2. Emits the role frame + `> _Thinking_` immediately, using a
   `payload.model`-based (model-independent) fingerprint.
3. *Then* classifies to pick the model.
4. Closes the banner line with the concrete model name (`✅ qwen3.6:35b`) —
   exactly as the tool-capable and deep paths already do.

The model name moved from the (now-instant) header to the closing line; the
banner is on the wire before the router call instead of after it.

**(b) Skip the classifier LLM for short, keyword-free prompts.** The
classifier already short-circuits with no LLM call on a *strong* keyword match.
New: prompts at or below `router.skip_llm_under_tokens` tokens that produced no
strong keyword route to `general` (the tool-capable fast path) without the
router round-trip. Short plain prompts ("thanks", "what's 2+2", "summarize
this") are almost always `general`, so the LLM call was low-value — and on a
busy GPU it was the main remaining source of banner lag. A *weak* keyword
signal (e.g. `code_weak`) is still honored over the bare `general` default.

## What's in scope (Part 2)

- **[`src/audrey/routes/openai.py`](../../src/audrey/routes/openai.py)** —
  `_stream_via_pipeline` restructured: routing decision first (local signals),
  fast branch emits role + `Thinking` before classify, model name on the
  closing line. Error/no-healthy-model paths emit under the already-open stream
  identity (a new `_fast_stop()` closure) instead of a second role frame. The
  now-unused `_emit_single_message` helper was removed.
- **[`src/audrey/pipeline/banners.py`](../../src/audrey/pipeline/banners.py)** —
  `PhaseTicker` gained `emit_header=False` so the tool-capable path can open a
  ticker on the already-emitted Thinking line without re-emitting the header.
- **[`src/audrey/pipeline/classify.py`](../../src/audrey/pipeline/classify.py)** —
  `classify(...)` gained `skip_llm_under_tokens`; threaded through
  `classify_with_registry` from `router.skip_llm_under_tokens`.
- **[`config.yaml`](../../config.yaml)** — new `router.skip_llm_under_tokens: 8`
  (conservative default; **0 disables** and restores the always-call-router
  behavior).
- **[`docs/lesson-ai/lesson-07-classification-and-routing.md`](../../docs/lesson-ai/lesson-07-classification-and-routing.md)** —
  re-anchored the two `classify.py` cites that shifted.

## Tuning the skip gate

`router.skip_llm_under_tokens` is the one knob. Default **8** tokens is
conservative — only the shortest prompts skip. Raise it if you want more chit-
chat to bypass the router (faster, slightly coarser routing); set **0** to
turn the skip off entirely and always call the router. The early banner is not
gated — it's unconditional.

## Behavior notes

- The early banner changes only *when* `> _Thinking_` appears, not its text or
  the final answer. The model name still shows, on the closing line.
- The `system_fingerprint` on fast-path frames is now `audrey-<ver>/<virtual>`
  (e.g. `audrey-7.0.0/audrey_fast`) instead of `…/<concrete model>`. OpenAI
  clients (incl. OWUI) don't pin the fingerprint across chunks — the deep path
  has always used a model-independent fingerprint — so this is safe.

---

# Deploy on Unraid (both parts)

No custom-tools change. **`config.yaml` changed** (Part 2's
`skip_llm_under_tokens`) — it's read at startup, so the rebuild picks it up.
From `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

# Verification

Hermetic (laptop, already green): **485 pytests pass** (+15 over Phase 15:
+8 fallback, +5 classify skip-gate, +2 PhaseTicker). Ruff clean on all touched
Python files (the 9 `kb/` ASYNC240 hints are pre-existing/accepted).

Live, on the box:

**Part 1 — fallback** (only shows under a real failure, so best-effort):

1. Normal traffic to **audrey_fast** / **audrey_auto** answers as before — no
   latency change on the happy path. Confirm the single-attempt log line:

   ```
   docker logs audrey-ai 2>&1 | grep -E "fast_path .* tools=off, attempt 1/"
   ```

2. If you can force a top-model failure (stop the local model so the first
   chat errors), the next request should fall back and still answer:

   ```
   docker logs audrey-ai 2>&1 | grep -E "fast_path: .* failed \(attempt 1/.*\)"
   docker logs audrey-ai 2>&1 | grep -E "fast_path .* attempt 2/"
   ```

   Without an induced failure, attempt-2 lines simply won't appear — correct.

**Part 2 — banner latency** (this is the one you reported):

3. Send a short plain prompt to **audrey_fast** (e.g. "hey") and watch the UI.
   `> _Thinking_` should appear **immediately**, not after a multi-second
   pause. The model name fills in on the closing line.

4. Confirm short prompts skip the router LLM:

   ```
   docker logs audrey-ai 2>&1 | grep -E "task=general\(short_skip"
   ```

   A longer or keyword-bearing prompt should still show `router:` or
   `keyword:` as the reason — the skip is gated, not blanket.

5. Regression: a long prompt still routes through the router (no quality
   regression), and a deep request (**audrey_deep**) still shows
   `Planning → Dispatching panel → Synthesizing` unchanged.

## Note for Phase 19

This change shifted line numbers in `routes/openai.py`, so the lesson cites
into that file (lesson-07 and others) now point a few lines off. They are
**intentionally not re-anchored here** — Phase 19 splits `routes/openai.py`
into a package and rewrites all 82 of those cites' paths wholesale, so fixing
the line offsets now would be throwaway work. The `classify.py` cites (which
Phase 19 won't touch) *were* re-anchored.

## What this unblocks

The fast path answers more reliably (Part 1) and *feels* immediate (Part 2):
the Thinking ack is on the wire before any model work, and chit-chat skips the
router entirely. Closes Item 1 of the 2026-06-23 optimization review
(`optimization-pass-plan.md`) and the fast-path banner-latency report.
