# Campaign 2 Phase 20 — Two concurrent cloud workers on `audrey_deep`

Gives the default deep pool a third, independent draft at near-zero wall-clock
cost: the text tasks now run **1 local + 2 cloud** workers instead of 1 local +
1 cloud. The two cloud workers run concurrently (they never touch the GPU gate),
so the synthesizer gets one more diverse take without the request taking longer.

**Config-only change. No code change.**

## What it does

`deep_panel` (the pool behind `audrey_deep` and `audrey_auto → deep`) used to
run one local + one cloud worker per task. The local worker holds the GPU gate;
the cloud worker overlaps it. This phase adds a **second cloud worker** to the
three text tasks (`code`, `reasoning`, `general`), from a different vendor than
the first for draft diversity:

| task      | before                                  | after                                                        |
|-----------|-----------------------------------------|--------------------------------------------------------------|
| code      | `qwen3-coder-next` + `kimi-k2.7-code`   | + `deepseek-v4-pro:cloud`                                     |
| reasoning | `qwen3.6:35b` + `kimi-k2.6`             | + `deepseek-v4-pro:cloud`                                     |
| general   | `qwen3.6:35b` + `kimi-k2.6`             | + `deepseek-v4-pro:cloud`                                     |
| vl        | `qwen3-vl:32b` + `kimi-k2.6`            | **unchanged** (see below)                                    |

It also pins the global cloud-concurrency cap explicitly:
`agentic.max_deep_workers_cloud: 3` (previously an implicit code default of 3),
and bumps `agentic.planning.max_subtasks` from 2 → 3 to match the new 3-worker
text pools so a 3-subtask plan isn't dropped.

## Why this is close to free

Cloud workers bypass `FairLocalGate` entirely — only local models contend for
the single GPU slot (`GPU_CONCURRENCY=1`). So the two cloud workers run in
parallel with each other *and* with the local worker. Adding the second cloud
draft costs ~0 extra wall-clock (the panel still finishes when the slowest
worker returns), and the synthesizer reconciles three independent takes instead
of two — which is exactly what deep mode is for.

The two cloud models per task are deliberately different vendors
(`kimi-*` + `deepseek-v4-pro`) so the third draft is genuinely independent, not
a near-duplicate of the first.

## Why `vl` is left at 1 local + 1 cloud

The deep path rebuilds prompts **text-only**, so cloud workers in a vl panel
never see the image — and image turns are forced onto the **fast** path anyway
(the local `qwen3-vl:32b` is the real worker). A second cloud draft there would
just merge a blind text take, so it isn't worth the call.

## Why the cap matters

`max_deep_workers_cloud` caps concurrent cloud workers **per deep request**.
With 2 cloud workers per request, `2 × concurrent users` can approach Ollama
Pro's parallel limit (≈3). The cap was already 3 as a code default; pinning it
in `config.yaml` makes it visible and tunable. If you ever run a LAN fleet
firing concurrent deep requests and see cloud 429s, this is the knob — and the
deferred "global passthrough cap" note in PROJECT_STATE is the related lever.

## What's in scope

- **[`config.yaml`](../../config.yaml)** — `deep_panel.{code,reasoning,general}`
  each gain `deepseek-v4-pro:cloud` as a third worker; `vl` unchanged.
  `agentic.max_deep_workers_cloud: 3` added explicitly;
  `agentic.planning.max_subtasks: 2 → 3`. The boot validator already requires
  every named model to be in `model_registry` (`deepseek-v4-pro:cloud` is), so
  a typo fails at startup.

## What's NOT in scope

- **`deep_panel_cloud`** (`audrey_cloud`) already runs 2 concurrent cloud
  workers — unchanged.
- **`deep_panel_local`** (`audrey_local`) stays local-only by contract — it must
  never get a cloud worker (privacy / offline path). Unchanged.
- No code change. The panel already supports N workers; this just lists more.

## Deploy on Unraid

`config.yaml` changed (read at startup). No custom-tools change. From
`/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Prerequisite

`deepseek-v4-pro:cloud` must be reachable through your Ollama-cloud bridge (it's
already a `deep_panel_cloud` worker, so if `audrey_cloud` works, this does too).
No new model to pull — cloud models aren't pulled locally.

## Verification

Hermetic (laptop): **494 pytests pass**; config validates (the real-config boot
test guards `config.yaml`); ruff clean.

Live, on the box:

1. Send a deep prompt to **audrey_deep** (or a long prompt to **audrey_auto**)
   on a text task and confirm three workers dispatch:

   ```
   docker logs audrey-ai 2>&1 | grep -E "dispatch|deep_panel:|workers="
   ```

   Expect three `worker_done` banners (one local + two cloud) in the streaming
   UI, then the synthesized answer.

2. Confirm wall-clock didn't regress — the panel still finishes about when the
   slowest worker returns, not the sum. Two cloud drafts should land close
   together.

3. Confirm the cloud cap holds under load: fire two deep requests at once and
   check you don't see a burst of cloud `429`/`error` outcomes:

   ```
   docker logs audrey-ai 2>&1 | grep -E "worker .* failed|429|cloud"
   ```

4. Regression: **audrey_local** still runs local-only (no cloud worker appears),
   and **audrey_cloud** is unchanged.

## What this unblocks

Better default deep-mode answers: three independent drafts (one local, two
cloud, three vendors) for the synthesizer to reconcile, at no meaningful
latency cost. Closes the "audrey_deep should always run ≥2 concurrent cloud
models" request (2026-06-24).
