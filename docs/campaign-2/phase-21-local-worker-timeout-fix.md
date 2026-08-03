# Campaign 2 Phase 21 — Fix the `audrey_local` worker timeout

Stops the `❌` you saw on the local deep panel: a worker was hitting the 240s
`deep_worker` ReadTimeout while cold-loading behind the GPU gate. Two
config-only changes — reorder the local pool and give cold-loading workers more
headroom.

**Config-only. No code change.**

## What you saw

```
Dispatching panel..... ❌ glm-4.7-flash:q8_0 ..... ✅ qwen3.6:35b ✅
```

The `❌` means that worker returned **no usable draft**. The log gave the
reason:

```
deep_panel: worker glm-4.7-flash:q8_0 failed in 240.00s:
  POST /api/chat transport error: ReadTimeout
```

`240.00s` exactly = it hit the `timeouts.deep_worker` deadline. Zero bytes came
back, so the model was almost certainly still **cold-loading** when the clock
ran out — it never started generating.

## Why it happened (and why it's structural, not a glm bug)

`deep_panel_local` (the `audrey_local` pool) runs **two local models**, and
with `GPU_CONCURRENCY=1` they **can't overlap** — they serialize through the
single GPU gate. So the worker that runs *second* has to cold-load into VRAM
after the first model is evicted, and its per-worker timeout counts the
load + generate together. `glm-4.7-flash:q8_0` (a heavier q8 quant) couldn't
finish both inside 240s when it ran second behind `qwen3.6:35b`.

The same risk applies to any second local worker — it's the serial-pool design,
not one model. So the fix has two parts.

## What changed

**1. Reorder the local pool; drop the timing-out model.** Worker *order*
decides which model runs (and cold-loads) first — the first-listed worker is the
de-facto primary. `reasoning` and `general` now lead with `deepseek-r1:32b`,
then `qwen3.6:35b`. `glm-4.7-flash:q8_0` is removed from `general`.

| local task | before                              | after                              |
|------------|-------------------------------------|------------------------------------|
| reasoning  | `qwen3.6:35b`, `deepseek-r1:32b`    | `deepseek-r1:32b`, `qwen3.6:35b`   |
| general    | `qwen3.6:35b`, `glm-4.7-flash:q8_0` | `deepseek-r1:32b`, `qwen3.6:35b`   |
| code, vl   | unchanged                           | unchanged                          |

**2. Raise the cold-load headroom.** `timeouts.deep_worker: 240 → 360`. Local
deep workers (which run second + cold-load) now have room to load and generate
without tripping a false timeout.

## Scope notes

- `timeouts.deep_worker` is shared by **both** local-holding pools: the mixed
  `deep_panel` (`audrey_deep`) and `deep_panel_local`. The bump helps both —
  `audrey_deep`'s one local worker gets the same headroom. `deep_panel_cloud`
  (`audrey_cloud`) is unaffected; it uses `timeouts.cloud` (240s), which is
  fine because cloud workers don't cold-load or hold the GPU gate.
- `deepseek-r1:32b` is registered under `reasoning` in `model_registry`; the
  deep-panel validator checks workers against the *flat set* of all registry
  names (not per-task), so using it as a `general` worker is valid and boots
  clean.

## Trade-offs (deliberate)

- **`deepseek-r1:32b` is a reasoning/"thinking" model.** As the primary
  `general` worker it'll emit longer `<think>` chains and be slower per turn
  than `qwen3.6:35b` was. That's a quality-first choice for the local-only path
  — accept the extra latency for stronger drafts. If `audrey_local/general`
  feels too slow for chatty prompts, swap the order back (qwen3.6 first) or
  make `general` single-worker.
- **The serial-pool tax remains.** Two local models still take turns through
  one GPU, so `audrey_local` is inherently ~2× a single-model turn plus a
  cold-load. 360s gives headroom; it doesn't make them parallel. If you want
  the local path faster, the real lever is a single-worker local pool.

## Deploy on Unraid

`config.yaml` changed (read at startup). No custom-tools change. From
`/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Prerequisite

`deepseek-r1:32b` must be pulled (it was already an `audrey_local/reasoning`
worker, so if that pool ever produced a deepseek draft, it's present):

```
docker exec ollama ollama list | grep deepseek-r1
```

## Verification

Hermetic (laptop): **494 pytests pass**; config validates (the real-config boot
test guards `config.yaml`); `pick_panel_timeout` returns 360s for `deep_panel`
and `deep_panel_local`, 240s for `deep_panel_cloud`.

Live, on the box:

1. Send a deep prompt to **audrey_local** (general or reasoning) and watch the
   banner — both workers should now finish `✅`, leading with `deepseek-r1:32b`:

   ```
   docker logs audrey-ai 2>&1 | grep -E "deep_panel: worker .* failed|dispatch"
   ```

   No `failed in 240.00s` line should appear. (`glm-4.7-flash:q8_0` shouldn't
   appear at all on `general` anymore.)

2. If a worker *still* times out, it'll now read `failed in 360.00s` — that
   would mean the model genuinely can't load+generate in 6 minutes on this box
   (a real capacity problem, not a tight deadline), and the next step is a
   single-worker local pool, not a bigger timeout.

3. Regression: **audrey_deep** and **audrey_cloud** still answer normally;
   `audrey_deep`'s per-worker timeout is now 360s too (longer grace, same
   behavior otherwise).

## What this unblocks

The `audrey_local` panel stops dropping a worker to a false timeout: a stronger
local lead (`deepseek-r1:32b`), no more `glm-4.7-flash:q8_0` ReadTimeout, and
enough cold-load headroom that a second serial local worker can actually finish.
Follows from the `❌`-banner diagnosis (2026-06-24).
