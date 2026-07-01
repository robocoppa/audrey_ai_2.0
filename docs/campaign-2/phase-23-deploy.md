# Campaign 2 Phase 23a — Pool-aware deep synthesis timeout

Makes the deep-panel **synthesizer** use the same pool-aware per-worker timeout
the **panel** already uses. Before this, both deep paths handed the synthesizer
the raw `deep_worker` budget (360s) even on the cloud-only pool, while that
pool's panel ran on `timeouts.cloud` (240s).

> Phase 23a is the code item of the Phase 23 audit-finding drain. 23b–d are
> lesson-doc drains with no runtime impact and are not part of this deploy.
> (Note: the number 23 was previously used for the reverted `web_search`
> grounding nudge — now deprecated; see PROJECT_STATE.)

## What it does

`pick_panel_timeout(cfg, pool_key)` exists so the streaming and non-streaming
deep paths can't drift on per-worker timeout selection. It returns
`timeouts.cloud` for the cloud-only `deep_panel_cloud` pool (those workers run
in parallel under the Ollama Pro concurrency cap) and `timeouts.deep_worker` for
the local-holding pools (`deep_panel`, `deep_panel_local`), where at least one
worker holds the single GPU gate and can't overlap.

The panel calls used it; the synthesizer calls didn't — they passed
`deep_worker_timeout` directly. So:

```text
audrey_cloud  panel: 240s (cloud)   synth: 360s (deep_worker)   ← mismatch
audrey_deep   panel: 360s           synth: 360s                 ← already aligned
audrey_local  panel: 360s           synth: 360s                 ← already aligned
```

After this change both call sites pass `pick_panel_timeout(cfg, pool_key)`, so
cloud-only synthesis now runs on 240s, matching its panel. The two local-holding
pools are byte-identical (the helper returns `deep_worker` for them).

## Why it matters

Not a crash and not a wrong answer — cloud-only synthesis simply had more
headroom than its panel. The real defect was **drift**: a future change to the
cloud pool's timeout would have silently failed to reach the synthesizer, which
is exactly what `pick_panel_timeout`'s docstring promises won't happen. This
re-unifies the synthesizer with the helper.

## What's in scope

- **[`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)** —
  `node_synthesize` now passes `timeout_s=pick_panel_timeout(cfg, pool_key)`
  (the `pool_key` was already derived in the node for the panel-pool lookup).
  The builder-scope `deep_worker_timeout` local was removed — it fed only this
  call (the panel node, `node_deep_panel`, computes its own `pick_panel_timeout`).
- **[`src/audrey/routes/openai/pipeline.py`](../../src/audrey/routes/openai/pipeline.py)** —
  the streaming synth (`_run_synth_stream` inside `_stream_via_pipeline`) now
  reuses the `timeout_s` already computed from `pick_panel_timeout` for the
  panel. The local `deep_worker_timeout` (computed only to feed this call) was
  removed.
- **[`tests/test_deep_panel.py`](../../tests/test_deep_panel.py)** — +2 tests.
  They build the graph with `synthesize_fn` stubbed, invoke just the compiled
  `synthesize` node per virtual model, and assert the forwarded `timeout_s`
  equals `pick_panel_timeout(cfg, pool_key)` — failing before this change for
  `audrey_cloud` (it would have captured 360s). `pick_panel_timeout` itself is
  already unit-tested in `test_config_validation.py`; these pin the *call-site*
  contract.

No `config.yaml` change. No custom-tools change.

## Behavior invariant

`deep_panel` and `deep_panel_local` synthesis are unchanged (the helper returns
`deep_worker` for both, exactly the previous value). Only `deep_panel_cloud`
synthesis changes — 360s → 240s — bringing it in line with its own panel. The
synthesizer's three-tier failure path (primary → fallback → degrade-to-longest-
draft) is untouched; only the deadline it runs under changed.

## Deploy on Unraid

Code-only change (no config, no custom-tools). From
`/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

## Verification

Hermetic (laptop): **503 pytests pass** (+2); ruff clean on the touched source
files (the 9 pre-existing `kb/` ASYNC240 hints remain); no new confident lesson
cite drift.

Live, on the box:

1. Send a deep request to **audrey_cloud** (the cloud-only pool). It should
   complete normally with the deep banners
   (`Planning → Dispatching panel → Synthesizing`) and a synthesized answer —
   the synthesizer now runs on the 240s cloud budget instead of 360s.

2. Regression: an **audrey_deep** or **audrey_local** deep request still
   synthesizes normally (those pools' synth timeout is unchanged at 360s).

There is no new log marker for this change — it's an internal timeout-selection
fix. If a cloud-only synthesis ever times out at exactly ~240s under heavy load
where it previously squeaked through at ~300s, that's the expected new ceiling;
the panel was already capped there, so a synth that needs >240s on cloud is the
signal to raise `timeouts.cloud`, not to special-case the synthesizer.

## What this unblocks

Closes the `consider` CODEBASE finding from the 2026-06-24 full-corpus audit
(`docs/lesson-ai/AUDIT.md`). The synthesizer and panel now share one timeout
source, so the cloud pool's budget can be tuned in one place without the
synthesizer silently keeping the old value.
