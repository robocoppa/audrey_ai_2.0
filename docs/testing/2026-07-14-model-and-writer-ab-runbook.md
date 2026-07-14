# Runbook — model sweep + writer route A/B (2026-07-14)

Two independent tests, prepped for you to deploy + run on the box. Scoring is
**structural (harness) + your human read** of the answers file, same as the
topics report.

- **Test 1 — model sweep:** 3 best local vs 3 best cloud, direct passthrough.
  No config change, no redeploy. One command.
- **Test 2 — writer route A/B:** local vs cloud *writer* stage inside
  `audrey_research`. Needs a config edit + redeploy per arm (2 arms).

Both case files are baked into `Dockerfile.eval`, so **rebuild the eval image
once** before running either (they're new files):

```
docker compose --profile eval build audrey-eval
```

---

## Test 1 — model sweep (no redeploy)

**Models.** Three best local, three best cloud (all already in
`passthrough.allowed_models`, so nothing to deploy):

| local | cloud |
|---|---|
| `qwen3.6:35b` (reasoning/general #1) | `deepseek-v4-pro:cloud` |
| `deepseek-r1:32b` (dedicated reasoner) | `kimi-k2.6:cloud` |
| `qwen3-coder-next:latest` (code #1) | `glm-5.2:cloud` |

**Cases** (`eval_prompts_models_ab.json`, 9 cases spanning the domains the
registry actually routes): 2 reasoning, 2 science-explain, 2 writing, 2
general-knowledge, 1 self-contained Python (`code_test` auto-scored).

**Run (on the box):**

```
CASES=eval_prompts_models_ab.json LABEL=models-ab \
  MODELS='audrey_passthrough/qwen3.6:35b,audrey_passthrough/deepseek-r1:32b,audrey_passthrough/qwen3-coder-next:latest,audrey_passthrough/deepseek-v4-pro:cloud,audrey_passthrough/kimi-k2.6:cloud,audrey_passthrough/glm-5.2:cloud' \
  scripts/eval-onbox.sh
```

That runs all 9 cases against all 6 models (54 completions) and drops an answers
file + a results JSON in the mounted testing-out dir, then Telegram-pings.

**Then, on the laptop, build the comparison table:**

```
.venv/bin/python scripts/eval_compare.py docs/testing/<the-results>.json \
  --out docs/testing/2026-07-14-models-ab-compare.md
```

**What the harness scores (objective):** `code_runs` (the Python case),
`answer_contains` (markup→82.8, element-W→tungsten/wolfram, Berlin→1989),
latency (ttft/total per model). **What you read (human):** prose quality on the
science + writing cases, hedge-preservation on the GK cases, and whether the
local coder's non-code answers are acceptable. Expect the local coder to look
weak on prose and strong on the code case — that contrast is the point.

---

## Test 2 — writer route A/B (config swap + redeploy per arm)

**What it tests:** the `audrey_research` route's final **writer** stage is
`qwen3.6:35b` (local). Does moving it to `glm-5.2:cloud` produce better final
answers, and at what latency/cost? The writer adds no new facts — it turns
verified findings into the answer — so this is a pure prose/assembly test.

**Cases** (`eval_prompts_writer_ab.json`, 5 cases): 3 science explainers +
1 multi-step reasoning + 1 GK-with-hedge (checks the writer preserves the
hedge). Anchors byte-identical to the topics set.

### Arm A — baseline (current config, writer = local)

No edit needed; current config already has `writer: "qwen3.6:35b"`.

```
MODEL=audrey_research CASES=eval_prompts_writer_ab.json LABEL=writer-local \
  scripts/eval-onbox.sh
```

Save that answers file — it's arm A.

### Arm B — writer = cloud

Edit `config.yaml`: change **all four** `writer:` lines under
`deep_panel_research` (lines ~192, 198, 204, 213 — code/reasoning/general/vl)
from:

```yaml
    writer: "qwen3.6:35b"
```
to:
```yaml
    writer: "glm-5.2:cloud"
```

(You only need code/reasoning/general for these cases, but flip all four for
consistency — vl won't be exercised.)

Redeploy the app service (config-only change, no rebuild):

```
docker compose up -d --force-recreate audrey-ai
```

Then run the identical cases:

```
MODEL=audrey_research CASES=eval_prompts_writer_ab.json LABEL=writer-cloud \
  scripts/eval-onbox.sh
```

### Revert

Change the four `writer:` lines back to `"qwen3.6:35b"` and
`docker compose up -d --force-recreate audrey-ai` again. **Don't leave the box
on the experimental config** — the local writer is the deliberate cost-discipline
choice for the research route.

### Compare

Put the two answers files side by side (LABEL keeps their filenames distinct:
`…-writer-local-…` vs `…-writer-cloud-…`). The human read is the whole point
here — look for: is the cloud writer's prose tighter/better organized? Did it
preserve the hedge on the GK case? And check the latency line: the local writer
pays a GPU cold-load (~30–98s) that the cloud writer skips, so cloud may be
*faster* end-to-end despite being a network call — that's the surprising result
to confirm or refute.

---

## Cleanup / notes

- Nothing here changes the default eval trio or any live lineup — both tests are
  opt-in and Test 1 touches no config at all.
- If you later want either case set in the routine suite, add a protocol entry
  to `scripts/run_all_evals.sh` (not done — these are one-off investigations).
- After Test 2, the box **must** be reverted to `writer: "qwen3.6:35b"`.
