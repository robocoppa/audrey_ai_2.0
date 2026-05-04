# Phase 9 — Deep-panel workers run ReAct (tool use) under the GPU gate

**Goal:** stop the deep panel being tool-blind. When a deep-panel worker's
model is in `fast_path.tool_capable_models`, it now runs a ReAct loop —
`kb_search`, `web_search`, memory tools — before answering. Drafts come
back tagged as tool-grounded; the synthesizer prefers them on factual
disagreements.

What's new vs Phase 8:

- **`agentic.react.deep_worker`** block in `config.yaml` — separate,
  tighter budget for deep workers (`max_rounds=2` default). Falls back to
  the fast-path `react` values for any field not set.
- **`deep_panel.run_panel`** accepts `tools` + `tool_capable_models` +
  per-worker ReAct params. Each worker dispatches through
  `pipeline/react.py` when its model is tool-capable, or a one-shot
  `ollama.chat` otherwise.
- **GPU gate holds for the whole ReAct loop.** Previously
  `_run_one_worker` only wrapped a single `chat` call; now the
  `gate.acquire()` block surrounds the entire ReAct loop (chat → tools →
  chat → … → final). Two local workers never have overlapping VRAM.
- **`WorkerDraft`** carries `tool_rounds` + `tool_calls`. Synth prompt
  tags drafts with `[tool-grounded: N rounds]` so the synthesizer can
  weight them.
- **Pool sizes trimmed to 2 workers** per `(pool, task)`. Mixed pools
  (`deep_panel`) pair one local + one cloud so the cloud worker runs
  while the local worker holds the gate. `deep_panel_cloud` keeps both
  cloud so Ollama Pro's 3-concurrent cap is exercised.
  `deep_panel_local` strictly serializes two local models.
- **Deep-panel log line** now includes `tool_grounded=<n>` alongside
  `ok=<n>`:
  `deep_panel: pool=<k> task=<t> workers=2 ok=2 tool_grounded=1 attempted=[…]`.

**What didn't change:** the escalation guard in `graph.py`'s
`route_after_fast_path` still returns `end` when `tool_rounds > 0`. A
fast-path tool answer is already grounded — re-running through deep
workers mostly adds latency. Revisit later if the smoke tests show deep
panels produce meaningfully better grounded answers than the fast path.

**Prereqs confirmed:**

- Phase 8 green on this host (KB endpoints healthy, `custom-tools`
  advertising 5 tools).
- At least one dataset populated enough to produce non-trivial KB hits
  for a worker (`servicenow` and `bjj` are the anchors from Phase 8).
- `ollama list` shows the tool-capable models configured in
  `fast_path.tool_capable_models`.

---

## Step 1 — Pull & rebuild

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

No new Python deps — the rebuild is fast. Config-only change to pool
sizes is picked up at container start (no rebuild strictly needed, but
`--build` is cheap and keeps the image in lockstep with the repo).

If the log shape in 2.1 doesn't appear, the image is stale — rebuild
with `--no-cache`.

---

## Step 2 — Smoke tests

All tests run on Unraid from `/mnt/user/appdata/audrey_ai_2.0`. For
inline JSON generation, run it inside the `audrey-ai` container (Unraid
host has no `python3`).

### 2.1 — Config parsed, pools are 2 workers each

```bash
docker exec audrey-ai python3 -c "
import yaml
c = yaml.safe_load(open('/app/config.yaml'))
for pool in ('deep_panel', 'deep_panel_cloud', 'deep_panel_local'):
    for task, spec in c[pool].items():
        ws = spec['workers']
        print(f'{pool:18s} {task:9s} workers={len(ws)} {ws}')
print()
dw = c['agentic']['react'].get('deep_worker', {})
print('deep_worker react:', dw)
"
```

**Expected:** every `(pool, task)` line says `workers=2`.
`deep_worker react: {'max_rounds': 2, 'compress_after_round': 2, 'max_tool_result_chars': 2000}`.

### 2.2 — Deep panel runs tools on a KB-relevant prompt (`audrey_local`)

This exercises the local-only pool (two local workers, strictly
serialized via `GpuGate`). `audrey_local` is used instead of
`audrey_deep` because `audrey_deep` only forces deep when the prompt
clears the complexity gate (500 tokens) — short KB-forcing prompts
would slip through to the fast path. `audrey_local` forces deep
unconditionally (`forced_by_virtual_model`).

The prompt is deliberately KB-forcing — it names `kb_search`, demands
quoted passages with source filenames, and explicitly forbids falling
back to parametric knowledge. Tool-capable models consistently call
`kb_search` on this shape; a generic "explain ServiceNow triage"
prompt is easy to bypass.

```bash
docker exec audrey-ai python3 -c "
import json
print(json.dumps({
  'model': 'audrey_local',
  'messages': [{'role': 'user', 'content':
    'Use kb_search against our ServiceNow KB to find the documented '
    'remediation steps for 500 errors on the incident form. Quote the '
    'specific passages you retrieve and cite the source filename. Do '
    'not answer from general knowledge — if the KB has nothing, say so.'}]
}))
" | curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @- | jq '.choices[0].message.content' | head -60
```

Then check the log for:

```bash
docker logs audrey-ai --tail 120 2>&1 | grep -E "classify:|complexity:|planner:|deep_panel:|react:|synth:|reflect:"
```

**Expected:**

- `complexity:` line shows `-> deep (forced_by_virtual_model)`.
- `planner:` line shows at most 2 subtasks (matches pool size — see
  `agentic.planning.max_subtasks`).
- One `react: round=<n> model=<tool_capable> tool_calls=<m>` line per
  worker, per round. At least one `tool_calls=1` expected.
- `deep_panel: pool=deep_panel_local task=general workers=2 ok=2 tool_grounded=<n> attempted=[…]`
  — `tool_grounded` should be `>= 1`. On a well-populated KB, `=2`.
- `synth: qwen3.6:35b ok in <s>s (attempt 1)`.
- Response body either quotes KB passages with filenames *or* plainly
  says the KB has no coverage for the exact question. Both are pass
  outcomes — the signal is that the model consulted retrieval instead
  of fabricating.

> **Note on `audrey_deep`:** that virtual model only forces deep when
> the prompt clears the 500-token complexity threshold (see
> `pipeline/graph.py:node_complexity`). A short KB-forcing prompt under
> `audrey_deep` will correctly run on the fast path. To exercise deep
> panel with a short prompt, use `audrey_local` / `audrey_cloud`, or
> pad the `audrey_deep` prompt past 500 tokens.

### 2.3 — Synth input tags tool-grounded drafts

The synth prompt now adds `[tool-grounded: N rounds]` to any draft that
ran tools. Verify the synth received the tag by temporarily raising log
verbosity or by calling the panel directly from inside the container:

```bash
docker exec -i audrey-ai python3 <<'PY'
import asyncio
from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.deep_panel import run_panel
from audrey.pipeline.semaphore import GpuGate
from audrey.pipeline.synthesize import _format_drafts_for_synth
from audrey.tools.discovery import discover_all, ToolRegistry

async def main():
    cfg = get_config()
    ollama = OllamaClient(cfg.env.ollama_host)
    health = HealthTracker()
    gate = GpuGate(concurrency=1)
    registry = ModelRegistry(cfg)
    tool_servers = list(cfg.tools.get("servers", []) or [])
    tools = await discover_all(tool_servers) if tool_servers else ToolRegistry()
    capable = set(cfg.raw['fast_path']['tool_capable_models'])
    prompt = ('Use kb_search to find anything about resetting a stuck '
              'ServiceNow incident workflow. Quote passages with filenames.')
    drafts, attempted = await run_panel(
        cfg, ollama, registry, health, gate,
        pool_key='deep_panel_local', task='general',
        messages=[{'role':'user','content':prompt}],
        subtasks=[], options={}, timeout_s=240, max_workers_cloud=3,
        tools=tools, tool_capable_models=capable,
        react_max_rounds=2, react_compress_after=2,
        react_max_tool_chars=2000, react_dispatch_timeout_s=30,
    )
    for d in drafts:
        print(f"model={d.get('model')} tool_rounds={d.get('tool_rounds',0)} "
              f"content_chars={len(d.get('content','') or '')} err={d.get('error','')}")
    print('---')
    print(_format_drafts_for_synth(prompt, drafts, [])[:2000])

asyncio.run(main())
PY
```

> **`-it` vs `-i`:** use `docker exec -i` (no `-t`) when piping a heredoc or
> redirected input — `-t` allocates a pseudo-TTY and fails with
> `the input device is not a TTY` when stdin isn't attached to a terminal.

**Expected:**

- Each draft prints with `tool_rounds=0` or `tool_rounds=>=1`.
- The `_format_drafts_for_synth` dump shows at least one `--- draft <n>
  (model=…, elapsed=…s) [tool-grounded: <n> rounds] ---` line when a
  worker used tools.

### 2.4 — Local workers strictly serialize through GpuGate (`audrey_local`)

This is the critical VRAM-safety test. Two local models
(`qwen3.6:35b` + `deepseek-r1:32b` for reasoning, or `qwen3.6:35b` +
`qwen2.5-coder:32b` for code) must not overlap.

```bash
docker exec audrey-ai python3 -c "
import json
print(json.dumps({
  'model': 'audrey_local',
  'messages': [{'role': 'user', 'content': (
    'Based on our BJJ knowledge base, explain the mechanical difference '
    'between a standard triangle choke setup from closed guard and a '
    'reverse triangle from mount. Cite which transitions our KB '
    'recommends.'
  )}]
}))
" | curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @- >/dev/null
```

Then inspect worker ordering in the logs:

```bash
docker logs audrey-ai --tail 200 2>&1 | \
  grep -E "react: round=|deep_panel:" | \
  tail -40
```

`GpuGate.acquire` / `.release` don't emit log lines — the evidence for
serialization is the *ordering and timing* of `react:` lines plus the
total panel wall-clock.

**Expected:**

- All `react: round=…` lines for worker A (a tool-capable local model)
  appear before any for worker B. If rounds from two local workers
  interleave in time, the gate isn't holding the full loop — stop and
  investigate `pipeline/deep_panel.py:_run_one_worker`.
- A non-tool-capable worker in the pool (e.g. `glm-4.7-flash:q8_0` for
  `general`, `deepseek-r1:32b` for `reasoning`) won't emit any
  `react:` line — it runs one-shot. Its serialization is implicit in
  the gate + the fact that the panel doesn't return until both
  workers finish.
- Total panel wall-clock ≈ `sum(worker_times) + synth_time`. If it's
  much less, parallelism leaked somewhere.

### 2.5 — Cloud workers run in parallel (`audrey_cloud`)

Opposite of 2.4 — two cloud models should start at nearly the same
wall-clock time (no gate).

```bash
time docker exec audrey-ai python3 -c "
import json
print(json.dumps({
  'model': 'audrey_cloud',
  'messages': [{'role': 'user', 'content':
    'Summarize current best practices for incident triage in ServiceNow, '
    'grounded in our KB where applicable.'}]
}))
" | curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @- >/dev/null
```

```bash
docker logs audrey-ai --tail 120 2>&1 | grep -E "react: round=0 model=|deep_panel:" | tail -20
```

**Expected:**

- Both `react: round=0 model=<cloud_model>` log lines appear within
  ~1-3 s of each other (true parallel start). This is the primary
  pass signal.
- `deep_panel: pool=deep_panel_cloud task=<t> workers=2 ok=2
  tool_grounded=<n>` — ideally `=2`. Cloud models are more aggressive
  tool-callers than locals.
- End-to-end wall time is dominated by `max(worker_time) +
  synth_time`, not `sum(worker_times)`.
- Workers routinely hit `max_rounds=2` and trigger the forced-final
  path (logged as `react: max_rounds=2 reached for <model>; forcing
  final answer without tools`). This is expected and the panel still
  completes cleanly — see the "forced-final" note below.

> **On hitting max_rounds:** at `deep_worker.max_rounds=2`, both cloud
> and local tool-capable models frequently exhaust the budget. The
> forced-final path (compress history + append a directive user turn +
> chat without tools) consistently produces a real answer in ~30-60s
> on cloud. If you observe deep-panel answers that feel consistently
> underresearched, consider bumping `agentic.react.deep_worker.max_rounds`
> to 3; the latency cost is ~+20-30s per worker × N workers.

### 2.6 — Non-tool-capable model → one-shot chat, no ReAct (regression)

If any pool entry is *not* in `tool_capable_models`, that worker should
skip ReAct entirely. `deep_panel_local.general` includes
`glm-4.7-flash:q8_0`, which is **not** in `tool_capable_models`. Run an
`audrey_local` general-category prompt and confirm that model's log
line.

```bash
docker exec audrey-ai python3 -c "
import json
print(json.dumps({
  'model': 'audrey_local',
  'messages': [{'role': 'user', 'content': 'Name three common rock types and one property of each.'}]
}))
" | curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @- >/dev/null

docker logs audrey-ai --tail 80 2>&1 | grep -E "react: round=|deep_panel:"
```

**Expected:**

- No `react:` log line for `glm-4.7-flash:q8_0` — it ran one-shot.
- `qwen3.6:35b` may or may not emit `react:` lines depending on whether
  it decided to call tools; if it did, `tool_grounded=1`.

### 2.7 — Escalation guard still holds (fast-path tool answer → no deep re-run)

A fast-path answer that used tools should **not** escalate to deep, even
if the answer looks short. Use `audrey_deep` — it's the only virtual
model that respects the complexity gate, so a short prompt goes fast
path. (`audrey_local` and `audrey_cloud` both force deep regardless of
length.)

```bash
docker exec audrey-ai python3 -c "
import json
print(json.dumps({
  'model': 'audrey_deep',
  'messages': [{'role': 'user', 'content': 'use kb_search to check our servicenow kb for anything about 500 errors on the incident form'}]
}))
" | curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @- >/dev/null

docker logs audrey-ai --tail 80 2>&1 | grep -E "classify:|complexity:|fast_path|react:|escalate:|deep_panel:"
```

**Expected:**

- `fast_path task=… -> <tool_capable_model> (tools=on)`.
- One or more `react: round=…` lines on the fast path.
- **No `escalate:` line, no `deep_panel:` line.** Fast path's
  `tool_rounds > 0` short-circuits escalation.

### 2.8 — Synthesis reflects tool grounding

Eyeball one `audrey_deep` answer from 2.2. The synthesizer's `##
Caveats` section should mention retrieved evidence or disagreements
between grounded vs ungrounded drafts — not just generic hedging. If
all drafts were tool-grounded and agreed, `- none` is acceptable.

---

## Pass criteria

- 2.1: every `(pool, task)` line reports `workers=2`; `deep_worker`
  react block loaded.
- 2.2: deep panel logs `tool_grounded >= 1`; answer has the three-header
  structure.
- 2.3: at least one draft carries a `[tool-grounded: N rounds]` tag in
  the synth input.
- 2.4: local workers serialize — no overlapping `gate.acquire` / ReAct
  rounds across models.
- 2.5: cloud workers start in parallel; panel wall time ≈
  `max(worker) + synth`.
- 2.6: non-tool-capable worker runs one-shot (no `react:` line).
- 2.7: fast-path tool answer ends without escalating to deep.
- 2.8: synthesis references evidence or acknowledges grounded-vs-not
  disagreement where applicable.

All green → update `CONTINUITY.md` with Phase 9 verification notes and
move on to Phase 10 (Open WebUI repoint + Cloudflare Tunnel wiring).

---

## Troubleshooting

- **No `tool_grounded=N` field in the `deep_panel:` log line**: the
  image is stale (still has the Phase 8 log format). Rebuild with
  `docker compose up -d --build audrey-ai`. If still missing,
  `--no-cache` the build.

- **All drafts have `tool_rounds=0` even for KB-relevant prompts**: the
  model chose not to call a tool — that's a model-level decision, not a
  wiring bug. Confirm by checking the same model runs tools on the fast
  path for a similar prompt. If the fast path runs tools but deep
  doesn't, check `tool_capable_models` in `config.yaml` includes both
  workers' names exactly as they appear in the pool.

- **Local workers overlap in logs (rounds interleave)**: the gate isn't
  held for the full ReAct loop. Check
  `src/audrey/pipeline/deep_panel.py:_run_one_worker` — `run_react` must
  be called *inside* the `async with gate.acquire(...)` block, not
  after it.

- **Panel takes 2× longer than expected on `audrey_cloud`**: cloud
  workers are serializing because the registry marked them `location:
  local`. Check `model_registry` entries in `config.yaml` — every cloud
  model must have `location: cloud`.

- **`deep_panel: no healthy pool workers for <pool>/<task>; falling
  back to registry`**: the two workers configured for that `(pool,
  task)` are both marked unhealthy. Check `docker exec audrey-ai curl
  -sS localhost:8000/v1/health/models | jq` and reset health tracker
  state if needed.

- **Fast-path tool answer DID escalate**: the guard check in
  `route_after_fast_path` fires before `tool_rounds` is populated. Check
  that `node_fast_path` returns `tool_rounds` in its state update
  (it does — `src/audrey/pipeline/graph.py` sets `"tool_rounds":
  int(react_meta.get("tool_rounds", 0))`). If it's always 0 on ReAct
  responses, the `_react` key isn't being attached in
  `fast_path.run_fast_path`.

- **ReAct loop burns max_rounds every call**: deep workers budget is
  `max_rounds=2`. If every tool-capable worker always hits 2 and
  returns a low-quality answer, raise `agentic.react.deep_worker.max_rounds`
  to 3 and watch latency. Don't touch the fast-path
  `agentic.react.max_rounds` — that still wants 3.
