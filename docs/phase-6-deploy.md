# Phase 6 — Deep panel + synthesis + reflection (Unraid)

**Goal:** replace the Phase 5 `deep_stub` with the real deep-panel pipeline.
`/v1/chat/completions` now runs:

```
classify → complexity ─► fast_path ─► escalate? ─► END
                     ╲                         ╲
                      ╲                         └► planner ─► deep_panel ─► synthesize ─► reflect ─► retry?
                       ╲                                                                          ↺ deep_panel
                        ╲                                                                         → END
                         └► (complex=True) ─► planner ─► deep_panel …
```

What's new vs Phase 5:
- **planner.py** — optional 2–3 sub-question decomposition, gated on
  `agentic.planning.min_prompt_tokens` (default 40, env override
  `PLANNING_MIN_TOKENS`). On parse error / single subtask, falls through to `[]`.
- **deep_panel.py** — picks worker pool per virtual model
  (`audrey_deep` → `deep_panel`, `audrey_cloud` → `deep_panel_cloud`,
  `audrey_local` → `deep_panel_local`). Cloud workers run via
  `asyncio.gather` (capped at `MAX_DEEP_WORKERS_CLOUD=3`); local workers
  serialize through `GpuGate` (`GPU_CONCURRENCY=1`).
- **synthesize.py** — single-call merge into `## Approach / ## Answer /
  ## Caveats`. Falls back to `fallback_synth` once; if both fail, returns
  the longest draft so the request never 502s.
- **reflect.py** — deterministic check (no LLM). Fails on empty,
  too-short (`agentic.reflection.min_answer_chars`), or missing section
  headers. One retry max.
- **escalation hook** in `fast_path`: if the fast answer is shorter than
  `agentic.escalation.min_chars` (default 100) or the classify confidence
  was below `confidence_ceiling`, the graph re-enters in deep mode.
  `escalated_from_fast=True` prevents looping back to fast.

**Prereqs confirmed:**
- Phase 5 done: `audrey-ai` on `ollama-net`, classify + complexity + fast_path green.
- All workers + synthesizers from `config.yaml` are present in `ollama list`
  (or are cloud models — those just need Ollama Pro auth, already configured).

---

## Step 1 — Rebuild the image

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker build -f docker/audrey.Dockerfile -t audrey-ai:latest .
docker restart audrey-ai
```

No new Python deps in this phase — `langgraph`, `langchain-core`, `tiktoken`
were all installed in Phase 5.

If the new log shapes (see Step 3) don't appear after restart, rebuild
with `--no-cache` to confirm the image isn't stale.

---

## Step 2 — Smoke tests

From the Unraid terminal (or any LAN host):

### 2.1 — Health still green

```bash
curl -s http://localhost:8000/health | jq
# → {"status":"ok","version":"7.0.0"}
```

### 2.2 — Simple prompt → fast path (no escalation)

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"what is the capital of Portugal?"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, ans: .choices[0].message.content[0:120]}'
# → fp contains "qwen3.6:35b" (top of `general` pool, fast path)
# → ans is a short factual answer (Lisbon)
```

If the answer is shorter than 100 chars (escalation threshold) the graph
escalates to deep — that's expected behavior. Verify in the log line
(Step 3.7) that you see `mode=fast` for short factual prompts and
`mode=deep escalated=True` if escalation fired.

### 2.3 — `audrey_local` (always deep) on a general prompt

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_local",
    "messages":[{"role":"user","content":"explain how btrfs raid0 differs from raid1 in 4 sentences"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, len: (.choices[0].message.content|length), head: .choices[0].message.content[0:160]}'
# → fp contains "qwen3.6:35b" (synthesizer for deep_panel_local/general)
# → len > 200
# → head should start with "## Approach"
```

### 2.4 — `audrey_cloud` (always deep, cloud-only workers)

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_cloud",
    "messages":[{"role":"user","content":"compare PostgreSQL and SQLite for an embedded analytics workload, 5 sentences"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, head: .choices[0].message.content[0:160]}'
# → fp contains "kimi-k2.6:cloud" (synthesizer for deep_panel_cloud/reasoning)
# → head should start with "## Approach"
```

### 2.5 — Long prompt forces deep on `audrey_deep`

```bash
LONG=$(docker exec audrey-ai python3 -c "print(' '.join(['lorem ipsum dolor sit amet']*200))")
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"audrey_deep\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG summarize\"}],\"stream\":false}" \
  | jq '{fp: .system_fingerprint, head: .choices[0].message.content[0:160]}'
# → fp contains the synthesizer (qwen3.6:35b for deep_panel/general)
# → head starts with "## Approach"
```

### 2.6 — Code prompt on `audrey_local`

```bash
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_local",
    "messages":[{"role":"user","content":"```python\ndef add(a,b):\n    return a-b\n```\nfind the bug and write a fix"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, head: .choices[0].message.content[0:200]}'
# → fp contains "qwen3.6:35b" (synthesizer for deep_panel_local/code)
# → head should mention the bug and contain a fenced code block in ## Answer
```

### 2.7 — Streaming for deep mode emits one chunk + [DONE]

Multi-worker + synthesis can't be coherently token-streamed, so deep
streams emit the synthesized answer as a single `delta.content` chunk
followed by `data: [DONE]`.

```bash
curl -s -N -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_local",
    "messages":[{"role":"user","content":"name three rocks"}],
    "stream": true
  }' | tail -3
# → final frame is `data: [DONE]`
# → second-to-last frame contains delta.content with "## Approach"
```

### 2.8 — Streaming on `audrey_deep` short prompt still token-streams

```bash
curl -s -N -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"count 1 to 5"}],
    "stream": true
  }' | grep -c '"delta"'
# → > 5 (multiple token-level chunks)
```

### 2.9 — Inspect pipeline decisions in logs

```bash
docker logs audrey-ai --tail 80 \
  | grep -E "classify:|complexity:|fast_path|planner:|deep_panel:|synth:|reflect:|escalate:|chat\.completions" \
  | tail -25
```

Expected line shapes:
```
classify: reasoning (router:reasoning, conf=0.90)
complexity: 12 tokens (threshold=500) -> fast
fast_path task=reasoning -> qwen3.6:35b
escalate: fast→deep (chars=42, conf=0.90, reason=too_short)
planner: 2 subtasks: ['How does btrfs raid0…', 'How does btrfs raid1…']
deep_panel: pool=deep_panel_local task=general workers=3 ok=3 attempted=['qwen3.6:35b','glm-4.7-flash:q8_0','olmo-3.1:32b']
synth: qwen3.6:35b ok in 4.21s (attempt 1)
reflect: attempt=1 passed=True reason=ok
chat.completions model=audrey_local task=general(router:general, conf=0.85) mode=deep -> qwen3.6:35b pool=deep_panel_local workers=3 ok=3 reflect=ok/attempts=1 escalated=False
```

Things to watch for:
- **At least one `deep_panel:` line per deep request** with `ok > 0`.
- **`synth:` line on every deep request** — `attempt=1` ok is normal,
  `attempt=2` means primary synth failed (acceptable, fallback covered it).
- **`reflect:` line `passed=True reason=ok`** for healthy answers.
  `passed=False reason=too_short` triggers one retry; if you see two
  `reflect:` lines for one request that's the retry path firing.
- **`escalate: fast→deep`** is informational, not an error. Means the
  fast path produced something the escalation gate considered weak.

---

## Step 3 — Report back

If 2.1–2.9 all pass, Phase 6 is done. Reply with **"phase 6 smoke tests passed"**
and we'll move to Phase 7 (custom-tools wiring + ReAct loop on the fast path).

If anything fails, paste:
- `docker logs audrey-ai --tail 120`
- The failing curl output
- Which test number failed

---

## Troubleshooting

- **All deep requests return "[deep panel produced no usable drafts]":**
  every worker in the pool failed or is unhealthy. Check:
  `docker exec ollama ollama ps` (anything loaded?), `docker logs audrey-ai | grep "worker.*failed"`.
  If a specific cloud model is the problem, check Ollama Pro auth and
  `ollama list | grep cloud`.

- **Deep requests time out at 240s:** worker timeout is `timeouts.deep_worker`.
  If local 35B models are too slow under load, drop one from the pool's
  `workers:` list in `config.yaml` (rebuild + restart), or raise the
  timeout for that specific phase.

- **Reflect always fails with `missing_sections`:** the synthesizer isn't
  obeying the structured prompt. Check `docker logs audrey-ai | grep "synth:"`
  to confirm which model produced the answer; some smaller fallback synths
  drift. Easiest fix is bumping the primary synth's priority in
  `config.yaml` so it doesn't get demoted.

- **Escalation fires on every short answer:** that's by design — the
  escalation gate's `min_chars` default is 100. Raise it in
  `config.yaml > agentic.escalation.min_chars` if you want fewer escalations,
  or set `agentic.escalation.enabled: false` to disable entirely.

- **Cloud workers all skipped (logs show "skipping unhealthy"):** the
  health tracker tripped. Cooldown is exponential (5s → 5min). Wait it
  out or restart the container to clear: `docker restart audrey-ai`.

- **`audrey_cloud` request returns the local synth model:** that means
  the cloud pool was empty (all 3 workers unhealthy) and the registry
  fallback kicked in. Check `docker logs audrey-ai | grep "no healthy pool workers"`.
