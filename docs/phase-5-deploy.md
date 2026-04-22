# Phase 5 — Classification + complexity + fast path (Unraid)

**Goal:** upgrade the Phase 4 pass-through to a real pipeline.
`/v1/chat/completions` now runs a LangGraph:

```
classify (keyword pre-filter → qwen3:4b → fallback)
  ↓
complexity gate (tiktoken; ≥500 tokens → deep)
  ↓
fast_path (pick top healthy model for task)       if not complex
  ↓
deep_stub (clear "Phase 6" message, not silent)   if complex
```

No tools, no panels, no KB yet. Reflection/escalation hooks exist but aren't
wired — they land in Phase 6 alongside real deep panels.

**Prereqs confirmed:**
- Phase 4 done: `audrey-ai` on `ollama-net`, `/health` + `/v1/models` + pass-through all green.
- `qwen3:4b` pulled into Ollama (you already have it per `ollama list`).

---

## Step 1 — Rebuild the image

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker build -f docker/audrey.Dockerfile -t audrey-ai:latest .
docker restart audrey-ai
```

Rebuild adds `langgraph`, `langchain-core`, `tiktoken`. ~30 s.

---

## Step 2 — Smoke tests

From Unraid terminal (or any LAN host):

```bash
# 1. Health still green
curl -s http://localhost:8000/health | jq
# → {"status":"ok","version":"7.0.0"}

# 2. Simple general prompt → classified general → fast path → qwen3.6:35b
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"what is the capital of Portugal?"}],
    "stream": false
  }' | jq '{fp: .system_fingerprint, ans: .choices[0].message.content[0:120]}'
# → system_fingerprint contains "qwen3.6:35b" (top of `general` pool)

# 3. Code prompt → classified code (keyword:code_strong) → qwen3-coder-next:latest
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"```python\ndef foo():\n    pass\n```\nwhat does this do?"}],
    "stream": false
  }' | jq '.system_fingerprint'
# → contains "qwen3-coder-next:latest"

# 4. Review-override: "review this code" → reasoning, not code
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_deep",
    "messages":[{"role":"user","content":"please review this python code for bugs: def add(a,b): return a-b"}],
    "stream": false
  }' | jq '.system_fingerprint'
# → contains the top `reasoning` pick (qwen3.6:35b)

# 5. Long prompt (>500 tokens of cl100k_base) → complexity=true → deep_stub
# Unraid host has no python3; run it inside the container.
LONG=$(docker exec audrey-ai python3 -c "print(' '.join(['lorem ipsum dolor sit amet']*200))")
curl -s -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"audrey_deep\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"stream\":false}" \
  | jq '{fp: .system_fingerprint, ans: .choices[0].message.content[0:120]}'
# → fp contains "deep_stub"
#   ans starts with "[deep-panel not yet implemented — Phase 6]"

# 6. Streaming still works (SSE frames end with [DONE])
curl -s -N -XPOST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"audrey_local",
    "messages":[{"role":"user","content":"count 1 to 5"}],
    "stream": true
  }' | tail -3
# → final frame is `data: [DONE]`

# 7. Check the logs show the pipeline decisions
docker logs audrey-ai --tail 25 | grep -E "classify:|complexity:|fast_path|deep_stub" | tail -10
# → lines like:
#   classify: general (router:general, conf=0.85)
#   complexity: 7 tokens (threshold=500) -> fast
#   fast_path task=general -> qwen3.6:35b
#   chat.completions model=audrey_deep task=general(...) mode=fast -> qwen3.6:35b
```

---

## Step 3 — Report back

If 1–7 all pass, Phase 5 is done. Reply with "phase 5 smoke tests passed"
and we'll move to Phase 6 (real deep panel: parallel workers + synth +
reflection).

If anything fails, paste:
- `docker logs audrey-ai --tail 80`
- The failing curl output
- Which test number failed

---

## Troubleshooting

- **Router timing out / 2-strike fallback firing every time:** check
  `qwen3:4b` is loaded: `docker exec ollama ollama list | grep qwen3:4b`.
  If missing, `docker exec ollama ollama pull qwen3:4b`.
- **All prompts classified general:** likely the router is returning
  non-JSON. Enable DEBUG logs:
  `docker exec audrey-ai sh -c 'kill -USR1 1'` (not implemented) — easier:
  bump the Python log level by rebuilding with `--build-arg LOG_LEVEL=DEBUG`
  in a future iteration. For now, check the raw classifier output:
  `docker logs audrey-ai | grep router`.
- **Long prompts going to fast path:** confirm
  `COMPLEXITY_TOKEN_THRESHOLD` isn't overriding via env; `config.yaml`
  default is 500.
