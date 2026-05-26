# Audrey AI 2.0

Self-hosted multi-model LangGraph orchestrator that exposes an OpenAI-compatible
`/v1/chat/completions` endpoint, runs on Unraid with dual RTX 3090 Ti GPUs, and
routes requests across local Ollama models + cloud-bridge models through a
classify → route → tool-call → reflect pipeline.

## High-level architecture

```
Open WebUI ──[Cloudflare Tunnel]──> audrey-ai ─────> Ollama (local + cloud)
                                       │             │
                                       │             └─ both 3090 Ti GPUs
                                       │
                                       ├─> custom-tools (7 OpenAPI endpoints)
                                       │     web_search, kb_search,
                                       │     kb_image_search, memory_store,
                                       │     memory_recall, memory_search,
                                       │     chat_history_search
                                       │
                                       ├─> Qdrant (text vectors + CLIP images,
                                       │   per-user collections + global KB
                                       │   + per-user chat archive)
                                       │
                                       └─> KB watcher (event-driven re-ingest)
                                           + reconcile (periodic orphan sweep,
                                             startup + 30 min cadence)

                                  + Prometheus + Grafana (metrics + alerts,
                                    dashboards provisioned from monitoring/)
                                  + OWUI-backed auth on all routes
```

All containers sit on Docker network `ollama-net`. Only Open WebUI is exposed
publicly via the Cloudflare Tunnel; every other service is internal.

## Virtual models

Five virtual models, each a different routing mode over the same registry:

| Model            | Routing                                                                  |
|------------------|--------------------------------------------------------------------------|
| `audrey_auto`    | Adaptive: fast path on short prompts, deep panel on long ones.           |
| `audrey_fast`    | Always fast path; never escalates, even on long prompts.                 |
| `audrey_deep`    | Always deep panel, mixed cloud + local pool.                             |
| `audrey_cloud`   | Always deep panel, cloud-only pool (up to 3 parallel workers).           |
| `audrey_local`   | Always deep panel, local-only pool (serialized through GPU gate).        |

Streaming responses (when the client sends `stream: true`) emit progress
banners during each phase (Thinking / Planning / Dispatching / Synthesizing)
and a per-worker tools-used footer after the answer.

## Pipeline shape

```
datetime injection
  → memory recall (per-user, semantic via custom-tools)
  → classify (keyword pre-filter + qwen3:4b router)
  → complexity gate (OWUI-task detect → virtual-model force → token threshold)
  → fast path (single model + ReAct if tool-capable)
    | planner (optional sub-question decomposition)
    → deep panel (N parallel workers, all tool-capable)
    → synth (cloud or local, streamed)
  → reflect (length + brevity-cue check)
  → retry on failure
```

Per-user fair scheduling (`FairLocalGate`) round-robins across users at the
local-GPU bottleneck. Per-user in-flight cap (default 3) prevents one user
from saturating the queue. Cloud calls bypass the gate. Every chat is captured
to the per-user chat archive (SQLite source-of-truth + Qdrant index) so the
model can search prior conversations via `chat_history_search`.

## Documentation

- **`AGENTS.md`** — start here. Tool-agnostic agent guide: project shape,
  runtime rules, deploy boundaries, git/change hygiene.
- **`docs/lessons/`** — codebase walk-through. Lesson 4 is the end-to-end
  request lifecycle; later lessons drill into specific subsystems
  (model layer, classify+route, deep mode, ReAct, KB ingest/lifecycle).
- **`docs/campaign-1/phase-N-deploy.md`** — original 31-phase build history.
- **`docs/campaign-2/phase-N-deploy.md`** — post-1.0 feature work
  (chat archive, prompt centralization, KB audit fixes, complexity-gate
  fixes, Thinking/Planning banners, Grafana dashboards).
- **`docs/unraid-ollama.md`** — clean Ollama container recreation on Unraid.
- **`monitoring/README.md`** — Prometheus + Grafana stack, including how to
  add a new dashboard via repo-managed provisioning.

## Dev workflow

```bash
# On the laptop
uv sync --extra dev          # install deps + pytest
.venv/bin/pytest tests/ -q   # run the hermetic test suite

# Manual run (rare — Unraid is the deploy target)
uv run audrey                # start FastAPI on :8000
uv run audrey-ingest --source /path/to/docs --topic geology
```

## Repo layout

```
src/audrey/         # orchestrator package (FastAPI + LangGraph)
tools-server/       # custom-tools FastAPI service (separate package)
tests/              # pytest suite — hermetic, no Ollama/Qdrant needed
docker/             # Dockerfiles (audrey-ai + custom-tools, base images pinned to digest)
monitoring/         # prometheus + grafana compose, scrape config, alert rules, provisioned dashboards
docs/               # campaign histories, lessons, deploy guides
scripts/            # model-pull, smoke tests, lesson-cite link checker
config.yaml         # model registry, fast_path, deep_panel*, fairness, KB, reconcile
.env.example        # BRAVE_API_KEY, GRAFANA_ADMIN_PASSWORD, etc.
compose.yaml        # audrey-ai + custom-tools (ollama/qdrant/owui stay on Unraid UI)
AGENTS.md           # canonical agent guide
```

## Status

Active development. Campaign 1 (phases 1-31) shipped the core orchestrator;
Campaign 2 is post-1.0 feature work. See `docs/PROJECT_STATE.md`
(gitignored, laptop-local) for the current priority, verified stack state,
and per-phase behavioral facts.
