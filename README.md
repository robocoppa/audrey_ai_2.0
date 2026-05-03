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
                                       ├─> custom-tools (6 OpenAPI endpoints)
                                       │     web_search, kb_search,
                                       │     kb_image_search, memory_store,
                                       │     memory_recall, memory_search
                                       │
                                       ├─> Qdrant (text vectors + CLIP images,
                                       │   per-user collections + global KB)
                                       │
                                       └─> KB watcher (event-driven re-ingest)
                                           + reconcile (periodic orphan sweep)

                                  + Prometheus + Grafana (metrics + alerts)
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
banners during each phase (Thinking / Dispatching / Synthesizing) and a
per-worker tools-used footer after the answer.

## Pipeline shape

```
datetime injection
  → memory recall (per-user, semantic via custom-tools)
  → classify (keyword pre-filter + qwen3:4b router)
  → complexity gate (token threshold)
  → fast path (single model + ReAct if tool-capable)  | deep panel (N parallel workers, all tool-capable)
  → synth (cloud or local, streamed)
  → reflect (length + brevity-cue check)
  → retry on failure
```

Per-user fair scheduling (`FairLocalGate`) round-robins across users at the
local-GPU bottleneck. Per-user in-flight cap (default 3) prevents one user
from saturating the queue. Cloud calls bypass the gate.

## Documentation

- `docs/phase-N-deploy.md` — one per phase, with smoke-test verification steps
- `docs/unraid-ollama.md` — clean Ollama container recreation on Unraid
- `docs/README.md` — categorized index of phase docs by feature area

## Dev workflow

```bash
# On the laptop
uv sync --extra dev          # install deps + pytest
.venv/bin/pytest tests/ -q   # run the test suite (110 tests, ~1s)

# Manual run (rare — Unraid is the deploy target)
uv run audrey                # start FastAPI on :8000
uv run audrey-ingest --source /path/to/docs --topic geology
```

## Repo layout

```
src/audrey/         # orchestrator package (FastAPI + LangGraph)
tools-server/       # custom-tools FastAPI service (separate package)
tests/              # pytest suite — hermetic, no Ollama/Qdrant needed
docker/             # Dockerfiles (audrey-ai + custom-tools)
monitoring/         # prometheus + grafana compose + scrape config + rules
docs/               # deploy guides, per-phase instructions
scripts/            # model-pull, smoke tests
config.yaml         # model registry, fast_path, deep_panel*, fairness, KB
.env.example        # BRAVE_API_KEY, GRAFANA_ADMIN_PASSWORD, etc.
compose.yaml        # audrey-ai + custom-tools (ollama/qdrant/owui stay on Unraid UI)
```

## Status

Active development. See `CONTINUITY.md` (gitignored, dev-machine only) for
the current phase, verified stack state, and per-phase behavioral facts.

Phases 1 → 31 verified.
