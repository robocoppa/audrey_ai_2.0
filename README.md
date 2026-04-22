# Audrey AI 2.0

Self-hosted multi-model LangGraph orchestrator that exposes an OpenAI-compatible
`/v1/chat/completions` endpoint, runs on Unraid with dual RTX 3090 Ti GPUs, and
routes requests across local Ollama models + cloud-bridge models through a
classify → route → tool-call → reflect pipeline.

**Status:** early build. See `CONTINUITY.md` (gitignored) for the current phase,
verified stack state, and "what a new session should do first".

## High-level architecture

```
Open WebUI (Cloudflare Tunnel)
        ↓ OpenAI API
    audrey-ai (FastAPI + LangGraph)
        ├── classify → route → fast path | deep panel
        ├── web search (Brave API)
        ├── KB query (Qdrant: text + CLIP images)
        └── custom-tools (5 OpenAPI endpoints, auto-discovered)
            ↓
        Ollama (local + cloud-bridge models, both GPUs)
```

All containers sit on Docker network `ollama-net`. Only Open WebUI is exposed
publicly via the existing Cloudflare Tunnel.

## Virtual models

| Model            | Behavior                                                          |
|------------------|-------------------------------------------------------------------|
| `audrey_deep`    | Auto-routes: fast path when confident + input is simple; deep panel otherwise. Mixed cloud + local. |
| `audrey_cloud`   | Always deep panel, cloud-only workers + synthesizer. Up to 3 parallel workers. |
| `audrey_local`   | Always deep panel, local-only workers + synthesizer. |

## Documentation

- `docs/unraid-ollama.md` — clean Ollama container recreation on Unraid
- `docs/unraid-audrey.md` — Audrey + tools + Qdrant + WebUI deployment
- `docs/cloudflared-routing.md` — existing tunnel → Open WebUI wiring
- `docs/kb-geology.md` — KB ingest workflow + rock ID query examples
- `docs/future-tools.md` — how to add run_python, read_document, sql_query, etc. later

Each phase gets a `docs/phase-N-deploy.md` with click-by-click Unraid steps and
the exact smoke-test command to verify.

## Dev workflow

```bash
# On the laptop
uv sync                    # install deps into .venv
uv run audrey              # start the FastAPI app
uv run audrey-ingest --source ./sample-datasets/geology  # test KB ingest
```

## Repo layout

```
src/audrey/         # orchestrator package
tools-server/       # custom-tools FastAPI service (separate package)
docker/             # Dockerfiles
docs/               # deploy guides, per-phase instructions
scripts/            # model-pull script, smoke tests
config.yaml         # model_registry, fast_path, deep_panel*, timeouts, cache, tools
.env.example        # BRAVE_API_KEY and other required env vars
```
