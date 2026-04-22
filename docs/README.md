# Audrey docs

Per-phase deploy guides and reference material. Written for Unraid operator use.

- `phase-1-deploy.md` — Ollama clean recreation (Phase 1)
- `phase-2-deploy.md` — custom-tools + Brave API wiring (Phase 2)
- `phase-3-deploy.md` — Qdrant container (Phase 3)
- `phase-N-deploy.md` — one per phase, added as we go
- `unraid-ollama.md` — canonical Ollama container config (referenced by Phase 1)
- `unraid-audrey.md` — canonical Audrey + tools + WebUI setup
- `cloudflared-routing.md` — existing tunnel → Open WebUI wiring
- `kb-geology.md` — KB ingest workflow + rock ID examples
- `future-tools.md` — how to add run_python, read_document, sql_query, etc. later

Each `phase-N-deploy.md` ends with an exact smoke-test command. Paste its output
back to Claude and we advance to the next phase.
