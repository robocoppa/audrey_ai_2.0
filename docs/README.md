# Audrey docs

Per-phase deploy guides + reference material. Written for Unraid operator use.
Each `phase-N-deploy.md` ends with smoke-test commands; running them top-to-
bottom verifies the phase shipped correctly.

## Reference

- [`unraid-ollama.md`](unraid-ollama.md) — canonical Ollama container config
  (referenced by Phase 1)

## Phase docs by feature area

The phase docs are historical — written when the feature shipped. Use the
groupings below for navigation; CONTINUITY.md (dev-machine only) holds the
authoritative current-state summary.

### Foundation (Phases 1–10)

- [phase-1-deploy.md](phase-1-deploy.md) — Ollama clean recreation
- [phase-2-deploy.md](phase-2-deploy.md) — custom-tools + Brave API
- [phase-3-deploy.md](phase-3-deploy.md) — Qdrant container
- [phase-4-deploy.md](phase-4-deploy.md) — audrey-ai pass-through
- [phase-5-deploy.md](phase-5-deploy.md) — model registry + health tracker
- [phase-6-deploy.md](phase-6-deploy.md) — fast-path + deep-panel routing
- [phase-7-deploy.md](phase-7-deploy.md) — streaming responses
- [phase-8-deploy.md](phase-8-deploy.md) — KB ingest CLI + watcher + KB query routes
- [phase-9-deploy.md](phase-9-deploy.md) — deep-panel workers run ReAct

### Knowledge base + memory (Phases 11, 12, 27, 30)

- [phase-11-deploy.md](phase-11-deploy.md) — per-user memory recall (sqlite-backed)
- [phase-12-deploy.md](phase-12-deploy.md) — memory backend rewrite to Qdrant + nomic-embed
- [phase-27-deploy.md](phase-27-deploy.md) — SSRF guards on image embed +
  watcher `on_deleted` / `on_moved` fixes
- [phase-30-deploy.md](phase-30-deploy.md) — periodic KB reconcile pass for
  global collections

### Per-user uploads + auth (Phases 13–16, 26)

- [phase-13-deploy.md](phase-13-deploy.md) — per-user uploads to Qdrant
- [phase-14-deploy.md](phase-14-deploy.md) — OWUI-backed auth on all routes
- [phase-15-deploy.md](phase-15-deploy.md) — sqlite uploads index
- [phase-16-deploy.md](phase-16-deploy.md) — admin auth-cache controls
- [phase-26-deploy.md](phase-26-deploy.md) — auth boundary fix (auth on chat
  completions, KB ingest, tools rediscover; memory_recall user filter)

### Observability + reliability (Phases 17, 22, 24)

- [phase-17-deploy.md](phase-17-deploy.md) — Prometheus metrics endpoint
- [phase-22-deploy.md](phase-22-deploy.md) — per-tool dispatch metrics +
  alert rules + targeted auth-cache eviction
- [phase-24-deploy.md](phase-24-deploy.md) — Prometheus + Grafana stack into
  the audrey repo
- [phase-24a-deploy.md](phase-24a-deploy.md) — `appdata/audrey/` →
  `appdata/runtime/` rename
- [phase-24b-deploy.md](phase-24b-deploy.md) — Dockerfile install-layer split
  (~6s source rebuilds vs ~30-60s)

### Routing + scheduling (Phases 18, 18a, 19, 20, 23, 25)

- [phase-18-deploy.md](phase-18-deploy.md) — streaming progress banners +
  five-virtual-model lineup (`audrey_auto`, `audrey_fast` added)
- [phase-18a-datetime-context.md](phase-18a-datetime-context.md) — ISO-8601
  datetime context injection
- [phase-19-deploy.md](phase-19-deploy.md) — synth token streaming
- [phase-20-deploy.md](phase-20-deploy.md) — per-user fair scheduling
  (`FairLocalGate` + per-user in-flight cap)
- [phase-23-deploy.md](phase-23-deploy.md) — fast-path GPU gating + Phase 20
  round-robin starvation fix
- [phase-25-deploy.md](phase-25-deploy.md) — synth context cleanup +
  brevity-cue-aware reflect

### Build + packaging (Phase 21, 31)

- [phase-21-deploy.md](phase-21-deploy.md) — Dockerfile pyproject conversion
- [phase-31-deploy.md](phase-31-deploy.md) — image digest pinning + docs cleanup

### UX (Phase 28)

- [phase-28-deploy.md](phase-28-deploy.md) — per-worker tools-used footer on
  streaming responses

### Test suite (Phase 29)

- [phase-29-deploy.md](phase-29-deploy.md) — starter test suite (110 tests
  hermetic + offline)

## Other reference

- [`owui-prompt-suggestions.json`](owui-prompt-suggestions.json) — OWUI
  preset prompt suggestions (drop into the Admin Panel → Settings →
  Suggestions field).
