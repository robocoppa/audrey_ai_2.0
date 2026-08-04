# Audrey docs

## Layout

- [`lesson-ai/`](lesson-ai/) — lesson plan teaching the codebase
  end-to-end. Start at [`lesson-ai/README.md`](lesson-ai/README.md).
- [`lesson-python/`](lesson-python/) — the language-level companion track.
- [`campaign-1/`](campaign-1/) — historical phase-by-phase build docs
  (Phases 1 → 33). Each `phase-N-deploy.md` ends with smoke-test commands
  that verified the phase when it shipped. `campaign-1/HISTORY.md` (gitignored,
  laptop-only) is the authoritative running state of the build campaign.
- [`campaign-2/`](campaign-2/) — the current build campaign, Phase 1 → 38.
- [`unraid-ollama.md`](guides/unraid-ollama.md) — canonical Ollama container config
  (referenced by Phase 1).
- [`owui-prompt-suggestions.json`](owui-prompt-suggestions.json) — OWUI
  preset prompt suggestions.

### Reading a phase number

**Phase numbers repeat across campaigns.** Campaign 1 ran 1 → 33 and
Campaign 2 is at 38, so "Phase 13" names two unrelated documents — the
per-user upload work in Campaign 1, the passthrough virtual model in
Campaign 2. Always say which campaign, and prefer the filename: since
Campaign 2's docs are named for what they contain, the name disambiguates
on its own.

Campaign 2 filenames are zero-padded (`phase-07-…`) so they sort in phase
order. Campaign 1's are not — it is closed, and its contents are indexed
below. A trailing `-plan` / `-deploy` appears only where a phase has both
documents; a bare topic name means it is the only one.

## Phase docs by feature area

The phase docs in `campaign-1/` are historical — written when the feature
shipped. Use the groupings below for navigation.

### Campaign 2 (current)

**Foundations and cleanup (1–12)**

- [01 chat archive](campaign-2/phase-01-chat-archive-plan.md) — per-user
  searchable chat archive: stored automatically, searched deliberately by tool
  call, kept out of automatic prompt context by default.
  ([deploy](campaign-2/phase-01-chat-archive-deploy.md))
- [02 prompt centralization](campaign-2/phase-02-prompt-centralization-plan.md)
  — one home for prompt defaults, plus compact task-role prompts.
  ([02a composer deploy](campaign-2/phase-02a-prompt-composer-deploy.md))
- [03 lesson-cite checker](campaign-2/phase-03-lesson-cite-checker.md) — the
  drift checker that keeps `file.py#L42` cites in the lesson docs honest.
- [04 manual UI smoke testing](campaign-2/phase-04-manual-ui-smoke-testing.md)
- [05 tools-server uv workspace](campaign-2/phase-05-tools-server-uv-workspace.md)
- [06 KB audit fixes](campaign-2/phase-06-kb-audit-fixes.md)
  ([06a complexity gate](campaign-2/phase-06a-complexity-gate-investigation.md))
- [07 fast-path Thinking banner](campaign-2/phase-07-fast-path-thinking-banner.md)
- [08 cleanup](campaign-2/phase-08-cleanup.md)
- [09 Grafana dashboard provisioning](campaign-2/phase-09-grafana-dashboard-provisioning.md)
- [10 KB reconcile bookkeeping](campaign-2/phase-10-kb-reconcile-bookkeeping.md)
- [11 image pinning audit](campaign-2/phase-11-image-pinning-audit.md)
- [12 chunk-tail fix](campaign-2/phase-12-chunk-tail-fix.md) — drop
  near-duplicate tail chunks.

**Routing, models and the chat path (13–23)**

- [13 passthrough virtual model](campaign-2/phase-13-passthrough-virtual-model.md)
  — `audrey_passthrough/<concrete>`, the route that puts LAN clients behind
  `FairLocalGate` and `UserInflightRegistry`.
- [14 fleet watchdog](campaign-2/phase-14-fleet-watchdog.md) — bot liveness.
- [15 inline image support](campaign-2/phase-15-inline-image-support.md)
- [16 fast-path model fallback](campaign-2/phase-16-fast-path-model-fallback.md)
- [17 deep-panel dedup](campaign-2/phase-17-deep-panel-dedup.md)
- [18 concurrent tool discovery](campaign-2/phase-18-concurrent-tool-discovery.md)
- [19 split openai routes](campaign-2/phase-19-split-openai-routes.md) —
  `routes/openai.py` becomes a package.
- [20 concurrent cloud workers](campaign-2/phase-20-concurrent-cloud-workers.md)
- [21 local worker timeout fix](campaign-2/phase-21-local-worker-timeout-fix.md)
- [22 depth-intent routing](campaign-2/phase-22-depth-intent-routing.md)
- [23 deep synthesis timeout](campaign-2/phase-23-deep-synthesis-timeout.md)
  — pool-aware. (Its heading says 23a; the phase is 23.)

**Research quality (25–28)**

- [25 research fact-check stage](campaign-2/phase-25-research-fact-check-stage.md)
- [26 research claim ledger](campaign-2/phase-26-research-claim-ledger.md) —
  claim/source ledger plus SearXNG fallback.
- [27 eval on box](campaign-2/phase-27-eval-on-box.md) — run the live eval on
  the box, independent of the laptop's internet.
- [28 research grounding diagnostic](campaign-2/phase-28-research-grounding-diagnostic.md)
  — the `read_url` failure.

**Fetching and access (29–31)**

- [29 web_fetch page-opener](campaign-2/phase-29-web-fetch-page-opener.md)
- [30 web_fetch SSRF hardening](campaign-2/phase-30-web-fetch-ssrf-hardening.md)
- [31 KB query auth](campaign-2/phase-31-kb-query-auth.md) — authentication on
  the query routes, which is what made publishing `audrey-ai:8000` to the LAN
  safe.

**Video ingest (32–38)** — one pipeline built in slices; each phase is
deployable and verifiable on its own.

- [32 video upload transport](campaign-2/phase-32-video-upload-transport.md) —
  chunked parts past Cloudflare's 100 MB body cap. **Landed.**
- [33 video job lifecycle](campaign-2/phase-33-video-job-lifecycle.md) — claim,
  lease, complete, fail, requeue. **Landed, fully verified.**
- [34 media-worker container](campaign-2/phase-34-media-worker-container.md) —
  ffmpeg in a sidecar, no model calls. **Landed, fully verified.**
- [35 video transcript](campaign-2/phase-35-video-transcript.md) — whisper,
  baked into the worker image. **Landed, fully verified** (step 5 moved to 38
  with `keep_source`).
- [36 video visual assessment](campaign-2/phase-36-video-visual-assessment.md)
  — keyframes through the `vl` pool. **Landed, working** — the gate keeps 6 of
  19 sampled frames and each describe costs ~62s, which is the cost data
  phase 38 was waiting for.
- [37 video summary](campaign-2/phase-37-video-summary.md)
- [38 video optimise](campaign-2/phase-38-video-optimise.md) — making it
  affordable, and where deleting the source video now lives.

**Retrieval quality**

- [39 hybrid retrieval](campaign-2/phase-39-hybrid-retrieval.md) — BM25
  alongside the vectors, so the KB can find what a document *says* and not
  only what it means. Prompted by measurements in phase 35: a 10-word
  paraphrase returned its chunk at 0.796 while a 6-word verbatim quote from
  the same transcript returned nothing. **Landed, fully verified** — that
  quote now comes back at rank 1, at a cosine the floor had been discarding.

There is no Phase 24 in Campaign 2.

### Foundation (Phases 1–10)

- [phase-1-deploy.md](campaign-1/phase-1-deploy.md) — Ollama clean recreation
- [phase-2-deploy.md](campaign-1/phase-2-deploy.md) — custom-tools + Brave API
- [phase-3-deploy.md](campaign-1/phase-3-deploy.md) — Qdrant container
- [phase-4-deploy.md](campaign-1/phase-4-deploy.md) — audrey-ai pass-through
- [phase-5-deploy.md](campaign-1/phase-5-deploy.md) — model registry + health tracker
- [phase-6-deploy.md](campaign-1/phase-6-deploy.md) — fast-path + deep-panel routing
- [phase-7-deploy.md](campaign-1/phase-7-deploy.md) — streaming responses
- [phase-8-deploy.md](campaign-1/phase-8-deploy.md) — KB ingest CLI + watcher + KB query routes
- [phase-9-deploy.md](campaign-1/phase-9-deploy.md) — deep-panel workers run ReAct

### Knowledge base + memory (Phases 11, 12, 27, 30)

- [phase-11-deploy.md](campaign-1/phase-11-deploy.md) — per-user memory recall (sqlite-backed)
- [phase-12-deploy.md](campaign-1/phase-12-deploy.md) — memory backend rewrite to Qdrant + nomic-embed
- [phase-27-deploy.md](campaign-1/phase-27-deploy.md) — SSRF guards on image embed +
  watcher `on_deleted` / `on_moved` fixes
- [phase-30-deploy.md](campaign-1/phase-30-deploy.md) — periodic KB reconcile pass for
  global collections

### Per-user uploads + auth (Phases 13–16, 26)

- [phase-13-deploy.md](campaign-1/phase-13-deploy.md) — per-user uploads to Qdrant
- [phase-14-deploy.md](campaign-1/phase-14-deploy.md) — OWUI-backed auth on all routes
- [phase-15-deploy.md](campaign-1/phase-15-deploy.md) — sqlite uploads index
- [phase-16-deploy.md](campaign-1/phase-16-deploy.md) — admin auth-cache controls
- [phase-26-deploy.md](campaign-1/phase-26-deploy.md) — auth boundary fix (auth on chat
  completions, KB ingest, tools rediscover; memory_recall user filter)

### Observability + reliability (Phases 17, 22, 24)

- [phase-17-deploy.md](campaign-1/phase-17-deploy.md) — Prometheus metrics endpoint
- [phase-22-deploy.md](campaign-1/phase-22-deploy.md) — per-tool dispatch metrics +
  alert rules + targeted auth-cache eviction
- [phase-24-deploy.md](campaign-1/phase-24-deploy.md) — Prometheus + Grafana stack into
  the audrey repo
- [phase-24a-deploy.md](campaign-1/phase-24a-deploy.md) — `appdata/audrey/` →
  `appdata/runtime/` rename
- [phase-24b-deploy.md](campaign-1/phase-24b-deploy.md) — Dockerfile install-layer split
  (~6s source rebuilds vs ~30-60s)

### Routing + scheduling (Phases 18, 18a, 19, 20, 23, 25)

- [phase-18-deploy.md](campaign-1/phase-18-deploy.md) — streaming progress banners +
  five-virtual-model lineup (`audrey_auto`, `audrey_fast` added)
- [phase-18a-datetime-context.md](campaign-1/phase-18a-datetime-context.md) — ISO-8601
  datetime context injection
- [phase-19-deploy.md](campaign-1/phase-19-deploy.md) — synth token streaming
- [phase-20-deploy.md](campaign-1/phase-20-deploy.md) — per-user fair scheduling
  (`FairLocalGate` + per-user in-flight cap)
- [phase-23-deploy.md](campaign-1/phase-23-deploy.md) — fast-path GPU gating + Phase 20
  round-robin starvation fix
- [phase-25-deploy.md](campaign-1/phase-25-deploy.md) — synth context cleanup +
  brevity-cue-aware reflect

### Build + packaging (Phase 21, 31)

- [phase-21-deploy.md](campaign-1/phase-21-deploy.md) — Dockerfile pyproject conversion
- [phase-31-deploy.md](campaign-1/phase-31-deploy.md) — image digest pinning + docs cleanup

### UX (Phase 28)

- [phase-28-deploy.md](campaign-1/phase-28-deploy.md) — per-worker tools-used footer on
  streaming responses

### Test suite (Phase 29)

- [phase-29-deploy.md](campaign-1/phase-29-deploy.md) — starter test suite (110 tests
  hermetic + offline)
