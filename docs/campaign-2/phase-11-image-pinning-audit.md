# Campaign 2 Phase 11 — Phase 30 cleanup: image pinning audit + README refresh

Closes the long-standing "Phase 30" cross-campaign followup. Like Phase 10
(which closed Phase 29), this is a verify-what's-already-done + ship-the-
loose-ends phase. The original Phase 30 spec had two parts; one turned out
to be fully done already, the other had real stale content worth fixing.

## What was already done — Part A (image digest pinning)

The original Phase 30 spec asked to pin `prom/prometheus:latest` and
`grafana/grafana:latest` to sha256 digests. Phase 31 already shipped that.

Audited every `Dockerfile` and `compose.yaml` in the repo:

| File | Image | State |
| --- | --- | --- |
| `docker/audrey.Dockerfile` | `python:3.12-slim` | pinned to digest (Phase 31) |
| `docker/audrey.Dockerfile` | `ghcr.io/astral-sh/uv:latest` | pinned to digest (Phase 31) |
| `docker/custom-tools.Dockerfile` | `python:3.12-slim` | pinned to digest (same as audrey-side) |
| `docker/custom-tools.Dockerfile` | `ghcr.io/astral-sh/uv:latest` | pinned to digest |
| `monitoring/compose.yaml` | `prom/prometheus` | pinned to digest (Phase 31) |
| `monitoring/compose.yaml` | `grafana/grafana` | pinned to digest |
| `compose.yaml` | `audrey-ai:latest`, `audrey-custom-tools` | local build targets, not upstream pulls — `:latest` here is just a name for the locally-built image, no pinning applies |

Conclusion: **Part A is fully done.** No edits needed.

The user's working rebuild command (`docker compose up -d --build && docker
image prune -a -f`) also limits the practical attack surface: rebuilds
happen on the operator's schedule, not upstream's. The aggressive `prune -a`
does mean every `up -d --build` after a prune re-pulls upstream images, so
having `prom/prometheus` and `grafana/grafana` pinned to digests is what
actually keeps tested-against and deployed images in lockstep across
prune cycles. Phase 31's pins do that work.

## What needed fixing — Part B (README + FastAPI description)

Both surfaces had drifted from current behavior. The README was the older
of the two — multiple references to pre-Campaign-2 state.

### README.md (substantive refresh)

Stale items found and fixed:

1. **Tool count** — said "6 OpenAPI endpoints" listing
   `web_search, kb_search, kb_image_search, memory_store, memory_recall,
   memory_search`. Now 7, with `chat_history_search` added (Campaign 2
   Phase 1). Refreshed the list and added a parenthetical mention of the
   per-user chat archive on the Qdrant line.
2. **Pipeline shape diagram** —
   - Complexity gate description now reflects the actual ordering
     (`OWUI-task detect → virtual-model force → token threshold`) that
     shipped in Campaign 2 Phase 6a, not just "token threshold."
   - Added the `planner` node on the deep branch (was missing entirely).
   - Streaming-banner list now includes `Planning` (Campaign 2 Phase 7
     renamed the deep helper's first stage from `Thinking` → `Planning`
     so users can tell at a glance which branch ran).
3. **KB watcher block** — added "+ reconcile (periodic orphan sweep,
   startup + 30 min cadence)" to surface the reconcile mechanism the
   Phase 10 deploy doc just formalized.
4. **Per-user chat archive sentence** added under the pipeline section so
   readers know the SQLite + Qdrant chat capture exists and powers
   `chat_history_search`.
5. **Documentation paths overhaul** — replaced the old `docs/phase-N-deploy.md`
   path (which no longer exists; campaigns now live under
   `docs/campaign-1/` and `docs/campaign-2/`) with a curated section
   pointing at the right entry points:
   - `AGENTS.md` first (the canonical agent guide)
   - `docs/lessons/` (codebase walk-through, with Lesson 4 highlighted)
   - Both campaign histories
   - `docs/guides/unraid-ollama.md`
   - `monitoring/README.md`
6. **Test-count statement** — said "(110 tests, ~1s)". Removed the count
   entirely; it was stale by a factor of 3 and the README isn't the right
   place to track that number anyway.
7. **CONTINUITY.md reference** — pointed at `CONTINUITY.md (gitignored,
   dev-machine only)`, but that file moved to `docs/PROJECT_STATE.md` per
   AGENTS.md. Reference updated.
8. **Phase status footer** — said "Phases 1 → 31 verified." Rewrote to
   reflect both campaigns: Campaign 1 (phases 1-31) shipped the core
   orchestrator; Campaign 2 is post-1.0 feature work. Removed the hard
   number since Campaign 2 is still active.
9. **Monitoring repo-layout line** updated to mention "provisioned
   dashboards" reflecting Phase 9.
10. **Docker repo-layout line** updated to note that base images are
    pinned to digest (so the layout line tells the reader the
    reproducibility story without needing to open the Dockerfiles).

Canonical-architecture-doc question (the squishy ask in the original spec):
decided **no new doc**. The right pair is already AGENTS.md (working in
this repo) + `docs/lessons/lesson-04-request-lifecycle.md` (how a request
flows). The README now points at both.

### main.py FastAPI app description

Smaller pass. The `description=` string passed to `FastAPI(...)` surfaces
on the OpenAPI spec page. Three additions:

1. Added `planner` to the pipeline-shape one-liner so the deep branch is
   accurately summarized.
2. Added `per-user chat-history search` to the tool-dispatch list
   (Campaign 2 Phase 1 capability).
3. Added "KB watcher + periodic reconcile keeping global collections
   drift-free" between the auth line and the streaming line.

Nothing in the old description was strictly wrong; it was just incomplete
relative to Campaign 2 features. Anyone hitting the OpenAPI page now sees
an accurate one-paragraph project summary.

## Closure verification — *pending*

This phase ships entirely in the repo. Verification is:

- Full test suite passes (no code-path behavior changed, but a sanity
  check that the FastAPI description edit didn't break import).
- Ruff clean on touched files.
- The FastAPI `description=` edit doesn't break the app boot — verify
  on Unraid that `docker compose up -d --build audrey-ai` produces a
  running container that responds to `/health`.

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai     # required: FastAPI description
                                           # is baked into the image
```

`custom-tools` doesn't need rebuilding — no changes touch its surface.
`monitoring/` likewise.

## 2. Smoke tests

### 2.1 Audrey boots

```bash
docker compose logs audrey-ai --tail=50 | grep ready
```

Expect the usual `ready: ollama=...; task types=...; ...` line. No new
config fields, so the boot log looks identical to pre-Phase-11.

### 2.2 OpenAPI description updated

```bash
curl -sS http://192.168.1.11:8000/openapi.json | jq -r '.info.description'
```

Expect the new description with `planner` in the pipeline string,
`per-user chat-history search` in the tool list, and the
`KB watcher + periodic reconcile keeping global collections drift-free`
clause.

### 2.3 README renders on GitHub

After pushing: <https://github.com/robocoppa/audrey_ai_2.0> should show
the refreshed README with the 7-tool list, the Planning banner mention,
the documentation section pointing at AGENTS.md and lessons, and no
"110 tests" or stale `phase-N-deploy.md` path.

## 3. Rollback

`git revert <commit>` restores the prior README, FastAPI description,
and (this) deploy doc. Then `docker compose up -d --build audrey-ai`
to revert the running container's OpenAPI description. No data
involved; nothing on the host filesystem changes.

## 4. Operational notes

- **Digest bump procedure** stays unchanged from Phase 31. Comments in
  both Dockerfiles + `monitoring/compose.yaml` document it. To pull
  newer upstream Prometheus or Grafana, run the commented bump
  procedure, replace the digest, commit, deploy.
- **README-vs-AGENTS.md split.** README is for "what is this and how
  do I get started." AGENTS.md is for "how to work in this repo as
  an agent or new contributor." Keep new project-shape facts in
  AGENTS.md; keep the README's "high-level architecture" diagram in
  sync as the system shape changes (the diagram is the only place in
  the README that drifts mechanically; everything else is stable
  conceptual content).

## 5. Followups

Two cross-campaign items remain on the queue after Phase 11. Both are
defer-until-evidence:

- **Per-task synth prompt variants** (Phase 25 followup). All five
  virtual models share `SYNTH_SYSTEM`. Split per pool_key if users
  complain about specific output styles.
- **Phase 19 reflect-on-stream.** Buffer first N synth tokens, run
  reflect against the partial draft, restart if needed. Only worth
  doing if too-short streamed answers become a real complaint.

Neither is urgent. Phase 11 effectively closes the active
cross-campaign hygiene backlog.
