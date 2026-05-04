# Phase 31 — Image digest pinning + docs cleanup

Audit-low cleanup phase. Two unrelated quality-of-life pieces bundled
because both are documentation-adjacent and no-risk.

**A. Pin base images to sha256 digests.** Pre-Phase-31, `python:3.12-slim`,
`ghcr.io/astral-sh/uv:latest`, `prom/prometheus:latest`, and
`grafana/grafana:latest` were unpinned. A rebuild on a different day
could pick up a base image with a CVE, a breaking change, or a subtle
behavioral difference — and we'd never know until something failed.
Phase 31 pins them to specific digests with a comment showing what
human-readable tag they pointed to on pin date.

**B. Refresh README + FastAPI app description + docs/README.** The
README still said "three virtual models" and "5 OpenAPI endpoints" —
both wrong since Phase 18 (5 virtual models) and Phase 11/12 (6
tools). FastAPI app description still said `Phase 4 build is a
pass-through; routing/panels/tools land in later phases.` The
docs/README.md indexed four files that don't exist. All fixed.

What stays the same:
- Runtime behavior. Pinning doesn't change what code runs; it just
  freezes the *how* so future builds are reproducible.
- All five virtual models, all six tools, every route, every metric.
- Phase doc filenames (`phase-N-deploy.md`) — historical record,
  unchanged.

What changed:
- **`docker/audrey.Dockerfile`** — `FROM python:3.12-slim` →
  `FROM python:3.12-slim@sha256:46cb...`. `COPY --from=ghcr.io/astral-sh/uv:latest`
  → `COPY --from=ghcr.io/astral-sh/uv@sha256:3b7b...`. Both with bump-procedure
  comments inline.
- **`docker/custom-tools.Dockerfile`** — same two pins. Same digests as
  audrey-ai's Dockerfile (both base off `python:3.12-slim` from the
  same registry pull). Comment notes "keep in sync when bumping."
- **`monitoring/compose.yaml`** — `prom/prometheus:latest` and
  `grafana/grafana:latest` pinned to digests. Bump procedure inline.
- **`src/audrey/main.py`** — FastAPI `description=` rewritten to
  describe current state (5 virtual models, fair scheduling, OWUI auth,
  metrics, banners + tools-used footer).
- **`README.md`** — full refresh. New architecture diagram (6 tools,
  KB watcher + reconcile, Prometheus/Grafana). 5-virtual-model table.
  Pipeline shape laid out. Dropped 4 dead doc-link references.
- **`docs/README.md`** — categorized by feature area (Foundation, KB +
  memory, Auth, Observability, Routing, Build, UX, Tests) rather than
  pretending phase numbers are navigation.

The four pinned digests as of 2026-05-02:

| Image                              | Digest                                                                  |
|------------------------------------|-------------------------------------------------------------------------|
| `python:3.12-slim`                 | `sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3` |
| `ghcr.io/astral-sh/uv:latest`      | `sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347` |
| `prom/prometheus:latest`           | `sha256:e4254400b85610324913f0dc4acf92603d9984e7519414c5a12811aa6146acc3` |
| `grafana/grafana:latest`           | `sha256:0f86bada30d65ef9d0183b90c1e2682ac92d53d95da8bed322b984ea78a4a73a` |

Out of scope (deliberately):

- **No automated digest-refresh tooling.** Renovate / dependabot is
  real work; manual refresh on intentional bumps is fine for our scale.
  The bump procedure is one `docker pull` + one `docker inspect` line,
  documented inline in each Dockerfile.
- **No tools-server pyproject conversion.** Separate followup. Tools-
  server still has its hardcoded dep list — adding a tool dep means
  editing both `tools-server/pyproject.toml` AND
  `docker/custom-tools.Dockerfile`. Pre-Phase-31 reality, unchanged.
- **No new "current architecture" doc.** README + phase-N-deploy
  history is enough. CONTINUITY is the working state for the dev
  machine; users get README.

**Prereqs:** all phases through 30 verified. No env vars, no compose
schema changes, no model registry changes.

---

## 1. Deploy

```bash
# Laptop:
git pull   # after the user has committed Phase 31

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build audrey-ai custom-tools

# Monitoring stack (separate compose):
cd monitoring
docker compose up -d
```

The audrey-ai + custom-tools build will pull `python:3.12-slim` and
`ghcr.io/astral-sh/uv` by digest — first build downloads, subsequent
ones use the cached image at the same digest. Same pinning for the
monitoring stack.

---

## 2. Smoke tests

### 2.1 Verify pinned digests appear in the build trace

```bash
docker compose build --no-cache audrey-ai 2>&1 | grep -iE "FROM python|Pulling.*python"
```

Expected: a line like
`FROM docker.io/library/python:3.12-slim@sha256:46cb7cc2877e60fb...`
in the trace. If you see `Using default tag: latest` or any line
referencing `python:3.12-slim` *without* a `@sha256:` suffix, an
unpinned reference slipped in — search the Dockerfile for the bare
tag.

Same check for the uv pin:

```bash
docker compose build --no-cache audrey-ai 2>&1 | grep -iE "ghcr.io/astral-sh/uv|Pulling.*uv"
```

Expected: a line referencing
`ghcr.io/astral-sh/uv@sha256:3b7b60a81d3c57ef...`. The `COPY --from=`
pattern doesn't print as cleanly as `FROM`, so you may need to
search for `astral-sh` instead.

### 2.2 What pinning does NOT guarantee

Pinning gives you **same base layer**, not **byte-identical builds**.
Even with pinned `FROM` digests, two `docker build --no-cache` runs
will produce different final image IDs because:

- `apt-get update` pulls the package index *at build time*.
- pip writes install timestamps into wheel metadata.
- BuildKit records `RUN` step timestamps in layer metadata.

That's fine. The point of pinning isn't byte-identical builds — it's
defense against a registry tag silently advancing under us. A CVE
patched in `python:3.12-slim` between yesterday and tomorrow stays out
of our build until we explicitly bump the digest.

If you want to verify the *base layer* is stable across builds (the
useful part), compare the first few entries of `RootFS.Layers`:

```bash
docker compose build --no-cache audrey-ai
docker inspect audrey-ai:latest --format='{{range .RootFS.Layers}}{{println .}}{{end}}' > /tmp/layers-1
docker compose build --no-cache audrey-ai
docker inspect audrey-ai:latest --format='{{range .RootFS.Layers}}{{println .}}{{end}}' > /tmp/layers-2
diff /tmp/layers-1 /tmp/layers-2 | head
```

Expected: the first ~5 layer SHAs (Python base + apt install) match
between builds. Layers further down (uv install, package install,
config copy) may differ for the timestamp reasons above.

### 2.3 Monitoring stack still scrapes

```bash
cd monitoring
docker compose up -d
sleep 5
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health}'
```

Expected: at least the audrey-ai scrape target as `health: "up"`. If
prometheus or grafana fails to start, check the digest is reachable
from your registry mirror.

### 2.4 Grafana renders a known panel

```bash
# In a browser: http://<unraid-ip>:3000
# Log in with the env-required password (Phase 26).
# Load the audrey dashboard, confirm panels show data from the last 1h.
```

Expected: same dashboards, same panels, same queries — pinning doesn't
touch dashboard JSON.

### 2.5 Tests + ruff still green

```bash
# On the laptop:
.venv/bin/pytest tests/ -q
# Expected: 110 passed in <2s

.venv/bin/ruff check src/audrey/ tests/ --statistics | tail -5
# Expected: 9 errors, all ASYNC240 (the accepted category from Phase 29)
```

### 2.6 README accurately reflects the build

Spot check:
- Does the model table show 5 entries? (audrey_auto, audrey_fast,
  audrey_deep, audrey_cloud, audrey_local)
- Does the architecture diagram mention 6 tools?
- Are there any links to docs that don't exist (`unraid-audrey.md`,
  `cloudflared-routing.md`, `kb-geology.md`, `future-tools.md`)?
  Should be none — they were all dead, all dropped.

---

## 3. Rollback

Revert the four pin commits (or unpin in place). To unpin:

```dockerfile
# Replace pinned form:
FROM python:3.12-slim@sha256:46cb...

# With floating tag:
FROM python:3.12-slim
```

Same for the other three. README + FastAPI description rollback is a
plain `git revert` of those file edits; production behavior is
unchanged either way.

---

## 4. Operational notes

- **Bump procedure (per image, when needed):**
  ```bash
  docker pull <image>:<tag>
  docker inspect --format='{{index .RepoDigests 0}}' <image>:<tag>
  # Copy the printed sha256 into the Dockerfile / compose.yaml
  # Update the date in the inline comment
  # Rebuild + redeploy
  ```
- **Two Dockerfiles share one Python digest.** When bumping
  `python:3.12-slim`, update both `docker/audrey.Dockerfile` AND
  `docker/custom-tools.Dockerfile` in the same change. The inline
  comment in custom-tools.Dockerfile says "keep in sync."
- **uv digest changes more frequently than Python's.** Bumping uv
  alone is fine — it's a single binary copy, low blast radius. If a
  uv release breaks the wheel build, the symptom is layer 1 of
  audrey.Dockerfile failing at `uv pip compile`.
- **Monitoring digest bumps are rarer.** `prom/prometheus` and
  `grafana/grafana` are stable-tagged; bump only when you have a
  specific reason (CVE, feature you want).
- **The README is a public-facing artifact.** It lives in git (unlike
  CONTINUITY.md, which is gitignored). Keep it accurate for anyone
  landing on the repo. Any future phase that meaningfully changes
  the architecture diagram, model count, or documented feature list
  should bump README in the same commit.
- **`docs/README.md` is the navigation aid.** Phase docs are
  historical — written when the feature shipped. The categorized
  index is the best way to find "where is the doc about auth" or
  "which phase added the KB watcher."
