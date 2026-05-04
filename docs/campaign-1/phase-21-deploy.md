# Phase 21 — Dockerfile installs from pyproject.toml

**Goal:** stop the hardcoded-deps-list footgun. Adding a runtime dep to
Audrey is now a single edit to `pyproject.toml` — `docker/audrey.Dockerfile`
no longer needs to track the same list separately.

Why: this footgun bit twice in 2026.
- **Phase 15** — added `aiosqlite` to `pyproject.toml` for the per-user
  uploads sqlite index. Container crash-looped on first deploy with
  `ModuleNotFoundError: aiosqlite` because the Dockerfile's hardcoded
  `uv pip install` list didn't include it.
- **Phase 17** — same thing with `prometheus-client` for the `/metrics`
  endpoint. Forgot to update the Dockerfile, container hit the same
  ImportError on `from prometheus_client import …`.

What changed:

- **`docker/audrey.Dockerfile`** — the long `RUN uv pip install --system
  "fastapi>=…" …` block is replaced with `RUN uv pip install --system .`
  reading deps from `pyproject.toml`. The package itself is wheel-built
  by hatchling, then symlinked at `/app/audrey` so the existing
  `PYTHONPATH=/app + import audrey` runtime continues to resolve the
  live `/app/src/audrey/...` source (PYTHONPATH wins over site-packages,
  so runtime hot-edits in the bind mount still work).
- **`docker/custom-tools.Dockerfile`** — *not* converted. Tools-server
  has a flat-script layout (`app.py`/`brave.py`/`db.py`/`settings.py`,
  no `src/<pkg>/`) which hatchling can't wheel-build cleanly. A comment
  was added pointing this out so future-you adds new deps to both the
  Dockerfile *and* `tools-server/pyproject.toml`.

What stays the same:

- Runtime layout: `import audrey` resolves `/app/audrey/*` (which is now
  a symlink to `/app/src/audrey/*`). Bind mounts at `/app/audrey/...`
  still work for hot-reload during dev (if you enable uvicorn `--reload`).
- All apt-level system packages (build-essential, libxml2, libxslt1.1,
  libmagic1, curl) — those still come from the apt block.
- `PYTHONPATH=/app` env var, `EXPOSE 8000`, `HEALTHCHECK`, `CMD`.

Out of scope (deliberately):

- **Custom-tools conversion.** Awkward without restructuring tools-server
  into a proper package. Discrepancy documented in the Dockerfile
  comment so the next person adding a dep notices.
- **Eliminating the layer-cache cost.** Today's install layer invalidates
  on any source change because hatchling needs `src/audrey` to build the
  wheel. A two-step approach (install deps → install package
  separately) would preserve cache but adds complexity. Not worth it
  for a Dockerfile rebuilt a few times a week.
- **uv.lock checked in.** Could pin transitive deps via `uv lock` and
  install from the lockfile in the Dockerfile. Maybe later — not load-
  bearing yet.

**Prereqs:** none. No env vars, no migrations.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

The first build with the new Dockerfile will not have a cached install
layer (everything's new), so expect ~2-3 minutes for the install step.
Subsequent rebuilds with the same `pyproject.toml` *and* unchanged
source will be cache-fast; rebuilds where source changed will re-run
the install (~30s). That's the documented tradeoff.

---

## 2. Smoke tests

### 2.1 Container starts cleanly

```bash
docker compose logs --tail 20 audrey-ai | grep -E 'ready|ERROR|Traceback'
```

Expect: one `ready: ollama=...; gpu_concurrency=1; max_inflight_per_user=3; ...`
line, no errors.

### 2.2 All deps actually installed

```bash
docker exec audrey-ai python -c "
import fastapi, uvicorn, httpx, pydantic, pydantic_settings, yaml, tenacity, tiktoken
import langgraph, langchain_core, dotenv, qdrant_client, sentence_transformers
import pypdf, docx, bs4, lxml, watchdog, PIL, magic, aiosqlite, markdown
import prometheus_client
print('all imports ok')
"
```

Expect: `all imports ok`. If any module is missing, the pyproject /
Dockerfile install line is wrong.

### 2.3 Audrey's own package importable

```bash
docker exec audrey-ai python -c "from audrey import __version__; print(__version__)"
```

Expect: `7.0.0` (or whatever the pyproject `version` is).

### 2.4 The symlink exists and resolves

```bash
docker exec audrey-ai ls -la /app/audrey
```

Expect: `/app/audrey -> /app/src/audrey`. If it's a regular directory,
the symlink step in the RUN didn't fire — usually means a previous
build artifact wasn't cleaned. `docker compose build --no-cache audrey-ai`
clears it.

### 2.5 audrey-ingest console script lands in PATH

The pre-Phase-21 Dockerfile didn't `pip install` the package, so the
`[project.scripts] audrey-ingest = "audrey.kb.cli:main"` console script
never landed in `$PATH`. Phase 8 documented the workaround as
`python3 -m audrey.kb.cli`. Phase 21 should fix this:

```bash
docker exec audrey-ai which audrey-ingest
```

Expect: `/usr/local/bin/audrey-ingest`. If it's not there, the wheel
build skipped the entry-point step — check `pyproject.toml` `[project.scripts]`
section.

### 2.6 An actual chat completion still works

End-to-end sanity:

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"one sentence on rsync"}]}' \
  http://192.168.1.11:8000/v1/chat/completions | jq -r '.choices[0].message.content'
```

Expect: an answer. Phase 21 should be invisible at runtime.

---

## 3. Rollback

```bash
git checkout <previous-sha> -- docker/audrey.Dockerfile docker/custom-tools.Dockerfile
docker compose up -d --build audrey-ai
```

Single-file revert. The pyproject.toml didn't change (it already had
all the deps), so no other rollback needed.

---

## 4. Adding a new dep going forward

1. Edit `pyproject.toml`:
   ```toml
   dependencies = [
       …existing…
       "newpkg>=1.2",
   ]
   ```
2. `docker compose up -d --build audrey-ai`. The install layer
   invalidates and runs `uv pip install --system .` against the new
   pyproject — `newpkg` lands.
3. **For tools-server** (custom-tools): you still need to edit *both*
   `tools-server/pyproject.toml` *and* `docker/custom-tools.Dockerfile`'s
   hardcoded list. The Dockerfile has a comment reminding you. This
   asymmetry is the cost of tools-server's flat-script layout.
