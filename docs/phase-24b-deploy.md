# Phase 24b — split Dockerfile install into deps layer + package layer

**Goal:** stop re-resolving the entire dep tree on every source change.
Today, editing one `.py` file under `src/audrey/` invalidates the
single big install layer in `docker/audrey.Dockerfile`, which then
re-runs `uv pip install --system .` against `pyproject.toml`. That
takes ~30-60s warm, ~3min cold, AND produces a fresh ~10 GB layer in
the Docker build cache. Over a couple weeks of dev rebuilds, this
accumulated to 101 GB of build cache (cleaned up 2026-04-29).

The fix: split the install into two layers with separate cache keys.

```
Layer 1 (deps):     cache key = pyproject.toml content
Layer 2 (package):  cache key = src/audrey/ + README.md
```

After this change:
- **Source edit** → only layer 2 reruns. ~5-10s, ~MB-scale layer.
- **`pyproject.toml` edit** → both layers rerun. Same ~30-60s as today
  (no regression on the rare path).
- **README/config edit** → only layer 2 reruns. ~5-10s.

What stays the same:

- Every Python dep installed in the final image — same `pyproject.toml`,
  same `uv pip` resolver, just split across two `RUN`s.
- Final image size (~10.8 GB).
- Runtime layout: `PYTHONPATH=/app` + symlink at `/app/audrey →
  /app/src/audrey`. App code is untouched; this is purely a build-time
  refactor.
- The Phase 21 win — adding a runtime dep is still a single edit to
  `pyproject.toml`, no Dockerfile change.

What changed:

- **`docker/audrey.Dockerfile`** — single install step replaced with two:
  1. `COPY pyproject.toml` + `uv pip compile -o /tmp/requirements.txt`
     + `uv pip install -r /tmp/requirements.txt`. README is *not*
     copied here — `compile` reads pyproject directly and doesn't
     need the wheel-build inputs.
  2. `COPY README.md` + `COPY src/audrey` + `uv pip install --no-deps
     /app`. Hatchling builds the audrey wheel (needs README for
     `readme = "README.md"` metadata) but uv skips dep resolution
     because deps are already installed from layer 1.
- Comment block updated to explain the two-layer rationale.

Out of scope (deliberately):

- **`uv.lock` checkin.** Would let us swap layer 1 to `uv sync --frozen`
  for fully-pinned transitive deps. Hygiene win, but introduces a
  discipline (must `uv lock` after editing deps) and a few-hundred-KB
  checked-in file. Defer until we have a reason.
- **`docker/custom-tools.Dockerfile`** — still has the flat-script
  layout that hatchling can't wheel-build cleanly (Phase 21 footnote).
  Separate followup.
- **Multi-stage build with `--from=builder`.** Would shave the
  `build-essential` apt deps from the final image, but the savings
  (~200 MB) aren't worth the layout churn given the image is already
  10+ GB of Python deps.

**Prereqs:** Phase 21 verified (the existing `uv pip install --system .`
step works). Phase 24/24a/25 all verified. No env vars, no migrations,
no schema changes.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

**First rebuild after Phase 24b is a cache miss for both layers** —
the new layer-1 cache key (just `pyproject.toml`) doesn't match
anything from prior builds (which keyed on the full `COPY . /app`
context). Expect ~2-3 minutes for this one rebuild. **Subsequent
rebuilds with unchanged pyproject** will hit the layer-1 cache
immediately and only re-run layer 2.

If you want to verify the build output explicitly, watch for
`CACHED [audrey-ai stage-0 ...] RUN uv pip compile`:

```bash
docker compose build audrey-ai 2>&1 | grep -E 'CACHED|RUN.*uv pip'
```

On the **first** rebuild, both `RUN` lines run. On the **second**
rebuild (without changing pyproject), the first `RUN` shows `CACHED`.

---

## 2. Smoke tests

### 2.1 Container starts cleanly

```bash
docker compose logs --tail 30 audrey-ai | grep -E 'ready|ERROR|Traceback'
```

Expect: one `ready: ollama=...; pipeline=compiled` line, no errors.

If you see `ModuleNotFoundError` for any dep at import time, layer 1
under-resolved — `uv pip compile` didn't include something the package
needs. Check the compile output (run it locally to inspect):
```bash
uv pip compile pyproject.toml
```
Compare against `pyproject.toml`'s `dependencies = [...]` block.

### 2.2 All deps installed

```bash
docker exec audrey-ai python -c "
import fastapi, uvicorn, httpx, pydantic, pydantic_settings, yaml, tenacity, tiktoken
import langgraph, langchain_core, dotenv, qdrant_client, sentence_transformers
import pypdf, docx, bs4, lxml, watchdog, PIL, magic, aiosqlite, markdown
import prometheus_client
print('all imports ok')
"
```

Expect: `all imports ok`. Same set of imports as Phase 21's smoke
test 2.2 — if anything is missing it's a layer-1 regression, not
something Phase 24b is supposed to change.

### 2.3 Audrey package itself importable

```bash
docker exec audrey-ai python -c "from audrey import __version__; print(__version__)"
```

Expect: `7.0.0` (or whatever pyproject's `version` says). Confirms
layer 2's `--no-deps` install of the audrey wheel succeeded.

### 2.4 Symlink at /app/audrey resolves correctly

```bash
docker exec audrey-ai ls -la /app/audrey
```

Expect: `/app/audrey -> /app/src/audrey`. The Phase-24b layer-2 RUN
uses `ln -sf` (vs Phase 21's `ln -s`) so a re-run on top of an existing
image overwrites the symlink instead of erroring.

### 2.5 End-to-end chat works

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"two-sentence intro to rsync"}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content'
```

Expect: a real answer. Phase 24b is a Dockerfile refactor — runtime
behavior is identical to pre-Phase-24b. If this fails, layer 1
under-installed something or layer 2's wheel build malformed the
package install.

### 2.6 The headline test — source edit is fast

This is where the win shows up. Touch a non-impactful Python file and
rebuild:

```bash
# Small edit to a comment in any Python file
echo "# Phase 24b cache test" >> /mnt/user/appdata/audrey_ai_2.0/src/audrey/__init__.py

# Time the rebuild
time docker compose build audrey-ai 2>&1 | tail -20
```

Expected build output should include:
```
CACHED [audrey-ai stage-0 6/9] COPY pyproject.toml /app/pyproject.toml
CACHED [audrey-ai stage-0 7/9] RUN uv pip compile ...
RUN [audrey-ai stage-0 8/9] COPY README.md ...      ← rerunning
RUN [audrey-ai stage-0 9/9] COPY src/audrey ...     ← rerunning
RUN [audrey-ai stage-0 10/9] RUN uv pip install --no-deps ...  ← rerunning
```

`time` output should be ~10-15s (vs pre-Phase-24b's ~30-60s).

Revert the edit and rebuild once more to leave things clean:

```bash
git checkout /mnt/user/appdata/audrey_ai_2.0/src/audrey/__init__.py
docker compose build audrey-ai 2>&1 | tail -5
```

This rebuild should be **fully cached** (both layers hit) and complete
in <2s — both layer 1 (pyproject unchanged) and layer 2 (source back
to checked-in state) match prior cache.

### 2.7 Pyproject change correctly invalidates layer 1

To prove the cache invalidation triggers when it should, add a
no-op comment to pyproject.toml:

```bash
# (Be careful: pyproject.toml IS used by the runtime, so don't break it.)
# Pick a line in [tool.hatch] or [build-system] that doesn't affect installs:
echo "# 24b cache test" >> /mnt/user/appdata/audrey_ai_2.0/pyproject.toml

time docker compose build audrey-ai 2>&1 | tail -10
```

Expect: layer 1 reruns (no `CACHED` for the `uv pip compile` line),
~30-60s wall-clock. This is the path where Phase 24b matches today's
behavior — no regression, just no improvement either.

Revert:
```bash
git checkout /mnt/user/appdata/audrey_ai_2.0/pyproject.toml
```

---

## 3. Rollback

```bash
git checkout <previous-sha> -- docker/audrey.Dockerfile
docker compose up -d --build audrey-ai
```

One-file revert. Old single-install pattern rebuilds from scratch
(~3 min cold), same as it always did. No data, config, or runtime
state to migrate.

---

## 4. Operational notes

### Build-cache growth is now slower, not zero

Source edits still produce small layer-2 deltas in build cache (the
new audrey wheel ~10 MB + the COPY of `src/audrey/`). Over many
edits these still add up — just at MB-scale instead of GB-scale.
Phase 23's reclaim (101 GB from ~10 rebuilds) won't recur at the same
rate, but `docker builder prune -af` is still good hygiene every few
weeks.

### Adding a new runtime dep

Same flow as Phase 21:

1. Edit `pyproject.toml`'s `dependencies = [...]` block.
2. `docker compose up -d --build audrey-ai`.
3. First rebuild after the dep change runs both layers (~30s-3min).
4. Subsequent source edits hit the layer-1 cache again.

### When to convert to `uv.lock`

If we ever want fully-pinned transitive deps (reproducible builds
across team members, CI, etc.), the path is:

```bash
# Locally, in the repo root:
uv lock
git add uv.lock
```

Then in the Dockerfile, replace layer 1 with:

```dockerfile
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project
```

And layer 2 with:

```dockerfile
COPY README.md src/audrey ./
RUN uv sync --frozen
```

That'd be Phase 24c. Defer until we have a reason — solo dev today
doesn't need cross-machine reproducibility.

### When uv pip compile output diverges from uv pip install .

`uv pip compile pyproject.toml` resolves the same dependency tree as
`uv pip install .`, just emits it as a flat list instead of installing.
If you ever observe a runtime `ModuleNotFoundError` for a dep that
should be there, run both locally and `diff` to spot the divergence:

```bash
uv pip compile pyproject.toml > /tmp/compiled.txt
# vs
uv pip install --dry-run . 2>&1 | grep -E '^[a-z]' > /tmp/installed.txt
diff /tmp/compiled.txt /tmp/installed.txt
```

This shouldn't happen — uv's resolver is consistent across both paths
— but documented here for future debugging.
