# Campaign 2 Phase 5 - tools-server pyproject install + uv workspace

Build-system phase. No runtime behavior change. Two related cleanups:

1. **`docker/custom-tools.Dockerfile`** now installs from
   `tools-server/pyproject.toml` (same pattern as audrey-side from
   Phase 21) instead of a manually-duplicated `uv pip install ...`
   line plus a per-file `COPY` list. Adding a new dep is now a
   single edit; adding a new `.py` file is just `git add`.

2. **`uv.lock`** is now a workspace lockfile covering both `audrey`
   and `audrey-custom-tools`. The redundant `tools-server/uv.lock`
   was deleted.

The bugs this prevents are not theoretical: we hit both in this
campaign alone. Phase 1 missed `tools-server/chat_archive.py` in the
per-file `COPY` list and the container restart-looped on
`ModuleNotFoundError`. Phase 12 (in the original campaign) hit the
same shape with `qdrant-client` when only the pyproject got edited
and the Dockerfile's pinned list went stale.

What stays the same:

- Tools-server is still a flat layout (`app.py` / `brave.py` /
  `chat_archive.py` / `db.py` / `settings.py` at the directory root).
  No imports change. `app:app` is still the uvicorn entry point.
- All seven tool endpoints still discover the same way.
- SQLite + Qdrant data directories are untouched. No migration.
- The base Python image and uv digest pins stay the same as audrey-side.

What changed:

- **`tools-server/pyproject.toml`** -
  - Already declared `packages = ["."]` and `build-backend =
    "hatchling.build"`. Hatchling actually does support flat-layout
    wheels — the previous Dockerfile comment claiming it "can't
    cleanly wheel-build" was wrong. A `uv build --wheel` produces a
    working wheel with every `.py` at the wheel root, importable as
    `from brave import …` exactly like today.
  - `[tool.uv] package = false` stayed (this is a flag for `uv sync`,
    not the wheel build).
- **`docker/custom-tools.Dockerfile`** -
  - Replaced the manual `uv pip install --system "fastapi>=0.115" …`
    block with the two-layer pattern from audrey-side:
    - Layer 1 (deps): `COPY pyproject.toml` → `uv pip compile` →
      `uv pip install -r requirements.txt`. Cache key is
      `pyproject.toml`'s hash.
    - Layer 2 (package): `COPY tools-server/README.md` and
      `COPY tools-server/*.py` → `uv pip install --no-deps .`. Cache
      key is the source files' contents.
  - Source edits now produce ~MB-scale layers and ~5s rebuilds
    instead of re-running `uv pip install` for the full dep tree.
  - The per-file `COPY tools-server/foo.py /app/foo.py` lines
    collapsed into one `COPY tools-server/*.py /app/`.
- **`pyproject.toml`** (repo root) -
  - Added `[tool.uv.workspace] members = ["tools-server"]` so
    `uv lock` resolves both packages together. The combined lockfile
    keeps dep versions pinned consistently across the two services
    (helpful because both depend on `httpx`, `pydantic`,
    `qdrant-client`, etc. and drift between them would be subtle).
- **`uv.lock`** (repo root) -
  - Regenerated as the workspace lockfile. Now includes
    `audrey-custom-tools` as a member alongside `audrey`. Resolution
    counts went from ~119 packages (audrey alone) to 122 (workspace).
- **`tools-server/uv.lock`** -
  - Deleted. The workspace lockfile is the single source of truth.

Out of scope:

- Rewriting tools-server to a `src/audrey_custom_tools/` layout. That
  would standardize against audrey-side but break the existing flat
  imports and require renaming every `from brave import` → `from
  audrey_custom_tools.brave import`. Hatchling can wheel the flat
  layout fine; no reason to spend the rename time.
- Generating a `tools-server/uv.lock` symlink to root for tools that
  expect a per-directory lockfile. Nothing in our toolchain does.
- Pulling `tools-server/README.md` into the repo-root README. They
  stay separate.

## 1. Deploy

Local first (laptop):

```bash
git pull   # after the Phase 5 commit lands
.venv/bin/python -m pytest    # 241 should still pass
```

Unraid (from `/mnt/user/appdata/audrey_ai_2.0`):

```bash
git pull
docker compose up -d --build custom-tools
docker compose logs --since 2m custom-tools | grep -E "ready|chat_archive"
docker compose up -d --build audrey-ai   # triggers fresh tool discovery
docker compose logs --since 1m audrey-ai | grep "tools="
```

Expected:

- `custom-tools` builds without errors. Image size is roughly the
  same as before (±~5 MB).
- The `chat_archive: ready ...` + `custom-tools ready ...` log
  lines appear as on the previous deploy.
- The `ready: ... tools=7 ([..., chat_history_search, ...]); ...`
  readiness line in `audrey-ai` shows all seven tools rediscovered.

If `audrey-ai` shows `tools=0`, the boot-order retry will pick up
within ~2 minutes (per the Phase 1 fix). Same recovery as before.

## 2. Smoke tests

This is a build-system change with no runtime semantics. The pytest
suite catches what it always caught; the real test is whether the
container runs and discovers tools.

### 2.1 Verify the image installs correctly

```bash
docker compose exec custom-tools python -c "import app, brave, db, chat_archive, settings; print('all imports ok:', app.app.title)"
```

Expected: `all imports ok: Audrey custom-tools`. If any `import` fails,
the wheel layout is off — open `docker compose logs custom-tools` and
look for the failure stack.

### 2.2 Verify discovery works

```bash
curl -sS http://localhost:8001/openapi.json | python -c "import json, sys; print([op for path in json.load(sys.stdin)['paths'].values() for op in path.values() if 'operationId' in op for _ in [print(op['operationId'])]])" 2>&1 | head -20
```

Or simpler — the `/v1/tools` audrey route lists the live registry:

```bash
curl -sS http://localhost:8000/v1/tools | jq '.tools[].name'
```

Expected: seven tools — `web_search`, `kb_search`, `kb_image_search`,
`memory_store`, `memory_recall`, `memory_search`, `chat_history_search`.

### 2.3 Run Phase 4 Category 6 (the 3-minute end-to-end)

See [`phase-4-testing.md`](phase-4-testing.md). The three Category 6
prompts (trivial fast, trivial deep, tool dispatch under deep) exercise
the full pipeline. If they pass, the conversion is verified.

### 2.4 Verify the build cache split actually works

Optional but worth confirming, especially before the first source edit
after deploy:

```bash
# Edit a comment in tools-server/app.py (no behavior change), then:
docker compose build custom-tools
```

Expected: the deps layer is `CACHED`, only the package layer rebuilds.
Total build time should be ~10s, not ~30s.

If you see the deps layer rebuilding on a source-only edit, the
`pyproject.toml` was accidentally modified — confirm with
`git diff tools-server/pyproject.toml`.

## 3. Rollback

Plain git revert. No state, no data, no migration.

```bash
git revert <phase-5-commit>
docker compose up -d --build custom-tools
```

The previous Dockerfile rebuilds custom-tools the old way. Tools-server
data (SQLite + Qdrant) untouched.

## 4. Operational notes

- **Adding a new dep:** edit `tools-server/pyproject.toml`, then run
  `uv lock` from the repo root. The workspace will pick it up.
  Rebuilding custom-tools picks up the new dep automatically.
- **Adding a new tools-server `.py` file:** just create it. The
  Dockerfile's `COPY tools-server/*.py` glob takes care of it. No
  Dockerfile edit needed (this was the Phase 1 trap).
- **The `tools-server/uv.lock` is gone.** Don't re-create it. The
  workspace lockfile at the repo root covers both packages.
- **Image size:** the new image carries `pyproject.toml`,
  `README.md`, and the source files at `/app/`. Tiny — a few KB more
  than the old per-file copy.
- **Dep drift between audrey-ai and custom-tools:** because both
  depend on `httpx`, `pydantic`, `qdrant-client`, etc., the workspace
  forces them to agree on versions. If you ever need to pin different
  versions per service, that's an explicit `[tool.uv.sources]` entry
  per workspace member. Don't expect to need it.

## 5. Followups

- Drop the long comment in
  [`AGENTS.md`](../../AGENTS.md) about per-file `COPY` (no longer
  applies) — done as part of this phase, see also the
  `project_custom_tools_dockerfile` memory which should be marked
  resolved.
- A pre-commit hook that runs `uv lock --locked` to verify the
  workspace lockfile is in sync with both pyprojects. Cheap belt-and-
  suspenders against forgetting `uv lock` after a dep edit.
- The audrey-ai Dockerfile installs `build-essential` + various lib
  headers (lxml, pillow, libmagic). Tools-server doesn't — its deps
  are pure-python. If a future tools-server dep needs compilation,
  the apt-get block needs adding. Not today.
