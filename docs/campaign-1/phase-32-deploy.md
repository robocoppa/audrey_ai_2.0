# Phase 32 — Lesson 5 audit cleanup (config + startup)

Hygiene phase, opened by the lesson-5 audit pass on `main.py`,
`config.py`, and `compose.yaml`. Six findings landed in
`docs/lessons/AUDIT.md`; this phase resolves five of them and accepts
the sixth with documentation. One real bug, the rest is consistency
and dead-code removal.

**The one real bug:** `EnvOverrides.tool_servers` and
`kb_dataset_paths` had non-`None` string defaults
(`"http://custom-tools:8001"` and `"/datasets/geology"`), and
`_apply_env_overrides` wrote them into the merged YAML
*unconditionally* — without the `is not None` guard the other fields
have. Result: the YAML's `tools.servers` list and the 13-topic
`kb.dataset_paths` list were silently clobbered on every boot, even
when the env vars were unset. The actual topic list came from
`compose.yaml`'s `${KB_DATASET_PATHS:-/datasets/geology}` fallback,
not from the YAML the operator was editing.

In production this was masked because `.env` typically sets
`KB_DATASET_PATHS` explicitly, but a fresh deployment without that
env var would silently boot with `/datasets/geology` only and a
reader who edited `config.yaml`'s 13-topic list would see no effect.
Fixed by switching the two fields to `str | None`, gating the merge
on `is not None`, and dropping the `compose.yaml` fallbacks for those
two vars so the YAML wins by default.

What stays the same:
- Runtime behavior in the existing deployment is unchanged — the
  `.env` on Unraid sets `KB_DATASET_PATHS` and `TOOL_SERVERS`
  explicitly, so the env-wins path still produces the same merged
  config.
- All five virtual models, every route, every metric.
- Phase doc filenames, model registry, deep-panel pools, fairness
  knobs, KB shape.

What changed:
- **`src/audrey/config.py`** —
  - `tool_servers` and `kb_dataset_paths` typed `str | None`,
    `default=None`. Their entries in `_apply_env_overrides` now use
    the same `if (v := …) is not None:` guard as every other tunable.
  - `data_dir` field deleted. No callers; the `/data/...` paths it
    was supposed to govern are hardcoded at the call sites
    (e.g. `kb.uploads_db_path`).
  - `get_env()` deleted and removed from `__all__`. No callers.
  - `kb_watcher_enabled: bool = Field(default=False, alias="KB_WATCHER_ENABLED")`
    added. Pydantic Settings handles `1`/`true`/`yes` parsing
    natively, so the bool is honest.
  - `Config._yaml` renamed → `Config._merged`. Private attribute, no
    external readers; pure rename to make "this is the post-override
    state" obvious from the name.
  - One-line comment on `_load_yaml` documenting the deliberate
    fast-fail-on-bad-config asymmetry vs. qdrant's degrade-on-boot
    path.
- **`src/audrey/main.py`** —
  - `if cfg.env.kb_watcher_enabled:` instead of
    `if os.environ.get("KB_WATCHER_ENABLED", "").strip() in (...)`.
  - `import os` dropped (no other usages remained).
  - `from audrey.tools.discovery import ToolRegistry, discover_all`
    on the top-of-file import (was `discover_all` only, with a
    function-local import of `ToolRegistry` inside the lifespan
    `else` branch). The function-local import line deleted.
- **`compose.yaml`** —
  - Dropped `TOOL_SERVERS: ${TOOL_SERVERS:-http://custom-tools:8001}`
    and `KB_DATASET_PATHS: ${KB_DATASET_PATHS:-/datasets/geology}`.
    Those two env vars now flow through `env_file: - .env` only;
    when unset, the YAML's `tools.servers` and `kb.dataset_paths`
    are authoritative. Comment in their place explains the choice.
  - `OLLAMA_HOST`, `QDRANT_HOST`, `QDRANT_PORT`, `KB_WATCHER_ENABLED`
    fallbacks kept — those have no YAML counterpart, so the compose
    default is load-bearing.

Out of scope (deliberately):

- **`os.environ` direct reads elsewhere in the codebase.** The
  `EnvOverrides`-bypass smell is fixed for `KB_WATCHER_ENABLED`. If
  other modules read `os.environ` directly, that's a separate
  cleanup; not in this phase's scope.
- **Test coverage for `EnvOverrides` defaults.** No tests touch
  `tool_servers` / `kb_dataset_paths` / `kb_watcher_enabled`; we're
  shipping the change without adding coverage. The behavior is small
  and self-evident from the diff. If a regression slips in we add
  the test then.
- **Wiring `data_dir` back up.** Deleting was the cheaper fix. If
  we ever want a single knob for "audrey's data root," we can
  reintroduce the field with actual readers.

## Verification on Unraid

The bug-fix changes runtime behavior only when `KB_DATASET_PATHS` or
`TOOL_SERVERS` are unset. Existing `.env` on Unraid sets at least
`KB_DATASET_PATHS=...` (per the historical compose default), so a
direct rebuild + restart should produce identical merged config.

```
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai | head -40
```

Look for the readiness log line:

```
ready: ollama=...; task types=...; gpu_concurrency=...;
  max_inflight_per_user=...; tools=N (...); qdrant=...;
  kb_watcher=on; kb_reconcile=on; pipeline=compiled
```

Then confirm the topic list matches expectation:

```
docker exec audrey-ai python -c "
from audrey.config import get_config
cfg = get_config()
import json
print(json.dumps(cfg.raw['kb']['dataset_paths'], indent=2))
print('tools.servers:', cfg.tools.get('servers'))
print('kb_watcher_enabled:', cfg.env.kb_watcher_enabled)
"
```

Expected: 13 topic paths, one tool server URL, `True`.

## Rollback

Revert the config.py + main.py + compose.yaml edits via git. No
schema changes, no migrations, no on-disk state involved. Restart
the audrey-ai container.

## Findings status after this phase

- `bug` — env defaults overwrite YAML lists → **resolved**.
- `smell` — `data_dir` dead → **resolved** (deleted).
- `smell` — `KB_WATCHER_ENABLED` bypasses EnvOverrides → **resolved**.
- `nit` — `get_env()` dead → **resolved** (deleted).
- `consider` — `_yaml` mutated in place → **resolved** (renamed to
  `_merged`).
- `consider` — config-load fast-fail vs. qdrant-degrade asymmetry →
  **accepted** (annotated with a one-line comment on `_load_yaml`).

Plus the previously-resolved `nit` from earlier the same day:

- `nit` — function-local `ToolRegistry` import → **resolved** (hoisted).

All Lesson 5 findings closed.
