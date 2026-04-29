# Phase 24a — rename `appdata/audrey/` → `appdata/runtime/`

**Goal:** disambiguate two directories that look related but aren't.
`/mnt/user/appdata/audrey/` is Audrey's runtime data (SQLite, caches,
upload bytes), bind-mounted as `/data` inside the container. The repo
itself lives at `/mnt/user/appdata/audrey_ai_2.0/`. The two names are
easy to confuse — especially when scanning `ls /mnt/user/appdata/`.

Renaming the runtime dir to `runtime/` makes the intent obvious. The
parent `audrey_ai_2.0/` already namespaces it; no need to repeat
"audrey" in the runtime dir name.

What changed (laptop, already in repo):

- `compose.yaml` — bind mount source path
  `/mnt/user/appdata/audrey:/data` → `/mnt/user/appdata/runtime:/data`.
- `docs/phase-4-deploy.md`, `docs/phase-13-deploy.md`,
  `docs/phase-15-deploy.md` — host-path references updated.
- `CONTINUITY.md` — appdata convention list + Stack state lines + the
  Phase 13 fact about uploads.sqlite location.

What stays the same:

- Container-internal path is still `/data`. App code never changes
  (it always writes to `/data/uploads.sqlite`,
  `/data/uploads/<sanitized_user>/...`, etc.).
- All other appdata directories (`ollama/`, `qdrant/`, `clip-cache/`,
  `custom-tools/`, `prometheus/`, `open-webui/`).
- TZ env, port mapping, network.

**Prereqs:** Phase 23 verified. No code changes — this is purely a
host-path rename.

---

## 1. Pull on Unraid

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

## 2. Rename the directory

```bash
docker compose stop audrey-ai
mv /mnt/user/appdata/audrey /mnt/user/appdata/runtime
docker compose up -d audrey-ai
```

The stop is required because Docker holds the bind mount open while
the container runs. With the container stopped, the rename is a
metadata-only operation (instant, no data movement on BTRFS).

## 3. Verify

```bash
# Confirm the bind mount resolves
docker exec audrey-ai ls -la /data | head -10

# Expect:
#   uploads.sqlite (Phase 15)
#   uploads/        (Phase 13)
#   any caches that have accumulated

# Confirm uploads.sqlite is intact (audrey-ai image has no sqlite3 CLI;
# use the python stdlib module instead).
docker exec audrey-ai python -c \
  "import sqlite3; print(sqlite3.connect('/data/uploads.sqlite').execute('SELECT COUNT(*) FROM uploads').fetchone()[0])"
# Expect: same row count as before the rename.

# Confirm a real chat completion still works (smoke)
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"two-sentence intro to rsync"}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content'
```

If the container starts without the volume properly mounting (because
the host path doesn't exist), Audrey will hit `FileNotFoundError`
trying to open `/data/uploads.sqlite` at startup. Logs show it
clearly. Rollback is just `mv` in reverse.

## 4. Rollback

```bash
docker compose stop audrey-ai
mv /mnt/user/appdata/runtime /mnt/user/appdata/audrey
git checkout HEAD~1 -- compose.yaml docs/phase-4-deploy.md \
  docs/phase-13-deploy.md docs/phase-15-deploy.md CONTINUITY.md
docker compose up -d audrey-ai
```

(Adjust the `HEAD~1` to whatever commit hash precedes the rename.)

---

## Notes

- **Why not bundle this into Phase 24?** Phase 24 is the Prometheus
  stack relocation — different scope, different containers. Keeping
  them as separate phases means clean rollback if either breaks
  independently.
- **Why not merge `runtime/` into the repo?** Tempting (it's right
  there next to `audrey_ai_2.0/`) but the runtime dir holds SQLite
  + WAL journals + per-user upload bytes that would constantly dirty
  the working tree and risk accidental `git add`. Cleaner to keep
  it as a sibling under `appdata/`.
- **What if some other process references `/mnt/user/appdata/audrey/`?**
  Run `grep -rn 'appdata/audrey\b' --include='*.sh' --include='*.yaml'
  --include='*.yml' /mnt/user/` on Unraid before the rename to catch
  any stragglers (cron jobs, backup scripts, custom Unraid templates).
  As of 2026-04-29, only the audrey-ai container references it.
