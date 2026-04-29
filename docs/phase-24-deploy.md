# Phase 24 — move Prometheus stack into the audrey repo

**Goal:** stop editing the Prometheus compose, scrape config, and
alert rules in-place at `/mnt/user/appdata/prometheus/` outside git.
Move them into `monitoring/` in the audrey repo so they get the same
`git pull` workflow as audrey-ai itself. Persistent state (TSDB,
Grafana SQLite) keeps living at the existing Unraid paths — only
*config* moves.

What stays the same:

- Prometheus + Grafana container names, image tags, ports, network.
- TSDB at `/mnt/user/appdata/prometheus/data/` — bind-mounted into
  the new compose at the same absolute path.
- Grafana state at `/mnt/user/appdata/prometheus/grafana-data/` —
  same. Saved dashboards, datasources, admin user all preserved.
- Scrape config (only audrey-ai + self-scrape).
- The four Phase 22 alert rules.
- The default `admin` / `changeme` Grafana login.

What changed:

- **NEW** `monitoring/compose.yaml` — moved from
  `/mnt/user/appdata/prometheus/compose.yaml`. Cleaned up duplicate
  volume entries from the old version. Adds `TZ=America/Denver` env
  var on both containers (per CONTINUITY.md gotcha — prometheus and
  grafana were missing it before).
- **NEW** `monitoring/config/prometheus.yml` — moved from
  `/mnt/user/appdata/prometheus/config/prometheus.yml`. Verbatim copy.
- **NEW** `monitoring/README.md` — short ops doc.
- The rules dir is now sourced directly from the repo's existing
  `monitoring/prometheus-rules/audrey.yml` (canonical since Phase 22).
  No more `cp` step in the Phase 22 deploy.

Out of scope (deliberately):

- **Merging with `compose.yaml` at the repo root.** Audrey rebuilds
  often; Prometheus+Grafana is set-and-forget. Keep them separate.
- **Putting TSDB or Grafana state into the repo.** Both are large
  and churn-heavy; bad fit for git.
- **Changing scrape config or alert rules.** This is a pure-move
  phase. Tweaks come later.

**Prereqs:** Phase 22 verified (rules already exist as
`monitoring/prometheus-rules/audrey.yml` in the repo).

---

## 1. Pull on Unraid

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

Should bring in `monitoring/compose.yaml`, `monitoring/config/prometheus.yml`,
`monitoring/README.md`, and this deploy doc.

## 2. Sanity-check the new files match what's running

Before stopping anything, diff the new files against what's currently
in production. Expect zero meaningful differences.

```bash
# Compare prometheus.yml
diff /mnt/user/appdata/prometheus/config/prometheus.yml \
     /mnt/user/appdata/audrey_ai_2.0/monitoring/config/prometheus.yml

# Compare compose.yaml — there WILL be diffs (cleaned up duplicate
# volume lines, added TZ env, persistent-state paths now absolute).
# Eyeball it and confirm the diffs are intentional.
diff /mnt/user/appdata/prometheus/compose.yaml \
     /mnt/user/appdata/audrey_ai_2.0/monitoring/compose.yaml
```

Expected diffs in the compose:
- Old had `./config/prometheus.yml:...:ro` and `./data:/prometheus`
  duplicated — new is single-entry.
- Old had `./rules:...:ro` (host directory) — new uses
  `./prometheus-rules:...:ro` (repo directory, canonical source).
- Old had `./data` and `./grafana-data` as relative paths — new uses
  absolute `/mnt/user/appdata/prometheus/data` and `.../grafana-data`
  so the working directory move doesn't break the bind mounts.
- New has `TZ: America/Denver` on both services.
- New has a header comment block.

If anything else looks unexpected, **stop and ask** before continuing.

## 3. Stop the old stack

```bash
cd /mnt/user/appdata/prometheus
docker compose down
```

Containers stop. TSDB + grafana-data stay on disk. `audrey-ai` keeps
running unaffected (`/metrics` endpoint just stops being scraped for
the next ~30s).

## 4. Start the new stack

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose up -d
docker compose logs --tail 30
```

Both `prometheus` and `grafana` should start clean.

Common failure mode: if the bind mount for `data/` or `grafana-data/`
points at the wrong path, Prometheus starts with empty TSDB
(metrics history wiped — recoverable from rollback) or Grafana starts
fresh with the default `admin/admin` login (dashboards wiped). The
absolute paths in the new compose match what was running, so this
shouldn't happen — but if you see "no data" in Grafana, **rollback
immediately** (step 8) before any new state writes complicate things.

## 5. Verify scrape targets

```bash
curl -s http://localhost:9090/api/v1/targets \
  | jq -r '.data.activeTargets[] | "\(.labels.job) \(.health) \(.lastError // "ok")"'
```

Expect:
```
audrey up ok
prometheus up ok
```

If `audrey` is `down`, the network isn't joined correctly. Check that
`ollama-net` is `external: true` and exists:

```bash
docker network ls | grep ollama-net
```

## 6. Verify alert rules loaded

```bash
curl -s http://localhost:9090/api/v1/rules \
  | jq -r '.data.groups[].rules[].name'
```

Expect four:
```
AudreyPipelineErrorRate
AudreyToolCallErrorRate
AudreyToolCallLatencyP95
AudreyCloudModelErrorRate
```

If you see fewer than four, the bind mount of the rules dir didn't
land. Check inside the container:

```bash
docker exec prometheus ls /etc/prometheus/rules
```

Should show `audrey.yml`.

## 7. Verify Grafana state preserved

Open <http://192.168.1.11:3000> in a browser, log in (`admin` /
`changeme`), and confirm:

- Datasource `Prometheus` is configured at `http://prometheus:9090`.
- The Audrey dashboard built in Phase 17 still renders.
- Recent metric history is still visible (last 30d).

If the login screen says "create admin user" instead of accepting
your old credentials, the bind mount missed the existing
`grafana-data/` — **rollback** (step 8).

If everything looks right, the migration is verified.

## 8. Rollback (only if step 4–7 failed)

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose down
cd /mnt/user/appdata/prometheus
docker compose up -d
```

State is preserved both ways because both composes bind at the same
host paths for `data/` and `grafana-data/`.

## 9. Clean up old config (only after successful verification)

Once the new stack has been running cleanly for at least an hour,
remove the now-superseded files at the old location:

```bash
rm /mnt/user/appdata/prometheus/compose.yaml
rm -rf /mnt/user/appdata/prometheus/config
rm -rf /mnt/user/appdata/prometheus/rules

# Keep these — they're live state:
ls /mnt/user/appdata/prometheus/  # → data/ grafana-data/
```

Don't `rm -rf /mnt/user/appdata/prometheus` itself — TSDB + Grafana
state still live there. Just remove the three superseded directories.

---

## Operational notes for future work

- **Editing alert rules.** Edit `monitoring/prometheus-rules/audrey.yml`
  in the repo, push, `git pull` on Unraid, then
  `curl -X POST http://localhost:9090/-/reload` for hot-reload (no
  restart needed).
- **Editing scrape config.** Same flow but for
  `monitoring/config/prometheus.yml`.
- **Adding a new container as a scrape target.** Edit the
  `scrape_configs:` block in `prometheus.yml`. The container must be
  on `ollama-net` so Prometheus can resolve its name. Reload
  Prometheus.
- **Bumping Prometheus or Grafana versions.** Change the image tag in
  `monitoring/compose.yaml`, `git pull` on Unraid,
  `docker compose pull && docker compose up -d`.
- **TZ env var.** New compose sets `TZ=America/Denver` explicitly on
  both containers. Old setup relied on Unraid container template
  defaults, which may or may not have applied to this stack. Post-
  migration, all timestamps in Prometheus + Grafana are consistent
  with audrey-ai's TZ.
