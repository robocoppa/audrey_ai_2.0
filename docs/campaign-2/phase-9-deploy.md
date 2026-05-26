# Campaign 2 Phase 9 — Grafana dashboard provisioning + per-tool panel

Two things ship in one phase:

1. **Grafana provisioning files in the repo.** Datasource and dashboard
   provider land under `monitoring/grafana/provisioning/`; dashboard
   JSONs land under `monitoring/grafana/dashboards/`. `git pull` on
   Unraid is now enough to deploy a new dashboard or panel — no
   clicking in the UI, no `grafana-cli` dance, no admin-password
   round-trip.
2. **Per-tool dispatch dashboard (`audrey-tools.json`).** Drains the
   long-standing followup ("Grafana per-tool dispatch panel —
   data exists, dashboard would make Phase 4 Categories 3 and 5
   faster to eyeball, ~1 hour"). Four panels: dispatch rate per tool
   (stacked), error rate per tool (% with thresholds), p95 latency
   per tool, chat-archive write health by result.

Phase 24 split Prometheus + Grafana out of the audrey compose; Phase 9
finishes the same arc by moving Grafana's *contents* into git too.
Both the scrape config and the dashboards are now versioned alongside
the code that produces the metrics.

## Closure verification — *pending*

Pending the Unraid-side deploy walk in §1. The repo-side changes are
ready; verification adds:

- `docker compose up -d grafana` recreates the container with the
  two new bind-mounts.
- Grafana logs show the datasource and dashboard provider loaded
  (`Datasources provisioned`, `Dashboards provisioned`).
- Browsing to <http://192.168.1.11:3000/dashboards> shows the
  **Audrey** folder containing **Audrey — Tools**.
- Each of the four panels renders against live Prometheus data.

---

## What's in scope

### `monitoring/grafana/provisioning/datasources/prometheus.yaml`

The Prometheus datasource. UID is pinned to `prometheus` so the
dashboard JSON can hard-reference it. URL is `http://prometheus:9090`
(container-name resolution inside `ollama-net`). `access: proxy` keeps
queries server-side so Prometheus stays LAN-only.

`editable: false` so a click-in-UI edit can't drift the datasource
config away from the file.

### `monitoring/grafana/provisioning/dashboards/audrey.yaml`

The dashboard provider. Points at `/etc/grafana/dashboards` (bind-mounted
from `monitoring/grafana/dashboards/` in the repo). Watches the
directory; reloads JSON changes within `updateIntervalSeconds=30`. No
container restart required for dashboard edits.

`allowUiUpdates: false` — file is the source of truth. UI edits don't
persist. Capture a UI experiment via the **JSON Model** view, paste
back into the file, commit.

`folder: Audrey` groups every provisioned dashboard under one folder
in the Grafana UI. When future dashboards land (fairness, KB search,
deep-panel timings), they cluster.

### `monitoring/grafana/dashboards/audrey-tools.json`

Four panels in a 2×2 grid:

| Panel | Query | What it tells you |
| --- | --- | --- |
| Tool dispatch rate (ok, by tool) | `sum by (tool) (rate(audrey_tool_calls_total{outcome="ok"}[5m]))` | Which tools are actually being used; a flat-zero line for a tool the model should be reaching for = upstream break (auth, registry rediscover, etc.). |
| Tool error rate (5m, by tool) | `sum by (tool) (rate(audrey_tool_calls_total{outcome!="ok"}[5m])) / clamp_min(sum by (tool) (rate(audrey_tool_calls_total[5m])), 1e-9)` | Per-tool error fraction. Thresholds: green <5%, yellow 5-20%, red ≥20%. |
| Tool dispatch p95 latency (5m, by tool) | `histogram_quantile(0.95, sum by (tool, le) (rate(audrey_tool_call_seconds_bucket[5m])))` | Per-tool p95. Failures often time out → p95 climbs first on a degraded tool, before error rate does. |
| Chat archive writes (5m, by result) | `sum by (result) (rate(audrey_chat_archive_writes_total[5m]))` | Stacked by `{ok, partial, fail, skipped}` with colors. Sustained `partial`/`fail` = tools-server back-pressured or SQLite store misbehaving. |

`clamp_min` on the error-rate denominator handles the division-by-zero
case during low-traffic windows (no tool calls = no error rate). 1e-9
is small enough to make the resulting fraction effectively zero, large
enough that Grafana doesn't render NaN.

### `monitoring/compose.yaml`

Two new bind-mounts on the `grafana` service:

```yaml
- ./grafana/provisioning:/etc/grafana/provisioning:ro
- ./grafana/dashboards:/etc/grafana/dashboards:ro
```

Read-only because Grafana only *reads* provisioning files — writing
back happens to the SQLite at `/var/lib/grafana`, which stays at the
existing absolute path (`/mnt/user/appdata/prometheus/grafana-data`).

### `monitoring/README.md`

New **Provisioning** section explains the layout, the source-of-truth
rule, and the "add a new dashboard" / "edit an existing dashboard"
workflows. References the datasource UID pinning so future dashboards
know what to bind against.

---

## 1. Deploy

Run from `/mnt/user/appdata/audrey_ai_2.0/monitoring`:

```bash
# Pull the repo changes (datasource YAML, dashboard provider YAML,
# dashboard JSON, README update, compose bind-mounts).
cd /mnt/user/appdata/audrey_ai_2.0
git pull

# Recreate the grafana container so the new bind-mounts attach.
# Prometheus is unaffected — leave it running.
cd monitoring
docker compose up -d grafana
```

`up -d grafana` does the right thing here: compose sees the new
volumes in the spec, stops the existing grafana container, and
recreates it with the bind-mounts. Persistent state under
`/mnt/user/appdata/prometheus/grafana-data/` survives the recreate
(it's an absolute bind-mount, not a named volume).

## 2. Smoke tests

### 2.1 Datasource provisioned

```bash
docker compose logs grafana --tail=200 | grep -iE 'datasource|provision'
```

Expect a line like `msg="inserting datasource from configuration" name=Prometheus uid=prometheus`. If you see `"datasource not found"`, the bind-mount didn't attach — re-check `docker compose config grafana` shows both `volumes` lines.

### 2.2 Dashboard provisioned

```bash
docker compose logs grafana --tail=200 | grep -iE 'dashboard'
```

Expect `msg="finished to provision dashboards"` and at least one
`inserting dashboard` line naming `audrey-tools`. The dashboard
provider logs the directory it watched — confirm it matched
`/etc/grafana/dashboards`.

### 2.3 Dashboard renders

Browse to <http://192.168.1.11:3000>, log in, navigate to
**Dashboards → Audrey → Audrey — Tools**. Confirm all four panels
render. With live traffic, the tool dispatch rate panel should show
non-zero series for whatever tools have been called recently.

If a panel says "No data," check:

- Is Prometheus actually scraping audrey? `curl -s
  http://localhost:9090/api/v1/query?query=audrey_tool_calls_total`
  should return non-empty `data.result`.
- Is the datasource UID right? Inspect the panel → Query → datasource
  picker — should say `Prometheus` (the human name backing UID
  `prometheus`).

### 2.4 Drive a tool call, watch the rate panel update

Hit Audrey with a prompt that triggers a tool — anything mentioning a
KB search or web search works. Within ~30 seconds the **Tool dispatch
rate** panel should add a new sample for the matching `tool` label.

## 3. Rollback

If anything breaks Grafana's startup:

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
git checkout HEAD~1 -- compose.yaml grafana/
docker compose up -d grafana
```

That reverts the compose bind-mounts and removes the provisioning
files from the container's view on the next start. Persistent state
(dashboards previously saved in the UI, datasources created by hand)
is untouched at `/mnt/user/appdata/prometheus/grafana-data/`.

The provisioning files themselves are *additive* — they don't delete
anything Grafana already has. If you provision a `Prometheus`
datasource and the UI already had a hand-created one with the same
name, Grafana renames the older one rather than overwriting. So the
rollback is genuinely safe; you don't lose UI work that pre-existed
this phase.

## 4. Operational notes

- **Dashboard reloads are hot.** Edit a JSON file, commit, `git pull`.
  Grafana picks it up within ~30 seconds. Bumping the `version` field
  at the bottom of the JSON is a polite signal that something
  changed, but not required.
- **Datasource changes are NOT hot.** A change to
  `provisioning/datasources/prometheus.yaml` requires a Grafana
  restart (`docker compose restart grafana`). Datasource provisioning
  is read once at startup.
- **UI edits don't persist.** `allowUiUpdates: false` means saving in
  the UI succeeds in-memory but vanishes on the next reload. If you
  want to keep a UI experiment, copy from **JSON Model** back into
  the file. (This is by design — every other repo-managed config
  works the same way.)
- **Adding panels.** Edit the JSON `panels` array. The dashboard's
  grid uses `gridPos: {h, w, x, y}` with a 24-column grid;
  `audrey-tools.json` uses two rows of two panels (12 wide, 8 tall
  each). Add a third row by appending two more panels with
  `y: 16`.

## 5. Followups

- **Per-virtual-model latency panel.** `audrey_model_seconds_bucket`
  exists and is labeled by model. A second dashboard breaking down
  fast-path vs. deep-panel latency by virtual model would be the
  natural next addition. Plenty of data — defer until a question
  needs it.
- **Per-user fairness dashboard.** Once Lesson 14 (fair scheduling)
  audits `audrey_gpu_gate_wait_seconds` and
  `audrey_user_inflight_blocked_seconds`, a fairness dashboard would
  pair nicely with that lesson's narrative.
- **Grafana folder structure.** Currently one folder (`Audrey`)
  with one dashboard. When the count grows past ~5, consider
  switching `foldersFromFilesStructure: true` in the dashboard
  provider so subdirectories under `dashboards/` map to folders.
