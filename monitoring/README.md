# monitoring/

Prometheus + Grafana stack for Audrey. Lives in the repo (Phase 24)
so config changes go through `git pull`, same as the rest of the
deployment.

## Layout

```
monitoring/
├── compose.yaml                              # prometheus + grafana services
├── config/
│   └── prometheus.yml                        # scrape config (audrey-ai + self-scrape)
├── prometheus-rules/
│   └── audrey.yml                            # 4 alert rules (Phase 22)
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yaml       # Prometheus datasource (uid: prometheus)
│   │   └── dashboards/audrey.yaml            # dashboard provider pointed at dashboards/
│   └── dashboards/
│       └── audrey-tools.json                 # per-tool dispatch dashboard (Phase 9)
└── README.md                                 # this file
```

Persistent state lives **outside the repo** at the existing Unraid
paths so it doesn't bloat git history:

```
/mnt/user/appdata/prometheus/
├── data/                  # prometheus TSDB (~30d retention)
└── grafana-data/          # grafana SQLite, dashboards, datasources
```

The compose's bind-mounts use absolute paths for those two directories.
TSDB and Grafana state survive container recreates and `git pull`s.

## Running

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose up -d
docker compose logs -f --tail 20
```

The containers join `ollama-net` (external network owned by the audrey
compose) so Prometheus can resolve `audrey-ai:8000` for scrapes.

## URLs

- Prometheus UI: <http://192.168.1.11:9090>
- Grafana UI: <http://192.168.1.11:3000> (default `admin` / `changeme`)

Both are LAN-only — not tunneled by `cloudflared`.

## Provisioning (Grafana datasource + dashboards)

Phase 9 moved Grafana off "click in the UI" onto file-based
provisioning. The datasource, dashboard provider, and dashboard
JSONs live in `grafana/` and are bind-mounted into the container:

- `grafana/provisioning/` → `/etc/grafana/provisioning` (read at
  Grafana startup; defines the Prometheus datasource and the
  dashboard provider).
- `grafana/dashboards/` → `/etc/grafana/dashboards` (the dashboard
  provider watches this directory and reloads JSON changes within
  `updateIntervalSeconds=30`).

`allowUiUpdates: false` on the dashboard provider — the JSON file
is the source of truth, UI edits don't persist. To change a
dashboard, edit the JSON, commit, `git pull` on Unraid.

The datasource UID (`prometheus`) is pinned in
`provisioning/datasources/prometheus.yaml` so dashboard JSONs can
hard-reference it. If you ever rename the UID, grep `grafana/`
for the old name and update every match in lockstep.

### Adding a new dashboard

1. Export the dashboard JSON from the UI (one-time, for the shape)
   or hand-write it using `audrey-tools.json` as a reference.
2. Drop the file into `monitoring/grafana/dashboards/`.
3. Make sure:
   - `uid` is unique across all dashboards (used as the stable
     identifier across reloads).
   - Every panel's `datasource` block uses
     `{"type": "prometheus", "uid": "prometheus"}` (not the
     human-readable name — UID is what provisioning binds).
   - `editable: false` at the top level — the JSON is the
     source of truth.
4. `git add` and commit.
5. `git pull` on Unraid. Grafana picks it up within ~30 seconds
   without a container restart.

### Editing an existing dashboard

Same flow as above — edit the JSON, commit, pull. The dashboard
reloads in place; bumping the `version` field at the bottom is
optional but lets you see in the UI that it actually reloaded.

If you accidentally edited in the UI, the changes won't survive
the next reload anyway (since `allowUiUpdates: false`). To capture
a UI experiment, use the UI's **JSON Model** view, copy the JSON
back into the file, and commit.

## Adding or editing alert rules

1. Edit `prometheus-rules/audrey.yml` in the repo.
2. `git pull` on Unraid.
3. Hot-reload Prometheus without restart:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```
   Works because the compose passes `--web.enable-lifecycle`.

The bind mount at `./prometheus-rules:/etc/prometheus/rules:ro` is
read directly — no `cp` step required (Phase 22 used to need one
because the rules dir was outside the compose's working directory).

## Verifying scrape + rules health

```bash
# Are the scrape targets up?
curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job) \(.health)"'

# Are alert rules loaded?
curl -s http://localhost:9090/api/v1/rules | jq -r '.data.groups[].rules[].name'
```

Expect 2 targets `up` (audrey, prometheus self-scrape) and 4 rules
loaded (AudreyPipelineErrorRate, AudreyToolCallErrorRate,
AudreyToolCallLatencyP95, AudreyCloudModelErrorRate).

## Why split from `audrey_ai_2.0/compose.yaml`?

Audrey rebuilds frequently (`docker compose up -d --build audrey-ai`
every code change). Prometheus and Grafana are set-and-forget. Mixing
them means a typo in audrey's image build would risk metrics/dashboard
downtime. Two compose files, one network — clean separation.

## Migration from the pre-Phase-24 layout

Pre-Phase-24, the same files lived at `/mnt/user/appdata/prometheus/`
outside git. To migrate, see `docs/phase-24-deploy.md`. Migration is
zero-downtime-safe because the persistent state directories
(`data/`, `grafana-data/`) stay at the same host path; only the
compose, config, and rules move into the repo.
