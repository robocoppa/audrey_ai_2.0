# monitoring/

Prometheus + Grafana stack for Audrey. Lives in the repo (Phase 24)
so config changes go through `git pull`, same as the rest of the
deployment.

## Layout

```
monitoring/
├── compose.yaml           # prometheus + grafana services, repo-managed
├── config/
│   └── prometheus.yml     # scrape config (audrey-ai + self-scrape)
├── prometheus-rules/
│   └── audrey.yml         # 4 alert rules (Phase 22)
└── README.md              # this file
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
