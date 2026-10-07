# Audrey monitoring

Operator Prometheus/Grafana stack, separate from the application Compose.
Persistent data stays in `/mnt/user/appdata/prometheus/{data,grafana-data}`.
Both services join `ollama-net`; Prometheus scrapes `audrey:8000` every 15 seconds.
URLs use the working LAN/WARP route:

- [Models dashboard](http://192.168.1.11:3000/d/audrey-models)
- [Tools dashboard](http://192.168.1.11:3000/d/audrey-tools)
- [Prometheus](http://192.168.1.11:9090)

Sign in to Grafana with the existing operator account. Use the configured
credentials; there is no documented default password. These are LAN operator
surfaces, not a new Audrey user or bot endpoint.

## This slice: model dashboard

**Run on Tower after the user's changes are committed:**

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

**No container rebuild or restart is needed for this slice.** The accepted
backend telemetry is already deployed. Grafana's mounted dashboard directory
reloads within 30 seconds. Its pinned Prometheus datasource remains unchanged.

**Check in a browser on the laptop:**

1. Open the Models dashboard above after the reload interval. Its title should
   be **Audrey — Models**. Leave the range at **Last 6 hours** / **All** models.
2. If there has been no recent traffic, open `https://ai.builtryte.xyz` and send
   one ordinary question. Wait for the answer and the next metrics scrape.
   No file upload or scripted evaluation is required.
3. Confirm the calls-since-restart panel contains model/outcome rows. Select
   a model in the **Model** filter and confirm the panels restrict to it.
4. Confirm usage totals appear where the provider reported them. Missing fields
   (often cached input) stay absent/**No data**, not fabricated zero or cost.
   Rate/latency charts require multiple scrapes and calls in their window;
   counter panels provide the immediate confirmation.

Pass: dashboard loads, observed model rows/filter work, and unsupported usage
is not invented. No forced cancellation or failure, extra model sweep, backend
rebuild, or repeat of accepted telemetry is required. This dashboard has been
validated locally for JSON, provisioning, layout, and parsed PromQL; live rendering is the
remaining user check.

## Metric meaning

| Metric | Meaning |
|---|---|
| `audrey_model_seconds` | One provider-terminal latency observation per call, labeled model/outcome |
| `audrey_model_tokens_total` | Sum of valid provider-reported input/output/cached-input fields |
| `audrey_model_usage_observations_total` | Calls reporting a valid value, separately for each usage kind |

Calls are provider generations, not chat turns; active calls are not yet counted.
Outcomes are `ok` (confirmed completion, including a length stop), `error`
(failure/unconfirmed end), and `cancelled` (interrupted before terminal).
Terminal timing excludes pipeline queue time and later consumer cleanup.
Histogram quantiles are estimates; the final finite bucket is 180 seconds.
The error fraction excludes cancellations; the latency distributions include
all terminal outcomes, so inspect outcome rates alongside latency.

Valid zero increments an observation. Missing/invalid counts stay unknown.
Output may include reasoning; cached input is a subset, not extra input or
measured billing savings. Coverage measures field reporting, not cache hit rate.
Counter panels reset with the backend process and ignore the dashboard time
range; rate panels use that range and require multiple samples. No costs,
prompts, account ids, or emails are added.

## Maintenance

Dashboard JSON lives in `grafana/dashboards/`, with stable unique UIDs and
`editable: false`. Datasource UID is `prometheus`; provisioned dashboards are
the source of truth. Pull dashboard edits on Tower; no build/restart.

For an actual monitoring Compose change, run **on Tower**:

```bash
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose up -d
```

Grafana requires `GRAFANA_ADMIN_PASSWORD` in its existing private environment
for Compose interpolation. Provisioning changes require a Grafana restart;
dashboard JSON changes do not.

Prometheus rule/config edits require a reload after pull, **on Tower**:

```bash
curl --fail --silent --show-error --write-out 'HTTP %{http_code}\n' \
  -X POST http://127.0.0.1:9090/-/reload
```

Pass: HTTP 200. For application rebuilds and runner selection, use the
[Campaign 3 main plan](../docs/campaign-3/README.md).
