# Phase 17 — Prometheus `/metrics` endpoint

**Goal:** observability for what audrey is actually doing — fast vs deep
ratio, which models are getting picked, how long calls take, whether the
GPU gate is contended, and whether KB queries return hits. Eight
metrics, each tied to a real question.

Why now: Phase 16 closed the auth-revocation latency gap, which means
the system is now correct enough to be worth measuring. Up to this
point the only signal was `docker compose logs`, which is fine for
debugging one request but useless for "is the deep panel slower this
week than last?" or "are we over-routing to a sick model?"

What changed:

- **`src/audrey/metrics.py` (new)** — eight Prometheus collectors plus
  a `render()` helper. All bounded label cardinality (no per-user
  labels — those would explode and leak emails).
- **`src/audrey/main.py`** — `GET /metrics` route. Unauthenticated;
  not published via cloudflared, so it's effectively LAN-only
  (Unraid's docker network).
- **`src/audrey/routes/openai.py`** — `_run_graph_with_metrics` wraps
  the three `graph.ainvoke()` sites; observes `audrey_pipeline_seconds`
  and increments `audrey_pipeline_total{outcome=ok|error}`.
- **`src/audrey/pipeline/fast_path.py`** — increments
  `audrey_dispatch_total{path=fast|fast_react}` per fast-path call.
- **`src/audrey/pipeline/deep_panel.py`** — one
  `audrey_dispatch_total{path=deep|deep_react}` per worker, before the
  panel kicks off.
- **`src/audrey/pipeline/synthesize.py`** —
  `audrey_dispatch_total{path=synth_primary|synth_fallback}` per
  attempt.
- **`src/audrey/models/ollama.py`** — `chat` and `chat_stream` observe
  `audrey_model_seconds{model, outcome}`.
- **`src/audrey/pipeline/semaphore.py`** — local-only
  `audrey_gpu_gate_wait_seconds`. Cloud calls don't pay any gate cost
  so they're omitted (otherwise the histogram fills with zeros).
- **`src/audrey/routes/kb.py`** —
  `audrey_kb_search_seconds{kind, had_user_collection}` and
  `audrey_kb_search_hits{kind}` per query.
- **`src/audrey/auth.py`** — `audrey_auth_cache_size` gauge,
  set after every cache mutation.
- **`pyproject.toml`** — `prometheus-client>=0.21`.

Out of scope (deliberately):

- Trace propagation / OTel. The metrics here answer aggregate
  questions — "which model got picked", "how often did fast-path
  miss the cache" — not "what happened on this one request."
- Per-user labels. Cardinality would explode and the labels would
  leak emails into Prometheus storage.
- Pre-built Grafana dashboards. Easier to evolve queries against the
  raw metrics than to maintain a dashboard JSON we don't read.
- A `/metrics` auth gate. The route is LAN-only because we don't
  publish it via cloudflared.

**Prereqs:**

- Phase 16 verified.
- A Prometheus instance reachable from the audrey container (Unraid
  ships one as a Docker template, or any external scraper that can
  hit `http://audrey-ai:8000/metrics` on the same network works).

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 20 audrey-ai | grep ready
```

`prometheus-client` is the only new pip dep — `--build` is required
so the wheel lands in the image.

No new env vars; no config keys.

---

## 2. Endpoint smoke

From the Unraid host (the audrey-ai container is reachable on the
docker network as `audrey-ai:8000`):

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics | head -40
```

Expected: a Prometheus text-format dump starting with HELP/TYPE lines
for `audrey_pipeline_seconds`, etc. If you see plain JSON or a 404,
the route didn't register — check `docker compose logs audrey-ai` for
import errors in `metrics.py`.

A few of the metric families have zero samples until traffic arrives;
that's normal. The HELP/TYPE lines are still present.

---

## 3. Drive each metric

The point of this section: hit each code path once, then verify the
corresponding metric moved. From your laptop with `$TOKEN` set to a
valid OWUI JWT.

### 3.1 Pipeline + dispatch + model calls (fast path)

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_deep","messages":[{"role":"user","content":"what is 2+2?"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq '.choices[0].message.content' | head
```

That should land in `_generate_via_pipeline` → fast path. Then on
Unraid:

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep -E '^audrey_(pipeline|dispatch|model)_(total|seconds_count)' | head -20
```

Expected:

- `audrey_pipeline_total{mode="fast",task_type="general",outcome="ok"} 1`
  (or whatever task_type your prompt classified to; "ok" is the key).
- `audrey_dispatch_total{model="<some-model>",task_type="general",path="fast"} 1`
  (path will be `fast_react` instead if the chosen model is in
  `fast_path.tool_capable_models`).
- `audrey_model_seconds_count{model="<same model>",outcome="ok"} 1`.

### 3.2 Pipeline + dispatch (deep path)

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","messages":[{"role":"user","content":"compare X and Y in detail"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq '.choices[0].message.content' | head
```

`audrey_cloud` forces `mode=deep`. Then:

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep -E 'mode="deep"|path="deep|path="synth' | head -20
```

Expected:

- `audrey_pipeline_total{mode="deep", ..., outcome="ok"}` incremented.
- One `audrey_dispatch_total{path="deep"}` (or `"deep_react"`) per
  worker that ran.
- One `audrey_dispatch_total{path="synth_primary"}`. If the primary
  failed and the fallback synth ran, `synth_fallback` will also be
  non-zero.

### 3.3 GPU gate

The gate only ticks for **local** workers. If your `audrey_cloud` test
above used cloud-only workers, the histogram stays empty — that's
correct behavior, not a bug. To force a local worker, run a prompt
through `audrey_local` (which uses the `deep_panel_local` pool):

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","messages":[{"role":"user","content":"summarize this short input"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq '.choices[0].message.content' | head
```

Then:

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep audrey_gpu_gate_wait_seconds_count
```

Expected: count > 0. If you have `gpu.concurrency: 1` and you fire
two `audrey_local` calls back-to-back, the second one will land in a
bucket > 0.05s — proof the gate is doing its job.

### 3.4 KB search

```bash
curl -sS -X POST -H "Content-Type: application/json" \
  -d '{"query":"steam deck oled panel","top_k":5}' \
  https://chat.builtryte.xyz/v1/kb/query | jq '.results | length'
```

Then:

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep -E 'audrey_kb_search_(seconds_count|hits_count)' | head
```

Expected:

- `audrey_kb_search_seconds_count{kind="text",had_user_collection="false"} >= 1`.
- `audrey_kb_search_hits_count{kind="text"} >= 1`.

If you pass a `user` field and that user has uploaded files,
`had_user_collection="true"` instead.

### 3.5 Auth cache gauge

```bash
docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep audrey_auth_cache_size
```

Expected: a value matching `GET /v1/admin/auth/status` (give or take
the request you just made to that endpoint, which itself gets cached).
Run a clear and watch it drop:

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/clear | jq

docker exec audrey-ai curl -sS http://127.0.0.1:8000/metrics \
  | grep audrey_auth_cache_size
```

Expected: drops to 0 immediately, then climbs back to 1 once the next
admin status call re-probes.

---

## 4. Wire to Prometheus

The Audrey side is done — `/metrics` is up at `http://audrey-ai:8000/metrics`
on the `ollama-net` bridge. What's missing is a Prometheus *server* to scrape
it. Below is a from-scratch setup using a separate compose file under
`/mnt/user/appdata/prometheus/`. If you already have Prometheus running on
Unraid, skip to §4.4 for the scrape stanza.

### 4.1 Prep the appdata directories

```bash
mkdir -p /mnt/user/appdata/prometheus/{config,data}
chown -R nobody:users /mnt/user/appdata/prometheus
```

`data` is where time-series get persisted (default ~15-day retention; tunable
later). `config` holds `prometheus.yml`.

### 4.2 Drop the scrape config

Create `/mnt/user/appdata/prometheus/config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s        # how often each target gets polled
  evaluation_interval: 15s    # how often alert/recording rules fire

scrape_configs:
  - job_name: audrey
    metrics_path: /metrics
    static_configs:
      - targets: ['audrey-ai:8000']
        labels:
          service: audrey-ai

  # Prometheus scraping itself — useful as a "is the scraper alive" canary.
  - job_name: prometheus
    static_configs:
      - targets: ['localhost:9090']
```

`audrey-ai:8000` resolves only when Prometheus is on the same docker network
as the audrey-ai container (next step). 15s is a reasonable default — every
audrey histogram has buckets that make sense at that resolution.

### 4.3 Add Prometheus to compose

Prometheus lives in its own compose file at
`/mnt/user/appdata/prometheus/compose.yaml`, separate from the audrey repo.
Keeps observability infra out of the application repo and lets the two
restart independently.

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    networks:
      - ollama-net
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

networks:
  ollama-net:
    external: true
```

Bring it up:

```bash
cd /mnt/user/appdata/prometheus
docker compose up -d
docker compose logs --tail 20
```

The `external: true` on the network is the same trick `compose.yaml` uses for
audrey — it joins the existing bridge instead of creating a new one. If you
forget this, Prometheus will sit on its own bridge and `audrey-ai:8000` won't
resolve.

### 4.4 Verify Prometheus reaches Audrey

Open `http://<unraid-ip>:9090/targets` in a browser. The `audrey` job should
show `UP` within ~30s, with `Last Scrape` ticking forward.

If it's `DOWN`:

- `dial tcp: lookup audrey-ai on 127.0.0.11:53: no such host` → Prometheus
  isn't on `ollama-net`. Fix: re-check the `networks:` block above and that
  `ollama-net` already exists (`docker network ls | grep ollama-net`).
- `connection refused` → audrey-ai is restarting or `/metrics` 500'd. Re-run
  §2 to confirm it's still serving.
- `403`/`401` → unlikely for /metrics, but if you see it, double-check you
  didn't accidentally route /metrics through the auth dependency.

Test a query in the Prometheus WebUI (`http://<unraid-ip>:9090/graph`):

```promql
audrey_pipeline_total
```

You should see one row per `(mode, task_type, outcome)` combination from
your earlier smoke tests. If the result is empty, the scrape is failing
silently — check `/targets` again.

### 4.5 Useful starter queries

These are the six queries the Audrey dashboard in Grafana (§4.7) is built
from. The Prometheus WebUI itself doesn't save queries — bookmark the URL
or use Grafana panels for persistence.

```promql
# Fast vs deep mix over the last 5 minutes
sum by (mode) (rate(audrey_pipeline_total[5m]))

# p95 pipeline latency by mode
histogram_quantile(0.95, sum by (mode, le) (rate(audrey_pipeline_seconds_bucket[5m])))

# Top 5 dispatched models in the last hour
topk(5, sum by (model) (increase(audrey_dispatch_total[1h])))

# KB miss rate (fraction of queries returning 0 hits)
sum(rate(audrey_kb_search_hits_bucket{le="0"}[5m]))
  / sum(rate(audrey_kb_search_hits_count[5m]))

# GPU gate p50/p95 wait (only meaningful if you run multiple local prompts)
histogram_quantile(0.5,  rate(audrey_gpu_gate_wait_seconds_bucket[5m]))
histogram_quantile(0.95, rate(audrey_gpu_gate_wait_seconds_bucket[5m]))

# Auth cache size right now (gauge — point-in-time, not rate)
audrey_auth_cache_size
```

### 4.6 Hot-reloading config

Once Prometheus is up, you don't need to restart the container to add new
scrape jobs. Edit `prometheus.yml`, then:

```bash
curl -sS -X POST http://<unraid-ip>:9090/-/reload
```

If reload returns non-empty body, the new config has a parse error — old
config stays in effect. Restart only if reload itself fails.

### 4.7 Grafana

Grafana runs alongside Prometheus in the same compose stack at
`/mnt/user/appdata/prometheus/compose.yaml`. Add the service:

```yaml
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    networks:
      - ollama-net
    ports:
      - "3000:3000"
    volumes:
      - ./grafana-data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: changeme
```

Prep the data directory and start it:

```bash
mkdir -p /mnt/user/appdata/prometheus/grafana-data
chown -R nobody:users /mnt/user/appdata/prometheus/grafana-data
cd /mnt/user/appdata/prometheus
docker compose up -d grafana
```

Open `http://<unraid-ip>:3000`, log in as `admin` / `changeme`, then walk
through the first-time setup:

1. **Add the Prometheus data source.** Connections → Data sources → Add new
   data source → Prometheus. URL: `http://prometheus:9090` (the docker DNS
   name — works because Grafana and Prometheus share `ollama-net`). Save & test.
2. **Create the Audrey dashboard.** Dashboards → New → New dashboard.
3. **Add one panel per query from §4.5.** For each: paste the PromQL into the
   query field, set the title, choose a viz type:
   - "Time series" for the rate / quantile queries (fast vs deep, p95
     latency, GPU gate p50/p95).
   - "Stat" for the gauge query (auth cache size).
   - "Bar gauge" or "Stat" for the topk query (top dispatched models).
4. **Save the dashboard** with a memorable name like "Audrey".

Network gotcha: Grafana resolves `prometheus:9090` only because both
containers share `ollama-net`. If you ever move Grafana to a different
bridge, swap the URL to `http://<unraid-ip>:9090`.

Storage gotcha: dashboards live in `./grafana-data`, not in `prometheus.yml`.
Back up `/mnt/user/appdata/prometheus/grafana-data/grafana.db` if the
dashboards become important. Or export each dashboard's JSON via the WebUI
(Dashboard settings → JSON Model → Copy) and check it into a repo.

Password gotcha: change `GF_SECURITY_ADMIN_PASSWORD` from `changeme` to
something real before exposing port 3000 anywhere outside the LAN. Grafana
on `:3000` is unauthenticated apart from this.

---

## 5. Rollback

Pure additive: removing the route and the metrics module would just
revert the observability — no data path is affected. If something
goes sideways:

```bash
git checkout <previous-sha> -- src/audrey/main.py src/audrey/metrics.py \
  src/audrey/auth.py src/audrey/models/ollama.py \
  src/audrey/pipeline/{fast_path,deep_panel,synthesize,semaphore}.py \
  src/audrey/routes/{kb,openai}.py
docker compose up -d --build audrey-ai
```

`prometheus-client` will linger as an installed dep — harmless until
the next image rebuild without it.

---

## 6. Follow-ups (not Phase 17)

- Per-tool dispatch metric inside the ReAct loop (which custom-tool
  got called, how long it took, did it error). Right now we see the
  *model* call but not the *tool* call.
- Histogram for the synth attempt count (today's `dispatch_total`
  with `path=synth_*` tells us by inference but not directly).
- A small Grafana dashboard with: deep-vs-fast pipeline rate, p95
  pipeline latency by mode, top-5 dispatched models, KB miss rate.
- Re-probe / cache-miss rate on the auth cache (today we see size
  but not hit/miss flow).
