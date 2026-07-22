# Box Operations Guide — the Audrey Unraid deployment

How the deployed Audrey stack is wired, and how to operate/probe it correctly.
Written because infra state kept being re-derived (and mis-derived) each session.
If you're an agent: **read this before handing the user any box command.**

> **Golden rule (from AGENTS.md):** the user deploys to Unraid; the assistant
> stays on the laptop checkout. Anything requiring box/deploy/remote state goes
> **through the user** — hand the command, don't infer the result. No write-side
> Docker/Unraid ops from the assistant.

---

## 1. The host

- **Unraid server**, LAN IP **`192.168.1.11`**. Shell prompt is `root@Tower`.
- Appdata root: **`/mnt/user/appdata/`**. Audrey's is
  `/mnt/user/appdata/audrey_ai_2.0/` (config, testing-out, eval.env live here).
- Knowledge datasets: host `/mnt/user/knowledge/<topic>` → container `/datasets/`.
- Deploys run **from `/mnt/user/appdata/audrey_ai_2.0`** (where `compose.yaml` sits).

## 2. Tooling reality — DON'T ASSUME `curl`

**`curl` is NOT installed in the app containers.** Verified 2026-07-15:
```
root@Tower:~# docker exec custom-tools sh -c 'curl ...'
sh: 1: curl: not found
```
The deploy docs (`docs/campaign-2/phase-1-deploy.md`) *do* use `curl` — so it
exists in **some** context (the host, or a container that bundles it), but **not
inside the Python app containers** (`audrey-ai`, `custom-tools`, `SearXNG`
clients). The trap is `exec`ing into a container and assuming its shell has curl.

**Use Python for HTTP probes inside app containers** — they're Python services,
so `python3` is present:
```bash
# SearXNG reachability from the tools server (internal network, no curl):
docker exec custom-tools python3 -c "import urllib.request,json; \
print(len(json.load(urllib.request.urlopen( \
'http://SearXNG:8080/search?q=test&format=json', timeout=10)).get('results',[])), 'results')"
```
When unsure whether a context has curl/wget, **ask or use `python3`** — don't hand
a curl command and hope.

## 3. Container / port / network map

All services join the Docker network **`ollama-net`**. **Container-name DNS
(e.g. `custom-tools`, `SearXNG`, `ollama`, `qdrant`) resolves only inside
`ollama-net`** — not from the Unraid host shell. Service-to-service calls use the
**internal** port + container name; the **host-published** port is only for
reaching a service from the host/LAN.

| Container | ollama-net IP | host → internal port | Role |
|---|---|---|---|
| **audrey-ai** | 172.18.0.10 | 8000 → 8000 | FastAPI app (THIS repo) |
| **custom-tools** | 172.18.0.16 | 8001 → 8001 | Tools server — serves `web_search` (Brave↔SearXNG logic lives here). Built from **`tools-server/` IN THIS REPO**, not a separate project |
| **SearXNG** | 172.18.0.15 | **8088 → 8080** | Search engine behind `web_search` |
| **open-webui** | 172.18.0.5 | **8080 → 8080** | OWUI frontend / public surface |
| **ollama** | 172.18.0.13 | 11434 → 11434 | Model runtime |
| **qdrant** | 172.18.0.14 | 6333/6334 | Vector DB (KB) |
| **bot-tools-mcp** | 172.18.0.7 | 9110 | Separate MCP (bot fleet — NOT Audrey web_search) |
| **fleet-watchdog** | 172.18.0.6 | 9099 | Notify hub (Telegram eval-completion pings) |
| **prometheus** | 172.18.0.11 | 9090 | Metrics |
| **grafana** | 172.18.0.4 | 3000 | Dashboards |
| nextcloud / -db / -redis, collabora (9980), radicale (5232), cloudflared | — | — | Adjacent services, not in the Audrey request path |

### Port traps (these have bitten before)
- **Host `8080` = open-webui, NOT SearXNG.** SearXNG is **`8088`** on the host,
  **`8080`** internally. A `localhost:8080` probe on the box hits OWUI and returns
  a misleading `200`.
- **SearXNG internal address is `http://SearXNG:8080`** (capital S, port 8080) —
  that's what `custom-tools` calls, not `:8088`.

## 4. Request path (who calls whom)

```
OWUI  ──►  audrey-ai:8000  ──►  custom-tools:8001  ──►  SearXNG:8080  ──► (web)
(public)   (FastAPI app)        (tools/web_search)      (search engine)
                │
                ├──►  ollama:11434        (models)
                └──►  qdrant:6333         (KB vectors)
```

- **`web_search` is served by `custom-tools`, not `audrey-ai`.** audrey-ai
  *discovers* it via OpenAPI from `http://custom-tools:8001` (see `tools:` in
  `config.yaml`) and calls it as a remote tool. The Brave/SearXNG provider logic
  lives in the **custom-tools repo** — a different codebase from this one.
- `config.yaml`'s `search: { backend: brave }` block (with `searxng (future)`)
  is **vestigial/misleading** relative to the live path — the real search
  provider selection + fallback is in custom-tools. Don't "fix search" by editing
  that block; it isn't the live lever.

## 5. Deploy (user runs these; assistant hands them)

From `/mnt/user/appdata/audrey_ai_2.0`:
```bash
# Code change (rebuild image):
docker compose up -d --build audrey-ai
docker compose up -d --build custom-tools

# Config-only change (no rebuild — config.yaml is bind-mounted):
docker compose up -d --force-recreate audrey-ai

# Logs:
docker compose logs -f audrey-ai

# Monitoring stack:
cd monitoring && docker compose up -d
```
Compose layout (AGENTS.md): root `compose.yaml` has only `audrey-ai` +
`custom-tools`; `monitoring/compose.yaml` has Prometheus+Grafana; Ollama, Qdrant,
Open WebUI, cloudflared are managed **outside** root compose.

**Which verb?**
- Edited `config.yaml` only → `--force-recreate` (it's bind-mounted, reloaded at
  startup).
- Edited any code under `src/audrey/` (or `custom-tools` source) → `--build`.
- Edited eval case files baked into the image → `docker compose --profile eval
  build audrey-eval` before the next on-box eval.

## 6. Editing config on the box vs. the repo

The running config is `/mnt/user/appdata/audrey_ai_2.0/config.yaml`. The clean
flow is **edit in the repo → push → pull on the box → recreate**, so the box
stays in sync with git. In-place `sed` edits on the box work for a quick,
about-to-be-reverted experiment, but they **drift from git** and a later
`git pull` can conflict/clobber. `sed` gotcha: use a delimiter other than `/`
when the pattern contains a URL/`#`, and anchor loosely (the box's YAML
indentation/quoting may differ from what a strict pattern assumes).

## 7. Evals on the box

`scripts/eval-onbox.sh` runs the eval container detached, waits, and Telegram-pings
on completion. Output lands in `/mnt/user/appdata/audrey_ai_2.0/testing-out/`.
Secrets sourced at runtime: OWUI key from `${APPDATA}/eval.env`, Telegram creds
from `/mnt/user/appdata/fleet-watchdog/.env`. Case files are **baked into the eval
image** — rebuild (`docker compose --profile eval build audrey-eval`) after any
case-file or harness change. Detached run pattern:
```bash
nohup env MODEL=audrey_research CASES=<file>.json LABEL=<label> \
  scripts/eval-onbox.sh \
  >/mnt/user/appdata/audrey_ai_2.0/testing-out/last-<label>-run.log 2>&1 &
```

## 8. Probing / diagnosing (no curl!)

- **Is a container up + healthy?** `docker ps` (shows host↔internal ports and
  health). Cross-check ports against §3 — don't trust a port number from memory.
- **What env does a service actually have?**
  `docker exec <container> printenv | grep -i <key>`.
- **HTTP probe between services** (internal network): use the Python one-liner in
  §2. `curl` is not in the app containers.
- **App logs for the failure window:**
  `docker compose logs --since 30m audrey-ai` (or `docker logs <id>`), grep for
  the subsystem (`web_search`, `brave`, `searxng`, `429`, `quota`).
- **A `200` HTTP code is not "healthy"** for search: a throttled SearXNG returns
  `200` with an **empty `results` array** ("SearXNG empties" — a known recurring
  condition, see PROJECT_STATE). Always check the **result count in the body**,
  not just the status code.

## 9. Known failure modes (so they aren't re-diagnosed from scratch)

- **`web_search` returns nothing on every case.** Usually a provider outage:
  Brave at its (monthly) usage cap, and/or SearXNG returning empties. The pipeline
  is designed to **degrade gracefully** — researchers fall back to prior knowledge
  and answers can still be correct, just unsourced (which then makes `hedge_policy`
  hedge more, since unsourced claims hedge by default). Empty search ≠ broken
  pipeline. Fix is at the provider/tools-server layer, not audrey-ai.
- **`kb_search` fails intermittently (`✅0 ❌1`) with no error body.** Diagnosed
  2026-07-22. NOT a tool, dispatch, or corpus bug: `OLLAMA_MAX_LOADED_MODELS=1`
  could not hold the 323 MB `nomic-embed-text` embedder alongside a ~24 GB panel
  worker model, so every `kb_search` following a generation paid a full model
  swap. Measured embed latency: **0.06s resident, 12.98s idle-swap, 28.53s and
  42.17s under a live local panel** — past the 30s tool-dispatch ceiling.
  **Fixed by `OLLAMA_MAX_LOADED_MODELS=2`** (Unraid UI → ollama container; do NOT
  add a Device entry while in there, it breaks GPU startup). `NUM_PARALLEL` stays
  `1` — it gives Audrey's gated local runs de-facto exclusivity against the
  ungated co-tenants (OpenClaw, OWUI), and KV cache is allocated per parallel
  slot at load time whether or not the slot is used.
  - **Diagnostic:** `docker exec -i custom-tools python3 - < scripts/embed_contention_probe.py`
    samples embed latency against Ollama residency. Single hand-timed probes are
    worthless here — three identical back-to-back calls returned 1.01s, 23.82s and
    0.08s. Judge by the distribution and the over-ceiling count, not one reading.
  - **Watch for:** a slow sample sitting next to a big model in the `resident`
    column is an eviction; a slow sample while a NEW model appears is a cold-load
    stall (narrower, and mostly disjoint from when real `kb_search` calls fire).
- **Wrong-port probe.** See §3 — `localhost:8080` is OWUI.
- **"Just renew the Brave key" is not always the cure.** A past PROJECT_STATE
  entry recorded this as a struck-through misdiagnosis; the real issue was
  SearXNG. Check the actual provider state before prescribing.

---

_Keep this in sync with reality: when a container/port changes, update §3. The
canonical durable memory pointers are `[[audrey-box-container-map]]` and
`[[audrey-box-ops-no-curl]]`._
