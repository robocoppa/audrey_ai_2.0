# Live testing — runner and deployment reference

This reference owns runner, authentication and rebuild instructions. Campaigns
are closed; no smoke or acceptance work is implied by these reusable commands.
English user input only; closed checks do not need another run.

## Where to run

| Work | Location and target |
|---|---|
| Hermetic tests/lint | Laptop checkout and its `.venv` |
| API-only smoke/eval | Laptop → `http://192.168.1.11:8000` (LAN/WARP) |
| Files/upload/chat acceptance | Browser → `https://ai.builtryte.xyz`; use a real file |
| Docker-only Ollama probe | Tower → `scripts/probes/probe-onbox.sh` |
| Native UI/proxy protocol smoke | Tower → `tests/smoke/smoke-native-onbox.sh` |
| Installed backend admission smoke, no models | Tower → `tests/smoke/smoke-agent-admission-onbox.sh` |

Laptop checkout: `/home/bart/Documents/github/audrey/audrey_ai_2.0`.
Tower checkout: `/mnt/user/appdata/audrey_ai_2.0`.
Tower has Docker, not a host Python/uv/repository `.venv`.
The backend is published on port 8000. The UI origin is Tower-loopback
`127.0.0.1:8090` or Docker DNS `http://audrey-ui:8080`; laptop access to that
origin needs an explicit tunnel. Tailscale `100.113.157.98:8000` remains
unreachable from the laptop. Repair is deferred until the user asks; use the
working LAN/WARP route for current checks.

## Credentials

Laptop `.env.test.local` is private, gitignored, mode 600. Inspect names/existence
without printing values before telling the user to source it.

| Variable | Use |
|---|---|
| `AUDREY_EVAL_BASE_URL` | API eval URL, normally `http://192.168.1.11:8000/v1` |
| `AUDREY_EVAL_API_KEY` | Audrey PAT with `compat:full` for compatibility calls |
| `AUDREY_USER_JWT` | Current Cloudflare Access application assertion for native user smokes |
| `AUDREY_ADMIN_JWT` | Distinct administrator's application assertion when required |

Tower `.env.smoke.local` contains only the two native JWT names. Use plain
`KEY=value`, no `export` or Markdown URLs. Application JWT values expire;
refresh them in place from the Audrey application's `CF_Authorization` cookie
or the existing cloudflared token workflow. A PAT is not a Cloudflare assertion.
Never print credentials, commit them, or paste them into chat.

## Choosing a check

Test the changed feature and its direct failure boundary. Inspect the real
script's environment names and arguments before handing off a command.
Prefer one targeted protocol case over a broad suite. Native upload/Files/chat
flows use manual browser acceptance; say whether an upload is required.
Create a focused reusable smoke when changed concurrency, deployment, restart or
service integration behavior benefits from it. Verify the runner locally, supply
its exact deployed command, and distinguish isolated checks from live-model evidence.

Every handoff states where commands run, what must rebuild/restart after pull,
and the visible pass condition. Stop at a failed step before a dependent one.
Laptop hermetic verification does not prove deployed behavior.

Short checks return results in the launching shell. For short direct Ollama
probes use `FOREGROUND=1` and a compact summary if supported; Telegram is skipped
unless `NOTIFY=1`. Detached runners are for long jobs needing disconnect safety.
Pings, exit codes and reports follow the [testing protocol](testing-protocol.md),
shared with ai-sec. API evals normally save to laptop `evals/results/`. If a necessary Tower report
is saved remotely, supply an explicit laptop copy step; do not rerun to retrieve
it. See [evals](../../evals/README.md) only for an actual evaluation task.

Delete temporary handoffs after acceptance. Keep a brief phase status and
regression code; retain model evaluation artifacts needed for decisions.

## Installed admission smoke

After deploying the interactive/background admission slice, run this on Tower:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
tests/smoke/smoke-agent-admission-onbox.sh
```

The wrapper streams the smoke into the running `audrey` container using its
installed package and `/opt/venv/bin/python`, and reads `/app/config.yaml`.
These smoke-only additions do not need an image rebuild. An older image without
admission support reports setup error 2; deploy the backend code/configuration
first using the service rebuild guidance below. No credentials, `.env.smoke.local`,
upload, model generation, network request or production account data is needed.

Expect five `PASS` lines and `PASS admission smoke (exit 0)`. Configured GPU/user
capacities and aging are printed separately from the deliberate cap-one fixture.
The checks cover configured capacity/cancel cleanup, chat passing queued background
work, GPU aging, running-call non-preemption and cancellation during a handoff.
Unaged-priority/handoff fixtures use a separately reported aging threshold of at
least 60 seconds so very short supported settings cannot age them prematurely.
The GPU-aging case uses the configured threshold and a backdated monotonic timestamp.
All overrides belong to isolated test queues; the running server's config is not changed.
The smoke normally takes a few seconds, with a ten-second deadline per check.
Short results return directly in the launching shell; no report transfer or
notifications are required. `--json` produces a compact machine-readable receipt.

This verifies installed scheduler/configuration behavior with isolated queues.
It does not inspect the server process's active queues, generate an answer, or
measure live GPU latency or HTTP wiring. Live contention acceptance remains separate.
Exit codes: 0 pass, 1 finding/timeout, 2 setup failure, 3 unexpected crash, 130 interrupted.
For a different checkout/container use `AUDREY_SMOKE_APPDATA` (or `APPDATA`) and
`AUDREY_SMOKE_CONTAINER`; forwarded Python options include `--config`, `--json`
and `--timeout-seconds` (positive, at most ten seconds).

## Service rebuilds

The user runs these actions on Tower from /mnt/user/appdata/audrey_ai_2.0 after
the normal pull. They describe how to deploy a change, not an outstanding queue.

| Changed files | Tower action |
|---|---|
| Documentation only | None |
| Backend Python or dependencies | `docker compose up -d --build audrey` |
| Browser source or assets | `docker compose up -d --build audrey-ui` |
| Shared KB code used by backend and tools | Rebuild `audrey` and `custom-tools` |
| Tools sidecar source | Rebuild `custom-tools`; also rebuild `audrey` when its code/declarations change |
| Media worker or fetcher source | Rebuild the corresponding `media-worker` or `media-fetcher` service |
| config.yaml or service environment | `docker compose up -d --force-recreate audrey`, using the affected service and adding `--build` when code also changes |
| Evaluation image contents/dependencies | `docker compose --profile eval build audrey-eval` |
| Monitoring source/provisioning | Follow [monitoring operations](../../monitoring/README.md) for the changed artifact |

Tool discovery refreshes automatically. A service restart alone does not reload
an updated bind-mounted configuration. Secrets and host-specific settings remain
in .env; do not edit the repository configuration directly on Tower.
