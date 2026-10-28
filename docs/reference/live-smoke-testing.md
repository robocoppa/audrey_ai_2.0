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
