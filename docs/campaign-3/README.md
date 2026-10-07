# Campaign 3 — main plan

**Updated October 7, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

- **Passed:** model outcome/usage telemetry (formerly slice 11A.1), confirmed
  by the user October 7. Navigation, Bots, token revocation, Files, uploads,
  Projects, and the native cutover remain accepted.
- **Passed:** [17A.2 model monitoring dashboard](phase-17-operations-and-api.md),
  confirmed by the user October 7. **Built:** 17B reasoning controls; targeted API acceptance pending.
- **Independent followups:** 13A inline-image protocol proof, Tailscale
  reachability, and decision-relevant observations from existing model traffic.
  These do not reopen accepted features or require a broad test sweep.

## Phases

| Phase | Document | Status |
|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening.md) | Complete |
| 02 | [Native application](phase-02-native-application.md) | Complete |
| 03 | [Skills](phase-03-skills.md) | Explicit skills complete; automatic selection parked |
| 04 | [File downloads](phase-04-file-downloads.md) | Complete |
| 05 | [Responses foundation](phase-05-responses-api.md) | Complete |
| 06 | [Router candidates](phase-06-system-one-routing.md) | Complete; retain qwen3.5:4b |
| 07 | [Scanned PDFs](phase-07-scanned-pdf-ocr.md) | Complete |
| 08 | [Audio ingestion](phase-08-audio-ingestion.md) | Complete |
| 09 | [Broader audio](phase-09-broader-audio.md) | Complete |
| 10 | [Answer provenance](phase-10-ordinary-answer-provenance.md) | Complete |
| 11 | [File explorer](phase-11-native-file-explorer.md) | Complete |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Complete |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | 13B–13F complete; 13A proof open |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Complete |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Complete |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Retired; Hermes uses its existing stack |
| 17 | [Operations and API improvements](phase-17-operations-and-api.md) | Monitoring complete; reasoning controls built |

## Remaining order

1. Accept the built reasoning controls with the targeted API check, then
   diagnose actual retry/quota failures only where evidence identifies a gap.
2. Close the small 13A inline-image proof when the user runs its laptop check.
3. Diagnose Tailscale from laptop route through Tower firewall/published port
   before editing configuration. Use the working LAN/WARP address meanwhile.
4. Read existing Kimi K3 panel and DeepSeek capability evidence. Request only
   an observation needed for a model decision; keep Kimi first in cloud panels
   and the current router. Update the model ledger when something is measured.
5. Optional required tools/catalog expansion follows only for a caller with a
   concrete Responses adoption benefit. Scope is in Phase 17.

## Standing decisions

- Hermes keeps its working Chat Completions adapter: Kimi K3 primary, GLM 5.3
  fallback. The GPT bot is outside this work. No forced protocol migration.
- Saved Responses/continuation already exist. They reduce client payload and
  history bookkeeping; Audrey still sends full retained context to the model.
  They do not establish input-token or billing savings.
- Hermes executes and approves its tools. Audrey validates model requests;
  required tool selection would not authorize execution.
- Further automatic-selection research and non-English input work are parked.
  Historical eval artifacts remain evidence, not new acceptance requirements.
- Background generation, webhooks, Conversations API, historical imports,
  broader media analysis, and model/config experiments require a real need.
- Document/spreadsheet authoring belongs to the existing Hermes Bot Tools MCP,
  Nextcloud, and Collabora setup, not Audrey's browser experience.
- Analyze ordinary accumulated traffic for cost/quality questions; do not
  generate workloads simply to satisfy an old checklist.

## Deploy after git pull

**Run deployment commands on Tower**, in `/mnt/user/appdata/audrey_ai_2.0`.
Pull the user's committed changes there first. Run only the rows relevant to
the slice; its handoff must state which rows apply.

| Changed files | Tower action after pull |
|---|---|
| Documentation only | None |
| Backend Python/dependencies | `docker compose up -d --build audrey` |
| Browser code/assets | `docker compose up -d --build audrey-ui` |
| Shared KB code used by both services | Rebuild `audrey` and `custom-tools` |
| Tools sidecar code | Rebuild `custom-tools`; restart `audrey` for startup discovery |
| Media worker/fetcher code | Rebuild the corresponding `media-worker` or `media-fetcher` |
| `config.yaml` or service environment | `docker compose up -d --force-recreate audrey` (use affected service; add `--build` when code changed) |
| Eval image contents | `docker compose --profile eval build audrey-eval` |
| Grafana dashboard JSON | No rebuild/restart; directory mount reloads within 30 seconds |
| Prometheus rules | Reload Prometheus after pull; see [monitoring](../../monitoring/README.md) |
| Monitoring Compose/provisioning | Run its Compose commands from `monitoring/`; provisioning needs Grafana restart |

Backend/API clients use `http://192.168.1.11:8000`. Browser testing uses
`https://ai.builtryte.xyz`. Tower's browser origin `127.0.0.1:8090` is private
loopback, not a laptop address. Tailscale `100.113.157.98:8000` remains unverified.

## Testing handoffs

Every handoff names **Tower or laptop**, the directory, rebuild/restart action,
the exact command or browser steps, and what a pass looks like. Mention whether
a file upload is needed and whether the test generates model calls.

Hermetic tests/lint run on the laptop; Tower runs Docker, not a host `.venv`.
Native Files/upload/chat acceptance uses the browser and real files. API-only
checks normally run on the laptop over LAN/WARP. Short results return in the
launching shell. Reports needed for evaluation stay accessible on the laptop.
Use the [short runner/credential reference](../reference/live-smoke-testing.md)
only when handing off a live command. Delete temporary instructions after a
pass; phase status is enough. Keep regression code and model evaluation data.
