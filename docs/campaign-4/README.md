# Campaign 4 — main plan

**Updated October 20, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

**20D.1 accepted October 20:** Private Projects, protected uploads/context,
refresh/isolation and new Private model choices passed. The user reported three
followups: deletion should open a clean draft, Private should remain changeable
before the first message, and its portrait should match Local. Those fixes and
the requested MiniMax cloud addition are built; targeted browser checks remain.
[Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md) holds the
privacy contract; [navigation](phase-12-sidebar-navigation.md) holds deletion.

**MiniMax M3.1 Flash:** configured as an unmeasured cloud draft/research candidate
in mixed, Cloud and selected Research panels. Kimi K3 stays first, panel worker
counts/caps are unchanged, and existing synthesis/verification roles remain.
Private/Local pools and fast primaries are unchanged. Direct model discovery,
tool/vision metadata and the shared passthrough allowlist include the installed
tag. No performance promotion or new model pull is implied.

Private document/calculator and the owned IRS page-read retry remain accepted.
The collector, published index and both embedding migrations stay closed.
Qwen3 Embedding 4B serves documents/files/references/memory; pinned Nomic serves
history; Qwen3.5 4B remains the router. Measurements stay in the model ledger.

## Phases

| Phase | Document | Status |
|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening.md) | Complete |
| 02 | [Native application](phase-02-native-application.md) | Complete |
| 03 | [Skills](phase-03-skills.md) | Explicit skills complete; native automatic selection built, acceptance pending |
| 04 | [File downloads](phase-04-file-downloads.md) | Complete |
| 05 | [Responses foundation](phase-05-responses-api.md) | Complete |
| 06 | [Router candidates](phase-06-system-one-routing.md) | History restoration accepted; document comparison complete, retain split |
| 07 | [Scanned PDFs](phase-07-scanned-pdf-ocr.md) | Complete |
| 08 | [Audio ingestion](phase-08-audio-ingestion.md) | Complete |
| 09 | [Broader audio](phase-09-broader-audio.md) | Complete |
| 10 | [Answer provenance](phase-10-ordinary-answer-provenance.md) | Complete; immediate answer summaries await acceptance |
| 11 | [File explorer](phase-11-native-file-explorer.md) | Complete; upload presentation followup built |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Desktop/mobile followup built; browser acceptance pending |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | Complete |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Complete |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Complete; shared overlay menus await acceptance |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Retired; Hermes uses its existing stack |
| 17 | [Operations and API improvements](phase-17-operations-and-api.md) | Monitoring complete; reasoning built; further bot work deferred |
| 18 | [Admin controls and project presentation](phase-18-admin-and-project-presentation.md) | 18A complete |
| 19 | [Calculations and deeper evidence tools](phase-19-evidence-tools.md) | Complete; user accepted October 13 |
| 20 | [Reference libraries and private tax assistance](phase-20-reference-libraries-and-private-tax-assistance.md) | Private Projects accepted; draft/deletion/portrait followup built; tax review next |

## Build queue

- [x] **20A–20B:** owned IRS library, published search and cited retrieval.
- [x] **20C:** Private document/calculator flow and successful IRS retry; planner
  fallback and consistent durable admission are in place.
- [x] **20D.1:** Private Projects, direct protected upload, project evidence,
  refresh/isolation and Local-to-Private new-choice consolidation.
- [ ] **Current followup:** verify delete-to-new-chat, model switching before the
  first message and Local portrait; confirm MiniMax appears in Cloud activity.
- [ ] **20D.2:** dedicated `tax_review` tool with saved facts, evidence, missing
  items and a deterministic case checklist.
- [ ] **20D.3:** one complete foreground tax review and a known-answer baseline.
- [ ] **20D.4:** durable Private tasks, checkpoints, progress, cancel and resume.
- [ ] **20D.5:** bounded local evidence verification and targeted repair loops.
- [ ] **20E:** compare single-pass and extended accuracy before extending defaults.

Tax year/jurisdiction and reference release belong to the case record; 20D.1 does
not add premature tax-specific Project fields. OCR/media, fine-tuning and state
source adapters wait for a concrete supported case. English-only scope remains.

Older unreported browser polish is grouped in the existing phase contracts:
[Files](phase-11-native-file-explorer.md), [navigation](phase-12-sidebar-navigation.md),
[composer](phase-15-composer-and-projects.md), [provenance](phase-10-ordinary-answer-provenance.md)
and [skills](phase-03-skills.md). Observe them during normal use; they do not reopen
passed ingestion, retrieval or model checks. The existing-file browser check and
memory-backup operation remain unreported in their owning references. No extra
smoke document or separate todo file is needed.

Domain cleanup is closed. Password login, Tailscale, 17D and further Hermes work
remain deferred. 17C is a future diagnostic note for an actual reported failure.

## Standing decisions

- Hermes keeps its working Chat Completions adapter: Kimi K3 primary, GLM 5.3
  fallback. The GPT bot is outside this work. Bot API enhancements require
  a concrete benefit and explicit user direction before resuming parked work.
- Saved Responses/continuation already exist. They reduce client payload and
  history bookkeeping; Audrey still sends full retained context to the model.
  They do not establish input-token or billing savings.
- Hermes executes and approves its tools. Audrey validates model requests;
  required tool selection would not authorize execution.
- Automatic English skill selection is limited to native chat, deterministic
  file intent, and the two accepted bundles. A model-based classifier or broader
  activation requires demonstrated benefit. Non-English input work is out of scope. Historical eval artifacts remain
  evidence; they are not fresh acceptance data after tuning.
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

Tower uses the repository's tracked `config.yaml`; config changes go through
the same commit/push/pull flow. Avoid in-place Tower edits. If an old edit blocks
a pull, the [operations guide](../reference/box-operations.md#6-keep-tower-config-identical-to-the-repository)
has the one-time backup and cleanup command.

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
loopback, not a laptop address. Tailscale repair is deferred; use the working LAN/WARP route until the user asks to revisit it.

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
