# Campaign 4 — main plan

**Updated October 10, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

- **12 — compact/mobile navigation:** unified left drawer built; browser
  acceptance pending. Header: Menu, builtryte logo, New chat. Drawer: user name,
  My Files, collapsible Projects, and swipeable chat history; Admin Panel and
  Log out stay at the bottom. Right swipe archives/restores; left reveals
  Delete/Cancel. Project sits beside Model; empty/short chats have no jump arrow. Loading periods
  animate, profile avoids field autofocus, and pull-to-refresh requires a
  deliberate downward pull followed by release. Desktop Archive/Delete and
  composer controls are larger; Model focus highlights only the outer control.
  Async history actions preserve a chat selected while the request is pending.
  These UI changes rebuild `audrey-ui`.
- **Platform audit fixes:** implemented October 10: authenticated tool catalog,
  safe rediscovery, retired legacy upload page, corrected cache headers, and
  current lifecycle/Compose comments and icon. Rebuild `audrey` and `audrey-ui`;
  recreate `custom-tools` for its icon label. Deployment remains pending.
- **3D.6 — native automatic English skill selection:** built; deployed
  document/video selection and successful video file reads are confirmed.
  `skills.auto_select: true` is now the repository default at the user's request,
  matching Tower's enabled setting. Deterministic rules use ready owned attachment
  and Project metadata; explicit choices win and unclear targets abstain.
  Remaining browser acceptance is pending.
- **Composer/provenance followup:** built, awaiting browser acceptance. Files and
  Tools & skills open below the composer. Sources, Models, and Tool calls stay
  beside each completed answer immediately, even with progress display off.
  Kimi K3 is included in the tool capability gate used by native panel workers.
- **18A — admin controls and project presentation:** accepted October 7.
- **Closed by the user October 7:** old domain cleanup and the model/runtime
  evidence and traffic-analysis followups. These require no further campaign
  work or testing; do not manufacture new measurements for the model ledger.
- Existing native features, 13A–13F, accounts/tokens, Projects, telemetry, and
  the Grafana model dashboard remain accepted. 17B reasoning controls are built;
  their live API acceptance and benefit to the existing Hermes adapter remain
  unverified, with no further bot integration work queued.
- **Deferred by the user:** Tailscale repair and 17D Responses tool contracts /
  catalog expansion. Keep the working Hermes adapter and settings.

## Phases

| Phase | Document | Status |
|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening.md) | Complete |
| 02 | [Native application](phase-02-native-application.md) | Complete |
| 03 | [Skills](phase-03-skills.md) | Explicit skills complete; native automatic selection built, acceptance pending |
| 04 | [File downloads](phase-04-file-downloads.md) | Complete |
| 05 | [Responses foundation](phase-05-responses-api.md) | Complete |
| 06 | [Router candidates](phase-06-system-one-routing.md) | Complete; retain qwen3.5:4b |
| 07 | [Scanned PDFs](phase-07-scanned-pdf-ocr.md) | Complete |
| 08 | [Audio ingestion](phase-08-audio-ingestion.md) | Complete |
| 09 | [Broader audio](phase-09-broader-audio.md) | Complete |
| 10 | [Answer provenance](phase-10-ordinary-answer-provenance.md) | Complete; immediate answer summaries await acceptance |
| 11 | [File explorer](phase-11-native-file-explorer.md) | Complete |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Desktop/mobile followup built; browser acceptance pending |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | Complete |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Complete |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Complete; menus below composer await acceptance |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Retired; Hermes uses its existing stack |
| 17 | [Operations and API improvements](phase-17-operations-and-api.md) | Monitoring complete; reasoning built; further bot work deferred |
| 18 | [Admin controls and project presentation](phase-18-admin-and-project-presentation.md) | 18A complete |

## Remaining work

1. Accept the compact/mobile layout on a phone and a narrow laptop window:
   left menu/footer, chat swipes, New chat, Project beside Model, empty-screen fit,
   profile keyboard behavior, normal scrolling, and deliberate pull-to-refresh.
   No upload or scripted live eval is required.
2. Accept the below-composer menus and immediate per-answer tool summaries in
   the browser, reusing the existing Ready video. Recorded video reads are passed;
   the remaining defect was their display, not missing tool execution.
3. Close remaining 3D.6 browser acceptance. Activation is already approved and
   recorded in config. No additional campaign work is required without a new
   user request.

Domain cleanup and operational evidence/traffic followups are closed by user
confirmation. Tailscale, 17D, and further Hermes changes stay deferred. 17C is
only a future diagnostic note for an actual reported failure.

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
