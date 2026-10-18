# Campaign 4 — main plan

**Updated October 17, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

- **20 — reference libraries and private tax assistance:** public collection
  is sufficient to proceed: logs confirm 26,706 downloaded/verified PDFs; the
  only known source failure is the corrupt 1976 Form 940. No further broad run.
  **20B.1 accepted October 14:** named IRS catalog, exact lookup and bounded page reading.
  Models see current official editions by default; explicit historical access
  is limited to the most recent five years (2022–2026). Older originals remain
  outside the accessible library. Tax-year applicability is verified separately
  from catalog/revision dates. Deployed tools, page citations and historical
  refusal are user-accepted. **20B.2 built:** local scoped hybrid content search,
  resumable bounded indexing, verified publication and page citations. The full
  Tower build saved 2,398 documents and 61,042 chunks; nine extraction failures
  prevented publication. Repairs preserve completed checkpoints, support freely
  readable permissions-encrypted PDFs, bound parser cache growth and report precise
  failures. Local public-source checks cover eight failed documents. The user
  approved Publication 5902 as an explicitly disclosed content-search gap; its
  original remains in the catalog. Exact-document, saved-release approval is
  implemented; unapproved failures still block publication. Search and reports
  retain the failed count, incomplete coverage and named gap. Deploy `audrey`
  after pull, then resume the saved release with its single approved gap.
  Live publication and cited browser acceptance remain pending.
  [Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md) holds
  the contract and remaining privacy/tax workflow. State jurisdictions remain
  unspecified; fine-tuning requires evidence.

- **Embeddings:** the broad Qwen migration and the later history-only Nomic
  restoration are deployed and user-accepted. October 17 acceptance passed the
  arithmetic paraphrase and absent-calculation checks; do not repeat them or
  either migration. Documents, files, references and memories retain Qwen4;
  history uses pinned Nomic/768 with its own readiness and formatting.
  **Document double-check completed October 17:** matched measurements with
  reference and actual upload chunk layouts support retaining Qwen for documents.
  All requests completed; unsupported-detail rejection remains a quality limit
  for every model. No threshold or model change follows. Reused public questions
  do not establish independent new-topic evidence. [Phase 06](phase-06-system-one-routing.md)
  holds the contract; measurements and limits are in MODEL-FACTS. The existing-file
  browser check remains unreported; IRS publication recovery is tracked above.
  Router assessment stays closed: retain Qwen 3.5 4B.

- **19 — deeper evidence tools:** complete; user accepted all five tools October 13.
  [Phase 19](phase-19-evidence-tools.md) defines calculator/conversions, CSV/XLSX
  analysis, PDF-page inspection, previous-conversation reading and video-frame
  inspection. Existing chat disclosures show their activity.

- **12 — compact/mobile navigation:** unified left drawer built; browser
  acceptance pending. Header: Menu, builtryte logo, New chat. Drawer: Username -
  My chats, search, Active/Archived, and history. The uniform bottom group holds
  collapsible My projects with a plus button and separators, My Files, Admin Panel,
  then Log out; expanded projects scroll independently. Right swipe archives/restores; left reveals
  Delete/Cancel. Project sits beside Model; empty/short chats have no jump arrow. Loading periods
  animate, profile avoids field autofocus, and pull-to-refresh requires a
  deliberate downward pull followed by release. Desktop Archive/Delete and
  composer controls are larger; Model focus highlights only the outer control.
  Async history actions preserve a chat selected while the request is pending.
  Latest polish: Cloudflare bootstrap shows **Processing Cloudflare Authentication**
  / **Please wait...**. Its portrait rim fills/erases in 60-second passes;
  ordinary loading uses two-second passes. The small orb is partly hidden
  by the photo, with backlight against its edge. Main portrait hover adds that
  close glow without brightening the photo. Projects uses a shared layers icon; the desktop divider
  follows the sidebar edge with balanced logo spacing and **AUDREY** in spaced
  capitals, with a larger initial A and a softer slate color.
  These UI changes rebuild `audrey-ui`; laptop verification passed, latest Tower
  deployment and browser acceptance are unconfirmed.
- **Course review:** the platform and system-map reviews are recorded in
  the local, gitignored [AUDIT](../lesson-ai/AUDIT.md). The current checkout
  requires rebuilding `audrey`, `custom-tools`, and `audrey-ui` after pull;
  deployment remains pending. Memory backup operation and verification are in
  the [backup guide](../reference/box-operations.md#10-memory-backups-and-isolated-restore-verification).
- **3D.6 — native automatic English skill selection:** built; deployed
  document/video selection and successful video file reads are confirmed.
  `skills.auto_select: true` is now the repository default at the user's request,
  matching Tower's enabled setting. Deterministic rules use ready owned attachment
  and Project metadata; explicit choices win and unclear targets abstain.
  Remaining browser acceptance is pending.
- **Composer/provenance followup:** built, awaiting browser acceptance. All four
  composer pickers use matching dark overlays directly above their controls,
  overlapping the message field without shifting the composer or portrait.
  Model aligns left, Tools right, Files/Project centered, with viewport clamping. Sources, Models,
  and Tool calls stay
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
- **Deferred by the user:** password login, Tailscale repair, and 17D Responses
  tool contracts / catalog expansion. Keep current Cloudflare sign-in and the
  working Hermes adapter and settings.

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
| 11 | [File explorer](phase-11-native-file-explorer.md) | Complete |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Desktop/mobile followup built; browser acceptance pending |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | Complete |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Complete |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Complete; shared overlay menus await acceptance |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Retired; Hermes uses its existing stack |
| 17 | [Operations and API improvements](phase-17-operations-and-api.md) | Monitoring complete; reasoning built; further bot work deferred |
| 18 | [Admin controls and project presentation](phase-18-admin-and-project-presentation.md) | 18A complete |
| 19 | [Calculations and deeper evidence tools](phase-19-evidence-tools.md) | Complete; user accepted October 13 |
| 20 | [Reference libraries and private tax assistance](phase-20-reference-libraries-and-private-tax-assistance.md) | 20A/20B.1 accepted; 20B.2 built, indexing/browser acceptance pending |

## Remaining work

1. Document comparison is complete; retain Qwen for documents and Nomic for
   history. Preserve accepted history/memory checks. New conversation focus and
   the existing-file browser check remain unreported. Resume the saved current-
   catalog IRS index with the approved Publication 5902 search gap, then accept
   one cited content search in the browser
   (20B.2). Follow with 20C enforced private processing, 20D's tax workflow and
   targeted 20E evaluation. Broad archive collection and both migrations stay
   complete; no repeat is queued.
2. Confirm Tower deployment and close pending targeted acceptance for the
   current course-reviewed work. The local, gitignored [AUDIT](../lesson-ai/AUDIT.md)
   holds its review record; the [operations guide](../reference/box-operations.md#10-memory-backups-and-isolated-restore-verification)
   holds the backup check. Reuse passed checks instead of restarting the suites.
3. Accept the latest branding and compact/mobile layout: stylistic **AUDREY** label,
   balanced desktop divider, Cloudflare loading message, distinct Projects icon,
   left menu/footer, chat swipes, New chat, Project beside Model, empty-screen fit,
   profile keyboard behavior, normal scrolling, and deliberate pull-to-refresh.
   No upload or scripted live eval is required for these UI checks.
4. Accept the shared composer overlays and immediate per-answer tool summaries in
   the browser, reusing the existing Ready video. Recorded video reads are passed;
   the remaining defect was their display, not missing tool execution.
5. Close remaining 3D.6 browser acceptance. Activation is already approved and
   recorded in config. Passed work remains closed.

Domain cleanup and operational evidence/traffic followups are closed by user
confirmation. Password login, Tailscale, 17D, and further Hermes changes stay deferred. 17C is
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
