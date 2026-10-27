# Campaign 4 — main plan

**Updated October 26, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

**Audit followup completed October 24.** The local course AUDIT records the
completed drain of its remaining queue. Source/tooling is locally verified;
runtime contracts are in [Phase 1](phase-01-platform-hardening.md). The user accepted
the runtime handoff October 24. This does not reopen accepted product work or change the
Tax review queue below.

**Accepted foundation:** the owned IRS library, Private Projects and local file
processing, conversation recovery, Tax review worksheet/task lifecycle and Toolbox
presentation remain closed. The editable direct-chat facts workflow is accepted.
The [Phase 20 contract](phase-20-reference-libraries-and-private-tax-assistance.md)
holds those behaviors; the build queue below records completed milestones.
The collector, index and embedding migrations need no further run.

**20E.1 diagnostic closed with a known limitation, October 23.** The original
six-case measurement passed five in both arms; checking improved none and approved
the unsupported deduction. Followups now ask the missing facts, withhold the
derived result and stop without approval. The latest raw applied-deduction field
still contradicts that unresolved state. Retain this failure in the model ledger;
do not tune prompts or repeat this scalar test until green. This diagnostic is
finished, not a full tax-accuracy pass. Other five passes stay recorded.

**20E.2 first native outcome, October 23:** the final cited worksheet gave
$50,000 − $15,750 = $34,250 and kept $6,000 withholding separate. Its first answer
applied the deduction before the necessary eligibility facts were supplied, so
this is not an accuracy pass. Redundant confirmations and a blank continuation
also made the flow unclear. Keep the concrete eligibility limitation recorded;
the closed scalar probe stays closed.

**Facts workflow accepted October 23.** Direct chat facts, the editable Toolbox
table and clear missing-information questions passed. Do not repeat that flow.

**Tax review projects and scope accepted October 24.** The user reported the
remaining checks passed. Automatic Private grouping, canonical facts/worksheets,
document access and scope presentation stay closed. The audit runtime handoff
also passed. The tax-year comparison has now been assessed once below.

**Composer Stop polish built October 24.** A red square icon replaces the word
Stop in a button matching Send's shape and dimensions. The user's followup makes
the inner square larger while keeping the outer control unchanged. Accessible
naming and cancellation stay intact. Rebuild audrey-ui for this UI-only change.

**Runtime maintenance built October 23.** The approved model/Private/research
audit is addressed and locally verified; the local course `AUDIT.md` holds the drain.
The retired client integration is removed. [Phase 2](phase-02-native-application.md#retained-compatibility-boundaries)
explains the retained data/protocol contracts. The affected lessons describe the
current behavior. Deploy backend, UI and sidecar together using the handoff below.

**Sidecar maintenance built October 24.** The approved Cycle 2b source changes
are locally verified and recorded in the local course AUDIT. Runtime behavior
is in [Phase 1](phase-01-platform-hardening.md#tool-availability-at-runtime).
Deploy audrey and custom-tools together. Lesson files are left to the author;
this handoff provides an outline only. Product queue order stays below.

**20E.2 scope decision implemented October 24.** Tax review supports selected
source review and conditional arithmetic worksheets; it does not determine
eligibility for deductions or credits. The server projects this boundary on new
and historical worksheets, the Toolbox and project views show it before findings,
and tool feedback instructs the writer to keep model-selected rule amounts conditional. A completed task
or checked support assessment does not establish entitlement. The premature
deduction remains a known model limitation, not a corrected accuracy result.
The user accepted this presentation October 24. No accuracy claim follows
from that acceptance.

**20E.2 comparison assessed October 24: incomplete.** The answer supported one
2025 EITC income-limit topic and explicitly left the other topics and 2024
comparison unresolved. Better uncertainty reporting does not pass comparison
coverage. The [retained assessment](../../evals/results/2026-10-24-tax-review-native-comparison.md)
records the evidence and missing execution metadata. The model's budget claim
does not identify the stopping cause. Keep broad comparison coverage as a known
limitation, with exact-source review and conditional arithmetic supported.

**Targeted source lookup passed October 24.** The same-case answer identified
December 2025 Form 8911 instructions, PDF page 1, and June 30, 2026 correctly.
Passage retention is not established: the answer said it could not save, while
the Toolbox showed a saved review. The broader comparison remains incomplete.

**20F accepted October 26 — saved review files.** The downloaded workbook
contains the complete Form 8911 termination-date excerpt, original extraction
spacing, both dates and citation/provenance. The older short fragment remains
visible. Passage retention and the chat XLSX download are now passed. Canonical
case data remains authoritative; Tax exports use a bounded revision cache, separate
from My files. Broader comparison/eligibility limits remain as recorded above.

**Shared generated files accepted October 26.** The user reports all three
ordinary DOCX/XLSX and Private Project Markdown outputs worked. Creation and owned
chat/My files/Project delivery stay closed. Generated outputs remain separate
from source evidence and Private renders locally; application schema stays 25.

**11A accepted October 26 — larger previews and native-style appearance.**
The user accepted section/row navigation, document paper, spreadsheet grids,
rendered Markdown and text/code presentation. Numeric spreadsheet values remain
left aligned. No repeat test is queued for that slice.

**11B built — uploaded PDF previews.** Owned ordinary and Private PDFs can show
original pages locally in My files, Projects and saved chat attachments. My files
keeps Summary and Transcript alongside Original PDF; navigation and zoom use
bounded PNG pages rather than an external viewer. Download still serves the
unchanged original. Backend/component and desktop/mobile browser checks,
typecheck, build and scoped lint pass; existing large-component lint diagnostics
are unchanged. User acceptance is pending. Rebuild audrey and audrey-ui after pull. No
schema/config/dependency/CSP/model/index change. [Phase 11](phase-11-native-file-explorer.md#uploaded-pdf-previews--11b)
owns the contract and the one existing-PDF browser check. The next unbuilt slice
is richer actual DOCX/XLSX formatting and supported preview fidelity (11C).

No broader repeat test is queued. A later
decision to improve broader autonomous review starts from this saved run's
actual tool trace, task usage and stop reason. Do not automatically raise caps,
change models, reindex or reopen the concluded scalar diagnostic.

**Toolbox worksheet layout followup:** reserve a complete worksheet-summary row
on short screens and scroll the expanded content rather than clipping its action.
Only audrey-ui rebuilds for this presentation fix.

**Toolbox edge-tab followup built:** the closed Toolbox opens from a larger,
vertically centered right-edge tab, with sideways text facing left. It hides while
the panel is open and restores keyboard focus on close. Only audrey-ui rebuilds.

**Private header followup built:** a subdued non-action status badge sits near
Archive/Delete, between those controls and the conversation title. Hover/focus
reveals processing/storage, administrator, Cloudflare, tool-scope and mode-lock
details; touch can reveal them and Escape/outside interaction dismisses them.
The compact header retains the badge while destructive chat actions stay in mobile
navigation. Composer privacy prose is removed. Tax review's filing caution and
settings behavior stay accepted. This presentation flow is accepted; no repeat check is queued.

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
| 11 | [File explorer](phase-11-native-file-explorer.md) | Creation/paged previews accepted; uploaded PDF preview built; richer formatting next |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Desktop/mobile followup built; browser acceptance pending |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | Complete |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Complete |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Complete; shared overlay menus await acceptance |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Retired; Hermes uses its existing stack |
| 17 | [Operations and API improvements](phase-17-operations-and-api.md) | Monitoring complete; reasoning built; further bot work deferred |
| 18 | [Admin controls and project presentation](phase-18-admin-and-project-presentation.md) | 18A complete |
| 19 | [Calculations and deeper evidence tools](phase-19-evidence-tools.md) | Complete; user accepted October 13 |
| 20 | [Reference libraries and private tax assistance](phase-20-reference-libraries-and-private-tax-assistance.md) | Product flow accepted; comparison assessed incomplete with known coverage limits |

## Build queue

- [x] **20A–20B:** owned IRS library, published search and cited retrieval.
- [x] **20C:** Private document/calculator flow and successful IRS retry; planner
  fallback and consistent durable admission are in place.
- [x] **20D.1:** Private Projects, direct protected upload, project evidence,
  refresh/isolation and Local-to-Private new-choice consolidation.
- [x] **Private chat followup:** delete-to-new-chat, model switching before the
  first message and MiniMax cloud configuration accepted.
- [x] **Private UI:** unused Audrey portrait and activity-sorted history accepted.
- [x] **Reference request repair:** oversized counts bounded; content search
  and page reads accepted. General tax accuracy remains unverified.
- [x] **Current UI followup:** Project header delete/checkmark and interrupted
  chat deletion recovery accepted; Cloudflare diagnostics closed.
- [x] **20D.2:** saved facts/evidence and deterministic case checklist accepted.
- [x] **Tax review presentation:** bounded settings, grouped conversational
  choices and automatic follow-up accepted.
- [x] **20D.3a:** explicit reference-library updates with retained evidence and
  required rechecks; locally verified.
- [x] **20D.3b:** foreground worksheet and limited numerical/separation baseline.
  Broader accuracy continues in 20D.5/20E.
- [x] **20D.4a:** durable owner-scoped task/checkpoint storage and recovery fences.
- [x] **20D.4b.1:** bounded step executor, durable resource accounting and atomic
  task/case write fencing.
- [x] **20D.4b.2a:** current native owner/model/source admission and preparation,
  exact source pins, permission fencing and explicit revalidated resume.
- [x] **Tax/Private notices:** concise disclaimers and settings-button polish.
- [x] **20D.4b.2b built:** native saved reviews, progress/control, safe restart
  recovery and idempotent delivery; user moved on from the lifecycle handoff.
- [x] **20D.5 built:** bounded local evidence checking and targeted repair;
  Toolbox presentation accepted October 22.
- [x] **20E.1 measurement:** six fixed-passage pairs completed; five passed both arms.
- [x] **20E.1 diagnostic concluded:** missing-fact stop and withheld result
  observed; retain the raw applied-value inconsistency as a known failure.
  No further scalar-probe rerun is queued; other five passes stay recorded.
- [x] **Facts followup:** editable current facts, direct chat statements and
  clear missing-information flow accepted.
- [x] **Tax project acceptance:** automatic Private grouping, saved
  facts/worksheet and project document access accepted October 24.
- [x] **Runtime maintenance:** approved audit source changes and retired-client
  cleanup implemented; see the local course AUDIT for the drain.
- [x] **20E.2 supported scope:** source review and conditional arithmetic;
  eligibility is not determined. Scope metadata and worksheet presentation are
  implemented and user-accepted October 24. The earlier premature deduction stays recorded.
- [x] **20E.2 comparison assessment:** completed once October 24; outcome is
  incomplete coverage, not a passed comparison or eligibility-accuracy result.
  Retain the limitation; no repeated broad test is queued.
- [x] **Targeted Form 8911 lookup:** correct edition/page/cutoff reported;
  full-passage persistence and workbook delivery subsequently passed in 20F.
- [x] **20F accepted — review files:** retained full source passage and real chat
  workbook download passed October 26. No repeat save/download test is queued.
- [x] **Shared file creation accepted:** all three ordinary DOCX/XLSX and Private
  Project Markdown outputs worked; no repeat generation/download smoke.
- [x] **Browser previews foundation:** local read-only generated-file and saved
  Tax workbook views; spreadsheet numeric values align left.
- [x] **11A accepted — larger previews and native-style appearance.** Bounded
  previous/next browsing for text/document sections and worksheet rows, with
  paper/grid/Markdown/text surfaces. User accepted October 26; no repeat check.
- [x] **11B built — uploaded PDF preview.** Original-page navigation and zoom
  for owned ordinary/Private PDFs in My files, Projects and saved chat attachments.
  Local verification passes; user acceptance pending. Phase 11 owns the check.
- [ ] **11C — generated-file formatting and preview fidelity.** Improve actual
  DOCX/XLSX headings, lists, tables and worksheet formatting, then reflect
  supported formatting in previews. Preserve literal cells and safe local
  generation; no office editor or Hermes adapter changes.

The approved file-preview followups run in that order. Passed creation and Tax
save/download work stays closed; deferred work remains deferred.

Tax year/jurisdiction and reference release remain in the case record. Project
views read that canonical record rather than duplicate it in Project instructions. OCR/media, fine-tuning and state
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
- Audrey may generate user-owned documents/spreadsheets for general chat tasks.
  Shared file creation is built in Phase 11; the existing Tax exporter is one
  scoped output path. Interactive office editing/collaboration stays in the
  existing Hermes Bot Tools MCP, Nextcloud and Collabora setup; no adapter change.
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
| Tools sidecar code | Rebuild `custom-tools`; Audrey refreshes discovery automatically. Rebuild `audrey` too when its code/declarations changed |
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
