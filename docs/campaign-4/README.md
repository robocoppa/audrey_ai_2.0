# Campaign 4 — main plan

**Updated October 22, 2026.** English user input only. Passed work stays closed.
This is the only campaign roadmap; each phase has one document containing its
behavior, decisions, and remaining work. Read this plan and the current phase.
Model measurements belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), consulted
when a model decision needs them. External assessments are already folded in;
they are not ongoing task lists.

## Current work

**20D.1 accepted October 20:** Private Projects, protected uploads/context,
refresh/isolation and new Private model choices passed. Delete-to-new-chat,
model switching before the first message and the MiniMax addition are accepted.
Private portrait, activity-sorted history and Automatic tools presentation are
accepted. The reference count repair is accepted: the latest Private comparison
completed content search and page reads. Accuracy remains open; retrieved pages
did not prevent outdated deduction figures and a wrong credit cutoff year.
Project deletion beside Edit and interrupted conversation-deletion recovery are
accepted. The original Cloudflare origin cause was not identified; diagnostics
are closed. [Navigation](phase-12-sidebar-navigation.md) holds the behavior.

**20D.2 and UI followup accepted October 21.** Tax review
settings open in a dialog bounded by the visible screen; the previous completed
year remains the editable default. Essential choices appear together in the
conversation. A saved choice becomes a visible user reply and Audrey continues
automatically, preserving unsent text and attachments. The compact box retains
settings and collapsed findings. Internal IDs/passing checklist rows stay hidden;
Audrey handles source verification herself. The generic degraded badge is removed
throughout chat, including after deletion; actual outage indicators remain.
The user moved on after the targeted presentation handoff; keep this flow closed.

**20D.3a built:** saved cases remain readable if the IRS library changes or is
unavailable. An explicit owner action adopts the exact available release;
confirmed personal facts and original passages are retained, while rule evidence
and related work require rechecking. Models cannot approve the update. Publication
races show a further update choice, without silently adopting it. No index rerun
or new library publication is needed to deploy this slice.

**20D.3b accepted for the limited worksheet:** a Tax review chat prompt runs one
local tool writer through the normal native run, with stop/reconnect and a ten-minute foreground limit. It saves
a goal-scoped worksheet with linked facts, official passages, arithmetic and
explicit limitations. Case edits make an earlier worksheet stale. The Toolbox
keeps findings expandable and user confirmations explicit. Selected-file inventory
is inspectable; it does not prove every page was read. The user-reported synthetic
baseline passed the requested arithmetic and year/withholding separation, with an
official passage retained. This limited pass does not establish legal accuracy or
certify a complete return. Broader accuracy and background work follow later.
[Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md) holds the contract.

**20D.4a built:** SQLite task/checkpoint foundation, with one active task per
Private conversation, expiring worker leases, revision/fence checks and replay-safe
checkpoints. Pause/cancel prevent late task writes; changed case/source/Project or
model/policy bindings require revalidation. Resource accounting and partial-work
stop reasons survive restart and existing backup/restore. Native integration is
built and accepted below.

**20D.4b.1 built:** one bounded leased step can use the verified Private runtime
and owned local tools. Tax-case changes and task revision advancement commit
atomically; cancellation, stale inputs and expired leases refuse late findings.
Observed token/tool costs persist during the step. An idle lease is not renewed,
and timeouts or failures retain partial work with a specific stop reason. This
executor now runs through the existing native run manager.

**20D.4b.2a built:** native task admission/preparation refreshes current account
and Private-model permissions, the case and published IRS release. Exact selected
originals, extracted text and semantic Project context are pinned by digest.
Permission changes are checked again inside case-write transactions. Explicit
resume revalidates inputs and retains spent resources; stale checkpoints cannot
start work.

**20D.4b.2b complete; user advanced past the lifecycle handoff October 22:** native Tax review turns create
or explicitly resume a saved task through the existing run manager and fair GPU
queue. One compact progress row provides Pause/Resume/Cancel; normal Stop cancels
the task. Refresh retains saved progress; review controls preserve unsent drafts.
Each attempt remains one
bounded local writer; generated-token usage is observed before output is accepted.
Checkpoint and final-answer outbox commit together, and delivery reuses the
original native message. Restart replays committed results, pauses unfinished
work and requires explicit resume with fresh validation. Restart never launches
model work. No repeat lifecycle check is queued.

**20D.5 and Toolbox accepted October 22.** Native worksheets receive bounded
source checks and targeted repair; owner confirmations and source applicability
retain their separate authority. Tax review settings/progress/choices live in the
right Toolbox rail (drawer on compact screens), with previous decisions above the
current prompt. Ordinary chat replaces idle Start/Continue actions. The open panel
hides its launcher; closing restores it and keyboard focus. Global scrollbars now
use thin muted blue thumbs. The user's functional/UI checks passed; broader tax
accuracy remains unverified.

**20E.1 built; live measurement pending:** a six-case fixed-passage diagnostic
compares the exact same initial local worksheet with the production evidence
checker and at most one repair/recheck. Independent hidden labels score amounts,
year/withholding separation, corrected/duplicate records, eligibility questions,
source selection and decimal rounding. Reports retain initial/final outputs,
regressions, false checker approvals, model/source/code identities and measured
costs. It uses synthetic confirmed records and frozen official excerpts, without
reading live cases or changing the library. This isolates check/repair behavior;
it does not measure native retrieval, uploaded-file extraction or full returns.
**Next:** read this measurement before choosing the smallest 20E.2 native
retrieval/accuracy comparison. No model promotion or default extension is implied.

**Notices and settings polish built:** Tax review shows a brief filing caution;
Private shows local server processing/storage with expandable privacy details.
Settings has a bordered dialog button without a dropdown arrow. Conversation
scrolling accounts for the complete composer height. The user moved on after
that handoff; no repeat visual check is queued.

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
| 20 | [Reference libraries and private tax assistance](phase-20-reference-libraries-and-private-tax-assistance.md) | Workflow accepted; fixed-passage accuracy measurement pending |

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
- [ ] **20E.1 built:** measure fixed-passage writer/check/repair quality; live results pending.
- [ ] **20E.2:** use those results to target the remaining native retrieval/accuracy
  comparison before extending defaults.

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
