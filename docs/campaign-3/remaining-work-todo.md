# Campaign 3 remaining work — items 5–10

**Status:** Items 1–6 were closed on 2026-10-01. The user reprioritized
[Phase 12](phase-12-sidebar-navigation.md) ahead of Items 7–10. Slice 12A is
laptop-complete with its My Files correction and awaits browser acceptance.
Item 7 has live-settled [Phase 13 Slice
13B](phase-13-responses-multimodal-input.md); Slice 13A remains laptop-complete
with its targeted live gate pending. Slice 13C adds owner-scoped image and
document references and passed its targeted protocol smoke on 2026-10-05.
Slice 13D's normal URL answer, private-URL guards, and corrected streamed retry
passed on 2026-10-05; its remote-input protocol gate is live-settled.
Slice 13E client-executed function tools passed its three-call protocol smoke
on 2026-10-05. Slice 13F saved Responses and text/function continuation is
live-settled after 3,665 laptop tests and the user's successful restart verify
received 2026-10-06. Claudette's cloud assessment is reviewed: keep the current
Kimi-primary/GLM-fallback Chat Completions connection; the 20-tool catalog exceeds
Responses admission. Further compatibility expansion needs a caller use case.
The user then
prioritized [Phase 14 Slice 14A](phase-14-bot-accounts-and-token-lifetimes.md),
which is laptop-complete and awaits native acceptance.
[Phase 15](phase-15-composer-and-projects.md) is complete. Slice 15A's composer control
rail passed user acceptance, Slice 15B's owner-scoped Projects storage and API
passed its restart smoke, and the user accepted Slices 15C–15D plus the
direct project-upload follow-up.
[Phase 16](phase-16-native-document-tools.md) is retired because document and
spreadsheet authoring already belongs to the separate Hermes bot workspace.

## Closed prerequisites

- [x] Deploy the retired-model and media-fetcher progress slice.
- [x] Pass the native Cloud turn with `glm-5.3:cloud` and `kimi-k2.6:cloud`.
- [x] Pass WAV, M4A, and FLAC native acceptance; Campaign 3 Phase 8 is complete.
- [x] Remove retired Ollama entries and return model inventory to clean.
- [x] Release and remove the stopped `audrey-ai-retired` cutover container.

## 5. Finish Phase 2D.5 administration and recovery proof

**State:** Complete and live-verified. The backup, account/model authority,
interactive provider and picker behavior, restart persistence, projection
rebuild, and isolated restore proof all passed.

### 5.1 Deploy and create the verified backup

- [x] Add `audrey-admin backup-app-state --to <new-path>`.
- [x] Refuse an existing destination, create mode `600`, use SQLite's online
  backup API, run `PRAGMA integrity_check`, and report the schema version.
- [x] Add hermetic CLI coverage.
- [x] Pass the full hermetic backend suite: 3,066 tests.
- [x] Rebuild `audrey` with the new command.
- [x] Create the persistent backup directory inside `/data`.
- [x] Create the pre-gate backup and preserve its JSON result.

On Unraid, after pulling the slice:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build --force-recreate audrey
docker compose exec audrey mkdir -p /data/backups
docker compose exec audrey audrey-admin backup-app-state --to /data/backups/audrey-app-before-2d5-20261001.sqlite
```

Success is exit code zero and JSON with `status: ok`, `integrity_check: ok`, a
positive byte count, and the current schema version. The command refuses to
overwrite a prior backup.

**Live result, 2026-10-01:** Passed. The backup at
`/data/backups/audrey-app-before-2d5-20261001.sqlite` is 675,840 bytes,
reported `integrity_check: ok`, and captured schema version 17.

### 5.2 Prove account and model administration

- [x] Confirm Tower's private `.env.smoke.local` contains accepted, distinct
  `AUDREY_USER_JWT` and `AUDREY_ADMIN_JWT` application assertions.
- [x] Rebuild Audrey and rerun the corrected account/model smoke through the
  standalone UI proxy.
- [x] Require `status: passed`, a successful direct-model run, zero tool events,
  restored access policy, restored publication profile, restored user access,
  repair `ready`, and no cleanup errors.

**First live attempt, 2026-10-01:** The smoke stopped before creating a
conversation because the ordinary account could see the selected direct model.
Cleanup passed and restored the account plus the original model state. The app
was applying a pre-existing Public publication profile whose roles still
included `users`; the old smoke changed only the separate enabled/audience
policy. The corrective changes both effective layers, reports their provenance,
and restores each source independently so an existing profile, display name,
and portrait survive. The full hermetic backend suite passes 3,068 tests.

**Corrected live result, 2026-10-01:** Passed. The users-only account received
HTTP 404 for the tester model, tester access completed one 66.571-second direct
Qwen turn with two canonical messages and zero tool events, disabled execution
returned HTTP 404 without appending messages, and repair reached `ready`.
Provider-only administration, PAT rejection, and self-disable/demotion
safeguards passed. Cleanup restored the access policy, publication profile,
user access, and removed the temporary conversation and archive projection.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_access_models.py
```

### 5.3 Complete the interactive browser checks

- [x] Sign in once with a genuinely new allowed Access identity and confirm it
  lands at the pending-account boundary.
- [x] Bootstrap the intended owner by exact canonical `usr_...` id with
  `docker compose exec audrey audrey-admin grant-admin <usr_id>`.
- [x] Confirm the Admin dialog exposes Accounts and Models.
- [x] With a second disposable identity, prove Pending → User → Tester and
  Disabled → Active transitions.
- [x] Confirm ordinary users cannot see tester/admin direct models.
- [x] Confirm testers can select an enabled tester model and complete a direct
  native turn.
- [x] Disable the selected direct model and confirm the picker falls back to
  the first allowed model.
- [x] Use **Reset default** and confirm the Customized marker clears.
- [x] Delete the newly created disposable account after the proof, or restore an
  existing disposable account to its recorded starting state. Reset the test
  model to its deployment default.

**Required Cloudflare applicant policy**

The current Access email allowlist rejects a new address before Audrey receives
identity evidence, so no Pending account can be created. In Cloudflare Zero
Trust, keep the Audrey application protected and add an **Allow** policy named
`Audrey verified applicants` with **Include → Login Methods → One-time
PIN**. Ensure One-time PIN is enabled for the application. This deliberately
lets any person who proves control of an email reach Audrey's account boundary;
Audrey still creates them as Pending and blocks its workspace, models, files,
conversations, runs, and administration routes until approval. Do not use an
Access Bypass policy.

Cloudflare references: [Access policies](https://developers.cloudflare.com/cloudflare-one/access-controls/policies/)
and [One-time PIN login](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/).

**Browser acceptance runbook**

1. Keep the administrator signed in in the normal browser. Open a separate
   browser profile, enter a genuinely new email at Cloudflare Access, complete
   its one-time PIN, and let the first Audrey request submit the account.
   Confirm the page says **Approval is pending**, shows the correct email, and
   offers **Check again**.
2. In the administrator browser, open **Admin Panel → Accounts**. Confirm the
   new identity is Pending, choose **Approve as user**, then use **Check again**
   in the disposable browser. The Audrey workspace must open.
3. In **Admin Panel → Models**, choose a currently unused direct model whose
   badge says **Default**. `qwen3.5:4b` is a suitable small candidate if it
   is still Default. Record its Enabled/Disabled and Public/Private starting
   state.
4. Change that model to **Public**, open **Edit…**, grant only **Tester**, and
   save. In **Accounts**, change the disposable identity's Role from **User** to
   **Tester**.
5. Refresh the disposable browser. Open the model picker, choose **Other
   models…**, select the test model, and send: `Reply with exactly
   BROWSER-DIRECT-PASS.` Confirm a direct answer completes.
6. In the administrator browser, disable the selected direct model. Refresh the
   disposable browser. Confirm the picker falls back to **Auto** and the
   conversation remains usable.
7. In **Accounts**, disable the disposable identity. Refresh its browser and
   confirm **This account is disabled**. Reactivate it, select **Check again**,
   and confirm the workspace returns. Move it through **User** and **Tester**
   once more to verify both role choices.
8. In **Models**, click **Reset default** for the test model. Confirm its badge
   changes to **Default** and its state matches step 3. Delete the newly created
   disposable account and wait for its row to leave the Accounts list.

Pass this slice when every visible state matches the steps, the direct turn
finishes, Auto appears after disabling the selected model, and the test model
returns to Default.

**Applicant-gate live result, 2026-10-01:** Passed. One-time PIN was enabled
for the Audrey Access application and the applicant policy was reduced to the
single `Login Methods -> One-time PIN` Include rule. A genuinely new address
received its code, authenticated successfully, reached Audrey as Pending, and
completed the administrator approval flow. Cloudflare no longer rejects new
applicants before Audrey can create their account.

**Full interactive result, 2026-10-01:** Passed. The user confirmed every
visible account, role, model-access, fallback, reset, and cleanup check in
this slice behaved as expected.

### 5.4 Prove restart persistence

**Implementation:** Laptop-complete. The read-only two-stage smoke and its
Tower runner support are ready for the live restart gate. The first live
capture exposed a probe-address bug: `/health` had been requested through
the UI origin, which returned its HTML shell. The corrected smoke calls
`http://audrey:8000/health` directly on `ollama-net` and keeps authenticated
state checks on the UI proxy. That failed attempt wrote no snapshot.

- [x] Record the intended owner groups, disposable account state, model policy,
  model order, and one conversation's selected model.
- [x] Restart Audrey.
- [x] Confirm those values and the selected conversation model survive.
- [x] Confirm `/health` returns `ok` and authenticated `/api/capabilities`
  returns `ready` after restart.

The two-stage smoke records both authenticated accounts, every stable model
policy in displayed order, and one real conversation selection. The verify
stage waits up to three minutes for the real health and capability routes,
then compares the post-restart values with the mode-600 snapshot. It does not
change application state.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_restart_persistence.py capture
docker compose restart audrey
bash tests/smoke/smoke-native-onbox.sh smoke_native_restart_persistence.py verify
```

The ordinary smoke account must already own at least one conversation. If
capture says none exists, sign in as that account, send one short native chat
message, and rerun only `capture`. A pass reports all four comparisons as
`true` and both readiness values as healthy.

**Live result, 2026-10-01:** Passed. Capture recorded the distinct owner and
ordinary accounts, seven workflow models, 34 direct models in displayed
order, and a real conversation selected to Deep. After restarting Audrey,
all four comparisons remained true: owner account, ordinary account, model
policy/order, and conversation model. Backend health returned `ok`, native
capabilities returned `ready`, and the smoke reported `status: passed`.

### 5.5 Prove projection rebuild

**Implementation:** Ready for the live gate. The smoke now refuses a rebuild
that reports zero reset projections.

- [x] Run the canonical chat projection smoke.
- [x] Require a successful native turn, a non-empty rebuild result, matching
  canonical/projected messages, deletion from both stores, repair `ready`, and
  no cleanup error.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_chat_projection.py
```

**Live result, 2026-10-01:** Passed. The disposable Fast turn produced two
matching canonical/projected messages and a 20-character answer. Rebuild
reset 11 projections and restored the disposable projection to the same two
messages. Canonical deletion returned 204, its follow-up read returned 404,
the projection disappeared, repair returned `ready`, and no cleanup error was
reported.

### 5.6 Complete the isolated restore proof

**Implementation:** Ready for the live gate. The supported
`verify-app-state-backup` command rejects the configured production path as
either source or destination. It inspects the saved backup read only, restores
it to a new mode-600 disposable file, then checks integrity, foreign keys,
current schema, and matching account, conversation, message, and run counts.
It removes a failed restore and leaves a successful one for inspection.

- [x] Add a supported verification command that restores the backup into a
  disposable path, opens it without touching production state, checks integrity
  and schema, and verifies representative account/conversation counts.
- [x] Run that command against the backup from 5.1.
- [x] Record the backup filename, size, schema, and verification result.
- [x] Keep production `/data/audrey_app.sqlite` untouched during the proof.

Run this on Tower after rebuilding the Audrey container with this slice:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build --force-recreate audrey
docker compose exec audrey audrey-admin verify-app-state-backup \
  --from /data/backups/audrey-app-before-2d5-20261001.sqlite \
  --restore-to /tmp/audrey-app-restore-check-20261001.sqlite
```

This reads the saved backup and writes only the new file under the container's
`/tmp`. It refuses to run if either argument resolves to
`/data/audrey_app.sqlite`. A pass reports `status: ok`, schema 17, the
integrity and foreign-key checks as `ok`, positive account and conversation
counts, and
`source_counts_match: true`. If the disposable filename already exists, use a
new filename; the command deliberately never overwrites a prior result.

**Live result, 2026-10-01:** Passed. The schema-17 backup and isolated
restore were both 675,840 bytes. Integrity and foreign-key checks returned
`ok`; the restore retained 5 accounts, 8 conversations, 22 messages, and 11
runs; all source counts matched. The verifier excluded production
`/data/audrey_app.sqlite` and wrote only
`/tmp/audrey-app-restore-check-20261001.sqlite`.

**Item 5 closed, 2026-10-01:** The backup, account/model smoke, interactive
provider and picker checks, restart persistence, projection rebuild, and
isolated restore proof all passed, with temporary policy and account changes
restored.

## 6. Build the next product phase

**State:** Complete. [Phase 9](phase-09-broader-audio.md),
[Phase 10](phase-10-ordinary-answer-provenance.md), and
[Phase 11](phase-11-native-file-explorer.md) are closed. The observed workflows
do not yet justify diarization or broader media analysis.

- [x] Write the Phase 9 plan and choose the first deployable slice.
- [x] Add OGG, Opus, and raw ADTS AAC admission using measured MIME/container
  pairs.
- [x] Reuse the existing durable audio queue, transcript, Summary, Files, and
  chat attachment boundaries.
- [x] Keep diarization and speaker labels in a separate measured slice.
- [x] Pass the Slice 9A manual Files and chat gate for OGG, Opus, and AAC.
- [x] Define current media-analysis scope from real user workflows and keep
  diarization, music, event, and scene analysis parked until an unmet workflow
  supplies a quality gate.
- [x] Design and implement Slice 10A ordinary-answer file provenance on the
  native message model without reintroducing tool cards into chat.
- [x] Pass the Slice 10A grounded-file and catalogue-only browser gate after
  refresh.
- [x] Replace paperclip file cards and Files listing cards with compact virtual
  folder explorers backed by the existing file-kind metadata.
- [x] Add measured single-request chat upload progress and explicit Preparing
  and Finishing phases.
- [x] Make Models and Tool calls mutually exclusive and outside-dismissible for
  live and saved answers.
- [x] Close Slice 11A after deployed visual acceptance, with Sources added to
  the Models/Tool calls mutual-exclusion and outside-dismissal behavior. The
  laptop has no Node runtime, so its frontend automated suites were not reported
  as passed.

## Next user-prioritized slice — Phase 12 sidebar navigation

**State:** Laptop-complete. Native browser acceptance is pending.

- [x] Replace the top Files and `+ New` pair with one full-width **New
  conversation** button.
- [x] Remove the Workspace label and username from the sidebar.
- [x] Move My Files into a distinct top-bar group separated from account
  actions.
- [x] Preserve the existing Files dialog and conversation history behavior.
- [ ] Pass desktop, narrow-screen, keyboard, empty-draft, and Files-launch
  browser checks.

Detailed scope: [phase-12-sidebar-navigation.md](phase-12-sidebar-navigation.md).

## Next user-prioritized slice - Phase 14 bot accounts and token lifetimes

**State:** Laptop-complete. Native acceptance is pending.

- [x] Add Bots as a protected schema-backed access group.
- [x] Allow pending approval and active role assignment as Bot.
- [x] Let model policies and publication profiles grant Bots explicit access.
- [x] Accept token lifetime `0`, return no expiry, and keep revocation and owner
  status enforcement.
- [x] Reload users, models, and roles without browser cache on every Admin Panel
  opening.
- [ ] Pass the refreshed applicant, Bot model visibility, permanent PAT, and
  revocation checks on the deployed application.

Detailed scope:
[phase-14-bot-accounts-and-token-lifetimes.md](phase-14-bot-accounts-and-token-lifetimes.md).

## Next build - Phase 15 composer controls and Projects

**State:** Complete. Slices 15A–15D and the direct project-upload follow-up
passed user or targeted live acceptance.

- [x] Build Slice 15A: narrow and center the composer; move Model, Add files,
  and Tools & skills into a labeled, symmetric rail below the message field.
- [x] Pass the Slice 15A desktop, narrow-screen, keyboard, picker-dismissal,
  upload, send/stop, and retry browser checks.
- [x] Build Slice 15B: schema 19 Projects storage, owner-scoped APIs, project
  file references, and nullable conversation membership.
- [x] Pass the Slice 15B deployment and restart-persistence smoke using one
  existing Ready file without deleting it.
- [x] Build Slice 15C: compact Projects navigation, project home, reusable My
  Files selection, new project conversations, and move/remove actions.
- [x] Pass the Slice 15C desktop, narrow-screen, keyboard, refresh, project
  management, conversation move/remove, and non-destructive deletion checks.
- [x] Build Slice 15D: server-resolved project instructions and bounded
  selected-file retrieval with persisted private-file provenance.
- [x] Pass the final two-document context, refresh, isolation, mutation
  snapshot, project deletion, and restart-persistence gates.
- [x] Build direct project upload with persistent processing membership and
  polish the My Files Add files and Choose files actions.
- [x] Pass direct project upload, progress, processing-to-Ready refresh, My
  Files styling, mobile layout, and hard-refresh behavior in the deployed UI.

Detailed scope:
[phase-15-composer-and-projects.md](phase-15-composer-and-projects.md).

## Retired Phase 16 - Audrey document tools

**State:** Closed without a replacement Audrey feature.

- [x] Confirm the existing Hermes path: Bot Tools MCP APIs, Nextcloud,
  Collabora, and `cloud.builtryte.xyz`.
- [x] Remove Audrey's document creation UI, document-job API, template worker,
  bundled template, tests, smoke, and future DOCX/PDF/spreadsheet roadmap.
- [x] Retain schema 20 only as inert deployed migration history with backup and
  privacy cleanup compatibility.
- [x] Record the standing boundary: Hermes workspace capabilities are API-driven
  and do not affect Audrey's user experience.

Detailed decision record:
[phase-16-native-document-tools.md](phase-16-native-document-tools.md).

## 7. Expand Responses API compatibility

- [x] Start the expansion as separate per-capability protocol slices.
- [x] Add typed `input_text` plus bounded inline `input_image` content parts
  for completed and streaming requests.
- [x] Build Slice 13C: add owner-scoped `file_id` image and document inputs
  through Audrey storage, with ready-state, type, count, and size limits.
- [x] Pass the targeted Slice 13C file-reference protocol smoke (2026-10-05).
- [x] Build Slice 13D: add remote HTTP(S) image/document inputs through a
  bounded SSRF-safe fetch contract, with temporary parsing and joint budgets.
- [x] Pass Slice 13D's normal URL answer and private-URL guards (2026-10-05).
- [x] Correct Fast stream token-limit/empty completion and remove progress from
  Responses answer text; preserve usage, run observations, and caller limits.
- [x] Pass Slice 13D's streamed-only retry with the explicit 8,192-token ceiling (2026-10-05).
- [x] Add bounded json_schema structured text output for completed and
  streamed Responses, with final validation and an explicit legacy json_object
  rejection.
- [x] Build Slice 13E: client-provided functions on permitted tool-capable
  passthrough models, typed completed/streamed calls, validated arguments, and
  stateless call/result replay; caller execution only, API only.
- [x] Pass Slice 13E's Qwen completed, streamed, and result-replay smoke (2026-10-05).
- [x] Review Claudette's Kimi K3 / GLM 5.3 assessment and actual Hermes catalog.
  Keep its current Chat Completions connection after the reported 20-tool catalog
  fails the optional Responses limit. Completed calls/results and text SSE are
  bot-reported evidence; streamed function arguments and production interruption
  handling were not shown. GPT Sol 6.1 is outside this task.
- [x] Build Slice 13F: opt-in stored text/function Responses, owner-scoped
  retrieval/deletion, bounded retention, and `previous_response_id` continuation.
- [x] Pass Slice 13F's targeted restart verify (received 2026-10-06): exact
  retained root/child objects, cascading deletion, and deleted-chain guard;
  zero new generations or uploads.
- [ ] Add the distinct Conversations API only when a caller needs its owner and
  lifecycle contract; deferred after the Hermes assessment.
- [ ] Add background execution, in-progress retrieval, and cancellation only
  for a demonstrated caller workflow; deferred after the Hermes assessment.
- [ ] Preserve the current explicit HTTP 400 response for every unsupported
  feature until its complete contract ships.

## 8. Evaluate automatic skill selection

**State:** Slice 3D.1 evaluation foundation is laptop-complete: 3,725 backend
tests passed. Rules baseline is measured (70% precision, three false activations,
seven misses). The pilot passed; two repeated hybrid studies completed all
calls on 2026-10-06 but measured 63.16% precision, 15/42 ordinary false
activations, and three rejected choices each. Automatic selection stays off.
Slice 3D.2 policy refinement is laptop-complete: 3,845 backend tests passed.
Rules on the same development set: nine correct activations, zero false
activations, five undecided positives; no model calls. First new-control result:
27/30 hybrid labels, 12/13 precision, two misses, one ordinary false activation;
zero model errors. Slice 3D.3 repairs those three exposed cases and adds compact
foreground results. Its targeted Tower check passed 3/3 on October 6 with zero
model calls. Slice 3D.4 is laptop-complete (3,973 backend tests): frozen
prospective study plans and 24 separately authored, agent-reviewed controls.
Tower preparation and first measurement completed: 69/72 matches, 36/39
activation precision, zero misses/errors, one ordinary request falsely selected
in all three repeats, and 24 planned/actual model calls. Proposed quality gates
remain unmet and human review remains pending; automatic selection stays off.
Terminal receipts and the operator-copied full report are verified on the
laptop. Slice 3D.5 repairs the exposed Japanese file-exclusion case;
4,031 backend tests passed. Its targeted four-case Tower check is pending.
[Recorded study](phase-03-skill-selection-evaluation.md) / [new-control result](phase-03-skill-selection-policy.md) / [terminal results](phase-03-skill-selection-terminal-results.md).

- [x] Keep `skills.auto_select: false` during the study.
- [x] Build 42 labeled positive, ambiguous, and ordinary-chat control cases.
- [x] Add an offline rules baseline and optional retained-router and hybrid
  measurement arms, with explicit errors, abstention, latency, and usage.
- [x] Pass the one-case direct-model pilot: rules abstained and one retained
  router call correctly selected document analysis; zero errors, auto off.
- [x] Collect the 42-case, three-repeat hybrid study (30 model calls per run);
  both received reports have identical selection findings.
- [x] Measure missed activation and false activation separately; current hybrid
  improves recall but worsens activation precision and ordinary false activation.
- [x] Compare deterministic rules with the retained-router hybrid on identical cases.
- [x] Refine evaluation policy: terminal abstention, excluded-file targets,
  scoped negation, and quoted content (Slice 3D.2).
- [x] Reserve 30 separately authored proposed controls without selector tuning;
  labels still require independent human review.
- [x] Preserve probe launch logfile identity through detachment; behavior tested.
- [x] Collect the first revision-2 measurement on those 30 controls once.
- [x] Review its three mismatches independently; proposed labels remain defensible.
- [x] Repair quotation, file-location comma scope, and bounded Spanish management
  regressions, preserving the exposed baseline (Slice 3D.3).
- [x] Restore foreground shell logs and compact selection results; mismatches
  now exit 1 even when model execution completed.
- [x] Pass the targeted three-case revision-3 Tower check in the launching shell
  (3/3, zero errors or model calls, accepted 2026-10-06).
- [x] Author 24 fresh proposed controls separately and obtain independent
  agent label review without evaluating or tuning on them (Slice 3D.4).
- [x] Add frozen study plans and drift/budget checks before model HTTP;
  separate proposed criteria from human attestations and production approval.
- [x] Prepare the Tower plan and collect its first prospective measurement;
  preserve the pasted terminal report on the laptop, including its findings.
- [x] Verify the operator-copied full revision-3 report on the laptop,
  including all repeated samples and qualification findings.
- [x] Repair the exposed supplied-text Japanese false activation with
  scoped file-access exclusions and quoted-data handling (Slice 3D.5).
- [ ] Pass the four-case revision-4 Tower regression check once; two guarded
  abstentions and two retained-router calls. Keep production selection off.
- [ ] Obtain independent human label review before broader selection-quality
  conclusions; earlier measured fixtures are now regression data.
- [ ] Consider a separate router/System One arm only if revised policy evidence
  warrants that comparison; no new candidate probe is queued now.
- [x] Propose separate precision, false-activation, and missed-activation
  thresholds; retain `auto_select: false` throughout measurement.
- [ ] Agree the final gates, validate a holdout set, and prove answer-quality
  and cost benefit before enabling anything.
- [ ] Ship only if the selector beats explicit selection without narrowing tools
  or activating a skill on unrelated chat.

## 9. Repair the Tailscale backend route

- [ ] Reproduce reachability from the laptop to `100.113.157.98:8000` while the
  LAN/WARP route at `192.168.1.11:8000` remains healthy.
- [ ] Check Tower's Tailscale address, route advertisement, host firewall, and
  Docker-published port independently.
- [ ] Identify the failing hop before changing Audrey or Compose.
- [ ] Restore `/health`, `/api/ready`, and authenticated `/v1` reachability.
- [ ] Update the live-smoke reference to prefer Tailscale only after it passes.

## 10. Finish model and production measurements

- [x] Run Clef's broad System One decision probe and review its text, JSON,
  image, accuracy, calibration, latency, token, and residency results.
- [x] Compare Clef and Clef Flash with `qwen3.5:4b`; retain the incumbent after
  full Clef failed cold/footprint gates and Flash failed cold-start reliability.
- [ ] Verify the installed `deepseek-v4.1-flash:cloud` capabilities with
  `ollama show` and one tool-call probe.
- [ ] Run the real A-B-A for `factcheck_worker.compress_keep_last` with a fresh
  config load between arms.
- [ ] Run the answer-quality A-B-A for
  `agentic.react.max_tool_result_chars: 6000`.
- [ ] Complete a normal video/audio ingest proof for the summary no-thinking
  policy.
- [ ] Measure fact-check fallback and ledger drop/unlinked counts from
  accumulated production traffic.
- [ ] Measure escalation cost and synthesis draft sizes once enough traffic has
  accumulated.
- [ ] Convert any decision-worthy result into `evals/MODEL-FACTS.md` in the
  same session.

## Deliberately deferred

- Historical chat import remains optional and requires a new explicit request.
- Automatic skill selection remains disabled until item 8 passes.
- The shared evidence pool, specialist prototype, math-classifier audit, and
  KB corpus split stay parked until a real failure or priority change reopens
  them.
