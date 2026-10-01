# Campaign 3 remaining work — items 5–10

**Status:** Items 1–4 were marked live-closed on 2026-10-01. Item 5 is in
progress. Items 6–10 remain queued in this order unless the user reprioritizes
them.

## Closed prerequisites

- [x] Deploy the retired-model and media-fetcher progress slice.
- [x] Pass the native Cloud turn with `glm-5.3:cloud` and `kimi-k2.6:cloud`.
- [x] Pass WAV, M4A, and FLAC native acceptance; Campaign 3 Phase 8 is complete.
- [x] Remove retired Ollama entries and return model inventory to clean.
- [x] Release and remove the stopped `audrey-ai-retired` cutover container.

## 5. Finish Phase 2D.5 administration and recovery proof

**State:** In progress. The repeatable account/model and projection smokes
already exist. The first new slice exposes Audrey's SQLite online-backup
primitive as `audrey-admin backup-app-state`.

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
Tower runner support are ready for the live restart gate.

- [ ] Record the intended owner groups, disposable account state, model policy,
  model order, and one conversation's selected model.
- [ ] Restart Audrey.
- [ ] Confirm those values and the selected conversation model survive.
- [ ] Confirm `/health` returns `ok` and authenticated `/api/capabilities`
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

### 5.5 Prove projection rebuild

- [ ] Run the canonical chat projection smoke.
- [ ] Require a successful native turn, a non-empty rebuild result, matching
  canonical/projected messages, deletion from both stores, repair `ready`, and
  no cleanup error.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_chat_projection.py
```

### 5.6 Complete the isolated restore proof

- [ ] Add a supported verification command that restores the backup into a
  disposable path, opens it without touching production state, checks integrity
  and schema, and verifies representative account/conversation counts.
- [ ] Run that command against the backup from 5.1.
- [ ] Record the backup filename, size, schema, and verification result.
- [ ] Keep production `/data/audrey_app.sqlite` untouched during the proof.

**Item 5 closes when:** the backup, account/model smoke, interactive provider
and picker checks, restart persistence, projection rebuild, and isolated restore
proof all pass, with temporary policy and account changes restored.

## 6. Build the next product phase

- [ ] Write the Phase 9 plan and choose the first deployable slice.
- [ ] Add OGG, Opus, and AAC admission using measured MIME/container pairs.
- [ ] Reuse the existing durable audio queue, transcript, Summary, Files, and
  chat attachment boundaries.
- [ ] Keep diarization and speaker labels in a separate measured slice.
- [ ] Define media-analysis scope from real user workflows before adding music
  or scene-specific analysis.
- [ ] After media work, design ordinary-answer provenance on the native message
  model without reintroducing tool cards into chat.

## 7. Expand Responses API compatibility

- [ ] Split each capability into its own storage/protocol slice.
- [ ] Add multimodal content parts.
- [ ] Add structured text output configuration.
- [ ] Add client-provided tools with explicit policy boundaries.
- [ ] Add stored responses and `previous_response_id`/conversation chaining.
- [ ] Add background execution, retrieval, cancellation, and retention.
- [ ] Preserve the current explicit HTTP 400 response for every unsupported
  feature until its complete contract ships.

## 8. Evaluate automatic skill selection

- [ ] Keep `skills.auto_select: false` during the study.
- [ ] Build labeled positive, ambiguous, and ordinary-chat control cases.
- [ ] Measure missed activation and false activation separately.
- [ ] Compare deterministic rules with the retained router and System One
  candidates.
- [ ] Define a precision threshold and rollback switch before enabling anything.
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
