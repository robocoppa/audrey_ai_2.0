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
- [ ] Rebuild Audrey and rerun the corrected account/model smoke through the
  standalone UI proxy.
- [ ] Require `status: passed`, a successful direct-model run, zero tool events,
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

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash tests/smoke/smoke-native-onbox.sh smoke_native_access_models.py
```

### 5.3 Complete the interactive browser checks

- [ ] Sign in once with a genuinely new allowed Access identity and confirm it
  lands at the pending-account boundary.
- [ ] Bootstrap the intended owner by exact canonical `usr_...` id with
  `docker compose exec audrey audrey-admin grant-admin <usr_id>`.
- [ ] Confirm the Admin dialog exposes Accounts and Models.
- [ ] With a second disposable identity, prove Pending → User → Tester and
  Disabled → Active transitions.
- [ ] Confirm ordinary users cannot see tester/admin direct models.
- [ ] Confirm testers can select an enabled tester model and complete a direct
  native turn.
- [ ] Disable the selected direct model and confirm the picker falls back to
  the first allowed model.
- [ ] Use **Reset default** and confirm the Customized marker clears.
- [ ] Restore the disposable account and model policy to their starting state.

### 5.4 Prove restart persistence

- [ ] Record the intended owner groups, disposable account state, model policy,
  model order, and one conversation's selected model.
- [ ] Restart Audrey.
- [ ] Confirm those values and the selected conversation model survive.
- [ ] Confirm `/api/ready` returns ready after restart.

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
