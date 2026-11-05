# Legacy identity cleanup

**Status:** Source implementation and local verification completed in the working
tree, November 5, 2026. Full suites retain reproduced baseline failures. Tower
deployment and owner-operated live acceptance remain pending. Lesson refresh is
a separate reviewed unit.

Audrey authenticates browser users with Cloudflare Access, scripts and bots with
Audrey personal access tokens (PATs), and trusted internal services with the
shared service token. Routes now use Audrey's `Principal` directly. This cleanup
removes the obsolete email-shaped adapter and moves browser/operator operations
to `/api`, while retaining the bot API and internal service protocols under `/v1`.

The owner authorized the corrected plan and implementation, and confirmed that
Hermes and other external bots do not use the user-facing `/v1/files` endpoints.
Their removal therefore has no outstanding external-caller prerequisite. Existing
Hermes adapters/settings and account storage namespaces are preserved. No account,
upload-directory or Qdrant namespace migration is included.

## Corrected identity and authorization contract

| Surface | Dependency and behavior |
|---|---|
| `GET /api/me` | `require_account_principal` returns a `Principal`; Cloudflare-authenticated pending/disabled accounts can still see their account status. PATs require an active account and explicit `account:read`. |
| Compatibility bot API, `/v1/tools`, `/v1/references/libraries`, and moved `/api/me` data operations | `require_user` returns an active `Principal` and explicitly requires `compat:full` for PATs, regardless of request path. Provider-authenticated active accounts retain access. |
| Token/credential management and account data-purge creation | `require_provider_principal` requires an active provider-authenticated account; a PAT cannot authorize these operations. |
| `/api/admin/*`, including KB ingest and tool rediscovery | `require_admin_principal` requires active provider authentication plus administrator rights. An administrator's PAT cannot invoke administration routes. |
| `/v1/kb/query` and `/v1/kb/query/image` | `require_service` accepts only `X-Audrey-Service-Token`; PATs and browser assertions do not replace the service credential. |
| Remaining `/v1/files` worker/artifact operations and reference service operations | Retain the service-token dependency and existing worker protocol. |

`require_user` remains as the name of the compatibility-scope dependency, but
returns no legacy wrapper. `require_account_user` likewise returns `Principal`
and preserves the special account-status behavior needed by `/api/me`. Ordinary
request authentication still rejects inactive accounts and unsupported bearers.

The passthrough model role gate must keep its former effective-role rule: a PAT
has effective role `user`, even when its owning `Principal` is an administrator.
Checking `principal.role` alone would silently broaden token privileges. Model
grants, allowlists and fair scheduling remain separate controls.

`Principal.user_id` and `Principal.storage_namespace` are distinct identifiers:

- Canonical Project and saved Responses ownership uses `user_id`.
- Files, retrieval, per-user collections, memories, chat-archive data and existing
  scheduling buckets use the opaque `storage_namespace` where they did before.
  Older namespaces can contain an email address; changing their spelling would
  require a separate data migration.
- A request's `payload.user` is ignored. It does not establish ownership and no
  longer triggers a comparison/log against the authenticated identity.

## Implementation units

### 1. Remove unused identity branches and repair the probe

Implemented in the working tree:

- Remove unauthenticated `GET /v1/kb/stats`; no caller or retained operator need
  was identified.
- Remove the obsolete `payload.user` comparison/log while continuing to ignore
  that client-provided field.
- Remove the impossible `role == "bot"` branches from Agent, Watch, Tax,
  Private-task and skill admission. Membership in the `bots` group continues
  to enforce bot exclusions.
- Keep `scripts/probes/kb_score_probe.py`. Read `KB_SERVICE_TOKEN`, send it as
  `X-Audrey-Service-Token`, and validate the endpoint's `results` list.
  Missing credentials return setup error 2;
  transport, HTTP and malformed-response failures return 1, rather than being
  reported as a valid query with no hits. A successful empty list is zero hits.
- Use `alice@example.com` in `kb/user_store.py` examples and remove the unused,
  gitignored local `src/audrey/static/app/` build.

### 2. Share file operations without an identity adapter

Implemented in `src/audrey/file_service.py`: owner-bound upload, upload-session,
URL-import, list and delete functions take `Principal`. Native file routes and
Project context call these functions directly; `_compat_user` is removed.

Low-level storage/lifecycle helpers and service routes remain in `routes/files.py`.
The extraction preserves their use by Private processing, generated file outputs,
Responses file inputs, media/fetch workers, quota reservations and deletion
workers. It does not replace or remove Private upload support, ownership checks,
pins, cleanup or worker callbacks. Responses reads continue through the native
owned-file interface; their persisted response owner remains canonical `user_id`.

### 3. Move browser and operator routes

Implemented route moves, with no legacy aliases:

| Previous route | Current route |
|---|---|
| `/v1/me/*` data/memory/chat-history/repair operations | `/api/me/*` |
| `/v1/admin/*` | `/api/admin/*` |
| `/v1/tools/rediscover` | `/api/admin/tools/rediscover` |
| `/v1/kb/ingest` | `/api/admin/kb/ingest` |

The browser client, frontend unit/E2E mocks, live smoke callers and smoke-harness
expectations use the current paths. The moved `/api/me` operations keep explicit
`compat:full` enforcement for tokens; purge creation keeps provider-only access.
The role-only `require_admin` helper is removed.

Browser and backend changes deploy together. An open browser tab needs a full
reload after deployment so its JavaScript uses the new paths.

### 4. Finish `/v1` and Principal migration

Implemented in the working tree:

- Remove the seven user-facing `/v1/files` route registrations: single upload,
  upload-session creation, part upload, completion, URL import, listing and delete.
  Their native `/api/files` equivalents remain. Keep service-only fetch/job,
  ingestion, OCR, list and artifact routes used by the sidecar/workers.
- Update `smoke_native_auth_cutover.py` to read `/api/files` and validate its
  `items` list rather than the retired endpoint's `files` field.
- Make both KB query routes service-only; delete `KBCaller` and
  `resolve_kb_caller`.
- Replace `AuthedUser` in OpenAI-compatible internals with `Principal`; use the
  correct ownership identifier for each subsystem and preserve PAT passthrough
  effective-role restrictions.
- Keep `/v1/tools` and `/v1/references/libraries` as bot metadata endpoints with
  explicit `compat:full` enforcement through `require_user`. Removing path-based
  scope enforcement must not make these endpoints available to `account:read`
  tokens.

### 5. Update references and isolate lesson refresh

The current Hermes integration reference describes Audrey PAT validation,
Principal namespaces, provider-only administration and the bot metadata boundary.
Historical campaign documents remain historical references.

The affected published lesson inventory is **1, 2, 3, 4, 5, 6b, 10, 13, 14 and
16**. These include dependency examples, identity wrappers, `/v1` scope statements,
route paths and source citations that move during extraction. Treat that inventory
as a separate refresh unit under the lesson workflow in [AGENTS.md](../../AGENTS.md):
audit current code, review findings/outline with the owner, then revise prose and
run the lesson convention check. No lesson prose is silently refreshed as part of
this implementation; the implementation authorization itself needs no further
permission.

## Verification and completion criteria

Local verification completed:

- Full Python coverage: **8,579 passed, seven failed** across 8,586 collected
  tests. All seven failures reproduce against restored unchanged source; no
  migration-induced failure remained. The first 876 passes were retained while
  the remaining tests ran in three disjoint batches.
- Full web unit suite: **655 passed, three failed**. All three failures reproduce
  with this task's changed web files restored to the unchanged baseline.
- Affected App tests: **46 passed**. The two affected production browser cases
  for saved memories and export/purge passed.
- Changed-file Ruff and ESLint, TypeScript, production build, source/auth route
  inventory, OpenAPI generation, plan links and diff whitespace checks passed.
  Ruff retains the existing accepted `kb/reconcile.py` ASYNC240 exception.
- Web checks used the available bundled Node 24.5.0, below the declared 24.15+
  requirement; no dependencies or runtime versions were installed or changed.
  The owner-operated image build below verifies the supported deployment setup.

Detailed local results and baseline evidence paths are in the gitignored
[validation receipt](../../testing-out/legacy-identity-cleanup/validation.json).
These isolated checks do not establish deployment, real model quality or live
browser acceptance. No remote deployment, live smoke or model inference was run.

The completed source/reference checks establish:

- No `AuthedUser`, `KBCaller`, role-only `require_admin`, `_compat_user`,
  `resolve_kb_caller` or `role == "bot"` remains in `src/`.
- `require_user` and `require_account_user` return `Principal`; explicit token
  scopes and pending-account handling survive the migration.
- No user-facing `/v1/files` registration, `/v1/me/*`, `/v1/admin/*`,
  `/v1/kb/ingest`, `/v1/tools/rediscover` or `/v1/kb/stats` remains.
- `/v1` contains only the bot API/metadata and service-token protocols, with
  `/v1/models` remaining the intentionally public catalogue of virtual and
  configured passthrough models.
- Application data operations retain their authentication dependencies.
  `/health` and `/metrics` remain public by design. FastAPI's documentation/schema
  routes (`/docs`, `/docs/oauth2-redirect`, `/redoc`, `/openapi.json`) are also
  public framework routes; this cleanup does not promise to remove them.
- Local tests/lint/build and the owner-operated checks below are reported
  separately. Live acceptance remains pending until the owner reports results.

## Owner-operated deployment and acceptance

Follow [live testing](../reference/live-smoke-testing.md) and the
[testing protocol](../reference/testing-protocol.md). These are handoff steps;
no remote deployment or live test is claimed here. No schema or configuration
migration is required. The backend and browser images must both rebuild.

On **Tower**, after the normal source sync, enter the checkout:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
```

Rebuild and start both changed services:

```bash
docker compose up -d --build audrey audrey-ui
```

Expect both services healthy/running before continuing. Refresh Tower's private
`.env.smoke.local` with a current `AUDREY_USER_JWT` Cloudflare application
assertion; keep its existing supported `KEY=value` format. The following short
foreground check needs no upload and no model generation:

```bash
tests/smoke/smoke-native-onbox.sh smoke_native_auth_cutover.py
```

Expect JSON `status: "passed"` and exit 0: an unsupported bearer is rejected,
a temporary PAT reaches the same account and the native `items` file list, and
revocation immediately rejects the token. This exercises the live UI proxy and
backend. The smoke revokes its temporary token, including on a failed check.
Results print in the launching shell; no report transfer is needed.

For a focused bot API check, use the **laptop** checkout. Private
`.env.test.local` exists with nonempty `AUDREY_EVAL_BASE_URL`,
`AUDREY_EVAL_API_KEY`, `AUDREY_USER_JWT` and `AUDREY_ADMIN_JWT` entries at handoff
preparation; their values and freshness are not printed or verified here. Its
PAT needs `compat:full` and access to `audrey_fast`. Enter the checkout:

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
```

Run the short Responses protocol smoke in a subshell, keeping credentials local:

```bash
(
  set -a
  source .env.test.local
  set +a
  .venv/bin/python tests/smoke/smoke_responses_api.py
)
```

The runner accepts `AUDREY_EVAL_BASE_URL` with or without its `/v1` suffix and
sends the PAT to `/v1/responses`. Expect JSON `status: "passed"` and exit 0,
a completed stream containing `RESPONSES_STREAM_OK`, and explicit rejection of
unsupported background mode. This performs actual model inference; it does not
require an upload. Results return to the laptop shell. It supplements the real
browser acceptance below and does not establish every retained bot feature.

In the real browser at `https://ai.builtryte.xyz`, fully reload after deployment:

1. Sign in normally. Confirm the account/profile and existing Projects, files
   and conversations remain visible. If a disposable pending/disabled provider
   account is available, its sign-in must display the awaiting-approval/disabled
   state rather than opening active-account operations; do not disable a working
   account just to perform this check.
2. Open **Settings → Saved memories**. Add key `identity-cleanup-check` with text
   `The identity cleanup test marker is CEDAR-42.`, save, edit it to `CEDAR-43`,
   reload Settings and verify the edit. Delete that disposable memory with the
   normal confirmation and confirm it disappears. Open **Your Audrey data →
   Download chat history** and open the downloaded JSON; expect the current
   account's retained archive, or an empty list if none exists.
3. Upload a real small English TXT/MD file through **My Files**, containing
   `The file acceptance marker is MAPLE-27.`. Wait for Ready, preview/read it,
   attach it to an ordinary Project, and verify the same file remains listed
   after reload. Select it in a chat using an allowed local model and ask: `What is the
   acceptance marker in the selected file? Reply with the marker only.` Expect
   `MAPLE-27`. Download/open the original and verify its bytes, then delete the
   disposable file and confirm it disappears from My Files and Project files.
4. If Private upload is enabled, repeat upload/preview/download/delete in a
   disposable Private Project with a real TXT file containing `PINE-18`. Confirm
   processing remains Private/local and that no ordinary Project can attach it.
   This step needs a real upload; do not replace it with mocked evidence.
5. As the provider-authenticated administrator, open administration and confirm
   account/model lists load. Leave existing account settings intact. In the
   existing Hermes bot, send `Reply with exactly: BOT_IDENTITY_OK` and confirm
   the response; no adapter/settings change is needed.

Visible pass criteria are intact account ownership and retained data, working
moved Settings/admin operations, successful local file reading and real downloads,
and the bot's normal reply. Account purge, archive deletion/rebuild and repair
are covered by isolated regressions; do not erase real user data for acceptance.
Record skipped optional account/Private checks explicitly and report failed steps
before proceeding to their dependents. Live acceptance stays pending until these
results are supplied.

## Optional, later

- Narrow `resolve_external_identity` to the Access sign-in it serves after tests
  have a separate account factory.
- Rename `KB_SERVICE_TOKEN` to reflect its wider use, changing environment,
  Compose and sidecar callers together.
- Make `admins` membership the only source of administrator rights and drop the
  mirrored role column.
- Rename pipeline/tool parameters called `user_id` that carry a storage namespace.
