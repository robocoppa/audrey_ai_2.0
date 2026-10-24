# Campaign 4 Phase 2 — native Audrey application

**Status:** Complete. The native application, identity, administration, recovery,
and standalone browser cutover are accepted. No Phase 2 acceptance gate remains.
User input is English-only for the current campaign.

## Goal and decisions

Audrey owns the application: identity, authorization, conversations, messages,
runs, attachments, preferences, and personal tokens. The browser renders and
initiates actions; it never owns authoritative history or the internal tool loop.

- React/TypeScript/Vite builds a static client in `web/`; production needs no
  Node server. AG-UI and OpenAI SSE adapt the same typed Audrey event spine.
- SQLite/WAL stores authoritative application records behind repositories and
  versioned migrations. Search/archive/Qdrant data are repairable projections.
- Cloudflare Access authenticates the public browser. Audrey verifies the
  signed application assertion and owns approval, roles, and model policy.
- Audrey PATs serve direct API clients and evals. Unknown bearer credentials
  are rejected locally; there is no external-client sign-in adapter.

## Identity and authorization contract

- Provider subject maps to a stable Audrey user id and private namespace.
  Email changes cannot rename private storage or implicitly merge accounts.
- Preserve existing exact namespaces; new identities use opaque namespaces.
- New Access identities start Pending. Only active accounts enter the workspace.
- Cross-owner and missing resources return the same `404`.
- Admin, credential management, preference mutation, and account-wide deletion
  require provider authentication. A PAT cannot become an admin credential.
- PATs have hashed secrets, explicit `account:read`/`compat:full` scopes,
  dated or intentionally permanent expiry (Phase 14), last-use tracking,
  and immediate revocation.
  Display each new secret once; never store it in browser persistent storage.
- Admin policy controls published models and direct-model audiences. Run-time
  authorization rechecks access; hiding a picker option is insufficient.
- Prevent self-disable, self-demotion, and removal of the last active admin.
  First-admin recovery uses an exact canonical id or unique exact email.

## Conversation, run, and event contract

- Audrey issues stable conversation, message, run, and tool-call ids.
  The server loads history; clients submit new text/actions and owned file ids,
  never an authoritative system prompt or prior assistant/tool transcript.
- Persist the user message and assistant/run records before streaming.
  Persist one immutable success, failure, or cancellation with partial output.
- One conversation has at most one active run. Run ownership survives browser
  disconnect; explicit Stop cancels. Startup settles interrupted run records.
- Sequenced typed events distinguish stages, answer deltas, tools, sources,
  usage, and terminal outcomes. Adapters do not parse each other's display text.
- Native replay is bounded and process-local. Expired replay falls back to
  durable run/message reads; a restart does not claim its event buffer survived.
- Native and `/v1` requests use the same model-choice rules. A task-like
  heading in user text does not override the requested model or skip archival.
- Empty abandoned drafts do not enter history; a saved Tax review case is
  durable work and retains its conversation even before the first message. Search, pagination, rename,
  archive/restore, deletion, canonical refresh, and retry remain owner-bound.

## Files, preferences, and data controls

- Upload once; validate quotas, readiness, and ownership before committing
  durable attachment snapshots with the user message. Source deletion removes
  content access while preserving the message's safe attachment metadata.
- Filenames and retrieved contents are data. Audrey constructs the model-facing
  attachment context and enforces tools at offer and dispatch.
- Validate IANA timezones and bounded preferences. Server-built preference
  context reaches models but does not change request-complexity routing.
- Settings export covers the chat-search archive, not a complete account backup.
- Account deletion retains identity/profile, resets preferences, and deletes
  tokens, canonical history, uploads, memories, and derived data through durable
  owner-bound cleanup receipts.

## Deployment and recovery

`audrey-ui` is the sole browser surface; `audrey` is API-only. Cloudflared
targets Tower loopback port 8090. The UI proxies same-origin `/api` and `/v1`
to its explicit `AUDREY_UI_UPSTREAM` using request-time Docker DNS.
Backend port 8000 must not become the public browser origin.

Take SQLite-aware online backups before migrations and verify isolated restore.
Keep projections rebuildable. Browser rollback redeploys a known-good UI image;
backend recovery uses the current backend service and preserves application data.
Runner and credential guidance lives in [live smoke testing](../reference/live-smoke-testing.md).

## Retained compatibility boundaries

The retired Open WebUI integration has been removed: no sign-in probe, token
cache, environment settings, roster-import command or special prompt routing.
Existing account storage namespaces and external-identity rows are deliberately
not renamed or deleted; changing them would orphan stored uploads and search
collections. Generic OpenAI-compatible fields such as `chat_id`, message metadata
and image parts stay because other clients use them. `history_import.py` stays
because it imports Audrey's own portable chat-export format, independent of any
external UI. Tests retain historical provider identifiers and error envelopes
only to cover account binding and compatible-client error handling. No test or
runtime path needs the retired UI to run.

## Optional historical import and deferred choices

Historical import is dormant and runs only on explicit request. Preview
`audrey-admin import-chat-export` before applying; verify exact owner id/email
and export markers. Apply requires a new private, integrity-checked online
backup. Unbound exports require independent ownership verification. Imports are
idempotent, start Archived, and respect deleted-import tombstones. Current limits
are 100 MiB and 100,000 messages. The archive export is not a full account backup.

Separate UI repository extraction, offline-LAN OIDC, and PostgreSQL require an
actual operational need; they do not reopen this completed phase.

## Source map

Identity and tokens: `src/audrey/identity/`. Canonical state and migrations:
`src/audrey/app_state/`. Events: `src/audrey/pipeline/run_events.py`. Native resources:
`src/audrey/routes/app/`. Browser/proxy: `web/`. Deployment: `compose.yaml`.
