# Phase 21 — legacy identity cleanup

**Status:** Closed by owner November 5, 2026. The owner reported the Responses
protocol smoke passed after correcting the laptop test URL to the working LAN
address. The earlier connection timeout is resolved. Lesson refresh is separate
scope.

## Delivered identity contract

Routes use Audrey's `Principal` directly. Browser accounts authenticate through
Cloudflare Access, scripts and bots through Audrey personal access tokens
(PATs), and trusted services through the shared service token. The obsolete
email-shaped identity wrapper is removed.

| Surface | Retained authorization |
|---|---|
| `GET /api/me` | `require_account_principal` permits provider-authenticated pending/disabled accounts to see their status. PATs require an active account and `account:read`. |
| Compatibility bot API, `/v1/tools`, `/v1/references/libraries`, and `/api/me` data operations | `require_user` returns an active `Principal` and explicitly requires `compat:full` for PATs regardless of path. Active provider-authenticated accounts retain access. |
| Token/credential management and account data-purge creation | `require_provider_principal` requires an active provider-authenticated account. |
| `/api/admin/*`, including KB ingest and tool rediscovery | `require_admin_principal` requires active provider authentication and administrator rights. An administrator's PAT cannot authorize these operations. |
| `/v1/kb/query` and `/v1/kb/query/image` | `require_service` requires `X-Audrey-Service-Token`. |
| Remaining `/v1/files` worker/artifact operations and reference service operations | Existing service-token protocol is retained. |

`require_user` and `require_account_user` return `Principal`, without a legacy
wrapper. Ordinary authentication continues to reject inactive accounts and
unsupported bearers. The passthrough model role gate treats every PAT as
effective role `user`, including tokens owned by administrators. Model grants,
allowlists and fair scheduling retain their separate controls.

Canonical Project and saved Responses ownership uses `Principal.user_id`.
Files, retrieval, per-user collections, memories, chat archives and existing
scheduling buckets retain `Principal.storage_namespace`. Existing namespaces
may contain email addresses; no account, upload-directory or Qdrant namespace
migration was performed. Client-provided `payload.user` remains ignored and no
longer triggers an identity comparison or log.

## File lifecycle and route boundaries

The shared [file service](../../src/audrey/file_service.py) accepts `Principal`
for owned uploads, upload sessions, URL imports, listing and deletion. Native
routes and Project context call it directly. Low-level lifecycle helpers and
service routes remain in `routes/files.py`, preserving Private uploads,
generated outputs, Responses file inputs, worker callbacks, quota reservations,
pins and deletion fences.

Browser and operator routes moved together with their browser clients,
unit/E2E mocks and smoke callers. Retired paths have no aliases:

| Previous route | Current route |
|---|---|
| `/v1/me/*` data/memory/chat-history/repair operations | `/api/me/*` |
| `/v1/admin/*` | `/api/admin/*` |
| `/v1/tools/rediscover` | `/api/admin/tools/rediscover` |
| `/v1/kb/ingest` | `/api/admin/kb/ingest` |

The seven user-facing `/v1/files` registrations for upload, upload-session
creation, part upload, completion, URL import, listing and deletion are removed.
Their `/api/files` equivalents remain; native listings return `items`. The
owner confirmed that Hermes and other external bots do not use the retired
routes. Existing Hermes adapters/settings and its separate document workspace
remain intact. Backend and browser changes belong to the same deployment;
an already-open browser tab needs a full reload to use the new paths. No schema
or configuration migration is required.

`/v1` retains the bot API/metadata and service protocols. `/v1/models` remains
the intentionally public catalogue. `/health`, `/metrics` and FastAPI's
documentation/schema routes remain public; application data operations retain
their authentication dependencies.

## Removed branches and maintained probe

- `AuthedUser`, `KBCaller`, `_compat_user`, `resolve_kb_caller` and the role-only
  `require_admin` helper are removed.
- Impossible `role == "bot"` admission branches are removed. Membership in the
  `bots` group continues to enforce bot exclusions.
- Unused unauthenticated `/v1/kb/stats` is removed, along with the unused local
  `src/audrey/static/app/` build. KB storage examples use `alice@example.com`.
- The [KB score probe](../../scripts/probes/kb_score_probe.py) reads
  `KB_SERVICE_TOKEN`, sends `X-Audrey-Service-Token`, and validates the returned
  `results` list. Missing credentials exit 2; transport, HTTP and malformed
  responses exit 1. A valid empty list is a successful zero-hit result.

## Verification and limits

Historical local verification recorded:

- Python: **8,579 passed, seven failed**, across 8,586 collected tests. All seven
  failures reproduced against unchanged source.
- Web unit suite: **655 passed, three failed**; all three failures reproduced
  against the unchanged baseline.
- Affected App tests: **46 passed**. Both affected production browser cases for
  saved memories and export/purge passed.
- Changed-file Ruff and ESLint, TypeScript, production build, source/auth route
  inventory and OpenAPI generation passed. Ruff retained the accepted
  `kb/reconcile.py` ASYNC240 exception.
- Web checks used Node 24.5.0, below the declared 24.15+ requirement.

The owner's acceptance report and closure are separate from these local
results; they do not erase the reproduced baseline failures. The owner later
reported the [Responses smoke](../../tests/smoke/smoke_responses_api.py) passed
from the laptop after replacing the obsolete Tailscale test URL with the working
LAN address. The authenticated text-only stream returned HTTP 200, delivered the
requested sentinel and ended with `response.completed`. The smoke validated
typed event order, contiguous sequence numbers, stable IDs, matching final text
and consistent usage totals. It also verified explicit HTTP 400 rejection of
`background: true` with `responses_feature_unsupported`. This establishes those
deployed Responses behaviors; other bot/API features retain their own evidence.

Reusable checks remain in the
[native auth cutover smoke](../../tests/smoke/smoke_native_auth_cutover.py) and
Responses smoke. Operational instructions live in
[live smoke testing](../reference/live-smoke-testing.md) and the
[testing protocol](../reference/testing-protocol.md). This record is not a
deployment or acceptance handoff.

## Teaching scope

Lessons **1, 2, 3, 4, 5, 6b, 10, 13, 14 and 16** contain affected identity
examples, scope statements, routes or source citations. No lesson prose was
authored or refreshed in this cleanup. Any refresh remains a separate reviewed
unit under the [lesson workflow](../../AGENTS.md); campaign closure does not
authorize automatic lesson rewriting.
