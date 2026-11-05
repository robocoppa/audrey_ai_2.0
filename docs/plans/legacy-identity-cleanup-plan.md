# Legacy identity cleanup

**Status:** Proposed, November 4, 2026. Both decisions made the same day; not
started.

Audrey's identity setup today is Cloudflare Access for the browser, personal
access tokens for scripts and bots, a shared service token for the tools sidecar
and media workers, and an Audrey-owned `Principal` for every request. Before the
cutover, routes identified callers by email. The cutover kept that older shape
wherever rewriting it wasn't needed: `AuthedUser` (`src/audrey/auth.py:35`),
whose `email` field now carries the storage namespace, the role-only
`require_admin` (`auth.py:283`), and a set of `/v1/...` routes built on them.
This plan finishes the move. Every route receives a `Principal`, browser and
operator routes live under `/api`, `/v1` keeps only the bot API and service
routes, and the old types are deleted.

Nothing from the old login mechanism still works. `require_user` (`auth.py:156`)
accepts only an Access assertion or an `aud_pat_` token, the `/v1` chat route
ignores the client-sent `user` field, and no old client headers or shared
secrets are read. The work is about shapes and duplicates, with one small
exposure: `GET /v1/kb/stats` has no authentication. It returns only the shared
knowledge-base collection names and their point counts.

## What's left, and who uses it

| Leftover | Where | Used today by | Action |
|---|---|---|---|
| `AuthedUser`, and `require_user` returning it | `auth.py:35`, `auth.py:156` | the `/v1` routes below; `_compat_user` shims in `routes/app/files.py:113` and `project_context.py:43` | Replace with `Principal`, then delete (phases 2–4) |
| `require_admin`, which checks only `role` | `auth.py:283` | `/v1/admin/*`, `/v1/kb/ingest` (`routes/kb.py:873`), `/v1/tools/rediscover` (`main.py:710`) | Use `require_admin_principal`, then delete (phase 3) |
| `/v1/admin/*`: readiness, repair, repair status, chat-archive rebuild, prune and stats, KB reconcile, skills rediscover | `routes/admin.py` | live smoke tests, with `AUDREY_ADMIN_JWT`; no browser caller | Move to `/api/admin/*` (phase 3) |
| `/v1/me/*`: memories, chat-history export and delete, data purge, repair status | `routes/user_data.py` | **browser Settings** (`web/src/api.ts:1130-1210`) and live smoke tests | Move to `/api/me/*` (phase 3) |
| User-facing `/v1/files`: upload, upload sessions, from-url, list, delete | `routes/files.py` | only `tests/smoke/smoke_native_auth_cutover.py`; the browser and bot API use `/api/files` | Delete, keeping the service routes (phase 4) |
| `KBCaller` and the user branch of `resolve_kb_caller` | `auth.py:49`, `auth.py:316` | the sidecar, through the service-token branch; no user caller found | Delete; `/v1/kb/query` becomes service-only (phase 4) |
| `/v1` bot API internals taking `AuthedUser` and `me.email` | `routes/openai/` | bots, through `/v1/chat/completions` and `/v1/responses` | Keep the API; pass a `Principal` and use `storage_namespace` (phase 4) |
| `GET /v1/kb/stats`, with no auth | `routes/kb.py:902` | nothing found | Delete, or move under `/api/admin` (phase 1) |
| The `payload.user` comparison | `routes/openai/routes.py:181` | nothing; it only logs | Delete; the field stays ignored (phase 1) |
| `role == "bot"` checks | ten places, listed below | nothing; roles are only `user` and `admin`, and each place also tests the `bots` group | Delete the role half (phase 1) |
| `scripts/probes/kb_score_probe.py` | line 58 | broken: it calls `/v1/kb/query` without credentials and gets 401 | Send the service token, or delete (phase 1) |
| A real-looking example address in a docstring | `src/audrey/kb/user_store.py:8` and `:14` | nothing | Use `alice@example.com` (phase 1) |
| Email-valued storage namespaces | data: accounts created before the cutover | per-user Qdrant collection names and payloads, upload directories, memories, the chat archive | Keep; see the decisions |
| `src/audrey/static/app/` | laptop only: gitignored, dockerignored, never mounted | nothing | Delete locally (phase 1) |

The `role == "bot"` checks are at `agent_executor.py:166`, `watch_executor.py:101`,
`agent_knowledge_drafts.py:138` and `:154`, `agent_file_inputs.py:241`,
`agent_files.py:159`, `tax_review.py:115`, `private_task_admission.py:122`,
`skills/native.py:64` and `app_state/private_tasks.py:184`.

No repository under `~/Documents/github` other than this one calls the routes
being moved or deleted. Before phase 4 deletes the user-facing `/v1/files`
routes, also check the Hermes adapter and any other bot that holds a token.

## Decisions

The owner decided both on November 4, 2026.

1. **`/v1/kb/query` becomes service-only.** Today a personal access token works
   as well as the service token, but only the sidecar calls it, through the
   service token. Phase 4 removes the token path, which deletes `KBCaller` and
   `resolve_kb_caller`.
2. **Email-valued storage namespaces stay.** Accounts from before the cutover
   keep their original email as `storage_namespace`, and their per-user Qdrant
   collections and upload directories stay named from it. Code treats the value
   as opaque, so nothing in this plan depends on it, and no data migration is
   planned.

## Phases

Each phase leaves Audrey working and can ship on its own. The browser, backend,
sidecar and smoke tests all live in this repository and deploy together, so
moved routes need no transition period. A browser tab left open across a deploy
needs a reload.

### Phase 1 — Small fixes

- Delete `GET /v1/kb/stats`, or move it under `/api/admin` if the owner uses it.
- Delete the `payload.user` comparison in `/v1/chat/completions`. The field stays
  ignored.
- Delete the role half of the ten `role == "bot"` checks and keep each `bots`
  group test. The existing bot-exclusion tests must still pass.
- Make `kb_score_probe.py` send the service token, or delete it.
- Replace the example address in `user_store.py`.
- Delete the local `src/audrey/static/app/`.

### Phase 2 — Shared helpers take a principal

The native file routes and Project context reach the older `/v1/files` code
through `_compat_user`, which builds an `AuthedUser` from a `Principal`. Move the
shared upload, list, read and delete logic out of `routes/files.py` into a module
whose functions take the `Principal`, or its `storage_namespace` and `user_id`.
Call it from `/api/files`, Project context and the Responses file readers, and
delete both `_compat_user` functions.

Check: the file, Project context, Responses file-input and response-storage
tests.

### Phase 3 — Browser and operator routes move to `/api`

- Move `/v1/me/*` to `/api/me/*` and keep each route's rule: `require_scope("compat:full")`
  where `require_user` stands today, and `require_provider_principal` where it
  already applies, as on data purge. Update `web/src/api.ts` and its tests.
- Move `/v1/admin/*`, `/v1/tools/rediscover` and `/v1/kb/ingest` to `/api/admin/*`
  behind `require_admin_principal`, which makes them browser-only like the rest of
  administration.
- Update the live smoke tests that call these paths
  (`grep -rln "/v1/admin/\|/v1/me/" tests/smoke`). Their Access-assertion
  credentials keep working.
- Delete `require_admin`.

Check: the route and web tests. The owner then runs the affected live smokes.

### Phase 4 — Only the bot API and service routes stay under `/v1`

- Delete the user-facing `/v1/files` routes and keep the `/v1/files` service
  routes the sidecar and workers use. Point `smoke_native_auth_cutover.py` at
  `/api/files`.
- Put `/v1/kb/query` and `/v1/kb/query/image` behind `require_service`, delete
  `resolve_kb_caller` and `KBCaller`, and update `tests/test_kb_query_auth.py`
  to expect a personal token to be refused.
- Give the `/v1` bot API a dependency that returns a `Principal` (an active
  account, with `compat:full` for tokens), and replace `me.email` with
  `principal.storage_namespace` throughout `routes/openai/`.
- Delete `AuthedUser`, `KBCaller` and `require_user`'s `AuthedUser` return.

Check: the full test suite, then the live bot smoke with `AUDREY_EVAL_API_KEY`.

### Phase 5 — Docs and lessons

Update `docs/guides/hermes-reference.md` (its note on `/v1/admin/*`), any operator
reference that names a moved path, and the lessons that describe the old shapes:
4, 5 and 13 (the `/v1/admin` routes), 6b §2.3 (the `AuthedUser` paragraph) and
14 (`require_user`, `require_admin` and the misleading `email` field). Lesson prose
follows the lesson workflow in AGENTS.md.

## Done when

- `grep -rn 'AuthedUser\|KBCaller\|require_admin\b\|_compat_user\|role == "bot"' src/`
  finds nothing.
- The only routes without an auth dependency are `/health`, `/metrics` and the
  static `/v1/models` list.
- `/v1` serves only the bot API and service-token routes.
- The test suite, the web tests and the build pass, and the owner's live smokes
  pass.

## Optional, later

- Narrow `resolve_external_identity` to the Access sign-in it serves. Tests use
  its extra parameters as an account factory, so give them a factory first.
- Rename `KB_SERVICE_TOKEN` to a name that matches its wider use, changing `.env`,
  compose and the sidecar together.
- Make `admins` membership the only source of administrator rights and drop the
  mirrored `role` column.
- Rename pipeline and tool parameters called `user_id` that actually carry the
  storage namespace.
