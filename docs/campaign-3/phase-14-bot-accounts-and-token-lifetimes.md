# Campaign 3 Phase 14 - bot accounts and token lifetimes

**Status:** Slice 14A is complete and live-accepted on October 7, 2026.

## Goal

Give automation accounts an explicit, limited access role; allow intentional
never-expiring personal access tokens; and make each Admin Panel opening read
current server state.

## Slice 14A

### Built-in Bot role

Schema migration 18 adds **Bots** as a protected access group and expands the
legacy model-audience constraint to accept `bots` without losing existing model
policies. A bot account receives `users` and `bots` memberships:

- `users` retains Audrey's normal workflow models;
- `bots` grants only models whose Public role list or audience includes Bots;
- the account remains a non-administrator and cannot use administration routes;
- existing personal-token scope checks still apply.

Pending accounts can be approved directly as a Bot. Active accounts can move
between User, Tester, Bot, custom roles, and Administrator from the existing
Role control. The model editor lists Bots beside the other protected roles, so
an administrator chooses which direct models an automation may invoke.

The approval endpoint now accepts `role: user | tester | bot`. Its former
`tester` boolean remains accepted for older clients and smokes.

### Explicit permanent personal tokens

The token lifetime field accepts `0` to mean no expiration. The API returns
`expires_at: null`, the settings list displays **Never**, and authentication
continues to validate the token's digest, owner status, scopes, and revocation
state on every use. Values from 1 through 365 keep the existing dated expiry;
negative values remain invalid.

The existing schema stores the no-expiration value as an empty timestamp in its
non-null legacy column. This avoids a destructive token-table migration while
the typed API exposes the intended nullable contract.

### Fresh administration data

The Admin Panel already unmounts when closed and loads users, models, and roles
when mounted. Those three list requests now use `cache: no-store`, preventing a
browser cache from returning the prior account roster when the panel is opened
again. The browser contract simulates a new signup between openings and
requires a second request plus the new account row.

## Automated contracts

- Schema 18 upgrades old databases, preserves model access policies, and creates
  the protected Bots group.
- A pending account can be approved as Bot and receives exactly `bots` plus
  `users`.
- A Public model assigned only to Bots is visible to that bot and remains
  subject to the ordinary model authorization path.
- A zero-day token is returned with no expiry, authenticates successfully, and
  still fails once revoked or its owner is disabled under existing tests.
- Negative token lifetimes fail request validation.
- The token settings UI submits `0` and renders **Never**.
- Closing and reopening the Admin Panel fetches a newly added account without a
  page reload.

## Laptop result

**Passed, 2026-10-02.** The focused backend gate passes 90 tests. Python
compilation and changed-file Ruff pass. The full hermetic backend suite passes
3,108 tests with one existing FastAPI deprecation warning. The diff check is
clean. The laptop has no Node runtime, so TypeScript, Vitest, build, and
Playwright remain part of the `audrey-ui` build and native browser gate.

## Native acceptance result

**Passed, 2026-10-07.** The user confirmed fresh applicant data when
reopening Admin Panel, Bot approval/role changes, Bot-only model visibility,
and zero-day tokens displaying Never. The final revocation check also
passed: the same token received HTTP 200 from protected `/api/models`
before revocation and HTTP 401 afterward. Slice 14A is closed.

The temporary smoke instructions were deleted after this confirmation.
Keep this result; do not request another check for unchanged behavior.
