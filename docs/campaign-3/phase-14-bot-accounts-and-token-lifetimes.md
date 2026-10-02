# Campaign 3 Phase 14 - bot accounts and token lifetimes

**Status:** Slice 14A is laptop-complete and awaits native acceptance.

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

## Native acceptance gate

1. Rebuild `audrey` and `audrey-ui`. Confirm startup migrates the application
   database to schema 18 and both services become ready.
2. Open **Admin Panel -> Accounts**, close it, and complete a sign-in with a new
   email in another browser profile. Reopen the panel without reloading Audrey.
   Confirm the new Pending account appears.
3. Click **Approve as bot**. Confirm the account becomes Active and its Role
   control says **Bot**. Change it to User and back to Bot once.
4. In **Models**, choose one disposable direct model, set it Public, open
   **Edit...**, grant only **Bots**, and save. Confirm an ordinary account cannot
   see that direct model while the bot account can.
5. Sign in as the bot, open account settings, create a personal token with a
   lifetime of `0`, and confirm its Expires value is **Never**. Use that token
   against `http://192.168.1.11:8000/v1/models`; confirm the bot-assigned model
   is present.
6. Revoke the token and confirm the same authenticated request returns HTTP 401.
7. Restore the model's original visibility and roles. Delete the disposable bot
   account or restore its original role.

## Completion gate

Close Slice 14A after the refreshed-account, Bot model access, permanent-token,
and revocation checks pass in the deployed application.
