# Phase 16 — admin auth-cache endpoints

**Goal:** force every cached OWUI token to re-probe on next request,
without restarting `audrey-ai`. Useful when a token is revoked in OWUI
mid-session, or when the 30-second cache is too long for an incident
response.

Why now: Phase 14 stood up the OWUI-backed auth cache, but the only
way to evict it was `docker restart audrey-ai`. That kicks every
in-flight request and rebuilds the model registry / tool discovery
state — wildly disproportionate to "drop a few cached tokens."

What changed:

- **`src/audrey/auth.py`** — new `require_admin` dependency. Wraps
  `require_user`; 403s when the OWUI role is not `"admin"`. OWUI v0.9.2
  emits role strings as lowercase (`"admin"`, `"user"`, `"pending"`).
  `clear_auth_cache()` now returns the count of evicted entries; new
  `cache_size()` helper exposes the count for read-only checks.
- **`src/audrey/routes/admin.py` (new)** — two endpoints under
  `/v1/admin/...`, both behind `require_admin`:
  - `POST /v1/admin/auth/clear` — evicts every cached AuthedUser.
    Logs WARNING with the admin email and the count. Includes the
    caller's own cache row (intentional — their next request re-probes).
  - `GET  /v1/admin/auth/status` — returns the current cache entry count.
- **`src/audrey/main.py`** — `admin_router` registered.

Out of scope (deliberately):

- Per-token / per-email targeted eviction. The blunt clear is fine at
  current scale (cache cap 1024 entries).
- Rotating role / impersonation. Admin role is whatever OWUI says it is.
- Persisted audit log. The WARNING log line is enough for one admin.

**Prereqs:**

- Phase 14 verified (OWUI-backed auth working end-to-end).
- The admin's OWUI account has `role == "admin"` in OWUI's user list.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 20 audrey-ai | grep ready
```

No new env vars; nothing to configure.

---

## 2. Smoke tests

Run from your laptop. You'll need two tokens:

- `$TOKEN_ADMIN` — JWT for an OWUI admin account.
- `$TOKEN_USER` — JWT for a non-admin OWUI account.

Pull each from the browser devtools while logged into the respective
account: `localStorage.getItem("token")`.

### 2.1 Unauthenticated → 401

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST https://chat.builtryte.xyz/v1/admin/auth/clear
```

Expected: `401`. Plain `Missing bearer token.` from the auth chain.

### 2.2 Non-admin → 403

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST -H "Authorization: Bearer $TOKEN_USER" \
  https://chat.builtryte.xyz/v1/admin/auth/clear
```

Expected: `403`. Confirms the role check fires *after* the token is
validated as a real OWUI user — i.e. a non-admin can't pretend not to
exist to bypass the permission gate.

### 2.3 Admin status

```bash
curl -sS -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/status | jq
```

Expected: `{"cached_entries": N}` — N depends on how many tokens have
hit the cache in the last 30 seconds. At least 1 (your own status call
just landed in cache).

### 2.4 Admin clear

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/clear | jq
```

Expected: `{"cleared": N, "by": "your-admin-email"}` — N matches the
status from 2.3 (give or take, depending on probe timing).

Status right after should report 1 (the admin's own re-probe just
re-populated the cache):

```bash
curl -sS -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/status | jq
```

### 2.5 WARNING log line shows up

On Unraid:

```bash
docker compose logs --tail 100 audrey-ai | grep "auth cache cleared"
```

Expected: a line like

```
admin: auth cache cleared by you@proton.me (3 entries evicted)
```

This is your audit trail. If you see clears you didn't run, that's
a real signal — either someone else has the admin token, or the OWUI
admin role is granted too widely.

### 2.6 Token revoked in OWUI no longer works after a clear

This is the real-world use case the endpoint exists for: a token gets
compromised or a user is offboarded; you revoke them in OWUI; you want
that revocation to take effect *now*, not 30 seconds from now when the
audrey cache happens to expire.

Setup — export both tokens so they survive the multi-step flow:

```bash
export TOKEN_ADMIN='<admin JWT from your browser localStorage.token>'
export TOKEN_USER='<non-admin JWT from a second OWUI account>'
```

**Step 1 — confirm the user token works.** Should be 200:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN_USER" \
  https://chat.builtryte.xyz/v1/files
# expected: 200
```

**Step 2 — populate the audrey cache for that token.** Just hitting it
once does this; the request you ran in step 1 already cached it. Verify:

```bash
curl -sS -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/status | jq
# expected: cached_entries >= 2  (admin from this status call + user from step 1)
```

**Step 3 — revoke in OWUI.** Open OWUI → Admin Panel → Users → click the
non-admin user → either:
- demote them to `pending` (fastest reversible test), or
- click "Sign out all sessions" if your OWUI build exposes it, or
- delete the user (irreversible — only if you really want them gone).

**Step 4 — observe the cache window.** Within ~30 seconds of the user's
last successful request, audrey will *still* serve them because of the
TTL cache. This is the gap the endpoint closes:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN_USER" \
  https://chat.builtryte.xyz/v1/files
# expected (for up to ~30s after step 1): 200, even though OWUI now rejects them
# after the cache TTL elapses: 401
```

**Step 5 — clear the cache as admin.** Forces audrey to re-probe OWUI on
the next request from any token:

```bash
curl -sS -X POST -H "Authorization: Bearer $TOKEN_ADMIN" \
  https://chat.builtryte.xyz/v1/admin/auth/clear | jq
# expected: {"cleared": N, "by": "your-admin-email"}
```

**Step 6 — retry as the revoked user.** Should now fail immediately:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN_USER" \
  https://chat.builtryte.xyz/v1/files
# expected: 401
```

If you got `200` here, the clear didn't take effect (check the WARNING
log from 2.5 — was it logged?) or OWUI hasn't actually revoked the
token yet (the demote-to-pending path keeps the JWT signature valid but
the `GET /api/v1/auths/` probe should now return 401 because the user's
role is `pending`).

**Step 7 — restore the user.** If you used the demote-to-pending path,
flip them back to `user` in OWUI's admin panel. Confirm:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN_USER" \
  https://chat.builtryte.xyz/v1/files
# expected: 200
```

That single 200 → 401 → 200 sequence is the whole point of the endpoint:
revocation latency drops from "up to TTL seconds" to "milliseconds after
the admin clicks clear."

---

## 3. Browser console (no curl)

The admin can run the same calls from any page on
`chat.builtryte.xyz`. Open devtools console:

```js
const t = localStorage.getItem("token");
fetch("/v1/admin/auth/status", { headers: { Authorization: `Bearer ${t}` } })
  .then(r => r.json()).then(console.log);

fetch("/v1/admin/auth/clear", { method: "POST", headers: { Authorization: `Bearer ${t}` } })
  .then(r => r.json()).then(console.log);
```

No UI for this yet — keeping it ops-shaped, not user-shaped.

---

## 4. Rollback

Pure additive — there's nothing to roll back unless `require_admin` is
broken in a way that locks you out (it shouldn't; non-admins get 403
from the endpoint, but every other route still works for every user).
If `routes/admin.py` blows up at import, comment out
`app.include_router(admin_router)` in `main.py` and rebuild.

---

## 5. Follow-ups (not Phase 16)

- Targeted eviction: `DELETE /v1/admin/auth/cache/{email}` to drop one
  user's entries.
- Webhook from OWUI on user deletion / role change → hit
  `/v1/admin/auth/clear` automatically. Removes the manual step in 2.6.
- Tie into the future `/metrics` endpoint: `audrey_auth_cache_size`,
  `audrey_auth_cache_clears_total`.
