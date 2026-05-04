# Phase 14 — Same-origin upload at `chat.builtryte.xyz/upload` with OWUI auth

**Goal:** make `/upload` reachable from outside the LAN, and make identity
come from OWUI instead of a user-typed email field.

Why now: Phase 13 only worked from a machine that could reach the Unraid
LAN IP, because the upload page asked the user to type their email and
trusted it. That's fine for a LAN tool; it's unacceptable for anything
published through the tunnel. Phase 14 fixes both problems at once by:

- Routing `/upload` and `/v1/files/*` through the existing
  `chat.builtryte.xyz` tunnel (path-based ingress, same origin as OWUI).
- Validating every request against OWUI's own JWT via
  `GET /api/v1/auths/`. The browser already stores that token in
  `localStorage.token` after login — the whole point of same-origin is
  that `/upload` can read it.

**What changed:**

- `src/audrey/auth.py` — new FastAPI dependency `require_user` that
  proxies the `Authorization: Bearer …` header to OWUI. 30-second token
  cache. 401 propagates; 502 on OWUI failure. CSRF-safe: we only read
  the header, never the OWUI cookie.
- `src/audrey/routes/files.py` — all three endpoints depend on
  `require_user`; `?user=` / `X-User` / form-field user params are gone.
  `user = me.email` is now the canonical identity.
- `src/audrey/static/upload.html` — no user input field. Reads
  `localStorage.token`, attaches `Authorization: Bearer` to every
  request, redirects to `/` on 401.
- `.env.example` — new `OWUI_URL` (defaults to `http://open-webui:8080`
  on `ollama-net`).
- Cloudflared tunnel config — ordered ingress rules so `/upload` and
  `/v1/files/*` go to `audrey-ai:8000` and everything else stays on
  `open-webui:8080`.

**Prereqs:**

- Phase 13 green on the LAN (upload, list, delete, cross-user isolation
  verified — see phase-13-deploy.md step 12).
- OWUI reachable at `http://open-webui:8080` on `ollama-net` from the
  `audrey-ai` container.
- `cloudflared` already fronting OWUI for `chat.builtryte.xyz`.

---

## 1. Deploy the code

Laptop:

```bash
git pull    # already committed; push if needed
```

Unraid:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 30 audrey-ai | grep -E "ready|auth|files:"
```

Add `OWUI_URL` to the Unraid `.env` (or the Unraid container template)
if it isn't already there. Default is `http://open-webui:8080`, which
matches the current `ollama-net` hostname — only override if you moved
OWUI.

---

## 2. Update the cloudflared tunnel

`cloudflared` picks the **first matching** ingress rule, so the order
matters. `/upload` and `/v1/files/*` must come *before* the
catch-all OWUI rule.

Edit the tunnel config (typically `/mnt/user/appdata/cloudflared/config.yml`
or the Unraid template):

```yaml
ingress:
  - hostname: chat.builtryte.xyz
    path: ^/upload(/.*)?$
    service: http://audrey-ai:8000
  - hostname: chat.builtryte.xyz
    path: ^/v1/files(/.*)?$
    service: http://audrey-ai:8000
  - hostname: chat.builtryte.xyz
    service: http://open-webui:8080
  - service: http_status:404
```

The tunnel container must be on `ollama-net` to reach `audrey-ai:8000`.
If it isn't already, add it to the compose network or the Unraid
template's extra networks.

Restart the tunnel:

```bash
docker restart cloudflared
docker logs --tail 20 cloudflared | grep -Ei "ingress|rules"
```

You should see the two new rules registered.

---

## 3. Smoke tests

All of these run from your laptop / phone / anything that can hit
`https://chat.builtryte.xyz`. Replace `<JWT>` with the token you pull
from the browser devtools: `localStorage.getItem("token")` after
logging into `chat.builtryte.xyz`.

### 3.1 Routing — `/upload` hits audrey, not OWUI

```bash
curl -sI https://chat.builtryte.xyz/upload | head -1
curl -s  https://chat.builtryte.xyz/upload | grep -oE "Audrey · Knowledge Upload|Open WebUI" | head -1
```

Expected: `HTTP/2 200` and `Audrey · Knowledge Upload`. If you see
`Open WebUI`, the ingress order is wrong — audrey's rules must come
before the catch-all.

### 3.2 Unauthenticated `/v1/files` returns 401

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://chat.builtryte.xyz/v1/files
```

Expected: `401`. If you get `200` or `403`, auth isn't wired.

### 3.3 Authenticated list works

```bash
TOKEN='<JWT>'
curl -sS -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files | jq '.user, .files | length, .total_bytes'
```

Expected: your OWUI email, file count, total bytes. The `user` field is
what the UI uses to show "Signed in as …".

### 3.4 Authenticated upload

```bash
echo "phase 14 smoke test" > /tmp/p14.txt
curl -sS -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/p14.txt" \
  https://chat.builtryte.xyz/v1/files | jq
```

Expected: `{file_id, filename: "p14.txt", mime: "text/plain", kind: "text", chunks: 1, …}`.

Then open `https://chat.builtryte.xyz/upload` in the browser. The
header should read **Signed in as you@…** and the file should show in
the list. Drop another file in — it should upload without prompting
for identity.

### 3.5 Forged / stale token → 401 and redirect

From the browser, open devtools console on `/upload` and run:

```js
localStorage.setItem("token", "garbage");
location.reload();
```

Expected: the page's own fetch to `/v1/files` returns 401, the JS wipes
the bad token, and the browser redirects to `/`. Log back in with OWUI
to restore normal operation.

From curl:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer garbage" \
  https://chat.builtryte.xyz/v1/files
```

Expected: `401`.

### 3.6 Cross-user delete blocked

Log in as user A, upload a file, note the `file_id`. Log in as user B
(different OWUI account) and try to delete A's file:

```bash
TOKEN_B='<JWT for user B>'
FILE_ID='<A's file id>'
curl -sS -X DELETE -H "Authorization: Bearer $TOKEN_B" \
  https://chat.builtryte.xyz/v1/files/$FILE_ID | jq
```

Expected: response says `deleted: true`, but A's file is **still
there** — the double-filter on `(file_id, user)` in Qdrant means B's
delete scrolled an empty filter set. Verify by logging back into A's
upload page; the file should still be listed.

### 3.7 OWUI down → 502, not 500 or silent 401

Stop OWUI briefly:

```bash
docker stop open-webui
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  https://chat.builtryte.xyz/v1/files
docker start open-webui
```

Expected: `502`. The distinction matters: 401 means "you're not logged
in" (user-fixable); 502 means "auth backend is broken" (ops problem).
Note the token cache is 30 s, so if you ran a request in the last 30 s
you'll still get 200 — wait it out or restart `audrey-ai`.

---

## 4. Disable OWUI's own file attach (if not already done in Phase 13)

Phase 14 doesn't change Phase 13's stance: OWUI should not also be
accepting uploads. Confirm the admin-side toggle is still off:

1. OWUI → Admin Panel → Settings → Interface → **File Upload** (toggle
   off for the default group), or
2. set `ENABLE_RAG_HYBRID_SEARCH=false` and revoke `file_upload` from
   the default user group — v0.9.2 exposes this as a per-group
   permission, not a top-level env var.

---

## 5. Rollback

If the tunnel ingress breaks OWUI, comment out the audrey rules and
restart `cloudflared`. Audrey keeps running on the LAN at
`http://<unraid-ip>:8000/upload` regardless — Phase 13's LAN access is
not disturbed by any of this.

If auth breaks for a single user but others are fine, clear the token
cache by restarting `audrey-ai` (the `require_user` cache is
in-process).

---

## 6. Follow-ups (not Phase 14)

- Persist upload metadata outside Qdrant payloads (sqlite) so list
  doesn't scroll the whole collection.
- Admin endpoint to force-evict tokens from the auth cache (`POST
  /admin/auth/clear` behind `require_user` + role check).
- A "shared docs" collection: admin-only upload, all authed users read.
