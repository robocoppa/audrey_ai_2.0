# Phase 26 — auth boundary fix

**Goal:** close the auth gaps surfaced by the 2026-04-29 audit. Six
small pieces, all touching the route-level identity story:

1. **`/v1/chat/completions` requires auth.** Was unauthenticated and
   trusted `payload.user`. Now `Depends(require_user)`; identity comes
   from `AuthedUser.email` (the canonical user-identity field shared
   with files/admin/memory routes). `payload.user` is logged-but-ignored.
2. **`/v1/kb/ingest` requires admin.** Was unauthenticated. Now
   `Depends(require_admin)`. Query routes (`/v1/kb/query`, `/query/image`,
   `/stats`) stay internal-only — they're called by custom-tools' ReAct
   path and don't get an auth header. The protection model is "untunnel
   `/v1/kb` from cloudflared so it can't be reached publicly."
3. **`/v1/tools/rediscover` requires admin.** Was unauthenticated and
   could mutate the live tool registry. Now `Depends(require_admin)`.
4. **`memory_recall` cross-user leak fixed.** Pre-fix: `recall(key)`
   scrolled across all users, returned newest match. Post-fix:
   `recall(key, *, user)` filters by both. Tools-server schema requires
   `user`; audrey's dispatcher overrides it with the pipeline's user_id
   via `_USER_SCOPED_TOOLS`.
5. **Grafana password forced via env var.** Was hardcoded `changeme`.
   Now `${GRAFANA_ADMIN_PASSWORD:?...}` — compose fails on `up` if
   unset.
6. **Cloudflared `^/v1/kb` rule removed (Unraid-side step).** Without
   this, the public tunnel exposes the now-internal-only KB routes.

**Audit findings addressed:**
- High #1 (chat/completions unauth) → piece 1
- High #2 (KB unauth) → pieces 2 + 6
- High #3 (memory_recall leak) → pieces 4 + 5 (tools-server side)
- Medium #4 (rediscover unauth) → piece 3
- Medium #6 (Grafana changeme) → piece 5

**Out of scope (future phases):**
- Audit medium #5 (SSRF in image URL embedding) — Phase 27.
- Audit medium #7 (watcher missing on_deleted) — Phase 27.
- Audit medium #8 (no tests) — Phase 28.
- Audit low #9 (docs drift) and #10 (mutable image tags) — Phase 29+.

**Prereqs:** all phases through 25 verified. Phase 24b verified
(rebuilds are now ~6s for source-only changes; this phase is mostly
source changes).

---

## 1. Set the Grafana password env var FIRST

Before any rebuild, set `GRAFANA_ADMIN_PASSWORD` somewhere compose
will pick it up. Two options.

**A. `.env` file alongside compose (cleanest).**

Run each of these three commands on its own line. Don't paste them
as a block — terminal paste behavior is inconsistent on multi-line
shell snippets and you can end up with commands smashed together.

Generate a random password and write it to the `.env`:
```bash
echo "GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 24)" >> /mnt/user/appdata/audrey_ai_2.0/monitoring/.env
```

Tighten the file's permissions:
```bash
chmod 600 /mnt/user/appdata/audrey_ai_2.0/monitoring/.env
```

Print it once so you can copy it somewhere safe (Grafana stores the
password hashed; you can't read it back from the running container):
```bash
cat /mnt/user/appdata/audrey_ai_2.0/monitoring/.env
```

**B. Shell export (one-shot, lost on shell exit).**
```bash
export GRAFANA_ADMIN_PASSWORD="<your-password>"
```

If you don't set it, `docker compose up -d` on the monitoring stack
will fail with:
```
GRAFANA_ADMIN_PASSWORD must be set in the environment
```

**Important:** if Grafana's existing admin login is `changeme`,
setting this env var **does not** change the live password (Grafana
stores hashed credentials in `grafana-data/grafana.db`; the env only
seeds initial creation). To change the existing password, after the
restart in step 3:

```bash
docker exec -it grafana grafana-cli admin reset-admin-password "<new-password>"
```

Or log into Grafana UI as admin/changeme one last time and change it
through Settings.

---

## 2. Pull on Unraid

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

Brings in:
- `src/audrey/routes/openai.py` — auth + identity changes
- `src/audrey/routes/kb.py` — `require_admin` on `/ingest`
- `src/audrey/main.py` — `require_admin` on `/v1/tools/rediscover`
- `src/audrey/tools/dispatch.py` — `memory_recall` in `_USER_SCOPED_TOOLS`
- `tools-server/app.py` + `tools-server/db.py` — memory_recall scoping
- `monitoring/compose.yaml` — `GRAFANA_ADMIN_PASSWORD` requirement

---

## 3. Rebuild and restart the affected containers

```bash
# audrey-ai (Phase 26 piece 1, 2, 3, 4 affect this image)
docker compose up -d --build audrey-ai

# custom-tools (Phase 26 piece 5 affects this image)
docker compose up -d --build custom-tools

# Grafana stack picks up the new env var
cd /mnt/user/appdata/audrey_ai_2.0/monitoring
docker compose up -d   # uses the .env file or shell-exported var
```

Phase 24b's layer split means audrey-ai rebuilds in ~6s. custom-tools
still has the old single-install Dockerfile (Phase 21 footnote — the
flat-script layout isn't pyproject-friendly), so its rebuild is ~30s.
Grafana doesn't rebuild — `up -d` just recreates the container with
the new env.

---

## 4. Untunnel `/v1/kb` (cloudflared rule edit)

Edit cloudflared's tunnel config on Unraid to **remove** the
`^/v1/kb` rule. The rule sends public traffic with that path to
`audrey-ai:8000` — Phase 26 makes those routes internal-only, so
public exposure becomes a leak.

The cloudflared config location varies by setup; if you manage rules
through Cloudflare's web UI, find the entry that maps a hostname +
path `^/v1/kb` and delete it. The catch-all `*` rule (which sends
unmatched paths to OWUI) takes over for any `/v1/kb*` requests
arriving at the tunnel — they'll get OWUI's 404.

Restart cloudflared after the edit:
```bash
docker restart Cloudflared
```

(The container name in CONTINUITY shows up as `Cloudflared` —
capital C.)

Verify from outside Unraid:
```bash
curl -sS https://<your-tunnel-host>/v1/kb/stats | head -5
```

Expect: HTML output starting with `<!doctype html>` — that's OWUI's
SPA index.html being served by the catch-all `*` rule. OWUI is a
single-page app, so cloudflared's catch-all returns 200 + index.html
for any unmatched path; the SPA router would then show its own 404
page client-side.

What you want to **NOT** see: a JSON response like
`{"collections": [...], "global": {...}}` — that would mean the
`^/v1/kb` rule is still routing to audrey.

To make the test less noisy, you can also check for an audrey
fingerprint specifically:
```bash
curl -sS https://<your-tunnel-host>/v1/kb/stats | grep -c '"collections"'
```
Expect: `0` (the substring `"collections"` won't appear in OWUI's
HTML shell).

---

## 5. Smoke tests

### 5.1 Container starts cleanly

```bash
docker compose logs --tail 30 audrey-ai | grep -E 'ready|ERROR|Traceback'
docker compose logs --tail 5 custom-tools | grep -E 'Application startup|ERROR|Traceback'
```

Expect: `ready: ...` on audrey, `Application startup complete` on
custom-tools, no errors.

### 5.2 Chat completions REJECTS unauthenticated requests

Each command below is a single line — copy-paste each one as a unit,
not as a block. Mixing multiple commands per paste tends to smash
them together if your terminal eats the trailing newline.

```bash
curl -sS -o /dev/null -w "%{http_code}\n" -X POST -H "Content-Type: application/json" -d '{"model":"audrey_fast","messages":[{"role":"user","content":"hi"}]}' http://localhost:8000/v1/chat/completions
```

Expect: `401`. Pre-Phase-26 this returned `200` and a real chat
response — that's the gap that's now closed.

### 5.3 Chat completions WORKS with auth

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"model":"audrey_fast","messages":[{"role":"user","content":"two-sentence intro to rsync"}]}' http://localhost:8000/v1/chat/completions | jq -r '.choices[0].message.content'
```

Expect: a real answer.

### 5.4 Spoofed `payload.user` is ignored

This is the headline behavior change. Send a request with a *valid*
auth token but a *spoofed* `user` field; the spoofed value should be
logged-as-ignored, and the real authenticated id should be what shows
up in fairness/inflight buckets.

Run this curl on its own:
```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"model":"audrey_fast","user":"evil-spoof@nope.com","messages":[{"role":"user","content":"hi"}]}' http://localhost:8000/v1/chat/completions -o /dev/null
```

Then on its own line, check the logs:
```bash
docker compose logs --since 1m audrey-ai | grep -E 'fast_path task=|payload.user'
```

Expect:
- A `chat.completions: payload.user='evil-spoof@nope.com' ignored
  (auth user='<real-admin-id>')` debug line (only at DEBUG log level
  — may not appear if log level is INFO).
- Subsequent log lines (`fast_path task=`, etc.) reference the real
  user, not `evil-spoof`.
- If you `curl /metrics` the `audrey_user_inflight_blocked_seconds`
  histogram should NOT show new buckets keyed off `evil-spoof`.

### 5.5 KB ingest requires admin

Three separate commands, each on its own line. Run one at a time.

Without auth (expect `401`):
```bash
curl -sS -o /dev/null -w "%{http_code}\n" -X POST -H "Content-Type: application/json" -d '{"paths":["/datasets/geology"]}' http://localhost:8000/v1/kb/ingest
```

With non-admin auth (expect `403`; skip if you only have admin tokens):
```bash
curl -sS -o /dev/null -w "%{http_code}\n" -X POST -H "Authorization: Bearer $REGULAR_USER_TOKEN" -H "Content-Type: application/json" -d '{"paths":["/datasets/geology"]}' http://localhost:8000/v1/kb/ingest
```

With admin auth (expect `200`, or `503` if KB isn't initialized —
either means auth passed):
```bash
curl -sS -o /dev/null -w "%{http_code}\n" -X POST -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"paths":["/datasets/geology"]}' http://localhost:8000/v1/kb/ingest
```

### 5.6 Tools rediscover requires admin

```bash
# Without auth → 401
curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST http://localhost:8000/v1/tools/rediscover

# With admin auth → 200 + a JSON body listing rediscovered tools
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/v1/tools/rediscover | jq
```

Expect: 401, then a body like
`{"tools": [...], "count": 6, "servers": [...]}`.

### 5.7 KB query routes still work via the ReAct path

A real chat that triggers a KB lookup. The ReAct loop's tool-dispatch
path goes audrey → custom-tools → audrey's `/v1/kb/query` (internal
docker-DNS, no auth required). If we accidentally added auth to those
routes, this would break.

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","messages":[{"role":"user","content":"use kb_search to find info about geology"}]}' \
  http://localhost:8000/v1/chat/completions \
  | jq -r '.choices[0].message.content' | head -10

docker compose logs --since 1m audrey-ai | grep -E 'react: round=|tool_calls='
```

Expect: a real answer + log lines showing `tool_calls=N` for kb_search.
If the ReAct loop got 401s on the kb_search tool call, you'll see
`tool_calls=1` followed by the model's "I tried to call kb_search but
got an error" recovery prose — that's the failure mode to look for.

### 5.8 memory_recall is now user-scoped

The setup needed:
1. User A writes a memory with key `"server_password"`.
2. User B tries `memory_recall(server_password)`.
3. Pre-fix: B gets A's memory.
4. Post-fix: B gets 404.

Without two real users it's hard to test directly. The mechanical
check: tools-server's `memory_recall` route now requires a `user`
field in the JSON payload.

```bash
# Direct call to custom-tools (use docker exec since localhost:8001
# is a bind from the host that may or may not be exposed):
docker exec custom-tools curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST -H "Content-Type: application/json" \
  -d '{"key":"anything"}' \
  http://localhost:8001/memory_recall
```

Expect: `422` (Unprocessable Entity — pydantic rejects the missing
`user` field). Pre-fix this would have returned 200 with whatever
memory matched the key.

```bash
# With a valid user:
docker exec custom-tools curl -sS -o /dev/null -w "%{http_code}\n" \
  -X POST -H "Content-Type: application/json" \
  -d '{"user":"some-user-id","key":"nonexistent"}' \
  http://localhost:8001/memory_recall
```

Expect: `404` (no memory for that user+key combination — or 200 if
that user really has a memory at that key, which is fine too).

### 5.9 Grafana login still works

Open <http://192.168.1.11:3000>. The existing admin password (set
during prior phases — likely `changeme` if you never changed it)
should still log you in. The env var only affects new admin creation.

If you used `grafana-cli admin reset-admin-password` in step 1, log
in with the new password.

### 5.10 KB is no longer publicly tunneled

From a non-LAN box:
```bash
curl -sS https://<your-tunnel-host>/v1/kb/stats | grep -c '"collections"'
```

Expect: `0`. The `"collections"` substring appears in audrey's real KB
stats response but not in OWUI's HTML index. Pre-fix this would have
returned a positive number (one match per JSON-key occurrence). For
visual confirmation that you're getting OWUI's frontend instead:

```bash
curl -sS https://<your-tunnel-host>/v1/kb/stats | head -3
```

Expect: lines starting with `<!doctype html>` and `<html lang="en">`
— that's OWUI's SPA shell being served by cloudflared's catch-all
`*` rule. Audrey's `/v1/kb/stats` endpoint is unreachable from the
public tunnel.

If you DO see `"collections"` in the response, the cloudflared rule
wasn't removed correctly — recheck step 4.

Also script-friendly: the HTTP code is `200` (catch-all serves the
HTML shell with a 200), NOT `404`. Don't write monitoring on `200 ==
healthy` for this path; check the body for the `"collections"`
substring or for an audrey-specific marker instead.

---

## 6. Rollback

If anything fails:

```bash
git checkout <previous-sha> -- \
  src/audrey/routes/openai.py \
  src/audrey/routes/kb.py \
  src/audrey/main.py \
  src/audrey/tools/dispatch.py \
  tools-server/app.py \
  tools-server/db.py \
  monitoring/compose.yaml
docker compose up -d --build audrey-ai custom-tools

cd monitoring
docker compose up -d
```

For the cloudflared rule, re-add `^/v1/kb` → audrey-ai:8000 in the
tunnel config and restart Cloudflared.

For Grafana, if the password reset broke something, the live state
in `grafana-data/grafana.db` is unchanged — only the env var changed.
Reverting `monitoring/compose.yaml` to the pre-Phase-26 hardcoded
`changeme` and rerunning `up` brings you back exactly.

---

## 7. Operational notes

### When OWUI returns a different identifier

`require_user` resolves the authenticated user via OWUI's `/api/v1/auths/`
probe. The `id` field of `AuthedUser` is whatever OWUI returns —
typically a uuid like `7c513b30-ed0d-...`. If OWUI ever changes that
schema, all `_USER_SCOPED_TOOLS` calls would target a new bucket key,
and pre-Phase-26 memories would become unreachable. Monitor the
existing Phase 17 `audrey_auth_cache_size` metric for unusual churn.

### `payload.user` drift logging

The new code logs at DEBUG when `payload.user` differs from `me.email`
(e.g. an OpenAI client that hardcodes a different user id). If you
ever bump audrey-ai to log level DEBUG, watch for that line — recurring
appearances might indicate a misconfigured client or a probe attempt.

### Custom-tools auth is implicit

custom-tools still calls audrey's `/v1/kb/query` over docker-DNS
without an Authorization header. That works because:
1. `/v1/kb/query` doesn't have `Depends(require_user)` (it's
   internal-only by design).
2. The `user` field in the kb_search payload is set by audrey's
   `_USER_SCOPED_TOOLS` override before custom-tools sees it — the
   model can't spoof it.
3. cloudflared's `^/v1/kb` rule is now removed, so external traffic
   can't reach `/v1/kb/query` directly.

This three-layer defense is fine for the current architecture. If we
ever add public KB access, the model becomes "auth on the public route,
internal trust on the docker-DNS path."

### Grafana password reset

If you forget the password later and `grafana-cli admin reset-admin-password`
fails (e.g. Grafana not running), the nuclear option is to delete
`/mnt/user/appdata/prometheus/grafana-data/grafana.db` and restart
Grafana — it'll re-create the admin user with whatever
`GRAFANA_ADMIN_PASSWORD` is currently set to. Loses dashboards.
Better: keep the password somewhere safe.
