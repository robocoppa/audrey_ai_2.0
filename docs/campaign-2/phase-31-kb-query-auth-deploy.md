# Campaign 2 Phase 31 — KB query-route authentication (safe LAN publish of audrey-ai:8000)

**Status:** DEPLOYED + verified on box 2026-07-27. Gate live (`resolve_kb_caller`
grep=3), internal path with `X-Audrey-Service-Token` → 200, no-token → 401, 8000
published, LAN `/v1/kb/query` → 401 while passthrough works. 844 hermetic tests pass
(+17), ruff clean. Two-image change (audrey-ai + custom-tools).

---

## Deploy checklist

**Code (laptop) — commit:**
- [ ] `feat(tools): send KB service token to Audrey`
- [ ] `feat(kb): authenticate KB query routes with a service token`
- [ ] `docs(kb): add phase-31 KB query-auth deploy plan`

**Box** (`/mnt/user/appdata/audrey_ai_2.0`), in order:
1. [ ] `echo "KB_SERVICE_TOKEN=$(openssl rand -hex 32)" >> .env`
2. [ ] `docker compose up -d --build custom-tools`
3. [ ] `docker compose up -d --build audrey-ai`
4. [ ] Smoke (below): 401 without token, 200 with; then a chat turn in OWUI still returns KB hits.
5. [ ] Publish 8000: add the `ports` block, `docker compose up -d --force-recreate audrey-ai`, retire the stopgap proxy/allowlist, update the compose security comment.

**Order matters:** custom-tools before audrey-ai — otherwise in-pipeline KB search 401s during the gap. No `compose.yaml` change for the secret (both services already `env_file: - .env`).

### Step 4 — smoke commands (box)

```bash
# deploy-state check (nonzero = new code is live):
docker exec audrey-ai grep -c resolve_kb_caller /app/src/audrey/routes/kb.py

# internal path WITH token → 200:
docker exec custom-tools python -c '
import urllib.request, json, os
tok = os.environ["KB_SERVICE_TOKEN"]
req = urllib.request.Request("http://audrey-ai:8000/v1/kb/query",
    data=json.dumps({"query":"test","user":"alice@example.com"}).encode(),
    headers={"content-type":"application/json","X-Audrey-Service-Token":tok}, method="POST")
print("with-token:", urllib.request.urlopen(req, timeout=30).status)'

# no token → 401:
docker exec custom-tools python -c '
import urllib.request, urllib.error, json
req = urllib.request.Request("http://audrey-ai:8000/v1/kb/query",
    data=json.dumps({"query":"test"}).encode(),
    headers={"content-type":"application/json"}, method="POST")
try: urllib.request.urlopen(req, timeout=30); print("FAIL: expected 401")
except urllib.error.HTTPError as e: print("no-token:", e.code)'
```

### Step 5 — the publish

`compose.override.yaml` (or fold into `compose.yaml`):
```yaml
services:
  audrey-ai:
    ports: ["8000:8000"]
```
Then from the LAN client: `/v1/chat/completions` works; `/v1/kb/query` without the secret → 401.

### Rollback

Revert both commits, remove the `ports` publish (back to unpublished + stopgap). No data migration; the `.env` secret can stay. Misconfigured/blank secret → in-pipeline KB search 401s (chat degrades to web/prior-knowledge, no crash) — caught by Step 4.

---

## Reference

### Why

`/v1/kb/query` ([kb.py:118](../../src/audrey/routes/kb.py#L118)) and `/v1/kb/query/image`
([kb.py:197](../../src/audrey/routes/kb.py#L197)) had no auth and passed a
caller-supplied `user` into the private-collection merge, so publishing 8000 to the
LAN let any device read another user's uploads by naming their email. That is why
the 2026-07-18 review unpublished the port — which also cut off a legitimate LAN
client (a Hermes gateway using `audrey_passthrough/*`). This gates the routes so the
port can be published again.

### Design — service token + act-as (dual-mode, settled 2026-07-27)

Shared secret `KB_SERVICE_TOKEN` between audrey-ai and custom-tools (both on
ollama-net). `resolve_kb_caller` resolves the effective user:
1. valid `X-Audrey-Service-Token` → trusted internal caller; honor `req.user` (act-as). ← custom-tools
2. else valid user bearer (`require_user`) → force `user = me.email`, ignore `req.user`. ← direct/LAN user
3. else 401.

`hmac.compare_digest` on the check; a blank configured secret never authenticates
(fail closed). Rejected the service-token-only variant.

### What made a naïve fix break the pipeline

`require_user` validates an OWUI bearer; the only caller of `/v1/kb/query` is
custom-tools ([app.py:457](../../tools-server/app.py#L457)), which holds no user JWT
— only the email `dispatch.py` already forced ([dispatch.py:126-134](../../src/audrey/tools/dispatch.py#L126)).
So plain `require_user` on the routes would 401 every in-pipeline KB search. The
service-token arm is what keeps that path working.

### Implemented

- `audrey/auth.py`: `verify_service_token`, `resolve_kb_caller`, `KBCaller`.
- `audrey/config.py` `EnvOverrides` + `tools-server/settings.py`: `kb_service_token`.
- `audrey/routes/kb.py`: both routes gated; `effective_user` replaces `req.user`.
- `tools-server/app.py`: `_service_headers` puts `X-Audrey-Service-Token` on the audrey client.

### Test coverage (hermetic, +17)

`tests/test_kb_query_auth.py`: `verify_service_token` (match / mismatch /
blank-fail-closed); `resolve_kb_caller` (service / user-bearer / no-creds-401 /
bad-token-falls-through-401); route matrix — service token acts-as body `user`, no
creds 401 (never searches), and the key assertion **user bearer is pinned to its own
email, body `user` ignored** (text + image). `tests/test_tools_service_token.py`:
`_service_headers` present when set / absent when blank.

### Scope

Fixes `audrey-ai:8000` only. `custom-tools:8001` stays unpublished (no LAN device
needs it) — out of scope.

### Open decisions

1. ~~Branch 2 dual-mode vs service-token-only~~ — **SETTLED: dual-mode.**
2. `X-Audrey-Service-Token` header chosen over reusing `Authorization: Bearer` so the
   service check doesn't collide with `require_user`'s bearer parsing.
3. Folding `custom-tools:8001` hardening into a follow-up now the service-token
   pattern exists — deferred (8001 is never published).
