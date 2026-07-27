# Campaign 2 Phase 31 — KB query-route authentication (safe LAN publish of audrey-ai:8000)

Gate Audrey's two unauthenticated KB query routes so `audrey-ai:8000` can be
**re-published to the LAN** without re-opening the private-KB exposure that got the
host port unpublished in the 2026-07-18 security review (Item 1). Motivating
consumer: a Hermes gateway on a separate LAN laptop that reaches Audrey via the
`audrey_passthrough/*` OpenAI route — a legitimate LAN client the Jul-18 review's
"legitimate consumers" list (OWUI, Prometheus) omitted, so unpublishing 8000 cut
it off. **Two-image change — deploy `--build audrey-ai` AND `--build custom-tools`.**

**Status: IMPLEMENTED in the working tree, NOT yet deployed (as of 2026-07-27).**
Commits A + B landed locally; **844 hermetic tests pass** (+17 new), ruff clean on
all touched files. Still to do on the box: set `KB_SERVICE_TOKEN` in `.env`, add it
to both services' compose `environment`, deploy custom-tools then audrey-ai, then
publish 8000 and retire the stopgap. This doc becomes the deploy record once it ships.

---

## Why this exists

`/v1/kb/query` ([kb.py:118](../../src/audrey/routes/kb.py#L118)) and
`/v1/kb/query/image` ([kb.py:197](../../src/audrey/routes/kb.py#L197)) have **no
auth dependency** and pass `user=req.user` — a value taken straight from the
request body — into the private-collection merge
([kb.py:168](../../src/audrey/routes/kb.py#L168) /
[kb.py:218](../../src/audrey/routes/kb.py#L218)). Any caller can POST
`{"query": "...", "user": "alice@example.com"}` and read Alice's private uploads.
Every *other* route on `:8000` already gates on a bearer (`require_user`) or admin
(`kb_ingest` → `require_admin`, [kb.py:235](../../src/audrey/routes/kb.py#L235)).

**Threat model:** any device on the LAN, once 8000 is published. Win condition to
deny: "read another user's private KB by naming their email." The Jul-18 review
denied it by *not publishing the port*; this phase denies it *at the route* so the
port can be published for the LAN clients that need it.

### What makes a naïve fix break the pipeline

- The existing "auth-forcing override" is [dispatch.py:126-134](../../src/audrey/tools/dispatch.py#L126):
  for user-scoped tools it overwrites `args["user"] = user_id`. That stops the
  **model** from naming another user, but it is a *pipeline-layer* guard — it does
  nothing for a direct HTTP caller to `/v1/kb/query`.
- The only legitimate caller of the route is **custom-tools**: `kb_search`
  ([app.py:451-457](../../tools-server/app.py#L451)) and `kb_image_search`
  ([app.py:482-499](../../tools-server/app.py#L482)) POST to `/v1/kb/query[/image]`
  via `app.state.audrey` — an `httpx.AsyncClient` with `base_url=http://audrey-ai:8000`
  ([app.py:76](../../tools-server/app.py#L76), `AUDREY_URL`
  [settings.py:50](../../tools-server/settings.py#L50)) and **no auth header**. It
  sends `user=<pipeline user_id>` that dispatch already forced.
- `require_user` ([auth.py:126](../../src/audrey/auth.py#L126)) validates an **OWUI**
  bearer. custom-tools holds no user JWT — only the email string. **So simply adding
  `require_user` to the KB routes 401s the internal pipeline path**, killing KB
  search for every chat turn. This is why the review chose the network mitigation.

---

## Design — shared service token + act-as (dual-mode)

Introduce `KB_SERVICE_TOKEN`, a secret shared only between audrey-ai and custom-tools
(both on `ollama-net`). A new dependency on the two KB routes resolves the effective
user in priority order:

1. Header `X-Audrey-Service-Token` matches `KB_SERVICE_TOKEN` (constant-time compare)
   → **trusted internal caller**; honor `req.user` (act-as — today's behavior). ← custom-tools
2. else a valid user bearer via `require_user` → **force `user = me.email`, ignore
   `req.user`** (a real user can query only their *own* KB). ← direct/LAN user
3. else **401**.

A LAN attacker has neither the service secret nor the victim's OWUI bearer, so the
private-KB read is denied while the port is published. The service secret never
leaves `ollama-net`.

**Constant-time compare** (`hmac.compare_digest`) on the token check — the routes are
LAN-reachable post-publish, so avoid a timing oracle on the secret.

> **DECISION (settled 2026-07-27): dual-mode.** Branch 2 is kept — a direct user
> bearer authenticates and is pinned to its own `me.email`. Rejected the
> service-token-only variant; the extra arm is a few lines and future-proofs a
> direct "my own KB" client. This is the implemented behavior.

---

## Implementation

### Prereq — the secret (`.env` + compose)

Both services already mount `.env` (`env_file: .env`). Generate once and reference
it from both:

```bash
# on the box, in the repo dir:
echo "KB_SERVICE_TOKEN=$(openssl rand -hex 32)" >> /mnt/user/appdata/audrey_ai_2.0/.env
```

`compose.yaml` — pass it to **both** services (env only; no new host ports):

```yaml
  audrey-ai:
    environment:
      KB_SERVICE_TOKEN: ${KB_SERVICE_TOKEN:?set in .env}
  custom-tools:
    environment:
      KB_SERVICE_TOKEN: ${KB_SERVICE_TOKEN:?set in .env}
```

`:?` makes a missing secret a hard boot failure rather than a silent open door.

### Commit A — custom-tools presents the service token  `feat(tools): send KB service token to Audrey`  ✅ in working tree

Deploy-safe to land first: audrey-ai still ignores the unknown header, so behavior is
unchanged until Commit B enforces it.

- `tools-server/settings.py`: add `kb_service_token: str = Field(default="", alias="KB_SERVICE_TOKEN")`.
- `tools-server/app.py:76`: give `app.state.audrey` a default header when the token is
  set — `httpx.AsyncClient(base_url=..., headers={"X-Audrey-Service-Token": settings.kb_service_token})`.
  (Empty token → omit the header, so local/dev without the secret degrades to the
  Commit-B `require_user` arm rather than sending an empty secret.)
- Tests: `kb_search` / `kb_image_search` outbound request carries the header
  (assert via `MockTransport`).

### Commit B — audrey-ai verifies the token and gates the routes  `feat(kb): authenticate KB query routes`  ✅ in working tree

- Env settings (where `owui_url` lives, `cfg.env`): add `kb_service_token`.
- `src/audrey/auth.py`: add `verify_service_token(token, expected) -> bool`
  (`hmac.compare_digest`, false on empty expected) and a dependency
  `resolve_kb_user(request, x_audrey_service_token: Header, ...) -> str` implementing
  the 3-way resolution. Returns the **effective user** string (or raises 401). Reuses
  `require_user` internally for branch 2.
- `src/audrey/routes/kb.py`: `kb_query` / `kb_query_image` take
  `effective_user: str = Depends(resolve_kb_user)` and pass `user=effective_user`
  (not `req.user`) into `_search_text_merged` / `_search_images_merged`. `req.user`
  is now advisory only on the service path and ignored on the user path.
- **Fail-closed:** if `kb_service_token` is empty while KB is enabled, log a loud
  warning at startup (the `:?` in compose already hard-fails the box); the service
  arm is disabled when the expected secret is empty, so a blank secret can never
  authenticate.
- Tests: see matrix below.

### Deploy step (post-code) — publish 8000, retire the stopgap

Only after A+B are live and smoked:

- Add the host publish for audrey-ai (`ports: ["8000:8000"]`) — the KB routes are now
  gated, so LAN exposure is safe.
- Retire any Phase-30.5 stopgap (the `compose.override.yaml` proxy / IP-allowlist put
  in place to unblock the LAN client before this landed).
- Update the `# No host ports publish (security review Item 1…)` comment on audrey-ai
  to record that the KB routes are gated as of Phase 31 and 8000 is intentionally
  published; add the LAN Hermes client to the "legitimate consumers" list.

**custom-tools:8001 stays unpublished** — every route there is unauthenticated and no
LAN device needs it (Audrey reaches it over `ollama-net`). Out of scope for this phase.

---

## Rollout ordering (zero-downtime)

1. Add `KB_SERVICE_TOKEN` to `.env` and to both services' compose `environment`.
2. `docker compose up -d --build custom-tools` (Commit A) — now sends the header;
   audrey ignores it, KB search unchanged.
3. `docker compose up -d --build audrey-ai` (Commit B) — now enforces; custom-tools
   already sends the token, so the pipeline KB path never sees a 401 window.
4. Add the `ports:` publish and `up -d --force-recreate audrey-ai`; retire the stopgap.

Doing Commit B before A (or publishing before both) would 401 in-pipeline KB search —
follow the order.

---

## Smoke testing

**Hermetic (`pytest tests/`), the auth matrix:**

| Call | Expect |
|------|--------|
| `/v1/kb/query` no creds | 401 |
| `/v1/kb/query` wrong `X-Audrey-Service-Token` | 401 |
| valid service token + `user=alice@example.com` | 200, Alice's collection merged (act-as) |
| valid user bearer (mock OWUI) + `user=alice@example.com` in body | 200, **caller's own** collection; body `user` ignored |
| image variant of each | same |
| `verify_service_token("", "")` / empty expected | False (blank secret never authenticates) |

Plus: custom-tools attaches the header (Commit A); a **pipeline integration** test —
a chat turn that emits `kb_search` still returns hits end-to-end; regression that
`dispatch._force_user_tag`/`args["user"]` override is intact (defense in depth).

**Functional (laptop):** `TestClient(app.app)` (no lifespan) — POST the matrix rows,
assert status codes; no OWUI/qdrant needed for the 401 rows (they fail before any
search). Branch-2 rows need a stubbed `require_user`.

**On-box (post-deploy)** — custom-tools has no curl; use `docker exec … python`.
Deploy-state check first (`docker exec audrey-ai grep -c X-Audrey-Service-Token
/app/src/audrey/routes/kb.py` → nonzero):

```bash
# from INSIDE ollama-net (custom-tools), the internal path with the token:
docker exec custom-tools python -c '
import urllib.request, json, os
tok = os.environ["KB_SERVICE_TOKEN"]
req = urllib.request.Request("http://audrey-ai:8000/v1/kb/query",
    data=json.dumps({"query":"test","user":"alice@example.com"}).encode(),
    headers={"content-type":"application/json","X-Audrey-Service-Token":tok}, method="POST")
print("service+user:", urllib.request.urlopen(req, timeout=30).status)'

# no token → 401:
docker exec custom-tools python -c '
import urllib.request, urllib.error, json
req = urllib.request.Request("http://audrey-ai:8000/v1/kb/query",
    data=json.dumps({"query":"test","user":"alice@example.com"}).encode(),
    headers={"content-type":"application/json"}, method="POST")
try: urllib.request.urlopen(req, timeout=30); print("FAIL: expected 401")
except urllib.error.HTTPError as e: print("no-token:", e.code)'
```

Then: send a real chat turn in OWUI that triggers `kb_search`, confirm KB hits still
appear (pipeline path). Finally publish 8000 and, from the LAN client, confirm
`/v1/chat/completions` (passthrough) works while `/v1/kb/query` without the secret
returns 401.

---

## Rollback

Revert Commits A+B, remove the `ports:` publish (back to unpublished + the stopgap
proxy/allowlist). No data migration; the secret can stay in `.env` harmlessly. A
misconfigured/blank secret degrades to: internal KB search 401s → chat turns lose KB
grounding (research falls back to web/prior knowledge; domain-KB answers suffer, but
no crash). The compose `:?` guard and the post-deploy chat smoke catch this before it
reaches users.

---

## Open decisions

1. ~~**Branch 2** (direct user-bearer → own KB): dual-mode vs service-token-only.~~
   **SETTLED 2026-07-27: dual-mode.** Implemented.
2. Header name `X-Audrey-Service-Token` vs reusing `Authorization: Bearer <service>`
   — separate header chosen so the service check can run without colliding with
   `require_user`'s bearer parsing; flag if you'd rather one auth header.
3. Whether to fold `custom-tools:8001` hardening (all routes unauth, caller-supplied
   `user` on memory/chat_history) into a follow-up now that the service-token pattern
   exists — currently out of scope because 8001 is never published.
