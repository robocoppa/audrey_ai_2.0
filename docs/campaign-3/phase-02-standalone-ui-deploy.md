# Campaign 3 Phase 2 — standalone Audrey UI deployment

This is the live gate for moving the native browser client out of the Audrey
backend process. It changes only the web traffic path. Audrey continues to own
identity, authorization, conversations, runs, files, and preferences.

## Transition boundary

Keep `AUDREY_NATIVE_UI_ENABLED=1` during this first deployment. The backend's
embedded shell is the rollback target until the standalone container has passed
the smoke, browser checks, and a normal-use soak. Do not remove Open WebUI or
change the Access application during this gate.

The host-network `cloudflared` instance reaches the new UI through
`http://127.0.0.1:8090`; host port 8088 remains assigned to SearXNG. The UI
reaches Audrey through the explicit
`AUDREY_UI_UPSTREAM` setting, which defaults to `http://audrey:8000` on
external network `ollama-net`. No Access JWT or API key belongs in the UI
container environment.

## Deferred Phase 2E regression gate

The standalone UI and public Cloudflare route are already live. The combined
browser regression below remains owed. The user explicitly authorized Slice
2F.1 before completing it, but the gate must still close before broader 2F
dependency removal. Keep the embedded backend shell, Open WebUI, temporary
network alias, and rollback container until that normal-use soak closes.

## Rename the backend and build the private origin

From the Unraid Audrey checkout:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose build audrey audrey-ui
docker stop audrey-ai
docker rename audrey-ai audrey-ai-retired
docker compose up -d audrey audrey-ui
docker compose ps audrey audrey-ui
curl --fail-with-body -sS http://127.0.0.1:8090/healthz
```

Both services should become healthy and the final command should print `ok`.
The UI port is bound to loopback, not the LAN. Keep the stopped
`audrey-ai-retired` container through the initial soak; Compose may identify
it as an orphan, which is expected. Do not use `--remove-orphans` during this
migration.

The new `audrey` service retains `audrey-ai` as a temporary network alias.
Existing OWUI and monitoring configuration can therefore continue to resolve
the former hostname while their settings move to `http://audrey:8000`.

Run the existing full native-client smoke through the standalone proxy:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
set -a
source .env.smoke.local
set +a
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_ui.py
```

The result must end with `"status": "passed"`, cross-owner reads must remain
`404`, and cleanup must return repair to `ready`. This one gate covers the
hashed SPA asset, security headers, identity, a real AG-UI run, canonical
history, conversation management, and same-origin API forwarding.

## Run the access and direct-model gate

After deploying the schema-v7/v8 build, run the focused 2D.5 smoke through the
same standalone proxy:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
set -a
source .env.smoke.local
set +a
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_access_models.py
```

The result must end with `"status": "passed"`. It proves provider-only admin
access, personal-token rejection at the admin boundary, self-protection,
ordinary/tester catalog filtering, stable direct-model selection, a real Qwen
AG-UI turn without tool events, disabled-model denial, canonical persistence,
and cleanup. The script restores the test user's original status/groups and the
model's original policy source and values even after a failed assertion.
Override the target only when the deployment intentionally uses a different
direct model:

```bash
AUDREY_DIRECT_SMOKE_MODEL_ID=direct/example-model:latest AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_access_models.py
```

Four checks remain interactive because bearer-token automation cannot reproduce
Cloudflare's signed browser assertion or the model picker's browser state:

1. Sign in with the first genuinely new allowed Access identity, confirm Audrey
   shows the account boundary, then bootstrap its exact email from inside the
   Audrey container:

   ```bash
   docker compose exec audrey audrey-admin grant-admin --email 'alice@example.com'
   ```

   Email matching is exact and case-insensitive. If more than one provider-bound
   account uses that email, the command refuses to choose; use the canonical id
   shown by `/api/me` instead:

   ```bash
   docker compose exec audrey audrey-admin grant-admin usr_replace_with_exact_id
   ```

   Reload and confirm that same account can open the Admin dialog. In Accounts,
   confirm User/Tester/Administrator role controls are present. In Models,
   confirm the status says `Live Ollama inventory`, every installed Ollama tag
   is listed, and direct entries default to Enabled + Admins only.

2. Sign in with a second new Access identity, confirm it remains pending, then
   approve it from the first account and assign the intended users/testers
   groups.
3. With a conversation currently selecting the direct model, disable that
   model in Admin and confirm the picker moves to the first available model.
   Use `Reset default` and confirm the Customized marker clears before leaving
   the gate.
4. Recreate Audrey after restoring groups and policies, then confirm both the
   restored access state and the selected conversation model survive restart.

The bootstrap command must report `"status": "ok"`, the expected email, and the
account's canonical user id. The browser must never expose a concrete model that
the current account cannot use.

## Switch the public hostname

In the existing Cloudflare Tunnel published-application route for
`ai.builtryte.xyz`, change only the origin service:

```text
from: http://127.0.0.1:8000
to:   http://127.0.0.1:8090
```

Leave the hostname, Access application, policies, audience tag, and Audrey team
domain unchanged. Then use an allowed browser identity to verify:

1. `https://ai.builtryte.xyz/` opens at the clean root URL.
2. Existing conversation history survives navigation and hard refresh.
3. A new Fast message streams progress and its final answer.
4. A small file uploads, can be attached to a turn, and can be removed.
5. Profile and preference changes still persist.
6. A second allowed identity cannot see the first identity's conversation.

The current Phase 2E browser regression also verifies:

1. An untouched blank conversation disappears when another new conversation
   opens.
2. The composer is centered before the first message and docked at the bottom
   afterward, including short conversations.
3. New image previews and document/video cards appear immediately and survive
   hard refresh.
4. Ordinary startup says `Loading`; the post-login Access callback says
   `Authenticating with Cloudflare`.
5. Hard refresh during an attached-file run does not cancel it; one centered
   recovery panel shows its larger orb, thinking text, and Stop run control, and
   the durable answer mounts when the run completes.
6. Explicit Stop settles promptly, exposes Retry, and Retry preserves the
   original validated attachments.
7. Research shows one source disclosure and one grouped tool disclosure without
   individual tool cards; both open into the chat instead of behind the sidebar.
8. Full-answer and fenced-code Copy controls work before and after refresh.
9. The attachment picker closes from its arrow, an outside click, and Escape
   without clearing already selected files.

Record each failure with the conversation id, run id when available, and whether
the behavior changed after refresh. Do not advance to Milestone 2F until this
regression and the normal-use soak pass.

The user explicitly deferred that gate on 2026-09-24 and authorized the first
bounded 2F slice. The regression remains owed; it was not reclassified as
passed.

## Cut over Open WebUI bearer authentication

Slice 2F.1 keeps a one-setting rollback while removing Open WebUI from Audrey's
runtime authentication path. First confirm `.env.smoke.local` contains a fresh
`AUDREY_SMOKE_USER_ACCESS_JWT` plus a distinct
`AUDREY_SMOKE_ADMIN_ACCESS_JWT` for the companion native UI smoke. The focused
cutover smoke itself uses the user assertion and refuses legacy OWUI credentials
by design.

In the Audrey deployment's `.env`, set:

```text
OWUI_AUTH_ENABLED=0
```

Recreate Audrey and read the effective startup state:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build --force-recreate audrey
docker compose logs --since=5m audrey
```

The logs must contain `auth: Open WebUI bearer adapter disabled`. Load the
private smoke environment, then run the cutover smoke through the standalone
proxy:

```bash
set -a
source .env.smoke.local
set +a
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_auth_cutover.py
```

Success ends with `"status": "passed"`, reports
`"legacy_bearer_rejected_locally": true`, and has no `cleanup_error`.

For the actual independence proof, stop Open WebUI temporarily and repeat the
focused smoke plus the existing native UI smoke:

```bash
docker stop open-webui
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_auth_cutover.py
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8090 .venv/bin/python scripts/smoke_native_ui.py
docker start open-webui
```

Run each command separately and always restart Open WebUI before investigating
a failed smoke. While it is stopped, also open the public Audrey URL, start one
native Fast turn, refresh it, open Files and Settings, and confirm Admin remains
available to the administrator. This proves browser authentication,
conversations, files, preferences, and administration do not fall through to
OWUI. The focused script separately proves a scoped Audrey token still reaches
the native account resource and protected `/v1/files`.

Starting the Open WebUI container does not make it a functional Audrey client
while the adapter remains disabled. Until evals and every other compatibility
client have moved to Audrey personal tokens, finish this bounded proof by
setting `OWUI_AUTH_ENABLED=1` in `.env`, recreating Audrey, and confirming the
startup log reports the adapter enabled. No data migration or identity rewrite
is involved. Once those consumers have migrated, the later permanent cutover
keeps the flag disabled and stops Open WebUI instead.

## Rollback

If the UI container, proxy, upload, or stream path misbehaves, change the tunnel
origin back to `http://127.0.0.1:8000`. No database rollback or data migration
is involved because the browser client never owns canonical state. Keep the
standalone container available for diagnosis and leave the Access application
protecting the hostname.

If the renamed backend itself fails before the gate, restore the retained
container without deleting either image:

```bash
docker compose stop audrey
docker rename audrey audrey-failed
docker rename audrey-ai-retired audrey-ai
docker start audrey-ai
```

The old Compose service remains outside the new graph, so do not run
`docker compose up` against it. Diagnose or remove `audrey-failed` only
after normal service has been restored.

## Extraction gate

After the standalone origin passes the smoke, browser checks, and a defined
normal-use soak:

1. Move `web/` intact into its own GitHub repository.
2. Give that repository its own release/image workflow and deployment Compose
   file while retaining external `ollama-net` and the same proxy contract.
3. Point Audrey's Compose deployment at the released UI image rather than a
   local build context.
4. In a later Audrey change, remove the backend Node build stage, embedded
   static router/flag, wheel artifact, and fallback assets.

The current UI protocol is Audrey-specific. Reuse by another project should be
implemented behind a typed platform adapter for identity, conversations, runs,
events, files, modes, and preferences; it must not make browser state
authoritative merely to appear generic.
