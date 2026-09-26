# Campaign 3 Phase 2 — standalone Audrey UI deployment

This is the live gate for moving the native browser client out of the Audrey
backend process. It changes only the web traffic path. Audrey continues to own
identity, authorization, conversations, runs, files, and preferences.

## Transition boundary

The transition is complete. `audrey-ui` is the sole browser surface and the
Audrey backend serves APIs only; its embedded shell, feature flag, packaged
assets, and Node build stage were removed in 2F.4. Open WebUI remains stopped
and is not a rollback target.

The host-network `cloudflared` instance reaches the new UI through
`http://127.0.0.1:8090`; host port 8088 remains assigned to SearXNG. The UI
reaches Audrey through the explicit
`AUDREY_UI_UPSTREAM` setting, which defaults to `http://audrey:8000` on
external network `ollama-net`. No Access JWT or API key belongs in the UI
container environment.

## Settled Phase 2E regression record

The standalone UI and public Cloudflare route are live. On 2026-09-24 the user
marked the combined browser regression and soak tested and settled. The 2F.3
native-independence proof passed with Open WebUI stopped, and 2F.4 removed the
backend browser fallback. The checklist remains below only as a concrete
regression reference.

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

The backend is reachable as `audrey` on its Docker networks. The temporary
`audrey-ai` alias is gone; monitoring and internal clients must use
`http://audrey:8000`.

Run the existing full native-client smoke through the standalone proxy using
the already-built Audrey image. Tower has no host Python environment:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash scripts/smoke-native-onbox.sh smoke_native_ui.py
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
bash scripts/smoke-native-onbox.sh smoke_native_access_models.py
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
cd /mnt/user/appdata/audrey_ai_2.0
AUDREY_DIRECT_SMOKE_MODEL_ID=direct/example-model:latest \
bash scripts/smoke-native-onbox.sh smoke_native_access_models.py
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

For any future regression, record the conversation id, run id when available,
and whether the behavior changed after refresh. The user closed this gate as
tested and settled on 2026-09-24; later failures reopen as new regressions.

## Make the native frontend authoritative

Slice 2F.2 completes the operational cutover started by 2F.1, removing Open
WebUI from Audrey's runtime authentication path. First confirm
`.env.smoke.local` contains a fresh
`AUDREY_SMOKE_USER_ACCESS_JWT` plus a distinct
`AUDREY_SMOKE_ADMIN_ACCESS_JWT` for the companion native UI smoke. The focused
cutover smoke itself uses the user assertion and refuses legacy OWUI credentials
by design. The authoritative runner and credential matrix is
`../reference/live-smoke-testing.md`: the focused API-only smoke normally runs
from the laptop over VPN, while the full standalone-proxy smoke runs on Tower
or through an explicit tunnel.

In the Audrey deployment's `.env`, set:

```text
OWUI_AUTH_ENABLED=0
```

Before stopping OWUI, migrate the laptop `.env.test.local` and Tower
`eval.env` to a direct Audrey `/v1` URL plus an Audrey PAT with
`compat:full`, following the live-smoke guide.

Recreate Audrey and read the effective startup state:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build --force-recreate audrey
docker compose logs --since=5m audrey
```

The authentication-cutover smoke, full standalone-proxy smoke, corrected
`fast-capital` eval, and targeted native Deep independence proof have all
passed. Open WebUI remained stopped for the Deep proof. These results are
settled; do not rerun them for 2F.4.

The 2F.4 live proof is intentionally limited to the affected boundary after
rebuilding `audrey`:

1. Confirm `audrey` and `audrey-ui` are healthy.
2. Refresh the public Audrey URL and confirm the authenticated account label
   and existing conversation history load.

That refresh covers the standalone proxy-to-API path and proves that removing
the backend shell did not disturb canonical application state. It does not
require a prompt, an eval, or a broad smoke suite. Leave `OWUI_AUTH_ENABLED=0`
and Open WebUI stopped. The dormant adapter is diagnostic-only and is not the
normal rollback path.

## Rollback

The backend no longer serves a browser shell, so port 8000 is API-only and must
not become the public browser origin. If the UI container, proxy, upload, or
stream path fails, redeploy a known-good `audrey-ui` image or repair the
standalone proxy while leaving Cloudflare Access on the hostname. Audrey owns
the canonical data, so a UI rollback does not require a database rollback or
identity rewrite.

If the Audrey backend itself fails, redeploy its known-good image through the
current Compose service. The retired pre-cutover container rename ceremony is
no longer part of normal recovery, and Open WebUI remains stopped.

## Optional UI repository extraction

The embedded backend fallback was removed in 2F.4. Moving `web/` to a separate
repository is an optional packaging decision, not a Phase 2 product-cutover
gate. If that split becomes useful:

1. Move `web/` intact into its own GitHub repository.
2. Give that repository its own release/image workflow and deployment Compose
   file while retaining external `ollama-net` and the same proxy contract.
3. Point Audrey's Compose deployment at the released UI image rather than a
   local build context.

The current UI protocol is Audrey-specific. Reuse by another project should be
implemented behind a typed platform adapter for identity, conversations, runs,
events, files, modes, and preferences; it must not make browser state
authoritative merely to appear generic.
