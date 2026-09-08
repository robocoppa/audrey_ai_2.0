# Campaign 3 Phase 2 — standalone Audrey UI deployment

This is the live gate for moving the native browser client out of the Audrey
backend process. It changes only the web traffic path. Audrey continues to own
identity, authorization, conversations, runs, files, and preferences.

## Transition boundary

Keep `AUDREY_NATIVE_UI_ENABLED=1` during this first deployment. The backend's
embedded shell is the rollback target until the standalone container has passed
the smoke, browser checks, and a normal-use soak. Do not remove Open WebUI or
change the Access application during this gate.

The host-network `cloudflared` instance currently reaches the new UI through
`http://127.0.0.1:8088`. The UI reaches Audrey through the explicit
`AUDREY_UI_UPSTREAM` setting, which defaults to `http://audrey-ai:8000` on
external network `ollama-net`. No Access JWT or API key belongs in the UI
container environment.

## Build and check the private origin

From the Unraid Audrey checkout:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey-ai audrey-ui
docker compose ps audrey-ai audrey-ui
curl --fail-with-body -sS http://127.0.0.1:8088/healthz
```

Both services should become healthy and the final command should print `ok`.
The UI port is bound to loopback, not the LAN.

Run the existing full native-client smoke through the standalone proxy:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
set -a
source .env.smoke.local
set +a
AUDREY_SMOKE_BASE_URL=http://127.0.0.1:8088 .venv/bin/python scripts/smoke_native_ui.py
```

The result must end with `"status": "passed"`, cross-owner reads must remain
`404`, and cleanup must return repair to `ready`. This one gate covers the
hashed SPA asset, security headers, identity, a real AG-UI run, canonical
history, conversation management, and same-origin API forwarding.

## Switch the public hostname

In the existing Cloudflare Tunnel published-application route for
`ai.builtryte.xyz`, change only the origin service:

```text
from: http://127.0.0.1:8000
to:   http://127.0.0.1:8088
```

Leave the hostname, Access application, policies, audience tag, and Audrey team
domain unchanged. Then use an allowed browser identity to verify:

1. `https://ai.builtryte.xyz/` opens at the clean root URL.
2. Existing conversation history survives navigation and hard refresh.
3. A new Fast message streams progress and its final answer.
4. A small file uploads, can be attached to a turn, and can be removed.
5. Profile and preference changes still persist.
6. A second allowed identity cannot see the first identity's conversation.

## Rollback

If the UI container, proxy, upload, or stream path misbehaves, change the tunnel
origin back to `http://127.0.0.1:8000`. No database rollback or data migration
is involved because the browser client never owns canonical state. Keep the
standalone container available for diagnosis and leave the Access application
protecting the hostname.

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
