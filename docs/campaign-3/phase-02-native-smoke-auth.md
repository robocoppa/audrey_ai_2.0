# Phase 2E–2F: native smoke credentials

Eight native smoke scripts can now authenticate without
Open WebUI. Supply two **different approved accounts**: a disposable ordinary
user and an Audrey administrator. Each credential must be the short-lived
Cloudflare Access **application** JWT for Audrey's hostname, not a service
token or the team-domain global-session token. Audrey verifies the JWT
signature, audience, issuer, expiry, and account binding on every request.

Set `AUDREY_SMOKE_USER_ACCESS_JWT` and `AUDREY_SMOKE_ADMIN_ACCESS_JWT` in the
private, root-owned `.env.smoke.local` file (mode `600`). Do not commit, print,
or paste either token into a ticket or chat. The scripts prefer these values
over `TEST_OWUI_TOKEN` and `ADMIN_OWUI_TOKEN`, which remain temporary fallbacks.
Refresh expired Access assertions before a long all-mode smoke. Identical
user/admin credentials fail before any live write.

Run the scripts from the repository checkout on the **Unraid host**, with that
private environment loaded and `AUDREY_SMOKE_BASE_URL` set to the standalone
UI's loopback proxy (`http://127.0.0.1:8090`). The scripts are not copied into
the Audrey image. Do not target the public Cloudflare hostname for assertion
replay: the edge can replace the origin assertion header. The proxy explicitly
forwards `Cf-Access-Jwt-Assertion` to Audrey.

Start with the native UI smoke, then the account/model smoke. The remaining
file, preferences, mode, chat-projection, and tool-event scripts extend the
parity gate; several create disposable runs or temporarily change test-user
state and perform cleanup, so use only a disposable account. Success means a
zero exit, JSON evidence, and no `cleanup_error`. Browser sign-in and the
Cloudflare handoff remain a separate manual gate; replaying a valid JWT to the
origin does not prove the browser login path.

The eighth script, `scripts/smoke_native_auth_cutover.py`, is the focused 2F.1
gate and deliberately refuses the legacy OWUI fallback. It needs the user
Access assertion, creates and revokes one disposable Audrey personal token, and
proves unknown legacy bearers are rejected locally. Run it after setting
`OWUI_AUTH_ENABLED=0` and recreating Audrey, including once while Open WebUI is
temporarily stopped.

The historical chat-export importer is optional and is not part of this gate.
