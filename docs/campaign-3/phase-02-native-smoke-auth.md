# Phase 2E–2F: native smoke credentials

The runner, endpoint, and credential decision table is maintained in
`../reference/live-smoke-testing.md`. Read it before constructing a live smoke
command; in particular, API-only smokes normally run from the laptop over VPN,
while the full standalone-proxy smoke runs on Tower or through a tunnel.

Eight native smoke scripts authenticate only with Cloudflare Access assertions.
Supply two **different approved accounts**: a disposable ordinary
user and an Audrey administrator. Each credential must be the short-lived
Cloudflare Access **application** JWT for Audrey's hostname, not a service
token or the team-domain global-session token. Audrey verifies the JWT
signature, audience, issuer, expiry, and account binding on every request.

Set `AUDREY_SMOKE_USER_ACCESS_JWT` and `AUDREY_SMOKE_ADMIN_ACCESS_JWT` in the
private, root-owned `.env.smoke.local` file (mode `600`). Do not commit, print,
or paste either token into a ticket or chat. Legacy OWUI variables are ignored.
Refresh expired Access assertions before a long all-mode smoke. Identical user/
admin credentials fail before any live write.

The file must use Docker-compatible `KEY=value` lines, without `export` or
spaces around `=`. It should contain only the two `AUDREY_SMOKE_*_ACCESS_JWT`
entries; remove obsolete `OWUI_API_KEY`, `TEST_OWUI_TOKEN`, and
`ADMIN_OWUI_TOKEN` lines. This same strict format can be sourced by Bash for a
laptop smoke.

Tower is Docker-only and has no host Python, `uv`, or repository `.venv`. Run
the scripts in a disposable container from the already-built `audrey:latest`
image. Mount the checkout's `scripts/` directory read-only at `/smoke`, pass the
root-owned credential file with Docker's `--env-file`, join `ollama-net`, and
set `AUDREY_SMOKE_BASE_URL=http://audrey-ui:8080`. The scripts are intentionally
not baked into the Audrey image. This reaches the standalone proxy through
Docker DNS. Do not target the public Cloudflare hostname for assertion replay:
the edge can replace the origin assertion header. The proxy explicitly forwards
`Cf-Access-Jwt-Assertion` to Audrey.

The checked Tower command is:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash scripts/smoke-native-onbox.sh smoke_native_ui.py
```

Start with the native UI smoke, then the account/model smoke. The remaining
file, preferences, mode, chat-projection, and tool-event scripts extend the
parity gate; several create disposable runs or temporarily change test-user
state and perform cleanup, so use only a disposable account. Success means a
zero exit, JSON evidence, and no `cleanup_error`. Browser sign-in and the
Cloudflare handoff remain a separate manual gate; replaying a valid JWT to the
origin does not prove the browser login path.

The eighth script, `scripts/smoke_native_auth_cutover.py`, is the focused 2F.1
gate. It needs only the user
Access assertion, creates and revokes one disposable Audrey personal token, and
proves unknown legacy bearers are rejected locally. Run it after setting
`OWUI_AUTH_ENABLED=0` and recreating Audrey, then again after Open WebUI is
stopped.

The historical chat-export importer is optional and is not part of this gate.
