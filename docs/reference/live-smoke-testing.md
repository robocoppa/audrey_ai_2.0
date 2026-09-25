# Audrey live smoke testing

Read this before giving or running a live smoke command. Audrey development
happens in the laptop checkout; Tower is the Docker-only deployment host.

## Choose the runner first

| Test | Runner | Target | Credentials |
|---|---|---|---|
| Hermetic pytest, lint, or build | Laptop | Local checkout | None |
| API-only live smoke, including `smoke_native_auth_cutover.py` | Laptop over WARP or Tailscale | Tower backend port `8000` | Credential required by that script |
| Eval harness | Laptop over WARP or Tailscale | Tower backend `/v1` on port `8000` | Audrey PAT with `compat:full` |
| Full native UI/proxy smoke | Disposable container on Tower, or laptop through an explicit SSH tunnel | `http://audrey-ui:8080` inside `ollama-net`, or tunneled loopback port `8090` | Native smoke user/admin assertions |

Current VPN addresses:

- WARP: `http://192.168.1.11:8000`
- Tailscale: `http://100.113.157.98:8000`

Tower does not have host Python, `uv`, or a repository `.venv`. Never hand the
user any of those commands at a Tower prompt. Conversely, do not move an
API-only smoke to Tower merely because Tower runs Docker; the laptop `.venv`
and VPN route are the normal path.

The standalone UI bind `127.0.0.1:8090` is Tower-loopback-only. A laptop cannot
reach it at either private IP without an SSH tunnel. Do not confuse that limit
with the published backend port `8000`.

## Keep the credential sets separate

The laptop's gitignored `.env.test.local` belongs to the eval harness. It
contains a direct Audrey URL and a first-party personal token:

```text
AUDREY_EVAL_BASE_URL=http://192.168.1.11:8000/v1
AUDREY_EVAL_API_KEY=aud_pat_...
```

Use `http://100.113.157.98:8000/v1` on Tailscale. Create the token in native
Audrey Settings with `compat:full` scope. This file is not a native
Cloudflare smoke environment and must never contain an Access assertion.

Do not assume `.env.smoke.local` exists on the laptop. Before telling the user
to source any file, confirm it exists and inspect key names without printing
values.

Every native smoke script ignores legacy OWUI credentials.
It requires a current Cloudflare Access **application** JWT for the user in
`AUDREY_SMOKE_USER_ACCESS_JWT`. Obtain it either with `cloudflared access token`
for Audrey's public application, or from the `CF_Authorization` cookie on the
protected Audrey hostname in browser developer tools. Do not use the separate
team-domain global-session cookie. Treat the value like a password: never print
it, commit it, or paste it into chat.

The broader native UI smokes require both
`AUDREY_SMOKE_USER_ACCESS_JWT` and a distinct
`AUDREY_SMOKE_ADMIN_ACCESS_JWT`. A Tower `.env.smoke.local`, when used, must be
root-owned or otherwise private and mode `600`.

## Run the 2F.1 authentication smoke from the laptop

Use WARP:

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  read -r -s -p 'Paste Audrey CF_Authorization value: ' AUDREY_SMOKE_USER_ACCESS_JWT
  printf '\n'
  export AUDREY_SMOKE_USER_ACCESS_JWT
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python scripts/smoke_native_auth_cutover.py
)
```

For Tailscale, use the same block with:

```bash
AUDREY_SMOKE_BASE_URL=http://100.113.157.98:8000 .venv/bin/python scripts/smoke_native_auth_cutover.py
```

The subshell keeps the token out of shell history and removes it from the
environment when the command exits. Success is exit code zero, JSON ending in
`"status": "passed"`, `"legacy_bearer_rejected_locally": true`, and no
`cleanup_error`.

## Run a full native UI smoke on Tower

This path is for scripts that must exercise the standalone proxy. Confirm the
private env file exists first; do not invent it in the command.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker run --rm \
  --network ollama-net \
  --env-file /mnt/user/appdata/audrey_ai_2.0/.env.smoke.local \
  --env AUDREY_SMOKE_BASE_URL=http://audrey-ui:8080 \
  --volume /mnt/user/appdata/audrey_ai_2.0/scripts:/smoke:ro \
  audrey:latest \
  /opt/venv/bin/python /smoke/smoke_native_ui.py
```

Success is exit code zero, `"status": "passed"`, and no `cleanup_error`.

## Before handing over any smoke command

1. Name the machine: laptop or Tower.
2. Classify the test: backend API, eval, or standalone UI/proxy.
3. Confirm the target is reachable from that machine.
4. Confirm the credential source exists and has the keys the script reads.
5. Read the script's environment-variable names instead of recalling them.
6. Keep URLs in code blocks as plain URLs, never Markdown link syntax.
7. State the expected success evidence and whether the script mutates live data.
8. Never claim an Unraid smoke passed until the user provides its output.

If a command fails with missing `uv`, `.venv/bin/python`, or `python`, first
check whether it was mistakenly aimed at Tower. If it fails because an env file
is missing, inspect the actual checkout before proposing another filename.
