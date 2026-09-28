# Audrey live smoke testing

Read this before giving or running a live smoke command. Audrey development
happens in the laptop checkout; Tower is the Docker-only deployment host.

## Choose the runner first

| Test | Runner | Target | Credentials |
|---|---|---|---|
| Hermetic pytest, lint, or build | Laptop | Local checkout | None |
| API-only live smoke, including `smoke_native_auth_cutover.py` | Laptop over Tailscale; WARP fallback | Tower backend port `8000` | Credential required by that script |
| Eval harness | Laptop over Tailscale; WARP fallback | Tower backend `/v1` on port `8000` | Audrey PAT with `compat:full` |
| Full native UI/proxy smoke | Disposable container on Tower, or laptop through an explicit SSH tunnel | `http://audrey-ui:8080` inside `ollama-net`, or tunneled loopback port `8090` | Native smoke user/admin assertions |

Choose the smallest proof that exercises the changed functionality and its
direct dependency or failure boundary. Do not rerun broad live suites merely
because they exist or because an older phase checklist names them. Reuse
current evidence for unchanged surfaces. Escalate to a broader smoke only when
the targeted check fails, the change crosses additional boundaries, or the
user explicitly asks for broader coverage. Hermetic laptop verification is a
separate required gate for code changes; it does not substitute for live proof.

Current VPN addresses:

- Primary — Tailscale: `http://100.113.157.98:8000`
- Fallback — WARP: `http://192.168.1.11:8000`

Tower does not have host Python, `uv`, or a repository `.venv`. Never hand the
user any of those commands at a Tower prompt. Conversely, do not move an
API-only smoke to Tower merely because Tower runs Docker; the laptop `.venv`
and VPN route are the normal path.

The standalone UI bind `127.0.0.1:8090` is Tower-loopback-only. A laptop cannot
reach it at either private IP without an SSH tunnel. Do not confuse that limit
with the published backend port `8000`.

## Keep credentials private and runner-specific

The laptop's gitignored `.env.test.local` is the permanent local credential
file for evals and native API smokes. It contains the direct Audrey eval URL,
the first-party personal token, and the two Cloudflare Access application
assertions:

```text
AUDREY_EVAL_BASE_URL=http://100.113.157.98:8000/v1
AUDREY_EVAL_API_KEY=aud_pat_...
AUDREY_USER_JWT=eyJ...
AUDREY_ADMIN_JWT=eyJ...
```

Use `http://192.168.1.11:8000/v1` only when falling back to WARP. Create the
personal token in native Audrey Settings with `compat:full` scope. Native smoke
commands source this file; the eval harness reads only its `AUDREY_EVAL_*`
entries. Keep it mode `600`. The variable entries are permanent, but expired
Access JWT values still need to be refreshed in place.

The Docker-only Tower fallback keeps a separate, gitignored
`.env.smoke.local`, because Docker's `--env-file` would otherwise pass the
unrelated eval PAT into the disposable container. That root-owned file contains
only these same two native credential names:

```text
AUDREY_USER_JWT=eyJ...
AUDREY_ADMIN_JWT=eyJ...
```

Both files use exactly `KEY=value`: no `export`, no spaces around `=`, no
Markdown link syntax, and no obsolete `OWUI_API_KEY`, `TEST_OWUI_TOKEN`, or
`ADMIN_OWUI_TOKEN` entries. Before telling the user to source a file, confirm it
exists and inspect key names without printing values.

Every native smoke script ignores legacy OWUI credentials. It requires a
current Cloudflare Access **application** JWT for the user in `AUDREY_USER_JWT`.
Obtain it either with `cloudflared access token` for Audrey's public application,
or from the `CF_Authorization` cookie on the protected Audrey hostname in browser
developer tools. Do not use the separate team-domain global-session cookie.
Treat the value like a password: never print it, commit it, or paste it into chat.

The broader native UI smokes require both `AUDREY_USER_JWT` and a distinct
`AUDREY_ADMIN_JWT`. Tower's `.env.smoke.local` must be root-owned or otherwise
private and mode `600`.

## Run the 2F.1 authentication smoke from the laptop

Use Tailscale. This consumes the credential already stored in
`.env.test.local`; do not prompt for it again:

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://100.113.157.98:8000 .venv/bin/python scripts/smoke_native_auth_cutover.py
)
```

If Tailscale is unavailable, use the complete WARP fallback:

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python scripts/smoke_native_auth_cutover.py
)
```

The subshell keeps the token out of shell history and removes it from the
environment when the command exits. Success is exit code zero, JSON ending in
`"status": "passed"`, `"legacy_bearer_rejected_locally": true`, and no
`cleanup_error`.

## Run the current built-in video-skill smoke from the laptop

This is the targeted API-only deploy proof for 3A.2. It checks that the tracked
`video-analysis` bundle is the one available catalog entry, skill readiness is
healthy, and admin rediscovery reloads it without diagnostics. It creates no
user data, conversations, tokens, files, or model calls. Use Tailscale first:

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://100.113.157.98:8000 .venv/bin/python scripts/smoke_skills_foundation.py
)
```

Use `http://192.168.1.11:8000` only as the WARP fallback. Success is exit code
zero and JSON ending in `"status": "passed"`, with catalog and readiness
`ready`, capabilities `available`, one `video-analysis` item, and zero
rediscovery diagnostics. The earlier all-`disabled` output remains the
recorded live proof for 3A.1; do not expect that result after deploying 3A.2.

This path is for scripts that must exercise the standalone proxy. Use the
checked wrapper so malformed or legacy env-file entries fail with a useful
message before Docker starts:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
bash scripts/smoke-native-onbox.sh smoke_native_ui.py
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
9. Never ask for a credential interactively when the chosen procedure already
   stores it in a private env file.

If a command fails with missing `uv`, `.venv/bin/python`, or `python`, first
check whether it was mistakenly aimed at Tower. If it fails because an env file
is missing, inspect the actual checkout before proposing another filename.
