# Phase 27 — Run the live eval ON THE BOX (laptop-internet-independent)

One small, additive deploy: package the existing eval harness
(`scripts/eval_research.py`) into a standalone container that runs **on the box's
Docker network**, so a long protocol run no longer dies when the laptop's
internet drops. No harness code change, no `compose.yaml` change — a
`Dockerfile.eval` + a secret-safe `docker run` recipe.

## The headline (read first)

**Three consecutive `hedge_policy: true` protocol runs died to laptop-side
connectivity** (2026-07-01): one to a mid-run Ollama update, two to laptop
internet drops (`ReadTimeout` → `ConnectError: [Errno 101] Network is
unreachable` on the tail cases). The box was healthy every time; the *laptop*
lost its path to it. The two ungrounded controls (`ctrl-explain-recursion`,
`ctrl-birthday-toast`) — the deciding canaries for whether `true` over-hedges —
have **never** produced data across four attempts.

> **Fix: move the harness onto the box.** Run the same script inside a container
> on `ollama-net`; it hits internal service names at LAN speed, and the laptop's
> connection becomes irrelevant. Kick it off, disconnect freely, read the
> results off the box's disk later.

## What it does

The harness is already **network-config-driven** — `--base-url` / `--api-key`
(or `AUDREY_EVAL_BASE_URL` / `AUDREY_EVAL_API_KEY`), `--cases`, `--model`,
`--save-file`, `--only`. Nothing about it needs to change to retarget it; we
just run it from *inside* the box network instead of over the laptop's link.

### Locked decisions (with user, 2026-07-01)

- **Auth = A2 — talk to OWUI's API by its internal service name with the durable
  `sk-…` key.** NOT direct-to-Audrey-with-a-JWT. Rationale: Audrey has no static
  key; it validates every Bearer JWT by proxying to OWUI (`auth.py` `_probe_owui`
  → `OWUI_URL/api/v1/auths/`). So "direct to Audrey" *still* needs a JWT, and
  JWTs expire (memory `reference_owui_session_auth`) — re-minting per run is the
  hassle the `sk-…` path was chosen to avoid. A2 keeps the durable key while
  still running entirely on the box. Base-url becomes `http://open-webui:8080/api`
  — the internal equivalent of the laptop's `http://192.168.1.11:8080/api`.
- **Packaging = B — standalone `Dockerfile.eval` + `docker run` recipe**, NOT a
  `compose.yaml` service. Keeps compose scoped to `audrey-ai` + `custom-tools`
  (memory `project_compose_scope`).

### Verified from source (no rebuild needed to know these)

- **Only external dep is `httpx`** — everything else is stdlib; `load_dotenv` is
  hand-rolled (no `python-dotenv`). So `python:3.12-slim` + `pip install httpx`
  is the whole image.
- **Internal service names on `ollama-net`:** `audrey-ai:8000`, `open-webui:8080`
  (Audrey's default `OWUI_URL=http://open-webui:8080` confirms OWUI is
  addressable there). `ollama-net` is an external network.
- Cases carried by the image: `scripts/eval_prompts{,_protocol,_deep,_fast}.json`.

## Files

- **`Dockerfile.eval`** (NEW, repo root) — `python:3.12-slim`, `pip install
  httpx`, copies `scripts/eval_research.py` + the prompt JSONs, entrypoint the
  script. No secret baked in.
- **`docs/testing/README.md`** — add an "eval on the box" section (build +
  `docker run` recipe + read-from-`/out`). The resilient path for long runs; the
  laptop `.venv` path stays for quick `--only` checks.
- **No change** to `scripts/eval_research.py`, `compose.yaml`, or any app code.

## Deploy

Every command below runs **on the box** (`root@Tower`), not the laptop. Docker
lives on the box; the laptop has no `docker`.

### 1. Build the image

The build needs three files present on the box: `Dockerfile.eval`,
`scripts/eval_research.py`, and the `scripts/eval_prompts*.json` cases. They
arrive via `git pull` — but only after they're committed on the laptop AND
pulled onto the box. **The box's repo is separate from the laptop's; nothing
syncs automatically.**

**1a. Be in the box's repo root.** That's the directory holding `Dockerfile.eval`
and the `scripts/` folder — currently:
```bash
cd /mnt/user/appdata/audrey_ai_2.0
```
The trailing `.` in the build command is the *build context* — the root Docker is
allowed to `COPY` from. The Dockerfile's `COPY scripts/eval_research.py …` lines
are written relative to this root, so you must build from the root, not from
inside `scripts/`.

**1b. Bring the box's repo up to date.** The `Dockerfile.eval` is committed on
the laptop; the box won't have it until you pull:
```bash
git status          # first: does the box have local uncommitted edits?
```
- If `git status` is **clean**, just:
  ```bash
  git pull
  ```
- If it shows a **modified `config.yaml`** — that's almost certainly the live
  `hedge_policy: true` edit on the box (the repo default is `false`). A plain
  `git pull` will refuse with *"local changes would be overwritten."* To keep the
  box's `true` across the pull:
  ```bash
  git stash          # set the local edit aside
  git pull           # bring in Dockerfile.eval + latest
  git stash pop      # reapply the hedge_policy edit (resolve a conflict on
                     # line ~239 by hand if git flags one)
  ```

**1c. Confirm the file is actually there** before building — this catches the
"nothing to build" failure:
```bash
ls -l Dockerfile.eval          # want: a real ~2 KB file, not "No such file"
```
> If the build errors with `failed to read dockerfile … no such file or
> directory` and the log shows `transferring dockerfile: 2B`, the box simply
> doesn't have the file yet — the `git pull` in 1b hasn't landed it. Re-check
> 1a (right directory) and 1b (pull succeeded).

**1d. Build.** The eval image is a **build-only compose service** on the `eval`
profile (`audrey-eval` in `compose.yaml`), so build it through compose:
```bash
docker compose --profile eval build audrey-eval
```
- `--profile eval` — activates the profile the service is gated on. WITHOUT it,
  compose doesn't know about `audrey-eval` at all (that's the point — a bare
  `docker compose up -d --build` never touches it, so the eval image never runs
  as a phantom service or gets deleted by a service-oriented prune).
- It builds from `Dockerfile.eval` with context `.` (both set in the compose
  stanza) and tags the result `audrey-eval:latest` — the name step 4 runs.

The old standalone form still works if you prefer it
(`docker build -f Dockerfile.eval -t audrey-eval:latest .`) — the compose stanza
just wraps the same build so "rebuild everything" is one flow:
```bash
docker compose up -d --build \
  && docker compose --profile eval build audrey-eval \
  && docker image prune -f
```

Expect a short build (base pull + one `pip install httpx` + a few `COPY`s). It
bakes in no secret, so the resulting image is inert until you pass the key at run
time (step 4).

### 2. One-time: create the secret file (yours only)

The image carries no key (see Security), so the run reads two values from a small
env-file you create once on the box. Put it somewhere you own, outside any
world-readable share — the appdata path below works:

```bash
cat > /mnt/user/appdata/audrey_ai_2.0/eval.env <<'EOF'
AUDREY_EVAL_BASE_URL=http://open-webui:8080/api
AUDREY_EVAL_API_KEY=sk-REPLACE_WITH_YOUR_OWUI_KEY
EOF
chmod 600 /mnt/user/appdata/audrey_ai_2.0/eval.env
```
- `AUDREY_EVAL_BASE_URL` — the OWUI API, by its internal service name (Audrey
  validates the key against OWUI; step 3 confirms this name resolves). If the
  step-3 probe fails, swap in `http://192.168.1.11:8080/api` (the box's own LAN
  IP — still works from a container on the box).
- `AUDREY_EVAL_API_KEY` — the same `sk-…` key from your laptop's
  `.env.test.local` (mint one in OWUI → Settings → Account → API Keys).
- `chmod 600` — readable only by you. This file is the credential; treat it like
  a password and never commit it.

### 3. Verify OWUI is reachable on the network (once, before the first run)

The `AUDREY_EVAL_BASE_URL` in your env-file points at OWUI by its internal
service name. Confirm that name actually resolves on `ollama-net` before a real
run — a throwaway curl container on the same network:
```bash
docker run --rm --network ollama-net curlimages/curl:latest \
  -s -o /dev/null -w "%{http_code}\n" http://open-webui:8080/health
```
- **`200`** → the name resolves; your env-file base-url is correct. Proceed.
- **anything else** (connection refused / name not known) → OWUI's service name
  or network differs on this box (it runs under the Unraid UI, not this compose
  file). Either use OWUI's real service name in `AUDREY_EVAL_BASE_URL`, or use the
  always-works fallback `http://192.168.1.11:8080/api` (the box's own LAN IP —
  still laptop-independent, since the container runs on the box).

### 4. Run the eval (detached — survives laptop disconnect)

First make the output directory the run writes into, and lock it to you:
```bash
mkdir -p /mnt/user/appdata/audrey_ai_2.0/testing-out
chmod 700 /mnt/user/appdata/audrey_ai_2.0/testing-out
```
Then launch:
```bash
docker run -d --name audrey-eval --network ollama-net \
  --env-file /mnt/user/appdata/audrey_ai_2.0/eval.env \
  -v /mnt/user/appdata/audrey_ai_2.0/testing-out:/out \
  audrey-eval:latest \
    --model audrey_research \
    --cases /eval/eval_prompts_protocol.json \
    --save-file /out/2026-07-01-research-onbox-answers.md
```
Flag by flag:
- `-d` — detached. The run keeps going after you close the terminal or the
  laptop drops. This is the whole point of the phase.
- `--name audrey-eval` — a fixed name so the follow-up commands can refer to it.
- `--network ollama-net` — puts the container on the box's internal network, so
  `open-webui:8080` resolves. Laptop internet is irrelevant once this launches.
- `--env-file …/eval.env` — supplies `AUDREY_EVAL_BASE_URL` + `AUDREY_EVAL_API_KEY`
  from your `chmod 600` file (NOT `-e`, which would leak the key — see Security).
- `-v …/testing-out:/out` — bind-mounts the host dir to `/out` in the container,
  so the answers file survives after the container exits.
- Everything after the image name (`--model` / `--cases` / `--save-file`) is
  passed straight to the harness. `--cases` and `--save-file` use in-container
  paths: cases live at `/eval/…` (baked into the image), output goes to `/out/…`
  (the mount).

Watch it, wait for it, collect the result:
```bash
docker logs -f audrey-eval                 # live progress; Ctrl-C stops watching, NOT the run
docker wait audrey-eval                     # blocks until done; prints the exit code (0 = all passed)
ls -l /mnt/user/appdata/audrey_ai_2.0/testing-out  # the answers .md is here
docker rm audrey-eval                       # clean up the finished container
```
Copy the answers file back into the repo's `docs/testing/` (on the box, or scp to
the laptop) and write the paired report as usual.

### 5. (Optional) Notify on completion — reuse the fleet-watchdog Telegram bot

A detached run has no natural "it's done" signal. Rather than poll, chain a
Telegram push onto `docker wait`, reusing the **watcher bot** already set up for
fleet-watchdog (phase-14) — its `WATCHDOG_TOKEN` + `WATCHDOG_CHAT_ID` are exactly
the "message me on my phone" channel, so no new bot/setup is needed.

Those two secrets already live on the box in the fleet-watchdog hub's `.env`
(`/mnt/user/appdata/fleet-watchdog/.env`). **Source them from there — don't
paste, copy, or commit them.** One source of truth:

```bash
nohup sh -c '
  set -a; . /mnt/user/appdata/fleet-watchdog/.env; set +a
  docker wait audrey-eval; rc=$?
  curl -s "https://api.telegram.org/bot${WATCHDOG_TOKEN}/sendMessage" \
    -d chat_id="${WATCHDOG_CHAT_ID}" \
    -d "text=✅ Audrey research eval finished (exit $rc)"
' >/dev/null 2>&1 &
```
- `. /mnt/…/fleet-watchdog/.env` loads `WATCHDOG_TOKEN` + `WATCHDOG_CHAT_ID` from
  the file they already live in — nothing duplicated, nothing to commit. (Confirm
  the path: `ls /mnt/user/appdata/fleet-watchdog/.env`.)
- `docker wait` blocks until the container exits, then the `curl` fires the push.
- `nohup … &` detaches it, so you can close the terminal / disconnect the laptop
  and still get the ping when the box finishes the run.

(A full Google Workspace tool integration — Gmail/Docs/Sheets/Calendar for the
bots — is a separate, larger effort, deliberately NOT bundled here.)

## Security — "only I can use it"

The `sk-…` OWUI key **is** the credential — anyone holding it can call
OWUI/Audrey as you; without it every request 401s. So "only I can use it" reduces
to *guard the key* + *keep box access yours*. This container adds **no new
network exposure**: any container on `ollama-net` can already reach
`open-webui:8080`/`audrey-ai:8000` — the network is the box's existing internal
trust boundary, and the key is the real gate on top of it.

1. **Key never in the image.** A baked secret survives in image layers +
   `docker history` and leaks if the image is shared. Pass at runtime only. The
   image is inert without the key — fine in the box's local registry, but **do
   not push it to any remote registry.**
2. **Key by mounted `--env-file`, not `-e`.** `-e SECRET=…` leaks into
   `docker inspect`, host `ps`/proc env, and shell history. An `--env-file` at a
   path you own keeps it out of all three.
3. **Lock the secret file:** `chmod 600`, owned by your user, under your appdata
   (not a world-readable share). Same posture as the laptop's gitignored
   `.env.test.local`. **Never commit it.**
4. **Lock `/out`:** `chmod 700`, yours — answers aren't secret, but don't scatter
   readable output into a shared share.
5. **`docker rm` the container after each run** so its stopped config/env-file
   reference doesn't linger for `docker inspect`.
6. **The real ceiling is host access.** Anyone who can run `docker` on the box is
   root-equivalent (can read mounted secrets, inspect running-container env). So
   "only I" ultimately means *only you (+ root) have shell/docker access to the
   box* — an Unraid-side control (your box login/SSH), not something the
   Dockerfile enforces. Keep box admin access restricted.

**Committable safely:** `Dockerfile.eval`, the recipe, this doc — no secret.
**Never committable:** the `sk-…` key / any `*.env` holding it.

## Verification

- On-box: OWUI-health probe (Deploy step 3) returns `200`.
- First payoff run: full `audrey_research` protocol on-box under `hedge_policy:
  true` completes **uninterrupted, 10/10**, including the two ungrounded controls
  — the canary read this whole exercise has been unable to capture from the
  laptop. Read `/out/…-answers.md`, write the paired report.

## Risks

- **Low.** Additive; no app-code / compose change; the laptop path is untouched.
  Worst case the image can't resolve `open-webui` on the net (step-3 probe
  catches it up front) — fall back to the box LAN-IP base-url.
- Secret-handling is the only real risk surface, fully covered by Security above
  (env-file over `-e`, no bake, `rm` after).

## Open items (on the box)

1. Confirm the `open-webui` service name/port on `ollama-net` (Deploy step 3).
2. Pick the `/out` path (proposed `/mnt/user/appdata/audrey_ai_2.0/testing-out`, or the
   repo's `docs/testing/` if the repo is checked out on the box).
3. Confirm the repo is checked out on the box (so `docker build` has the files),
   else copy the two files over / build on the laptop and load the image.

## Follow-up (offered, not in this phase)

Per-case connection-retry in the harness would harden **both** laptop and on-box
runs against transient blips (retry a case on `ConnectError`/`ReadTimeout` with
backoff; optionally checkpoint completed cases so a re-run resumes). Separable
from this phase; hermetically unit-testable.
