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

### 1. Build (on the box, where the repo/files are)
```bash
docker build -f Dockerfile.eval -t audrey-eval:latest .
```

### 2. One-time: the secret file (yours only)
```bash
#   /mnt/user/appdata/audrey/eval.env   (chmod 600, owned by you)
#     AUDREY_EVAL_BASE_URL=http://open-webui:8080/api
#     AUDREY_EVAL_API_KEY=sk-...
```

### 3. Run (detached — survives laptop disconnect) — SECRET-SAFE
```bash
docker run -d --name audrey-eval --network ollama-net \
  --env-file /mnt/user/appdata/audrey/eval.env \
  -v /mnt/user/appdata/audrey/testing-out:/out \
  audrey-eval:latest \
    --model audrey_research \
    --cases /eval/eval_prompts_protocol.json \
    --save-file /out/2026-07-01-research-onbox-answers.md
```
- `--network ollama-net` → `open-webui:8080` resolves internally; laptop internet
  is irrelevant once launched.
- `--env-file` (NOT `-e`) — see Security. The key comes from your `chmod 600`
  file, not the command line.
- `-v …/testing-out:/out` — answers land on the box disk (dir `chmod 700`);
  copy back into the repo's `docs/testing/` afterward.
- `-d` detached — `docker logs -f audrey-eval` to watch; keeps running if the
  laptop drops. `docker wait audrey-eval` for the exit code.
- After: `docker rm audrey-eval`.

### 4. VERIFY before the first real run — OWUI reachable on the net
```bash
docker run --rm --network ollama-net curlimages/curl:latest \
  -s -o /dev/null -w "%{http_code}\n" http://open-webui:8080/health
```
If not `200`, OWUI's service name/network differs (it lives under the Unraid UI,
not this compose file) — use its real name in `AUDREY_EVAL_BASE_URL`. Fallback
that always works: `AUDREY_EVAL_BASE_URL=http://192.168.1.11:8080/api` (the box's
own LAN IP, hit from a container ON the box — still laptop-independent).

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

- On-box: OWUI-health probe (Deploy step 4) returns `200`.
- First payoff run: full `audrey_research` protocol on-box under `hedge_policy:
  true` completes **uninterrupted, 10/10**, including the two ungrounded controls
  — the canary read this whole exercise has been unable to capture from the
  laptop. Read `/out/…-answers.md`, write the paired report.

## Risks

- **Low.** Additive; no app-code / compose change; the laptop path is untouched.
  Worst case the image can't resolve `open-webui` on the net (step-4 probe
  catches it up front) — fall back to the box LAN-IP base-url.
- Secret-handling is the only real risk surface, fully covered by Security above
  (env-file over `-e`, no bake, `rm` after).

## Open items (on the box)

1. Confirm the `open-webui` service name/port on `ollama-net` (Deploy step 4).
2. Pick the `/out` path (proposed `/mnt/user/appdata/audrey/testing-out`, or the
   repo's `docs/testing/` if the repo is checked out on the box).
3. Confirm the repo is checked out on the box (so `docker build` has the files),
   else copy the two files over / build on the laptop and load the image.

## Follow-up (offered, not in this phase)

Per-case connection-retry in the harness would harden **both** laptop and on-box
runs against transient blips (retry a case on `ConnectError`/`ReadTimeout` with
backoff; optionally checkpoint completed cases so a re-run resumes). Separable
from this phase; hermetically unit-testable.
