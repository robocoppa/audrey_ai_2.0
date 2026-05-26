# OpenClaw — prototype plan

Not a deploy doc yet. This is the design plan for running OpenClaw
(open-source autonomous AI agent — `github.com/openclaw/openclaw`)
against the local Ollama instance on Unraid. Experimental; teardown
notes included.

OpenClaw is structurally similar to Audrey's ReAct loop — LLM picks a
tool, executes, observes, iterates. The difference is that OpenClaw
exposes itself through messaging platforms (Telegram, Discord, etc.)
rather than an OpenAI-compatible HTTP surface, and ships with 100+
built-in skills covering filesystem, email, browser, and API
automation. Bring-your-own-LLM, so it can point at Ollama just as
easily as Anthropic's API.

Channel setup (Telegram bot creation, allowlist, etc.) is out of
scope — done post-install. This plan covers hosting + Ollama
wiring only.

Two hosting options below; recommends one.

## Decisions locked in

- **LLM backend:** local Ollama on Unraid (existing
  `http://<unraid-lan-ip>:11434`). No cloud keys.
- **Model:** `qwen2.5-coder:32b` at Q4_K_M as the first pick
  (~20 GB VRAM, fast tool-loop iterations on a 3090 Ti, strong
  function-calling). Fall back to `llama3.3:70b` Q4_K_M (~42 GB) if
  tool-selection reliability isn't there. Both fit comfortably in
  the 48 GB budget with `GPU_CONCURRENCY=1`.
- **Hosting:** Option B (LAN Linux desktop) recommended over Option A
  (Unraid VM). Reasoning below.
- **Posture:** experimental. Teardown notes at the bottom.

## Why Option B over Option A

Both work. Option B is recommended because:

- **No Unraid surface area added.** OpenClaw asks for broad
  permissions (filesystem, email, calendar, messaging). Sandboxing
  that on the same host as Audrey/Ollama/Qdrant/OWUI is a permission
  story you don't want to debug. A separate LAN host keeps it
  isolated by network boundary instead of by VM boundary.
- **GPU isn't needed on the OpenClaw host.** OpenClaw is a thin
  orchestrator that calls Ollama over HTTP. The desktop only runs
  Docker + a few hundred MB of Python — no passthrough config.
- **Easier to throw away.** `docker compose down -v` on the desktop
  removes everything. An Unraid VM is a heavier teardown and risks
  leaving artifacts (vDisk images, network bridges) behind.
- **Doesn't share Ollama's GPU contention.** OpenClaw's tool-loop
  calls hit Ollama; Audrey's calls also hit Ollama. They already
  share the `GPU_CONCURRENCY=1` gate. Running OpenClaw on Unraid
  doesn't change that — but running it elsewhere makes the
  isolation obvious.

Option A is the right call **only** if the always-on desktop isn't
actually always on, or if you want the entire stack to live on one
piece of hardware for travel reasons.

---

## Option A — Unraid VM

### Shape

- A Linux VM created via Unraid's VM Manager (Settings → VM
  Manager → enabled). Minimum 2 vCPU, 4 GB RAM, 20 GB vDisk.
  Headless Ubuntu Server 24.04 is the easiest path.
- Docker installed inside the VM. OpenClaw runs as a Docker
  Compose stack.
- VM network: bridged (`br0`) so it gets its own LAN IP. Ollama
  is reachable at the Unraid host's LAN IP (not `localhost` from
  the VM — the VM is its own host).

### Steps

1. **Create the VM.**
   - Unraid UI → VMs → Add VM → Linux template.
   - 2 vCPU, 4 GB RAM, 20 GB vDisk on the cache pool.
   - Network: `br0` (bridged). Note the MAC for DHCP reservation.
   - Install Ubuntu Server 24.04 from the ISO (or use a cloud
     image with cloud-init for unattended setup).
2. **Inside the VM, install Docker.**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   # log out / back in to pick up the group
   ```
3. **Pull the OpenClaw repo + write `compose.yaml`** (see
   "Compose file" section below — identical between options).
4. **Verify Ollama reachability from the VM.**
   ```bash
   curl http://<unraid-lan-ip>:11434/api/tags
   ```
   Should list the models. If it fails: Unraid's Ollama container
   needs to bind to the LAN, not just `ollama-net`. Check the
   container's port mapping (`11434:11434`).
5. **Pull the model.** From the Unraid Ollama UI or:
   ```bash
   docker exec ollama ollama pull qwen2.5-coder:32b
   ```
6. **`docker compose up -d`** in the VM. Watch logs:
   ```bash
   docker compose logs -f openclaw
   ```
7. **Configure the channel** per OpenClaw's docs (out of scope here).
8. **Smoke test** by sending the bot a single-step request — e.g.
   "what's the current time?" — and confirm a tool call fires.

### Costs / caveats specific to Option A

- VM Manager adds a maintenance surface (vDisk backups, kernel
  updates inside the VM, Unraid VM Manager itself).
- If Unraid reboots, the VM has to autostart (set in VM config).
- VirtIO networking is fine for HTTP traffic but adds one more
  layer to debug if something doesn't reach Ollama.
- Snapshotting the VM before risky experiments is easy; that's the
  one real advantage of Option A.

---

## Option B — LAN Linux desktop (recommended)

### Shape

- Existing always-on Linux desktop on the LAN runs Docker.
- OpenClaw runs as a Docker Compose stack pointed at
  `http://<unraid-lan-ip>:11434` for Ollama.
- If the chosen channel uses outbound polling (Telegram, Discord),
  no inbound port-forward is needed on the desktop.

### Steps

1. **Confirm Docker on the desktop.**
   ```bash
   docker --version
   docker compose version
   ```
   If missing: `curl -fsSL https://get.docker.com | sh` + add user
   to the `docker` group.
2. **Verify Ollama reachability.**
   ```bash
   curl http://<unraid-lan-ip>:11434/api/tags
   ```
   Same check as Option A. If this fails, fix it before touching
   OpenClaw — nothing else will work.
3. **Pull the model on Unraid** (one-time, lives in the Ollama
   container's volume):
   ```bash
   docker exec ollama ollama pull qwen2.5-coder:32b
   ```
4. **Create a directory for the OpenClaw stack** on the desktop,
   e.g. `~/openclaw/`. Drop the `compose.yaml` below in it.
5. **`docker compose up -d`** in `~/openclaw/`. Logs:
   ```bash
   docker compose logs -f openclaw
   ```
6. **Configure the channel** per OpenClaw's docs (out of scope here).
7. **Smoke test.** Try a single-step task first ("what time is
   it?"), then a two-step ("create a file `hello.txt` in `/tmp`
   with the text 'hi'").

### Costs / caveats specific to Option B

- The desktop has to actually stay on. Schedule wake-on-LAN or
  disable suspend if it doesn't already.
- LAN dependency: if the desktop or its network drops, OpenClaw
  is offline. Audrey is unaffected.
- Backups: OpenClaw state (auth tokens, chat history) lives in
  the Docker volume. Back up `~/openclaw/data/` periodically if
  you care about retention.

---

## Compose file (both options)

Identical between A and B — the only thing that changes is which
host you put it on. `OLLAMA_BASE_URL` points at Unraid's LAN IP.

```yaml
# ~/openclaw/compose.yaml  (Option B) or /home/<user>/openclaw/compose.yaml  (Option A)
services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    environment:
      OLLAMA_BASE_URL: http://<unraid-lan-ip>:11434
      OPENCLAW_MODEL: qwen2.5-coder:32b
      # Channel credentials (e.g. TELEGRAM_BOT_TOKEN) live in .env —
      # set up post-install per OpenClaw's docs.
    env_file:
      - .env
    volumes:
      - ./data:/data
```

Notes:

- Pin the image tag (`openclaw/openclaw:v1.2.3` or a digest) before
  treating this as anything other than experimental. `:latest` is
  fine for prototyping; not fine for "I'd like this to keep
  working."
- Add a healthcheck once you know what OpenClaw exposes — the
  README will document `/health` or similar.
- The exact env var names above (`OLLAMA_BASE_URL`,
  `OPENCLAW_MODEL`) are educated guesses based on OpenClaw being
  typical of this category. Confirm against `openclaw/openclaw`
  README before first run; correct any mismatches in the compose
  file.

## Validation plan

Tasks to walk before declaring it usable, in order of escalating
trust:

1. **Single-step text:** "what time is it?" → expect a `time`
   tool call.
2. **Read-only filesystem:** "what's in `/tmp`?" → expect an `ls`
   or equivalent.
3. **Two-step chain:** "find the largest file in `/tmp` and tell me
   its size." → tool → tool → answer.
4. **Mutating operation:** "create `/tmp/openclaw-test.txt` with
   the text 'hello'." → verify the file exists.
5. **Multi-tool agentic task:** "summarize the README at
   github.com/openclaw/openclaw and save it to
   `/tmp/openclaw-summary.md`." → web fetch + file write +
   summarization.

If step 5 works reliably on three out of three tries, the model is
good enough. If it fails repeatedly with `qwen2.5-coder:32b`, swap
to `llama3.3:70b` Q4_K_M and re-run.

## Risks worth naming

- **Tool-selection hallucination.** Local models pick tools less
  reliably than GPT-4 / Claude. Expect occasional "I'll use the
  email tool" when the user asked about files. Mitigation:
  start with the recommended model, don't drop below 32B Q4.
- **Permission scope.** OpenClaw can touch the entire filesystem
  of its host. Run it as a non-root user inside the container, and
  consider mounting only the directories you actually want it to
  reach (e.g. `./workspace:/workspace` instead of binding `/`).
- **Channel credentials.** Keep tokens out of git. Don't commit
  `.env`. Rotate at the channel provider if it leaks.
- **Ollama contention with Audrey.** Both call the same Ollama
  instance under `GPU_CONCURRENCY=1`. If OpenClaw is mid-loop,
  Audrey's requests queue. Acceptable for a prototype; revisit
  if it bites.
- **Token cost looks like zero — until it isn't.** Local inference
  doesn't bill, but agentic loops can rack up thousands of tokens
  per task. Watch the Ollama logs for runaway loops.

## Teardown

If the experiment doesn't pan out:

**Option A (VM):**
```bash
# Inside the VM, then shut it down:
cd ~/openclaw && docker compose down -v
# On Unraid:
# VMs → openclaw-vm → Stop → Remove → Delete vDisks
```
Then delete the model from Ollama if it's not used elsewhere:
```bash
docker exec ollama ollama rm qwen2.5-coder:32b
```

**Option B (desktop):**
```bash
cd ~/openclaw && docker compose down -v
rm -rf ~/openclaw
# Delete the model from Ollama on Unraid if no longer wanted:
docker exec ollama ollama rm qwen2.5-coder:32b
```

**Channel cleanup:** delete the bot / app / integration at the
channel provider. This permanently invalidates the token; even if
`.env` leaks later, it can't be reused.

## Next-step decisions, when it's working

Not in scope for the prototype, but worth flagging for the
post-experiment debrief:

- Pin to a specific OpenClaw release tag or digest.
- Add it to monitoring (`monitoring/compose.yaml`)? It would
  share Audrey's Grafana, but only if Option A — Option B sits
  on a different host.
- Decide whether OpenClaw should be allowed to call Audrey via
  OWUI's API (giving it Audrey's full virtual-model surface as a
  tool). Interesting but not the prototype's job.
- Consider a per-tool allowlist in OpenClaw config — start with
  filesystem + web fetch + shell, add others only after a real
  use case appears.
