# Campaign 2 Phase 14 — Fleet watchdog (bot liveness monitoring)

Adds a central health monitor for the fleet of LAN bots (OpenClaw,
Hermes) that run on laptops around the network. Before this phase a
bot could die — machine asleep, daemon crashed, or, subtlest of all,
*running but unable to send on Telegram* (blocked, kicked from a
chat, rate-limited) — and nothing would tell you. After this phase a
small always-on service on Unraid watches every bot and DMs you on
Telegram the moment one goes dark, and again when it recovers.

This is a **separate project**, not part of Audrey. It lives in its
own repo at `github.com/robocoppa/fleet-watchdog` and deploys as its
own compose stack on Unraid, alongside (not inside) Audrey. This doc
is filed under campaign-2 as the period's ops log; the canonical
runbook also ships as `DEPLOY.md`/`README.md` inside that repo.

## What it does

A **push-based** heartbeat watchdog. Each bot on each laptop POSTs a
heartbeat to a receiver on Unraid every ~2 minutes; a background loop
on Unraid grades each bot's health and fires edge-triggered Telegram
alerts on any bad transition.

Three failure signals, each its own alert:

  - **stale** — no heartbeat within the bot's threshold. Machine off,
    asleep, or daemon crashed.
  - **send_fail** — the bot is alive and beating, but its last
    `sendMessage` probe failed. This is the signal that catches
    "running but can't talk to its chat" (blocked / kicked / chat
    migrated / rate-limited) — the case a plain token check misses.
  - **token_fail** — `getMe` failed. Token revoked, or Telegram
    unreachable from that laptop.

Plus an opt-in **backend_fail** (model backend Ollama/Audrey
unreachable from the bot).

The `send_fail` signal is the heart of it. The laptop-side probe does
a **real `sendMessage` to the bot's actual target chat, then deletes
it** — proving end-to-end send capability against the chat that
matters, with no visible spam. `getMe` alone would report a
blocked/kicked bot as healthy; the send+delete probe does not.

## Why push, not poll

The bots run on laptops that come and go — sleep, roam, shut down.
A central poller would have to *find* each laptop (changing IPs,
sleep states, no route when off-network) and would constantly
false-alarm on machines that are simply asleep. Push inverts this:
each bot reports in, Unraid only watches for staleness, and a
sleeping laptop just stops beating. No SMB mounts, no SSH from the
hub, no IP chasing.

Per-bot **tiers** in `registry.yaml` handle the roaming case: a
laptop you treat as transient gets `expected_up: false` so it goes
quiet when asleep instead of nagging; an always-on bot gets a tight
`stale_after` and alerts fast.

## Architecture

```text
each laptop, per bot                        Unraid hub (always on)
┌──────────────────────────┐               ┌──────────────────────────────┐
│ heartbeat.sh (timer/cron) │               │ fleet-watchdog container :9099 │
│  • getMe self-check        │  POST /beat   │  • records last-seen per bot   │
│  • sendMessage+delete probe│ ───JSON────►  │  • watchdog loop every 60s:    │
│  • optional backend probe  │               │     stale / send_fail /        │
│  POSTs {bot,host,send_ok,  │               │     token_fail / backend_fail  │
│         send_error,...}     │               │  • edge-triggered alert via    │
└──────────────────────────┘               │     dedicated WATCHER bot      │
                                            └──────────────────────────────┘
                                                        │  🔴/🟠/✅ DM to you
```

Alerts are **edge-triggered**: one message on entering a bad state,
one on recovery, never per-tick spam. Last-seen state is persisted to
JSON so a container restart does not replay stale alerts.

## Two distinct Telegram identities — don't conflate them

This trips people up. There are two separate bots and two separate
chat ids in play:

  - **Watcher bot** (`WATCHDOG_TOKEN` + `WATCHDOG_CHAT_ID`, set on
    Unraid). A *new* bot you make solely to send *you* alerts. It
    must be separate from any monitored bot — a down bot cannot
    deliver its own down-alert. `WATCHDOG_CHAT_ID` is **your personal
    user id** (a positive integer) so the bot DMs you.
  - **Monitored bot** (`BOT_TOKEN` + `TARGET_CHAT`, set per-bot on
    each laptop). The Hermes/OpenClaw bot being watched. `BOT_TOKEN`
    is that bot's own token; `TARGET_CHAT` is the chat that bot
    normally posts to (often a group — a *negative* id), used as the
    send-probe target.

Mnemonic: the watcher talks to *you*; each monitored bot proves it
can talk to *its* chat.

---

## Deploy — Step A: the watcher bot (Telegram, ~2 min)

1. Message **@BotFather**, `/newbot`, give it a name and a
   username ending in `bot`. Save the token it returns —
   that's `WATCHDOG_TOKEN`.
2. Open a chat with your new bot and **send it a message** (tap
   Start, then type anything). A bot cannot DM you until you've
   started a chat with it.
3. Get your personal user id (this is `WATCHDOG_CHAT_ID`): message
   **@userinfobot**, which replies with `Id: <number>`. (The
   `getUpdates` URL works too, but @userinfobot avoids the
   offset-consumed gotcha — see Troubleshooting.)

## Deploy — Step B: the hub on Unraid

Run on Unraid (SSH or terminal). `/mnt/user/appdata/` is where other
container configs live, so clone there:

```bash
cd /mnt/user/appdata
git clone https://github.com/robocoppa/fleet-watchdog.git
cd fleet-watchdog
```

Create `.env` with the watcher-bot secrets from Step A:

```bash
cp .env.example .env
nano .env
```

```text
WATCHDOG_TOKEN=<watcher bot token>
WATCHDOG_CHAT_ID=<your personal user id>
```

Edit `registry.yaml` for your real bots. The id you pick here is what
each laptop sends as `BOT_ID` — they must match exactly. The file is
bind-mounted and re-read every tick, so later edits need no restart:

```yaml
bots:
  hermes-claudette:
    expected_up: true
    stale_after: 300
  openclaw-desk1:
    expected_up: true
    stale_after: 600
    alert_on_backend_fail: true
```

Confirm the external network exists, then build and start:

```bash
docker network ls | grep ollama-net
docker compose up -d --build
```

Verify the hub is alive:

```bash
docker compose ps
docker compose logs --tail=20 fleet-watchdog
curl -s http://localhost:9099/healthz   # {"ok":true}
curl -s http://localhost:9099/status    # {"now":...,"bots":[]}  (empty is correct pre-checkin)
```

End-to-end alert test — POST a fake `send_fail` beat and watch the
watcher bot DM you within one tick (~60s):

```bash
curl -s -X POST http://localhost:9099/beat \
  -H 'Content-Type: application/json' \
  -d '{"bot":"hermes-claudette","host":"smoke-test","send_ok":false,"send_error":"manual end-to-end test"}'
```

Expect: *🟠 hermes-claudette can't send on Telegram… manual
end-to-end test*. Clear it and confirm the recovery DM:

```bash
curl -s -X POST http://localhost:9099/beat \
  -H 'Content-Type: application/json' \
  -d '{"bot":"hermes-claudette","host":"smoke-test","send_ok":true}'
```

Expect: *✅ hermes-claudette recovered*. Both DMs arriving means the
hub is fully working.

### Unraid restart survival

Containers on Unraid don't auto-start after an array reboot on their
own. `restart: unless-stopped` (in the compose) covers crash-restart
and reboots *while it's running*. For robust survival across array
reboots, add it as a stack in the **Compose Manager** plugin and use
its autostart toggle.

## Deploy — Step C: the heartbeat sender on each laptop

Run on **each laptop**, once per bot it hosts. Needs only `bash` +
`curl`. systemd path shown (best for always-on daemons); cron is in
the repo README.

Get the script and install the timer templates:

```bash
cd ~
git clone https://github.com/robocoppa/fleet-watchdog.git
cd fleet-watchdog
sudo cp heartbeat/heartbeat.sh /usr/local/bin/fleet-heartbeat.sh
sudo chmod +x /usr/local/bin/fleet-heartbeat.sh
sudo cp heartbeat/fleet-heartbeat@.service /etc/systemd/system/
sudo cp heartbeat/fleet-heartbeat@.timer   /etc/systemd/system/
sudo mkdir -p /etc/fleet-heartbeat
```

Create one env file per bot. The filename stem **must match the bot
id** in Unraid's `registry.yaml`:

```bash
sudo cp heartbeat/env.example /etc/fleet-heartbeat/hermes-claudette.env
sudo nano /etc/fleet-heartbeat/hermes-claudette.env
sudo chmod 600 /etc/fleet-heartbeat/hermes-claudette.env   # holds a token
```

```text
BOT_ID=hermes-claudette                             # matches registry.yaml
BOT_TOKEN=<the MONITORED bot's own token>           # NOT the watcher bot
TARGET_CHAT=<chat the bot normally posts to>        # send-probe target
WATCHDOG_URL=http://192.168.1.11:9099/beat
HOST_LABEL=claudette-laptop
# BACKEND_URL=http://192.168.1.11:11434/api/version  # uncomment to also check Ollama
```

Test once by hand, then check the hub saw it:

```bash
sudo env $(grep -v '^#' /etc/fleet-heartbeat/hermes-claudette.env | xargs) \
  /usr/local/bin/fleet-heartbeat.sh; echo "exit: $?"
curl -s http://192.168.1.11:9099/status   # hermes-claudette now present, send_ok:true
```

Enable the timer (fires every ~2 min):

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fleet-heartbeat@hermes-claudette.timer
systemctl list-timers 'fleet-heartbeat@*'
```

Second bot on the same laptop: repeat with a new id — drop
`openclaw-desk1.env`, enable `fleet-heartbeat@openclaw-desk1.timer`.
The `@` template serves any number of bots.

### Order matters

Do **B before C**. The laptops POST to the hub, so the hub must be
listening first. Beats sent before the hub is up just fail silently
and retry next tick — harmless, but they won't show in `/status`.

### A note on the send probe and human chats

The probe sends-then-deletes a message in `TARGET_CHAT` every ~2 min.
Invisible on most clients, but a few may briefly flash it, and it
counts against Telegram rate limits. If a monitored bot lives in a
*human* group chat where even a flicker would annoy people, point its
`TARGET_CHAT` at a private muted chat instead.

## Troubleshooting

Symptoms seen during the first deploy, with fixes:

**`getUpdates` returns `{"ok":true,"result":[]}`.** Telegram has no
pending updates for that token. Two causes: (a) you didn't send the
bot a message first — tap Start and send text, *then* reload; or (b)
the update was already consumed by a prior `getUpdates`/polling call,
which advances the offset. Sending a fresh message fixes (b). Easiest
path overall: skip `getUpdates` and use **@userinfobot** for your
personal id.

**`curl http://localhost:9099/healthz` returns nothing.** The
container isn't running or didn't bind. Diagnose in order:

```bash
docker compose ps                              # Up, or Exit/Restarting/absent?
docker compose logs --tail=50 fleet-watchdog   # read the failure
```

Common log causes:
  - `network ollama-net not found` → the external network name
    differs. `docker network ls` to find the real name; fix the
    `external:` block at the bottom of `compose.yaml`.
  - build/pip traceback → paste it.

If `docker compose ps` shows `Up` but curl is still empty, test from
inside the container to isolate host-port vs. app-bind:

```bash
docker compose ps    # expect 0.0.0.0:9099->9099/tcp
docker exec fleet-watchdog curl -s http://127.0.0.1:9099/healthz
```

Inside works but host doesn't → host-port issue (port not published,
or something else on 9099). Inside also empty → app didn't bind; back
to the logs.

Note: a missing or empty `.env` does **not** stop the container — it
logs a warning and runs with alerts disabled. So a blank `.env` is
never the cause of "no response from curl"; that always points at the
container not running or the network.

**A bot never appears in `/status`.** The laptop's `BOT_ID` doesn't
match `registry.yaml` (unmatched bots still appear, so total absence
means the beat isn't arriving), or the laptop can't reach
`192.168.1.11:9099`. Run the by-hand test in Step C and read its exit
code; `curl -v` the `WATCHDOG_URL` from the laptop to check
reachability.

## Operating it day-to-day

  - **Add a bot:** edit `registry.yaml` on Unraid (no restart), then
    install a heartbeat env + timer on its laptop (Step C).
  - **Mute a bot temporarily:** set `expected_up: false` in
    `registry.yaml` — it stays in `/status` but stops alerting on
    staleness. (send/token failures still alert while it's awake if
    those flags are on.)
  - **Check the fleet any time:** `curl -s http://192.168.1.11:9099/status`.
  - **Tune sensitivity:** per-bot `stale_after`, or the global
    `WATCHDOG_DEFAULT_STALE` / `WATCHDOG_TICK` in `.env`.

## Relationship to Audrey

None at the code level — fleet-watchdog is independent. The only
overlap is operational: some monitored bots (OpenClaw, Hermes) talk
to Audrey's passthrough surface (Phase 13) or direct to Ollama, and
their optional `BACKEND_URL` probe can point at Audrey's `/healthz`
or Ollama's `/api/version`. Monitoring those bots' liveness is this
project's job; Audrey neither knows nor cares about the watchdog.
```
