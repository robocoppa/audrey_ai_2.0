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
heartbeat to a receiver on Unraid every 5 minutes; a background loop
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
a **real `sendMessage` to a shared, muted probe channel** (not deleted) —
proving the bot can actually post on Telegram, without cluttering any real
chat. `getMe` alone would report a rate-limited or restricted bot as
healthy; the send probe does not.

> **Probe channel, not the real chat.** An earlier design posted to the
> bot's real chat and immediately deleted the message — which flickered on
> some clients and was unpleasant. The probe now posts to a private channel
> you create just for this and mute. Trade-off: it proves the bot can send
> *somewhere*, not specifically to its real chat, so it won't catch the bot
> being kicked from one particular real chat. Every other failure (token
> revoked, rate-limited, Telegram down, daemon dead) is the same signal.
> The alerts to *you* are unaffected — those always go to your watcher-bot
> DM and are never deleted.

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
  - **Monitored bot** (`BOT_TOKEN` + `PROBE_CHAT`, set per-bot on
    each laptop). The Hermes/OpenClaw bot being watched. `BOT_TOKEN`
    is that bot's own token; `PROBE_CHAT` is the **shared probe
    channel** (a *negative* `-100…` id) every monitored bot posts its
    heartbeat to. One channel for the whole fleet; bots are told apart
    by the message text. Each monitored bot must be an admin of that
    channel so it can post.

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

### Make the shared probe channel

The monitored bots prove they can send by posting a heartbeat to a
**single private channel** you create just for this (the `PROBE_CHAT`).
You mute it and never read it — the messages are the probe, not content.

1. In Telegram, create a new **channel** (or group), private. Name it
   something like "Fleet Probes" and **mute it**.
2. Add **each monitored bot** (every Hermes/OpenClaw bot you'll watch) as
   an **admin** of the channel, so it has permission to post.
3. Get the channel's chat id — it's a `-100…` number. Easiest: post any
   message in the channel, forward it to **@userinfobot**, and read the
   `Forwarded from chat` id. That number is `PROBE_CHAT`, the same value
   for every bot's env file.

One channel serves the whole fleet; bots are told apart by their
heartbeat text (`🩺 <bot-id>@<host> <time>`).

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
# Senders beat every 5 min (300s). Set stale_after >= 2x the interval +
# margin so one dropped beat doesn't false-alarm. 700s = "two missed beats".
bots:
  my-bot:               # an always-on bot — alert if it misses two beats
    expected_up: true
    stale_after: 700
  desk-bot:             # always-on, and watch its model backend too
    expected_up: true
    stale_after: 700
    alert_on_backend_fail: true
  roaming-bot:          # a laptop that sleeps — present but don't nag
    expected_up: false
    stale_after: 900
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
  -d '{"bot":"my-bot","host":"smoke-test","send_ok":false,"send_error":"manual end-to-end test"}'
```

Expect: *🟠 my-bot can't send on Telegram… manual
end-to-end test*. Clear it and confirm the recovery DM:

```bash
curl -s -X POST http://localhost:9099/beat \
  -H 'Content-Type: application/json' \
  -d '{"bot":"my-bot","host":"smoke-test","send_ok":true}'
```

Expect: *✅ my-bot recovered*. Both DMs arriving means the
hub is fully working.

### Smoke tests — exercise every signal

The end-to-end test above proves the `send_fail` → recovery edge. These
walk each of the **five** watchdog states independently, so you can
confirm the whole alerting matrix before trusting it with the real fleet.

**Run these from the Unraid host via `docker exec`.** The slim container
has `python` but **no `curl`**, and the Unraid host has no reliable
`python3` — so the cleanest path is to drive everything *inside* the
container with its own `python`. That also dodges host networking
entirely: the calls hit `127.0.0.1:9099` from within the container, which
is always reachable if the container is up.

After each "down" POST, the alert lands within **one watchdog tick** —
`WATCHDOG_TICK`, default **60s** — because alerting is edge-triggered on
the loop, not on the POST. `/status` updates immediately; the *DM* waits
for the next tick. So the rhythm is: POST → check state now → wait up to
~60s for the DM.

Use a throwaway bot id `smoke-bot` so you don't pollute real bots'
state. `smoke-bot` isn't in `registry.yaml`, so it inherits the defaults:
`expected_up: true`, `stale_after: WATCHDOG_DEFAULT_STALE` (700s),
`alert_on_send_fail`/`token_fail` on, `alert_on_backend_fail` **off**.

> **One DM per tick, not per POST — pace yourself.** Alerting fires on
> the *net* state change since the last alerted state, evaluated once per
> tick. If you fire several POSTs faster than one tick, intermediate
> states collapse and you'll see only the final transition's DM (the
> `/status` state is still correct at each step — only the DMs coalesce).
> To see each signal's *own* DM, leave ~one tick (~60s) between the
> "down" POSTs. State checks can be immediate; the DMs need the gap.

Define two shell functions on the Unraid host (they shell into the
container and use its `python` + `urllib` — no curl, no host python):

```bash
beat() { docker exec fleet-watchdog python -c "
import urllib.request, sys
urllib.request.urlopen(urllib.request.Request(
    'http://127.0.0.1:9099/beat', data=sys.argv[1].encode(),
    headers={'Content-Type':'application/json'}))" "$1" >/dev/null; }

seestate() { docker exec fleet-watchdog python -c "
import urllib.request, json
d = json.load(urllib.request.urlopen('http://127.0.0.1:9099/status'))
for b in d['bots']:
    if b['bot']=='smoke-bot':
        print(' ', b['bot'], '->', b['state'], '|', b.get('send_error') or '')"; }
```

> Prefer one-liners? Each test below also works as a single
> `docker exec fleet-watchdog python -c "..."` — but the helpers keep the
> test bodies readable. To watch DMs as they fire (handy while testing),
> tail the log in another terminal: `docker compose logs -f fleet-watchdog`.

**Test 1 — healthy baseline.** A good heartbeat should produce state `ok`
and **no** DM.

```bash
beat '{"bot":"smoke-bot","host":"smoke","send_ok":true,"getme_ok":true,"backend_ok":true}'
seestate          # →  smoke-bot -> ok |
```

Expect: state `ok`, and your watcher bot stays silent. (A silent healthy
bot is the most important non-event to confirm — it proves you won't get
nagged for working bots.)

**Test 2 — `send_fail` (bot up, can't post to its chat).** The headline
case.

```bash
beat '{"bot":"smoke-bot","host":"smoke","send_ok":false,"send_error":"bot was blocked by the user"}'
seestate          # →  smoke-bot -> send_fail | bot was blocked by the user
```

Expect within ~60s: *🟠 smoke-bot can't send on Telegram … bot was
blocked by the user*.

**Test 3 — `token_fail` (getMe failed; takes priority over send).** Note
`getme_ok:false` *and* `send_ok:false` — the watchdog should report
`token_fail`, because a failed getMe means the send result can't be
trusted.

```bash
beat '{"bot":"smoke-bot","host":"smoke","getme_ok":false,"send_ok":false}'
seestate          # →  smoke-bot -> token_fail |
```

Expect within ~60s: *🟠 smoke-bot token/Telegram problem*. This also
confirms the **priority order** (token over send), not just that an alert
fires.

**Test 4 — recovery edge.** Send a clean beat; the bot should return to
`ok` and emit exactly one recovery DM.

```bash
beat '{"bot":"smoke-bot","host":"smoke","send_ok":true,"getme_ok":true}'
seestate          # →  smoke-bot -> ok |
```

Expect within ~60s: *✅ smoke-bot recovered*. **Confirm it fires once** —
leave it a few minutes and check no repeat DMs arrive. That proves the
edge-trigger (alert on transition, not every tick).

**Test 5 — `backend_fail` is gated by registry.** `smoke-bot` inherits
`alert_on_backend_fail: false`, so a backend failure should **not** alert.

```bash
beat '{"bot":"smoke-bot","host":"smoke","send_ok":true,"backend_ok":false}'
seestate          # →  smoke-bot -> ok |   (NOT backend_fail — gated off)
```

Expect: state stays `ok`, **no DM**. To prove the gate works the *other*
way, repeat against a real id that ships `alert_on_backend_fail: true`
(`desk-bot` in the registry example) — change `smoke-bot` to that id in the
`beat` payload — and you should get *🟠 desk-bot model backend
unreachable*. Then clear it with a `backend_ok:true` beat for that id.

**Test 6 — `stale` (no heartbeat).** This one is **time-based**: stop
beating and wait `stale_after` seconds. With `smoke-bot`'s inherited 700s
default that's a ~12-minute wait, so either be patient or temporarily set
`WATCHDOG_DEFAULT_STALE=90` in `.env` and `docker compose up -d` first.

```bash
beat '{"bot":"smoke-bot","host":"smoke","send_ok":true}'   # last beat
# ...then send nothing. After stale_after seconds:
seestate          # →  smoke-bot -> stale |
```

Expect after the threshold: *🔴 smoke-bot is DOWN … No heartbeat for Ns*.
This is the signal that catches a powered-off laptop or a crashed daemon —
the one you can't test with a POST, only by *withholding* one.

> **Transient bots don't go stale.** If you want to confirm the
> `expected_up: false` behavior, run Test 6 against a transient id
> (`roaming-bot` in the registry example) instead: stop beating it and it
> should stay `ok` (a sleeping roaming laptop is not an incident), never
> emitting a 🔴.

**Cleanup.** `smoke-bot` lingers in `/status` (and `state.json`) after
testing. It's harmless — unregistered, and you can ignore it — but to
remove it entirely, stop the container, delete its entry from
`data/state.json` on the host, and restart:

```bash
docker compose stop fleet-watchdog
# edit data/state.json, remove the "smoke-bot" key (or: echo '{}' > data/state.json to clear all)
docker compose up -d fleet-watchdog
```

If you set `WATCHDOG_DEFAULT_STALE=90` for Test 6, revert it and
`docker compose up -d` again.

### Unraid restart survival

Containers on Unraid don't auto-start after an array reboot on their
own. `restart: unless-stopped` (in the compose) covers crash-restart
and reboots *while it's running*. For robust survival across array
reboots, add it as a stack in the **Compose Manager** plugin and use
its autostart toggle.

## Deploy — Step C: the heartbeat sender on each laptop

Run on **each laptop**, once per bot it hosts. Needs only `bash` +
`curl` — both present on Linux and macOS. The *scheduler* differs by OS:

- **Linux → systemd timer** (§C-Linux below). Best for always-on daemons.
- **macOS (e.g. a Mac mini) → launchd** (§C-macOS below). systemd does
  **not** exist on macOS — don't try the `systemctl` steps there.
- Either OS → **cron** is a simpler fallback (repo README + cron-wrapper.sh).

`heartbeat.sh` itself and the `BOT_ID`/`BOT_TOKEN`/`PROBE_CHAT`/
`WATCHDOG_URL` config are identical across both; only the scheduler and
a couple of paths change.

### C-Linux (systemd)

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

Replace `my-bot` below with this bot's id — the filename stem **must
match the bot id** in Unraid's `registry.yaml`:

```bash
sudo cp heartbeat/env.example /etc/fleet-heartbeat/my-bot.env
sudo nano /etc/fleet-heartbeat/my-bot.env
sudo chmod 600 /etc/fleet-heartbeat/my-bot.env   # holds a token
```

```text
BOT_ID=my-bot                                       # matches registry.yaml
BOT_TOKEN=<the MONITORED bot's own token>           # NOT the watcher bot
PROBE_CHAT=<shared probe channel -100… id>          # same for every bot
WATCHDOG_URL=http://192.168.1.11:9099/beat
HOST_LABEL=my-laptop
# BACKEND_URL=http://192.168.1.11:11434/api/version  # uncomment to also check Ollama
```

Test once by hand, then check the hub saw it:

```bash
sudo env $(grep -v '^#' /etc/fleet-heartbeat/my-bot.env | xargs) \
  /usr/local/bin/fleet-heartbeat.sh; echo "exit: $?"
curl -s http://192.168.1.11:9099/status   # my-bot now present, send_ok:true
```

Enable the timer (fires every 5 min):

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fleet-heartbeat@my-bot.timer
systemctl list-timers 'fleet-heartbeat@*'
```

Second bot on the same laptop: repeat with a new id — drop
`<other-bot>.env`, enable `fleet-heartbeat@<other-bot>.timer`.
The `@` template serves any number of bots.

### C-macOS (launchd) — e.g. a Mac mini

macOS has no systemd. The native scheduler is **launchd**, where a
LaunchAgent `.plist` with `StartInterval` plays the role of the systemd
timer. launchd doesn't read an env file the way systemd does, so each bot
gets a small **wrapper** that exports its vars then execs `heartbeat.sh` —
the repo's `cron-wrapper.sh` already does exactly that, so we reuse it.

Replace `brigitte` below with this bot's id (must match `registry.yaml`):

```bash
cd ~
git clone https://github.com/robocoppa/fleet-watchdog.git
cd fleet-watchdog

# 1. The probe script (shared by all bots on this Mac).
sudo cp heartbeat/heartbeat.sh /usr/local/bin/fleet-heartbeat.sh
sudo chmod +x /usr/local/bin/fleet-heartbeat.sh

# 2. A per-bot wrapper holding this bot's config. Edit the exported vars
#    inside it: BOT_ID=brigitte, BOT_TOKEN, PROBE_CHAT, WATCHDOG_URL,
#    HOST_LABEL (and optional BACKEND_URL).
cp heartbeat/cron-wrapper.sh /usr/local/bin/fleet-heartbeat-brigitte.sh
nano /usr/local/bin/fleet-heartbeat-brigitte.sh
chmod +x /usr/local/bin/fleet-heartbeat-brigitte.sh

# 3. Test once by hand before scheduling.
/usr/local/bin/fleet-heartbeat-brigitte.sh; echo "exit: $?"
curl -s http://192.168.1.11:9099/status   # brigitte now present, send_ok:true
```

> **Apple Silicon path note.** `/usr/local/bin` works on both Intel and
> Apple Silicon for a hand-placed script. (Homebrew uses `/opt/homebrew/bin`
> on Apple Silicon, but you're not installing via brew here.) If you prefer,
> drop the scripts anywhere on `PATH` and point the wrapper/plist at that
> path.

Now the launchd job. Rename the example plist per bot and edit its
`Label`, `ProgramArguments`, and log paths to match:

```bash
# Edit Label → com.fleet-watchdog.brigitte, ProgramArguments → the wrapper,
# and the log paths, inside the example first.
cp heartbeat/com.fleet-watchdog.heartbeat.plist.example \
   ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
launchctl load ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
```

`StartInterval` is 300s in the example, matching the systemd timer's 5-min
cadence; `RunAtLoad` fires the first beat immediately on load. Verify and
manage:

```bash
launchctl list | grep fleet-watchdog          # is it loaded?
launchctl start com.fleet-watchdog.brigitte   # fire once now
cat /tmp/fleet-heartbeat-brigitte.err         # any errors from the last run
# to stop:
launchctl unload ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
```

A **LaunchAgent** (`~/Library/LaunchAgents`, runs as the logged-in user)
is right when the bot runs as that user — the usual case. Use a
**LaunchDaemon** (`/Library/LaunchDaemons`, runs as root) only if the bot
is a system-wide service. Second bot on the same Mac: another wrapper +
another `.plist` with a new `Label`.

> **macOS gotchas.** (1) The first run may prompt for permissions
> (e.g. network); approve it. (2) launchd jobs don't fire while the Mac is
> *asleep* — a Mac mini left on won't sleep its scheduler, but a closed
> laptop will, and its heartbeat simply stops (which the hub treats as
> stale, correctly, if `expected_up: true`). (3) If `launchctl load`
> reports "service already loaded," `unload` first then `load` again.

### Order matters

Do **B before C**. The laptops POST to the hub, so the hub must be
listening first. Beats sent before the hub is up just fail silently
and retry next tick — harmless, but they won't show in `/status`.

### A note on the send probe

Each bot posts a heartbeat to the shared `PROBE_CHAT` channel every 5 min
and leaves it there (no delete). Because it's a dedicated, muted channel
you never read, this never touches a real chat. The messages just
accumulate — harmless, but if the channel ever gets visually noisy you can
clear its history in Telegram any time; it has no effect on monitoring.
The probes do count against Telegram's per-bot rate limits, but at one
small message per 2 min per bot that's negligible.

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
