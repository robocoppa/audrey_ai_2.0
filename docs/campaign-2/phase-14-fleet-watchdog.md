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
each host, per bot                          Unraid hub (always on)
┌────────────────────────────┐             ┌──────────────────────────────┐
│ heartbeat.sh — run from the │             │ fleet-watchdog container :9099 │
│ git checkout, NOT a copy:   │  POST /beat │  • records last-seen per bot   │
│  • getMe self-check          │ ──JSON───►  │  • watchdog loop every 60s:    │
│  • sendMessage probe (no del)│             │     stale / send_fail /        │
│  • optional backend probe    │             │     token_fail / backend_fail  │
│ scheduler:                   │             │  • edge-triggered alert via    │
│  • Linux → systemd timer      │             │     dedicated WATCHER bot      │
│  • Mac   → launchd KEEP-ALIVE │             └──────────────────────────────┘
│           loop (heartbeat-loop)│                       │  🔴/🟠/✅ DM to you
│ fleet-update.sh (slow timer) ──┘ git checkout released  ▼
│  converges the checkout on the `released` tag every ~15 min
└────────────────────────────┘
```

Alerts are **edge-triggered**: one message on entering a bad state,
one on recovery, never per-tick spam. Last-seen state is persisted to
JSON so a container restart does not replay stale alerts.

**Deploy model in one line:** each host runs `heartbeat.sh` *straight
from a git checkout* (no `/usr/local/bin` copy), and a slow
`fleet-update` timer keeps that checkout on the `released` tag — so
shipping a change is `git tag -f released && git push -f origin
released`, and the fleet converges within ~15 min. No per-host
reinstall.

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

Create your real registry from the example. `registry.yaml` is
**gitignored** — it holds your actual fleet (bot ids + per-bot
thresholds), which is deployment-specific, so it stays local and a
`git pull` on the hub never reverts your hand-tuned values:

```bash
cp registry.yaml.example registry.yaml
nano registry.yaml
```

The id you pick here is what each host sends as `BOT_ID` — they must
match exactly. The file is bind-mounted and re-read every tick:

```yaml
# Senders beat every 5 min (300s). Set stale_after >= 2x the interval +
# margin so one dropped beat doesn't false-alarm. 700s = "two missed beats".
bots:
  alice-bot:            # an always-on Linux bot — alert if it misses two beats
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

> **Edit `registry.yaml` IN PLACE (`nano`), never by replacing it.**
> The file is bind-mounted into the container by inode. `nano` edits
> keep the inode, so the re-read-each-tick, no-restart behavior works.
> Anything that *swaps* the file — `git pull`, `git checkout`,
> `cp something registry.yaml` — replaces the inode, and the running
> container is then left holding a **stale file handle**: `/status`
> starts returning `500` with `OSError: [Errno 116] Stale file handle`.
> Cure is a `docker compose restart fleet-watchdog` to re-open the
> mount. (Gitignoring `registry.yaml` largely prevents this, since you
> stop running git operations against it — but a stray `cp` still
> trips it.)

> **Mac bots and `stale_after`.** A Mac on the keep-alive loop model
> (see Step C-Mac) beats as reliably as a Linux systemd timer, so
> `700` is fine. Only a Mac left on the older launchd `StartInterval`
> scheduler needs a looser value (~1100+) to absorb launchd's timer
> jitter — the loop model is preferred precisely so you don't have to.

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

**Cleanup.** `smoke-bot` (and any other test ids) linger in `/status`
after testing. They're harmless — unregistered, and you can ignore them —
but to clear them entirely, wipe the persisted state and restart. Real bots
re-register on their next beat; dead test ids don't.

```bash
docker compose stop fleet-watchdog
echo '{}' > data/state.json
docker compose up -d fleet-watchdog
```

Restarting (rather than hand-editing `data/state.json` while the container
runs) is deliberate: the file is bind-mounted, and replacing it underneath a
running container can leave a **stale file handle** (`/status` then 500s
with `OSError: Stale file handle`). The stop/wipe/up cycle re-opens the
mount cleanly. Likewise, edit `registry.yaml` **in place** (`nano`), not by
replacing it (`cat >` / git checkout swaps the inode) — in-place edits keep
the no-restart, re-read-each-tick behavior; an inode swap needs a restart.

If you set `WATCHDOG_DEFAULT_STALE=90` for Test 6, revert it and
`docker compose up -d` again.

### Unraid restart survival

Containers on Unraid don't auto-start after an array reboot on their
own. `restart: unless-stopped` (in the compose) covers crash-restart
and reboots *while it's running*. For robust survival across array
reboots, add it as a stack in the **Compose Manager** plugin and use
its autostart toggle.

## Deploy — Step C: heartbeat senders (read first)

A heartbeat sender runs on **each machine that hosts a bot**, once per
bot. The script (`heartbeat.sh`) and the config keys
(`BOT_ID`/`BOT_TOKEN`/`PROBE_CHAT`/`WATCHDOG_URL`/`HOST_LABEL`) are
**identical on every OS**, and on every OS the script runs **from the git
checkout, not a `/usr/local/bin` copy**. Only the *scheduler* differs:

- **Linux → systemd timer.** See [Step C-Linux](#deploy--step-c-linux-add-a-linux-host).
- **Mac → launchd KEEP-ALIVE loop** (not `StartInterval` — it's too
  throttled). See [Step C-Mac](#deploy--step-c-mac-add-a-mac).
  systemd does **not** exist on macOS — don't run the `systemctl` steps there.

Two rules that apply to every machine:

1. **`BOT_ID` must match the bot's key in Unraid's `registry.yaml`.** A
   mismatch means the heartbeat arrives but is treated as an unknown bot.
2. **Do Step B before Step C.** Senders POST to the hub, so the hub must be
   listening first. Beats sent before the hub is up just fail silently and
   retry on the next tick — harmless, but they won't show in `/status`.

### The release / self-update model

Once a host is set up, you never hand-redeploy code to it. Each host runs a
slow `fleet-update` timer that converges its checkout on the **`released`**
git tag. The workflow:

- Push to `main` as often as you like — **nothing deploys**.
- When a commit is fleet-ready, move the tag and push it:

  ```bash
  git tag -f released <commit>     # default: current HEAD
  git push -f origin released
  ```

- Every host's updater fetches and checks out that tag on its next run
  (~15 min), and the schedulers already run `heartbeat.sh` straight from the
  checkout — so updating the checkout **is** the deploy. No `sudo`, no
  reinstall. A half-finished push to `main` never reaches the fleet; only the
  tag does. (The Unraid **hub** is updated by hand, not by this timer — it
  tracks `main`; see day-to-day ops.)

**About the probe.** Each bot posts a heartbeat to the shared `PROBE_CHAT`
channel every 5 min and leaves it there (no delete). It's a dedicated,
muted channel you never read, so it never touches a real chat; the messages
just accumulate harmlessly (clear the channel history any time — no effect
on monitoring). The bot **must be an admin of that channel** or its probe
fails and you'll see `send_ok:false`.

---

## Deploy — Step C-Linux: add a Linux host

Self-contained runbook for one Linux machine. Repeat the per-bot block for
each bot it hosts. Example bot id below is `alice-bot`, login user `alice` —
substitute yours. (This runbook assumes the login user owns the checkout and
runs the bot; replace `alice`/`/home/alice` throughout.)

Linux uses a **systemd timer** as the scheduler, and runs `heartbeat.sh`
**directly from the checkout** — no `/usr/local/bin` copy. Updating the
checkout (which the self-updater does for you) is the whole deploy.

### 1. Clone the repo in the user's home (once per machine)

If it's already cloned, skip to step 2 — the self-updater keeps it current.

```bash
cd ~
git clone https://github.com/robocoppa/fleet-watchdog.git
cd fleet-watchdog
git fetch --tags --force origin
git checkout --detach released
sudo mkdir -p /etc/fleet-heartbeat
```

The `git checkout --detach released` pins this host to the blessed tag (not
a moving branch) — exactly what the self-updater will keep it on.

### 2. Create the per-bot env file (once per bot)

The filename stem **must match the bot id** in `registry.yaml`.

```bash
sudo cp ~/fleet-watchdog/heartbeat/env.example /etc/fleet-heartbeat/alice-bot.env
sudo nano /etc/fleet-heartbeat/alice-bot.env
sudo chmod 600 /etc/fleet-heartbeat/alice-bot.env   # holds a token
```

```text
BOT_ID=alice-bot                                    # matches registry.yaml
BOT_TOKEN=<the MONITORED bot's own token>           # NOT the watcher bot
PROBE_CHAT=<shared probe channel -100… id>          # same for every bot
WATCHDOG_URL=http://192.168.1.11:9099/beat
HOST_LABEL=alice-laptop
# BACKEND_URL=http://192.168.1.11:11434/api/version  # uncomment to also check Ollama
```

> A root-owned `600` env file is fine for the systemd unit even though the
> unit runs as a non-root `User=` (below): systemd's PID 1 (root) reads
> `EnvironmentFile=` *before* dropping to the unit's user, so the user
> never needs read access to the file itself.

### 3. Install the heartbeat timer template, pointed at the checkout

Copy the `@`-templated unit + timer, then edit the unit to run as your user
and exec from your checkout. (The template ships with `youruser` placeholders
in `User=`/`Group=`/`ExecStart`.)

```bash
sudo cp ~/fleet-watchdog/heartbeat/fleet-heartbeat@.service \
        ~/fleet-watchdog/heartbeat/fleet-heartbeat@.timer /etc/systemd/system/
sudo nano /etc/systemd/system/fleet-heartbeat@.service
```

Set these three lines for your user (`alice`):

```text
User=alice
Group=alice
ExecStart=/home/alice/fleet-watchdog/heartbeat/heartbeat.sh
```

> **`%i` shortcut (optional).** If your login username equals the bot id and
> you run exactly one bot per host (the common case), you can leave the
> template generic with `User=%i`, `Group=%i`,
> `ExecStart=/home/%i/fleet-watchdog/heartbeat/heartbeat.sh` — `%i` expands
> to the instance/bot id at start time, so one unedited template serves the
> host. **Caveat:** this breaks the moment a host runs a *second* bot whose
> id isn't a local username (it'd try `User=<that-id>` → `203/EXEC`). For
> strict one-bot-per-host with user==bot-id, `%i` is clean; otherwise use
> the literal form above.

### 4. Test by hand, then enable

The env file is `chmod 600` (root-owned), so a by-hand test runs the **whole**
pipeline as root — `sudo bash -c '...'`, not `sudo env $(grep ...)` (the
latter runs `grep` as your user → "Permission denied" → no config):

```bash
sudo bash -c 'env $(grep -v "^#" /etc/fleet-heartbeat/alice-bot.env | xargs) \
  /home/alice/fleet-watchdog/heartbeat/heartbeat.sh'; echo "exit: $?"
curl -s http://192.168.1.11:9099/status   # alice-bot now present, send_ok:true
```

`exit: 0` + a `🩺 alice-bot@…` message in the probe channel = working.

> **`exit: 0` proves the script RAN, not that the beat LANDED.** The script
> always exits 0 by design (so the scheduler never spams mail) — even when
> the hub POST fails. The real proof a beat arrived is the bot appearing in
> the hub's `/status`, not the exit code.

Then enable the timer (fires every 5 min):

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fleet-heartbeat@alice-bot.timer
systemctl list-timers 'fleet-heartbeat@*'
```

A healthy oneshot run shows `inactive (dead)` with `status=0/SUCCESS` — that
is success for a oneshot, not a failure.

### 5. Install the self-updater (once per machine)

A slow `--user` timer that converges the checkout on the `released` tag, so
future deploys need no per-host work. It runs as your user (no sudo to ship
code, matching the no-copy layout):

```bash
mkdir -p ~/.config/systemd/user
cp ~/fleet-watchdog/heartbeat/fleet-update.service \
   ~/fleet-watchdog/heartbeat/fleet-update.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fleet-update.timer
sudo loginctl enable-linger alice      # let the --user timer run while logged out
```

The `enable-linger` line matters on an always-on host: without it, a `--user`
timer only runs while you're logged in. Confirm it's armed:

```bash
systemctl --user list-timers fleet-update.timer   # shows a NEXT run time
```

**Another bot on the same host?** Repeat steps 2–4 with a new id (the `@`
template serves any number of bots). Steps 1 and 5 are once per machine.

---

## Deploy — Step C-Mac: add a Mac

Self-contained runbook for one Mac (mini or laptop). macOS has no systemd;
the scheduler is **launchd**. Example bot id below is `brigitte`, home
`/Users/brigitte` — substitute yours (`echo $HOME` if unsure).

Like Linux, the Mac runs `heartbeat.sh` **from the checkout** (no
`/usr/local/bin` copy). The two Mac-specific wrinkles:

- launchd doesn't read env files, so each bot gets a small **wrapper** that
  exports its vars and execs the heartbeat — the launchd equivalent of the
  Linux `.env`. It lives in your home dir (it holds a token), `chmod 700`.
- **Use the KEEP-ALIVE LOOP model, not `StartInterval`.** launchd's
  `StartInterval` is a loose, power-managed scheduler: on an idle Mac it
  coalesces/throttles an interval job hard — a "5-min" beat can slip to
  15+ min, overrunning the hub's `stale_after` and firing false 🔴 pages.
  Instead, run `heartbeat-loop.sh` (which beats, `sleep`s, repeats) as a
  launchd **`KeepAlive`** job: launchd just keeps the process alive, and the
  script's own `sleep` is the honest cadence. No throttling.

### 1. Clone the repo (once per machine)

If it's already cloned, skip to step 2 (the self-updater keeps it current).
macOS ships `git` (you may be prompted to install Command Line Tools first).

```bash
cd ~
git clone https://github.com/robocoppa/fleet-watchdog.git
cd fleet-watchdog
git fetch --tags --force origin
git checkout --detach released
```

No `/usr/local/bin`, no `sudo`, no Apple-Silicon `mkdir` dance — the scripts
run from the checkout in your home dir.

### 2. Create the per-bot wrapper (once per bot)

It holds this bot's config (including `BOT_TOKEN`), so it lives in your home
dir, `chmod 700`. **Point its final `exec` at `heartbeat-loop.sh`** (the
keep-alive loop), using an **absolute path** — see the warning below.

```bash
mkdir -p ~/.fleet-heartbeat
cp ~/fleet-watchdog/heartbeat/cron-wrapper.sh ~/.fleet-heartbeat/brigitte.sh
chmod 700 ~/.fleet-heartbeat/brigitte.sh   # owner-only — it holds a token
nano ~/.fleet-heartbeat/brigitte.sh
```

Set the exports (`BOT_ID` must match `registry.yaml`); the final line execs
the loop from the checkout:

```bash
export BOT_ID="brigitte"
export BOT_TOKEN="<the MONITORED bot's own token>"
export PROBE_CHAT="<shared probe channel -100… id>"
export WATCHDOG_URL="http://192.168.1.11:9099/beat"
export HOST_LABEL="brigitte-mac"
# export BEAT_INTERVAL="300"   # seconds between beats (default 300)
# export BACKEND_URL="http://192.168.1.11:11434/api/version"

exec "/Users/brigitte/fleet-watchdog/heartbeat/heartbeat-loop.sh"
```

> **Use an ABSOLUTE path in the `exec`, never `$HOME`.** launchd execs the
> wrapper in a stripped environment where `$HOME` is often **unset** — so
> `exec "$HOME/fleet-watchdog/..."` resolves to `/fleet-watchdog/...`, the
> exec fails, and launchd reports **exit 78** with an **empty `.err`**
> (the script never ran, so it logged nothing). This exact trap produced a
> false "DOWN" page. Hardcode `/Users/<you>/...`.

### 3. Test by hand, then check the hub

The loop runs forever, so test it in the foreground briefly and Ctrl-C once
you've seen a beat (or test the one-shot `heartbeat.sh` directly):

```bash
~/fleet-watchdog/heartbeat/heartbeat.sh; echo "exit: $?"   # one beat, then exits
curl -s http://192.168.1.11:9099/status   # brigitte now present, send_ok:true
```

`exit: 0` + a `🩺 brigitte@…` message in the probe channel = working.
Remember: the bot showing up in `/status` is the real proof, not `exit: 0`
(the script always exits 0).

### 4. Schedule it as a KeepAlive LaunchAgent (once per bot)

Use the **keep-alive** plist (not the `StartInterval` one). Copy it, edit the
absolute wrapper path in `ProgramArguments`, validate, and bootstrap:

```bash
cp ~/fleet-watchdog/heartbeat/com.fleet-watchdog.heartbeat-keepalive.plist.example \
   ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
nano ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
plutil -lint ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fleet-watchdog.brigitte.plist
```

In the plist, set `ProgramArguments` to the absolute wrapper path
(`/Users/brigitte/.fleet-heartbeat/brigitte.sh`) and the `Label` to
`com.fleet-watchdog.brigitte`. The plist has `KeepAlive=true` and
**no `StartInterval`** — launchd keeps the loop process alive; the loop
self-paces.

> **Use `bootstrap`/`bootout`, not `load`/`unload`.** The legacy
> `launchctl load` throws an opaque `Load failed: 5: Input/output error`
> when an agent is already registered (and is generally flaky after repeated
> reloads). The modern `launchctl bootstrap gui/$(id -u) <plist>` /
> `launchctl bootout gui/$(id -u)/<label>` pair gives real errors and a
> clean re-register. Run them as **you** (`gui/$(id -u)`), not as root — it's
> a LaunchAgent.

Verify it's a live, resident process (the keep-alive signature):

```bash
launchctl print gui/$(id -u)/com.fleet-watchdog.brigitte | grep -iE 'state|pid'
```

Want `state = running` **and a numeric PID** — that PID is the loop sitting
alive between beats. (A one-shot/interval job shows *no* resident PID; a live
PID is how you know the keep-alive model took.) Watch the loop's log:

```bash
tail -f /tmp/fleet-heartbeat-brigitte.err   # "starting heartbeat loop: every 300s", then quiet
```

To stop it: `launchctl bootout gui/$(id -u)/com.fleet-watchdog.brigitte`.

**Another bot on the same Mac?** Repeat steps 2–4 with a new id — another
wrapper, another `.plist` with a new `Label`. Step 1 is done once.

### 5. Install the self-updater LaunchAgent (once per machine)

A separate keep-alive-free LaunchAgent that periodically converges the
checkout on the `released` tag:

```bash
cp ~/fleet-watchdog/heartbeat/com.fleet-watchdog.update.plist.example \
   ~/Library/LaunchAgents/com.fleet-watchdog.update.plist
nano ~/Library/LaunchAgents/com.fleet-watchdog.update.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fleet-watchdog.update.plist
```

Edit **every** `/Users/brigitte` in that plist to your real home (there are
three: the `ProgramArguments` path and two `FLEET_REPO_DIR` values — launchd
does not expand `~`). Confirm both agents are loaded:

```bash
launchctl list | grep fleet-watchdog   # com.fleet-watchdog.brigitte + .update
```

> **macOS gotchas.** (1) The first run may prompt for a network permission —
> approve it. (2) A keep-alive job pauses while the Mac is *asleep* and
> launchd relaunches it on wake — so a sleeping laptop still goes stale
> (correct: it *is* unreachable). A Mac mini left on beats continuously.
> (3) zsh paste traps: don't paste a line with a trailing `# comment` or a
> `?` — zsh errors with `command not found: #` / `no matches found`. Run each
> command on its own line.

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

**The by-hand test prints `set BOT_ID` (or every var unset), with a
`grep: …: Permission denied` line above it.** The env file is `chmod 600`
(root-only), and `sudo env $(grep …)` runs the `grep` as *your* user — it
can't read the file, so `env` gets nothing. Run the **whole** pipeline as
root instead:

```bash
sudo bash -c 'env $(grep -v "^#" /etc/fleet-heartbeat/<bot>.env | xargs) \
  /home/<user>/fleet-watchdog/heartbeat/heartbeat.sh'; echo "exit: $?"
```

**A scheduled run records exit 78 with an EMPTY `.err` (Mac), but a manual
run works.** Exit 78 is `EX_CONFIG`. Empty `.err` + nonzero exit means the
script *never started* — the failure is before it can log. The classic cause:
the per-bot wrapper's final `exec` uses `$HOME` (e.g.
`exec "$HOME/fleet-watchdog/..."`), but launchd runs the wrapper with `$HOME`
**unset**, so the path resolves to `/fleet-watchdog/...` and the exec fails.
Fix: hardcode an **absolute** path in the wrapper
(`/Users/<you>/fleet-watchdog/heartbeat/heartbeat-loop.sh`). Reproduce the
launchd environment to confirm: `env -i /Users/<you>/.fleet-heartbeat/<bot>.sh;
echo $?` — 78 before the fix, 0 after.

**A Mac bot keeps flapping to `stale` even though it's beating** (probe
messages land in the channel, but at irregular 10–18 min gaps). launchd's
`StartInterval` is being throttled/coalesced — it is *not* a punctual
scheduler. **Switch that bot to the keep-alive loop model** (Step C-Mac):
`heartbeat-loop.sh` + the `KeepAlive` plist, which self-paces with `sleep`.
As a stopgap, widen the bot's `stale_after` in `registry.yaml` to ~1100s.
Confirm the keep-alive model took with
`launchctl print gui/$(id -u)/<label> | grep -iE 'state|pid'` — you want a
live **PID** (a resident loop process), which an interval job never shows.

**`/status` shows the bot with `send_ok:false`.** The bot reached Telegram
(`getme_ok:true`) but its `sendMessage` to the probe channel was rejected.
Read `send_error`:
  - `…not enough rights…` / `chat not found` with `getme_ok:true` → **the
    bot isn't an admin of the probe channel.** Add *that specific bot* as a
    channel admin (each bot must be added individually — another bot posting
    fine doesn't grant this one access).
  - `chat not found` and the bot *is* an admin → wrong `PROBE_CHAT` id in
    its env/wrapper. Fix to the channel's `-100…` id.
  - If `getme_ok:false` too → bad `BOT_TOKEN`.

**`/status` shows `send_ok:true` but you don't see the bot's messages in
the channel.** The probe *succeeded* — it's posting somewhere, just not where
you're looking. Almost always a **wrong `PROBE_CHAT`**: the bot is admin of
some *other* chat and posting there. Compare its id to a working bot's
(`grep PROBE_CHAT …`); they must be identical. Also remember probes are sent
silent (`disable_notification`), so scroll the channel — they won't ping.

**Mac: `launchctl load` fails with `Load failed: 5: Input/output error`.**
The legacy `load`/`unload` API throws this opaque error when an agent is
already registered, and is generally flaky after repeated reloads. Use the
modern pair instead: `launchctl bootout gui/$(id -u)/<label>` (ignore "no such
process") then `launchctl bootstrap gui/$(id -u) <plist>`. Run as **you**
(`gui/$(id -u)`), not root. If `launchctl` commands themselves *hang*, the
user launchd domain is wedged — log out and back in (or reboot) to rebuild it.

**`/status` returns `500 Internal Server Error` with
`OSError: [Errno 116] Stale file handle: '/config/registry.yaml'`.** The
bind-mounted `registry.yaml` (or `state.json`) was *replaced* (its inode
swapped) by a `git pull` / `git checkout` / `cp` while the container held the
old handle. Fix: `docker compose restart fleet-watchdog` to re-open the mount.
Prevent it by editing `registry.yaml` **in place** (`nano`) — and note it's
now gitignored, so git operations no longer touch it.

## Operating it day-to-day

  - **Add a bot:** add its entry to `registry.yaml` on Unraid (no
    restart — re-read each tick), then install the sender on its machine
    (Step C-Linux or C-Mac).
  - **`expected_up` — the alerting switch.**
    - `true` → you get **one** 🔴 when it goes stale and **one** ✅ when
      it returns. No repeats while it stays down. This is the right
      setting for a machine you *want* to know about — including laptops
      shut overnight (you'll get a down each night, an up each morning).
    - `false` → it stays in `/status` but **never** alerts on staleness
      (a machine allowed to come and go silently). Send/token failures
      still alert while it's awake.
  - **Clear stale junk entries** (old test bots, renamed bots) that linger
    in `/status` and sit as false 🔴s: on Unraid,
    `docker compose stop fleet-watchdog && echo '{}' > data/state.json &&
    docker compose up -d fleet-watchdog`. Real bots re-register on their
    next beat; dead ids don't.
  - **Check the fleet any time:** `curl -s http://192.168.1.11:9099/status`.
  - **Tune sensitivity:** per-bot `stale_after` (edit `registry.yaml` **in
    place**), or the global `WATCHDOG_DEFAULT_STALE` / `WATCHDOG_TICK` in
    `.env`. Keep `stale_after` >= 2× the 5-min beat (700s = two missed beats).
    Mac bots on the keep-alive loop model take 700s like Linux; only a Mac
    still on launchd `StartInterval` needs a looser value.
  - **Ship a code change to the whole fleet:** commit, push to `main`, then
    `git tag -f released <commit> && git push -f origin released`. Each host's
    `fleet-update` timer converges within ~15 min — no per-host work. The
    **Unraid hub** is *not* on this timer; update it by hand:
    `cd /mnt/user/appdata/fleet-watchdog && git pull && docker compose up -d
    --build fleet-watchdog` (then mind the stale-handle note if the pull
    touched `registry.yaml` — it won't, it's gitignored).

## Relationship to Audrey

None at the code level — fleet-watchdog is independent. The only
overlap is operational: some monitored bots (OpenClaw, Hermes) talk
to Audrey's passthrough surface (Phase 13) or direct to Ollama, and
their optional `BACKEND_URL` probe can point at Audrey's `/healthz`
or Ollama's `/api/version`. Monitoring those bots' liveness is this
project's job; Audrey neither knows nor cares about the watchdog.
```
