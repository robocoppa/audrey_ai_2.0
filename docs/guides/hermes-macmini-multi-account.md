# Hermes on the Mac mini — one account per agent

Setup for running **two or more Hermes gateways on a single Mac mini**, each in
its own macOS account, each writing to its own home directory, with no agent
able to read another's config, secrets, state or logs.

**Design premise: the bots are independent and belong to different people.**
There is no shared directory and no shared group — each agent is a separate
tenant, and the account boundary is a privacy boundary between their operators,
not just tidiness. Everything below follows from that. If a shared workspace is
ever wanted, the appendix has the recipe; don't open one before then.

Companion to [`hermes-reference.md`](hermes-reference.md) (Hermes behaviour and
config schema) and bound by
[`../reference/macos-command-best-practices.md`](../reference/macos-command-best-practices.md)
— every command below runs on the **Mac mini**, is zsh-safe, and uses BSD tool
flags.

## 1. Why accounts and not Hermes profiles

Hermes already supports multiple gateways in one home via **profiles**:

| | Profiles | Separate accounts |
|---|---|---|
| Config | `~/.hermes/profiles/<name>/` | `~/.hermes/` per user |
| PID file | `~/.hermes/profiles/<name>/gateway.pid` | `~/.hermes/gateway.pid` |
| Logs | `~/.hermes/profiles/<name>/logs/` | `~/.hermes/logs/` |
| Runs as | **one shared UID** | one UID each |
| Can read the others | **yes, entirely** | no |

Profiles separate *configuration*. They do not separate *access*. Every profile
runs as the same user, out of the same `~/.hermes`, and shares
`.env`, `state.db`, `auth.json` and `google_token.json` as siblings in one
readable tree.

That matters more here than it would for most software, because the Hermes core
toolset is **47 tools including terminal and file tools**
(`_HERMES_CORE_TOOLS`, see `hermes-reference.md`). An agent that can run shell
commands can read anything its UID can read. Under profiles that is every other
profile's secrets. Under separate accounts the kernel stops it at the home
directory.

So: **one macOS account per agent, each running the default profile.** You get
real containment, and the simpler unprofiled paths (`~/.hermes/config.yaml`,
`~/.hermes/gateway.pid`) rather than the `profiles/<name>/` variants.

## 2. What each agent owns

Everything Hermes writes lives under `~/.hermes` — which is exactly why a
per-account home is the whole boundary:

- `config.yaml` — the live config, plus `config.yaml.bak.*` auto-snapshots
- `.env` — provider keys and the Telegram bot token
- `state.db` + WAL — sessions, history, kanban, memories (SQLite)
- `sessions/`, `memories/`, `skills/`, `cron/`, `sandboxes/`, `logs/`
- `gateway.pid`, `gateway.lock`, `gateway_state.json`
- `channel_directory.json`, `kanban.db`, `auth.json`, `google_token.json`

Two agents sharing this directory would contend on one SQLite file and one
gateway lock. Separate homes are the fix, not a hardening extra.

> **Each agent needs its own Telegram bot token.** Telegram permits one
> long-polling consumer per bot; two gateways on the same token will steal
> updates from each other. One bot, one token, one account.

## 3. Create the service account

Group first, so the user can be created with the right primary group in one
shot. UIDs in the 600s stay clear of the 501+ range macOS gives real people.

```bash
sudo dseditgroup -o create -i 601 -r "Hermes Agent 1" hermes1
```

```bash
sudo sysadminctl -addUser hermes1 \
  -fullName "Hermes Agent 1" \
  -UID 601 -GID 601 \
  -shell /bin/zsh \
  -home /Users/hermes1 \
  -password -
```

The trailing `-password -` prompts interactively rather than leaving a password
in shell history. The account never logs in, so the value only matters as
something you do not reuse.

Hide it from the login window and fast-user-switching menu:

```bash
sudo dscl . -create /Users/hermes1 IsHidden 1
```

Do **not** add these accounts to `admin` or `wheel`. An agent with a terminal
tool that can call `sudo` reads every other agent's `.env`, and the boundary is
decorative.

## 4. Lock the home directory

Do not rely on the default — macOS ships new homes traversable at the top level
and leans on per-folder modes underneath.

```bash
sudo chown -R hermes1:hermes1 /Users/hermes1
sudo chmod -R -N /Users/hermes1
sudo chmod 700 /Users/hermes1
```

Confirm the mode and that no inherited ACL survived:

```bash
ls -led /Users/hermes1
```

Expect `drwx------`, owner `hermes1`, and nothing listed beneath it.

## 5. Install Hermes into the account

Run the install as the agent user so the checkout and venv land in its own home.

Homebrew at `/opt/homebrew` (Apple Silicon) is shared and fine to share — but
note the asymmetry: the agent accounts can **run** brew-installed binaries,
because the prefix is world-readable and executable. They cannot **install**
into it, because `/opt/homebrew` is owned by the admin account that set Homebrew
up. Any shared system dependency is your job, once, from your own account. It is
the per-agent `~/.hermes` that must not be shared.

```bash
sudo -u hermes1 -H /bin/zsh -l
```

Then, inside that shell, run the normal Hermes install and confirm where it
believes its config lives:

```bash
hermes config path
```

It must print paths under `/Users/hermes1/.hermes`. If it prints another
account's home, `HOME` was not set — exit and re-enter with `sudo -u hermes1 -H`.

Tighten the secrets file once it exists:

```bash
chmod 600 /Users/hermes1/.hermes/.env
```

## 6. Point the agent at Audrey

Per-account `~/.hermes/config.yaml`, same shape as the laptop install — the
provider `base_url` is what crosses the wire to Audrey, which fronts Ollama on
the Unraid box:

```yaml
providers:
  audrey:
    base_url: http://192.168.1.11:8000/v1
    api_mode: openai-completions
    request_timeout_seconds: 600

streaming:
  enabled: false
```

Keep `streaming.enabled: false` — see the 2026-05-29 resolution in
`hermes-reference.md`. Per-agent secrets (bot token, provider keys) go in that
account's `~/.hermes/.env` at mode 600.

**MCP servers do not collide between agents.** The `bot-tools` server is a
remote HTTP endpoint (`http://192.168.1.11:9110/mcp`), so nothing binds a local
port and two agents can point at it simultaneously. What separates them there is
the `Authorization: Bearer` header — per `hermes-reference.md`, **the per-bot
token is the identity** the tools server sees. That token lives in the agent's
own `.env`, which is exactly why that file is mode 600 inside a 700 home: it is
not just a credential, it is the thing that stops one bot acting as another.

If you ever switch a server to the stdio form (`command:` + `args:`), the
subprocess is spawned by the gateway and inherits the agent's UID, `HOME` and
umask — so it stays inside the same boundary with no extra work.

## 7. Supervision — the part that needs real care

Two facts collide here.

**Hermes installs its own LaunchAgent** at
`~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist` when you run
`hermes gateway start`.

**A LaunchAgent only runs while its user has a login session.** That is fine
for the laptop, where you are logged in. It does not work for headless service
accounts on the mini: nothing bootstraps a per-user launchd domain for an
account that never logs in, so after a reboot neither agent comes back.

> This is a deliberate departure from
> `macos-command-best-practices.md`, which makes LaunchAgent the default and
> reserves LaunchDaemons for system services. The justification is narrow and
> specific: **headless accounts with no login session**. A LaunchDaemon with a
> `UserName` key still runs unprivileged as that agent — it is not a root
> service, it just lives in a domain that exists at boot.

**The second fact:** the Hermes gateway **daemonizes**. `hermes gateway start`
forks, writes `gateway.pid` / `gateway.lock`, and returns. Pointing a
`KeepAlive` LaunchDaemon straight at it produces a restart loop — launchd sees
the starter exit immediately and fires it again forever.

So the daemon runs a **supervisor wrapper** that starts the gateway, then blocks
on its PID.

### `/Users/hermes1/agent-run.sh`

```bash
#!/bin/zsh
umask 077

export HOME=/Users/hermes1
export TMPDIR="$HOME/agent-tmp"
export PATH="$HOME/.hermes/hermes-agent/venv/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

HERMES=$(command -v hermes) || exit 1

pidfile="$HOME/.hermes/gateway.pid"

trap '"$HERMES" gateway stop; exit 0' TERM INT

"$HERMES" gateway start || exit 1

i=0
while [ $i -lt 30 ]; do
  [ -s "$pidfile" ] && break
  sleep 1
  i=$((i + 1))
done

[ -s "$pidfile" ] || exit 1
pid=$(cat "$pidfile")

while kill -0 "$pid" 2>/dev/null; do
  sleep 10
done

exit 1
```

**The `PATH` line is the one to get right.** Hermes lives in a venv inside the
account's own home (`~/.hermes/hermes-agent/`, per `hermes-reference.md`), not
in `/opt/homebrew/bin`. A daemon `PATH` of just the system directories will not
find `hermes`, and the failure is quiet and ugly: the wrapper exits non-zero,
`KeepAlive` restarts it, and you get a 30-second crash loop with nothing in the
log but `command not found`. Resolving `HERMES` with `command -v` and failing
loudly is what keeps that debuggable.

Confirm the resolved path before you trust the wrapper:

```bash
sudo -u hermes1 -H /bin/zsh -lc 'command -v hermes'
```

Exiting non-zero when the gateway dies is what hands the restart back to
launchd. The `trap` matters: on `bootout`, launchd signals the wrapper, and
without it the detached gateway would survive as an orphan holding the lock.

```bash
sudo chown hermes1:hermes1 /Users/hermes1/agent-run.sh
sudo chmod 700 /Users/hermes1/agent-run.sh
sudo -u hermes1 mkdir -p /Users/hermes1/agent-tmp
sudo -u hermes1 chmod 700 /Users/hermes1/agent-tmp
```

Then remove Hermes's own LaunchAgent so the two supervisors do not both try to
own the gateway:

```bash
sudo rm -f /Users/hermes1/Library/LaunchAgents/ai.hermes.gateway-default.plist
```

> **Verify on the box:** whether `hermes gateway start` *re-creates* that plist
> on every invocation is not documented. Check after the first daemon start; if
> it comes back, the wrapper needs an `rm` after the start line. Append the
> finding to `hermes-reference.md`.

## 8. The LaunchDaemon

One plist per agent in `/Library/LaunchDaemons`. `UserName` and `GroupName` are
what make a root-loaded daemon run as the unprivileged agent. launchd does not
expand `~`, so every path is absolute.

### `/Library/LaunchDaemons/ai.hermes.agent1.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.agent1</string>

    <key>UserName</key>
    <string>hermes1</string>
    <key>GroupName</key>
    <string>hermes1</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/hermes1/agent-run.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/hermes1</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Users/hermes1</string>
    </dict>

    <key>Umask</key>
    <integer>63</integer>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>/Users/hermes1/agent-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/hermes1/agent-daemon.err</string>
</dict>
</plist>
```

Three keys that bite if you get them wrong:

- **`Umask` is decimal.** launchd passes the integer straight to `umask(2)` and
  plist integers are decimal. Octal `077` is **63**. Writing `77` gives you
  octal 115 — group and other keep read on directories.
- **`HOME` is not set for system daemons.** Without it Hermes will not find
  `~/.hermes` and may scatter files into `/`. Set it here *and* in the wrapper.
- **`ThrottleInterval`** keeps a crash-looping gateway from hammering the box
  every 10 seconds, which is the launchd default.

Pre-create the daemon log files as the agent user. launchd opens
`StandardOutPath` before it drops privileges, so if they do not exist they can
end up root-owned and the agent's writes fail:

```bash
sudo -u hermes1 /bin/sh -c 'umask 077; : > /Users/hermes1/agent-daemon.log; : > /Users/hermes1/agent-daemon.err'
```

The gateway's own logs stay where Hermes puts them, at
`/Users/hermes1/.hermes/logs/gateway.log`.

### Rotation

`agent-daemon.log` and `.err` grow without bound — a crash-looping gateway can
fill a 256 GB SSD faster than you would expect. macOS rotates with `newsyslog`,
not logrotate. Create `/etc/newsyslog.d/hermes.conf`:

```text
/Users/hermes1/agent-daemon.log hermes1:hermes1 600 5 1024 * J
/Users/hermes1/agent-daemon.err hermes1:hermes1 600 5 1024 * J
```

Five generations, rotate at 1 MB, bzip2 the old ones, recreated 600 and owned by
the agent so the daemon keeps writing after a rotation.

## 9. Load and start

launchd refuses any plist in `/Library/LaunchDaemons` that is not root-owned and
non-writable by group or other:

```bash
sudo chown root:wheel /Library/LaunchDaemons/ai.hermes.agent1.plist
sudo chmod 644 /Library/LaunchDaemons/ai.hermes.agent1.plist
```

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.hermes.agent1.plist
sudo launchctl enable system/ai.hermes.agent1
```

| Task | Command |
|---|---|
| State and last exit code | `sudo launchctl print system/ai.hermes.agent1` |
| Restart after editing the wrapper | `sudo launchctl kickstart -k system/ai.hermes.agent1` |
| Stop and unload | `sudo launchctl bootout system/ai.hermes.agent1` |
| Reload after editing the plist | `bootout` then `bootstrap` again |

`bootstrap` / `bootout` / `kickstart` are the current subcommands. `launchctl
load` and `unload` still work but report less and hide failures.

## 10. Prove the isolation

Run these once, after the second agent exists.

The other agent cannot traverse in:

```bash
sudo -u hermes2 ls /Users/hermes1
sudo -u hermes2 cat /Users/hermes1/.hermes/.env
```

Both must fail with `Permission denied`. Anything else means the home is not at
700, or an ACL survived step 4.

New files are born private:

```bash
sudo launchctl kickstart -k system/ai.hermes.agent1
ls -l /Users/hermes1/.hermes
```

Files should read `-rw-------` and directories `drwx------`.

Each gateway runs as its own user, with its own PID:

```bash
ps -axo user,pid,ppid,command | grep -E "hermes_cli|gateway" | grep -v grep
```

Each should appear under its own username. A process still showing `root` means
`UserName` was ignored — usually a typo in the key.

Each agent sees only its own gateway:

```bash
sudo -u hermes1 -H /bin/zsh -lc 'hermes gateway list'
sudo -u hermes2 -H /bin/zsh -lc 'hermes gateway list'
```

Each must list exactly one running profile. If either lists both, `HOME` is
wrong somewhere.

## 11. What the agents can and cannot do

The boundary is only correct if it still lets the bots work. Concretely, with
the setup above and **no admin rights at all**:

**They can, with no further grants:**

- Run the gateway and the full 47-tool core toolset — terminal, file, web,
  vision — anywhere inside their own home
- Read, write and execute anything under `/Users/hermesN`, including
  `sandboxes/`, `skills/` and `cron/`
- Reach the network outbound: Telegram, Audrey on the Unraid box, the
  `bot-tools` MCP endpoint
- Spawn stdio MCP subprocesses, which inherit the agent's UID and stay contained
- `pip install` into their own venv at `~/.hermes/hermes-agent/venv`
- Bind unprivileged ports above 1024
- Run any binary already installed in `/opt/homebrew` or the system

**They cannot, by design:**

| Blocked | Why | Deliberate? |
|---|---|---|
| Read another agent's home, `.env`, `state.db` | Home is 700 | **Yes — the point** |
| Read your personal home | Same | Yes |
| `brew install`, write `/opt/homebrew` | Prefix owned by your admin account | Yes |
| Write `/usr/local/bin`, `/Library`, `/etc` | Root-owned | Yes |
| Bind ports below 1024 | Requires root | Yes |
| Touch Documents, Desktop, Downloads, external volumes | TCC, with no UI to prompt | Yes, but see below |
| `sudo` anything | Not in `admin` or `wheel` | Yes |

The only row worth a second look is TCC. A headless daemon gets **no permission
dialog** — it just fails silently. If an agent genuinely needs one of those
locations, grant Full Disk Access to the resolved `hermes` binary in System
Settings → Privacy & Security once, from your admin account. Better: keep the
agent's work inside its own home and avoid the whole subsystem, which is what
the layout here does.

## 12. Independent bots, different operators

Each bot belongs to a different person. That turns the account boundary from
tidiness into a **privacy boundary**, and it changes what is worth checking.

### What the boundary is now protecting

| File | Contains |
|---|---|
| `state.db` + WAL | One operator's sessions, history, memories, kanban |
| `channel_directory.json` | Their registered Telegram chats |
| `.env` | The bot token that **is** their bot's identity |
| `sessions/`, `memories/` | Conversation content |

Every one of those is another person's data. The 700 home is what keeps it from
the other operators, and it is the whole reason this is worth doing rather than
running Hermes profiles in one account.

### The toolset is the real perimeter

This is the item to get right before anyone else touches these bots.

Per `hermes-reference.md`, the Telegram channel's tools come from
`platform_toolsets:`. On the laptop install the config said `[hermes-telegram]`
but **the observed behaviour was the channel inheriting `hermes-cli`** — the
47-tool core set, terminal included.

If that happens here, **whoever can DM the bot can run shell commands on the Mac
mini as that bot's UID.** The account boundary does its job — the blast radius
stops at that one agent's home, which is exactly why it exists — but it is still
a shell handed to whoever finds the bot.

So decide per bot, explicitly, rather than inheriting:

```yaml
platform_toolsets:
  telegram: []
```

for pure chat, or a deliberate minimum:

```yaml
platform_toolsets:
  telegram: [clarify, web_search]
```

**Do not trust the config alone** — that install proved config and behaviour can
diverge. Verify from the operator's side by asking the bot, over Telegram, to
read a file or run a command, and confirm it cannot. Then check the agent's own
logs to see whether a tool call was attempted:

```bash
sudo -u hermes1 -H /bin/zsh -lc 'tail -n 50 ~/.hermes/logs/gateway.log'
```

### Rules that follow from multi-tenancy

- **One person, one bot, one token, one account.** Never two people on one bot —
  they would share `state.db`, and each would see the other's history.
- **Operators do not get the mini's admin account.** Admin reads every agent's
  home, which is every other operator's conversations.
- **Telegram bots answer anyone who finds them** unless restricted. Whatever
  toolset you grant is granted to strangers too, so the `platform_toolsets`
  decision above is what actually bounds this until an allowlist is confirmed.
- **Say what you are keeping.** The mini's admin (you) can read every home. If
  operators expect their conversations to be private from you, that is a
  conversation to have, not an assumption to leave standing.

### Operational independence is already there

Separate LaunchDaemons mean one bot crashing, wedging or restarting does not
touch the others — each has its own `KeepAlive` and its own `ThrottleInterval`.
Contention for the model happens at Ollama on the Unraid box, not on the mini,
so a heavy turn from one operator slows the shared model rather than the other
agents' processes.

## 13. Do you need an agent running as admin?

**No. Don't.** Give none of them admin, and do the privileged work yourself.

Three reasons, in order of how much they should worry you:

1. **Hermes has a terminal tool.** An agent in `admin` is an LLM with `sudo`.
   Everything reaching it — a Telegram message, a fetched web page, a tool
   result — is untrusted input that can steer it. Prompt injection stops being
   an annoyance and becomes arbitrary root execution.
2. **It collapses the boundary you just built.** root reads every other agent's
   `.env` and `state.db`. One admin agent means zero isolated agents.
3. **Those tokens are identity.** The per-bot MCP token is how `bot-tools`
   tells the bots apart. An agent that can read the others' `.env` files can
   impersonate every one of them to the tools server.

### What that actually costs you in time

The privileged work is front-loaded and rare, not ongoing:

| Task | When | Time |
|---|---|---|
| Create account, lock home | Per new agent | ~2 min, scripted in §14 |
| Install Hermes into the account | Per new agent | ~5 min |
| Write and bootstrap the LaunchDaemon | Per new agent | ~2 min, copy and substitute |
| `brew install` a shared dependency | Rare | ~1 min |
| Grant Full Disk Access | Rare, GUI | ~1 min |

One-time setup is well under an hour. After that a new agent is a few minutes,
and steady-state operation needs **no admin at all** — the bots run, restart and
recover entirely on their own. There is no recurring tax to avoid here, which is
what makes "just don't give them admin" the cheap answer as well as the safe one.

### If one privileged action really does recur

Do not reach for `admin`. Add a single narrowly-scoped sudoers rule for exactly
that command, via `sudo visudo -f /etc/sudoers.d/hermes`:

```text
hermes1 ALL=(root) NOPASSWD: /usr/sbin/mydiagnostic --readonly
```

Rules that hold, and the ways this goes wrong:

- **No wildcards.** `/usr/bin/foo *` lets the agent pass any argument, including
  ones that read or write arbitrary files.
- **Nothing that can spawn a shell.** Never grant an editor, `find`, `awk`,
  `tar`, `git`, or any interpreter — each has a documented shell escape.
- **Never `ALL`.** `hermes1 ALL=(root) NOPASSWD: ALL` is admin with extra steps.
- Absolute paths only, and the target binary must not be writable by the agent —
  otherwise it just rewrites the thing it is allowed to run as root.

If the privileged action is complex enough that a safe sudoers line is hard to
write, that is the signal to invert it: have the agent drop a request file in
the shared directory and run a small root-owned LaunchDaemon that validates and
executes it. The agent never gets privilege; it gets an API.

## 14. Add agent N

`/Users/Shared/provision-hermes-agent.sh`, run as `provision-hermes-agent.sh 2`:

```bash
#!/bin/zsh
set -euo pipefail

n="$1"
user="hermes${n}"
uid=$((600 + n))
home="/Users/${user}"

sudo dseditgroup -o create -i "$uid" -r "Hermes Agent ${n}" "$user"

sudo sysadminctl -addUser "$user" \
  -fullName "Hermes Agent ${n}" \
  -UID "$uid" -GID "$uid" \
  -shell /bin/zsh -home "$home" -password -

sudo dscl . -create "/Users/${user}" IsHidden 1

sudo chown -R "${user}:${user}" "$home"
sudo chmod -R -N "$home"
sudo chmod 700 "$home"

sudo -u "$user" /bin/sh -c "umask 077; \
  mkdir -p ${home}/agent-tmp; \
  : > ${home}/agent-daemon.log; \
  : > ${home}/agent-daemon.err"

echo "provisioned ${user} (uid ${uid}) at ${home}"
```

Then per agent: install Hermes in the account, copy the wrapper and plist with
the user name and `Label` substituted, and bootstrap. **One label per agent** —
two daemons sharing a `Label` means the second silently never loads.

## 15. How many fit on the mini

The model does not run here. Hermes on the mini is an agent loop and a Telegram
adapter; inference happens on the Unraid box via Audrey. So per-agent memory is
a Python process plus SQLite — hundreds of MB, not gigabytes.

Budget ~4 GB for macOS, then divide. On a 16 GB mini that is comfortably a
dozen agents before RAM is the constraint; well before that you will hit
Telegram bot limits, Ollama concurrency on the Unraid box, or your own ability
to tell them apart. Isolation is not the thing that caps this.

Measure a real one rather than trusting the estimate:

```bash
sudo -u hermes1 -H /bin/zsh -lc 'hermes gateway status'
ps -axo user,pid,rss,command | grep -E "hermes_cli" | grep -v grep
```

Divide free RAM by that RSS. `footprint -p <pid>` gives the honest number if you
want it — plain `rss` double-counts pages shared between the agents.

## 16. What this boundary does not stop

| Threat | Held? | Detail |
|---|---|---|
| Agent reads another's `.env` or `state.db` | **Held** | Kernel-enforced at the 700 home |
| Agent leaks into shared `/tmp` | **Held** | umask 077 plus a private `TMPDIR` |
| Agent escalates to root | Open | root reads everything. Never grant `admin` |
| Agent reaches another's gateway over the network | Open | Shared loopback and LAN |
| Agent reads another's command line | Open | Process table is world-readable — never pass secrets in argv |
| Kernel-level compromise | Open | One shared kernel; only VMs change this |

The Hermes terminal tool makes the first row load-bearing rather than
theoretical. If the threat model is an agent that might be actively adversarial
rather than merely buggy, accounts are the wrong tier — give each one a VM and
treat the mini as a hypervisor host.

## 17. Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Agents gone after reboot | Hermes's own LaunchAgent, no login session | LaunchDaemon with `UserName`, per step 8 |
| Daemon restarts every few seconds | `KeepAlive` pointed at the forking `gateway start` | Use the supervisor wrapper from step 7 |
| `command not found: hermes`, 30s restart loop | Daemon `PATH` missing the venv in `~/.hermes/hermes-agent/` | Resolve with `command -v` in the wrapper, per step 7 |
| Shared-dir files unreadable by the other agent | `umask 077` strips the group bits setgid would have given | Inherited ACL on the directory, see the appendix |
| Agent cannot `brew install` | `/opt/homebrew` is owned by your admin account | Install shared deps yourself, per step 13 |
| Files come out 644 | `Umask` written as octal in a decimal field | `63`, and `umask 077` in the wrapper too |
| Hermes cannot find its config | `HOME` unset for system daemons | Export in wrapper and `EnvironmentVariables` |
| Empty daemon logs | Log files root-owned from first load | Pre-create as the agent user |
| Silent file-read failures, no dialog | TCC blocking a daemon with no UI to prompt | Keep agent work out of Documents, Desktop, Downloads and external volumes |
| `Load failed: 5: Input/output error` | Plist not root-owned or group-writable | `chown root:wheel`, `chmod 644` |
| `gateway start` refuses, shows not-running | Stale PID, process still alive | `kill -TERM` the PID from `gateway.pid`, then start — see `hermes-reference.md` |
| Telegram updates vanishing at random | Two gateways on one bot token | One token per agent |
| Operator can run shell commands via the bot | Telegram channel inherited `hermes-cli` | Set `platform_toolsets.telegram` explicitly, per step 12 |

## 18. Teardown

Unload before deleting the account, or you leave a daemon referencing a UID that
no longer resolves.

```bash
sudo launchctl bootout system/ai.hermes.agent1
sudo rm /Library/LaunchDaemons/ai.hermes.agent1.plist
sudo sysadminctl -deleteUser hermes1
sudo dseditgroup -o delete hermes1
```

`sysadminctl -deleteUser` removes the home directory with the account. Add
`-keepHome` to keep the agent's `~/.hermes` state.

## Appendix — if you later want a shared workspace

**Not part of the current design.** The bots are deliberately independent and
belong to different people, so there is no shared directory and no shared group.
Keep it that way until there is a concrete reason not to.

If that reason arrives, this is the recipe. Separate homes mean the agents
cannot hand each other files, so a common drop has to be made **explicit and
narrow** rather than by loosening the homes.

```bash
sudo dseditgroup -o create -i 650 -r "Hermes Shared" hermes-shared
sudo dseditgroup -o edit -a hermes1 -t user hermes-shared
sudo dseditgroup -o edit -a hermes2 -t user hermes-shared
```

```bash
sudo mkdir -p /Users/Shared/hermes-work
sudo chgrp hermes-shared /Users/Shared/hermes-work
sudo chmod 2770 /Users/Shared/hermes-work
```

The setgid bit (`2770`) makes new entries inherit the group — **and on its own
that is not enough here.** The agents run with `umask 077`, which strips the
group bits off every file they create, so the other agent still cannot read
them. The fix is an inherited ACL, which is applied at creation time and is not
subject to the umask:

```bash
sudo chmod +a "group:hermes-shared allow list,search,add_file,add_subdirectory,delete_child,readattr,writeattr,readextattr,writeextattr,file_inherit,directory_inherit" /Users/Shared/hermes-work
```

Verify the ACL landed, then confirm a file written by one agent is readable by
the other:

```bash
ls -lde /Users/Shared/hermes-work
sudo -u hermes1 /bin/sh -c ': > /Users/Shared/hermes-work/probe'
sudo -u hermes2 cat /Users/Shared/hermes-work/probe
```

Group membership is read at process start, so bounce the daemons after adding
an agent to the group:

```bash
sudo launchctl kickstart -k system/ai.hermes.agent1
```

> Anything in this directory is readable by every agent in the group — and
> therefore by every *operator* of those bots. It is a deliberate hole in the
> boundary. Never put per-agent secrets or tokens there, and think twice before
> opening it at all while different people own different bots.

## Open items to verify on the mini

- Does `hermes gateway start` re-create
  `~/Library/LaunchAgents/ai.hermes.gateway-default.plist` on every invocation?
- Does Hermes offer a foreground / no-daemonize gateway mode? If so it replaces
  the supervisor wrapper in step 7 with a plain `exec`, which is cleaner.
- Does `hermes gateway stop` reliably release `gateway.lock` when signalled from
  the wrapper's `trap`?
- **Does Hermes support a per-bot Telegram allowlist** (permitted user or chat
  IDs)? The reference does not document one, and `channel_directory.json` is
  only guessed at. Until this is answered, `platform_toolsets.telegram` is the
  only verified control over what a stranger who finds the bot can make it do.
- Does `platform_toolsets.telegram` reliably override the global `toolsets:` on
  this install, or does it inherit as it did on the laptop?

Append answers to [`hermes-reference.md`](hermes-reference.md), per its
maintenance note.
