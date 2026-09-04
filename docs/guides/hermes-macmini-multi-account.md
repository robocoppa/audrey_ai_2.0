# Hermes on the Mac mini — one account per agent

Setup for running **two or more Hermes gateways on a single Mac mini**, each in
its own macOS account, each writing to its own home directory, with no agent
able to read another's config, secrets, state or logs.

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
Homebrew is shared at `/opt/homebrew` (Apple Silicon) and is fine to share — it
is the per-agent `~/.hermes` that must not be.

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
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin

pidfile="$HOME/.hermes/gateway.pid"

trap 'hermes gateway stop; exit 0' TERM INT

hermes gateway start || exit 1

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
sudo -u hermes1 -H hermes gateway list
sudo -u hermes2 -H hermes gateway list
```

Each must list exactly one running profile. If either lists both, `HOME` is
wrong somewhere.

## 11. Add agent N

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

## 12. How many fit on the mini

The model does not run here. Hermes on the mini is an agent loop and a Telegram
adapter; inference happens on the Unraid box via Audrey. So per-agent memory is
a Python process plus SQLite — hundreds of MB, not gigabytes.

Budget ~4 GB for macOS, then divide. On a 16 GB mini that is comfortably a
dozen agents before RAM is the constraint; well before that you will hit
Telegram bot limits, Ollama concurrency on the Unraid box, or your own ability
to tell them apart. Isolation is not the thing that caps this.

Measure a real one rather than trusting the estimate:

```bash
sudo -u hermes1 -H hermes gateway status
ps -axo user,pid,rss,command | grep -E "hermes_cli" | grep -v grep
```

Divide free RAM by that RSS. `footprint -p <pid>` gives the honest number if you
want it — plain `rss` double-counts pages shared between the agents.

## 13. What this boundary does not stop

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

## 14. Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Agents gone after reboot | Hermes's own LaunchAgent, no login session | LaunchDaemon with `UserName`, per step 8 |
| Daemon restarts every few seconds | `KeepAlive` pointed at the forking `gateway start` | Use the supervisor wrapper from step 7 |
| Files come out 644 | `Umask` written as octal in a decimal field | `63`, and `umask 077` in the wrapper too |
| Hermes cannot find its config | `HOME` unset for system daemons | Export in wrapper and `EnvironmentVariables` |
| Empty daemon logs | Log files root-owned from first load | Pre-create as the agent user |
| Silent file-read failures, no dialog | TCC blocking a daemon with no UI to prompt | Keep agent work out of Documents, Desktop, Downloads and external volumes |
| `Load failed: 5: Input/output error` | Plist not root-owned or group-writable | `chown root:wheel`, `chmod 644` |
| `gateway start` refuses, shows not-running | Stale PID, process still alive | `kill -TERM` the PID from `gateway.pid`, then start — see `hermes-reference.md` |
| Telegram updates vanishing at random | Two gateways on one bot token | One token per agent |

## 15. Teardown

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

## Open items to verify on the mini

- Does `hermes gateway start` re-create
  `~/Library/LaunchAgents/ai.hermes.gateway-default.plist` on every invocation?
- Does Hermes offer a foreground / no-daemonize gateway mode? If so it replaces
  the supervisor wrapper in step 7 with a plain `exec`, which is cleaner.
- Does `hermes gateway stop` reliably release `gateway.lock` when signalled from
  the wrapper's `trap`?

Append answers to [`hermes-reference.md`](hermes-reference.md), per its
maintenance note.
