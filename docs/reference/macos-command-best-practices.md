# macOS command best practices

Reference for handing the user commands that run on a **Mac** — Mac mini or
MacBook bot hosts (e.g. brigitte). macOS is Unix, but it is **not Linux**:
the default shell is **zsh**, the core tools are **BSD** (not GNU), and the
filesystem layout differs (especially on Apple Silicon). A command that's
correct on Linux can silently misbehave or error here.

Read this before giving a Mac user a shell command. The general scripting
rules in [bash/Linux best practices](bash-linux-best-practices.md) still
apply; this doc covers what's *different*.

## The shell is zsh — interactive paste traps

macOS defaults to **zsh**. The traps below hit when the user pastes a
command interactively (they don't apply inside a `#!/usr/bin/env bash`
script, but they do apply to pasted one-liners — which is most of what we
hand over).

- **`?` triggers globbing.** A pasted `grep foo # is it loaded?` errors with
  `zsh: no matches found: loaded?` — zsh tries to glob `loaded?`. **Never
  put a `?` in a trailing comment** on a pasted line.
- **`#` is not stripped in interactive zsh by default.** A pasted
  `cat file   # note` runs `cat` against `file`, `#`, and each word of the
  note → `cat: #: No such file or directory`. **No inline `#` comments on
  pasted command lines** (this is also in the Linux doc; it's worse here).
- **`~` is expanded by the shell but NOT by everything.** Interactive zsh
  expands `~`, but **launchd plists, config files, and many programs do
  not** — use absolute paths (`/Users/<you>/…`) in any plist/config.
- **Consequence for our docs:** put explanations in prose above the block;
  give each command its own line; no trailing annotations. We hit every one
  of these during the brigitte deploy.

> Bash exists on macOS but it's **3.2** (frozen for licensing). Don't rely
> on bash 4+/5 features (`declare -A` assoc arrays, `${var^^}`, `mapfile`)
> in anything that must run with the system bash. `#!/usr/bin/env bash`
> picks up a Homebrew bash if installed; the system one if not.

## BSD vs GNU tools — the flags that differ

macOS ships BSD versions of the core utilities. Same names, different flags.
These are the ones that actually break scripts:

| Task | Linux (GNU) | macOS (BSD) |
|---|---|---|
| in-place sed | `sed -i 's/a/b/' f` | `sed -i '' 's/a/b/' f` (mandatory backup-ext arg, empty string for none) |
| PCRE grep | `grep -P '\d+'` | not supported — use `grep -E '[0-9]+'` |
| date from string | `date -d "2023-10-01" +%s` | `date -j -f "%Y-%m-%d" "2023-10-01" +%s` |
| relative date | `date -d "yesterday"` | `date -v-1d` |
| file size | `stat -c %s f` | `stat -f %z f` |
| canonical path | `readlink -f f` | no `-f` — use `realpath` (may need brew) or `cd`+`pwd` |
| find | `find -name '*.sh' .` (lenient) | `find . -name '*.sh'` (path **must** come first) |
| ps columns | `ps -eo pid,comm` | `ps -ao pid,comm` |
| base64 decode | `base64 -d` | `base64 -D` (capital D) |

Rules of thumb when handing a Mac command:
- **Prefer the portable spelling** that works on both: `grep -E` over
  `grep -P`; a temp-file rewrite over `sed -i`; `find . -name …` (path
  first) which is valid on both.
- **`sed -i` is the single most common breakage.** If you must give a Mac
  `sed -i`, it's `sed -i '' '…'`. Better: avoid in-place edits in handed
  commands; have the user open the file in an editor (`nano`) instead.
- **When unsure a flag is portable, don't guess** — give the BSD form
  explicitly for a Mac, or check.

## Filesystem layout — Apple Silicon especially

- **`/usr/local/bin` does NOT exist by default on Apple Silicon.** Homebrew
  lives in `/opt/homebrew` there, so nothing creates `/usr/local/bin`.
  Copying into it fails with `No such file or directory` until you
  `sudo mkdir -p /usr/local/bin`. (Exactly the brigitte failure.) On Intel
  Macs it exists.
- **`/usr/local/bin` may not be on `PATH`** even when it exists. Doesn't
  matter if you call scripts by absolute path (which schedulers require
  anyway).
- **Homebrew prefix differs by arch:** `/opt/homebrew` (Apple Silicon) vs
  `/usr/local` (Intel). Don't hardcode one; use `$(brew --prefix)` if a
  script needs it.
- **APFS is case-insensitive by default.** `File.txt` and `file.txt` are
  the same path on a stock Mac but different on Linux ext4/xfs. Don't rely
  on case to distinguish files in cross-platform scripts.
- **Files that hold secrets:** put them in the user's home
  (`~/.config/…`, `~/.fleet-heartbeat/…`), owned by the user, `chmod 600`.
  Don't shove token-bearing files into root-owned `/usr/local/bin` — it
  forces `sudo` for every edit and forces root-running schedulers. (The
  fleet-watchdog Mac path puts the shared script in `/usr/local/bin` but
  per-bot token wrappers in `~/.fleet-heartbeat/`.)

## Scheduling: launchd, not systemd/cron

macOS has **no systemd**. Don't hand a Mac `systemctl …` — it doesn't
exist. The native scheduler is **launchd**:

- A **LaunchAgent** (`~/Library/LaunchAgents/*.plist`, runs as the
  logged-in user) is the default choice for user-owned jobs. A
  **LaunchDaemon** (`/Library/LaunchDaemons`, runs as root) is only for
  system services.
- `StartInterval` (integer seconds) is the cron-interval equivalent;
  `RunAtLoad` fires once on load.
- **plists need absolute paths** in `ProgramArguments` — launchd does not
  expand `~`.
- Manage with `launchctl load|unload|list|start`. `launchctl list | grep
  <label>` shows if a job is loaded.
- cron *does* exist on macOS but is deprecated by Apple and may need "Full
  Disk Access" granted; prefer launchd.

## Permissions / first-run prompts

- **TCC prompts.** The first time a job touches the network, files, or
  automation, macOS may pop a permission dialog — a background launchd job
  can appear to "do nothing" until the user approves it once.
- **`sudo` works** the same as Linux (invisible password). Same rule:
  whole pipeline under sudo when reading root-owned files, not just the
  tail command.

## Quick pre-flight before handing a Mac command

1. Any `?` or `#` in the line? → move comments to prose. (zsh)
2. Any GNU-only flag (`sed -i` w/o `''`, `grep -P`, `date -d`,
   `stat -c`, `readlink -f`)? → use the BSD form or a portable one.
3. Writing into `/usr/local/bin`? → precede with `sudo mkdir -p`
   (Apple Silicon) and use `sudo`.
4. Scheduling? → launchd + absolute paths, never `systemctl`.
5. Token-bearing file? → home dir, `chmod 600`, no sudo needed.

## Sources

- [Write Cross-Platform Shell: Linux vs macOS Differences That Break Production](https://tech-champion.com/programming/write-cross-platform-shell-linux-vs-macos-differences-that-break-production/)
- [Differences Between macOS and Linux Scripting — DEV](https://dev.to/aghost7/differences-between-macos-and-linux-scripting-74d)
- [Replacing macOS BSD utils with GNU coreutils](https://devedge.github.io/2026/03/14/replacing-macOS-BSD-utils-with-GNU-coreutils/)
- Plus gotchas verified firsthand during the brigitte (Mac mini, Apple
  Silicon) fleet-watchdog deploy: missing `/usr/local/bin`, zsh `#`/`?`
  paste traps, launchd `~` non-expansion, home-dir token wrappers.
