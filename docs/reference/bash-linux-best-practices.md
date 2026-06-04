# Bash / Linux command + script best practices

Reference for handing the user (or writing) shell commands and scripts that
run on **Linux** — the Unraid host, Linux laptops (claudette, donna), and
any Linux bot machine. macOS has its own rules; see
[macOS command best practices](macos-command-best-practices.md). When a
command must work on *both*, follow the stricter of the two and prefer the
portable forms noted here and in the macOS doc.

This is a "before you give a command" checklist, not a style essay. The
goal is that a command pasted into the user's shell does what we said it
would, the first time.

## Commands handed to the user (one-liners + pastes)

These are the rules that actually bite when the user copy-pastes what we
give them.

- **No inline `# comments` on command lines meant to be pasted.** They are
  a real paste hazard — especially trailing comments on a `git clone`/`cp`
  line, and *especially* in zsh (see the macOS doc). Put the explanation in
  prose above the block, not after the command. If a block has more than
  one command, give each its own fenced block when the user runs them
  one-by-one and reads output between them.
- **One command per line; don't chain with `;` what the user needs to read
  between.** Chaining hides which step failed.
- **Absolute paths in anything a scheduler runs.** cron/systemd/launchd do
  not share your interactive `PATH`. Call scripts by full path.
- **Quote every variable and path that could contain spaces or globs.**
  `"$f"`, not `$f`.
- **When a file is `chmod 600`/root-owned, the WHOLE pipeline needs sudo,
  not just the final command.** `sudo env $(grep … file)` runs `grep` as
  the *user* (permission denied) — wrap it: `sudo bash -c 'env $(grep …)
  …'`. (This bit us live on the fleet-watchdog env files.)
- **Tell the user what success looks like.** "exit: 0 and X appears in
  /status" beats "run this." A command with no expected-output line is
  half a command.
- **`sudo` password prompts are invisible.** Mention it when a `sudo`
  command is the first in a while, so the user isn't confused by a blank
  prompt.

## Script header — strict mode, every time

Start non-trivial scripts with:

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
```

- **`#!/usr/bin/env bash`** — not `#!/bin/bash`. Finds bash on `PATH`;
  portable across distros and macOS (where `/bin/bash` is an ancient 3.2).
- **`set -e` (errexit)** — exit on any unhandled non-zero. For a command
  whose failure is acceptable, append `|| true` explicitly.
- **`set -u` (nounset)** — error on unset variables. For an intentionally
  optional var, write `"${VAR:-default}"` / `"${VAR:-}"`.
- **`set -o pipefail`** — a pipeline fails if *any* stage fails, not just
  the last. Without it, `false | true` succeeds.
- **`IFS=$'\n\t'`** — drop space from the field separator so filenames with
  spaces don't word-split. Optional but cheap insurance.

Caveat on `set -e`: it does **not** fire inside `if`/`while` conditions, `&&`/`||`
chains, or command substitutions used in tests. Don't rely on it as your
only error handling for those; check explicitly.

## Quoting and expansion

- **Always double-quote variable and command-substitution expansions:**
  `"$var"`, `"$(cmd)"`, `"${arr[@]}"`. Unquoted, the shell word-splits and
  glob-expands the result — the #1 source of shell bugs.
- **`"${arr[@]}"`** (quoted) iterates array elements safely; `${arr[*]}`
  joins them — rarely what you want.
- **Brace-protect when concatenating:** `"${name}_suffix"`.

## Conditionals and tests

- **Use `[[ … ]]`, not `[ … ]`/`test`** in bash. It's a keyword (no
  word-splitting inside), supports `&&`/`||`, `=~` regex, and `<`/`>`
  string compares without escaping.
- **Numeric comparison:** `(( a > b ))` or `[[ "$a" -gt "$b" ]]`.
- **Existence:** `[[ -f "$path" ]]`, `[[ -d "$path" ]]`, `[[ -n "$s" ]]`
  / `[[ -z "$s" ]]`.

## Commands and subshells

- **`$(…)`, never backticks.** Nestable, readable, no escaping surprises.
- **`local` for every function variable** — otherwise it's global and leaks
  between functions. `local x; x="$(cmd)"` on two lines if you need the
  exit status of `cmd` (declaring and assigning on one line masks it).
- **Don't parse `ls`.** Glob (`for f in ./*.txt`) or use `find … -print0 |
  while IFS= read -r -d '' f`. `ls` output is for humans.
- **`read -r`** always (raw — don't let backslashes be eaten).

## Structure and cleanup

- **`trap 'cleanup' EXIT`** for temp files / partial state, so they're
  removed on any exit path including errors. Create temps with `mktemp`,
  not hardcoded `/tmp/foo`.
- **A `main` function** called as `main "$@"` at the bottom keeps top-level
  scope clean and makes the script sourceable for tests.
- **Errors to stderr:** `echo "msg" >&2`, and exit non-zero on failure so
  callers (and `set -e` upstream) can detect it.
- **`cd` defensively:** `cd "$(dirname "$0")"` early if the script assumes
  its own directory — but prefer absolute paths over relying on cwd.

## Lint it

- **Run `shellcheck` on any script before shipping it.** It catches the
  unquoted-var, missing-`-r`, `[ ]`-vs-`[[ ]]`, and word-split classes
  automatically. Treat its warnings as findings to resolve, not noise.
- For scripts in this repo's orbit (e.g. `fleet-watchdog/heartbeat/*.sh`),
  `bash -n <script>` is a fast syntax-only pre-check when shellcheck isn't
  installed.

## Portability notes (so it also survives on macOS)

If a script might ever run on a Mac, avoid GNU-isms that BSD tools lack —
the full list is in the macOS doc, but the headline traps:

- `sed -i` (GNU) vs `sed -i ''` (BSD). Prefer a temp-file rewrite if it
  must be portable.
- `grep -P` (GNU only) — use `grep -E`.
- `date -d` / `date -v` differ entirely.
- `readlink -f`, `stat -c`, `mktemp` flags differ.

When in doubt, stick to POSIX-ish forms and the portable spellings.

## Sources

- [Shell Script Best Practices — sharats.me](https://sharats.me/posts/shell-script-best-practices/)
- [Writing Safe Shell Scripts — MIT SIPB](https://sipb.mit.edu/doc/safe-shell/)
- [ShellCheck](https://www.shellcheck.net/)
- Plus gotchas verified firsthand during the fleet-watchdog deploy
  (root-owned env-file sudo trap, scheduler PATH, paste-time `#` comments).
