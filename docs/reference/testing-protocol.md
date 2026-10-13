# Testing protocol — runs, pings and reports

Shared word for word by ai-sec and Audrey, at `docs/reference/testing-protocol.md` in
each: change both together. It says how a run is handed over, how it reports back, and
how its report is read. What to test, and where, stays in each repo's own docs.

## The loop

1. **The agent prepares the run.** It checks every premise it can on the laptop. Then it
   hands over one block per machine, naming the machine, and says what the run proves
   and roughly how long it takes.
2. **The user starts the run and may walk away.** A long run pings Telegram as it
   starts, and again however it ends, with its report attached.
3. **The agent reads the report, not a paste.** It reads it from
   `~/Downloads/Telegram Desktop/`, newest first, checking the stamp in its name against
   the run it handed over. For a laptop run, it can read the run's folder instead. A
   short check prints in the shell, and the user pastes that.

## Which runner

| run | how | pings |
|---|---|---|
| a short check, a minute or two | in the foreground, results in the shell | no; `NOTIFY=1` asks for them |
| a long run, minutes to hours | detached (`nohup … &`, or a named container on the Tower), or a laptop script that pings | at its start and at every exit; `NOTIFY=0` skips them |

A run that must survive an SSH drop or a sleeping laptop runs detached.

## The ping

One message, then the report as a document:

```text
<verdict> <repo> <run> finished (exit <n>) → <report path>
<the run's own one-line result>
```

| exit | verdict | meaning |
|---|---|---|
| 0 | ✅ | clean: every check passed, or every criterion was met |
| 1 | ⚠️ | finished with findings: read the report, this is usually the point |
| 2 | ❌ SETUP ERROR | a precondition failed, and nothing ran |
| 3 | ❌ CRASHED | the harness failed mid-run; the traceback is in its log or terminal |
| 130 | ⛔ INTERRUPTED | stopped by hand; no result |

- **Every exit pings,** including a setup error or an interruption. A run walked away
  from without a ping is found hours later.
- **The report always goes as a document.** Telegram refuses a message over 4,096
  characters. Audrey tried inlining a log tail on 2026-08-15, and reverted it.
- **A report's name says which run it is**, with the run's name and its stamp:
  `b6-drill-20261012T013403Z.txt`, `report-dev-2026-10-05-202427.txt`. A chat full of
  `report.txt` files cannot be told apart.

## The bot

- **The bot** is fleet-watchdog's watcher bot, set by `WATCHDOG_TOKEN` and
  `WATCHDOG_CHAT_ID`.
- **On the laptop** they come from `$AISEC_NOTIFY_ENV`, else
  `~/.config/ai-sec/notify.env`, else `~/Documents/github/fleet-watchdog/.env`. On the
  Tower they come from `/mnt/user/appdata/fleet-watchdog/.env` (`WATCHDOG_ENV`).
- **The token is never printed.** It is read as data or sourced in a subshell. A failed
  send names only the kind of error, since an error's text can carry the address the
  token is part of.
- **A ping never fails a run.** A missing bot or a failed send warns on stderr, and the
  run's own exit stands. After a `WARN` line, the report is still on disk: read it there.

## What a report may hold

A report may hold verdicts, counts, digests, timings and the run's own lines. Telegram is
a third-party service, so a report never holds credentials or client data. In ai-sec, it
also never holds lab alert content or a model's reply to it. Those stay in the run's
folder (`refused.json`, the scratch hubs) and are read there.

## Before handing a run over

- **Preflight first.** Everything checkable is checked before the first request, and a
  failure exits 2 before anything starts.
- **One run at a time on the box.** ai-sec takes the GPU lease
  (`scripts/eval/gpu-lease.sh`). Audrey never launches two evals at once, since its eval
  container's name is fixed (`docs/guides/eval-runbook.md`).
- **Say what proves what.** In ai-sec, the criteria are written before the result. In
  Audrey, name the targeted check and its pass condition.

## The runners

| repo | runner | where | pings |
|---|---|---|---|
| ai-sec | `scripts/eval/eval-model.sh`, `scripts/eval/eval-arms.sh` | laptop, under `nohup` | yes |
| ai-sec | `tests/smoke/smoke-run.sh` | laptop, detached | yes |
| ai-sec | `scripts/b6_drill.py`, `scripts/b8_drill.py`, and each long drill after them | laptop, ten to twenty-five minutes | yes, by `scripts/notify.py` |
| ai-sec | `scripts/b9_drill.py`, which signs and sends B9's updates, and the `check` of `scripts/b10_drill.py` and `scripts/b10_retention.py`, which read B10's restore and its collectors' purge | laptop, under a minute | no: each prints its result |
| ai-sec | `scripts/box/b8_drill.sh`, `b9_drill.sh`, `b10_drill.sh` and `b10_retention.sh`, shipped to `/boot/config/aisec/` | Tower, as root, three to twenty minutes | yes, through fleet-watchdog's `.env` |
| Audrey | `evals/eval-onbox.sh` | Tower, a detached container | yes |
| Audrey | `scripts/probes/probe-onbox.sh` | Tower | detached, yes; with `FOREGROUND=1`, only with `NOTIFY=1` |
| Audrey | `evals/eval_research.py`, `evals/run_all_evals.sh` | laptop, in the background | not yet: the results are in `evals/results/` |

To check the bot answers, run `python3 scripts/notify.py "a test ping"` on the laptop,
from ai-sec's root. It uses only Python's standard library, so it needs no venv.

**More:** ai-sec's `AGENTS.md` (*Commands for the user*) and the phase doc of the run;
Audrey's `docs/reference/live-smoke-testing.md`, `docs/guides/eval-runbook.md` and
`evals/README.md`.
