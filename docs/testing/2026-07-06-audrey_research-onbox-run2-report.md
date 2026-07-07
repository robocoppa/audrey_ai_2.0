# Eval report — 2026-07-06 research protocol RUN 2 (post-fix deploy) — 2/10, killed mid-run

Paired with [`2026-07-06-audrey_research-onbox-run2-answers.md`](2026-07-06-audrey_research-onbox-run2-answers.md).
First run after deploying the three fixes from the morning assessment
([run-1 report](2026-07-06-audrey_research-onbox-report.md)). **2/10 — the two
completed cases validate the fixes; the run then died to an infrastructure
event, not code.**

## Headline

All three fixes are confirmed live on real traffic. The run stopped being a
protocol run at case 3: `bio-archimedes` died mid-stream
(`RemoteProtocolError: peer closed connection without sending complete message
body`) and every subsequent case got `ConnectError: [Errno 111] Connection
refused`. The eval talks to OWUI (`open-webui:8080/api`), so connection-refused
means **OWUI itself stopped accepting connections** — no audrey-ai code path
can produce that. One event explains both errors: the stack (or the OWUI
container) went down ~23 minutes into the run and stayed down. Needs box-side
triage (below), then a clean re-run.

## Fix validation (the two completed cases)

- **Eval sources line is now trustworthy under the trace flag.** Both cases
  report `sources:8`, matching their real rendered blocks exactly — and
  pythagoras's trace contains a researcher-note `## SOURCES` H2 heading that
  would have hijacked the old `rfind` (run 1 counted a note list on euclid and
  25 ledger URLs on parallel-postulate). The hijack is dead on live data.
- **No `</think>` anywhere in the file** (run 1 had a full duplicated draft
  around a dangling tag).
- **Ledger linkage is clean** — claims carry real source IDs (`w0_s4`,
  `w1_S1…`), zero title-string refs, and both cases render authority-ranked
  Sources blocks. (Sources' own `supports` lists are empty — the one-directional
  linkage the dual-direction logic exists for.)
- **Corrections chain visibly steering prose (strong positive):** the euclid
  fact-check corrections were applied nearly verbatim — the answer
  distinguishes the 1482 Ratdolt Latin edition from the 1533 Grynaeus first
  *Greek* edition (the exact CORRECT instruction), softens "second only to the
  Bible" to "often said", "over 1,000 editions" to "well over a thousand
  editions have been estimated", and attributes the Einstein anecdote ("is
  also said to have reflected"). That is the pipeline doing precisely what it
  was built to do.

## Not a fix regression: the disposition wall at scale

Run-2 euclid reads more hedged than run-1 euclid (9 hedge-phrases vs 2 in the
body). Two things to keep straight:

1. **The fixes didn't grow the wall — it shrank.** Run 1: euclid 80 action
   lines (150-claim ledger), pythagoras 105 (105 claims). Run 2: 61 (136
   claims) and 69 (88 claims). The starve-proof suppression only fires on
   zero-usable-URL ledgers; these are grounded cases, so it correctly stays
   out of the way.
2. Much of run-2's extra hedging is **instructed** (the corrections above);
   the rest is spillover pressure from a ~60-line HEDGE wall, including
   `hedge_or_cite_strongly` on trivially-verifiable bibliographic facts
   (Heiberg 1883–1888, Billingsley 1570) that the structuring model marks
   `risk: high` reflexively. `hedge_policy` checks risk before authoritative
   backing (by design, +37), so reference-backed dates still hedge.

**Standing design issue, not tuned this session (deliberately):** dense-bio
ledgers produce 60–105 action lines, dwarfing the few dispositions that
matter. Candidate levers, all needing their own eval: teach the structuring
prompt to reserve `high` risk for genuinely contestable claims; let
authoritative backing beat `high` risk for dates/editions; cap rendered
action lines. Writer hedge-application also varies run-to-run (run-1 euclid
stated plainly under a *bigger* wall), so don't tune on one sample.

## Watch item: fact-check produced 0 per-claim verdicts on both cases

- euclid: no verdicts section at all — the fact-checker returned prose
  corrections (rich and correct ones), not parseable checks.
- pythagoras: `0 checks` + 3 fatal errors (`contradiction — w0_c12` etc.),
  second run in a row this case's fact-check aborts.

Run-1 euclid managed 8 checks; run-1 rust-async managed 79. At 100+-claim
ledger scale the per-claim verdict path looks unreliable — the fail-soft
(prose corrections still flow) is carrying it. Worth a look if it persists
on the clean re-run.

## Latency

euclid 445.0s (run 1: 409.7s), pythagoras 577.3s (run 1: 399.5s) — pythagoras
+45%, driven by a 203.6s glm researcher round and the big ledger. Ledger bloat
costs time in every downstream stage.

## Disposition

- **Root cause CONFIRMED (box triage, same day):** at ~10:00 box time an
  Unraid scheduled job restarted every container except `ollama` and
  auto-updated OWUI 0.9.x → **0.10.2** (cold-start banner at 10:00:51 in its
  log; box uptime spans the event, so no reboot). The eval's three
  `POST /api/chat/completions` lines (09:37 / 09:44 / 09:54) put archimedes
  mid-stream at exactly 10:00. Suspect CA Auto Update / Appdata Backup —
  check the schedule in the Unraid UI; don't straddle that window with
  protocol runs. Saved to auto-memory
  (`project_unraid_scheduled_stack_restart`).
- **Before the clean re-run:** OWUI 0.10.2 is un-smoke-tested — verify the
  Audrey Connection still has Auth=`Session` and run the smoke set first.
- The three fixes are validated; no code changes from this run.
- Carry-forward candidates: disposition-wall scale + risk-inflation (design
  discussion first), fact-check verdict reliability at ledger scale, optional
  eval resilience (one delayed retry on `ConnectError` would have ridden out
  the 51-second OWUI restart gap).
