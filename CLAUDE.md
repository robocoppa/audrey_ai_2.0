# Working in this repo

## On every new session, before answering the user's first prompt

Read `CONTINUITY.md` at the repo root. It is the source of truth for current
project state — phase status, behavioral facts, queued followups, container
layout, model registry, and decisions already locked. The codebase alone does
not tell you which phases are shipped vs. queued, why certain choices were
made, or which gotchas not to re-discover.

`CONTINUITY.md` is **gitignored** and laptop-only — it lives at this path on
the development laptop and is not synced to Unraid or anywhere else. Treat it
as the durable working memory for this project.

After reading it, skip ahead to its **Next session should** section. That is
where current priority lives.

## At the end of every phase

When a phase ships (deploy doc verified, smoke tests passing, work
acknowledged by the user as done), update `CONTINUITY.md` before moving on.
Specifically:

- Bump the `_Last updated:` line at the top with the current date and a
  one-line summary of what shipped.
- Update the **Status** section: current phase, last completed step, new
  behavioral facts worth remembering. Preserve older phase notes — they have
  load-bearing context that future sessions need.
- Rewrite the **Next session should** list: remove items now done, promote
  the next priority, capture any new followups discovered during the phase.
- Update **Stack state**, **Containers table**, **Model registry**, etc. if
  the phase changed any of them.

The goal: a new Claude session that reads `CONTINUITY.md` cold should be able
to pick up right where the last one left off, without re-deriving state from
git log or codebase scans. If after a phase you realize a fact would have
saved you 30 minutes of re-discovery, that's exactly the kind of thing that
belongs in the **behavioral facts** bullets.

## After every meaningful change, suggest a commit message

The user runs all git operations themselves — never run `git commit`,
`git push`, `git rebase`, or any other write-side git command. After
finishing a meaningful change (a real edit, not a one-liner question
or a discussion), end the reply with a suggested commit message in a
fenced code block so the user can paste it straight into
`git commit -m "…"`. Format:

```
<type>: <one-sentence summary>
```

Types follow conventional-commits roughly: `feat`, `fix`, `docs`,
`chore`, `refactor`. No `Commit:` prefix. No trailing period. Keep
the summary under ~70 chars where possible. If the change spans
multiple concerns, list one message per concern in separate fenced
blocks rather than cramming them into one line.

Skip the suggestion when the turn was just a question, exploration,
or non-code edit (a docs typo correction is borderline — include it
if the user clearly wants the change committed).
