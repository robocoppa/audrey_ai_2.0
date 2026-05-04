# Working in this repo

## On every new session, before answering the user's first prompt

Read **`docs/campaign-1/HISTORY.md`** for the full Phase 1→31 build history
(behavioral facts, phase decisions, container/model layout) AND **`docs/lessons/CONTINUITY.md`**
for the current lesson-plan state. Together these are the source of truth
for what's been built and what we're working on right now.

Both files are **gitignored** and laptop-only — they live at these paths on
the development laptop and aren't synced to Unraid or anywhere else. Treat
them as the durable working memory for this project.

Path note: `HISTORY.md` was renamed from `CONTINUITY.md` (was at repo root)
on 2026-05-03 when the project transitioned from "build phases" to "lesson
plan." The build phases are complete and now live in `docs/campaign-1/`
alongside their phase-N-deploy.md docs.

After reading both, skip ahead to the **Next session should** section in
the lessons CONTINUITY (or HISTORY's section if there's no lesson-side
priority yet). That is where current priority lives.

## At the end of every phase or lesson

When a phase ships or a lesson concludes (deploy doc verified, smoke tests
passing, work acknowledged by the user as done), update the relevant
continuity-style doc before moving on:

- For **build phases** (historical now, but the convention persists for
  any future phase work): update `docs/campaign-1/HISTORY.md`.
- For **lessons**: update `docs/lessons/CONTINUITY.md`.

Specifically:

- Bump the `_Last updated:` line at the top with the current date and a
  one-line summary of what shipped.
- Update the **Status** section: current phase/lesson, last completed step,
  new behavioral facts worth remembering. Preserve older notes — they have
  load-bearing context that future sessions need.
- Rewrite the **Next session should** list: remove items now done, promote
  the next priority, capture any new followups.
- Update **Stack state**, **Containers table**, **Model registry**, etc. if
  the phase changed any of them.

The goal: a new Claude session that reads the file cold should be able
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
