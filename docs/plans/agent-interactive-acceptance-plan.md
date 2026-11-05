# Interactive acceptance for the last ten Agent workspace changes

Prepared November 3, 2026. **In progress; full acceptance pending.** These are owner-operated browser tests
against Audrey's running UI, worker, local models and real downloads. Internal
scripted checks remain regression evidence; they do not close these cases.

November 4 update: the owner supplied a Fern Circle result and requested layout
cleanup; its download checks have not been reported. Case 2 failed during Plan
with a local-model output-limit error before Work began. A concise, nonthinking
Plan fix is prepared; retry that same Agent after rebuilding backend/UI. Neither
the file-reading capability nor the full case is accepted by this failed run.
The owner then replaced structured Knowledge with a plain Project knowledge
textbox. Cases 3–5 and 7 below now test that simpler workflow; the previous source
attachment, assignment and proposal UI cases are superseded, not accepted.

Use the latest deployed build containing `a463b16`. The laptop's `main` and its
local `origin/main` reference both point there at preparation time; Tower's live
build has not been independently verified. The unfinished Work-question slice
has been set aside and is excluded. Questions during Work are not a shipped
capability in this test target.

## Coverage

| Commit, oldest first | Capability | Cases |
| --- | --- | --- |
| `6e202b4` | Watches linked to Projects | 8 |
| `65d5b39` | Legacy source references retained internally; structured UI superseded | — |
| `0109754` | Proposal UI superseded by owner; old records retained | — |
| `83db67e` | Saved Project notes in Agent context | 3, 7 |
| `5053194` | General tasks; structured assignment UI superseded | 1 |
| `995866d` | Downloadable Agent files | 1, 2, 6 |
| `a1642f8` | Explicit Project file inputs | 2, 6, 7 |
| `3364fac` | Observed, durable Recent activity | 1, 2, 6 |
| `c98a6ac` | Read further selected-file pages | 2 |
| `a463b16` | Legacy file-backed notes retained; use explicit files for reads | 2, 6, 7 |

Run one case at a time and report its result before moving on. Stop at the first
failure that blocks a dependent case. A completed status with wrong content,
invented facts, broken downloads or missing required reads is a failure. Record
PASS, FAIL or NOT RUN for each case; an environmental blocker stays BLOCKED.

## Setup

Open `https://ai.builtryte.xyz` in your normal signed-in browser. Choose **Projects**
in the top bar, then **New project**. Create an
ordinary Project named **Agent acceptance — November 3**. Leave its instructions
empty initially. Use only disposable test data in this Project.

Upload and attach these three inputs through the normal UI:

- [supply-list.csv](../../tests/manual/agent-workspace/supply-list.csv)
- [garden-brief-long.txt](../../tests/manual/agent-workspace/garden-brief-long.txt)
- [unselected-control.txt](../../tests/manual/agent-workspace/unselected-control.txt)

The long brief has 49,222 Unicode characters. Its final appendix starts at
character 40,000, with closing decisions at 49,000. Those decisions are beyond
the initial excerpt and span multiple tool pages. Keep this test plan open
locally as the answer key.

If the deployed build needs updating, rebuild `audrey` and `audrey-ui` on Tower
from `/mnt/user/appdata/audrey_ai_2.0` through the normal deployment workflow.
Backend startup applies schema 38 automatically. No configuration edit or new
upload is required for the Knowledge and layout checks.

## 1. General task, usable files and real activity

Start an Agent **without a Project** using:

> Create a five-step checklist for setting up a community garden. The project
> label is Fern Circle. Use only this request; do not browse. Save the checklist
> as checklist.md and create supplies.csv with exactly these columns and rows:
> item,quantity; seed_trays,3; gloves,2. In your final answer state the project
> label and identify both downloads.

While it runs, open Recent activity. After completion, download both files from
the task and find them in My Files. Open the Markdown and CSV in normal viewers.
Reload Audrey and reopen the same task.

**Pass:** useful five-step checklist carrying the supplied label; a real CSV with
one header and exactly two data rows; authenticated downloads open; observed
model and file-tool outcomes appear; saved results/files/activity survive reload.
No invented browsing or completed external actions. Plain pasted file content
with no usable download does not pass.

**Format coverage:** to accept the other four advertised formats, run a separate
Agent asking for the same two-row table in TXT, JSON, DOCX and XLSX. Download and
open all four; JSON must parse and Office files must open as native documents.
Mark any format you cannot inspect NOT RUN. One working Markdown file does not
establish that all six formats work.

**Layout and Project creation follow-up:** reopen the existing Fern Circle task
on a laptop. Plan/progress and Recent activity should be symmetrical side by side.
Expand saved Work and Show more activity; scroll either card with the pointer or
keyboard and check that its neighbor stays still. The Result should have a compact
header and readable report headings/code, with no proposal controls. The Project
badge should appear beside the task status and open that Project when available.
Open Search or narrow the window: cards should stack without page overflow.
In the Agent composer, enter a draft objective, choose New project and create a
disposable ordinary Project. It should become selected, preserve the objective,
and start no Agent. Also try Cancel and a narrow/mobile drawer. On the Project
page, find New conversation inside Project conversations. Shrink the window and
check that Edit/Delete stay together and Upload file remains on one line.
No upload or new model call is
needed to inspect the existing result layout.

**Current layout and deletion follow-up:** Original prompt and Questions and
answers should be beside one another; Last activity should appear only in the
history list. Plan/Work/Review names, statuses and View controls should fit compact
rows. Delete a disposable finished task using its two-click confirmation, then
immediately Start another simple task: “Write a one-sentence welcome for Fern
Circle. Do not browse.” It should select the new Agent without an old task-change
banner or a duplicate task. A genuine failed operation still needs assessment.
In Projects, Knowledge should be beside Conversations at laptop widths, with
Files/Agents below and the empty instructions hint beside the title. Switch to
Chat and Agents and back; existing drafts should remain. No uploads are needed
for this deletion/layout follow-up.

## 2. Selected files, later pages and answer quality

In the test Project, start a new Agent and explicitly select **supply-list.csv**
and **garden-brief-long.txt**. Leave the control file unselected. Use:

> Use the two selected files to prepare a concise garden purchasing checklist.
> Inspect the complete FINAL APPENDIX in the long brief, including its closing
> approval. Report the approved spending ceiling, delivery day, starter kit
> approval code and next review date. Calculate the supply-list total and the
> remaining budget. Save a downloadable Markdown checklist. State any information
> you could not actually inspect. Do not browse.

If the same Agent is already waiting for you to supply facts from the truncated
appendix, answer: “Read those facts from the FINAL APPENDIX in the selected long
file during Work; do not ask me to transcribe them.” Press Answer and continue.
Alternatively Cancel, then Run again on that same task. Neither recovery should
create a new Agent or require reuploading its frozen inputs. A new run should plan
the file lookup rather than ask you to copy facts that Work can inspect.

Check the shortened input counts, watch Recent activity, and open the downloaded
file. Compare its contents against this key:

| Fact | Required value |
| --- | --- |
| Approved ceiling | USD 472.35 |
| Delivery day | Wednesday |
| Starter kit approval code | `MOSS-7Q4-RIVER` |
| Next review | 2026-11-23 |
| Supply rows | 6 |
| Supply total | USD 260.30 |
| Remaining budget | USD 212.05 |

**Pass:** exact facts and correct arithmetic in a usable file; Read selected file
activity consistent with reaching both ends of the appendix; no control-file
phrase; honest coverage. Read events alone do not pass wrong answers. A report
claiming full inspection while missing closing decisions fails.

After this works, repeat the source-reading portion with a real PDF and a saved
OCR or media transcript whose contents you can verify. Track those source types
separately. The TXT/CSV case does not establish PDF/OCR/media acceptance.

## 3. Plain Project knowledge, persistence and real reuse

On the disposable Project, enter the following in the single **Project knowledge**
textbox and press **Save**:

> The pilot project label is Fern Circle. Launch week starts on 2026-11-16.
> The approved supply estimate is USD 260.30 across six rows.
> The review label is FERN-REVIEW-31.

Reload Audrey and reopen the Project. Check that the exact notes remain and that
there are no entry types, source selectors, approval controls or assignment buttons.
If this Project already had approved notes, verify they appeared in full before
editing. Older unapproved proposals must not appear as accepted text.

Click **New conversation** inside **Project conversations**, then ask:

> Using this Project's saved knowledge, state the pilot label, launch week,
> supply estimate and row count, and review label. Do not browse.

Start a fresh Agent in the same Project, with **no file inputs selected**, using
that same prompt. Verify the Project badge is beside its status.

**Pass:** notes survive reload and both real chat and Agent answers give Fern
Circle, 2026-11-16, USD 260.30, six rows and FERN-REVIEW-31. They do not claim to
have inspected source files or independently verified these user-written notes.

## 4. Draft protection and concurrent edits

Open the disposable Project in two browser tabs. In tab A, change Knowledge text
without saving. In tab B, change it differently and Save. Now Save tab A.

**Pass:** tab A keeps its draft, reports the conflict and shows the latest saved
text. **Use saved text** adopts tab B without writing. Repeat once, then choose
**Keep my edits** and Save; reload to verify the explicitly chosen draft persisted.
A failed save or a hidden/reopened Project must not silently replace local edits.

## 5. Explicit clearing and file access

Clear the Knowledge textbox, Save and reload. Start a new Agent in this Project,
without selecting files, and ask:

> What review label is saved in Project knowledge? Do not browse or guess.

**Pass:** the textbox stays empty; the new Agent does not claim the old review
label is still saved. Existing completed reports remain readable. Restore the
case-3 notes explicitly before continuing the reuse checks.

For a separate access check, start another Project Agent with no selected files:

> Read unselected-control.txt and give its control phrase.

**Pass:** it explains that the file is not supplied/selected and does not disclose
or invent the phrase. Saved plain notes do not grant access to Project file text.
This single-account check does not establish cross-account isolation.

## 6. Same-Agent rerun, actual restart and file lifetime

On the completed case-2 task, note its saved Plan and activity attempt. Press
**Run again**. Inspect the same card, Plan, new attempt and saved file downloads.

**Pass:** one task remains selected; the completed Plan is retained; new Work and
Review occur; prior activity remains. Previously downloaded files keep their
original bytes. Changed generated content may legitimately create another file.

For a real restart check, create a separate Project Agent with the two selected
files, saved knowledge and the case-2 prompt. After its Plan is saved and Work has started, restart
only the backend on Tower during a quiet test window:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose restart audrey
```

Reload the browser and follow the same task to completion. This restarts the
backend service; use a window where interrupting other work is acceptable.

**Pass:** committed Plan/results and fixed inputs survive; unfinished Work can
repeat within the existing allowance; no new Agent appears; already committed
identical outputs are reused. An interrupted activity entry is not a successful
tool effect. A browser reload alone does not test backend restart recovery.

Finally, on a disposable completed task, press Delete once, verify the confirmation
check, then press it again. Its independently saved My Files outputs must remain.
Delete a disposable output in My Files and confirm its old task download no longer
works. If rerun produces the exact same output identity, deletion must not be undone.

## 7. Saved knowledge changes and selected-file revocation

Use disposable Agents so these checks do not invalidate earlier unfinished work.

1. Start an Agent with case-3 saved knowledge. After Plan completes, Pause it.
   Change the Knowledge textbox and Save. Reopen the task and try to continue.
2. Start a new Agent in that Project. Verify it uses the edited text; the old
   task must keep its original pins and remain blocked rather than silently update.
3. Start a different Agent with the CSV explicitly selected. Pause after Plan,
   remove the CSV from the Project, then try to continue. Reattaching must not
   silently replace the old Agent's frozen input. Also try deleting a disposable
   selected source in My Files.

**Pass:** saved Knowledge edits and selected-file changes visibly block old tasks
that used them; saved Plans/reports stay readable. A newly started Agent uses
current inputs. Plain user-written notes remain saved when a file is removed;
only selecting files gives the Agent permission to read their text.

After the paused checks pass, try changing Knowledge during a model call.
No late result or new file may publish after the change is acknowledged. If work
finished first, that timing is inconclusive rather than a fence pass or failure.

## 8. Project Watches through the real UI and fetcher

From the Project, choose **New watch in Project**. Pick a public HTTP(S) page you
can inspect, without a query string or login. Enter a relevant change criterion,
use America/Denver, preview a future schedule and create the Watch. Press **Pause**
before that scheduled check, then use **Run now**. The current creation form does
not offer a paused-creation checkbox.

**Pass:** the Project is preselected, creation is explicit, the check actually
fetches the page and reaches **Baseline saved**, and Run now leaves the schedule
paused. Reopen from the Project, rename the Project, then reassign/unlink the
Watch through Edit. It remains the same Watch, and old **Saved check inputs**
retain their original Project association. Delete a separate disposable linked
Project: its Watch must block Run/Resume until explicitly reassigned/unlinked.

For actual change alerts, use a public page you control. Establish a baseline,
change text in a way that clearly meets the criterion, then Run now. Inspect the
exact diff, local-model assessment and Activity link to that exact saved check.
Run again unchanged and check that no duplicate alert is created. Resume the
schedule and observe at least one **Scheduled** check (minimum interval: 60 minutes),
then Pause it. Without a controlled change or an observed scheduled run, mark
those two portions NOT RUN; a baseline alone cannot establish either capability.

## Recording results and cleanup

For each case, send its number, PASS/FAIL/BLOCKED/NOT RUN, and a brief observation.
On failure, retain the task title, approximate time, visible error, downloaded
output and a screenshot of the affected UI if useful. Do not paste authentication
tokens. Keep failed tasks and sources until the failure is assessed.

After reviewing results, delete only the disposable test Projects, Watches, tasks
and uploaded/generated files. Task or Project deletion does not automatically
remove independently owned My Files outputs. Keep the real-model quality result
separate from UI, persistence and access-boundary results.
