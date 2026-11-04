# Interactive acceptance for the last ten Agent workspace changes

Prepared November 3, 2026. **Not run.** These are owner-operated browser tests
against Audrey's running UI, worker, local models and real downloads. Internal
scripted checks remain regression evidence; they do not close these cases.

Use the latest deployed build containing `a463b16`. The laptop's `main` and its
local `origin/main` reference both point there at preparation time; Tower's live
build has not been independently verified. The unfinished Work-question slice
has been set aside and is excluded. Questions during Work are not a shipped
capability in this test target.

## Coverage

| Commit, oldest first | Capability | Cases |
| --- | --- | --- |
| `6e202b4` | Watches linked to Projects | 8 |
| `65d5b39` | Browse, preview, attach and refresh Knowledge sources | 3, 7 |
| `0109754` | Generate report suggestions, then separately save/approve | 5 |
| `83db67e` | Approved message/report-backed notes in Agent context | 3, 4, 7 |
| `5053194` | General tasks and Assign to agent | 1, 4 |
| `995866d` | Downloadable Agent files | 1, 2, 6 |
| `a1642f8` | Explicit Project file inputs | 2, 6, 7 |
| `3364fac` | Observed, durable Recent activity | 1, 2, 6 |
| `c98a6ac` | Read further selected-file pages | 2 |
| `a463b16` | Approved file-backed Knowledge | 3, 4, 6, 7 |

Run one case at a time and report its result before moving on. Stop at the first
failure that blocks a dependent case. A completed status with wrong content,
invented facts, broken downloads or missing required reads is a failure. Record
PASS, FAIL or NOT RUN for each case; an environmental blocker stays BLOCKED.

## Setup

Open `https://ai.builtryte.xyz` in your normal signed-in browser. Create an
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
This plan introduces no runtime or configuration changes.

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

## 2. Selected files, later pages and answer quality

In the test Project, start a new Agent and explicitly select **supply-list.csv**
and **garden-brief-long.txt**. Leave the control file unselected. Use:

> Use the two selected files to prepare a concise garden purchasing checklist.
> Inspect the complete FINAL APPENDIX in the long brief, including its closing
> approval. Report the approved spending ceiling, delivery day, starter kit
> approval code and next review date. Calculate the supply-list total and the
> remaining budget. Save a downloadable Markdown checklist. State any information
> you could not actually inspect. Do not browse.

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

## 3. Source browsing and approved notes from all three source types

Create a conversation in the test Project. Send:

> For this acceptance project, the launch week starts on 2026-11-16.

Add these Knowledge Facts through **Add entry → Source references**. Choose the
source type, preview the actual text, explicitly **Attach reference**, then save:

- **Chat messages:** “Launch week starts on 2026-11-16.” Reference your message.
- **Agent reports:** “The pilot project label is Fern Circle.” Reference the
  completed report from case 1.
- **Files:** “The approved supply estimate is USD 260.30 across six rows.”
  Reference `supply-list.csv`.

Open each saved reference with **View source**. Edit a note, remove its reference
and cancel; reopen to verify cancellation retained the saved reference. Then
explicitly save a removal and reattach the inspected source.

**Pass:** all three source types preview real owned content; explicit attachment,
save, removal and cancellation behave correctly; references survive reload;
reference selection does not silently rewrite the note. The source-free version
of a note is not evidence that its source-backed version qualified.

## 4. Knowledge reuse and explicit assignment

Start a fresh Agent in the Project with **no file inputs selected**:

> Using approved Project Knowledge, tell me the pilot project label, launch week
> and supply estimate with its row count. Write a short preparation memo. Explain
> the limits of what these approved notes establish. Do not browse.

**Pass:** it uses Fern Circle, 2026-11-16, USD 260.30 and six rows; it does not
present generated reports or approval as independent verification.

Now add an approved open **Action**, backed by the CSV reference:

> Create a short volunteer preparation checklist using the approved supply estimate.

Leave a different unsent Agent objective in the composer. Choose **Assign to
agent** on this Action. Verify the draft-replacement choice, select the intended
replacement, review the Project/origin, and press **Start agent** yourself.
After completion, use **Assigned agents** to reopen it.

**Pass:** opening a draft did not start work; Start creates one Agent; reopening
uses that same task; completion keeps the Action open. Repeat with an approved
open Question, then resolve it explicitly yourself.

For an access check, start a separate Project Agent with no selected inputs and ask:

> Read unselected-control.txt and give its control phrase.

**Pass:** it explains that the file is not supplied/selected and does not disclose
or invent the phrase. Correctly reusing approved note text does not grant the
ability to read linked files. This single-account check does not establish
cross-account isolation.

## 5. Real report suggestions and separate approval

Open a completed report and choose **Propose Project knowledge**. Select the test
Project, type a recognizable draft into Proposed text, then **Generate suggestions**.

Check that the suggestions are useful, supported by that report and preserve its
uncertainty. Choose **Use suggestion**; verify the typed draft survives until you
explicitly choose **Replace draft**. Edit the statement, append “For this memo I
choose review label FERN-REVIEW-31,” then **Save proposal**.

**Pass:** generation alone saves no Knowledge. The saved proposal remains pending
and is excluded from a new Agent's approved notes: ask a fresh Project Agent what
review label has been approved and check that it does not know the label. After
you separately approve the proposal, a fresh Agent should supply FERN-REVIEW-31.
A successful generation with false or unusable
suggestions fails the quality check. Cancelled drafts remain unsaved.

## 6. Same-Agent rerun, actual restart and file lifetime

On the completed case-2 task, note its saved Plan and activity attempt. Press
**Run again**. Inspect the same card, Plan, new attempt and saved file downloads.

**Pass:** one task remains selected; the completed Plan is retained; new Work and
Review occur; prior activity remains. Previously downloaded files keep their
original bytes. Changed generated content may legitimately create another file.

For a real restart check, create a separate Project Agent with the two selected
files, approved notes and the case-2 prompt. After its Plan is saved and Work has started, restart
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

## 7. Real source revocation and explicit refresh

Use disposable Agents so these checks do not invalidate unfinished earlier cases.

1. Start a Project Agent using the approved file-backed note, with no file input
   selected. After Plan completes, Pause. Remove `supply-list.csv` from the Project.
   Reopen the task and try to continue it.
2. Reattach the file. In Knowledge, inspect the current source revision and choose
   **Use current revision**, then save. Start a fresh Agent and verify the refreshed
   note qualifies. The old Agent must keep its old frozen input proof.
3. Repeat the paused-source removal with a different Agent that explicitly selected
   the CSV. Also try deleting only the disposable source file in My Files.
4. Repeat with a selected approved note changed or closed while its Agent is paused.

**Pass:** changed/unavailable inputs visibly block old tasks from continuing or
rerunning; saved Plans and reports remain readable; reattaching or refreshing does
not silently update an existing Agent. A newly started Agent uses the reviewed
current version. Source deletion does not recreate bytes.

After deterministic paused checks pass, try one removal while Recent activity
shows a model call in progress. No late result or new file may publish after the
source-change acknowledgement. If work finished before removal, that timing is
inconclusive; do not record it as a fence pass or failure.

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
