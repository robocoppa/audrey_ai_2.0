# Manual Agent workspace inputs

These reusable fictional fixtures support the [Agent workspace](../../../docs/plans/agent-workspace-plan.md).
Upload them through Audrey's ordinary Project UI. Checks use the running application and real
local inference. There is no scripted model transport or automatic pass runner.

- `supply-list.csv`: six rows; extended cost USD 260.30.
- `garden-brief-long.txt`: 49,222 Unicode characters; final appendix starts at
  offset 40,000 and closing decisions at 49,000, beyond the opening excerpt.
- `unselected-control.txt`: attach it to the Project, but leave it unselected in
  the Agent input chooser when checking source scope.

The long brief's final appendix sets a USD 472.35 spending ceiling, Wednesday
delivery, starter-kit code `MOSS-7Q4-RIVER` and review date `2026-11-23`.
The CSV total leaves USD 212.05 under that ceiling. A file-paging check should
inspect real downloaded content carrying the required facts from multiple pages;
a tool event alone is insufficient. The previous interactive acceptance document
was owner-closed and removed on November 13, 2026; these inputs remain reusable.

## Selected-file search

After syncing the slice, rebuild `audrey` and `audrey-ui` on Tower from
`/mnt/user/appdata/audrey_ai_2.0`:

```bash
docker compose up -d --build audrey audrey-ui
```

Schema 44 applies at startup; no configuration change is needed. In the real
Audrey browser, open an ordinary Project containing `garden-brief-long.txt`.
Upload that fixture only if it is not already present, and wait for Ready.
Start a new Agent in that Project and explicitly select that file. Use:

> Use search_selected_file to find the literal phrase “FINAL APPENDIX — CLOSING
> APPROVAL” in the selected file, then read the surrounding passage with
> read_selected_file. Save its delivery day, starter kit approval code and next
> review date in final-decisions.md. Also search for the literal phrase
> “NOT-IN-THIS-FIXTURE-731” and report whether that exact phrase matched. Do not
> browse or guess missing facts.

Open Recent activity: Work must show completed Search selected file, Read selected
file and Create file entries. The no-match query is a successful search, not an
error. Review may make its own native checks. Download and open
`final-decisions.md`: it must contain **Wednesday**, **MOSS-7Q4-RIVER** and
**2026-11-23**. The report must say the second literal phrase had no match without
claiming that search inspected every file passage. Correct-looking prose without
actual search/read events and the real downloaded contents does not pass.

The selected-file tool must not appear in Plan. Other Project files remain outside
the Agent's selected scope. Fast isolated regressions exercise those access and
restart boundaries separately; they do not replace this real local-model check.

## Saved-output inspection

After syncing this slice, rebuild `audrey` and `audrey-ui` on **Tower** from
`/mnt/user/appdata/audrey_ai_2.0`:

```bash
docker compose up -d --build audrey audrey-ui
```

Schema 45 applies automatically. No configuration change or upload is needed.
In the real Audrey browser, start a new Agent without selected files using:

> This is a test of checking an actual saved deliverable. During Plan, outline
> the steps without questions. During Work, create budget-check.md with these
> six purchases: 8 at 12.50, 6 at 3.25, 4 at 7.75, 2 at 18.40, 5 at 9.60 and
> 4 at 6.25, and a spending ceiling of 472.35. Intentionally put the incorrect
> total 270.95 and remainder 201.40 in that first draft. Reopen the saved draft
> with read_output_file and confirm what it actually contains. During Review,
> reopen that saved file with read_output_file, use calculate to recompute both
> figures, then use create_file to save the complete corrected budget-check.md.
> Reopen the corrected file with read_output_file before returning its download.
> Keep this deliberate draft error out of the final deliverable. Do not browse.

Plan should produce an outline without asking whether the later tools exist or
how to call them. If an Agent is already waiting on that question from an earlier
build, sync the guidance correction and rebuild only `audrey` on Tower from the
same checkout:

```bash
docker compose up -d --build audrey
```

Refresh, reopen that same Agent and answer its saved question:

> Use the tools Audrey supplies during Work and Review. Complete the Plan now;
> no tool calls are needed during Plan. Continue with the original test.

This resumes the existing Agent's unfinished Plan. A fresh Agent with the original
prompt checks that Plan no longer needs the workaround; a saved answer alone does
not establish that behavior.

Pass only when all of these are visible:

1. Recent activity shows successful **Read output file** operations in both Work
   and Review, successful Review **Calculate** entries and a Review **Create file**.
   Use **Show more activity** if needed. Plan has no tool operations.
2. Open the actual current `budget-check.md` download from the Agent's Files area.
   Its total is **260.30** and remaining budget is **212.05**, with the six purchase
   quantities and prices intact. The final Result links that corrected file.
3. Expand **Previous files** and open the earlier draft. It still contains
   **270.95 / 201.40**, with a different file identity. The correction preserves
   the old bytes rather than silently modifying a previous download.

The model's claim that it read or corrected a file is insufficient: inspect the
events and both actual downloads. This checks real local-model tool choice,
the running worker and browser/proxy wiring. Scripted local regressions separately
check owner/task isolation, changed/deleted files, paging and revoked attempts;
they do not close this interactive acceptance check.

## Resetting a Watch baseline

After syncing, rebuild the backend and UI on **Tower**.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
```

1. Open **Agents → Watches** and select a public-page Watch that has already
   saved a baseline. Use a paused Watch or pause it first. No file upload is needed.
2. Click **Reset baseline** beside **Check history**, then **Confirm baseline
   reset**. Wait for the new check to finish.
3. Verify history shows **Baseline saved** and **Baseline reset** on the new
   check. Its detail should say the check saved a new baseline. Open
   the prior check from history and verify its result remains available. The
   Watch header should still say **Paused** and the sidebar should show
   **Next: Not scheduled**. The reset produces no change notification.
4. Click **Run now**. If the page text has stayed the same, the check should say
   **Unchanged**, and the schedule should still be paused. A page that changed
   between checks is a separate comparison against the new baseline.

## Viewing Watch page changes

After syncing, rebuild the UI on **Tower**.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey-ui
```

Open a changed check in **Agents → Watches → Check history**, or create a
temporary Watch named `UTC clock check` using the
[public UTC clock](https://www.timeanddate.com/worldclock/timezone/utc).
Use **Hourly** and the following criterion.

> Notify me whenever the displayed UTC clock time changes. Treat any change to
> the displayed clock value as relevant.

Pause the Watch and click **Run now** to save a baseline. Wait about ten seconds,
then click **Run now** again. Wait for the check to finish. If the page returned the
same text, wait briefly and run another manual check.

Open the changed check and expand **Page changes**. The **Before** and **After**
excerpts should show the two clock values with the changed passage highlighted.
The model assessment belongs above the excerpts. A clock change that matches the
criterion should appear as **Relevant change**. For a **Baseline saved** or
**Unchanged** check, the **Page changes** heading explains the outcome in place.
Select a different saved check and verify its details appear without a loading
message flashing between them. No upload is needed.

## Viewing saved Watch page text

After syncing, rebuild the backend and UI on **Tower**.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
```

1. Refresh Audrey and open **Agents → Watches**. Select a Watch with a
   completed **Baseline saved** or **Unchanged** entry in **Check history**.
2. Select that entry and expand **Saved page text**. Verify that recognizable
   text from the Watch's page appears below the disclosure. A long page should
   scroll within the check detail at laptop widths.
3. If the text spans several pages, use **Next page** and **Previous page**.
   Verify that the character range advances and then returns with the original
   text.
4. Select a different completed check. Open its **Saved page text** and verify
   that the viewer starts on the first page for that check. The history and
   next scheduled check should stay the same throughout the test.

Use existing entries for this check. No live fetch, model call, or upload is
needed. If there is no completed check with retained text, choose **Run now**
once and wait for a successful result before opening the viewer.

## Finding text in a saved Watch check

After syncing, rebuild the backend and UI on **Tower**.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
```

1. Refresh Audrey and open **Agents → Watches → Check history**. Select an
   existing successful check for the Python downloads Watch and expand
   **Saved page text**. No upload is needed.
2. Click **Next page** and copy a release name or distinctive phrase from that
   page. Click **Previous page** to return to the beginning.
3. Enter the phrase in **Find text** and press Enter. Audrey should jump to the
   matching page, highlight the phrase, and display the match count. If the
   first match is earlier in the capture, use **Next match** to reach the later
   occurrence.
4. Search for `release notes` to check case-insensitive matching and use
   **Next match** and **Previous match** to move through the highlights. Search
   for `audrey-no-such-release-731` to check the empty result.

## Watching Project file attachments

After syncing, rebuild the backend and UI on **Tower**. Schema 46 applies at
backend startup.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
```

1. Create an ordinary Project named `Attachment Watch Test` and leave it empty.
   In **Agents → Watches → New watch**, set **Watch source** to **Project files**,
   choose the Project, and enter `Attachment changes` as the title. Select
   **Hourly**, click **Preview next checks**, then **Create watch**.
2. Open the new Watch and click **Pause**, then **Run now**. The check should
   finish with **Baseline saved** and **Attached files: 0**.
3. In the Project's **Files** box, use **Choose from My Files** to attach one
   existing ordinary file. If needed, use **Upload file** with
   [supply-list.csv](supply-list.csv). Wait for the file to appear in the box.
4. Return to the Watch and click **Run now**. Expect **Files changed** with
   **Added: 1**, **Removed: 0**, and **Attached files: 1**. Open its change in
   **Agents → Updates** and verify it opens that exact check.
5. Run another check with the attachment still in place. Expect **Unchanged**
   and a single addition Update. Remove the file from the Project and run the
   Watch again. Expect **Files changed**, **Added: 0**, **Removed: 1**, and
   **Attached files: 0**. The Watch should stay **Paused** throughout.

## Watching a Project file's saved text

After syncing, rebuild the backend and UI on **Tower**. Schema 47 applies at
backend startup.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --build audrey audrey-ui
```

1. On the laptop, prepare two small text files in separate folders. Name both
   `release-notes.txt`. Put `Release status: Draft` in the first and
   `Release status: Approved` in the second. Upload the first into an ordinary
   Project using **Upload file**. Wait for it to be ready.
2. Go to **Agents → Watches → New watch**. Set **Watch source** to **Project file
   text**, choose the Project and select `release-notes.txt` under **Filename**.
   Enter a title, select **Hourly**, click **Preview next checks**, then
   **Create watch**.
3. Click **Pause** and **Run now**. Expect **Baseline saved** and the result
   **A saved text baseline was recorded.**
4. Upload the revised `release-notes.txt` into the same Project, keeping the
   first copy attached. Once it is ready, click **Run now** on the Watch. Expect
   **File text changed**, and verify that the notification in **Updates** opens
   that check. Another manual check should say **Unchanged**.
5. Remove the newest copy from the Project and run the Watch again. Expect a
   fresh **Baseline saved** for the earlier copy. The Watch stays paused
   throughout the manual checks.
