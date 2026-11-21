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
