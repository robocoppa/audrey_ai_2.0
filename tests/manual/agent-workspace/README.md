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
