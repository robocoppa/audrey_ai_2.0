# Manual Agent workspace inputs

Use these files with the [interactive acceptance plan](../../../docs/plans/agent-interactive-acceptance-plan.md).
Upload them through Audrey's ordinary Project UI. The fictional input data has
known answers; the acceptance exercise uses the running application and real
local inference. There is no scripted model transport or automatic pass runner.

- `supply-list.csv`: six rows; extended cost USD 260.30.
- `garden-brief-long.txt`: 49,222 Unicode characters; final appendix starts at
  offset 40,000 and closing decisions at 49,000, beyond the opening excerpt.
- `unselected-control.txt`: attach it to the Project, but leave it unselected in
  the Agent input chooser when checking source scope.

The plan contains expected facts and test instructions. Keep those open locally
while uploading the three input files. Case 2 requires a real downloadable output
with correct facts from multiple pages; a tool event alone is insufficient.
