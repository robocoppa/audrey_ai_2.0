# Campaign 3 Slice 3D.3 — measured repairs and terminal results

**Status:** Live-accepted, 2026-10-06. The targeted three-case Tower check passed.
Automatic runtime skill selection remains disabled. Backend verification is
recorded below.

## What the previous measurement established

[Slice 3D.2](phase-03-skill-selection-policy.md#received-result--october-6-2026)
completed its first separately authored 30-control measurement, with zero
execution errors. The hybrid matched 27/30 proposed labels: 12/13 correct
activations, two missed positive requests, and one ordinary false activation.
Eight model calls recovered seven positive requests and falsely selected video
analysis for a Spanish project-management request. Its old exit-zero status
represented execution completion and did not establish acceptable quality.

The three exposed labels remain defensible after independent agent review.
This review is not human validation. The measured fixture now supplies
regressions; later tuning cannot turn its result into new untouched holdout
validation. The full original report and all earlier measurements are retained.

## Evaluation policy revision 3

- A question locating or explaining quoted content inside an identified file
  stays eligible for analysis. Translating or rewriting text supplied in the
  prompt remains a task on that supplied text, including when a preceding
  sentence names its source file.
- File-location prefixes remain attached to their actions across ordinary
  commas, such as “From rehearsal-walkthrough.mp4 alone, describe…”. A separate
  negative clause excluding a poster cannot block analysis of the requested
  video. Scoped exclusions still block the excluded file.
- Bounded Spanish move, rename, and project-membership forms receive a firm
  `file_management` abstention. Spanish content questions remain eligible for
  the retained router; management followed by an explicit content request can
  still reach analysis. This does not implement general multilingual parsing.

These changes affect only the standalone evaluator. Its model prompt, schema,
production router, explicit skills, runtime configuration, and native UI are
unchanged. No skill is automatically activated.

## Copyable shell results

Use `FOREGROUND=1` for a short probe. It waits for completion, prints combined
stdout/stderr in the launching shell, and saves the same output in
`testing-out/probes`. It skips Telegram by default; an explicit prefix
`NOTIFY=1` opts into notification after the logfile is complete. Existing logs
are preserved on timestamp collisions. Detached execution remains available
for long jobs that need to survive an SSH disconnect.

The skill evaluator's new `--summary` prints compact JSON with its proposed-label
check, actual model call count, measured usage, and any incorrect selections.
It distinguishes `passed`, `findings`, and `failed`, and also reports the
underlying `measurement_status`. Here, `passed` means that the requested study
arm matched this selected synthetic fixture. It does not approve production
automatic selection. Full reports keep their execution `status` and add an
explicit `selection_check`.

The CLI now exits 1 for either mismatched proposed labels or model errors, in
both full and compact output. Setup errors exit 2. Only the requested arm is
checked: successful hybrid recovery is not penalized for the rules baseline's
misses. The optional `--save-json` still saves the full private report and
refuses overwrites, even when the shell prints a summary.

## Laptop verification

Full backend result: 3,888 passed, one existing FastAPI HTTP422 deprecation warning, 78.86 seconds.
Focused evaluator/policy/reporting/runner checks: 246 passed. Scoped Ruff,
Python compilation, and shell syntax passed. ShellCheck is unavailable.
Behavioral runner tests use fake Docker/curl commands; no live call, container
change, or Telegram message was made by laptop verification.

The three measured regressions now match all three proposed labels offline:
two correct activations, one guarded management abstention, zero misses,
zero false activations, and zero model calls or tokens. Neighboring checks
cover supplied-text transformations, scoped exclusions, Spanish content, and
mixed management/content requests. These results prove regression behavior,
not broad model precision or final-answer benefit.

## Targeted Tower check

After updating the Tower checkout, run this once from its shell. No upload,
Audrey token, new model, browser action, or container rebuild is needed. Keep
this short foreground run in the open shell until its result appears.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
FOREGROUND=1 scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_holdout.json \
  ARGS='--backend hybrid --config /app/config.yaml --cases /tmp/skill_selection_holdout.json --only holdout-document-quoted-content-question,holdout-video-excludes-unrelated-document,holdout-ordinary-project-management-spanish --repeats 1 --summary'
```

Success is exit zero, `status: passed`, `measurement_status: completed`,
`rules_revision: 3`, `case_count: 3`, `selection.correct: 3`, zero misses and
ordinary false activations, `execution.errors: 0`, and `findings: []`. Expected
`execution.model_called: 0`: both analysis requests are now handled by rules,
and project management is a firm abstention. This verifies the deployed probe
copy and foreground runner; it supplies no new model quality or speed evidence.

Copy the JSON directly from the shell. No Telegram download is needed. The
wrapper also prints the saved logfile location. The direct probe runs through
Docker DNS; it does not use the laptop's LAN/WARP or Tailscale route.

## Next decision

Do not repeat the settled pilot or old studies. Fresh independently reviewed
controls, agreed precision/miss/cost limits, and a real final-answer comparison
against explicit selection remain prerequisites for automatic activation.
The exposed cases are regressions, and adding more matching development cases
alone does not meet that gate. No new router candidate comparison is queued.

## Received Tower acceptance — October 6, 2026

The user supplied the foreground result created at
`2026-10-06T22:23:57.801179+00:00`: revision 3, hybrid, three cases, one repeat,
and the expected canonical cases hash. All three proposed labels matched:
two analysis activations and one guarded management abstention. There were
zero errors, misses, ordinary false activations, model calls, or tokens. The
shell returned exit zero and copyable JSON. The [preserved result](../../evals/results/2026-10-06-skill-selection-v3-regressions-smoke.json)
settles this targeted gate; do not repeat it unchanged.

This acceptance supplies no new Qwen model performance or qualification
evidence. [Slice 3D.4](phase-03-skill-selection-protocol.md) freezes a prospective
study before model measurement; automatic runtime selection remains disabled.
