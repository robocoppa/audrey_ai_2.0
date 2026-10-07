# Campaign 3 Slice 3D.5 — scoped Japanese file exclusions

**Status:** Laptop-complete. The full backend suite passed 4,031 tests, and the
four-case targeted Tower check is pending. Automatic skill selection remains
disabled. This changes the standalone evaluator's policy, not Audrey's runtime.

## Why this slice

The operator copied the [full revision-3 report](../../evals/results/2026-10-06-skill-selection-v3-prospective-full.json)
to the laptop. It matches the saved preparation and terminal summary. The
study executed successfully, but its quality criteria did not pass: 69/72
labels matched, 36/39 activation precision, and one Japanese rewrite request
wrongly activated document analysis in all three repeats. Rules alone missed
21 positive observations, all recovered by the model. The first model request
took 6.216 seconds, so the low median/p95 do not prove fast first responses.

The failing request supplies its complete sentence and explicitly says not
to read the attached PDF. Independent review upheld `none`. The original
fixture, full report, and preparation remain unchanged. They are now exposed
regression evidence, not untouched controls for future qualification.

## Evaluation policy revision 4

- Treat Japanese quotation marks `「…」` and `『…』` as quoted data. Preserve
  exact filenames inside those quotes so genuine file questions still resolve.
- Recognize bounded reading, opening, and reference prohibitions immediately
  after a known filename, outside quoted prose. These return a firm abstention
  instead of allowing the router to activate analysis for the excluded evidence.
- Scope Japanese sentences and commas after explicit denials separately. A
  denied PDF does not block an independently requested document or video.
  Ordinary location commas and punctuation inside exact filenames stay intact.
- Preserve Japanese content questions and actions on identified files as
  eligible router decisions. Asking for a summary so the user need not read
  the file, or requesting partial reading, is not an access prohibition.

These are bounded grammar patterns, not general Japanese parsing. Unknown
forms remain subject to the existing conservative policy and retained router.
An unquoted filename immediately joined to Japanese particles remains an
existing resolution limitation; these controls use spaces or quoted filenames.
The model prompt, schema, production router, runtime configuration, explicit
skills, and UI are unchanged. `skills.auto_select: false` remains set.

Changing evaluator source and rules revision invalidates revision-3 plans for
future runs. Preserve that historical plan; do not rerun the old study or change
its labels to obtain a pass. A future qualification study needs a new plan and
independently reviewed, unexposed controls after quality gates are agreed.

## Laptop verification

The focused evaluator checks passed 357 tests, including 58 new hermetic
Japanese boundaries. Full backend verification: 4,031 passed, with one existing
FastAPI HTTP422 deprecation warning, in 76.21 seconds. Scoped Ruff and diff
checks passed.
Router responses are mocked: these tests make no live requests, read no real
uploads, and provide no new model-quality or speed measurement.

The single exposed case also passed the offline CLI: revision 4, one guarded
abstention, zero model calls, errors, or false activations. This proves the
changed guard, rather than claiming the rules' earlier undecided `none` was
already a complete fix.

## Targeted Tower check

After updating the Tower checkout, run this once **in Tower's shell**. Keep the
shell open until its copyable JSON appears. No file upload, Audrey token, new
model, container rebuild, or browser action is needed. This direct Ollama check
uses Docker DNS, so neither the laptop LAN route nor Tailscale is involved.
Leave Audrey idle while the retained local model serves the two short requests.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
FOREGROUND=1 scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_japanese_regressions.json \
  ARGS='--backend hybrid --config /app/config.yaml --cases /tmp/skill_selection_japanese_regressions.json --repeats 1 --summary'
```

The [four-case development fixture](../../evals/cases/skill_selection_japanese_regressions.json)
contains the exposed rewrite, a quoted filename with a real access denial, a
question about quoted text inside a PDF, and a denied PDF followed by a separate
video request. Independent agent review confirmed its labels and the expected
two-call budget. It is a regression check, not a prospective quality study.

Success: exit zero, `status: passed`, `measurement_status: completed`,
`rules_revision: 4`, `case_count: 4`, `selection.correct: 4`, two correct
activations, zero misses/ordinary false activations/errors, `guarded: 2`,
`model_called: 2`, and `findings: []`. Copy the JSON from the launching shell.
It contains all findings and measured usage; no remote report transfer is
needed. The runner saves its log and skips Telegram unless `NOTIFY=1` is requested.

If a request fails, preserve that output and diagnose only that boundary. Do
not expand to a broad study or repeat the settled earlier checks. Passing this
regression gate does not approve production activation. Human label review,
agreed quality/cost gates, and actual final-answer/workflow benefit remain open.
