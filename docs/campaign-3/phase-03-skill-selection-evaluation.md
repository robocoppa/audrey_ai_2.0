# Campaign 3 Slice 3D.1 — skill selection evaluation foundation

**Status:** Laptop-complete, 2026-10-06. Full hermetic backend verification:
3,725 passed. The one-case live pilot passed on 2026-10-06. Two repeated
42-case studies subsequently collected all calls but exposed unsuitable
selection behavior. Automatic selection remains disabled. [Slice 3D.2](phase-03-skill-selection-policy.md)
now adds firm abstention and scoped evidence resolution; its new-controls live
measurement is pending.

## Purpose

Measure when Audrey should suggest one of its two existing skills before adding
any automatic behavior. The earlier explicit-skill answer tests remain passed;
they did not measure whether a selector activates the right skill.

This slice adds a synthetic labeled fixture and a standalone study harness.
It does not change Audrey's browser, request handling, skill prompts, router
configuration, or selected production models.

## What the study measures

[The fixture](../../evals/cases/skill_selection_cases.json) contains 42 cases:

- 14 clear requests: seven documents and seven videos;
- 14 ambiguous requests where the proposed selector should abstain;
- 14 ordinary-chat controls where no skill should activate.

Six cases use languages other than English. Controls include quoted instructions,
negation, unrelated attachments, missing or processing files, incompatible modes,
and mixed relevant media. Every case includes a human-reviewable label and reason.
The fixture contains synthetic filenames and metadata, with no file contents.

The proposed study policy prefers `grounded-document-analysis` for document
questions and `video-analysis` for video questions. It abstains when readiness,
identity, mode, or the relevant media set is uncertain. Image/audio-only and
mixed-media abstention are conservative study labels; they do not redefine
Audrey's existing file-analysis capabilities. These labels need review before
using them as a production selection contract.

## Evaluation methods

`evals/eval_skill_selection.py` supports:

| Backend | Behavior |
|---|---|
| `rules` | Offline conservative rules; no model or network access |
| `router` | Compare the rules baseline with the retained router's eligible decisions |
| `hybrid` | Use rules first; ask the retained router only where rules abstain and a skill is eligible |

The router receives only catalog metadata, the user request, file metadata, and
mode. Gold labels, reasons, full skill instructions, resources, and document
contents are excluded. Unsupported modes and unavailable eligible evidence are
recorded as guarded abstentions rather than successful model decisions.

Model calls run serially with bounded output and timeouts. Live evaluation
requires `skills.auto_select: false` and uses the configured retained local router,
`qwen3.5:4b`. It does not unload models, change settings, execute tools, or activate
skills. Run it while Audrey is idle to avoid adding selector traffic to real work.

Reports keep precision, false activation, missed activation, ordinary-chat false
activation, abstention, model errors, latency, and token usage separate. A wrong
skill counts as both a false activation and a missed useful selection. No
activations means precision is undefined, not 100%. Invalid output and provider
errors are not successful abstentions. Model-reached and guarded samples have
separate provenance and denominators.

A completed report means the study collected its measurements. It does not mean
the selector passed a deployment gate. Poor decisions remain visible in the
report. Output saved with `--save-json` is private and never overwrites an
existing artifact.

## Small live check on Tower

No upload, browser action, Audrey token, model pull, or container rebuild is
needed. Update the Tower checkout first. The supported runner copies the new
harness and fixture into the existing Audrey container and preserves its log
under `testing-out/probes`. It reaches Ollama through Docker DNS; laptop
Tailscale/WARP reachability is irrelevant to this direct model study.

Run on **Tower**:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_cases.json \
  ARGS='--backend hybrid --config /app/config.yaml --only positive-document-spanish --repeats 1'
```

This checks one Spanish document request that the conservative English rules
abstain on. It makes one router call using synthetic ready-document metadata.
The runner prints a log path and sends its usual completion notification.

A successful transport check has a completed report, zero model errors, and one
model-reached sample. The expected choice is `grounded-document-analysis`.
If the choice is `none` or another skill, keep the report: that is a measured
selector miss, not a reason to enable it or rerun all previous smokes.

## Full measurement after the small check

The pilot and two repeated studies are settled on 2026-10-06. The command
below records how those studies ran; do not repeat unchanged measurements.
Future measurements should follow the policy refinement described below.
When needed after that change, the runner executes on Tower while Audrey is idle:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_cases.json \
  ARGS='--backend hybrid --config /app/config.yaml --repeats 3'
```

This repeats the 42 cases three times. With the current fixture and rules,
it makes 30 router calls (10 eligible abstentions per repeat). Rules make no
model calls; the hybrid asks the router only for eligible rule abstentions.
The current findings require policy refinement before another model arm or
candidate comparison. No broad answer-quality rerun is needed for this study.
A synthetic development fixture cannot establish production precision or show
that selected skills improve final answers.

## Laptop evidence, 2026-10-06

The full hermetic backend suite passed 3,725 tests with one existing FastAPI
HTTP422 deprecation warning. The focused evaluator/runner modules passed 83
checks. Scoped Ruff, Python compilation, shell syntax, and diff checks passed.
Coverage includes metadata-only requests, hard guards, model truncation/errors,
wrong-choice accounting, serial repetitions, model calls only on eligible rule
abstentions, and private reports that refuse overwrite. The on-box wrapper was
exercised with Docker replaced by a local stub; no live service was called.

The final offline rules report completed with 42 valid observations and no
infrastructure errors:

| Measurement | Result |
|---|---:|
| Correct labels overall | 32 / 42 |
| Correct activations / all activations | 7 / 10 (70% precision) |
| Useful selections missed | 7 / 14 |
| False activations | 3 |
| Ordinary-chat false activations | 2 / 14 |
| Model calls / tokens | 0 / 0 |

The false activations were an unresolved report identity, translation of a
quoted instruction, and a question about the delete button. Misses included
four non-English positives, two explicit mixed-attachment subsets, and a
question about conflicting document sections. These expose policy limitations;
the rules are not ready for automatic activation. Hybrid can fill rule
abstentions but inherits positive rule mistakes, so it cannot by itself repair
those false activations.

Local result: `testing-out/evals/2026-10-06-skill-selection-rules-final.json`.
This is a development fixture, not an independent production quality estimate.
No live model was called by the laptop verification.

## Live pilot evidence, 2026-10-06

Tower log: `2026-10-06-105732-eval_skill_selection.log`. Its report was created
at `2026-10-06T16:57:40.340643+00:00`. The [retained result](../../evals/results/2026-10-06-skill-selection-pilot-results.json)
records exactly one `positive-document-spanish` case, one repeat, and one
`qwen3.5:4b` call. Rules abstained; the hybrid called the router and selected
`grounded-document-analysis`, matching the label. Output was valid, with zero
errors, retries, wrong choices, or missed selection in this case. Both harness
activation and configured runtime auto selection remained false.

The call took 7.35355 seconds and used 277 input / 11 output tokens with
temperature 0 and a 128-token output ceiling. Model loading was not controlled,
so this single elapsed time does not establish warm or cold latency. The
`conditional_router` section repeats the same sample; it is not another call.
No ordinary or ambiguous case was included, so this pilot supplies no evidence
about false activation. The repeated study below supplies those development
measurements. The pilot is settled; do not repeat it for unchanged behavior.

## Repeated hybrid evidence, 2026-10-06

Both Tower logs contain all 42 cases repeated three times, 126 decisions per
full arm, and 30 actual model calls per study. Preserve the reports separately:

- [First report: 110216](../../evals/results/2026-10-06-skill-selection-hybrid-110216-results.json), created at `2026-10-06T17:02:22.219753+00:00`.
- [Second report: 110258](../../evals/results/2026-10-06-skill-selection-hybrid-110258-results.json), created at `2026-10-06T17:03:04.733490+00:00`.

Decisions, errors, and token counts match exactly across repeats and both
studies. Wrapper timestamps show the first finished at 17:02:22 UTC and the
second started at 17:02:58 UTC; these runs did not overlap. They use the same
42 development cases, so the second run is replication, not an independent
holdout. The conditional-router section duplicates model-reached observations
from the hybrid; it does not represent additional calls.

| Per-study measurement | Rules | Hybrid |
|---|---:|---:|
| Correct decisions overall | 96 / 126 | 96 / 126 |
| Correct activations / all valid activations | 21 / 30 (70%) | 36 / 57 (63.16%) |
| Useful selections recovered | 21 / 42 (50%) | 36 / 42 (85.71%) |
| Useful selections missed | 21 / 42 | 6 / 42 |
| Valid false activations | 9 | 21 |
| Ordinary-chat false activations | 6 / 42 | 15 / 42 |
| Invalid selections | 0 | 3 |
| Model calls | 0 | 30 |

Each report exits 1 because `ambiguous-video-negated` returns
`grounded-document-analysis` in all three repeats, although only video evidence
is eligible. The post-response eligibility check rejects that choice. There
are no reported transport, timeout, malformed-JSON, or truncation failures.
Those three rejected attempts remain separate from the 21 valid false
activations; they are not counted as successful abstentions.

Among the 30 model-reached observations, the model activates for every request:
15 correct positive selections, 12 valid false activations, and three rejected
ineligible attempts. It never returns `none`. The five recovered positive cases
cover conflicting document sections, Spanish/French document requests, and
Portuguese/German video requests. New valid false choices concern negated
document analysis, file renaming, an unrelated attachment, and project
organization. The hybrid also inherits the rules' unresolved-report, quoted
instruction, and delete-button mistakes. The remaining six missed positives
are two explicitly selected attachment subsets repeated three times; the
guard treats an explicitly excluded file as another target.

For the second study, model-only request latency is 0.16197s median and
0.21186s p95 over 30 calls. The first study records 0.16070s / 0.20747s.
Each uses 8,223 input / 318 output tokens; median usage is 273 input / 11 output
tokens. Whole-hybrid median latency is zero because guards return immediately;
it is not a model-speed measurement. Temperature is 0, output ceiling is 128,
retries are zero, and loading is uncontrolled. These are observed request
timings, not proven warm/cold benchmarks or candidate comparisons.

**Decision:** The foundation collected useful evidence, but the current rules
and hybrid do not support automatic activation. Both harness and configured
runtime auto selection remain false. Keep the existing production task router
and explicit skills unchanged; this study measures a different decision.

## Follow-up: 3D.2 evaluation policy refinement

[Slice 3D.2](phase-03-skill-selection-policy.md) is laptop-complete: 3,845 backend
checks passed. It implements the policy refinements below and reserves a
separately authored 30-case proposed control set for its first measurement.
Automatic runtime activation remains deferred; the revision-1 evidence above
is preserved.

1. Separate a terminal abstention from an undecided rules result. Explicitly
   forbidden file analysis, quoted commands, interface-management questions,
   and unresolved targets must not fall through to a model that activates.
2. Resolve explicitly excluded attachments conservatively while preserving
   readiness, supported-mode, and evidence-identity checks.
3. Add independently reviewed controls before refining rules or model prompts;
   retain an untouched holdout and preserve this revision-1 baseline. Do not
   change labels merely to make current decisions look correct.
4. Measure affected cases and new controls after the change. Restricting the
   output enum alone is insufficient: it could replace a rejected document
   choice with an unwanted video choice while concealing the underlying error.
5. Agree quality and cost gates before testing automatic behavior. No unchanged
   pilot/full-study rerun or new model candidate is requested by these findings.

## Requirements before enabling automatic selection

A future activation slice must add an independently reviewed holdout set and
real workflow evidence. Proposed starting thresholds are at least 95% activation
precision, zero ordinary-chat false activations in repeated controls, and at most
10% missed useful selections. These are proposals, not an agreed deployment gate.
Latency/token ceilings, answer-quality comparison with explicit selection, and
visible reversible selection remain required. Keep `skills.auto_select: false`
until those decisions and repeated measurements are accepted.
