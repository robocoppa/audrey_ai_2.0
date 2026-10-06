# Campaign 3 Slice 3D.2 — evaluation policy refinement

**Status:** Laptop-complete, 2026-10-06. Full hermetic backend: 3,845 passed;
focused evaluator/policy/runner gate: 203 passed. Automatic runtime selection
remains disabled. The first new-control measurement completed on Tower with
three selection findings; see the received result below and the
[Slice 3D.3 repairs](phase-03-skill-selection-terminal-results.md).

## Why this slice exists

[Slice 3D.1](phase-03-skill-selection-evaluation.md#repeated-hybrid-evidence-2026-10-06)
collected two identical repeated studies. The hybrid recognized more genuine
analysis requests but lowered valid-activation precision to 63.16%, falsely
selected skills for 15/42 ordinary requests, and produced three ineligible
choices per study. It never abstained on its 30 model calls.

The rules used `none` for two different outcomes: a request that should receive
no skill, and a request the rules could not decide. The hybrid sent both kinds
to the model when the file metadata was eligible. Excluded attachment names also
remained in the evidence target set.

## Changes to the evaluator

- A firm abstention carries `terminal_abstention: true` and a `policy_reason`.
  It ends selection in both direct-router and hybrid study arms, with no model
  call. Unsupported modes and unresolved or unavailable evidence also end
  selection.
- An undecided eligible request can still reach the retained router. Rules do
  not treat every unknown phrase or language as a user prohibition.
- Request scope distinguishes an excluded attachment from evidence the user
  asks to analyze. Excluding a file cannot substitute unrelated ready evidence
  for a missing or processing target.
- Quoted commands remain content. A quoted exact filename can still identify
  evidence. Restrictions on editing, inventing numbers, or other unrelated
  actions do not by themselves prohibit analysis.
- Reports retain invalid model choices as errors and keep model latency
  separate from immediate guard decisions. Revision and case-set fingerprints
  distinguish later reports from the preserved revision-1 measurements.

This is a standalone study policy. It is not connected to Audrey request
handling, the native browser, or runtime skill selection. Existing explicit
skills and production task routing retain their prior acceptance evidence.
The rules are conservative lexical checks rather than a complete natural
language parser. Ambiguous intent and unsupported wording still need measured
review before any activation decision.

## Reserved proposed controls

[The new fixture](../../evals/cases/skill_selection_holdout.json) contains 30
synthetic proposed controls: 14 positive, eight ambiguous, and eight ordinary
requests. A separate agent authored it without opening the selector,
development cases, or model logs. Required repository-state reading exposed
summary study results. These are separately authored proposed labels, not a
validated or blind human-authored holdout.

Revision 2 implementation and regression tests did not inspect or evaluate
these prompts before the first measurement. That measurement is now received
and reviewed. Slice 3D.3 repairs its three exposed findings, so this set now
serves as regression evidence, rather than an untouched validation set. The
original fixture and all earlier reports remain preserved.

Reserved fixture file SHA-256:
`098ac965713ea811ec9f90a32fb66fbc43061dbb20375258636ceeeeb80af4ba`.
The report's `case_set_sha256` fingerprints the canonical evaluated cases;
it is distinct from this raw-file checksum.

## Probe launch identity

The detached wrapper now retains the parent timestamp and logfile. Previously,
the child read the clock again; a second boundary could make Telegram reference
another path while stdout still wrote to the original file. Behavioral tests
force that boundary using local fake commands and verify the announced log,
captured output, and attached document match. Telegram delivery time still does
not establish execution time; use the report and log timestamps.

## Laptop verification

The full hermetic backend passed 3,845 tests in 75.50 seconds, with one existing
FastAPI HTTP422 deprecation warning. The focused evaluator/policy/runner gate
passed 203 checks. Scoped Ruff, Python compilation, shell syntax, and diff checks
passed. ShellCheck is unavailable on the laptop; shell syntax and behavioral
runner tests provide the available shell verification.

The unchanged 42-case development fixture now produces these offline results:

| Rules-only measurement | Result |
|---|---:|
| Correct labels overall | 37 / 42 |
| Correct activations / all activations | 9 / 9 |
| Valid false activations, including ordinary controls | 0 |
| Positive requests still undecided | 5 / 14 |
| Firm abstentions | 28 |
| Actual model calls / tokens | 0 / 0 |

The five undecided positives are the contradiction question, Spanish/French
document requests, and Portuguese/German video requests. They remain eligible
for an optional model decision. With this development set, a three-repeat hybrid
would now make 15 model calls; this is a calculated budget, not a new live
hybrid result. Nine correct rule activations do not establish production
precision. These numbers preceded the new-control measurement recorded below.

Private final development artifact:
`testing-out/evals/2026-10-06-skill-selection-rules-v2-final.json`.
No live model call or deployed behavior was tested by the laptop gate.

## First measurement on Tower — completed with findings

The historical command below produced the received revision-2 report. Do not
repeat it unchanged; use Slice 3D.3's targeted foreground check for the repairs.

Update the Tower checkout to include this slice, then run the command once
while Audrey is idle. No video/PDF upload, browser action, Audrey token, model
pull, or container rebuild is needed. This uses direct Ollama through Docker
DNS; the laptop LAN/Tailscale route is not involved.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_holdout.json \
  ARGS='--backend hybrid --config /app/config.yaml --cases /tmp/skill_selection_holdout.json --repeats 1'
```

This evaluates 30 new controls once. It makes at most 30 short model requests;
firm abstentions and positive rule choices make no model call. Finishing in
seconds is possible. The printed logfile identifies this run, and Telegram
sends its usual summary and full report after execution.

Check that provenance reports `rules_revision: 2`, `case_count: 30`, and
`repeats: 1`. A completed measurement has `status: completed` and zero model
errors; that alone does not establish acceptable selector quality. The report
must keep false activations, missed positives, firm abstentions, actual model
calls, and model-only latency visible. Zero model calls supplies no model-speed
or model-choice evidence.

Send the resulting log for review. Any incorrect selection or missed positive
is a finding to retain, not a reason to change its label. Follow-up repetitions
or targeted comparisons depend on this first measurement. Independent human
label review, agreed quality/cost limits, and final-answer benefit remain
required before automatic runtime activation.

## Received result — October 6, 2026

The 15:55:26 Tower log reports revision 2, 30 cases, one repeat, retained
`qwen3.5:4b`, and automatic selection disabled. Its status is `completed`, with
zero transport, format, or ineligible-choice errors. Under this older CLI,
exit zero indicated execution completion, even with valid incorrect choices.
The raw fixture checksum matches the reserved fixture; the canonical case hash
is `26b27546a68ca68f145ee637a6da970809046f35d7196059cfb20d6fad243b97`.

| Proposed-label measurement | Rules | Hybrid |
|---|---:|---:|
| Correct labels | 21 / 30 | 27 / 30 |
| Correct activations / activations | 5 / 5 | 12 / 13 |
| Missed positive requests | 9 / 14 | 2 / 14 |
| Ordinary false activations | 0 / 8 | 1 / 8 |
| Actual model calls | 0 | 8 |

The hybrid blocked a document question locating the phrase “Stop the recording”
and a video description request that excluded an unrelated poster. It also
activated video analysis for a Spanish request to move a file into a project
and rename that project. A separate agent reviewed the three mismatches and
retained their proposed labels. Human label validation remains outstanding.
All eight ambiguous controls abstained correctly. The model recovered seven
other positives and abstained on none of its eight calls.

Model-only median latency was 0.167845 seconds and nearest-rank p95 was
7.506008 seconds; the first request took 7.506008 seconds. Loading was
uncontrolled. Usage totaled 2,261 input and 80 output tokens. These are eight
observations from one run, not controlled warm/cold timing or answer-quality
evidence. Proposed production thresholds remain unagreed and unmet.

[Preserved full report](../../evals/results/2026-10-06-skill-selection-policy-v2-holdout-results.json),
created `2026-10-06T21:55:36.393253+00:00`; source Tower log
`2026-10-06-155526-eval_skill_selection.log`. The measurement is complete;
automatic activation remains deferred. The [next bounded repair](phase-03-skill-selection-terminal-results.md)
addresses the three findings and restores copyable foreground shell results.
