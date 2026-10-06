# Campaign 3 Slice 3D.1 — skill selection evaluation foundation

**Status:** Laptop-complete, 2026-10-06. Full hermetic backend verification:
3,725 passed. Automatic selection remains disabled. The rules baseline is
measured; live router measurements are pending.

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

Only after the small model call works, run:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_cases.json \
  ARGS='--backend hybrid --config /app/config.yaml --repeats 3'
```

This repeats the 42 cases three times. With the current fixture and rules,
it makes 30 router calls (10 eligible abstentions per repeat). Rules make no
model calls; the hybrid asks the router only for eligible rule abstentions. An independent router arm
can be run later with `--backend router` if the hybrid evidence warrants it.
Do not run model candidates or a broad answer-quality suite for this foundation.

Review the saved errors, activation mistakes, latency, and usage before deciding
whether to expand the study. This small development fixture cannot establish
production precision or show that selected skills improve final answers.

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
No live router sample or selector cost measurement has run on the laptop.

## Requirements before enabling automatic selection

A future activation slice must add an independently reviewed holdout set and
real workflow evidence. Proposed starting thresholds are at least 95% activation
precision, zero ordinary-chat false activations in repeated controls, and at most
10% missed useful selections. These are proposals, not an agreed deployment gate.
Latency/token ceilings, answer-quality comparison with explicit selection, and
visible reversible selection remain required. Keep `skills.auto_select: false`
until those decisions and repeated measurements are accepted.
