# Campaign 3 Phase 6 - System One decision routing

**Status:** The original comparison completed on 2026-09-30 and retained
`qwen3.5:4b`. A probe-only Clef and Clef Flash follow-up is prepared and awaits
live measurement; production routing remains unchanged.

## Goal

Evaluate Ollama's `/v1/systemone` decision-model API as a typed replacement for
the generative JSON call used by Audrey's task router. A candidate ships only
if Audrey-specific evidence shows that it preserves routing quality, latency,
GPU residency, and deep-panel cost.

Official references:

- https://docs.ollama.com/api/systemone
- https://ollama.com/library/clef
- https://ollama.com/library/clef-flash
- https://ollama.com/library/tev1
- https://ollama.com/library/nimble

System One accepts a state plus named `choice`, `noul`, or `score` questions.
For Audrey's first experiment, one `choice` question selects among `code`,
`reasoning`, `general`, and `vl` and returns a probability distribution.
Slice 6A calls the HTTP endpoint through a narrow probe-local async client
rather than adding the TypeSafe SDK or changing the production Ollama client.

## Why this fits Audrey

The current router asks `qwen3.5:4b` for JSON containing a task label and a
self-reported confidence. Audrey constrains that output with a schema, parses
it, retries malformed responses, and falls back to keyword or `general`
routing. A decision model can provide the label and distribution as a typed API
result, removing generated-JSON format risk from this boundary.

The incumbent remains the gate, not a generic benchmark. Its recorded
production-matching probe produced 24/30 correct labels, 0.48-second median
latency, and 27/30 confidence values at or above Audrey's 0.95 escalation
ceiling. The comparison must use the same cases and an expanded labeled routing
set because ten prompts are only a feasibility smoke.

## Boundaries that remain unchanged

- Strong keyword signals still short-circuit the model.
- Short prompts still use the configured cheap-route rule.
- Explicit tool names still route through the tool-capable general path.
- A `vl` result without an image is still demoted to `general`.
- Router failure still degrades through the existing weak-keyword and general
  fallbacks.
- Explicit virtual-model choices and the deterministic fast/deep complexity
  ordering remain authoritative.
- The production default remains the incumbent router until the evaluation
  gate passes, with one configuration setting restoring it afterward.

## Candidate order

1. `tev1:0.8b` is the 811 MB footprint baseline and the best physical match
   for Audrey's ungated router slot.
2. `tev1:latest` is the 4B Tev1 comparison, close to the incumbent's size.
3. `nimble:latest` is the 9B comparison. Its 9.5 GB package may buy accuracy,
   but the larger residency cost remains part of the gate rather than being
   accepted from generic benchmark results.

The completed run confirmed Ollama 0.35.0 and every exact candidate tag before
generation. Model presence and successful endpoint behavior are live evidence
from the deployed Ollama host.

## Slice 6A - probe-only adapter

This slice adds a probe-local System One client and an expanded router
comparison without changing production classification.

For every candidate and the incumbent, record:

- selected label and agreement with the expected label;
- the full option distribution, winning probability, first-to-second margin,
  and System One `confidence`;
- cold latency plus warm p50 and p95 latency;
- transport, HTTP, and response-shape failures;
- model size and observed residency displacement; concurrent active-worker
  eviction remains a separate ship-gate observation;
- expensive misroutes into `reasoning`, separately from nearly free
  `general`/`code` swaps;
- the projected number of fast-to-deep escalations under any proposed mapping.

System One defines `confidence` as distribution concentration, not the chance
that the label is correct. Do not copy it into `classify_confidence` or reuse
Audrey's existing 0.95 threshold. Evaluate winning probability, margin, and
abstention behavior against labeled cases before choosing a mapping.

### Slice 6A implementation

`scripts/probes/systemone_router_probe.py` is intentionally probe-local. It
uses Audrey's real `router_classify` path for `qwen3.5:4b` and calls
`/v1/systemone` directly for candidates, so the experiment adds no production
backend or dependency. `evals/cases/systemone_router_cases.json` preserves the
original ten prompts and adds twenty-six unique cases, balanced at twelve each
for `code`, `reasoning`, and `general`. `vl` remains an output option but is not
an expected case because current attached-image turns bypass the router.

For every warm sample the report retains the selected label, expected label,
full probability distribution, winner probability, first-to-second margin,
System One concentration, latency, and validity. It separately totals false
routes into costly `reasoning`, missed reasoning cases, cheap code/general
swaps, and projected fast-to-deep escalations under an explicitly provisional
winner-and-margin rule. The incumbent projection continues to use its actual
0.95 self-reported-confidence ceiling, so the two confidence meanings never
share a column.

The probe checks Ollama's version and exact installed tags before generation.
It deliberately unloads each model for one cold sample, records `/api/ps`
before and after, runs candidates first, and runs the incumbent last so Audrey's
current router is warm when it finishes. Residency displacement is evidence
about loading; it is not a concurrent-contention proof.

### Laptop verification

- 58 focused new and incumbent router-probe contract tests pass.
- The full hermetic suite passes: 3,002 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff and standalone probe compilation pass.
- `config.yaml`, the production classifier, and `OllamaClient` are unchanged.

### On-box measurement and decision

The single three-candidate run completed on Ollama 0.35.0 with all 148 calls
valid and no transport, HTTP, or response-shape failures. The full report is
[`2026-09-30-systemone-router-probe-results.json`](../../evals/results/2026-09-30-systemone-router-probe-results.json).

Raw results across all 36 balanced fixture cases:

| Model | Correct | Costly false reasoning | Missed reasoning | Projected escalations | Warm p50 / p95 | Cold |
|---|---:|---:|---:|---:|---:|---:|
| `tev1:0.8b` | 22/36 | 14 | 0 | 8 | 0.089s / 0.092s | 4.98s |
| `tev1:latest` | 30/36 | 4 | 0 | 4 | 0.169s / 0.174s | 5.43s |
| `nimble:latest` | 33/36 | 2 | 1 | 2 | 0.200s / 0.215s | 15.21s |
| `qwen3.5:4b` | **34/36** | **1** | **0** | **1** | **0.184s / 0.199s** | **11.06s** |

Thirteen fixture prompts are resolved before a production router call: nine by
the strong keyword gate and four by the short-prompt rule. On the 23 cases that
actually reach the model, the result is clearer:

| Model | Correct | Costly false reasoning | Cheap code/general swaps | Uncertain |
|---|---:|---:|---:|---:|
| `tev1:0.8b` | 13/23 | 10 | 0 | 7 |
| `tev1:latest` | 18/23 | 3 | 2 | 3 |
| `nimble:latest` | 22/23 | 1 | 0 | 1 |
| `qwen3.5:4b` | **23/23** | **0** | **0** | **0** |

Nimble was the closest candidate. Its one reachable error sent the SQL window
function case to `reasoning`; its low winning probability and margin would also
abstain under the provisional mapping. That still adds one costly route and one
escalation relative to the incumbent. Nimble's warm p50 was about 9% slower,
its cold load about 38% slower, and its observed residency was 8.97 GB versus
4.20 GB for the incumbent. Loading Nimble displaced Tev1 4B; loading the
incumbent then displaced Nimble. The existing 19.64 GB `qwen3.8:32k` residency
survived every stage, so this is observed sequential displacement rather than a
concurrent contention result.

The deterministic gates themselves disagree with two legacy fixture labels and
route both cases to `reasoning`; that shared behavior does not distinguish the
models. With those gates applied end to end, Nimble remains at 33/36 with three
costly false-reasoning outcomes, while the incumbent remains at 34/36 with two.

**Decision:** retain `qwen3.5:4b`. No candidate matched its production-reachable
accuracy, escalation behavior, and footprint. A concurrent worker-contention
test would not rescue a candidate that already failed the earlier accuracy and
cost gates, so no further live smoke is required for the original candidates.

## Clef follow-up evaluation — prepared 2026-10-03

Ollama now publishes the 27B `clef:latest` and latency-focused 9B
`clef-flash:latest` decision models. Both use `/v1/systemone`, require Ollama
0.35.1 or later, and are evaluated as decision models rather than ordinary chat
generators. This follow-up is measurement only: it does not register either
model, edit `config.yaml`, or open Slice 6B.

The Audrey router comparison reuses the same 36-case fixture and the real
incumbent path. Both Clef tags run before `qwen3.5:4b`, and the probe records
the original accuracy, costly false-reasoning, uncertainty, latency, token,
package-size, and residency fields. Clef or Clef Flash must match the
incumbent's 23/23 model-reached accuracy without adding a costly reasoning
route or escalation before footprint or speed can justify a production trial.

`scripts/probes/systemone_decision_probe.py` adds a separate broad System One
measurement for Clef. Its tracked fixture contains 12 cases and 29 questions:
11 choice, 12 yes/no, and 6 score decisions across eight text inputs, three
structured JSON states, and one generated image. It reports exact-case and
per-question accuracy, calibration signals, score-range error, input/output
tokens, cold and warm latency, package metadata, and residency. This benchmark
tests Clef's wider decision contract; it does not claim conversational answer
quality because the endpoint does not generate chat answers.

Run the broad Clef probe first and the router comparison second using the exact
Tower commands in `docs/reference/live-smoke-testing.md`. The broad probe
unloads Clef after measurement; the router probe runs the incumbent last so
Audrey's current router is warm at the end. Live results remain pending and no
ship decision should be recorded until both logs have been reviewed.

## Slice 6B - not opened

No candidate cleared Slice 6A. The implementation conditions below remain the
gate if a later model merits another evaluation.

- Add an explicit `chat` versus `systemone` router backend setting, defaulting
  to the incumbent behavior during development.
- Keep the common keyword, short-prompt, tool-name, image, retry, and fallback
  logic around both backends.
- Validate the typed response defensively; an incomplete distribution or an
  unknown choice is a router strike.
- Preserve current routing reason and observation fields while naming the
  backend in logs and bounded metrics.
- Switch the deployed default only after the targeted live probe confirms the
  model, endpoint, latency, and fallback path on Audrey's actual Ollama host.

## Ship gate

A candidate must demonstrate:

- no meaningful accuracy regression on the expanded Audrey routing set;
- no increase in costly false `reasoning` routes;
- a measured uncertainty mapping that does not inflate deep-panel traffic;
- latency acceptable on Audrey's hardware, including cold load;
- no harmful eviction or contention with active fast and deep workers;
- clean failure fallback when System One is unavailable or malformed;
- full hermetic tests and changed-file Ruff before the live gate.

The 2026-09-30 run rejected all three candidates. Audrey retains
`qwen3.5:4b`, and the measured rejection closes this phase without a production
backend change.

## Later skill-selection experiment

If a small candidate proves reliable, Phase 3 Milestone 3D may evaluate the
same typed-choice mechanism against `none`, `grounded-document-analysis`, and
`video-analysis`. That is a separate precision and abstention study using only
catalog metadata and deterministic attachment compatibility checks.
`skills.auto_select` remains false until that study passes its own gate.

## Non-goals

- Replacing Audrey's deterministic complexity gate.
- Inferring file or media types already known from validated metadata.
- Moderating read-only tools without a demonstrated product need.
- Using a decision model as the final judge of answer correctness or quality.
