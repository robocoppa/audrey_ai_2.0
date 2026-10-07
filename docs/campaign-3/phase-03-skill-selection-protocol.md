# Campaign 3 Slice 3D.4 — frozen prospective skill-selection studies

**Status:** Laptop-complete, 2026-10-06. Tower preparation and the first
measurement remain pending. Automatic selection stays disabled. This slice
changes only the standalone evaluation harness and its study documents.

## Starting evidence

The [3D.3 targeted Tower check](phase-03-skill-selection-terminal-results.md#received-tower-acceptance--october-6-2026)
passed on October 6 at 22:23:57 UTC: three of three proposed labels matched,
two correct activations, one guarded abstention, zero errors or model calls,
and zero tokens. This settles that regression check. It does not supply a new
Qwen performance measurement. The earlier measured controls remain exposed
regression data; do not repeat them to claim untouched validation.

## Freeze before measuring

`evals/eval_skill_selection.py --prepare-plan PATH` creates a private plan
without constructing an HTTP client or making model requests. It binds:

- evaluator source SHA-256, covering the rules, prompt, schema, and protocol;
- canonical case-set and skill-catalog metadata SHA-256;
- retained local model tag, Ollama origin, backend, repeats, and timeout;
- temperature zero, at most 128 output tokens, serial calls, and no retries;
- the expected model-call count, maximum call budget, and proposed criteria;
- explicit operator claims about label review and agreed gates, false by default.

`--plan PATH` checks these inputs before constructing the model client. Drift,
conflicting overrides, case filtering, malformed plans, and an existing output
report are rejected before HTTP. Plans and full reports refuse overwrites and
are created with mode 0600. No runtime configuration is changed.

The plan freezes the model **tag**, not an independently verified weight digest.
It does not control cold loading or authenticate human approval. The review
flags are operator attestations; neither a prepared plan nor a successful
measurement grants them. A later source change requires a new plan and a clear
explanation of the change, rather than silently reusing the old result.

## Fresh proposed controls

There are 24 new synthetic cases: twelve positive requests (six document and
six video), six ordinary requests, and six ambiguous requests. Each supported
mode, `auto`, `fast`, and `deep`, has eight cases. The
[review packet](../testing/2026-10-06-skill-selection-prospective-review.md)
contains every proposed prompt, file metadata, label, and rationale.

The author worked separately without reading selector source, old fixtures,
model logs, or the model ledger. Required project-state guidance exposed a
summary of earlier findings. A second agent independently compared all 24
labels and rationales to the new fixture, without inspecting the selector or
running it; it found all labels defensible and no inconsistencies. This is
agent review, not a blind human-authored or human-validated holdout.

No prospective selector or model outcomes were collected on the laptop, and
the new cases have not been used to tune the selector. Rules revision 3 remains
unchanged. Human label review remains pending.

| Artifact identity | SHA-256 |
|---|---|
| Raw prospective fixture | `0e34a52f8a5ff9dab123df38aa002962f7a47de40c5147ba37cace8676e7ac71` |
| Canonical evaluated cases | `99f5a308720b4013a02f8b16c49b4de6d66be8f9204fd29bdc00df9f87650842` |
| Review packet | `cc0205668ae6c8fe5f42d568a7ebe1b531ff4659705e9fb75b0d7f319ae054a7` |

## Proposed criteria and reporting

These are study proposals, not previously agreed production acceptance limits:

| Criterion | Proposal |
|---|---|
| Activation precision | At least 95% |
| Missed positive requests | At most 10% |
| Ordinary false activations | Zero |
| Model response errors | Zero |
| Repeats | At least three |
| Evidence | Positive, ordinary, and ambiguous categories; actual model calls and activations |
| Cost control | At most 36 model calls by default; actual calls must match the planned count |
| Request duration | Frozen timeout, at most 60 seconds per call |

Preparation refuses a study whose predicted model-call count exceeds its budget.
Repeated cases are repeated measurements, not independent new examples. Small
synthetic samples cannot establish production precision. Model latency and token
usage remain recorded; agreed total-token and latency limits, controlled loading,
and final-answer/workflow benefit remain future gates.

Compact results keep three questions separate:

1. `measurement_status` reports execution completion or errors.
2. Top-level `status` reports `passed`, `findings`, or `failed` against proposed
   labels. A label pass does not approve automatic activation.
3. `qualification.criteria_status` reports proposed criteria separately from
   `qualification.status`, which stays `pending_review` while human label or
   gate attestations are missing. Inadequate evidence cannot report criteria met.

Exit 1 means a label mismatch, model error, unmet criterion, or insufficient
evidence. Setup errors exit 2. Pending review alone does not make an otherwise
adequate measurement fail. Every qualification still reports
`production_activation: false` and requires final-answer/workflow proof.
The compact view includes fingerprints and checks; the full private report
retains all samples, criteria observations, and the complete plan.

## Tower check — prepare, then measure

Update the Tower checkout first. No upload, Audrey token, new model, browser
action, or container rebuild is needed. The runner copies the evaluator and
fixture into the existing probe container and uses Ollama over Docker DNS.
Keep the foreground shell open; output is copyable in that shell and a log is
saved. Telegram is skipped unless explicitly requested.

### 1. Prepare the plan

This step makes zero model requests. Review its call budget and fingerprints.
The handoff deliberately leaves human attestations false.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
FOREGROUND=1 scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_prospective.json \
  ARGS='--backend hybrid --config /app/config.yaml --cases /tmp/skill_selection_prospective.json --repeats 3 --prepare-plan /data/c3-skill-selection-v3-prospective-plan.json --summary'
```

Expected: exit 0, `status: prepared`, `case_count: 24`, `repeats: 3`,
`model_called: 0`, `review_status: pending_review`, and
`production_activation: false`. The canonical case hash must match the table
above. `planned_router_calls` must fit `max_router_calls`; if it does not,
preparation fails before any requests and the proposed budget needs review.
If a plan already exists, preserve it and use a new descriptive filename in
both commands instead of deleting or overwriting it.

### 2. Run the first frozen measurement

Run only after preparation succeeds. Leave Audrey idle for the local-model
measurement. This performs up to the frozen call budget in serial; it may take
several minutes if loading or requests are slow. Backend, repeats, and timeout
come from the saved plan.

```bash
FOREGROUND=1 scripts/probes/probe-onbox.sh eval_skill_selection.py \
  COPY=skill_selection_prospective.json \
  ARGS='--config /app/config.yaml --cases /tmp/skill_selection_prospective.json --plan /data/c3-skill-selection-v3-prospective-plan.json --save-json /data/c3-skill-selection-v3-prospective-results.json --summary'
```

Copy the terminal JSON even if the wrapper exits 1; findings are useful results.
Check that fingerprints match the plan, actual calls match the predicted count,
and `qualification.status` is still `pending_review`. A `passed` label check or
exit 0 cannot close human review or enable automatic selection. The full report
is retained at `/data/c3-skill-selection-v3-prospective-results.json`.

Do not add `--labels-reviewed` or `--gates-agreed` merely to obtain a green
result. Those preparation flags require actual prior operator review and
agreement; the current handoff does not claim either.

## Verification and next step

Full hermetic backend gate: 3,973 passed in 77.31 seconds, with one
existing FastAPI HTTP422 deprecation warning. The new protocol module passed
85 focused checks; scoped Ruff, Python compilation, and diff checks passed.
Tests use separate synthetic fixtures and fake model responses; they do not
evaluate the new prospective cases or call live services. An independent
read-only source review found no blocking bug in drift checks, call counts,
status reporting, or the separation of evidence and operator attestations.

After the first frozen measurement, preserve its report and review findings
without relabeling cases to match results. Human label review, agreed quality
and cost limits, and an evaluation-only final-answer comparison using the
existing explicit-skill API are required before production auto selection.
The accepted 3C explicit-skill comparison is separate evidence and stays passed.
No additional Responses API capability is queued without a concrete caller need.
