# Campaign 3 Phase 6 - System One decision routing

**Status:** In progress. Slice 6A is laptop-complete and awaits its first
`tev1:0.8b` on-box measurement. No production router, config, or Ollama
deployment change has been made.

## Goal

Evaluate Ollama's `/v1/systemone` decision-model API as a typed replacement for
the generative JSON call used by Audrey's task router. A candidate ships only
if Audrey-specific evidence shows that it preserves routing quality, latency,
GPU residency, and deep-panel cost.

Official references:

- https://docs.ollama.com/api/systemone
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

1. `tev1:0.8b` first. Its small footprint is the best match for Audrey's
   ungated router slot.
2. `tev1:4b` second. It is the closest purpose-built size comparison with the
   incumbent 4B router.
3. `nimble` only if the smaller candidates leave a material quality gap. Its
   9.5 GB package is a poor default for a router that can evict an active local
   worker under `GPU_CONCURRENCY=1`.

Ollama 0.35 or newer is a prerequisite. Confirm the deployed version before
pulling candidates. Model presence and a successful endpoint request are
deployment facts and require a user-confirmed live check.

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

### First on-box measurement

This is an Ollama-host probe, not a browser or Audrey API smoke. It needs no
uploaded image, video, document, login token, or manual UI step. Run it while
Audrey is idle because cold-load measurement changes model residency.

First confirm the externally managed Ollama container is version 0.35 or newer:

    docker exec ollama ollama --version

Install the first candidate if it is not already present:

    docker exec ollama ollama pull tev1:0.8b

After the new probe files reach the Unraid checkout, start the detached run:

    cd /mnt/user/appdata/audrey_ai_2.0
    scripts/probes/probe-onbox.sh systemone_router_probe.py \
      COPY=systemone_router_cases.json CANDIDATES=tev1:0.8b

The command prints a log path and returns immediately. The probe makes 74 model
calls: one cold plus 36 warm calls for the candidate, then the same for the
incumbent. Completion is exit code zero and a JSON report with
`"status": "measured"`. That status means evidence collection worked; it is
not an automatic ship decision. Review candidate versus incumbent accuracy,
false `reasoning` routes, projected escalations, p50/p95 and cold latency, and
residency displacement before opening Slice 6B. Record the accepted numbers in
`evals/MODEL-FACTS.md` in the same session.

## Slice 6B - guarded production backend

Open this slice only after one candidate clears 6A.

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

If no candidate clears every item, retain `qwen3.5:4b` and record the measured
result. The experiment remains useful even when it rejects the new backend.

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
