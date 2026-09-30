# Campaign 3 Phase 6 - System One decision routing

**Status:** Planned after the Campaign 3 Phase 5 Slice 5B laptop and live gate.
No production router or Ollama deployment change has been made.

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
Audrey will call the HTTP endpoint directly through its existing async client
rather than add the TypeSafe SDK.

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

Add a narrow System One method to `OllamaClient` or a probe-local equivalent
and extend the router probe without changing production classification.

For every candidate and the incumbent, record:

- selected label and agreement with the expected label;
- the full option distribution, winning probability, first-to-second margin,
  and System One `confidence`;
- cold latency plus warm p50 and p95 latency;
- transport, HTTP, and response-shape failures;
- model size, observed residency, and whether it evicts an active worker;
- expensive misroutes into `reasoning`, separately from nearly free
  `general`/`code` swaps;
- the projected number of fast-to-deep escalations under any proposed mapping.

System One defines `confidence` as distribution concentration, not the chance
that the label is correct. Do not copy it into `classify_confidence` or reuse
Audrey's existing 0.95 threshold. Evaluate winning probability, margin, and
abstention behavior against labeled cases before choosing a mapping.

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
