# Phase 24 — `audrey_research`: a grounded, role-based deep mode

_Filed 2026-06-25. Status: PLAN ONLY, not started. Gated on the Euclid
re-test of the synth anchor (see PROJECT_STATE +16)._

## Why this exists

The Euclid "deep dive" produced a well-written answer with uncited
model-memory flourishes ("ho Geōmetritēs", "pinhole camera obscura"). Two
fixes were considered:

1. **Synth-side factual anchoring** — SHIPPED (PROJECT_STATE +16). The
   synthesizer now treats `[tool-grounded]` drafts as the factual spine and
   softens/drops claims only the non-grounded drafts vouch for. This is the
   first line of defense and applies to **all** deep modes.
2. **Worker-side grounding** — the failed Phase 23 (+10) forced *every*
   worker to search, which homogenized the panel and made Euclid worse.

This plan is the **safe form of #2**: instead of changing how the existing
panel behaves, add a **new opt-in virtual model** `audrey_research` with a
role-based pipeline (Researcher / Verifier / Writer). The user selects it
deliberately when they want a grounded, fact-disciplined answer.

### Why a dedicated mode, not a gate

The original proposal wanted a "factual prompt detector" to decide when to
force search. A dedicated mode **deletes that classifier entirely** — picking
`audrey_research` *is* the signal. No heuristic to mis-fire on "write a
birthday toast." And the existing panel's diversity (Audrey's "offer tools,
let each model decide" default) is **never touched** — the role pipeline lives
only in the new mode. This is the same additive shape as Phase 13
(passthrough), which the pipeline audit confirmed was byte-identical for every
non-passthrough path.

### Why role-based dispatch is acceptable here (but not in the default panel)

`deep_panel` is **parallel by design** — workers fire concurrently (cloud via
`gather`, local serialized through the GPU gate). A Researcher→Verifier→Writer
pipeline is **sequential** (Verifier needs the Researcher's findings; Writer
needs verified evidence). That sequential cost — and the ~2–3× wall-clock on a
single GPU — is unacceptable as a default but **fine in an opt-in mode** the
user chose precisely because they want thoroughness over speed.

## Step 0 — RE-TEST FIRST (gate) — ✅ RESOLVED 2026-06-25

**Done.** The synth anchor (+16) was deployed and the Euclid prompt re-run on
the box — **the answer was improved**, with panel diversity intact (no worker
forced to search). The gate resolves toward **"anchor sufficient"**:

> Build `audrey_research` only if you want a dedicated high-assurance mode for
> its own sake (it has value beyond Euclid). It is **no longer a bug fix** — the
> Euclid accuracy problem that motivated it is already addressed synthesis-side.

The worker-side grounding path stays deferred indefinitely; there is no signal
forcing it. If this mode is ever built, it's a feature, not a remedy.

## Architecture — LOCKED (decided 2026-06-25)

`audrey_research` is a fourth forced-deep virtual model. It reuses the deep
path's *outer* shape (forced-deep routing, planner, reflect) but replaces the
flat panel with a **hybrid staged pipeline: parallel fan-out _within_ a stage,
sequential _between_ stages.** This is the user's refinement of Option A — the
literal 3-role pipeline, but the Research stage is not a single worker; it's a
fan-out of several researchers running concurrently. The sequential cost is only
between the three stages, not across N serial workers.

```text
audrey_research request
  → forced deep (new entry in both gates' forced_deep tuples)
  → pool key "deep_panel_research"
  → STAGED PIPELINE (new executor):
      Stage 1  RESEARCH (parallel fan-out)
        N researchers run concurrently (cloud via gather; local serialized
        through the GPU gate). Each is tool-capable and grounds with
        web_search/kb_search. Starting mix: 2 cloud + 1 local — but the
        worker list is CONFIG-DRIVEN (see schema below), so the mix is tuned
        in config.yaml, not hardcoded. Output: a merged evidence/findings
        block from all successful researchers.
      Stage 2  VERIFY (single pass)
        One verifier consumes the merged findings + the original request,
        flags unsupported / overconfident / anachronistic / disputed /
        too-precise claims. Output: a critique block.
      Stage 3  WRITE (single pass)
        One writer turns verified findings into the final answer. Introduces
        NO new facts beyond the evidence. Output: the user-facing draft.
  → synthesize (light: the synth anchor already privileges grounded material;
    for research mode synth mostly formats the Writer's draft + folds the
    Verifier's caveats — it does NOT re-merge competing drafts)
  → reflect
```

**Why parallel research is the right call:** research is the
embarrassingly-parallel step — independent searches, no cross-dependency — and
cloud workers don't touch the GPU gate, so 2 cloud + 1 local costs barely more
wall-clock than one researcher while buying real source breadth. Verify and
Write are inherently sequential (each needs the prior stage's output), so they
stay single-pass. This is the whole reason the mode is viable on one GPU: the
expensive parallelism is spent where it's free (cloud), the serial cost is paid
only twice (verify, write), not five times.

## Build steps

### 1. Register the virtual model
- `routes/openai/routes.py` — add `"audrey_research"` to `VIRTUAL_MODELS`.
- `graph.py:218` and `routes/openai/pipeline.py:219` — add `"audrey_research"`
  to **both** `forced_deep` tuples. (Complexity-gate ordering in AGENTS.md must
  not be reordered — this is an addition to the existing forced-deep branch, not
  a new branch.)
- `deep_panel.py:52` `_POOL_KEYS` — `"audrey_research": "deep_panel_research"`.
- `state.py:29` + the docstrings in `graph.py:30`, `routes/openai/__init__.py`,
  `main.py:255` — extend the "virtual models" enumerations.
- `pick_panel_timeout` (`deep_panel.py:73`) — decide research's per-worker
  timeout. Research workers search, so they need the larger budget; map
  `deep_panel_research` to `deep_worker` (360s) like the local/mixed pools, or
  add a dedicated `timeouts.research` if 360s proves tight under multi-round
  search.

### 2. Add the pool to `config.yaml` — NEW SCHEMA SHAPE
The existing pools are flat (`workers`/`synthesizer`/`fallback_synth`). The
research pool is **staged**, so it needs a richer per-task shape:

```yaml
deep_panel_research:
  reasoning:                       # (+ code / general / vl)
    researchers: ["deepseek-v4-pro:cloud", "qwen3.5:397b-cloud", "qwen3.6:35b"]
    verifier: "deepseek-v4-pro:cloud"
    writer: "qwen3.6:35b"
    fallback_synth: "glm-5.2:cloud"   # degrade path if write stage fails
```

- `researchers` is the **config-driven fan-out list** (the 2-cloud + 1-local
  starting mix lives here, tunable without code).
- `verifier` / `writer` are single models per stage.
- `_validate_deep_panel_pools` (config.py:136) currently assumes the flat
  shape — **extend it** to recognize `deep_panel_research`'s staged keys and
  validate every researcher/verifier/writer/fallback name against
  `model_registry` (mirrors the existing flat-pool check; +tests).

### 3. Cloud caps + ReAct budget (config)
- **Dedicated research cloud cap (decided):** add
  `agentic.max_research_workers_cloud` (start 2). The research fan-out caps its
  cloud researchers at this value, independent of `max_deep_workers_cloud` (3)
  — research deliberately fans wider than normal deep, so it gets its own
  ceiling kept under the Ollama Pro ~3 limit. `select_workers` (or a research
  variant) honors it.
- **Dedicated research ReAct budget:** add `agentic.react.research_worker`
  (bigger than `deep_worker`: more `max_rounds`, larger `max_tool_result_chars`)
  so researchers can actually read sources instead of rushing from truncated
  snippets — the failure the +10 post-mortem flagged. The Writer/Verifier are
  NOT tool-loops, so they don't use this.
- **Timeout:** map `deep_panel_research` in `pick_panel_timeout` to the larger
  budget (≥`deep_worker` 360s); add `timeouts.research` if multi-round search
  needs more headroom.

### 4. Role prompts (new constants in `prompts.py`)
- `RESEARCHER_SYSTEM` — "Find the factual backbone. Prefer reliable sources.
  Include dates and uncertainty. Do not speculate. Ground claims with the
  tools available; note what you used."
- `VERIFIER_SYSTEM` — "Check the findings for false / overconfident /
  anachronistic / disputed / too-precise claims. Flag each with why. Prefer
  cautious phrasing for ancient biography, disputed authorship, dates,
  rankings, and attribution."
- `WRITER_SYSTEM` — "Turn the verified findings into a clear, engaging answer.
  Introduce NO new facts beyond the evidence provided. Soften or drop anything
  the verifier flagged."
- Each gets a `prompt_from_config` override key in `_PROMPT_KEYS` (so
  `agentic.prompts.researcher`/`verifier`/`writer` are tunable + kill-switchable).
- **Each new constant needs a byte-for-byte regression test** in
  `test_prompts.py` (mirroring `test_synth_system_unchanged`) — hard convention.

### 5. Forced tool use — SCOPED TO THE RESEARCH STAGE ONLY
The critical containment. The reverted Phase 23 forced search on the *whole
panel*; here, forced/strongly-nudged `web_search` applies **only to Stage-1
researchers**, never to the Verifier, Writer, or any other mode. The Verifier
*may* be tool-capable (to spot-check a flagged claim) but is not forced. The
Writer is strictly parametric over the provided evidence — no tools.

### 6. Executor — new staged runner
A new `run_research_pipeline` in `deep_panel.py` (+ a streaming variant for the
banner path), separate from `run_panel`/`run_panel_streaming`:
1. Build + run the researcher fan-out concurrently — reuse `_run_one_worker`
   (it already handles the ReAct loop + GPU gate per worker) and the
   `asyncio.gather` pattern from `run_panel`. Cap cloud researchers at
   `max_research_workers_cloud`.
2. Merge successful researcher drafts into a findings block.
3. Single Verify call (findings → critique).
4. Single Write call (findings + critique → answer).
Each stage degrades gracefully (a dead stage falls through with what it has) —
the pipeline never raises, mirroring `run_panel`'s contract. `state.py` gains
fields for the intermediate stage outputs if reflect/synth need them.

### 7. Synth stage
For research mode the synthesizer is **light** — the Writer already produced the
answer and the anchor already demotes unsupported claims. Options: (a) skip synth
entirely and stream the Writer's draft, or (b) a thin research-synth that formats
the Writer draft + appends the Verifier's surviving caveats as `## Caveats`.
Decide at build time; (b) is safer (keeps the caveat surface) and reuses the
shipped synth path with a research prompt override.

### 8. Banners / streaming — DESIGNED (the risk item, de-risked 2026-06-25)

**Key realization from reading `_stream_deep_with_banners` (pipeline.py:465):**
the deep stream is *already* a sequential chain of phases (Planning →
Dispatching → Synthesizing), each built from the same three-move dance:
`async with PhaseTicker(BANNER, emit)` → `asyncio.create_task(_phase_fn)` →
`_drain_q_until_task` then `_drain_q_now`. Research mode is **the same dance
with five beats instead of three** — not a new streaming paradigm. The earlier
"machinery wasn't built for sequential stages" worry was overstated; it was
built sequential, just never with >3 stages or a fan-out feeding a later stage.

**Reuse verbatim:** the queue/ticker/drain helpers (`PhaseTicker`, `emit`,
`banner_q`, `_drain_q_until_task`, `_drain_q_now`), the frame builders
(`_delta_frame`/`_stop_frame`), the role-delta first frame, and the **entire
Planning phase** — `_phase_thinking` is mode-agnostic, so research gets memory
recall + planner for free.

**Stage mapping:**
- **Research** = `_phase_dispatch` almost unchanged. Stage-1 research IS a panel
  fan-out with per-worker completion banners — exactly what `_phase_dispatch` +
  `run_panel_streaming` already do (`worker_done` → `ticker.append_tail`). Emits
  "researcher done ✓/✗" tails. Almost no new streaming code.
- **Verify** = one new `PhaseTicker(BANNER_VERIFYING)` block, single non-streamed
  call (banner dots → ✓), same dance.
- **Write** = one new `PhaseTicker(BANNER_WRITING)` block that **streams tokens
  live** (user decision) — reuse the synth `first_token` → close-banner → forward-
  deltas logic at pipeline.py:636–664 nearly verbatim, since the Writer's output
  IS the final answer.

**New code (scoped — extensions, not rewrites):**
1. `_stream_research_with_banners` — `_stream_deep_with_banners` + two extra
   PhaseTicker blocks (Verify, Write). Routed via `payload.model ==
   "audrey_research"` in `_stream_via_pipeline` (~pipeline.py:260).
2. `run_research_pipeline` streaming variant in `deep_panel.py` exposing stage
   events: `researcher_done`, `findings_ready`, `verify_done`, `write_delta`,
   `write_done` (modeled on `run_panel_streaming`'s event shapes).
3. New banner constants `BANNER_RESEARCHING` / `BANNER_VERIFYING` /
   `BANNER_WRITING` in `banners.py`.

**Empty-research degrade (user decision):** if Stage 1 yields no usable findings
(all researchers fail/empty), Verify is skipped and Write still runs — but with a
"no grounding was retrieved; answer from general knowledge with explicit low-
confidence framing" instruction. Mirrors deep mode's "always answer something"
contract; the user gets a flagged answer, never a dead end. The Writer prompt
needs a branch (or an injected system note) for this case.

**Non-streaming path:** the compiled graph also needs a research branch (for
non-streaming clients), mirroring how `graph.py` mirrors the streaming deep
path. Keep the two in sync — the same caution `_phase_thinking`'s docstring
already gives applies (add a stage here → add the matching graph node).

### 9. Tests + verification
- Staged-pool validation test (config validates; `deep_panel_research`'s
  researchers/verifier/writer all resolve against the registry).
- Role-prompt regression tests (step 4).
- Routing test: `audrey_research` forces deep in both gates.
- `run_research_pipeline` tests: research fan-out runs concurrently, cloud
  capped at `max_research_workers_cloud`, stage order (research→verify→write)
  holds, and each stage degrades gracefully when its model fails.
- Full `.venv/bin/pytest tests/ -q` + `.venv/bin/ruff check .`.
- Cite check on every touched `src/audrey/` file
  (`scripts/check-lesson-links.py`) — adding to `VIRTUAL_MODELS` /
  `_POOL_KEYS` / prompts will shift lesson cites (Lessons 7, 8, 15 cite these).
- **Box smoke test (user):** select `audrey_research` in OWUI, run the Euclid
  prompt, confirm (a) the research stage grounds the facts, (b) flourishes
  absent/hedged, (c) the stage banners render, (d) wall-clock acceptable.

### 10. OWUI surface
`audrey_research` appears automatically in `/v1/models` once in
`VIRTUAL_MODELS` (the route lists them). No OWUI config needed beyond the user
picking it. Optionally document it in the model-description text at
`main.py:255`.

## Risk register

- **Reintroducing the Phase 23 regression.** Mitigated structurally: forced
  search is scoped to the Stage-1 research fan-out inside an opt-in mode; the
  default panel is untouched. The thing that broke last time (homogenizing the
  *default* panel) cannot happen here — research mode homogenizing its own
  research stage is the intended design, not a regression.
- **ReAct budget too tight for real research.** The +10 post-mortem flagged
  truncated snippets (`max_tool_result_chars=2000`, `max_rounds=2`). Addressed
  by the dedicated `agentic.react.research_worker` budget (step 3) — researchers
  get more rounds + larger result chars than `deep_worker`.
- **Wall-clock.** The two sequential stages (verify, write) add latency the
  parallel panel doesn't have, and a local researcher holds the GPU gate for its
  whole search loop. Mitigations: keep the research fan-out cloud-heavy (cloud
  doesn't touch the gate), and the config-driven `researchers` list lets the
  mix be tuned toward cloud if local proves too slow.
- **Streaming machinery (was flagged as the riskiest part).** DOWNGRADED after
  reading the code (step 8): the deep stream is already a sequential phase chain
  built from one reusable dance; research is that dance with two extra beats.
  Research stage reuses `_phase_dispatch` wholesale; Write reuses the synth
  token-streaming logic. The new code is extensions, not a new generator. Real
  residual risk is narrow: keeping the streaming and non-streaming (graph) paths
  in sync, and the empty-research Writer branch.
- **Cloud 429 under concurrency.** Research fans wider than normal deep; the
  dedicated `max_research_workers_cloud` cap (kept under Ollama Pro's ~3) plus
  the existing per-request cloud accounting keeps a single research request in
  budget. Concurrent research requests still contend — same as today's deep
  modes; revisit only if a real fleet makes it bite.
- **Scope creep into a gate.** Resist adding a "factual prompt auto-detector"
  that routes `audrey_auto` into research mode. That reintroduces the heuristic
  the dedicated mode was designed to avoid. Keep it opt-in.

## What this plan deliberately does NOT do

- Does not change `audrey_auto`/`audrey_deep`/`audrey_cloud`/`audrey_local`
  behavior at all.
- Does not add a factual-prompt classifier or auto-routing.
- Does not force tool use outside the Stage-1 research fan-out.
- Does not touch the synth anchor already shipped (reused as-is for the light
  research-synth stage).
