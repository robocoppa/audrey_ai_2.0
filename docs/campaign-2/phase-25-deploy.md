# Campaign 2 Phase 25 — `audrey_research` fact-check stage

Adds a dedicated **fact-checking stage** to the `audrey_research` staged
pipeline. It web_search-confirms the high-risk/dated/current claims in the
researchers' findings and feeds corrections to the writer, so factual errors are
fixed *before* the answer reaches the user — not flagged after.

New pipeline shape:

```text
research (fan-out) → verify → FACT-CHECK (web lookups) → write
```

> Follow-on to Phase 24 (`audrey_research`). Motivated by the 2026-06-26 eval
> (`docs/testing/2026-06-26-accuracy-stress-*`): the mode produced good answers
> but shipped subtle dated errors fluently (e.g. DeepSeek-R1 "Jan 26" vs. the
> official Jan 20; "async-std deprecated in 2024" vs. the 2025 visibility push).
> The just-stabilized verifier was **left untouched** — this is an additive
> stage, not a verifier change.

## What it does

A single tool-capable model (the `factchecker`) runs a bounded ReAct loop
(`pipeline/react.py`) between verify and write. It takes the request + merged
findings + verifier critique, uses `web_search` to confirm a handful of the most
load-bearing checkable claims (dates, versions, licenses, names, and
"deprecated/first/only/proved/authored" assertions), and emits a short
corrections list:

```text
- CONFIRMED: <claim> (source)
- CORRECT: findings say <X>, but <source> shows <Y> — use <Y> (url)
- UNVERIFIED: <claim> — no reliable source; the writer should hedge
```

The writer's prompt now says a `FACT-CHECK CORRECTIONS` block overrides the
findings on any claim it touches. The block is only added to the write prompt
when it carries actionable `CORRECT:`/`UNVERIFIED:` lines.

**Fail-soft and optional, like every stage.** It runs only when a `factchecker`
is configured, healthy, tool-capable, tools exist, *and* the request was
grounded. On no-factchecker / no-tools / any error it returns empty corrections
and the pipeline runs the old verify→write flow unchanged. Streaming emits a new
`Fact-checking` banner (Researching → Verifying → **Fact-checking** → Writing);
when the stage is skipped, no banner is shown.

## Files changed

- `config.yaml` — `deep_panel_research.*.factchecker` (set to
  `deepseek-v4-pro:cloud`, the proven searcher) on all four task types;
  `agentic.react.factcheck_worker` budget (max_rounds 3, max_tool_result_chars
  6000 — tighter rounds than researchers, same large read budget).
- `src/audrey/config.py` — `_validate_deep_panel_pools` validates the optional
  `factchecker` slot when present.
- `src/audrey/pipeline/prompts.py` — `FACTCHECK_SYSTEM`; `factchecker` override
  key; one writer-prompt line for applying corrections.
- `src/audrey/pipeline/deep_panel.py` — the stage in
  `run_research_pipeline_streaming` (the load-bearing change); `_factcheck_user_block`;
  `_write_user_block` gains an optional `corrections` arg; `_factcheck_react_budget`;
  `factcheck_done` event + `corrections` on the `done` event. Broad
  exception-catch so the optional stage can never break the answer.
- `src/audrey/pipeline/banners.py` — `BANNER_FACTCHECKING`.
- `src/audrey/routes/openai/pipeline.py` — Fact-checking banner block in
  `_stream_research_with_banners`.
- `src/audrey/pipeline/graph.py` + `state.py` — surface `research_factcheck`.

## Verification (laptop, hermetic)

- **534 pytests pass** (+6: 1 factcheck prompt regression, 3 executor tests
  — stage runs + threads corrections, skipped when unconfigured, stage order —,
  1 streaming Fact-checking-banner-order test, 1 config validation test).
- `ruff` clean on all touched files (only the pre-existing accepted `kb/`
  ASYNC240 hints remain).
- Real `config.yaml` boots: graph compiles, staged pool validates with the new
  `factchecker`.
- AI-course cite check **0 confident DRIFT / 0 broken** (re-anchored 3 cites the
  config/prompts growth shifted: config.yaml 350→368, prompts.py 187→213,
  pipeline.py 97→106).

## Deploy

Config + code only; no new dependency, no schema migration. From
`/mnt/user/appdata/audrey_ai_2.0`:

```bash
docker compose up -d --build audrey-ai
docker compose logs -f audrey-ai
```

`custom-tools` is unchanged (the stage reuses the existing `web_search` tool).

## Box smoke test (the real verification)

This is judged by the testing protocol, not by eye. VPN'd into the LAN:

```bash
.venv/bin/python scripts/eval_research.py \
    --cases scripts/eval_prompts_protocol.json \
    --save-file docs/testing/$(date +%F)-factcheck-stage-answers.md
```

Then **diff against the 2026-06-26 baseline** and check the specific claims the
eval caught:

- `current-2025-recent`: does the DeepSeek-R1 date now read **Jan 20** (or hedge),
  not Jan 26? Qwen3 **Apr 29**, not Apr 28?
- `current-rust-async`: is the async-std "deprecated in 2024" framing corrected
  toward the 2025 discontinuation/visibility?
- Watch the **Fact-checking banner** renders live in OWUI between Verifying and
  Writing.
- Controls (`ctrl-birthday-toast`, `ctrl-explain-recursion`): the stage should
  no-op (no checkable claims) without bloating latency — confirm no Fact-checking
  banner / no spurious corrections on those.

Keep the stage if the dated errors soften and quality holds; if it over-corrects
(a web lookup misleads the writer) or adds unacceptable latency, tune the
`factcheck_worker` budget or disable per task by removing the `factchecker` key —
or override live via `agentic.prompts.factchecker`. No rebuild needed for the
prompt override.

## Risks

- **Latency/cost** — one more web-using stage on an already-slow opt-in mode.
  The factchecker budget is deliberately tighter than the researchers'.
- **The fact-checker is fallible** — a lookup can mislead and "correct" a right
  claim. Mitigated: it only *advises* (the writer owns the prose), and the prompt
  prefers official/primary sources and `UNVERIFIED` over forcing a change.
- **Verifier untouched** — the stabilized `+30` verifier is deliberately
  separate. If fact-checking proves redundant with it, reconsider; start additive.
