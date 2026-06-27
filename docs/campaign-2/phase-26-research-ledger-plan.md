# Phase 26 — Research claim/source ledger (staged plan)

One deploy doc, built and shipped in **stages**, each independently deployable
and **eval-gated against the baseline** (`docs/testing/2026-06-26-factcheck-*`
via `scripts/eval_research.py`). The end state: `audrey_research` proves to
itself where each load-bearing claim came from — a structured ledger of claims
and sources — before the writer is allowed to make it read well, with
deterministic, *selective* hedging.

## The headline risk (read first)

This repo already built a structured citation/source mandate (+21) and **fully
reverted it (+26)** because it degraded answers: models did source bookkeeping
over reasoning, padded weak sources to satisfy the rule, and dropped sound
parametric facts for lacking a URL. This design is *more* structured than what
was reverted. So the controlling principle of this plan is:

> **The ledger is internal scaffolding the models reason over — it does not
> become user-facing bookkeeping.** The answer the user reads stays clean prose
> plus a short "Sources used" list at the end. Structure serves verification
> and hedging; it must not leak into the prose as citation pressure.

Every stage is a **measured variable**: build it, deploy, run the protocol, diff
against the prior run, keep only if quality holds or improves. If a stage
regresses like +21 did, it backs out at its own boundary without losing the
others. This is the discipline the whole +21→+26 saga was missing.

---

## Current shape (what we're extending)

`audrey_research` today (`deep_panel.py`):

```
research fan-out (N workers, ReAct + tools) → _format_findings (prose blob)
  → Verify (prose critique) → Fact-check (prose corrections) → Write (prose)
```

Everything between stages is **free prose** in a `findings` string. Sources
that `web_search` returns dissolve into that prose; nothing structures them,
carries them forward, or displays them. The role prompts live in `prompts.py`
(`RESEARCHER_SYSTEM` / `VERIFIER_SYSTEM` / `FACTCHECK_SYSTEM` / `WRITER_SYSTEM`),
each overridable via `agentic.prompts.{role}`. Stage user-blocks are
`_verify_user_block` / `_factcheck_user_block` / `_write_user_block`.

**Key constraint found in source:** `OllamaClient.chat` does NOT forward
Ollama's `format` (JSON-schema) field today — only `model/messages/stream/
options/tools`. Structured output needs a small client addition (Stage 0). And
a worker can't run a ReAct *tool loop* and be pinned to a final JSON schema in
the same call — structure comes **after** the tool loop (a parse/format step),
not instead of it.

---

## Stage 0 — Plumbing: structured output + the schema module (no behavior change)

Foundation only; ships dark (nothing calls it yet), so it can't regress.

- **`OllamaClient.chat`** gains an optional `format: dict | None` param that
  forwards to the Ollama payload as `"format"` (JSON schema). Streaming variant
  left alone (ledger calls are non-streaming). +1 unit test (payload includes
  `format` when passed, omitted otherwise).
- **New `pipeline/ledger.py`** — the Pydantic models, verbatim to the user's
  design, as the internal contract:
  - `Source(id, title, url: HttpUrl, source_type: Literal[...], supports: list[str])`
  - `Claim(id, text, source_ids, risk: low|medium|high, needs_hedge, hedge_reason)`
  - `ResearchResult(summary_notes, claims, sources, unresolved_questions)`
  - `ClaimCheck(claim_id, verdict, corrected_text, notes)`
  - `FactCheckResult(checks, fatal_errors)`
  - `source_type` includes `company_claim` (so vendor benchmarks are typed, not
    treated as independent fact — central to Stage 4 hedging).
- **Robust parsing helper** — `parse_research_result(raw: str) -> ResearchResult
  | None` that tolerates a model wrapping JSON in prose/code fences, and returns
  `None` (never raises) on malformed output. Same for the fact-check result.
  Hermetic tests over good/fenced/garbage inputs.

Deploy: ship with the next rebuild; **no eval needed** (dark code). Unit tests
+ ruff are the gate.

---

## Stage 1 — Researchers emit a ledger (the structured-output switch)

Make Stage-1 workers return `ResearchResult` JSON instead of prose notes. This
is the riskiest *quality* stage (it changes how workers think), so it ships
**alone** and is eval-gated hard.

- **Two-step per worker** (because tools + schema can't share one call): the
  worker runs its existing ReAct tool loop as today, then a **final
  structuring call** (`format=ResearchResult.schema`, no tools) converts the
  gathered material into the ledger. The structuring call is cheap and
  deterministic-ish; if it fails to parse, **fall back to the worker's prose
  draft unchanged** (Stage 1 must degrade to today's behavior, never break).
- **`RESEARCHER_SYSTEM`** rewritten to the user's researcher prompt: return only
  structured JSON; attach `source_ids` to each important claim; mark `risk=high`
  for dates/rankings/"first"/"only"/"invented"/"proved"/authorship/release-specs/
  benchmarks/laws/prices/current-status; type vendor benchmark claims as
  `company_claim`. Keep the "don't pad, don't invent a URL you didn't use" guard
  (the anti-+21 language).
- **`_format_findings`** gains a structured path: when workers return ledgers,
  merge them into a combined `ResearchResult` (dedup sources by URL, concat
  claims with worker-prefixed ids) AND render a human-readable findings blob for
  the downstream prose stages that don't yet consume structure. Both the
  structured object and the rendered blob flow forward.

**Eval gate:** run the protocol; diff every case. The watch items: do the
biographies stay as good (the +21 failure was here — workers padding/dropping)?
Does `current-2025-recent` still hedge-don't-drop? If quality holds, proceed; if
it regresses, this stage backs out (prose researchers) and Stages 2–4 still have
value on prose — but the ledger benefit is lost, so this is the make-or-break
stage. **Budget note:** the extra structuring call adds latency/cost per worker;
record it.

---

## Stage 2 — Fact-check operates on claims, not prose

With a ledger flowing, the fact-checker reviews **claims against their sources**
instead of re-reading prose.

- **`FACTCHECK_SYSTEM`** rewritten to the user's fact-checker prompt, producing
  `FactCheckResult` (`format=` schema): per-claim `verdict`
  (supported/unsupported/conflicting/needs_hedge/irrelevant), optional
  `corrected_text`, and `fatal_errors` for claim-vs-claim contradiction. Rules
  baked in: official dates/names/licenses not hedged unless sources conflict;
  company benchmarks phrased as company claims; ancient bios hedged unless
  attested; disputed authorship → "attributed to"; strong-support words
  ("first"/"only"/"proved"/"invented"/"founded"/"definitively"/"worldwide"/
  "complete"/"all") require strong support or get `corrected_text`.
- **`_factcheck_user_block`** feeds the claim ledger (not prose) when present;
  falls back to the prose path when Stage 1 degraded.
- The fact-checker keeps its `web_search` capability (tool loop → structuring
  call, same two-step as Stage 1) so it can still *confirm* high-risk claims,
  but now writes verdicts onto specific claim ids.
- **This is the stage that catches the "Surviving: Conics…" class** (Conics is
  lost) — a claim whose source doesn't actually support it gets
  `verdict=unsupported`, and the writer (Stage 3) omits it.

**Eval gate:** protocol diff. Watch that `fatal_errors`/`unsupported` verdicts
don't over-trigger and gut good answers (the over-flagging that sank +21–+25).
Still **fail-soft**: any parse/tool failure → empty checks → Stage-1 findings
flow to the writer unchanged.

---

## Stage 3 — Source-bound writer + end-of-answer "Sources used"

The writer consumes the ledger + checks and is **source-bound**, with the
clean-output rule the user originally asked for (list at the end, no inline
citation bookkeeping).

- **`WRITER_SYSTEM`** rewritten to the user's writer prompt: ledger is the
  factual backbone; fact-check corrections are mandatory; introduce no new
  facts/dates/names/titles/rankings/authorship/specs not in **supported**
  claims; `needs_hedge` → preserve the hedge; `unsupported`/`conflicting` →
  omit. End with a short **"Sources used"** list (the supported claims' sources)
  — **no inline `[n]` markers** (the user's call: "list its sources at the end
  but not incorporate citations").
- **Empty-grounding path:** when no grounding was retrieved, **omit the Sources
  section entirely** (user decision earlier today) — keep the existing
  low-confidence caveat; never print an empty or fabricated list.
- **`_write_user_block`** passes the structured ledger + checks; the "Sources
  used" content is derived from the ledger's sources, so the writer can't invent
  a URL — it can only list what the researchers carried.

**Eval gate:** protocol diff. New structural check is cheap to add to the
harness: for grounded research cases, assert a "Sources used" section renders
with ≥1 well-formed URL (reuse the existing `_sources_block`/`_extract_urls`
helpers — they already exist from the reverted work). Ungrounded controls
assert NO sources section. This is the one stage that changes user-visible
output, so eyeball it closely.

---

## Stage 4 — Deterministic selective hedging (`hedge_policy`)

The "hedge the right things, not everything" piece — a **pure function**, not a
prompt, so it's testable and consistent.

- **`pipeline/ledger.py`** (or a sibling) gains `hedge_policy(claim, source_types)
  -> Literal["state_plainly","attribute_to_company","hedge",
  "hedge_or_cite_strongly"]`, verbatim to the user's logic: official + not-high
  → state plainly; company_claim → attribute to company; `needs_hedge` → hedge;
  high risk → hedge-or-cite-strongly; else plainly.
- **Where it runs:** between fact-check and write. For each supported claim, the
  policy computes a hedging *disposition* from its source types + risk, and that
  disposition is rendered into the writer's per-claim guidance (e.g. the
  `_write_user_block` annotates each claim "STATE PLAINLY" / "ATTRIBUTE TO
  COMPANY: <name>" / "HEDGE"). The writer applies the disposition rather than
  guessing — this is what makes "DeepSeek released R1 on January 20, 2025"
  (official, low-risk → plain) vs. "Meta claimed Maverick beat GPT-4o" (company
  → attributed) deterministic instead of vibes.
- **Pure unit tests** over the policy table (the user's three examples become
  test cases) — no model needed; this is the most testable part of the whole
  design.

**Eval gate:** protocol diff. The payoff case is `current-2025-recent` (official
dates stated plainly, vendor benchmarks attributed) and the bios (ancient
anecdotes hedged). Watch for under-hedging now that plain-statement is
explicitly allowed.

---

## What we are deliberately NOT doing

- **No hard gate that fails ungrounded answers** (user decision earlier today).
  The user's original §"final gate that fails if no source trace" recreates the
  reverted +21 pressure; we get the benefit via the source-bound writer + the
  "Sources used" list instead. `fatal_errors` exists in the ledger but drives
  *claim omission*, not a whole-answer failure.
- **No inline `[n]` citations** — sources list at the end only.
- **No structured output on the streaming write path** — the writer still
  streams prose to the user; structure is upstream only.
- **No change to `audrey_deep`/fast** — this is `audrey_research`-only. (The new
  deep/fast eval protocols built this session still apply unchanged.)

---

## Build / deploy order

| Stage | Ships | User-visible | Eval-gated | Backs out independently |
|-------|-------|--------------|------------|------------------------|
| 0 plumbing + schema | dark | no | unit only | n/a |
| 1 researcher ledger | rebuild | no | **yes (make-or-break)** | → prose researchers |
| 2 claim fact-check | rebuild | no | yes | → prose corrections |
| 3 source-bound writer + Sources list | rebuild | **yes** | yes | → unbound writer |
| 4 hedge_policy | rebuild | yes (wording) | yes | → prompt-only hedging |

Each stage: build → `docker compose up -d --build audrey-ai` → run
`scripts/eval_research.py --cases scripts/eval_prompts_protocol.json
--save-file docs/testing/<date>-ledger-stageN-answers.md` → write the paired
report → diff vs. the prior baseline → keep or back out. All stages are
prompt+config-tunable live via `agentic.prompts.{role}` once shipped, so a
regression can be softened without a rebuild while deciding.

## Fail-soft contract (every stage)

Any parse failure, tool failure, or schema-validation error at any stage →
that stage degrades to the current prose behavior for that request, never raises,
never breaks the answer. This mirrors the existing fact-check stage's broad-except
design (`deep_panel.py`) and is what lets us ship structure without making the
mode brittle.

## Open decisions for you (before Stage 1 build)

1. **Stage 1 two-step cost** — the extra structuring call per worker adds
   latency/quota. OK to absorb for the "thorough" mode, or cap the structuring
   call to a cheaper model?
2. **Merge strategy for multi-worker ledgers** — dedup sources by URL is
   obvious; for *conflicting* claims across workers, surface both as
   `conflicting` for the fact-checker, or let the merge pick? (Recommend:
   surface both — that's what the fact-checker is for.)
3. **Outline-before-prose** for each stage's prompt rewrite, or trust the
   user-supplied prompts as the spec and implement directly? (The prompts above
   are yours; I'd treat them as the spec and implement, eval-gating each.)
