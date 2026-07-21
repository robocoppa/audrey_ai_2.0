# Shared evidence pool — deep panel

Status: **deferred behind a cheaper fix.** The gating eval ran
(`2026-07-21-135212-deep-engines`) and resolved to a branch this plan did not
anticipate — see "Gate resolution" at the bottom before building anything here.

## Why

`deep-2025-recent` (2026-07-21 eval): qwen and deepseek dispatched overlapping
searches and held comparable evidence. qwen wrote a grounded draft. deepseek
wrote a refusal. The evidence deepseek held was not bad — it was unused.

The argument for pooling is **resilience, not cost**. Today a worker's evidence
is only as useful as that worker's willingness to write from it. One model's
refusal silently deletes everything it retrieved. A pool decouples *retrieval*
from *willingness*: the panel keeps every fact any worker found, regardless of
which worker found it or what it decided to do with it.

Deduplication and quota savings are real but secondary — worth noting because
they change the sizing math, not because they justify the work.

## The blocking architectural fact

Evidence is discarded at the worker boundary.

`ReactResult.tool_calls` carries `ToolResult` objects with the full `.content`
(the search-result JSON). [deep_panel.py:232-235](../../src/audrey/pipeline/deep_panel.py#L232-L235)
projects them down to `{name, elapsed_s, is_error}` when building the
`WorkerDraft`. `WorkerDraft` ([state.py:16-25](../../src/audrey/pipeline/state.py#L16-L25))
has no field that could hold a body.

So there is no pool to share, at any level, until capture exists. Capture is
cheap — the data is already in hand at the projection site — but it is a
prerequisite for every option below.

## Three things "shared pool" could mean

### A. Pool → synthesizer (recommended first)

Collect every worker's evidence, dedupe, and hand the union to the synthesizer
as an `EVIDENCE` block alongside `DRAFTS`.

- Deterministic. No new concurrency.
- Rescues the case **all** workers refuse — synth still has raw sources and can
  answer. That is the strongest resilience case, and A is the only option that
  covers it cheaply.
- Does not literally satisfy "every worker sees the union" — the synthesizer
  does, the workers don't.
- Cost: one prompt gets bigger. No extra model calls, no extra latency beyond
  prefill.

### B. Live shared cache during the panel run

A per-request store workers read before dispatch and write after.

**Recommend against.** Workers run concurrently, so whether worker B sees worker
A's result depends on who finished first — the panel becomes non-deterministic
run to run and the eval loses its baseline. It also solves the wrong problem: a
cache prevents *duplicate work*, it does not deliver the *union*, so it buys
nothing for refusal resilience. The `tools-server` SearXNG cache
([`searxng.py`](../../tools-server/searxng.py), keyed on query alone since
`267eabd`) already absorbs the exact-duplicate case at the layer where it
belongs.

### C. Two-phase panel: gather → pool → draft

Phase 1 workers search only. Pool the union. Phase 2 every worker redrafts from
the same evidence with tools off.

This is the literal ask, and it does more than A: a refusing worker gets a second
prompt in a shape ("here is the evidence, write the answer") that the research
pipeline's Write stage already demonstrates the same models comply with.

Costs: roughly doubles chat calls; the local worker takes the GPU gate twice;
`pick_panel_timeout`'s budget assumptions change. And it converges deep mode on
research mode's staged shape — at which point the honest question is whether deep
should borrow research's Write stage rather than grow its own.

**Recommendation: A now, C only if A's evidence block shows synth actually
rescuing answers.** A is a strict prerequisite for C (same capture, same pool,
same dedupe), so nothing is thrown away.

---

## Phase 1 — capture

**`state.py`** — add to `WorkerDraft`:

```python
evidence: list[dict]   # pooled-evidence items harvested from this worker's tool results
```

**`deep_panel.py`** — a new `_harvest_evidence(results: list[ToolResult]) -> list[dict]`,
called where the `tool_calls` projection already happens
([deep_panel.py:232](../../src/audrey/pipeline/deep_panel.py#L232)).

Per successful `ToolResult`:
- Parse `.content` as JSON. Unparseable → skip (bodies are truncated at
  `max_tool_result_chars`, so a tail-cut result is *normal*, not exceptional).
- `web_search` shape (`{"query": …, "results": [{"title","url","snippet"}]}`) →
  one item per result: `{"title", "url", "snippet", "query", "tool"}`.
- `kb_search` shape → same item shape with `url` absent; keyed on doc id.
- Anything else → skip.

Fail-soft is mandatory: wrap the whole harvest in `try/except Exception` and
return `[]`. This runs inside `_run_one_worker`, which is documented to never
raise. A malformed body must not cost the panel a draft.

Cap at capture: `_MAX_EVIDENCE_PER_WORKER` items, snippets truncated to
`_EVIDENCE_SNIPPET_CHARS`. Capping here rather than at pool time keeps
`PipelineState` bounded even when Phase 2 is off.

**`banners.py`** — verify `panel_drafts_block` /
`_draft_section_lines` ([banners.py:217](../../src/audrey/pipeline/banners.py#L217))
reads only the fields it names and will not start dumping `evidence` into the
debug block. It reads named keys today; add a test pinning that, because
`debug_panel_drafts: true` is currently on and the eval artifact would balloon.

Phase 1 is inert — nothing consumes `evidence` yet. Ship and eval it separately
if convenient; it cannot change an answer.

## Phase 2 — pool and dedupe

**`deep_panel.py`** — `_pool_evidence(drafts: list[WorkerDraft]) -> list[dict]`:

1. Concatenate every draft's `evidence`.
2. Dedupe by normalized URL — `url.strip().rstrip("/").lower()`, the same key
   `_merge_ledgers` uses ([deep_panel.py:731](../../src/audrey/pipeline/deep_panel.py#L731))
   so the two dedupe paths can't disagree. URL-less items (kb) dedupe on
   `(tool, title)`.
3. On a hit, keep the longer snippet and record that a second worker
   independently surfaced it — corroboration is signal the synthesizer can use.
4. Rank: multi-worker items first, then original order. Cap at
   `_MAX_POOLED_EVIDENCE`.

The earlier duplication measurement predicts this collapses hard. That run showed
45 searches with **zero** exact-duplicate queries but severe *semantic*
duplication (three clusters consumed 30 of 45) — semantically duplicate queries
return overlapping URLs, so URL dedupe catches what query dedupe missed.

## Phase 3 — deliver to synth

**`synthesize.py`** — `_format_drafts_for_synth`
([synthesize.py:42](../../src/audrey/pipeline/synthesize.py#L42)) gains an
`evidence` parameter and emits, after `DRAFTS`:

```
SHARED EVIDENCE (retrieved by the panel; may include sources no draft used):
- [title](url) — snippet   (found by 2 workers)
```

**`prompts.py`** — `SYNTH_SYSTEM` needs one clause, and its wording is
load-bearing in the same way the compaction stub's was: the block must read as
*material available to you*, not as *claims to repeat*. Something to the effect
of: evidence here was retrieved by the panel; a draft declining to use a source
is not a judgment against it; do not cite an evidence item the drafts don't
support beyond what it plainly states.

**Threading.** `_pool_evidence` runs in `synthesize`/`synthesize_stream`, which
already receive `drafts`. No signature change on `run_panel*`, no new event
shape, both deep call sites
([graph.py:357](../../src/audrey/pipeline/graph.py#L357),
[pipeline.py:663](../../src/audrey/routes/openai/pipeline.py#L663)) unchanged.

## Sizing — the main risk

Ceiling today: 3 workers × `max_web_searches: 4` × `max_tool_result_chars: 4000`
= **48,000 chars** of raw bodies. The `synth_draft_sizes` log line
([synthesize.py:75](../../src/audrey/pipeline/synthesize.py#L75)) is the ground
truth for what drafts actually run; an unbudgeted pool would be several times the
drafts it's supposed to support, and drown them.

Distillation is what makes this viable — dropping to title/url/snippet, URL
dedupe, and per-snippet truncation should land the block in low single-digit
thousands of chars. Budget it explicitly with `_MAX_POOLED_EVIDENCE` and
`_EVIDENCE_SNIPPET_CHARS` and pin both with a test asserting a worst-case pool
renders under a stated ceiling.

Note this promotes a deferred AUDIT item: `_format_drafts_for_synth` has **no
per-draft cap today**. Adding a second unbounded block to the same prompt makes
that gap load-bearing. Cap the evidence block at minimum; consider capping drafts
in the same pass.

## Invariants to hold

- **User scoping.** `kb_search`/`memory_search` are user-scoped
  ([dispatch.py:56](../../src/audrey/tools/dispatch.py#L56)). Pooling happens
  strictly within one request, so all evidence shares one `user_id` and no
  cross-user path opens. State it in the docstring so a future "pool across
  requests" idea has to confront it deliberately.
- **Never raise.** Harvest and pool sit inside code paths documented to degrade,
  never raise. Bare `except Exception` + log, matching the ledger-structuring
  posture ([deep_panel.py:1407](../../src/audrey/pipeline/deep_panel.py#L1407)).
- **Research mode untouched.** It has its own evidence path (the claim/source
  ledger). This work is deep-panel only.
- **Errors excluded.** Harvest only `is_error=False` results, mirroring
  `web_search_chars` accounting in [react.py:307](../../src/audrey/pipeline/react.py#L307).

## Flag and deploy

`agentic.shared_evidence.enabled`, default **false**, read via the same shape as
`_ledger_enabled` ([deep_panel.py:592](../../src/audrey/pipeline/deep_panel.py#L592)).

`config.yaml` is bind-mounted, so A/B is a flag flip plus
`up -d --force-recreate audrey-ai` — no rebuild between arms. That matters: the
baseline and treatment then run against a byte-identical image.

## Measurement gate

Decide from the compaction eval currently running, before writing Phase 1:

- **Refusals gone, drafts grounded** → Phase 1–3 is resilience insurance for a
  problem not currently firing. Still worth it, but as a low-priority build, and
  the eval to justify it needs a case where a worker refuses.
- **Refusals persist while `web_search_chars` is large** → the worker held
  evidence and declined. Build Phase 1–3, and C becomes a live candidate.
- **`web_search_chars` near zero on refusing workers** → retrieval is still
  broken upstream. Pooling nothing is still nothing. Fix retrieval first.

Success metric for the A/B, once built: on the same protocol, count answers
containing refusal/hedge language while the pool held on-topic sources. The
`debug_panel_drafts` block already puts every draft in the eval artifact, so a
refusing worker whose evidence reached the answer anyway is visible without new
tooling.

## Not doing

- Option B (live cross-worker cache) — non-deterministic, wrong problem.
- Embedding-based semantic dedupe — URL dedupe is the cheap 80%; revisit only if
  the pool measurably overflows its cap with near-duplicate URLs.
- Pooling across requests — a different (and user-scoping-sensitive) feature.

---

## Gate resolution — `2026-07-21-135212-deep-engines`

18/18 PASS. Three findings, in order of what they change.

### 1. The compaction fix landed

**Zero** compaction narration anywhere in the run — no "history compacted," no
"compacted out before I could read them," no "searches returned empty results."

The ctx numbers are NOT the evidence for this. `web_search_chars` accumulates at
dispatch ([react.py:307](../../src/audrey/pipeline/react.py#L307)), before
compaction runs, so a large ctx never proved the bytes reached the final prompt.
The evidence is that the *shape of the complaint changed*: workers went from
claiming they didn't have results (contradicted by their own ctx) to judging the
results they read as thin. That shift is the fix's signature.

### 2. Refusals persist — and they are one model, not a panel property

| Worker | Grounded runs | Declined |
|---|---|---|
| kimi-k2.6:cloud | 6 | **3** |
| deepseek-v4-pro:cloud | 5 | 0 |
| qwen3.6:35b | 4 | 0 |

Two near-controlled comparisons within a single case:

- **deep-pythagoras** — kimi held 6,668 chars and wrote "I cannot fabricate it."
  qwen held **6,719** and wrote a complete draft.
- **deep-library-alexandria** — kimi held 7,431 and refused. deepseek held
  **7,545** and produced the draft the final answer is built from.

Same case, same evidence volume, opposite outcome. Not retrieval, not
compaction — disposition.

### 3. Redundancy already absorbed both refusals

Both refusal cases shipped strong final answers, because a sibling worker was
grounded and synth used it. **Option A's rescue case is all workers refusing,
and three runs have now failed to produce it.** Its measured value in this run
is zero.

### Revised next step: a deep-worker role prompt, first

Deep-panel workers receive **no role system prompt at all**. Research mode
injects `RESEARCHER_SYSTEM`; `_prepare_panel` forwards the raw conversation. So
there is nowhere in the deep path to say "write from what you gathered."

The hook already exists:
[`compose_system_messages(task_role=...)`](../../src/audrey/pipeline/prompts.py#L440-L442)
is a slot built for exactly a deep-worker prompt and is always `None` today.

Do that first — same move that worked on `SYNTH_SYSTEM`, a fraction of this
plan's size, and it targets the only behaviour the data actually shows. Re-run
the protocol. If kimi's decline rate falls to deepseek's, the pool's remaining
justification is the correlated all-refuse case alone, and this plan stays
parked.

### Carried forward, unrelated to the pool

- **Latency is the compaction fix's bill, and it lands on the local worker.**
  code-lru-cache 263.5s total with qwen3-coder-next at 214.7s — for an
  `OrderedDict` LRU cache. deep-pythagoras 175.0s (qwen 106.1s), deep-rust-async
  154.4s (qwen 114.9s). The local worker now carries up to
  `compress_keep_last` × `max_tool_result_chars` = 32k chars into every round.
- **New one-off:** qwen dropped on deep-contested-recommendation with
  `500: expected element type <function> but have <parameter>` — Ollama
  rejecting a malformed tool call the model emitted. Single occurrence.
- **Retrieval ambiguity, separate from refusal:** deepseek on deep-2025-recent
  reported that many results were about golf's "The Open" tournament. Real
  query-ambiguity noise, and it still answered.
