# Campaign 2 Phase 28 — research grounding diagnostic: the `read_url` failure

Why do individual `audrey_research` workers still come back with thin grounding
even with SearXNG healthy and the empty-fallback deployed? The
`2026-07-15-113045-research-trace-diag` run (5 cases: attention, plate-tectonics,
mrna, race-order, gk-element-w) answers it — and the answer is **none of the three
starting points we'd written down.** It's a fourth cause the run's own
`_Tools used:_` footers exposed directly.

**Status:** root cause identified + evidenced. Two fixes: (1) the in-repo prompt
mitigation is **shipped** (needs deploy); (2) the primary fix lives in the
**separate `custom-tools` repo** and is **not started** (spec below). Last updated
2026-07-15.

## The headline (read first)

**The thinness is a failing `read_url` / `web_fetch` tool, not a retrieval,
budget, or merge problem.** `web_search` succeeds and returns URLs in its snippets;
the follow-up page-open tool (`read_url` / `web_fetch`) then fails on essentially
every call; and some researcher models misread that fetch failure as "search
returned nothing" and write `SOURCES: none`, discarding the snippet URLs they
already had. The ledger then has no usable URL to render → the case reads THIN.

This corrects two claims from the earlier inline analysis (both were wrong):

- **`web_fetch`/`read_url` is NOT the cloud model's opaque native tool.** It is a
  real registered tool served by `custom-tools:8001`, dispatched through Audrey's
  own `tools/dispatch.py`. That's *why* it shows up in the `_Tools used:_` footer
  (which is built only from real pipeline dispatches). It is failing at the
  dispatch layer (`is_error=True`), which Audrey **can** see and act on.
- **The writer does NOT merge to a "thin common source set."** `_merge_ledgers`
  concatenates claims and unions sources (deduped by URL). A rich worker's sources
  are never intersected away. Thinness is upstream: the researcher emitted no
  usable URL in the first place.

## The evidence (from the run's own footers + ledgers)

The `_Tools used:_` footer is assembled by `banners._format_calls` from each
worker's real `tool_calls` log (`{name, elapsed_s, is_error}` per dispatch). So a
tool that appears there with ✅/❌ counts went through `dispatch_one` — it is one of
Audrey's registered tools, not model-internal.

| Case | web_search | read_url / web_fetch | Ledger URLs | Final |
|---|---|---|---|---|
| **attention** | ✅6, ✅6 (both workers) | `read_url ✅0 ❌2` | 1 source, **no url** | **THIN (sources:0)** |
| **plate-tectonics** | ✅6 ×3 | (none logged) | 8 sources w/ real URLs | sources:2 GOOD |
| **mrna** | ✅6 ×2 | `web_fetch ✅0 ❌1` | 4 sources, 2 w/ URL | sources:2 GOOD |
| **gk-element-w** | ✅ small ×3 | (none) | 3 sources w/ URLs | sources:2 GOOD |
| **race-order** | (pure reasoning) | — | 0 (no web needed) | correct, no sources |

Two facts kill the obvious hypotheses:

1. **The fetch failure is NOT what makes a case thin.** mrna's `web_fetch` also
   failed (`✅0 ❌1`) yet mrna grounded fine (2 GOOD sources). The difference is
   purely whether a worker *cited the search-snippet URL anyway*. mrna's qwen
   worker cited the JHU + Nebraska URLs straight from the `web_search` snippets;
   attention's workers did not.
2. **`web_search` succeeded everywhere it was thin.** attention's glm worker ran
   `web_search ✅6` — six successful searches — then wrote *"my web and
   knowledge-base searches returned no usable results ... from my own prior
   knowledge, not freshly-retrieved sources."* It conflated "I couldn't `read_url`
   the page" with "search found nothing," and threw away six searches' worth of
   snippet URLs. That is the behavioral bug.

The `web_search` result shape (from the sibling in-repo `tools-server/app.py`,
`WebSearchResult`) is `{title, url, snippet}` — **the URL is in every result
without any `read_url` needed.** So a worker always has a citable URL after a
successful search; the fetch is only for reading the *full page*.

## The confirmed causal chain

1. `custom-tools:8001` (a **separate repo** — see `docs/reference/box-operations.md`
   §3 and memory `[[audrey-box-container-map]]`) serves `web_search` (returns
   `{title,url,snippet}`) **plus** a `read_url` / `web_fetch` page-opener that the
   in-repo `tools-server/` does not define. That's why `read_url` appears nowhere
   in this checkout yet fires in the footer.
2. Worker runs `web_search` → **succeeds**, URLs land in context via the snippets.
3. Worker calls `read_url` / `web_fetch` to open a page → **fails** (timeout /
   network / http-4xx-5xx). `dispatch_one` returns `{"error": ...}` with
   `is_error=True` into the model's context (footer: `✅0 ❌n`).
4. Some models misread that as "no sources found" → write `SOURCES: none` or a
   URL-less title (attention glm: `w1_s1 (unknown) "Attention Is All You Need" —
   no url`).
5. `_structure_one_draft` produces a ledger with no usable URLs →
   `_render_sources_block` renders nothing (its `_usable_url` gate requires an
   http(s) host) → **THIN**.
6. mrna survived only because one worker cited snippet URLs without depending on
   `read_url`.

## Fix 1 — in-repo prompt mitigation (SHIPPED, needs deploy)

`RESEARCHER_SYSTEM` (`src/audrey/pipeline/prompts.py`) now tells the researcher
that a `web_search` snippet's URL is itself a citable source, and that a failed
`read_url` / `web_fetch` is "I could not read the full page," NOT "the search
found nothing" — so it must cite the search URL rather than fall back to
`SOURCES: none`. This would have rescued attention (whose workers *had* the
paper/arxiv info from search and just declined to attach a URL).

- It does **not** weaken the anti-fabrication guard. "Do not invent a URL for
  something you already knew" is intact — a *search-returned* URL is not invented;
  the fix only rescues URLs the tool actually surfaced this session.
- The paired byte-regression test (`tests/test_prompts.py::
  test_researcher_system_unchanged`) was updated in lockstep.
- `RESEARCH_STRUCTURE_SYSTEM` was deliberately **left untouched** — it already
  wires a claim to a `SOURCES:`-block source when the researcher emits one; the
  gap is upstream (the researcher emitting the block at all), which the
  `RESEARCHER_SYSTEM` edit targets. Editing structuring would treat a symptom and
  risk loosening its strict no-fabrication rule.
- Gate: `.venv/bin/pytest` 758 pass, ruff clean on the changed files.
- **Deploy:** `docker compose up -d --build audrey-ai` (prompt lives in the app).
- This is a **mitigation, not the cure** — it makes workers salvage a snippet URL
  when the page won't open, but they still lose the full-page content `read_url`
  was meant to provide (deeper quotes, exact figures). Fix 2 is the real cure.

## Fix 2 — repair `read_url` / `web_fetch` on `custom-tools` (NOT started; separate repo)

The primary cause. `read_url` / `web_fetch` is failing on ~100% of calls
(`✅0 ❌{1,2}` everywhere it appears). This lives in the **`custom-tools` repo**,
not here, so it can't be fixed in this checkout. Spec for that work:

- **Reproduce + classify the error.** `dispatch_one` collapses the cause into one
  of `timeout` / `network_error` / `http_<code>` / `arguments_*` in the tool
  message body — but the footer only shows ❌. Read the **`custom-tools` container
  logs** on the box (`docker logs custom-tools`) to see the real failure per call.
  Likely candidates, in order:
  - **Timeout** — the research worker's `dispatch_timeout_s` is 30s
    (`agentic.react.research_worker`); a heavy or slow page fetch blows it.
  - **Blocked fetch** — many sites 403 a default user-agent / datacenter IP; the
    fetcher may need a browser UA, or the box's egress is being refused.
  - **Service bug / dependency** — the fetcher endpoint itself erroring (missing
    dep, bad parse, upstream lib).
- **Once classified, the fix is in `custom-tools`**, e.g. a longer fetch timeout +
  a realistic UA + graceful partial-content return. Out of scope for THIS repo.
- **In-repo follow-up worth considering** (separable): a `read_url` result today
  costs a full 30s dispatch window per failure and is unbudgeted (only
  `web_search` is capped in `react.py`). If `read_url` stays flaky, a per-worker
  `read_url` cap + a shorter fetch timeout would stop a dead fetcher from eating
  the worker's wall-clock. Not shipped; flagged only.

## Verification

- **Fix 1:** re-run the same 5-case set
  (`CASES=eval_prompts_writer_ab.json LABEL=research-trace-diag`) after
  `--build audrey-ai`. Success = attention + plate-tectonics come back with a
  non-empty `## Sources` list (they have real search URLs available every run;
  the only reason they were empty was workers dropping them). The new
  `web_search→ctx: N chars` per-worker trace line (Phase-27 follow-on) confirms
  the search content reached the model.
- **Fix 2:** in a later run the `read_url` / `web_fetch` footer shows ✅ counts,
  and thin cases ground *richer* (full-page detail, not just snippet-level).

## Risks

- **Fix 1 low.** Additive prompt sentence + matching test; degrades to today's
  behavior if a model ignores it. Worst case it cites a snippet URL for a page it
  couldn't fully read — which is exactly what mrna's good worker already does, and
  is more honest than `SOURCES: none`.
- **Fix 2 not in this repo** — no risk to this codebase; the risk is entirely on
  the `custom-tools` side and gated by that repo's own tests/deploy.

## What this supersedes

PROJECT_STATE's OPEN QUESTION framed the residual as "per-worker web_search
RELIABILITY (truncation/budget/web_fetch failures)" and asked whether `web_fetch`
was "a separate tool failing independently." Answer: **yes, it's a separate tool
(`read_url`/`web_fetch` from the custom-tools repo), and it's failing ~always —
but `web_search` itself is healthy.** The budget and merge angles were both dead
ends (budget is generously tuned at `max_web_searches:6`/`keep_last:5`; the merge
unions rather than intersects). Update the OPEN QUESTION to point here.

## Sub-finding: the mrna KB junk — a co-mingled corpus, not a bug

The mrna case (`sources:0`, 2026-07-15 trace) had a second cause beside the
`read_url` failure: all three workers' `kb_search` returned totally off-topic docs
(PowerApps, ServiceNow, Forest-Service medicinal plants) for a vaccine query.

**Root cause (data-proven, not theorized):** `kb.dataset_paths` in `config.yaml`
**deliberately loads `powerapps` and `servicenow`** alongside the research corpus
(geology/botany/bushcraft/fishing/survival/first-aid/…). One KB serves two
purposes — research grounding AND ops/work docs — and a research worker's
`kb_search` sees both. So a research query that happens to be nearest an ops doc
retrieves it; the ops doc is a *genuine* high-similarity match for its own topic,
not corrupt data.

**A cosine floor (`kb.min_score`) was tried and is only a PARTIAL fix.** The
floor (added this phase in `routes/kb.py` `_search_text_merged`, applied
before the `top_k` cut, over-fetching so a below-floor near-neighbour can't
starve a real hit) is set to **0.53** and is **deployed + verified live**
(2026-07-16). Its measured net effect, from `scripts/kb_score_probe.py` (22
labeled queries) + a live `/v1/kb/query` probe:
- **Win:** genuinely-foreign queries now return EMPTY instead of nearest-junk —
  `capital of France`, `TCP vs UDP`, `sourdough`, and `how do mRNA vaccines work`
  all → `{"results":[]}` (their top hit fell below 0.53). This is the direct fix
  for the mrna symptom: no more PowerApps injected into a vaccine researcher's
  context when nothing relevant exists.
- **Does NOT fix ops-doc leakage:** a `PowerApps pipeline` query still scores 0.69
  and survives, because it's a *real* match for the deliberately-loaded PowerApps
  docs. No floor below the real-hit floor (~0.54, e.g. a fishing-knot hit) can
  exclude it.
- **Small cost:** cuts the weakest real hits (a medicinal-plants hit at 0.53).

**Why a floor can't fully solve it, and the harness that proved it:** the initial
0.53 came from two hand-probes that showed a clean valley (on-domain 0.57–0.59 vs
off-domain 0.49–0.52). `scripts/kb_score_probe.py` (built to get more data) swept
22 queries and found the distributions actually **OVERLAP** — real on-domain hits
as low as 0.54, off-domain/ops hits as high as 0.69 — so no single global cosine
floor cleanly separates them. The two-point estimate was a sampling artifact; the
harness caught it. **Keep the harness; re-run it after any corpus change.**

**The real fix (NOT done — bigger, separable):** research `kb_search` should not
see the ops datasets at all. Either drop `powerapps`/`servicenow` from
`kb.dataset_paths`, or route research queries to a research-only collection
(separate from the ops KB). That's an ingest/collection-topology change; the 0.53
floor is the cheap partial win kept in the meantime. **Decision (2026-07-16, with
user): keep the 0.53 floor as-is; corpus separation is the deferred proper fix.**

**Harnesses shipped this phase** (both hermetic where possible):
- `scripts/kb_score_probe.py` + `kb_probe_queries.json` — probe `/v1/kb/query` with
  labeled on/off-domain queries, report score distributions + the safe-floor
  window (or "OVERLAP — no clean cut"). Run on the box via a throwaway container
  on `ollama-net` mounting the host scripts dir (the running `audrey-ai`/
  `custom-tools` images don't carry newly-pulled scripts until rebuilt).
- `scripts/sources_block_probe.py` — replay captured/real ledgers through the REAL
  `_render_sources_block` to catch Sources-rendering regressions (seeds the
  attention url-less-linked shape + surrounding contracts). Fully laptop-hermetic.
