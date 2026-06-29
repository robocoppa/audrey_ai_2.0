# Campaign 2 Phase 26 — research claim/source ledger + SearXNG fallback

`audrey_research` gains a structured claim/source **ledger** the pipeline reasons
over (Stages 0–3), plus a self-hosted **SearXNG fallback** for `web_search` when
Brave is quota-exhausted. Built and deployed in eval-gated stages; this doc covers
everything through Stage 3 (the deterministic Sources list), last updated
2026-06-28.

**Status:** Stages 0–2 are deployed and **validated on the box at 3/3 workers**
(see the worker-drop fixes below). Stage 3 is **built + hermetically tested,
awaiting its box deploy.**

See `phase-26-research-ledger-plan.md` for the full staged design and the
controlling principle (the ledger is INTERNAL scaffolding, not user-facing
bookkeeping — never the +21 citation-mandate that was reverted).

## What it does

**The ledger (Stages 0–2):**
- **Stage 0 (dark):** `OllamaClient.chat` forwards Ollama's `format` (JSON-schema)
  field; `pipeline/ledger.py` holds the Pydantic schema (`Source`/`Claim`/
  `ResearchResult`/`ClaimCheck`/`FactCheckResult`, with `company_claim` a
  first-class source type) + fail-soft parsers that never raise.
- **Stage 1:** after researchers return, a second mechanical pass per worker
  structures their prose into a `ResearchResult` (claims + sources), merged
  (sources URL-deduped, claims kept across workers, ids worker-prefixed).
  `RESEARCHER_SYSTEM` now ends notes with a `SOURCES:` section so sources are
  captured. Opt-in via `agentic.research_ledger.enabled` (bind-mounted → toggles
  live).
- **Stage 2:** the fact-checker's prose is structured into a `FactCheckResult`
  (per-claim verdicts) and rendered into the existing `corrections` string the
  writer applies — `unsupported → DROP` (omit), `conflicting → UNVERIFIED`
  (hedge), `needs_hedge → CORRECT`. Writer prompt learned `DROP:`.
- **Stage 3:** after the writer finishes a clean answer, a deterministic
  `## Sources` list is **appended by the pipeline** (NOT asked of the writer —
  that's the +21 citation pressure that was reverted). It lists the ledger's
  *surviving* sources — those backing a claim the fact-checker did **not** drop —
  deduped by URL, capped at 8, http(s)-host URLs only. The block is **omitted
  entirely** when there's no ledger, no surviving source, or no usable URL, so
  creative/ungrounded answers stay clean (no empty header). When the model left
  claim↔source linkage empty, it falls back to listing all consulted sources.

All ledger work is **fail-soft**: any parse/tool error degrades to the prior
prose behaviour for that request, never breaks the answer.

### Worker-drop fixes (Stages 1–2 robustness — validated 3/3 on box)

The per-worker structuring pass was silently **discarding whole worker ledgers**
when the model emitted structurally-valid JSON with off-spec values that strict
Pydantic rejected (one bad field nuked all of a worker's claims+sources; the box
showed 1/3, then 2/3 workers surviving). Found via the per-field
`ValidationError on fields [...]` log, which named each offender in turn. All
fixed in `ledger.py` with tolerant `BeforeValidator`s that coerce instead of
reject — never discard a worker:

- required `id` → optional + positional backfill (`c1`, `s1`, …)
- integer ids → string-coerced
- off-enum `risk` / `verdict` / `source_type` (`"High"`, ints, `"wikipedia"`) →
  normalized to the nearest enum member (or `medium`/`irrelevant`/`unknown`)
- **`url` / `title` = `null` or non-string → `""`** (`_to_str_or_empty`)

**Confirmed on box: 3/3 workers, no ValidationErrors, total time ~640s → ~300s**
(the silent retries were the time sink). Durable lesson (also in AGENTS.md): a
`format=`-pinned model "returning unusable JSON" that fails on a DIFFERENT model
each run is a parse/schema problem, not a model problem — log the raw reply /
per-field error before swapping models.

**SearXNG fallback:** when Brave returns **402** (quota exhausted) or **429**
(rate-limit), `web_search` falls back to a self-hosted **SearXNG** meta-search
instance (JSON API) so research grounding survives a Brave outage. Same response
shape — Audrey's research path is unchanged. Configured via `SEARXNG_URL` (empty
→ no fallback → 503 on a Brave outage).

> A DuckDuckGo HTML-scrape fallback was tried first and **abandoned**: under the
> research panel's query volume DDG served an HTTP 202 "anomaly" bot-block page
> (0 results), causing empty ledgers. SearXNG runs on your own infra, so no
> bot-blocking. (No DDG artifacts remain in the tree.)

## Files changed

**audrey-ai image** (`src/`):
- `models/ollama.py` — `chat(..., format=…)` param.
- `pipeline/ledger.py` — NEW: schema, parsers (`_strip_fence`/`_extract_json`,
  `strict=False`, bare-array normalization), `inlined_schema()`, and the tolerant
  `BeforeValidator`s that closed the worker-drop bug (`_to_str`/`_to_str_list`,
  `_to_str_or_empty`, `_norm_risk`/`_norm_verdict`/`_norm_source_type`,
  `_backfill_ids`, per-field `ValidationError` logging).
- `pipeline/prompts.py` — `RESEARCH_STRUCTURE_SYSTEM`, `FACTCHECK_STRUCTURE_SYSTEM`,
  researcher `SOURCES:` line, writer `DROP:` line, 2 new prompt keys.
- `pipeline/deep_panel.py` — structuring calls, ledger merge, factcheck-ledger
  render, diagnostic logging, and the **Stage 3** Sources renderer (`_usable_url`,
  `_surviving_source_ids`, `_render_sources_block`; `fc_result` hoisted to pipeline
  scope so verdicts reach the renderer; block streamed as a `write_delta` after
  the writer).
- `config.yaml` — `agentic.research_ledger.enabled` flag; research-pool
  researcher swap `qwen3.5:397b-cloud → glm-5.2:cloud`.

**custom-tools image** (`tools-server/`):
- `searxng.py` — NEW: SearXNG JSON-API client (the web_search fallback).
- `settings.py` — NEW `SEARXNG_URL` (empty → no fallback).
- `brave.py` — NEW `BraveQuotaError` for 402 (not retried → triggers fallback).
- `app.py` — `web_search` handler falls back to SearXNG on quota/rate-limit;
  SearXNG client (optional) constructed/closed in lifespan.

**SearXNG instance (you run this on Unraid):**
- `searxng/searxng` container on a LAN port (e.g. 8088).
- Its `settings.yml` must enable the JSON API: `search.formats: [html, json]`
  (off by default — the API returns 403 without it).
- Set `SEARXNG_URL=http://192.168.1.11:8088` in the custom-tools env.

## Verification (laptop, hermetic)

- **601 pytests pass**, ruff clean. New across this phase: `test_ledger.py`
  (schema + parsers + inliner + strict/fence/array cases + the off-enum and
  null-url/title tolerance cases), `test_searxng.py` (JSON parser + empty-url skip
  + 402→quota-error), Stage-1/2 helper tests + the Stage-3 Sources tests in
  `test_deep_panel.py` (`_render_sources_block` unit cases + one end-to-end test
  that drives the ledger-enabled streaming pipeline and asserts the `## Sources`
  block reaches the stream).
- **SearXNG client** is hermetically tested against a representative JSON
  response; the live verification is the box smoke test (needs the SearXNG
  instance running).

## Deploy

Two images. The ledger work (incl. Stage 3) is in `audrey-ai`; the SearXNG
fallback is in `custom-tools`.

```bash
# audrey-ai (ledger stages + Stage 3 — code baked in, needs rebuild)
docker compose up -d --build audrey-ai
# custom-tools (SearXNG fallback)
docker compose up -d --build custom-tools
```

The ledger flag (`agentic.research_ledger.enabled`) is in the bind-mounted
`config.yaml` — toggle live with `docker compose restart audrey-ai`, no rebuild.

**Confirm the new code is actually in the running image** before probing (a stale
image is the usual reason a change "doesn't take"):

```bash
# Stage 3 (this deploy):
docker exec audrey-ai grep -c "_render_sources_block" /app/src/audrey/pipeline/deep_panel.py
# worker-drop fix (already live):
docker exec audrey-ai grep -c "_to_str_or_empty" /app/src/audrey/pipeline/ledger.py
```

`≥1` = new code is live; `0` = the image predates the commit, rebuild.

## Box smoke test (the real verification)

1. **SearXNG fallback live** — send one `audrey_research` request, then:
   ```
   docker logs custom-tools 2>&1 | grep "falling back to SearXNG"
   docker logs custom-tools 2>&1 | grep "SearXNG returned"
   ```
   With Brave 402'd, every `web_search` should log the fallback AND
   `SearXNG returned N results` with N > 0 (the DDG attempt returned 0 — that's
   the regression this replaces).
2. **Ledger builds at full strength** —
   ```
   docker logs audrey-ai 2>&1 | grep "ledger built" | tail -1
   docker logs audrey-ai 2>&1 | grep "factcheck ledger" | tail -1
   docker logs audrey-ai 2>&1 | grep "ValidationError on fields" | tail -3
   ```
   Expect `ledger built — N claims, M sources from 3/3 workers` (M > 0),
   `factcheck ledger — N checks (X drop, Y hedge)`, and the `ValidationError`
   grep **empty** (a non-empty line names a field a model is still emitting
   off-spec — add a tolerant validator for exactly that field, same as the
   url/title fix).
3. **Stage 3 Sources list renders (and stays absent when ungrounded)** —
   ```
   docker logs audrey-ai 2>&1 | grep "Sources block appended" | tail -1
   ```
   Probe a grounded case (`bio-euclid`): the answer must END with a `## Sources`
   list of clickable URLs. Probe a creative/ungrounded case
   (`ctrl-birthday-toast`): there must be **NO** `## Sources` block (and no log
   line for it). The eval harness prints the full answer, so eyeball both.
4. **Full eval** — re-run the protocol into a new
   `docs/testing/<date>-ledger-stage3-answers.md` and diff vs the Stage-1/2
   baselines. Does a bio/history case correctly DROP/hedge a claim its sources
   don't support, while `current-2025-recent` keeps its releases — and does each
   grounded answer now carry a clean Sources list without any inline-citation
   clutter in the prose?

## Known issues / notes

- **Brave key is quota-exhausted (402).** SearXNG fallback keeps grounding alive;
  renewing Brave restores the primary provider (SearXNG then idles as fallback).
- **SearXNG must have the JSON API enabled** (`search.formats: [html, json]` in
  `settings.yml`) or every fallback returns 403. See
  `searxng-unraid-setup.md`.
- **Research totals are long** (~300s for `bio-euclid`, down from ~640s before the
  worker-drop fixes removed silent retries). Not a blocker, but watch for
  near-timeout runs on heavier prompts.
- **Stage 4 (deterministic hedge policy + per-claim writer disposition) not yet
  built.**

## Risks

- **Low** for the ledger + Stage 3: opt-in flag, fail-soft everywhere, prose path
  unchanged when disabled or on any error; the Sources block is a pure append that
  only fires on a clean grounded answer and omits itself otherwise.
- **Low** for SearXNG: only fires on Brave 402/429; a clean 503 if both providers
  fail (same shape as before). Self-hosted, so no bot-blocking or external quota.
