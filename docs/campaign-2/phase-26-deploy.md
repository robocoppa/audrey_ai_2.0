# Campaign 2 Phase 26 — research claim/source ledger + DuckDuckGo fallback

`audrey_research` gains a structured claim/source **ledger** the pipeline reasons
over (Stages 0–2 so far), plus a keyless **DuckDuckGo fallback** for `web_search`
when Brave is quota-exhausted. Built and deployed in eval-gated stages; this doc
covers what's shipped through the DDG-fallback deploy (2026-06-28).

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

All ledger work is **fail-soft**: any parse/tool error degrades to the prior
prose behaviour for that request, never breaks the answer.

**SearXNG fallback:** when Brave returns **402** (quota exhausted) or **429**
(rate-limit), `web_search` falls back to a self-hosted **SearXNG** meta-search
instance (JSON API) so research grounding survives a Brave outage. Same response
shape — Audrey's research path is unchanged. Configured via `SEARXNG_URL` (empty
→ no fallback → 503 on a Brave outage).

> A DuckDuckGo HTML-scrape fallback was tried first and **abandoned**: under the
> research panel's query volume DDG served an HTTP 202 "anomaly" bot-block page
> (0 results), causing empty ledgers. SearXNG runs on your own infra, so no
> bot-blocking.

## Files changed

**audrey-ai image** (`src/`):
- `models/ollama.py` — `chat(..., format=…)` param.
- `pipeline/ledger.py` — NEW: schema, parsers (`_strip_fence`/`_extract_json`,
  `strict=False`, bare-array normalization), `inlined_schema()`.
- `pipeline/prompts.py` — `RESEARCH_STRUCTURE_SYSTEM`, `FACTCHECK_STRUCTURE_SYSTEM`,
  researcher `SOURCES:` line, writer `DROP:` line, 2 new prompt keys.
- `pipeline/deep_panel.py` — structuring calls, ledger merge, factcheck-ledger
  render, diagnostic logging.
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

- **583 pytests pass**, ruff clean. New: `test_ledger.py` (schema + parsers +
  inliner + strict/fence/array cases), `test_searxng.py` (JSON parser + empty-url
  skip + 402→quota-error), Stage-1/2 helper tests in `test_deep_panel.py`.
- **SearXNG client** is hermetically tested against a representative JSON
  response; the live verification is the box smoke test (needs the SearXNG
  instance running).

## Deploy

Two images. The ledger work is in `audrey-ai`; the DDG fallback is in
`custom-tools`.

```bash
# audrey-ai (ledger stages — prompts baked in, needs rebuild)
docker compose up -d --build audrey-ai
# custom-tools (DDG fallback)
docker compose up -d --build custom-tools
```

The ledger flag (`agentic.research_ledger.enabled`) is in the bind-mounted
`config.yaml` — toggle live with `docker compose restart audrey-ai`, no rebuild.

## Box smoke test (the real verification)

1. **SearXNG fallback live** — send one `audrey_research` request, then:
   ```
   docker logs custom-tools 2>&1 | grep "falling back to SearXNG"
   docker logs custom-tools 2>&1 | grep "SearXNG returned"
   ```
   With Brave 402'd, every `web_search` should log the fallback AND
   `SearXNG returned N results` with N > 0 (the DDG attempt returned 0 — that's
   the regression this replaces).
2. **Ledger builds** —
   ```
   docker logs audrey-ai 2>&1 | grep "ledger built"
   docker logs audrey-ai 2>&1 | grep "factcheck ledger"
   ```
   Expect `ledger built — N claims, M sources` (M > 0 with grounding restored)
   and `factcheck ledger — N checks (X drop, Y hedge)`.
3. **Full eval** — re-run the protocol into a new
   `docs/testing/<date>-ledger-stage2-rerun-answers.md` and diff vs the Stage-1
   baseline. THIS is the valid Stage-2 measurement (the prior run was confounded
   by the Brave outage): does a bio/history case now correctly DROP/hedge a claim
   its sources don't support, while `current-2025-recent` keeps its releases?

## Known issues / notes

- **Brave key is quota-exhausted (402).** DDG fallback keeps grounding alive but
  is lower quality; renewing Brave is the better long-term fix.
- **DDG parser is HTML-scrape** — fragile to DDG markup changes (regex-based,
  tested). If results suddenly go empty, check `duckduckgo.py`'s regexes against
  the current page.
- **Stages 3 (source-bound writer + end-of-answer Sources list) and 4
  (deterministic hedge policy) not yet built.**

## Risks

- **Low** for the ledger: opt-in flag, fail-soft everywhere, prose path unchanged
  when disabled or on any error.
- **Low** for DDG: only fires on Brave 402/429; a clean 503 if both providers
  fail (same shape as before). No new dependency (httpx + regex).
