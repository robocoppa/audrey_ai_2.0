# Campaign 2 Phase 26 — research claim/source ledger + SearXNG fallback

`audrey_research` gains a structured claim/source **ledger** the pipeline reasons
over (Stages 0–4), plus a self-hosted **SearXNG fallback** for `web_search` when
Brave is quota-exhausted. Built and deployed in eval-gated stages; this doc covers
the whole arc through Stage 4 (deterministic hedging) + the operational lessons
learned deploying it. Last updated 2026-06-29.

**Status:** all five stages (0–4) are built, deployed, and **validated on the box.**
The close-out eval (Stage 4 flag on, over-hedge fix + source ranking live) confirmed
all three changes: the ungrounded `ctrl-explain-recursion` control is a **clean,
confident tutorial again** (zero over-hedge markers — the all-hedge suppression
works), grounded Sources lists **lead with authoritative domains** (Wikipedia/arXiv/
tokio.rs; no facebook-groups/scribd/slideshare survived the ranked cap), and the
hedging behaves correctly on real uncertainty (when a case's live grounding came
back thin — e.g. `current-2025-recent` this run — it correctly caveated rather than
faking confidence, which is the *right* hedge, not the over-hedge the fix targets).
**Note on run-to-run variance:** which cases get a Sources block varies between runs
because the live web (via SearXNG) returns different results per query — the +38 run
grounded the bios richly, this run grounded fewer. That's grounding variance, not a
code regression; the pipeline shaped each answer correctly for the grounding it got.
See "What we learned" below for the full deploy history, including the two
operational failures that cost the most time (a stopped SearXNG container, a silent
fallback 503).

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
  **ranked by domain authority** (`_source_rank`: official/primary_paper/
  scholarly/reference = top tier, then news, then company_claim/blog, then
  unknown), deduped by URL, capped at 8, http(s)-host URLs only. The ranking
  (stable sort, ties keep ledger order) means the cap keeps the *most
  authoritative* sources — a Stanford/arXiv/Wikipedia link isn't crowded out by a
  facebook-groups or scribd URL SearXNG happened to surface. The block is
  **omitted entirely** when there's no ledger, no surviving source, or no usable
  URL, so creative/ungrounded answers stay clean (no empty header). When the model
  left claim↔source linkage empty, it falls back to listing all consulted sources.
- **Stage 4:** deterministic **selective hedging**. A pure function
  `hedge_policy(claim, source_types)` assigns each surviving claim a disposition
  — `state_plainly` (authoritative source + not high-risk), `attribute_to_company`
  (a `company_claim` source — name the vendor, don't endorse), `hedge`
  (`needs_hedge`, or no authoritative grounding), `hedge_or_cite_strongly` (high
  risk) — rendered into a `CLAIM DISPOSITIONS` block the writer applies (STATE
  PLAINLY = assert with NO hedging, the lever that stops well-grounded facts from
  being over-hedged). Opt-in via a **separate** flag
  `agentic.research_ledger.hedge_policy` (so it can be A/B'd against a ledger-on
  baseline). **Over-hedge guard:** the block is suppressed entirely when *every*
  disposition is `hedge` — an all-hedge block carries no signal beyond "be
  cautious about everything," which turned a plain "explain recursion" into hedged
  mush on the ungrounded controls; it earns its place only with ≥1
  plain-statement/attribution to make.

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
  render, diagnostic logging; the **Stage 3** Sources renderer (`_usable_url`,
  `_surviving_source_ids`, `_render_sources_block`; `fc_result` hoisted to pipeline
  scope so verdicts reach the renderer; block streamed as a `write_delta` after
  the writer); **source-authority ranking** (`_SOURCE_RANK`/`_source_rank`, stable
  sort before the cap); and the **Stage 4** hedging wiring (`_hedge_policy_enabled`,
  `_source_types_for_claim`, `_render_dispositions_block` with the all-hedge
  suppression guard; `dispositions` threaded into `_write_user_block`).
- `pipeline/ledger.py` — also the **Stage 4** `hedge_policy()` pure function +
  `HedgeDisposition` type (the policy table; unit-tested, no model needed).
- `config.yaml` — `agentic.research_ledger.enabled` flag; **`hedge_policy` flag**
  (separate, ships `false`); research-pool researcher swap
  `qwen3.5:397b-cloud → glm-5.2:cloud`.

**custom-tools image** (`tools-server/`):
- `searxng.py` — NEW: SearXNG JSON-API client (the web_search fallback).
- `settings.py` — NEW `SEARXNG_URL` (empty → no fallback).
- `brave.py` — NEW `BraveQuotaError` for 402 (not retried → triggers fallback).
- `app.py` — `web_search` handler falls back to SearXNG on quota/rate-limit;
  SearXNG client (optional) constructed/closed in lifespan. **Logs the
  fallback-failure path** (`SearxngError → log.warning` before the 503) — added
  after a stopped SearXNG container failed *silently* (see "What we learned").

**SearXNG instance (you run this on Unraid):**
- `searxng/searxng` container on a LAN port (e.g. 8088).
- Its `settings.yml` must enable the JSON API: `search.formats: [html, json]`
  (off by default — the API returns 403 without it).
- Set `SEARXNG_URL=http://192.168.1.11:8088` in the custom-tools env.

## Verification (laptop, hermetic)

- **624 pytests pass**, ruff clean. New across this phase: `test_ledger.py`
  (schema + parsers + inliner + strict/fence/array cases, the off-enum and
  null-url/title tolerance cases, and the **Stage-4 `hedge_policy` policy-table**
  cases — incl. the plan's three worked examples: official date→plainly,
  vendor benchmark→attribute, ancient anecdote→hedge), `test_searxng.py` (JSON
  parser + empty-url skip + 402→quota-error), and in `test_deep_panel.py`:
  Stage-1/2 helper tests, the Stage-3 Sources tests (`_render_sources_block` units
  + an end-to-end test asserting the `## Sources` block reaches the stream), the
  **source-ranking** tests (authoritative-beats-weak-at-cap, stable-within-tier),
  and the **Stage-4** tests (disposition rendering, all-hedge suppression,
  flag-gating, two end-to-end proving dispositions reach the writer with the flag
  on and are absent with it off).
- **SearXNG client** is hermetically tested against a representative JSON
  response; the live verification is the box smoke test (needs the SearXNG
  instance running — and *running*: see "What we learned").

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
docker exec audrey-ai grep -c "_render_sources_block" /app/src/audrey/pipeline/deep_panel.py  # Stage 3
docker exec audrey-ai grep -c "_to_str_or_empty"      /app/src/audrey/pipeline/ledger.py      # worker-drop fix
docker exec audrey-ai grep -c "any_non_hedge"         /app/src/audrey/pipeline/deep_panel.py  # Stage 4 over-hedge guard
docker exec audrey-ai grep -c "_source_rank"          /app/src/audrey/pipeline/deep_panel.py  # source ranking
```

`≥1` = new code is live; `0` = the image predates the commit, rebuild.

And confirm the Stage-4 flag is actually ON (it ships `false` — Stage 4 is a
no-op until enabled in the bind-mounted config + a `docker compose restart`):

```bash
docker exec audrey-ai grep "hedge_policy:" /app/config.yaml   # want: hedge_policy: true
```

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
4. **Stage 4 hedging (flag on)** — the payoff case is `current-2025-recent`:
   official release dates should read **plainly** (no "reportedly"), while vendor
   benchmark claims are **attributed** ("Meta reports…", "industry coverage
   notes…"). Bios should keep ancient anecdotes hedged. The ungrounded control
   `ctrl-explain-recursion` must be a **clean, confident tutorial** — if it's full
   of "often described as / commonly understood to," the all-hedge suppression
   guard isn't live (rebuild) or the flag flipped a stale image.
5. **Source ranking** — grounded Sources lists should **lead with authoritative
   domains** (Wikipedia/Stanford/arXiv/Britannica); weak domains
   (facebook-groups, scribd, slideshare) should be pushed down or off the
   capped-at-8 list.
6. **Full eval** — re-run the protocol into a new
   `docs/testing/<date>-ledger-stageN-answers.md` and diff vs the prior baseline.
   Does a bio/history case correctly DROP/hedge a claim its sources don't support,
   while `current-2025-recent` keeps its releases — clean Sources list, no
   inline-citation clutter in the prose, authoritative sources on top?

## What we learned (the deploy history that cost the most time)

This phase was built and shipped in eval-gated stages, and most of the calendar
time went not into writing features but into **diagnosing failures that masqueraded
as something else**. The durable lessons:

**1. A `format=`-pinned model "returning unusable JSON" that fails on a DIFFERENT
model each run is a PARSE/SCHEMA problem, not a model problem.** The Stage-1
structuring pass silently discarded whole worker ledgers when a model emitted
structurally-valid JSON with off-spec field values that strict Pydantic rejected
(saw 1/3, then 2/3 workers surviving). Two deploy cycles were wasted swapping the
model and inlining the schema before the fix that actually mattered: **log the raw
reply and the per-field `ValidationError` first.** The per-field log named each
offender in turn, and the fix was tolerant `BeforeValidator`s that coerce instead
of reject (optional+backfilled ids, str-coerced ints, off-enum normalization,
null/non-string url→""). Never discard a worker; degrade the field. (Also in
AGENTS.md.)

**2. A dead dependency can fail SILENTLY and look like a feature regression.** The
worst single time-sink: after deploying Stage 3, an eval showed `web_search`
failing on every case and answers degrading to "retrieval returned nothing." It
looked like a code/Stage-3 bug. It wasn't — the **`searxng` container had no
Autostart set**, so after a restart it was simply stopped. From inside custom-tools
the `searxng` hostname gave `Name or service not known`; the fallback fired, hit a
dead host, and 503'd. **Two compounding causes made it invisible:**
   - The `except SearxngError` branch raised the 503 **without logging** — the only
     log evidence was "falling back to SearXNG" with nothing after it. *Fixed:* a
     `log.warning` before the raise. **Lesson: log the failure path before you raise
     from it** — same instinct as #1, applied to a fallback.
   - Stage 3 still rendered its (empty) behaviour fine, so the *answer* looked
     plausible — just quietly ungrounded. **Lesson: when grounding silently dies,
     the answer degrades gracefully enough to hide it. Watch the tool footers and
     the provider logs, not just the prose.**
   - **Fast triage** (beats re-running the ~30-min eval): probe the exact fallback
     path from inside the container —
     `docker exec custom-tools python -c "import httpx; print(httpx.get('http://searxng:8080/search', params={'q':'x','format':'json'}, timeout=10).status_code)"`.
     `Name or service not known` = stopped/off-network; `403` = JSON API disabled;
     `200` = healthy, look elsewhere. **SearXNG must have Autostart ON** — it's a
     grounding dependency for `audrey_research`, not optional, while Brave is 402'd.

**3. The `❌` in the "Tools used" footer means "a tool call errored ≥once," NOT
"retrieval failed."** During the SearXNG outage every `web_search` showed `❌`,
which made a *successful* design look broken. With the fallback healthy the footers
show `web_search ×N` with no `❌`. Read the provider logs to tell a real outage from
a noisy footer.

**4. Deterministic hedging over-corrects on UNGROUNDED answers.** Stage 4 worked as
designed on grounded factual content (the `current-2025-recent` payoff: plain dates,
attributed vendor claims). But its first box eval over-hedged a plain "explain
recursion" into mush — an ungrounded answer's claims have no authoritative source,
so `hedge_policy` returned `hedge` for *all* of them and the writer blanket-hedged.
*Fixed:* suppress the whole disposition block when every disposition is `hedge` (it
carries no signal the writer's own caution rules don't already cover). **Lesson:
"hedge the uncertain things" needs a floor — an all-hedge instruction is just
blanket caution, which is the over-cautious regression this whole research arc
(see the +21→+26 citation revert) kept fighting.** First instinct (gate on empty
`findings`) was *wrong* — the control HAD findings, it just had no authoritative
sources; verified by checking the saved answer had no Sources block.

**5. Judge the source URL, not the provider.** SearXNG surfaces whatever its engines
return, including SEO/forum sludge (facebook-groups, scribd, slideshare). The fix
was NOT to distrust SearXNG wholesale — it was to **rank by domain authority** so the
capped Sources list keeps the best URLs regardless of which provider found them. A
provider-level tiebreak was considered and dropped: the ledger doesn't track
provider per source, and that signal can't survive the prose→structuring path
(the structuring model isn't told the provider).

**6. The eval harness proves liveness, not truth — and quality regressions only
show on a careful read of the saved answers.** Every quality finding this phase
(the over-hedge, the weak-domain Sources, the worker drops) came from *reading* the
saved `docs/testing/*-answers.md`, not from the harness's structural PASS. The
harness is the gate that the plumbing works; the human/Claude read is the gate that
the answer is good. Keep both, and diff each run against the prior baseline.

## Known issues / notes

- **Brave key is quota-exhausted (402).** SearXNG fallback keeps grounding alive;
  renewing Brave restores the primary provider (SearXNG then idles as fallback).
- **SearXNG must have the JSON API enabled** (`search.formats: [html, json]` in
  `settings.yml`) or every fallback returns 403. See
  `searxng-unraid-setup.md`.
- **Research totals are long** (~300–600s; `bio-euclid` ~300s, `bio-pythagoras`
  ran 624s on the Stage-4 eval). Not a blocker, but watch for near-timeout runs on
  heavier prompts.
- **Stage-4 verbosity (open watch-item).** On the Stage-4 eval `bio-pythagoras`
  came back bloated with near-duplicate claim restatements — the
  `CLAIM DISPOSITIONS` block may be too long/numerous, nudging the writer to
  enumerate rather than compose. Candidate follow-up: shorten the per-claim text or
  cap the disposition count. Not yet addressed.
- **All five stages are built and deployed**; Stage 4 ships behind `hedge_policy:
  false` (flip live + restart to enable).

## Risks

- **Low** for the ledger + Stages 3/4: opt-in flags, fail-soft everywhere, prose
  path unchanged when disabled or on any error; the Sources block is a pure append
  that only fires on a clean grounded answer and omits itself otherwise; the
  disposition block is suppressed unless it has real (non-hedge) signal to add.
- **Low** for SearXNG: only fires on Brave 402/429; a clean 503 if both providers
  fail (same shape as before). Self-hosted, so no bot-blocking or external quota.
  *Operational* risk is the real one — it must be running and Autostart-enabled
  (see "What we learned" #2).
