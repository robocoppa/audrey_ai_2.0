# Plan — Improve deep-mode answers on recency/grounding questions

_Created 2026-07-18. Trigger: `audrey_deep` answer to a "what happened at this
year's World Cup" prompt was confidently-worded on structure but hedged on
everything time-sensitive, despite a worker having fetched 8000 chars of web
evidence._

## Symptom (the observed run)

Deep-mode ("DeepThink") answer to a 2026 World Cup question gave correct
structural facts (48 teams, three hosts, format) but hedged the whole answer
("either just wrapping up or has just concluded") and could not report any
results. Panel debug drafts:

- `deepseek-v4-pro:cloud` — 10.4s, **2 tool rounds, web_search→ctx: 8000 chars**,
  then: *"the earlier web searches were compacted out of context, and the
  tool-call budget has been reached, so I can't run new searches."*
- `kimi-k2.6:cloud` — 20.9s, **0 tool rounds** (never searched).
- `qwen3.6:35b` — 88.3s, 1 tool round, web_search→ctx: 2835 chars, no results
  surfaced.

The datetime injection is working — kimi and qwen both correctly stated "July 20,
2026" in their drafts (they can only know that from the injected system message,
`pipeline/context.py`). So this is **not** a date bug.

## Root cause — TWO independent defects

### Defect A (primary): ReAct evicts fetched grounding before the worker can use it

Confirmed by reading `src/audrey/pipeline/react.py`:

- Compaction fires at `react.py:190` once `round_idx >= compress_after_round`,
  replacing older `role=tool` results with a *"[history compacted: an earlier
  `web_search` result is omitted…]"* stub (`_summarize_tool_message`,
  `react.py:88`), keeping only `compress_keep_last` tool messages verbatim
  (`_compress_history`, `react.py:114`).
- The web_search budget (`max_web_searches`) stubs any over-budget call with
  `_WEB_SEARCH_BUDGET_STUB` (`react.py:96`) and stops offering web_search in
  later rounds (`react.py:282`).

deepseek's own words map 1:1 to these two code paths. With the `deep_worker`
budget, it fetched good evidence in round 1, then in round 2 compaction evicted
that evidence **and** the web_search budget blocked a re-fetch — so it was left
holding a compacted scratchpad it couldn't rebuild. The 8000 chars reached
context, then were thrown away before the answer step.

This is a **config-value** problem (the knobs exist and are live-tunable via
`config.yaml`, no rebuild), NOT a code bug. The four knobs, under
`agentic.react.deep_worker` (falling back to `agentic.react.*`), read in
`deep_panel._react_budget` (`deep_panel.py:1234`):

| Knob | Effect | Suspected wrong setting |
|---|---|---|
| `compress_after_round` | round index at which older tool results get stubbed | too low → round-1 evidence stubbed by round 2 |
| `compress_keep_last` | how many recent tool messages survive verbatim | too low (1) → only the newest result kept |
| `max_web_searches` | per-worker web_search cap (0 = unlimited) | too low → can't re-fetch after eviction |
| `max_rounds` | total ReAct rounds | if tight, fetch-then-use doesn't fit |

### Defect B (secondary): deep-mode synthesizer has no anti-over-hedge guardrail

`SYNTH_SYSTEM` (`prompts.py:81`, used by `pipeline/synthesize.py`) instructs the
synthesizer only toward caution ("soften it or drop it", add a `## Caveats`
section). It has **no** counterbalancing "state confident known facts plainly"
clause. Research mode's `WRITER_SYSTEM` (`prompts.py:283`) DOES — e.g. *"a
confident known fact plus an honest 'I can't verify the rest' beats a wall of
hedged maybes"* — plus the `_render_dispositions_block` all-plain/all-hedge
suppression machinery in `deep_panel.py`. Research mode was hardened against the
over-hedge trap (the +21/+36 evals); deep mode never got that treatment.

So even when deep mode DID know the structural facts confidently, `SYNTH_SYSTEM`
wrapped them in caution. Fixing B without A yields a confidently-worded answer
that still can't report results; fixing A without B may still hedge facts it
knows. Both matter, A first.

## Open questions (need the box / can't resolve from the laptop)

1. **Live `agentic.react.deep_worker` values** on the deployed `config.yaml` —
   the whole of Defect A hinges on the current numbers. Laptop can't read them.
2. **Was search actually returning results?** kimi searched 0×; deepseek's
   grounding was thin. Per prior notes, an empty SearXNG result shows as ✅ in the
   footer, so "web_search ✅4" doesn't prove useful results. If the grounding
   source was dry at query time, tuning A/B won't help — confirm SearXNG/Brave
   were returning World Cup hits.
3. **Should this prompt route to research mode at all?** A "what happened at this
   year's X" question is a grounding task — `audrey_research`'s wheelhouse. Deep
   mode is a parallel-panel/synthesis design, better at reasoning than fresh-fact
   retrieval.

## Plan (ordered by leverage)

### Step 1 — Read the live deep_worker ReAct budget (diagnostic, box) — DONE

Live values read from the deployed `config.yaml` (2026-07-18):

| Knob | `deep_worker` (wrote the bad answer) | `research_worker` (hardened) |
|---|---|---|
| `max_rounds` | **2** | 5 |
| `compress_after_round` | **2** | 3 |
| `compress_keep_last` | **1** | 5 (`== max_rounds`) |
| `max_tool_result_chars` | **2000** | 6000 |
| `max_web_searches` | 4 | (bigger) |

**This confirms the diagnosis exactly, and the fix already exists in-repo.** With
`deep_worker`'s `max_rounds=2` + `compress_keep_last=1`, the forced-final-answer
compaction at `react.py:297` runs `_compress_history(keep_last=1)` — replacing
ALL but the single most-recent tool message with the contentless
*"[history compacted…]"* stub. deepseek's round-0 web_search results (the useful
ones) got stubbed before it wrote its answer, so it narrated the stub:
*"compacted out of context."*

This is the SAME failure the `research_worker` config comment documents across
five separate eval regressions (current-2025-recent 2026-06-30, parallel-postulate
run-3, pythagoras/archimedes/library/current-2025 run-6). The terminal fix landed
there was **`compress_keep_last == max_rounds`** (keep every round verbatim so no
search result is ever stubbed before the worker writes). Deep mode never received
that fix — it's still at the `keep_last=1` the comments call broken.

### Step 2 — Fix Defect A: apply the proven research_worker pattern to deep_worker (config, live)

Not a guess — port the known-good pattern. Proposed `agentic.react.deep_worker`:

```yaml
    deep_worker:
      max_rounds: 3                 # was 2 — one more round to fetch-then-use
      compress_after_round: 3       # was 2 — don't stub mid-run at 3 rounds
      max_tool_result_chars: 4000   # was 2000 — the +10 Euclid post-mortem tied
                                    # 2000 to "rushed answers from truncated
                                    # snippets"; research_worker uses 6000
      compress_keep_last: 3         # was 1 — == max_rounds, so NO search round is
                                    # ever replaced by the contentless stub
                                    # (the terminal research_worker fix)
      max_web_searches: 4           # unchanged — Brave quota × panel width
```

Rationale for the specific numbers: mirror research_worker's *shape*
(`keep_last == max_rounds`, later `compress_after_round`, bigger char cap) without
going all the way to its 5-round thoroughness budget — deep mode is the faster,
lighter tier and stays that way. `keep_last == max_rounds` is the load-bearing
change; the others prevent the tight budget from re-truncating.

Config edit only, no rebuild. Requires `up -d --force-recreate audrey-ai` (config
bind-mount stale-handle gotcha). Trade-off: bigger surviving context = more prompt
tokens/worker = slower + more VRAM/round. Deep mode's whole point is being lighter
than research, so this is a deliberate middle setting, not research parity.

**Open decision for the user:** how far to move deep mode toward research's budget.
The table above is the recommended middle. A more conservative option changes ONLY
`compress_keep_last: 1 → 2` (the single highest-leverage line) and leaves the rest;
a more aggressive option matches research_worker outright (loses the deep/research
tier distinction). Recommend the middle.

### Step 3 — Fix Defect B: give SYNTH_SYSTEM a plain-by-default clause (code + eval)

Port research mode's anti-over-hedge guidance into `SYNTH_SYSTEM` (`prompts.py:81`):
state facts confident from general knowledge plainly; reserve softening for
specific checkable claims a grounded draft failed to verify or that drafts
disagreed on. Preserve the existing FACTUAL ANCHORING rule (don't trust a lone
tool-free claim). Code change → wants a deep-protocol eval run
(`scripts/run_all_evals.sh`, or `eval_research.py` against the deep cases) to
confirm it doesn't overshoot into overconfident assertion.

### Step 4 — Confirm grounding source health (box, parallel with 1–2)

Rule out empty-throttle: from inside custom-tools, probe SearXNG/Brave for a live
World-Cup-style query and confirm non-empty results. (The ✅ footer counts
`is_error` only; an empty 200 shows as ✅.)

### Step 5 (optional, larger) — Routing

If recency questions should reliably ground, consider routing them to
`audrey_research`, or teaching `audrey_auto` to escalate recency-sensitive prompts
to the research pipeline rather than deep. Bigger change; defer until 1–4 land and
we see whether tuned deep mode is enough.

## Verification

- After Step 2: re-ask the World Cup prompt (or the closest eval case) in deep
  mode; the debug drafts should show a worker's web_search chars SURVIVING to the
  answer (no "compacted out of context" self-narration), and the synthesized
  answer should carry actual retrieved specifics.
- After Step 3: deep-protocol eval run diffed against the prior answers file —
  confident on known facts, honest-but-not-blanket-hedged on the unverifiable.
- Watch for overcorrection: deep mode asserting unverified specifics. The eval
  read is the gate.

## Status log

- **2026-07-18** — Plan created. Root cause traced to `react.py` compaction +
  web_search budget evicting fetched grounding (Defect A) and `SYNTH_SYSTEM`
  lacking research mode's anti-over-hedge clause (Defect B). Blocked on reading
  live `config.yaml` deep_worker values (Step 1, box-only) before tuning.
- **2026-07-18** — Step 1 DONE (user ran the greps). Live values confirm the
  diagnosis: `deep_worker` runs `compress_keep_last=1` + `max_rounds=2`, so the
  forced-final compaction stubs round-0 search results before the answer. This is
  the exact failure `research_worker` was hardened against with
  `compress_keep_last == max_rounds`. Step 2 upgraded from "candidate direction"
  to a concrete config edit (port the research_worker pattern, middle setting).
- **2026-07-18** — User chose the MIDDLE setting + BOTH defects. Applied in the
  working tree:
  - **Defect A** — `config.yaml` `agentic.react.deep_worker`: `max_rounds` 2→3,
    `compress_after_round` 2→3, `max_tool_result_chars` 2000→4000,
    `compress_keep_last` 1→3 (== max_rounds). Added an explaining comment.
  - **Defect B** — `src/audrey/pipeline/prompts.py` `SYNTH_SYSTEM`: inserted a
    "PLAIN BY DEFAULT" bullet after FACTUAL ANCHORING — state confident facts
    plainly, hedge only a lone-tool-free claim / a draft conflict / an
    unverified specific. Updated the `test_prompts.py` snapshot to match.
  - Gates: 763/763 pytest pass, ruff clean, cite-checker hard-DRIFT 0 (fixed two
    mechanical line-shift cites in lesson-07 and lesson-13 caused by the added
    lines; the 10 remaining `drift?` are advisory and pre-existing).

  **Working-tree changes complete; not deployed.** Next: user deploys with
  `up -d --force-recreate audrey-ai` (config bind-mount stale-handle gotcha —
  a plain restart won't pick up the config change), then run the deep-protocol
  eval and re-ask the World Cup prompt to confirm the debug drafts show
  web_search chars SURVIVING to the answer (no "compacted out of context"
  narration) and the synth reads confident-but-honest, not blanket-hedged.
- **2026-07-21 07:56** — Pre-fix baseline run
  (`2026-07-21-075615-deep-baseline-onbox-answers.md`, 18/18 structural PASS).
  Established the fingerprint of the OLD config: every `web_search→ctx` value
  was an exact multiple of 2000 (8000/6000/6000/4000/4000 — 6000 appears three
  times and is not reachable under a 4000 cap), and NO worker anywhere in 18
  cases reported more than 2 tool rounds. Also confirmed the structural checks
  can't grade this: a worker that answers "I couldn't retrieve anything" still
  scores `has_answer:✅ banners:✅ no_error_marker:✅`. The signal lives only in
  the panel debug drafts.
- **2026-07-21 08:26** — Post-fix run
  (`2026-07-21-082651-deep-baseline-onbox-answers.md`, 18/18 structural PASS).
  **Both fixes confirmed live. Defect A is closed.**
  - **Proof the config is active:** workers now report **3 tool rounds**
    (library-alexandria ×2, pythagoras, rust-async, 2025-recent ×3). Three
    rounds is unreachable under the old `max_rounds: 2` — this cannot be a
    stale deploy.
  - **Proof the eviction window is closed:** zero occurrences of "compacted out
    of context" / "tool-call budget has been reached" narration anywhere in the
    run. That phrasing was `_summarize_tool_message` / `_WEB_SEARCH_BUDGET_STUB`
    leaking into worker prose; it is gone.
  - **Truncation fingerprint changed:** `ctx` values are now arbitrary
    (788 / 8489 / 5313 / 11516 / 4230 / 4369 / 2577 / 1178 / 1759 / 209 / 85 /
    1913 / 0), no longer quantized to the old 2000-char cap.
  - **Defect B behaving as designed:** synth answers are confident by default.
    The one case that still hedges (`deep-2025-recent`) hedges *in scope* — it
    states the DeepSeek-R1 / Mistral Small 3 / Qwen 2.5-Max material plainly,
    then isolates a "What I Couldn't Verify" section for mid-to-late 2025. That
    is exactly the PLAIN BY DEFAULT contract ("say so plainly for THAT detail
    … but keep the rest of the answer confident"), not the blanket hedging the
    World Cup answer showed.
  - **NEW, INDEPENDENT defect surfaced (Defect C): web_search grounding is thin
    or empty.** Closing the eviction window revealed that some of what was being
    evicted was empty to begin with:
    - `deep-2025-recent` — all three workers ran the full 3 rounds
      (4369 / 2577 / 1178 chars) and all three independently reported the
      results were unusable. This is the case most exposed to it: the query
      needs 2025 coverage and every panel model's training data stops before it.
    - `deep-library-alexandria` — kimi burned 3 rounds for **788 chars** and
      bailed entirely ("None of the search results returned any material").
      deepseek got 8489 chars and still opened with "the web search results …
      were not returned."
    This is the known SearXNG signature (`200 + 0 results` = upstream engines
    throttling the instance), not an Audrey code path, and the ✅/❌ footer counts
    `is_error` only so empties render as ✅. Defect C is a grounding-source
    problem and is tracked separately from this plan — do NOT tune the ReAct
    budget further in response to it; more rounds against an empty index just
    costs latency (see `deep-2025-recent`: 3 wasted rounds per worker).
- **2026-07-21 (analysis) — CORRECTION to the entry above, and a sharper read.**
  - **RETRACTED:** "three workers logged `web_search→ctx: 0 chars` … the call
    succeeded and returned nothing." That was wrong. `banners.py:203-204` emits
    the `web_search→ctx:` label whenever `tool_rounds > 0`, *whether or not the
    worker ever called web_search*, and `react.py:274-275` only accumulates
    `web_search_chars` for `name == "web_search"`. All three `ctx: 0` workers
    have tools-used footers containing **no web_search at all**
    (synthesis-tradeoff qwen = `memory_search`+`chat_history_search`;
    contested-recommendation qwen = `chat_history_search`; code-lru-cache
    deepseek = `memory_store` ❌1). `ctx: 0` means **"never searched"**, not
    "searched and got nothing". *Reading note: `ctx` is only interpretable when
    paired with the web_search call count from the tools-used footer.*
  - **The real empty-result signature.** `dispatch.py:185` sets content to
    `json.dumps(payload)`, so a zero-result web_search is a ~63–73 char envelope
    (`{"query": "…", "results": []}`) — measured, not estimated. Dividing ctx by
    the footer's web_search count gives per-search chars:

    | case | worker | searches | ctx | chars/search | read |
    |---|---|---|---|---|---|
    | rust-async | qwen3.6:35b | 1 | 85 | 85 | **empty** |
    | rust-async | kimi-k2.6 | 2 | 209 | ~104 | **both empty** |
    | 2025-recent | qwen3.6:35b | 4 | 1178 | ~294 | near-empty |
    | 2025-recent | kimi-k2.6 | 4 | 2577 | ~644 | thin |
    | library-alexandria | kimi-k2.6 | 2 | 788 | ~394 | thin |
    | 2025-recent | deepseek | 4 | 4369 | ~1092 | real |
    | rust-async | deepseek | 4 | 4230 | ~1057 | real |
    | pythagoras | deepseek | 4 | 11516 | ~2879 | real |

  - **SearXNG is not down.** In `deep-rust-async`, concurrently: qwen's 1 search
    and kimi's 2 searches came back empty while deepseek's 4 came back full.
    Same instance, same moment, different queries → the failure is **per
    request**, not a dead dependency. (This also means the fast-triage "is
    SearXNG up?" probe will say "healthy" and prove nothing.)
  - **Why one empty is fatal rather than retried.** `app.py:241-248`
    `_prefer_searxng` blake2b-hash-splits queries ~50/50 between Brave-primary
    and SearXNG-primary, with cross-fallback on error *or* empty. But with Brave
    402'd (`brave.py:106-108` → `BraveQuotaError`), **both paths terminate at
    SearXNG**: a Brave-primary query fails over to SearXNG, and a SearXNG-primary
    empty calls Brave, gets `BraveQuotaError`, `_try_other` returns `None`, and
    `app.py:338` `return result or hits` hands back the empty. The two-provider
    design is currently decorative — the entire defense is the single 1.5s
    empty-retry at `searxng.py:140-142`.
  - **Hypothesis (untested): our Defect A fix increased the burst.**
    `max_rounds` 2→3 with `max_web_searches` left at 4 means workers now
    actually *reach* 4 searches; in the pre-fix baseline no worker exceeded 2
    rounds and every ctx was truncation-capped (≥2000 chars/search — zero
    empties). 3 workers × up to 4 searches ≈ 12 searches at one keyless
    instance, and there is **no concurrency limit anywhere in tools-server**
    (verified: no `Semaphore`, no httpx `limits=`). Confounded by time-of-day —
    needs the probe below before acting.
  - **Next step (decisive, read-only, box):** run
    `scratchpad/searxng_burst_probe.py` (sequential 12 vs concurrent 12) to
    separate burst-induced throttle from baseline degradation, and grep
    `custom-tools` logs for the eval window — `app.py:336` already logs
    `"returned 0 results; trying …"` and `:321` logs `"unavailable (…)"`, so the
    logs can confirm Brave's 402 and count the empties directly.
    **Do not add retries or tune the ReAct budget until the probe reads out** —
    the two candidate fixes (rate-limit/stagger vs. renew the Brave key) are
    opposites, and the probe picks between them.
- **2026-07-21 — DEFECT C ROOT-CAUSED AND FIXED (engine layer). Both of my
  hypotheses above were wrong; the burst one is explicitly disproven.**
  Full detail now lives in
  [`docs/guides/searxng-unraid-setup.md`](../guides/searxng-unraid-setup.md)
  ("2026-07-21 — engines also quota-drop mid-session"); summary here:
  - **Measured:** 16 of 57 `web_search` calls over 6h returned 0 results (28%),
    with a further 12% returning 1–3. Brave logged **zero** successful calls —
    every path 402'd, so SearXNG was carrying 100% of grounding alone.
  - **Cause:** `mojeek` silently stops contributing after ~6 queries *without*
    appearing in `unresponsive_engines`. Two back-to-back sequential passes over
    six queries showed five of six dropping by **exactly 10** results (mojeek's
    per-query contribution); the query whose only contributor was mojeek went
    **10 → 0**. Not a dead engine — a quota-dropout with no error surface.
  - **RETRACTED — burst/concurrency:** a 12-query burst had the same empty rate
    as sequential (17% each). These engines rate-limit per *request window*, not
    per simultaneity. So the `asyncio.Semaphore` in the tools-server web_search
    handler I proposed would have added latency and fixed nothing, and capping
    searches-per-round would only have helped by cutting volume (~14%), not by
    cutting concurrency as I argued. **Nothing was built — the probe ran first.**
  - **RETRACTED — `safesearch: 1`:** A/B'd against the same queries; the runs
    *with* safesearch returned more results, not fewer. `searxng.py:108` is fine.
  - **Fix applied (config, box-only — SearXNG `settings.yml` is not tracked):**
    added `bing` + `yandex` (each answered every probe query) + `wiby` +
    `encyclosearch`; disabled `seznam` (timeout — stalls every search),
    `crowdview` (silent zero on all probes), `wikidata` (persistent access
    denied). Probe queries went **10/23/40 → 32/54/92 results**, and the
    acceptance-test query (`tokio vs smol vs glommio`, the one that zeroed) went
    **0 → 32**. Worst-case surviving engines: 1 → 3.
  - **Still open:** the Brave key is still 402'd. It remains worth renewing — it
    is a separate quota pool that does not share SearXNG's per-engine rate
    limits, so it degrades independently, and it is the only source of
    major-index result *quality* (the pre-fix junk — cram.com, able2know.org —
    came from grounding on three small independent crawlers).
  - **Next:** re-run the deep eval now that grounding is healthy, and re-check
    `deep-2025-recent` / `deep-library-alexandria`, which are the two cases whose
    workers reported unusable search results.
- **2026-07-21 — third run (`…-093005-deep-postfix…`) IS post-engine-fix.**
  - **THE BOX IS PDT (UTC−7).** Verified with `date; date -u`, not inferred.
    An earlier revision of this entry called run 3 "pre-engine-fix" by assuming
    UTC−6 from a git-commit offset; that was wrong and is retracted. **Always run
    `date` on the box before reasoning about a timestamp** — `eval-onbox.sh:65`
    stamps filenames from host `date`, `docker inspect` reports UTC with a `Z`,
    and mixing them silently inverts conclusions.

    | event | UTC | PDT (host) |
    |---|---|---|
    | SearXNG restart (new engines live) | 16:29:21 | **09:29:21** |
    | run 3 start (filename stamp) | 16:30:05 | **09:30:05** |
    | run 3 finish (file mtime) | 16:52 | **09:52** |

    Run 3 began 44 s after the restart and ran entirely inside the new engine set.
  - **The engine fix, not variance, explains the jump.** Run 2 (08:26–08:52,
    pre-fix) vs run 3 (09:30–09:52, post-fix), identical code config:

    | case / worker | run 2 (pre-engines) | run 3 (post-engines) |
    |---|---|---|
    | rust-async / kimi | 209 (empty) | **12,433** |
    | rust-async / deepseek | 4,230 | 12,158 |
    | 2025-recent / deepseek | 4,369 | 12,447 |
    | 2025-recent / qwen | 1,178 | 9,399 |
    | alexandria / kimi | 788 | 6,072 |

    `deep-2025-recent` went from "couldn't verify anything past Q1 2025" to
    precise dated specs (R1 2025-01-20, V3-0324, R1-0528, Llama 4 Scout/Maverick
    expert counts, Qwen3 Apache 2.0). rust-async/kimi — the acceptance-test topic
    — went from an empty envelope to 12.4 k chars.
  - **VERIFIED: zero-result rate 35% → 0%.** Measured per-run, after resolving
    the clock offset (custom-tools logs are MDT, filenames are host PDT — logs
    run 1 h AHEAD of filenames; see [[project-three-clocks-gotcha]]):

    | | run 2 (pre-engines) | run 3 (post-engines) |
    |---|---|---|
    | container-log window | 09:26–09:52 | 10:30–10:52 |
    | web_search calls (tools-used footers) | 34 | 47 |
    | **empty results** | **12 → 35%** | **0 → 0%** |
    | result-count spread | 5s/2s/1s + 12 zeros | 13×10, 12×5, 1×8 |

    Every one of run 3's 47 searches returned results, and 25 of the 26 logged
    were **count-capped** (model asked 5 or 10, got 5 or 10). **Defect C is
    closed.** This also settles the open question: bing/yandex did NOT
    quota-drop across a full 47-search eval — the enlarged pool holds under real
    load, so no volume reduction is required for correctness.
    *Measurement note:* zeros are ALWAYS logged (`app.py:336` primary-empty,
    `app.py:269` fallback), but SearXNG-primary successes log nothing — so take
    the denominator from the eval file's tools-used footers, never from the log.
  - **NEW — Defect D: a worker can hold ample grounding and still refuse.**
    `deep-pythagoras`/deepseek — 3 rounds, `web_search` ✅4, **13,602 chars** —
    wrote "The web searches I attempted … all hit a tool-call budget limit before
    returning any results." `deep-compare-crises`/deepseek did the same with
    6,818 chars. Both had exhausted `max_web_searches` and hit
    `_WEB_SEARCH_BUDGET_STUB` (`react.py:96`), then narrated the stub as total
    failure. Notable because that stub was **already hardened for exactly this**
    (`react.py:91-95`: nothing may read as a failed search) and even instructs
    "Answer from the evidence already gathered" — deepseek ignored both. So this
    is not a rewording bug; it is a model-behaviour failure that survived a
    deliberate mitigation. **Panel redundancy absorbed both** (kimi covered
    Pythagoras, kimi+qwen covered crises; both syntheses are among the run's
    best), which is why it shows up as a footnote rather than two failed cases —
    and is itself evidence the multi-worker design earns its cost.
  - **Not yet re-tested:** the original trigger prompt. There is no World Cup /
    current-sports case in `scripts/eval_prompts_deep.json` (16 `deep-*` cases +
    2 borrowed `code-*`); `deep-2025-recent` is the closest recency analog. A
    direct OWUI re-ask in `audrey_deep` is still the only way to close the loop
    on the reported symptom — and it will be gated by Defect C, not Defect A.

## Deploy + verify (box, for the user)

```bash
cd /mnt/user/appdata/audrey_ai_2.0 && git pull
# --build is REQUIRED, not just --force-recreate: config.yaml is bind-mounted (ro)
# so it needs a recreate, but prompts.py (SYNTH_SYSTEM) is BAKED INTO THE IMAGE
# (docker/audrey.Dockerfile does `uv pip install --system /app`; there is no source
# bind-mount). A force-recreate alone would deploy the config fix with the OLD
# prompt — you'd test a half-applied change and misread the result.
docker compose up -d --build --force-recreate audrey-ai
docker compose logs -f audrey-ai                     # confirm clean boot

# Then, from the laptop over LAN/VPN — deep-protocol eval (background; ~20-40 min):
.venv/bin/python scripts/eval_research.py \
    --cases scripts/eval_prompts_deep.json \
    --save-file docs/testing/$(date +%F)-deep-hedge-answers.md

# And a direct re-ask of the trigger prompt in deep mode (OWUI, audrey_deep):
#   "what were the key moments of this year's World Cup?"
# Watch the panel debug drafts: a worker's web_search chars should now survive
# to its final answer (no "compacted out of context" self-narration).
```

Diff the new `-deep-hedge-answers.md` against the prior deep answers file. Watch
for OVERCORRECTION (deep mode now asserting unverified specifics confidently) —
that's the failure mode of Defect B's fix, and the eval read is the gate. If it
overshoots, dial the PLAIN BY DEFAULT bullet back toward caution.
