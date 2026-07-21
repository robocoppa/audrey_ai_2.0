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

## Deploy + verify (box, for the user)

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --force-recreate audrey-ai      # picks up config.yaml + prompt change
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
