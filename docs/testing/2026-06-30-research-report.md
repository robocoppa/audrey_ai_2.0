# Eval report — 2026-06-30 research protocol (post run_all_evals suite)

Paired with [`2026-06-30-research-answers.md`](2026-06-30-research-answers.md)
(the raw outputs). Run: `scripts/eval_research.py --model audrey_research --cases
eval_prompts_protocol.json` against the box over the VPN. **10/10 structural
PASS, exit 0.** This is the first run after building out the suite docs +
`run_all_evals.sh`; the harness itself is unchanged.

## Headline

Clean structural pass and the strong cases are excellent — well-grounded,
correctly hedged, 7–8 authoritative sources each. But **two cases surfaced a new
failure mode that I initially misread as a SearXNG content problem.** It is not.
`current-2025-recent` narrates *"several searches returned sparse or elided
results"* — and "elided" is a **literal string our own ReAct history-compaction
emits** ([`react.py:66`](../../src/audrey/pipeline/react.py#L66)), not anything
SearXNG returns. The model is reading its own compacted scratchpad and reporting
it as a search failure. Separately, `bio-archimedes` dropped to a single grounded
worker. Both are **our-side behaviors, not upstream throttle** — distinct from
the empty-result issue we shipped a fix for earlier today.

## What the structural checks confirmed (10/10 PASS)

Every case: reachable, no error marker, answer present, banners in order
(Planning → Researching → Verifying → Fact-checking → Writing).

## Latency

| Case | total | TTFT |
|---|---|---|
| ctrl-birthday-toast | 63.5s | 13.6s |
| ctrl-explain-recursion | 86.2s | 15.0s |
| tech-transformer-attention | 204.7s | 0.5s |
| current-rust-async | 250.5s | 12.9s |
| current-2025-recent | 279.4s | 17.2s |
| bio-pythagoras | 386.7s | 13.4s |
| hist-library-alexandria | 431.7s | 15.1s |
| bio-euclid | 453.4s | 8.4s |
| hist-parallel-postulate | 486.0s | 0.5s |
| **bio-archimedes** | **667.3s** | 13.4s |

The slow cases are the deep biographies/histories where all three workers run a
full search loop. `bio-archimedes` is both the slowest **and** the
single-worker-grounded case — consistent with two workers spending their whole
budget and contributing nothing usable (see Finding 2).

## Quality read — the strong cases

- **`bio-euclid`, `bio-pythagoras`, `hist-library-alexandria`,
  `hist-parallel-postulate`** are the suite at its best. All three workers
  contributed, each rendered a deterministic authority-ranked **Sources** block
  (7–8 real sources: Stanford Encyclopedia, IEP, MacTutor, Britannica,
  world/ancient-history references), and the **hedge discipline is exactly
  right** — disputed/legendary claims marked "reportedly"/"likely
  apocryphal"/"disputed by historians," firmly attested facts stated plainly.
  `hist-parallel-postulate` in particular is a genuinely good piece of writing.
- **Control suppression still works:** `ctrl-birthday-toast` and
  `ctrl-explain-recursion` correctly got **no Sources block and no hedging** —
  the ungrounded ⇒ no-Sources, no-dispositions path. Both are fast (63s / 86s).

## Finding 1 — "elided" is our compaction string, not a search failure

**Case:** `current-2025-recent` (and, more mildly, `tech-transformer-attention`).

**Symptom:** the answer opens *"I couldn't fully verify these details … several
searches returned sparse or elided results,"* renders **no Sources block**, and
hedges every 2025 fact with "reportedly" — *despite the footer showing 34
successful searches across the three workers* (`✅10 ❌2`, `✅13`, `✅11 ❌1`). On
the earlier same-day empty-retry validation run this exact prompt produced a
confident, dated answer with 3 real Sources. So this is a **regression in
narration quality**, and the worker-reported reason is "elided results."

**Root cause (our side):** "elided" is not a word SearXNG or the dispatch layer
produces. It comes from ReAct history-compaction:

```
src/audrey/pipeline/react.py:66
    return f"[earlier tool call: {name} -> {len(content)} chars elided]"
```

`_compress_history` ([`react.py:69`](../../src/audrey/pipeline/react.py#L69))
replaces older `role=tool` messages with that one-line stub once a worker passes
`compress_after_round`, keeping only the last `compress_keep_last` tool results
verbatim. For research workers the budget is deliberately large
([`config.yaml`](../../config.yaml) `research_worker`): **`max_rounds: 5`,
`compress_after_round: 3`, `compress_keep_last: 1`.** So a worker that searches
across rounds 1–4 has, by the time it writes its findings, seen its **own earlier
`web_search` results collapsed to `…chars elided`**, keeping only the single most
recent round. A heavy-search prompt like the 2025-recency case (the most
search-dependent in the protocol) reliably trips this. The model then faithfully
narrates that its searches were "elided" — it is describing its compacted
scratchpad, not the upstream results.

**Why this matters:** it is a *self-inflicted, prompt-shaped* degradation that
looks identical to a grounding failure in the answer, and it leaks the internal
word "elided" to the reader. It is independent of the SearXNG throttle (the
searches succeeded and returned content; we compacted it away before the worker
finished reasoning). The empty-result retry we shipped today cannot touch it —
the results weren't empty.

**Correction to my first read:** I initially attributed this to SearXNG returning
"thin/elided snippets." That was wrong. SearXNG has no part in it; the string is
ours.

**Candidate fixes (not yet applied — flagging for decision):**
1. **Reword the compaction stub** so it can't be mistaken for (or narrated as) a
   tool result — e.g. `[history compacted: {name} result omitted to save context]`
   with no leading "[earlier tool call". Cheapest; stops the word "elided"
   leaking and makes a model that quotes it sound less like a failure. Hermetic
   test in `tests/test_react.py`.
2. **Raise `compress_keep_last` for `research_worker`** (1 → 2 or 3) so a
   multi-round researcher keeps more of its own evidence verbatim before writing.
   Costs context budget; research mode is thoroughness-first so likely
   acceptable. Needs a box re-run to confirm it recovers the 2025 case.
3. **Both** — reword the stub *and* keep more rounds. Recommended.

## Finding 2 — `bio-archimedes` grounded on a single worker

**Symptom:** the answer opens with the "couldn't be fully verified" hedge and has
**no Sources block**; the footer lists only **`glm-5.2:cloud`** (the other two
workers, deepseek and qwen, contributed nothing that survived). The other two
biographies had all three workers and 7–8 sources. It was also the slowest case
(667s).

**Where it drops (our side, fail-soft):** two filters in the ledger-build path
silently exclude a worker:
- [`deep_panel.py:1214`](../../src/audrey/pipeline/deep_panel.py#L1214) — a draft
  with empty/whitespace `content` is skipped before structuring.
- [`deep_panel.py:1218`](../../src/audrey/pipeline/deep_panel.py#L1218) — a draft
  whose structuring returned `None` (unparseable ledger) is dropped from `usable`.

Both are deliberately fail-soft (a bad worker must not break the answer), so the
result is a quietly thinner answer rather than an error. The
`"ledger built — %d claims, %d sources from %d/%d workers"` log line
([`deep_panel.py:1222`](../../src/audrey/pipeline/deep_panel.py#L1222)) records
exactly which it was — **needs a box-log check to confirm** whether the two
missing workers returned empty content or an unparseable ledger.

**Why this is plausibly the same family as past drops:** we've hardened ledger
parse-tolerance before (null/off-enum fields, optional ids, bare top-level
arrays). A single-worker-survived case on an *easy* biography is the signature of
another tolerated-but-dropped shape, or of two workers timing out within the
667s. Can't confirm from the laptop — the box logs have the `%d/%d workers` line.

**Action:** check the box log for that run's `research: ledger built — … from
1/3 workers` line for the archimedes task; if the drop is a parse `None`, capture
the raw reply (per the AGENTS.md "log the raw reply FIRST" rule) and add it to
the ledger regression shapes. If it's a worker timeout, that's a latency story,
not a parse one.

## Net

- **No regression in the shipped deterministic-shaping work** — Sources ranking,
  hedge discipline, and control suppression all behave correctly on the cases
  that grounded fully.
- **Two our-side issues, both previously hidden:** the ReAct compaction stub
  narrating as a search failure (Finding 1, reproducible and prompt-shaped), and
  the single-worker ledger drop on `bio-archimedes` (Finding 2, needs a box-log
  confirm). Neither is the SearXNG throttle.
- **The throttle is still the operational backdrop** (the `❌` counts on the
  current-* cases), but it is *not* what produced the two soft answers this run —
  that was us. Brave-key renewal remains the durable grounding cure; it does not
  fix Finding 1 or 2.

## Follow-ups

**Done in the working tree (hermetic, 637 pytests green, ruff clean):**

1. **Reworded the compaction stub** — [`react.py:63`](../../src/audrey/pipeline/react.py#L63)
   `_summarize_tool_message` now emits `[history compacted: an earlier
   \`<tool>\` result is omitted here to save context]` (was `[earlier tool call:
   … N chars elided]`). No count to misread as "thin," and it can't read as a
   failed/empty search. Pinned by `test_summarize_tool_message_reads_as_compaction_not_failure`
   (asserts the wording carries none of elided/error/failed/empty/sparse).
   *(Finding 1, the leak.)*
2. **Bumped `research_worker.compress_keep_last` 1 → 2** —
   [`config.yaml`](../../config.yaml). A 5-round researcher now keeps its two
   most recent search rounds verbatim before writing, so it has real evidence to
   ground on instead of narrating the stub. *(Finding 1, the cause.)*
3. **Worker-drop warning** — [`deep_panel.py:1190`](../../src/audrey/pipeline/deep_panel.py#L1190)
   now logs `research: N/M researchers produced content; dropped: <model>
   (err=…, …s)` whenever a researcher returns empty content. The next
   archimedes-style drop is a one-line grep, not a reconstruction. *(Finding 2,
   observability — the part fixable without box access.)*

**Still needs the box (can't be done from the laptop):**

4. **Re-run `current-2025-recent`** after deploy to confirm Findings 1's two
   fixes recover the confident, sourced answer (the empty-retry run got it; this
   run lost it to compaction). Single-case: `scripts/eval_research.py --only
   2025-recent`.
5. **Root-cause `bio-archimedes`** from the new warning line: was it an
   OllamaError/timeout (err=… populated) or a genuinely empty reply (err=empty)?
   If the worker produced prose but its *ledger structuring* returned `None`,
   that's a parse drop — capture the raw reply per the AGENTS.md "log the raw
   reply FIRST" rule and add the shape to the ledger regression tests. *(The
   warning narrows it to one of these without guessing.)*
