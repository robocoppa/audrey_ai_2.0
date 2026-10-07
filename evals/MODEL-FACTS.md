# Audrey model evaluation report and evidence ledger

**Last updated:** October 6, 2026

This document records measured model behavior through Audrey and attributed
client assessments. The public report below summarizes decisions that affect
the product. The detailed ledger
that follows preserves the evidence, limitations, and unresolved questions used
to reach those decisions. Vendor specifications are identified separately from
Audrey measurements.

## Public report: local decision models for Audrey routing

### Executive summary

Audrey currently uses `qwen3.5:4b` for task routing. Five purpose-built local
decision models or variants have been evaluated as possible replacements. Full
Clef produced the strongest raw classification result, but its 55-second cold
start and 32.48 GB runtime footprint make it unsuitable for a latency-sensitive,
always-available router. Clef Flash approached the incumbent's warm latency but
failed multiple requests while loading. The smaller Tev and Nimble candidates
were fast enough but introduced routing errors or additional deep-panel
escalations. Audrey therefore retains `qwen3.5:4b`.

### Question and method

The primary question was whether a typed decision model could replace Audrey's
generative JSON router while preserving routing quality, latency, reliability,
and GPU headroom. The tracked fixture contains 36 balanced prompts: 12 code,
12 reasoning, and 12 general. Audrey's deterministic keyword and short-prompt
gates resolve 13 prompts before model inference, leaving 23 prompts that
represent the production router's actual work.

The September comparison used one warm pass per model on Ollama 0.35.0. The
October Clef follow-up used three warm passes per model on Ollama 0.35.1.
Results with different repeat counts are reported with their actual
denominators rather than pooled. Cold requests deliberately begin with the
model unloaded. Residency is sequentially observed through Ollama and does not
claim concurrent workload performance.

### Routing results

| Model | Production-reached result | Warm p50 | Cold request | Observed residency | Decision |
|---|---:|---:|---:|---:|---|
| `tev1:0.8b` | 13/23 | 0.089s | 4.98s | 0.89 GB | Rejected: ten costly false-reasoning routes |
| `tev1:latest` | 18/23 | 0.169s | 5.43s | 4.67 GB | Rejected: three costly false-reasoning routes |
| `nimble:latest` | 22/23 | 0.200s | 15.21s | 8.97 GB | Rejected: one costly error and one added escalation |
| `clef:latest` | **69/69** | 0.341s | >20s; 55.13s in broad probe | 32.48 GB | Rejected: cold latency and footprint |
| `clef-flash:latest` | 67/69 | 0.204s | >20s | 12.78 GB | Rejected: two reached-case load timeouts |
| `qwen3.5:4b` | **23/23 and 69/69** | 0.184s and 0.197s | 11.06s and 10.29s | 4.20 GB | **Retained** |

Full Clef matched the incumbent on every production-reached sample and made no
costly false-reasoning error. Its observed residency was 7.7 times the
incumbent's. Combined with the 19.64 GB `qwen3.8:32k` residency observed in the
September run, it would require about 52.1 GB, beyond the host's 48 GB GPU
capacity. Clef Flash returned correct decisions once responsive, but its cold
request and the next four sequential requests each reached Audrey's 20-second
timeout.

### Broader Clef decision benchmark

Clef was also measured on 12 cases containing 29 typed questions across text,
structured JSON, and a generated image. Three warm rounds produced 36 requests
and 87 scored answers.

| Measure | Result |
|---|---:|
| Valid requests | 36/36 |
| Correct questions | 84/87 |
| Exact cases | 33/36 |
| Choice | 30/33 |
| Yes/no | 36/36 |
| Score | 18/18 |
| Structured JSON exact cases | 9/9 |
| Image exact cases | 3/3 |
| Warm p50 / p95 | 0.381s / 0.457s |

The single repeated error labeled a legitimate password-change notification as
phishing in all three rounds. This result establishes strong typed-decision
performance over the tested cases. It does not measure conversational writing,
long-form reasoning, or general answer quality because System One is a decision
endpoint.

### Decision and retained router

`qwen3.5:4b` remains the production router, and no additional router swap is
scheduled. No smaller Qwen 3.8 parameter model is currently published on
Ollama's official Qwen 3.8 page; its published tags are 27B variants. The
related Qwen 3.5 family publishes `qwen3.5:2b` at 2.7 GB and
`qwen3.5:0.8b` at 1.0 GB, but Audrey has not measured either and does not need
a smaller replacement for the current 4B router. These remain documented
alternatives rather than active candidates.

Official model pages:

- [Qwen 3.8 on Ollama](https://ollama.com/library/qwen3.8)
- [Qwen 3.5 on Ollama](https://ollama.com/library/qwen3.5)

### Limitations and reproducibility

- The routing set is Audrey-specific and intentionally small; it is not a
  general benchmark or a claim about every classification workload.
- The September candidates have one warm pass, while the Clef candidates have
  three. Denominators are retained so repeat depth remains visible.
- Residency observations are sequential. Only arithmetic establishes that full
  Clef and the previously resident 32K Qwen model cannot fit together in 48 GB.
- The broad Clef suite contains one image and three structured states. It is a
  capability check, not comprehensive vision or structured-data coverage.
- Complete machine-readable reports: [September router comparison](results/2026-09-30-systemone-router-probe-results.json), [Clef router comparison](results/2026-10-03-clef-systemone-router-results.json), and [broad Clef decisions](results/2026-10-03-clef-systemone-decision-results.json).

---

## Evidence standards

A line goes in the per-model sections **only** if it is:

1. **Sourced** — a dated run artifact, a probe result, `ollama list`, or a
   config value, named inline as `[source, date]`; and
2. **Not a single unseeded draw** — see *How to read the numbers* below. One run
   of one case is an anecdote. It goes in "Not established".

Everything that feels true but does not clear that bar goes in
**[Not established](#not-established)** instead, with what it would take to
settle it. That section is the point of the file: the failure mode this ledger
exists to prevent is an impression hardening into a fact through repetition.

⚠️ When a fact is superseded, **replace it and date the replacement** — do not
append a contradiction and leave the reader to guess which is current.

---

## How to read the numbers

These make the difference between a comparison and a coincidence.

- ⚠️ **`evals/eval_research.py` sets no `seed` and no `temperature`.** Options
  are built from the request body only, so every case is **one unseeded draw
  from the model's default sampler**. Two runs disagreeing is ordinary variance.
  This is why suite numbers here are quoted with their repeat count, and why
  n=1 results are quarantined below rather than recorded as facts.
- ⚠️ **Case 1 of every run is a cold model load.** `nemotron-3.5-lightning`
  showed ttft 40–59s cold against 0.4s warm. Drop case 1 before reading latency.
- ⚠️ **`eval_compare.py` merges by `(case, model)`, last file wins.** Globbing
  two arms into one invocation silently drops one. Different models in one glob
  are safe; the same model twice is not.
- Repeats pool into a single cell as a pass rate (`⚠️ 3/5`), and the `flaky`
  column counts cases that neither always pass nor always fail.
- ⚠️ **Scores from before 2026-08-19 evening under-count.** Three checks were
  failing correct answers, and they hit whichever model wrote most plainly:
  `has_answer` had a flat 20-char floor that failed `2,140 ms` (right value,
  whole question); `_DISCLAIMS_ABSENCE` carried eighteen verbs but not
  `describe`, so "the notes do not describe X" read as a failure to disclaim;
  and `synth-absent-subtopic` forbade `back-pressure is handled`, which is the
  phrasing of the correct REFUSAL. Four of seven `gap-r5b` failures were these.
  Fixed and guarded by tests. ▶ **`gap-r5b` real scores: qwen3.8 60/60,
  muse-glimmer 60/60, nemotron 60/60, ornith-1.5:35b 59/60, glm-4.7-flash
  58/60.** `[gap-r5b, 2026-08-19]`
- ⚠️ **`docker logs audrey-ai | grep …` DOES NOT FILTER.** Python logging writes
  to **stderr**, `docker logs` keeps the streams separate, and the pipe only
  ever sees stdout — so the grep returns the whole log and looks like a broken
  filter. Two separate 2026-08-19 attempts to confirm a thinking arm this way
  returned unfiltered output and cost several rounds each. Always redirect
  first: `docker logs audrey-ai --since 3h 2>&1 | grep 'model=<name>'`.
- The thinking arm is per-request as of 2026-08-19 and is recorded in the
  results JSON as `think_requested`, so an arm can no longer be lost to a
  container recreate.

**Suite sizes:** `eval_prompts_models_ab.json` = 13 cases (general quality:
reasoning, world knowledge, science explanation, writing, code).
`eval_prompts_local_models.json` = 12 cases (grounding, synthesis over supplied
passages, instruction-following, code).

---

## Local models

### `muse-glimmer:latest`

- **122/125** across both suites, `--repeat 5`, thinking on. ab 62/65, gap
  **60/60** — the only clean sweep of any model on the grounding suite.
  `[ab/gap-think-r5, 2026-08-19]`
- ✅ **123/125 with thinking OFF** — ab 63/65, gap **60/60**. The only model
  that does not move when thinking is removed, and the highest thinking-off
  score of the five. Both failures are `reasoning-decimal-compare`
  (`9.11 > 9.9`). ⚠️ It is NOT the fast-path primary — qwen3.8 holds that at
  priority 100 against muse's 84. `[ab-off-r5b + gap-off-r5b, 2026-08-19]`
- ▶ **LEADS `deep_panel_local.general` AND `deep_panel_local.reasoning` from
  2026-08-25**, retaking `general` from ornith-1.5:35b. The two local pools are
  now identical. The trade is accuracy over cost — see the latency line below,
  which is what that costs and is the first thing to check if deep-local p95
  becomes a complaint. `[config.yaml]`
- ⚠️ **The slowest local by a wide margin.** Thinking off, warm, grounding
  suite: ttft mean 3.55s, total mean 5.71s, 7 of 59 cases over 8s. That is
  **2.5x qwen3.8 and 6.3x nemotron** on total. Its accuracy lead has to be paid
  for in latency. `[gap-off-r5b, 2026-08-19]`
- Slowest of the three measured: mean ttft 15.2s, mean total 20.5s, mean answer
  1094.6 chars (ab suite). `[ab-think-r5, 2026-08-19]`
- Wrote the explicit `sorted(key=lambda x: (-x[1], x[0]))` tie-break on
  `code-word-frequency` in **5/5** runs. `[ab-think-r5, 2026-08-19]`
- Failed `writing-eli5-rewrite` 3/5 with source jargon (`electromagnetic
  radiation`, `photolysis`) surviving into the rewrite body — genuine, not a
  check artifact. `[ab-think-r5, 2026-08-19]`
- Failed `reasoning-decimal-compare` 1/5 (`9.11 > 9.9`). `[ab-think-r5, 2026-08-19]`
- On `gk-nonexistent-paper` it named only **real** adjacent work when redirecting
  (Transformer-XL, Tensor2Tensor, T5). No invented citations in 5 runs — the
  only measured model of which that is true. `[ab-think-r5, 2026-08-19]`

### `glm-4.7-flash:q8_0` — ⛔ RETIRED 2026-08-19

Dropped as out of date at the user's call: removed from `model_registry.general`
(it held priority 80), `passthrough.allowed_models` and `scripts/ops/pull-models.sh`.
It was already out of every deep panel — pulled from `deep_panel_local.general`
earlier on a 240s `deep_worker` ReadTimeout. ⚠️ `ollama rm` it on the box; a tag
present but unlisted trips the inventory check to exit 1. The measurements below
are kept because two of the harness fixes shipped 2026-08-19 exist because of
them. ⚠️ Not to be confused with `glm-5.2:cloud`, which stays — that is
Claudette's gateway model.

- **115/125**. ab 60/65, gap 55/60. Mean ttft 11.5s, total 14.4s, 1297.4 chars
  (longest answers measured). `[ab/gap-think-r5, 2026-08-19]`
- ⚠️ **Answers that begin mid-document.** Three separate failures opened at a
  subheading (`### What it Changed about Architectures`, `### 1. The Main
  Findings…`) with the leading section simply absent, so the actual question
  went unanswered. Seen on `gk-nonexistent-paper` ×2 and
  `science-mrna-vaccines` ×1. `[ab-think-r5, 2026-08-19]`
- ⚠️ **Invents grounding.** `synth-absent-subtopic` 3/5: two runs asserted the
  pipeline handles back-pressure via "the protocol's built-in flow control"
  from a passage that says nothing about back-pressure. `[gap-think-r5, 2026-08-19]`
- Wrapped strict-JSON output in ```json fences 2/5 on `instruction-strict-json`.
  `[gap-think-r5, 2026-08-19]`
- Failed `instruction-negative-constraint` 1/5 by dodging so hard it never used
  the word "buffer". `[gap-think-r5, 2026-08-19]`
- Failed `reasoning-decimal-compare` 1/5. `[ab-think-r5, 2026-08-19]`
- Correct tie-break on `code-word-frequency` 5/5. `[ab-think-r5, 2026-08-19]`
- ⛔ **107/125 with thinking OFF** — ab 56/65, gap 51/60, the largest grounding
  drop of any model (−7). `instruction-strict-json` went **0/5**: fenced in
  ```json every time, with `replicas` rendered `3`, `"between two and five"`,
  `{"min":2,"max":5}` and `"between 2 and 5"` — never the required form, against
  2/5 fenced with thinking on. It also leaked raw scratchpad into answers
  ("Alright, let's tackle this puzzle step by step", all six permutations, a
  `### Final Answer` header). `[ab-off-r5b + gap-off-r5b, 2026-08-19]`

### `ornith:latest` — ⛔ REMOVED FROM THE BOX 2026-08-19

Deleted as out of date, superseded by `ornith-1.5`. The measurements below are
kept because they are the baseline `ornith-1.5` has to beat, and because two of
them are the reason it is being replaced. ⚠️ They can no longer be re-measured.

- **111/125** — lowest measured. ab 53/65, gap 58/60. `[ab/gap-think-r5, 2026-08-19]`
- **Fastest by a wide margin**: mean ttft 1.8s, total 3.8s — 4–8× faster than
  the other two. Mean answer 814.9 chars (shortest). `[ab-think-r5, 2026-08-19]`
- ⚠️ **Reaches for the terse stdlib call and misses the edge case.**
  `code-word-frequency` **1/5**: used `Counter.most_common(n)`, which does not
  tie-break alphabetically, returning `[('b',2),('a',2),('c',1)]` in 4 runs.
  Same shape on `code-rle-roundtrip` (4/5): `int(s[i+1])` assumes a single-digit
  count. `[ab/gap-think-r5, 2026-08-19]`
- ⚠️ **Fabricates confidently on history.** `gk-berlin-wall` 3/5: one run
  attributed the trigger to "Berlin Mayor **Walter Momper**" with an invented
  quote, stated flatly with no hedge — failed `contains` and `calibrated`
  together. `[ab-think-r5, 2026-08-19]`
- Omitted `ribosome` from `science-mrna-vaccines` 2/5. `[ab-think-r5, 2026-08-19]`
- Ignored an explicit output-format instruction 1/5 on `reasoning-race-order`
  (wrote `Cal (1st) > Ada (2nd) > Ben (3rd)` where the prompt demanded three
  bare names) — reasoning correct, instruction violated. `[gap-think-r5, 2026-08-19]`
- Only model to go **5/5** on `reasoning-decimal-compare`. `[ab-think-r5, 2026-08-19]`
- ✅ Its `science-attention` failure was a **harness** false positive, not a
  model defect (`not_misattributed` fired on the teaching phrase "you say"),
  fixed 2026-08-19. `[ab-think-r5, 2026-08-19]`

### `nemotron-3.5-lightning:latest`

- **120/125** — ab 60/65, gap **60/60**. `[ab/gap-next4, 2026-08-19]`
- ⛔ **115/125 with thinking OFF** — ab 60/65, gap 55/60. ▶ **It invents metrics
  when the fact is absent.** `ground-fact-absent` answered **`2322 ms`** on one
  draw and **`p99 latency: 1424 ms`** on another — numbers that appear nowhere
  in the passage, stated bare, no hedge, in a two-word reply, on a question
  whose only correct answer is that the passage does not say. It was 60/60 on
  this suite WITH thinking. ⛔ This disqualifies it from any thinking-off
  grounding role, including the fast path, on trust rather than accuracy.
  It also shipped literal placeholder code on `code-rle-roundtrip`
  (`s[result[-1] - result[-1] ...]  # placeholder to make syntax happy`) for a
  SyntaxError exit 1. `[gap-off-r5b, 2026-08-19]`
- ✅ **The fastest local measured.** Thinking off, warm, grounding suite: ttft
  mean **0.32s**, total mean **0.90s**, max 1.9s — 6.3x faster than muse and
  2.5x faster than qwen3.8 on total. Cold load 39.0s, the slowest cold of the
  three. ▶ On speed alone it is the strongest fast-path candidate on the box;
  the fabrication above is the only thing keeping it out.
  `[gap-off-r5b, 2026-08-19]`
- Fast: most cases ttft 2–4s, total 2–15s. `[ab-next4, 2026-08-19]`
- Correct alphabetical tie-break on `code-word-frequency` **5/5**, and clean on
  both gap code cases. `[ab/gap-next4, 2026-08-19]`
- ⚠️ `science-mrna-vaccines` **0/5** — every answer opens at a `###` heading
  partway in ("How mRNA Vaccines Differ…", "Traditional Attenuated Vaccines…")
  with the requested "how mRNA vaccines work" section absent. See
  [answers truncated to their conclusion](#answers-truncated-to-their-conclusion)
  — this may be the arm, not the model. `[ab-next4, 2026-08-19]`
- Volunteers calibration language unprompted ("Both the date and the
  press-conference miscommunication are well-documented historical facts; I'm
  not guessing"). `[ab-next4, 2026-08-19]`
- **What `ollama show` reports:** architecture `nemotron_h_moe`, a mixture of
  experts, 32.9B in total, embedding length 2,688, context length 1,048,576,
  Q4_K_M, requires Ollama 0.32.9. Capabilities completion, tools and thinking,
  with thinking levels `false`/`true`/`medium` and default `true`. Parameters
  `top_p 0.95`, `draft_num_predict 2`, `temperature 1`. Licence: NVIDIA Open Model
  License Agreement, last modified October 24, 2025. `[ollama show, 2026-10-01]`

### `qwen3.8:latest`

- **116/125** — ab 57/65, gap 59/60. `[ab/gap-next4, 2026-08-19]`
- **112/125 with thinking OFF** — ab 54/65, gap 58/60. Flipped
  `reasoning-decimal-compare` to `9.11 > 9.9` **3/5**.
  `[ab-off-r5b + gap-off-r5b, 2026-08-19]`
- **17 GB** — ruled out for the router slot on size, not ability. `[config.yaml]`
- ▶ **THE `general` FAST-PATH PRIMARY, at priority 100** — and it leads `code`
  at 100 too, and sits third in `reasoning` at 94. muse-glimmer is 84 and
  ornith-1.5:35b 83, so neither is reached while qwen3.8 is healthy.
  ⚠️ Read this before reasoning about the fast path: it is NOT the
  highest-scoring local (muse beats it by 6 points thinking-on and 11
  thinking-off). It holds the slot on **tool calling**, which is what this path
  is actually for and which was measured directly — 5/5 correct selection in
  all three thinking states, `false` 2x faster on under half the tokens. See
  the fast-path table below. `[config.yaml + thinking_probe, 2026-08-19]`
  ⚠️ **A duplicate stub of this section existed lower in the file** and said
  only "leads both the `code` and `general` fast-path pools". That sentence is
  what got read as "muse-glimmer is the primary" across three consecutive
  answers on 2026-08-19. Stub deleted 2026-08-20 — **one section per model**.
- **Middle of the three on latency.** Thinking off, warm, grounding suite: ttft
  mean 0.49s, total mean 2.29s, median 2.00s. Cold load 29.1s.
  `[gap-off-r5b, 2026-08-19]`
- ⚠️ **Fabricated a named official.** `gk-berlin-wall` 4/5: one run attributed
  the press conference to "the regime's spokesman **Josef Ahern**" — an invented
  person — and then invented a scholarly debate about whether Ahern "misspoke or
  was misquoted". Same class as the `ornith` Walter Momper failure.
  `[ab-next4, 2026-08-19]`
- `writing-cold-email` **2/5** — lowest of any case for this model.
  `[ab-next4, 2026-08-19]`
- Strongest measured refusal behaviour on `gk-nonexistent-paper` (5/5): states
  the paper does not exist AND says why it will not guess ("I'd rather flag the
  gap than invent a plausible-sounding summary"). `[ab-next4, 2026-08-19]`

### `qwen3.8` quant variants — `27b-mtp-q4_K_M`, `27b-mtp-q8_0`

Bake-off 2026-08-25 against the incumbent `qwen3.8:latest` (Q4_K_M, 17 GB).
Same 27.3B weights, three builds. `eval_prompts_code_hard_models.json`,
5 cases x `--repeat 5` = 25 draws per arm, all 5 cases `executed`.
`[quant-mtp + quant-q8, 2026-08-25]`

- ⛔ **NEITHER VARIANT EARNED PROMOTION. `qwen3.8:latest` stays.**
- **Quality is identical across all three.** latest 24/25, mtp-q4 25/25,
  mtp-q8 24/25. One failure each in two arms, on different cases, both real
  code bugs (latest: `lru-ttl` omits the expired-purge from `put`; q8:
  `parse-duration` writes `\d+[hms]+` where it needs `(?:\d+[hms])+`, so
  `'1h30m'` is rejected). **Q8_0's extra 12 GB bought no measurable quality.**
- ✅ **MTP is genuinely active** — `ollama show --modelfile` reports
  `PARAMETER draft_num_predict 4`, a Modelfile default needing no client
  involvement. This is a real measurement of MTP, not of an inert tag.
- **MTP delivered NO speedup.** At Q4 it was slower on 5/5 cases, medians
  +3% to +38%. Sign test p = 0.062 and pooled Mann-Whitney z = -0.86, so the
  DIRECTION is consistent but significance is not reached. What is established
  is the absence of the advertised 1.4-2.2x, not the presence of a penalty.
- ▶▶ **WHY MTP COULD NOT HELP, and this is the durable finding.** MTP
  accelerates token generation only. The generation window (`total - ttft`) is
  **1.3-4.5s** in ALL THREE arms, against 6-100s totals — TTFT is **65-94%** of
  every request. A perfect 2x on generation would save 1-2s of a 20s request.
  ⚠️ Generalise this before benchmarking any generation-side optimisation on
  this box: on a thinking model the reasoning tokens are emitted BEFORE the
  first content token, so they land inside TTFT. You are timing reasoning, not
  answering.
- ⚠️ **`mtp-q8_0` has a heavy tail.** Median 12.7s is the LOWEST of the three,
  but mean 23.4s is the highest and max is **100.8s** against latest's 56.8s.
  On `lru-ttl` its five draws spread 63.3s (37.5 / 49.1 / 50.6 / 84.9 / 100.8)
  and the 97.4s ttft was repeat #4, NOT the cold load. Most requests are fine;
  some are catastrophic. See *Not established* for what this is.
- **29 GB against 24 GB per card** (`ollama list`), so it cannot be resident on
  one card and must span both. ✅ **It does, entirely on GPU** (replaces "same
  class of risk as `llama4:latest`", 2026-10-01). Served as `qwen3.8:q8-32k`
  with a 32k window, `ollama ps` read 30 GB at **100% GPU**, and `nvidia-smi`
  showed 15,900 and 16,777 MiB in use on the two cards, with `nomic-embed-text`
  resident beside it. Unlike `llama4`, it is not split onto the CPU.
  `[ollama ps + nvidia-smi during ai-sec's eval, 2026-10-01]`

#### Thinking-OFF re-run — the decisive arm `[q4-thinkoff + q8-thinkoff, 2026-08-25]`

Same three arms, same suite, `--repeat 5`, `THINK=off`. ⚠️ `THINK` forces
Audrey-direct, so these numbers are NOT comparable to the thinking-ON block
above (no OWUI system prompt, sampling params or retrieval context). Compare
the three models WITHIN this arm only.

- ✅✅ **TTFT WAS REASONING, CONFIRMED.** Median TTFT collapses from 4-97s to
  **0.20s** — every arm, every case. The 65-94% TTFT share recorded above was
  thinking tokens being generated before the first content token, not prompt
  processing. Generation is now ~100% of the request.
- ⛔⛔ **MTP IS A NULL RESULT, AND THIS SUPERSEDES THE READING ABOVE.** With
  thinking off, MTP has the ENTIRE request to accelerate. It changes nothing:
  per-case median deltas vs `latest` are **+0.00 / +0.00 / +0.00 / -0.10 /
  +0.10 s**, mean delta **0.000s**, largest **0.10s**, Mann-Whitney
  **z = +0.10**. Not faster, not slower — flat.
  ▶ The thinking-ON "slower on 5/5 cases" is **WITHDRAWN as thinking-length
  variance**, not an MTP penalty. It was measured through TTFT, which MTP does
  not govern, on unseeded draws of differing reasoning length.
  ▶▶ **The advertised 1.4-2.2x does not appear on this box on any setting,
  including the one where MTP had everything to work with.**
- **`mtp-q8_0` is genuinely slower, and this reading IS clean** (TTFT 0.2s, so
  it is pure generation rate). Slower on 5/5 cases, +6% to +59%, Mann-Whitney
  z = -1.81. Median total ratio **1.32x** against a **1.71x** weight ratio
  (29 GB / 17 GB) — the shape expected of bandwidth-bound generation.
- ✅ **The 100.8s q8 tail was thinking, not memory pressure.** Warm max drops
  from 100.8s to **14.3s** and the cold load completes in 40.7s. See
  *Not established* — the hypothesis is largely retired, not proven wrong.
- **Quality: 70/75 across all three arms, against 73/75 thinking-ON.** latest
  24/25, mtp-q4 23/25, mtp-q8 23/25. Directionally consistent with this model's
  known thinking penalty (116/125 on vs 112/125 off) but 3 cases is noise.
  Failures concentrate on `lru-ttl`, the hardest case.
- **Warm median total, `qwen3.8:latest`: 13.4s thinking-ON → 3.90s
  thinking-OFF.** ⚠️ The request path differs too, so this is NOT a clean
  thinking delta — it is directionally consistent with the fast path's measured
  6x token reduction and should not be quoted as a thinking coefficient.
- ▶ **Both Q4 arms co-resided.** In the sweep, `mtp-q4` took no cold load after
  `latest` (first case ttft 0.2s), confirming 17+17 GB sits in 48 GB without
  eviction. The run-shape rule holds.
- **What `ollama show` reports for `27b-mtp-q8_0`:** architecture `qwen35`, 27.3B,
  Q8_0, context length 262,144, embedding length 5,120, requires Ollama 0.32.12.
  Capabilities completion, vision (a 460.73M `clip` projector), tools and
  thinking, with thinking levels `false`/`low`/`medium`/`xhigh` and default
  `medium`. Apache 2.0. `[ollama show, 2026-10-01]`

### `qwen3.8:32k`, `qwen3.8:16k`, `qwen3.8:q8-32k` — ai-sec's tags, not Audrey's

The ai-sec project (`~/Documents/github/ai-sec`) writes its report narratives with
these. Audrey's config names none of them, so `check_model_inventory.py` lists
them as unreferenced and reclaimable. ⛔ **That list is a report, not a delete
list:** removing one of these breaks ai-sec's eval. ai-sec sends temperature 0
and both penalties 0, and at those settings these tags measured
bit-deterministic (below), so one ai-sec draw is a measurement rather than a
sample. `[check_model_inventory.py + ai-sec eval, 2026-10-01]`

- **How they are built.** Each is its parent's weights plus one `PARAMETER
  num_ctx` line. `qwen3.8:16k` and `qwen3.8:32k` are `FROM qwen3.8:latest`, at
  16384 and 32768. `qwen3.8:q8-32k` is `FROM qwen3.8:27b-mtp-q8_0` at 32768,
  created 2026-10-01. `ollama create` reused every existing layer, so a tag like
  this costs no disk. All other parameters are the parent's: `top_p 0.95`,
  `draft_num_predict 4`, `min_p 0`, `presence_penalty 0`, `repeat_penalty 1`,
  `temperature 1`, `top_k 20`. `[ollama show --parameters + /api/show, 2026-09-30
  and 2026-10-01]`
- **The Q4 and Q8 tags share one chat template**, sha256 `b507b9c2f6ca…`, so
  they think the same way. `[ollama show --template | sha256sum, 2026-10-01]`
- ✅ **`qwen3.8:32k` is bit-deterministic at temperature 0 on a quiet box.** In
  five draws of ai-sec's 42 cases, all 36 outcomes were identical in every draw
  and identical to the run a day earlier. `qwen3.8:16k` before it: 22 narratives
  byte-identical across five draws, and one packet byte-identical across a real
  reload (`/api/ps` read before and after). ⚠️ That is ai-sec's request, not
  this file's harness, which sets no temperature or seed.
  `[ai-sec eval-out/model-2026-09-24-134656, 2026-09-24; model-2026-08-31-213302,
  2026-08-31]`
- **A 32k window fits on one card, 100% GPU, at no measurable cost.** The 16k
  and 32k arms of one 38-case run finished in 1,021.94s and 1,018.84s, with
  `num_ctx 32768` served in every case. `[ai-sec
  eval-out/constrain-2026-09-01-160823, 2026-09-01]`
- **Throughput on ai-sec's workload:** 42 cases in 1,765s. The slowest case took
  92s for 7,956 completion tokens. Counting each case's whole span from file
  timestamps, which includes the non-model steps, that comes to about 82
  completion tokens per second, so generation runs at least that fast. The
  largest prompt plus reply was 18,226 tokens. `[ai-sec
  eval-out/model-2026-09-30-200843, 2026-09-30]`
- **Thinking was on although ai-sec never asked for it.** ai-sec sends no
  `think` field to `/v1/chat/completions`. Its accepted narratives run 0.45-1.31
  characters per completion token (median 0.72, n=25), against this file's
  plain-prose baseline of ~4. So most completion tokens were reasoning, which is
  the thinking probe's "omitted IS thinking" result on another endpoint. `[ai-sec
  eval-out/model-2026-09-30-200843, 2026-09-30]`
- ⚠️ **It writes an absent setting as a missing protection.** On ai-sec's
  hardening findings where a check found a setting absent rather than wrong,
  the model is told the host then runs a default, which may be the secure one.
  Even so, under ai-sec's narrative contract 2.14 it described 3 of 17 such
  absences as the host doing without the protection, down from 14 of 18 under
  2.12. This is the same class as nemotron's invented metric: a consequence the
  source does not state, written as fact. `[ai-sec
  eval-out/model-2026-09-30-200843, 2026-09-30]`

### `gemma4:31b-it-q4_K_M` and its ai-sec tag `gemma4:31b-32k`

Pulled 2026-10-01 for ai-sec's second model comparison, and kept on the box
afterwards by the user's decision. Audrey's config names neither, so
`check_model_inventory.py` lists both as reclaimable. Leave them unless the user
says otherwise.

- **What it is:** Gemma 4 31B, dense, 30.7B, Q4_K_M, from Google DeepMind under
  Apache 2.0, released 2026-04-02, with configurable thinking. Ollama also
  offers `31b-it-q8_0` (34 GB) and `31b-it-qat` (19 GB). `[Ollama library +
  Hugging Face model card, 2026-10-01]`
- **Built-in parameters:** `draft_num_predict 3`, `temperature 1`, `top_k 64`,
  `top_p 0.95`. The ai-sec tag adds `num_ctx 32768` and `repeat_penalty 1`.
  `[/api/show via ai-sec, 2026-10-01]`
- ⚠️ **`ollama ps` under-reports it.** With a 32k window it read 3.5 GB at
  `100% GPU`, while `nvidia-smi` showed 13,544 and 13,947 MiB in use on the two
  cards with only `nomic-embed-text` beside it. So it spans both cards. Read
  `nvidia-smi` for this model's footprint, not `ollama ps`. `[ollama ps +
  nvidia-smi, 2026-10-01]`
- **Thinking is on by default:** with no `think` field sent, its accepted
  narratives run 0.27-0.69 characters per completion token. `[ai-sec
  eval-out/model-2026-10-01-142858, 2026-10-01]`
- **ai-sec eval, one draw at temperature 0:** 574/574 gates, 19/19 thresholds,
  26 narratives, the most of the three models tried. It had none of the
  authority-phrasing refusals qwen3.8 had (3 at Q4, 4 at Q8). But it misread an
  absent setting as a missing protection in 4 of 18 cases. It printed
  packet-local references such as "(4)" or "[9, 10]" in 18 of 26 narratives,
  against 5 of 25 for qwen3.8:32k. It returned non-JSON on 2 of 3 adversarial
  cases, at 368s and 479s. 3,871s for the 42 cases, 2.2x qwen3.8:32k. ⛔ **Not
  adopted by ai-sec.** `[ai-sec eval-out/model-2026-10-01-142858, 2026-10-01]`

### `llama4:latest`

- ⛔ **DROPPED FROM THE MODEL SWEEP 2026-08-19**, replaced by `ornith-1.5:35b`.
  The disqualifier is the VRAM split below, not the missing thinking: every
  latency figure it produced measures the GPU/CPU boundary rather than the
  model, so it cannot be compared with anything else in this file on speed and
  its quality score cannot be compared on the thinking arm either. ▶ It is
  still the only model here with a clean **thinking-off** measurement, so keep
  these numbers as the reference point if a `THINK=off` arm is ever run.
- **107/125** — ab 52/65, gap 55/60. Lowest measured. `[ab/gap-next4, 2026-08-19]`
- ⛔ **Its 107/125 is a THINKING-OFF result and is NOT comparable to the rest of
  this file.** `llama4` does not declare the `thinking` capability, so
  `ollama.thinking_flag` omits the field entirely rather than sending it —
  Ollama hard-errors on `think` for a model that lacks it. The logs read
  `wanted=True resolved=None src=request` and `thinking_len=0` on **every**
  call, with `chars_per_tok` 2.35–5.36 (healthy prose, no reasoning overhead)
  against 0.06–2.9 for the thinking models. ▶ **The arm is not uniform across a
  sweep**: `THINK=on` silently degrades to no-thinking for any model without the
  capability, and nothing in the results artifact says so — only the logs do.
  `[audrey-ai logs, 2026-08-19]`
- ⚠️ **67 GB against 48 GB of VRAM.** It cannot be resident; Ollama splits it
  across GPU and CPU. Its ttft sat at **21–23s on essentially every case** with
  a 155.9s cold load — that is a memory-bandwidth measurement, not a measurement
  of the model. Treat every llama4 latency figure here as a floor imposed by the
  split. `[config.yaml + ab-next4, 2026-08-19]`
- `instruction-strict-json` **0/5** — fenced the JSON in ```json every time, and
  emitted a **duplicate `replicas` key** in 4 of 5. `[gap-next4, 2026-08-19]`
- `science-mrna-vaccines` **0/5**, `code-word-frequency` **1/5** (its
  tie-break loop raises `IndexError`, and once timed out at 15s).
  `[ab/gap-next4, 2026-08-19]`

### `qwen3.5:4b` — the router

Production router (`router.model`). Probed 10 cases × 3 rounds.

| arm | latency median | parse | accuracy | conf median |
|---|---|---|---|---|
| thinking | 4.75–4.96s | 93–100% | 89–90% | 0.95 |
| no_thinking | **0.46s** | **100%** | 80% | **0.97** |

`[router probe, 2026-08-16, recorded in config.yaml]`

- Production runs `no_thinking: true` **and** `pin_schema: true`, both read in
  `classify.py`. Thinking costs ~10× latency on the hot path of every
  non-skipped turn and confidence went *up* without it. `[config.yaml, verified 2026-08-19]`
- The 9–10 point accuracy loss is **the cheap kind**: the case it loses
  ("draft a polite email" → code) routes `general` vs `code`, and both fast-path
  pools lead with `qwen3.8:latest` — the same model answers either way. The
  expensive misroute (a DB question → `reasoning`) occurs in *every* arm.
  `[config.yaml]`
- `timeout_s: 20`, `max_failures_before_fallback: 2`, `skip_llm_under_tokens: 8`.
- ⛔ **NOT targetable via passthrough.** It is absent from the
  `passthrough.allow` list in `config.yaml`, and passthrough gates on **that list
  alone** (`routes/openai/passthrough.py:95`) — a `model_registry` entry does not
  grant access. A 2026-08-19 attempt to put it through the suites returned
  `HTTP 403 Passthrough not allowed for model 'qwen3.5:4b'` on all 125 samples.
  ▶ Its quality can only be measured by adding it to that list, or via
  `router_probe.py`, which calls Ollama directly. `[ab/gap-next4, 2026-08-19]`

### System One router candidates — rejected 2026-09-30

Ollama 0.35.0 completed all 148 calls: one cold and 36 warm calls for each of
`tev1:0.8b`, `tev1:latest`, `nimble:latest`, and the incumbent
`qwen3.5:4b`. Every response was valid; there were no transport, HTTP, or shape
failures.

| Model | Correct | False reasoning | Missed reasoning | Escalations | warm p50 / p95 | cold | package / resident |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tev1:0.8b` | 22/36 | 14 | 0 | 8 | 0.089s / 0.092s | 4.98s | 0.81 / 0.89 GB |
| `tev1:latest` | 30/36 | 4 | 0 | 4 | 0.169s / 0.174s | 5.43s | 4.48 / 4.67 GB |
| `nimble:latest` | 33/36 | 2 | 1 | 2 | 0.200s / 0.215s | 15.21s | 9.53 / 8.97 GB |
| `qwen3.5:4b` | **34/36** | **1** | **0** | **1** | **0.184s / 0.199s** | **11.06s** | **3.39 / 4.20 GB** |

Thirteen fixture cases never reach the production model router: nine hit a
strong keyword gate and four hit the short-prompt rule. Restricting the result
to the 23 model-reached cases removes those shared deterministic outcomes:

| Model | Correct | False reasoning | Cheap swaps | Uncertain |
|---|---:|---:|---:|---:|
| `tev1:0.8b` | 13/23 | 10 | 0 | 7 |
| `tev1:latest` | 18/23 | 3 | 2 | 3 |
| `nimble:latest` | 22/23 | 1 | 0 | 1 |
| `qwen3.5:4b` | **23/23** | **0** | **0** | **0** |

Nimble was closest, but its SQL window-function case went to costly
`reasoning`. The provisional winner >= 0.55 and margin >= 0.15 rule would
abstain on that low-confidence error, which still creates an escalation that
the incumbent avoids. Nimble was about 9% slower at warm p50, about 38% slower
cold, and used about 2.1 times the resident memory. It displaced Tev1 4B during
sequential loading; the incumbent later displaced Nimble. `qwen3.8:32k`
remained resident throughout, so this records displacement rather than active
worker contention.

Two legacy expected-code cases are deterministically routed to `reasoning`
before any model call. End to end, that leaves Nimble at 33/36 with three
costly false-reasoning outcomes and the incumbent at 34/36 with two. No System
One candidate clears the accuracy, escalation, and footprint gate. **Keep
`qwen3.5:4b`; do not open the production-backend slice.** System One
`confidence` is distribution concentration and was not treated as correctness
probability. `[2026-09-30-systemone-router-probe-results.json, 2026-09-30]`

### Clef System One follow-up — rejected 2026-10-03

Ollama 0.35.1 measured `clef:latest` (27B Q4_K_M, 17.99 GB package) and
`clef-flash:latest` (9.1B Q8_0, 10.93 GB package) on the 36-case router fixture
for three warm rounds. Full Clef returned 108/108 correct warm samples with no
failures, no costly false-reasoning route, 0.341s/0.347s p50/p95, and 32.48 GB
resident. Clef Flash returned 104/108 successful and correct samples with four
20-second timeouts, 0.204s/0.215s p50/p95, and 12.78 GB resident. The incumbent
returned 102/108 raw-fixture correct with no failures, 0.197s/0.210s p50/p95,
and 4.20 GB resident.

After Audrey's deterministic gates, full Clef and the incumbent were both
69/69 across the 23 production-reached cases repeated three times. Clef Flash
was 67/69 because two reached cases timed out during cold loading; it made no
wrong selection after becoming responsive. Both Clef models exceeded the
router's 20-second cold timeout. The broad probe measured full Clef's cold
request at 55.13s. Clef Flash then timed out on the cold request and first four
warm requests before responding.

Full Clef's separate broad benchmark completed 36/36 warm requests and scored
84/87 questions, 33/36 exact cases, choice 30/33, yes/no 36/36, and score 18/18.
JSON was 9/9 exact and image input 3/3. It consistently labeled a legitimate
password-change notification as phishing. Warm p50/p95 was 0.381s/0.457s;
yes/no Brier mean 0.000614, choice expected-label log loss 0.262787, and score
out-of-range distance zero.

**Keep `qwen3.5:4b`.** Full Clef passes routing quality but fails cold latency
and footprint: its observed residency is 7.7 times the incumbent's and would
exceed 48 GB beside the 19.64 GB `qwen3.8:32k` residency seen in the original
run. Clef Flash fails cold-start reliability at Audrey's production timeout.
No production implementation slice opens.
`[2026-10-03-clef-systemone-router-results.json;
2026-10-03-clef-systemone-decision-results.json, 2026-10-03]`

### `ornith-1.5:35b`

- **59/60 on the grounding suite**, `--repeat 5`. One real failure; see below.
  `[gap-r5b, 2026-08-19]`
- ✅ **Thinking confirmed applied.** Every call logged
  `wanted=True resolved=True src=request` with a non-zero `thinking_len`, so
  this score IS comparable to muse/glm/nemotron and is NOT the llama4 case.
  `[audrey-ai logs, 2026-08-19]`
- **22 GB.** Cold load ~44s; warm cases 2.6-12.6s elapsed.
  `[ollama list + audrey-ai logs, 2026-08-19]`
- ✅ **The best chars-per-token of any thinking model measured here** on
  ordinary prose: 0.53-1.90, against muse's 0.06-1.19 on the same suite. It
  spends its budget on the answer rather than on reasoning the user never sees.
  `[audrey-ai logs, 2026-08-19]`
- ⚠️ **Its reasoning budget is unstable on hard code, and the instability is
  large.** On `code-rle-roundtrip` — five draws of one identical prompt — it
  generated 4,008 / 7,354 / 12,108 / 21,854 / 24,169 chars of thinking against
  500-675 chars of answer, a six-fold spread run to run and `chars_per_tok`
  down to 0.08. Every other case in the suite sat between 223 and 5,108. muse
  on the same case was stable at ~5,700-6,000. ▶ It is also **the one case it
  got wrong**: after 24k characters of reasoning it still shipped
  `int(s[i:j])` where `j` starts at `i+1` while the slice starts at `i`, so it
  parses `'a3'` as an integer and raises. **Thinking longer did not make it
  more correct, only slower.** `[gap-r5b + audrey-ai logs, 2026-08-19]`
- Refuses well: on the absent-subtopic case it states outright that it will not
  invent an answer, and separately flags its own AMQP prefetch assumption as
  world knowledge rather than grounding. `[gap-r5b, 2026-08-19]`
- ⛔ **NO LONGER IN ANY PANEL, 2026-08-25 (user decision).** It held
  `deep_panel_local.general` from 2026-08-19, having displaced muse-glimmer on
  COST rather than accuracy; muse has been put back at the top of that pool to
  favour accuracy. Superseded record: it took the slot on warm 2.6-12.6s
  against muse's 13-20s, at 59/60 against muse's 60/60.
  ▶ It remains in `model_registry` (priority 83, below muse's 84, so fast-path
  selection is unchanged either way) and in `passthrough.allowed_models`, so it
  is still targetable by `eval_research.py --models`. ⚠️ Its registry comment
  still reads "REGISTERED SO IT CAN BE A PANEL WORKER" — stale as of this
  change, harmless where it sits.
  ⛔ Was deliberately kept OUT of `deep_panel_local.code` — see the thinking
  instability above; that reasoning stands if it is ever reconsidered.
- **What `ollama show` reports:** architecture `qwen35moe`, a mixture of experts,
  35.5B in total, embedding length 2,048, context length 262,144, Q4_K_M.
  Capabilities tools, thinking, completion and vision (a 446.57M `clip`
  projector). ⚠️ **It prints no licence and no parameters**, so its terms of use
  are not stated anywhere on the box. `[ollama show, 2026-10-01]`
- ⛔ **Worst of the five on general quality with thinking OFF** — 48/65, against
  2 failures on the grounding suite in the same arm. Four defect classes, all
  absent when it thinks: answers truncated to their closing sentence
  (`science-attention` 4/5 returned only "In one sentence: …" plus an offer to
  elaborate), `reasoning-race-order` landing on a wrong final order 4/5 after
  reasoning correctly in the body, `string.punctuation` used without
  `import string` 2/5, and one substituted name on `gk-berlin-wall`. ▶ It is
  the most thinking-dependent model measured. ▶ **That risk retired 2026-08-25
  when it left the pool** — it was a hard dependency for as long as it led
  `deep_panel_local.general`, and would become one again on any re-entry.
  `[ab-off-r5b, 2026-08-19]`

### `ornith-1.5:9b` — ⛔ REMOVED FROM THE BOX 2026-08-19

Pulled as a router candidate, probed, disqualified, deleted the same day.
The measurements below are kept because they are the reason, and because
they are the clearest evidence on file that **router accuracy is not the
binding constraint — confidence calibration is.** ⚠️ They can no longer be
re-measured. Dropped from `passthrough.allow` and `pull-models.sh` with it.

Probed head to head against the production router, 10 cases x 3 rounds,
`NOTHINK=1 FORMAT=1` (the production-matching arm).

| | `qwen3.5:4b` | `ornith-1.5:9b` |
|---|---|---|
| parse | 30/30 | 30/30 |
| accuracy | 24/30 | 24/30 |
| latency median | 0.48s | 0.49s |
| conf median | 0.97 | **0.90** |
| **conf at/above 0.95** | **27/30** | **9/30** |

`[router probe, 2026-08-19]`

- ⛔ **Not on accuracy — they are identical**, down to the same two failing
  prompts (Postgres query plan -> reasoning; polite email -> code). Latency is
  a tie. It loses on **confidence calibration alone**.
- ▶ Escalation fires on `conf < 0.95` STRICTLY. That is 3 escalations in 30 for
  the incumbent against **21 in 30** for this candidate — roughly seven times
  the deep-panel traffic, at three cloud calls each, for identical routing.
  Against a hard credit budget that settles it. This is exactly the "routes
  correctly but timidly" failure `router_probe.py` was written to catch.
- Also 6.6 GB against a slot that is small on purpose: the router is not
  GPU-gated, so under `GPU_CONCURRENCY=1` it would evict the deep worker.
- **Keep `qwen3.5:4b`.** The router question is closed.

### `laguna-xs-2.1:latest`

No quality facts on any suite. It is still on the box as of 2026-10-01, though
queued for removal (see *Not established*).

- **What `ollama show` reports:** architecture `laguna`, 33.4B, embedding length
  2,048, context length 262,144, Q4_K_M, requires Ollama 0.32.3. Capabilities
  completion, tools and thinking, with thinking levels `false`/`true` and
  default `true`. No parameters. Licence: OpenMDW License Agreement, version 1.1.
  `[ollama show, 2026-10-01]`

### Other installed local models

`nemotron-3.5-lightning:latest`, `llama4:latest`, `laguna-s-2.1:latest`,
`laguna-xs-2.1:latest`, `qwen3-vl:32b` (vision), `llava:34b` (vision),
`nomic-embed-text:latest` (embeddings). `[pull-models.sh]`
No quality facts established for any of them on the current suites.

✅ **Inventory drift closed 2026-08-19.** `ornith:latest` and `ornith-1.5:9b`
are both gone from the box and from `config.yaml` + `pull-models.sh`;
`ornith-1.5:35b` is the surviving tag and is allow-listed.
`scripts/ops/check_model_inventory.py` should be clean.

---

## Cloud models

The reported Kimi K3 / GLM 5.3 Responses capability check is recorded under
[Not established](#kimi-k3-and-glm-53-client-protocol-assessment-received-october-6-2026).
Its individual timings are retained as observations from one bot assessment;
they do not establish a cloud speed or quality ranking.

`deepseek-v4-pro:cloud`, `kimi-k2.6:cloud`, `kimi-k2.7-code:cloud`,
`qwen3.5:397b-cloud`, `deepseek-v3.2:cloud`, `deepseek-v4-flash:cloud`,
`nemotron-3-super:cloud`, `glm-5.2:cloud`. `[pull-models.sh]`

⚠️ **Cloud credit is a hard budget.** No cloud model may hold the `general`
fast-path primary slot; cloud earns deep-pool slots only.

⚠️ **`reasoning` is the only fast-path task that lands on cloud** —
`deepseek-v4-pro:cloud` leads that pool at priority 100. Anything the router
labels `reasoning` spends credit on a *fast* turn, which is why router
confidence and accuracy are a budget concern and not only a quality one.
`[config.yaml]`

---

## Thinking: measured on all five locals, 2026-08-19

`scripts/probes/thinking_probe.py`, SAMPLES=5, NUM_PREDICT=2048, default reasoning
prompt. Content chars per state:

| model | omitted | true | false | false wall | false eval |
|---|---|---|---|---|---|
| muse-glimmer | 1,382 | 769 | 2,338 | 19.7s | 912 |
| glm-4.7-flash | **0** | 461 | 2,244 | 4.6s | 511 |
| nemotron-3.5-lightning | 757 | **0** | 2,936 | 3.4s | 653 |
| ornith-1.5:35b | **0** | **0** | 2,661 | 3.8s | 607 |
| qwen3.8 | **0** | **0** | 3,794 | 14.6s | 900 |

- ⛔ **SIX OF TEN thinking cells returned ZERO visible content.** `ornith-1.5:35b`
  and `qwen3.8` returned nothing in BOTH thinking states, five samples each.
  This is the `len=0` defect, and it is not confined to cloud structuring calls
  — it reproduces on every local model on an ordinary prompt.
- ▶ **Mechanism: the budget is spent before the answer starts.** Every thinking
  run pinned `eval_count` at 2048; every `false` run finished naturally at
  511-912. Reasoning length is flat across all five models (7,689-8,784 chars)
  regardless of size or family. ⚠️ 2048 is the PROBE's budget — check what
  `num_predict` production sends before concluding this is silent rather than
  merely slow there.
- ✅ **All five honour `think=false` exactly** (0 reasoning chars, every run).
  None is a `qwen3-vl` case.
- ✅ **`omitted` IS thinking**, confirmed on five more models: 7,689-8,532
  against 7,704-8,784 for `true`. Indistinguishable. Every non-vision path that
  leaves the field unset pays full reasoning cost, unchosen.
- ⚠️ **This measures LENGTH, not QUALITY.** `false` yields an answer where
  thinking yielded none; whether it is a good answer needs a `THINK=off` suite
  arm. Do not promote a config change on this alone.

### Fast path — `TOOLS=1`, qwen3.8, the role-matching mode

| state | tool called | choice | wall | eval |
|---|---|---|---|---|
| omitted | 5/5 | `get_file_text` ×5 | 1.5s | 100 |
| true | 5/5 | `get_file_text` ×5 | 1.5s | 111 |
| false | 5/5 | `get_file_text` ×5 | 0.8s | 45 |

- ✅ **`fast_path.no_thinking: true` is CORRECT for the model that serves it.**
  Tool selection is identical in all three states; `false` is 2x faster on
  fewer than half the tokens. The config line was justified on `qwen3.6:35b`
  and never re-measured after `qwen3.8` replaced it 2026-08-15. Now measured.
- ⚠️ **`0c` content in this table is NOT the defect above.** A tool-calling turn
  has no prose by design — the model returned `tool_calls` instead of talking,
  in all three states, which is the correct move. Same number, opposite
  meaning. Read the `tools=` column in this mode, never the content column.

---

## Thinking OFF: the quality arm, both suites, 2026-08-19

The measurement the length probe above could not make. `THINK=off`, all five
locals, `--repeat 5`, both suites, run after the three check fixes — so it is
comparable to the `THINK=on` runs of the same evening and NOT to the older
per-model numbers earlier in this file, which predate those fixes.

| suite | thinking on | thinking off | cost |
|---|---|---|---|
| `eval_prompts_models_ab.json` (general quality) | 308/325 | **281/325** | −27 |
| `eval_prompts_local_models.json` (grounding) | 297/300 | **282/300** | −15 |

`[ab-off-r5b + gap-off-r5b, 2026-08-19]`

### Per model, grounding suite — the one clean paired comparison

| model | on | off | delta |
|---|---|---|---|
| muse-glimmer | 60/60 | **60/60** | 0 |
| ornith-1.5:35b | 59/60 | 58/60 | −1 |
| qwen3.8 | 60/60 | 58/60 | −2 |
| nemotron-3.5-lightning | 60/60 | 55/60 | −5 |
| glm-4.7-flash | 58/60 | 51/60 | −7 |

Thinking-off failures on the general-quality suite, for contrast: ornith-1.5:35b
17, qwen3.8 11, glm-4.7-flash 9, nemotron 5, muse-glimmer 2.

### Latency, thinking off, warm cases only

Computed from the 59 warm cases of the grounding run (the first case of each
model is a cold load and is excluded). This is the like-for-like speed table —
the older per-model ttft figures earlier in this file were measured with
thinking ON and are not comparable to these.

| model | ttft mean | total mean | total median | cold load |
|---|---|---|---|---|
| nemotron-3.5-lightning | **0.32s** | **0.90s** | 0.90s | 39.0s |
| qwen3.8 (fast-path primary) | 0.49s | 2.29s | 2.00s | 29.1s |
| muse-glimmer | 3.55s | 5.71s | 5.00s | 27.7s |

`[gap-off-r5b, 2026-08-19]`

### The paired on/off latency delta

Same 59 warm cases, same suite, thinking the only variable:

| model | ttft on → off | total on → off |
|---|---|---|
| muse-glimmer | 13.04s → 3.55s (**3.7x**) | 15.17s → 5.71s (**2.7x**) |
| nemotron-3.5-lightning | 4.73s → 0.32s (**14.7x**) | 5.15s → 0.90s (**5.7x**) |
| qwen3.8 | 4.59s → 0.49s (**9.3x**) | 6.25s → 2.29s (**2.7x**) |

Worst case: muse 37.2s → 12.2s, nemotron 20.0s → 1.9s, qwen3.8 17.9s → 8.1s.
`[gap-r5b vs gap-off-r5b, 2026-08-19]`

- ⛔ **Thinking costs ~4s on a PROSE turn, not ~0.7s.** The 0.7s figure is the
  TOOL-turn delta (1.5s → 0.8s, `thinking_probe` TOOLS=1) and does not transfer
  to prose — it was briefly written into `config.yaml` as the justification for
  `no_thinking_prose: false` and is corrected here. The prose price for
  qwen3.8's +4/125 is **6.25s vs 2.29s mean total**, and **4.59s vs 0.49s ttft**
  — the user waits nearly 5 seconds before the first token instead of half a
  second. ▶ Any future argument about fast-path thinking must use the prose
  numbers; the tool numbers flatter it by ~6x.
- ▶ **The accuracy ranking and the speed ranking are inverted.** muse-glimmer is
  the most accurate thinking-off model (123/125) and the slowest (6.3x
  nemotron on total, 7 of 59 cases over 8s). nemotron is the fastest and
  fabricates metrics. qwen3.8 is second on both and holds the slot.
- ⚠️ Cold load is 27-39s for all three and is a per-eviction cost, not a
  per-turn one — under `GPU_CONCURRENCY=1` whichever model serves the fast path
  stays resident. Do not read it as a latency difference between them.

- ▶ **The cost is task-shaped, not model-shaped.** `ornith-1.5:35b` is the worst
  model in the run on open-ended generation and nearly the best on grounded
  synthesis over supplied passages — 17 failures against 2, same model, same
  arm, same evening. Thinking buys reasoning-from-world-knowledge. It buys
  very little when the answer is already in the prompt. ▶ The deep panel's job
  is the second shape, which is the one that degrades least.
- ✅ **`fast_path.no_thinking: true` is safe where it is applied — but the
  reason is TOOL SELECTION, not prose.** ⚠️ **CORRECTED 2026-08-25.** This
  bullet previously read "`muse-glimmer` is the fast-path primary and is the
  one model thinking-off does not move", and was wrong twice over:
  1. **The fast-path primary is `qwen3.8:latest`** (priority 100 in `code` and
     `general`) against muse-glimmer's 84. This is the SAME stale claim the
     duplicate stub caused on 2026-08-19 and that was deleted 2026-08-20 — it
     survived here, in a different section. ▶ One section per model, and check
     `config.yaml` priorities before asserting who serves a path.
  2. **It justified the flag with the wrong branch's evidence.** `no_thinking`
     governs the **ReAct/tool branch only**; the plain-chat branch is
     `no_thinking_prose: false`, i.e. thinking stays ON. So muse's prose scores
     could not justify this line even if muse held the slot.
  ▶ **The actual argument, on the model that actually serves it:** tool
  selection is **5/5 correct in ALL THREE thinking states**, so thinking buys
  nothing on that branch and costs ~0.7s per ReAct round, which compounds over
  a loop. `[thinking_probe TOOLS=1, 2026-08-19]`
- ✅ **`fast_path.no_thinking_prose: false` is the deliberate opposite**, and
  qwen3.8's −4/125 thinking-off is exactly why. Prose is one call, not a loop,
  so the ~4.0s does not compound the way it would on the tool branch.
  `[ab/gap-off-r5b, 2026-08-19 + config.yaml]`
- ⛔ **Do NOT extend `no_thinking` to the deep panel.** −27 on general quality is
  the whole argument.

### What breaks, by class

- ⛔ **`nemotron-3.5-lightning` invents metrics when the fact is absent.**
  `ground-fact-absent` answered **`2322 ms`** on one draw and
  **`p99 latency: 1424 ms`** on another — numbers that appear nowhere in the
  passage, stated bare, no hedge, in a two-word answer. It was **60/60** on this
  suite with thinking on. A grounding model fabricating a metric is the exact
  failure that ends user trust, and it appears only in this arm.
  `[gap-off-r5b, 2026-08-19]`
- ⚠️ **`glm-4.7-flash` loses strict formatting completely.** `instruction-strict-json`
  **0/5** — every draw wrapped in ```json fences, and `replicas` came back as
  `3`, `"between two and five"`, `{"min":2,"max":5}` and `"between 2 and 5"`,
  never the required form. With thinking on it fenced 2/5. Thinking off makes it
  total. `[gap-off-r5b, 2026-08-19]`
- ⚠️ **Reasoning relocates into the answer.** With no thinking channel the
  scratchpad has nowhere else to go. `glm` opened `reasoning-race-order` with
  "Alright, let's tackle this puzzle step by step", enumerated all six
  permutations, took a tangent about the prompt saying "four friends", and
  closed with a `### Final Answer` header — correct answer, transcript output,
  `no_reasoning_leak` fired. Same mechanism produced ornith's mid-answer
  self-corrections on the other suite ("Wait — let me double-check"), which
  land on the wrong order when they do not complete.
- ⚠️ **Truncation to the closing paragraph persists.** `ornith-1.5:35b` returned
  only the trailing `**Note:**` caveat on `synth-merge-three-drafts` — the
  briefing itself absent. 1/5 here against 4/5 on `science-attention` in the
  general-quality arm. See
  [answers truncated to their conclusion](#answers-truncated-to-their-conclusion).
- ⚠️ **`nemotron` emitted literal placeholder code.** `code-rle-roundtrip` shipped
  `if ch == s[result[-1] - result[-1] ...]:  # placeholder to make syntax happy,
  will be overwritten`, then wrote a working implementation below it. Exit 1,
  SyntaxError.

### A check gap this run exposed

`synth-absent-subtopic` caught `glm` inventing back-pressure handling on one
draw and **passed it on another** that asserted "if a single consumer stalls,
the remaining consumers in the group can pick up the slack" — equally invented,
from a passage that says nothing about it. `nemotron` passed a similar draw
attributing back-pressure to AMQP ACK mechanics. ▶ The check fires on some
fabrications and not others; it is scoring shape where it should be scoring
whether the claim is in the passage. Worth a pass.

---

## Cross-model observations

- **Secondary fabrication passes every check we have.** On
  `gk-nonexistent-paper`, `glm-4.7-flash` scored PASS while attributing
  "Self-Attention with Linear Biases" and "Inferencing 1D Transformers
  Efficiently" to Vaswani's group in 2019; `ornith` passed twice while inventing
  "When to Use Recurrent Networks" and a 2020 Wang et al. paper. The check
  catches *asserting the fake paper exists*, not *inventing replacements while
  correctly denying it*. Blacklisting those titles is the trap
  `_CORPUS_FICTIONS` documents. Secondary citations stay human-read.
  `[ab-think-r5, 2026-08-19]`
- **`9.11 > 9.9` is a live failure mode** at roughly 1-in-5 for both
  `muse-glimmer` and `glm-4.7-flash`; `ornith` was clean 5/5. A single draw
  would have ranked these three models three different ways — this is the
  clearest argument on record for `--repeat 5`. `[ab-think-r5, 2026-08-19]`
- ### Answers truncated to their conclusion

  ⚠️ **Open — but NOT explained by content loss. See the thinking-cost entry
  below; the first hypothesis was measured and did not hold.** Across
  three unrelated models, some answers arrive as *only the closing paragraph* of
  an answer that was clearly longer, or open at a mid-document `###` heading with
  the first requested section missing:
  - `qwen3.8` `science-attention` #1/#3/#5 returned only a recap ("That's the
    whole mechanism…", "Recap of the logical chain:"), while #2 and #4 returned
    full multi-section explanations. Same case, same run, same prompt.
  - `qwen3.8` `synth-merge-three-drafts` #4 returned the single line
    "*End of briefing. No additional sources were consulted.*"
  - `nemotron-3.5-lightning` `science-mrna-vaccines` **5/5** opened partway in.
  - `glm-4.7-flash` showed the same shape on 3 cases in the earlier run.

  ▶ ⚠️ **Checked 2026-08-19 and the obvious explanation FAILED.**
  `passthrough.stream` log lines for `glm-4.7-flash` and `muse-glimmer` show
  `content_len` matching the answers that actually appeared — 809 chars for a
  full race-order deduction, 3,763 for a full attention explanation, 36 for a
  one-line decimal comparison. **No content was lost in transit for those two
  models.** So "the body went into `thinking`" is not a general explanation.
  ▶ Still unexplained for `qwen3.8` and `nemotron`, whose truncated cases fall
  outside the log window that was read.
  ▶ **Narrow the next check to those calls specifically:**
  `docker logs audrey-ai --since 5h | grep -A2 'last_head=.Explain how the attention' | grep stream`
  and read `content_len` on the `science-attention` repeats. If content_len is
  small there, it is the model choosing to answer in the reasoning channel; if
  it is large, the loss is downstream of Audrey and is a harness bug.
  ▶ ⚠️ It also passes checks it should not: `qwen3.8` `science-attention` #1 and
  #5 scored **PASS** on recap-only answers, because a good recap still contains
  the needles `softmax` and `dot product`.
- ### Thinking costs 3–15× the tokens, confirmed

  `THINK=on` reaches Ollama on every call — `think=True resolved=True
  src=request` appears on all of them, so the arm is real and per-request
  resolution works. `[audrey-ai logs, 2026-08-19]`

  What it costs, measured from `content_len` vs `thinking_len`:

  | case | content chars | thinking chars | thinking share |
  |---|---|---|---|
  | `reasoning-decimal-compare` (glm) | 36 | 1,221 | **97%** |
  | `instruction-strict-json` (muse) | 57 | 2,502 | **98%** |
  | `reasoning-race-order` (glm) | 809 | 13,557 | 94% |
  | `ground-fact-present` (muse) | 130 | 825 | 86% |
  | `science-attention` (glm) | 3,763 | 10,245 | 73% |
  | `science-mrna-vaccines` (glm) | 4,434 | 4,734 | 52% |

  ▶ **The shorter the required answer, the worse the ratio.** A 36-character
  answer cost 427 generated tokens. `chars_per_tok` ran **0.08–2.2** against a
  plain-prose baseline of ~4, so on short-answer cases 86–98% of everything
  generated was reasoning the user never sees.
  ▶ ⚠️ **The arm is not uniform.** `llama4` ran this same sweep with **no
  thinking at all** (`resolved=None`, `thinking_len=0`) because it lacks the
  capability. Any cross-model comparison in this file mixes a thinking arm and a
  non-thinking one, and only the logs reveal which is which.
  ▶ This is a latency cost locally and would be a credit cost on cloud. It is
  the strongest argument on file for running the suites `THINK=off` before
  treating any latency number here as the model's.
- **Speed and trustworthiness are inversely ordered** in everything measured so
  far: `ornith` is 4–8× faster and last on quality; `muse-glimmer` is slowest
  and first. No measured model is both.

---

## Video-summary role probe (2026-09-29)

One synthetic BJJ transcript plus visual-description block was sent with the
production 2-3 sentence summary prompt and a 240-token output limit. No uploaded
video content was used. This is a single instruction-following check, not a
general quality benchmark, and no per-model latency was recorded.

| Model | Location | Result |
|---|---|---|
| qwen3.8:latest | Local | HTTP 200; complete natural three-sentence summary; zero thinking characters |
| ornith-1.5:35b | Local | HTTP 200; complete natural three-sentence summary; zero thinking characters |
| muse-glimmer:latest | Local | HTTP 200; stopped mid-sentence, so it is excluded from this role |
| deepseek-v4-pro:cloud | Cloud | HTTP 200; complete concise three-sentence summary |
| kimi-k2.6:cloud | Cloud | HTTP 200; complete three-sentence summary before the 80-word sanitizer |
| qwen3.5:397b-cloud | Cloud | HTTP 410: model retired on 2026-09-25 |

The shipped decision is qwen3.8 primary with ornith-1.5:35b fallback. This keeps
private video-derived text on the Audrey host and makes an unusable primary
response switch models instead of repeating the same failed call. The cloud
outputs are evidence that those models followed the synthetic prompt only; they
were not selected because changing private-data providers needs an explicit
privacy decision.

## Not established

Open questions, and what would close each.

### Retained Qwen skill-selection pilot (October 6, 2026)

**Observed protocol success, not an established selector-quality result.**
Tower ran the metadata-only hybrid evaluator for one Spanish document request
(`positive-document-spanish`), one repeat, using `qwen3.5:4b`. Rules abstained;
one model call correctly selected `grounded-document-analysis`, with valid
output and zero errors or retries. Automatic selection remained disabled.

| Observed measurement | Result |
|---|---:|
| Cases / model calls | 1 / 1 |
| Valid / correct model choices | 1 / 1 |
| Request elapsed time | 7.35355s |
| Input / output tokens | 277 / 11 |
| Temperature / output ceiling | 0 / 128 tokens |

Source: Tower log `2026-10-06-105732-eval_skill_selection.log`, report timestamp
`2026-10-06T16:57:40.340643+00:00`; [retained JSON](results/2026-10-06-skill-selection-pilot-results.json).
The conditional-router section duplicates the same call, not an independent
observation. Model loading was uncontrolled, and no seed was recorded. This
single positive sample cannot establish warm/cold latency, false activation,
production selection precision, or comparative model quality. Keep it separate
from task-router measurements above and from the 42-case offline rules report.
The subsequent full study is recorded below; it found excessive false activations.
Holdout and workflow evidence remain required before enabling automatic selection.

### Retained Qwen automatic skill-selection study (October 6, 2026)

**Decision: Keep automatic selection disabled.** The rules-first hybrid recovered
positive requests missed by the rules, but also activated skills for unrelated or
explicitly excluded file tasks. Its selection precision fell from 70.00% to
63.16%, and ordinary-request false activations rose from 6/42 to 15/42. These
results fail the proposed selection-quality gate. They do not overturn the
separate task-routing results that retain `qwen3.5:4b` as Audrey's router.

**Method.** Tower ran the same 42 synthetic cases three times in each of two
serial studies, using `qwen3.5:4b`, temperature 0, a 128-token output ceiling,
a 20-second timeout, and no retries. Each study produced 126 decisions per full
arm: rules only, and rules first with a model call for an eligible abstention.
The latter made **30 actual model calls per study**. Models received catalog and
file metadata plus the request, without document contents, skill bodies, or gold
labels. No final answers or native user workflows were evaluated. The harness
and runtime configuration kept automatic selection disabled.

The later study is the primary report; both studies produced the same decisions,
error categories, and token counts. The first finished at 17:02:22 UTC before the
second started at 17:02:58 UTC, so these logs do not indicate overlapping study
calls. Repeating the same fixtures is not independent holdout evidence.

| Measurement per 126-decision study | Rules only | Hybrid |
|---|---:|---:|
| Correct decisions | 96/126 | 96/126 |
| Valid decisions / rejected choices | 126 / 0 | 123 / 3 |
| Correct activations / valid activations | 21/30 | 36/57 |
| Activation precision | 70.00% | 63.16% |
| Correct selections for positive requests | 21/42 | 36/42 |
| Missed positive requests | 21/42 | 6/42 |
| Valid false activations | 9 | 21 |
| Ordinary-request false activations | 6/42 | 15/42 |
| Actual model calls | 0 | 30 |

Activation precision uses valid activations as its denominator; the three rejected
choices remain recorded as errors. Each hybrid study exited 1 because
`ambiguous-video-negated` selected the document skill for video evidence on all
three repeats. The eligibility guard rejected those choices. There were **zero
transport, malformed-JSON, or truncated-output errors**; suppressing the three
rejections would not fix the selection-quality failure.

**What failed and what improved.** All repeats agreed within and across studies:

- The model recovered five positive case types: document contradiction checks,
  Spanish and French document requests, and Portuguese and German video requests.
  The two remaining positive misses concern explicitly selected document/video
  subsets among mixed attachments; the conservative eligibility guard abstained.
- Three false case types already selected by the rules remained: uncertain
  document identity, translating quoted document instructions, and finding a
  video delete control. The hybrid does not reconsider a positive rule choice.
- Model calls added valid false activations for a negated document request,
  file renaming, an unrelated attached document, and project organization. The
  negated video case produced the three rejected document-skill attempts.
- Among the **same 30 model calls** reported again in `conditional_router`,
  15 selected the expected skill, 12 made valid false activations, and 3 made
  ineligible attempts. There were **zero model abstentions**. This subset is
  conditional on a rules abstention; it is not another 30 calls or a complete
  model-only comparison.

**Measured model cost.** Latencies below include only the 30 model requests in
each study. The whole-hybrid median is zero because the report records guards at
zero elapsed time; it does not describe model response speed. Quantiles use
nearest-rank sampling.

| Model-call measurement | First study | Later study |
|---|---:|---:|
| Request latency median / p95 | 0.16070s / 0.20747s | 0.16197s / 0.21186s |
| Input tokens, total / measured calls | 8,223 / 30 | 8,223 / 30 |
| Output tokens, total / measured calls | 318 / 30 | 318 / 30 |
| Median input / output tokens per call | 273 / 11 | 273 / 11 |

Model loading was uncontrolled and no seed was recorded. These observations
cannot establish controlled cold/warm latency or a speedup relative to the
7.35-second single-case pilot. The fixture labels are proposed conservative
selection policy, with three repeats per case at temperature 0, and require
independent review. This study does not compare router candidates or measure
answer quality, workflow benefit, or production selection precision.

**Follow-up: evaluation policy revision 2.** [Slice 3D.2](../docs/campaign-3/phase-03-skills.md)
adds firm abstention and scoped target resolution to the standalone evaluator.
Its 3,845-test laptop gate passed. Offline rules on the unchanged development
set select nine of 14 positives correctly and abstain on all 28 negative
controls: 37/42 correct labels, zero false activations, five positive abstentions.
No model was called. These development results supply no new Qwen quality,
latency, or token measurements and do not replace the revision-1 model evidence
above. The first measurement on 30 separately authored proposed controls is
now recorded below; its three exposed findings are repaired in Slice 3D.3.
Those cases now serve as regression data, and their labels still need human
review. Automatic runtime selection remains disabled.

Sources: Tower logs `2026-10-06-110216-eval_skill_selection.log` and
`2026-10-06-110258-eval_skill_selection.log`; report timestamps
`2026-10-06T17:02:22.219753+00:00` and
`2026-10-06T17:03:04.733490+00:00`.
Retained raw reports: [first study](results/2026-10-06-skill-selection-hybrid-110216-results.json)
and [later study](results/2026-10-06-skill-selection-hybrid-110258-results.json).

### Retained Qwen skill selection on new controls (October 6, 2026)

**Decision: Keep automatic selection disabled.** Evaluation policy revision 2
improved selection on 30 separately authored synthetic proposed controls, but
retained three quality findings. The report completed without model errors;
its older exit-zero contract was not a selection-quality pass.

The controls were authored separately without access to selector code,
development prompts, or model logs; required project-state reading exposed
summary results. Their first measurement used `qwen3.5:4b`, one repeat,
temperature 0, serial requests, no retries, and at most 128 output tokens.
Gold labels and reasons stayed out of model requests. Automatic selection was
disabled. Labels received independent agent review, not human validation.

| Proposed-label measurement | Rules | Hybrid |
|---|---:|---:|
| Correct labels | 21/30 | 27/30 |
| Correct activations / activations | 5/5 | 12/13 |
| Activation precision | 100% (five activations) | 92.31% (13 activations) |
| Missed positive requests | 9/14 | 2/14 |
| Ordinary false activations | 0/8 | 1/8 |
| Actual model calls | 0 | 8 |
| Model errors / rejected ineligible choices | 0 / 0 | 0 / 0 |

The hybrid's router recovered seven eligible positives, falsely activated
video analysis on its sole ordinary request, and abstained zero times. It
recovered Spanish/Japanese document questions, a document request excluding an
unrelated video, a misleading filename, a French video question, an unrelated
negation, and a question about quoted video content. The two remaining missed
positives were incorrectly blocked by rules: locating a quoted phrase inside a
document, and describing a video while excluding an unrelated poster. The
ordinary false activation was Spanish file/project management. All eight
ambiguous controls abstained correctly. Proposed quality limits remain
unagreed and unmet.

The eight actual calls totaled 2,261 input and 80 output tokens. Model-only
median latency was 0.167845 seconds and nearest-rank p95 was 7.506008 seconds;
the first request took 7.506008 seconds. Loading was uncontrolled and there was
no seed or repeat distribution. These observations do not establish controlled
warm/cold speed, comparative router quality, production precision, or benefit
to final answers.

**Follow-up:** [Slice 3D.3](../docs/campaign-3/phase-03-skills.md)
repairs the three exposed regressions in evaluation policy revision 3 and
makes terminal results distinguish findings from successful execution. The
three-case Tower check passed on October 6 (report 22:23:57 UTC): all three labels
matched, zero errors, zero model calls, zero tokens. Its [preserved terminal JSON](results/2026-10-06-skill-selection-v3-regressions-smoke.json)
settles the regression gate. This is not a new Qwen performance measurement.
The measured controls are now exposed to tuning;
fresh reviewed controls and real workflow evidence are still required.
[Slice 3D.4](../docs/campaign-3/phase-03-skills.md) now supplies
frozen prospective study plans and 24 new agent-reviewed proposed controls.
The first frozen measurement is recorded in the following entry; quality gates
remain unmet and automatic selection stays disabled.

Source: Tower log `2026-10-06-155526-eval_skill_selection.log`, report timestamp
`2026-10-06T21:55:36.393253+00:00`;
[preserved full JSON](results/2026-10-06-skill-selection-policy-v2-holdout-results.json).
Raw fixture SHA-256: `098ac965713ea811ec9f90a32fb66fbc43061dbb20375258636ceeeeb80af4ba`;
canonical evaluated-case SHA-256:
`26b27546a68ca68f145ee637a6da970809046f35d7196059cfb20d6fad243b97`.
Both match the reserved fixture. Previous reports remain unchanged.

### Retained Qwen prospective skill-selection study (October 6, 2026)

**Decision: Keep automatic selection disabled.** The first frozen revision-3
hybrid study completed without model errors but failed proposed activation
precision and ordinary-false-activation criteria. Its 24 proposed synthetic
cases had three repeats. All three mismatches were the same ordinary request;
the other 23 distinct cases matched their proposed labels on every repeat.

| Measure | Observed |
|---|---:|
| Matches | 69/72 repeat observations |
| Correct activations / activations | 36/39 (92.31% precision) |
| Positive misses | 0/36 |
| Ordinary false activations | 3/18 observations; one of six distinct ordinary cases |
| Actual / planned model calls | 24 / 24, within the 36-call cap |
| Errors / rejected ineligible choices | 0 / 0 |
| Input / output tokens | 7,068 / 246 |
| Model-only median / nearest-rank p95 | 0.158339 / 0.176427 seconds |

`prospective-ordinary-03` asks in Japanese to rewrite a supplied sentence more
politely without reading an attached PDF. The retained `qwen3.5:4b` router
selected document analysis on all three repeats. Independent agent review
after measurement upheld the proposed `none` label. This is unnecessary skill
selection, not proof of a file read or degraded final answer: the standalone
study never activates skills or reads real uploaded content.

Temperature was zero, calls serial, no retries, and output capped at 128 tokens.
Frozen source/cases/catalog identities and planned call count match the pasted
preparation/result. Model tags, not verified weight digests, were bound; loading
was uncontrolled. The synthetic agent-authored labels remain pending human
validation. Repeat observations are not independent new examples; neither
production precision nor final-answer benefit is established.

Sources: operator-pasted shell output, report timestamp
`2026-10-07T02:29:27.091570+00:00` (October 6 locally), Tower logfile
`2026-10-06-202916-eval_skill_selection.log`;
[preserved laptop terminal result](results/2026-10-06-skill-selection-v3-prospective-summary.json)
and [preparation](results/2026-10-06-skill-selection-v3-prospective-preparation-summary.json).
The operator-copied [full report](results/2026-10-06-skill-selection-v3-prospective-full.json)
is now verified on the laptop. It confirms rules alone missed 21/36 positive
observations, all recovered by the hybrid. The first model call took 6.216
seconds despite the low median/p95; loading remains uncontrolled.
[Revision-4 policy repairs](../docs/campaign-3/phase-03-skills.md)
address the exposed request; hermetic guard tests supply no new model-quality
or latency measurement and do not change the original study result.
The measured fixture is now exposed regression data. Preserve the result;
do not repeat it unchanged or tune labels to the outputs.

### Revision-4 Japanese regression check — closed out of scope (October 6, 2026)

The operator reported a four-case hybrid regression run, timestamp
`2026-10-07T03:09:11.145355+00:00`: 3/4 matches, two guards, two model calls,
and one rejected ineligible choice. For the separate video request with an
excluded PDF, Qwen returned `grounded-document-analysis`; the evaluator rejected
it. That is an error/miss, not an accepted activation or a successful gate.
Input/output usage was 583/22 tokens; model latency was 0.170345/6.216807 seconds
at median/nearest-rank p95 with only two samples and uncontrolled loading.
The [private terminal receipt](results/2026-10-06-skill-selection-v4-japanese-smoke-summary.json)
preserves the exact supplied result.

The user reiterated a hard English-only input scope. This branch is closed
as out of scope without relabeling, a rerun, or another language repair.
Neither this check nor the earlier mixed-language aggregates establish English
production acceptance. Automatic selection remains disabled and further studies
are parked until a concrete English workflow warrants them. This adds no
production model recommendation or runtime-selection change.

### Kimi K3 and GLM 5.3 client protocol assessment (received October 6, 2026)

**Source and scope.** Claudette, a Hermes bot, supplied
`audrey-responses-assessment-telegram.md` and a follow-up conversation shared
October 5–6. The report's heading says October 3; the actual execution date is
not independently established. This entry records receipt on October 6.
The bot reported Hermes v0.21.0 and retained its existing
`/v1/chat/completions` connection throughout: primary
`audrey_passthrough/kimi-k3:cloud`, fallback
`audrey_passthrough/glm-5.3:cloud`. The assessment used isolated Responses
requests and local execution; it did not migrate the production client.

**Reported measurements.** Each row is an individual reported observation,
with no repeated samples, controlled sampler settings, or raw machine-readable
results supplied. Token pairs are reported input/output usage; a dash means no
measurement was provided. These cases establish reported protocol success only,
and do not establish general answer quality, latency distributions, billing
savings, or a speed ranking between the models.

| Kimi K3 case | Reported result | Latency | Input/output tokens |
|---|---|---:|---:|
| Basic completion | Completed; output text `OK` | 1.19s | 147 / 40 |
| Function call | Completed; `get_weather` with `city: Tokyo` | 2.04s | 235 / 85 |
| Function result → final answer | Completed; weather data used in the answer | 1.84s | 330 / 65 |
| Streamed text SSE | Completed; report records nine typed events | 2.23s | 150 / 55 |
| File write dispatch | Completed; expected function and arguments | 2.11s | — |
| Local execution → final answer | Completed; written file content verified | 1.63s | — |

| GLM 5.3 case | Reported result | Latency | Input/output tokens |
|---|---|---:|---:|
| Basic completion | Completed | 1.77s | 17 / 254 |
| Function call | Completed | 0.45s | 172 / 40 |
| Function result → final answer | Completed | 0.61s | 202 / 50 |
| Streamed text SSE | Completed; report records all nine event types | 0.75s | 20 / 94 |
| File write dispatch | Completed | 0.47s | — |
| Local execution → final answer | Completed; written file content verified | 0.44s | — |

The documented stream coverage is text: creation, progress, output item and
content part events, text deltas, and `response.completed`. The report does not
supply streamed function-argument deltas, so streamed function calls remain
unestablished for these two models. Non-streamed calls and client result replay
were reported successful on both models.

The isolated cross-model replay tests reportedly preserved history before a
function call and preserved the same call ID after a result, in both Kimi → GLM
and GLM → Kimi directions, without executing the function twice in those tests.
The pre-call failure was simulated using an unavailable model ID that returned
HTTP 403, followed by a manual retry with the other model. This does not exercise
an actual provider outage, a partially emitted stream, or the installed Hermes
adapter's retry and execution-deduplication behavior.

**Workload fit and decision.** The bot measured 20 model-visible functions and
35.4 KB of definitions in its actual request dump. The byte count fits Audrey's
64 KiB Responses allowance, but the count exceeds its 16-function maximum.
Audrey also permits eight calls per group and 128 replay items. The bot described
regular groups of four to eight calls, possible larger bursts, and a 150-turn
client setting; no oversized group or long-session payload was supplied here.
Keep the existing Chat Completions connection. The small Responses checks show
that both models can use its tested subset; they do not establish compatibility
with the bot's complete catalog and conversation workload.

**Corrections to the report and later follow-up.** These follow from Audrey's
request schemas and replay path, rather than additional model measurements:

- Responses supports `tool_choice: auto|none`. Required or named selection is
  unavailable, but function selection does not authorize execution. Hermes's
  executor must apply its approval policy before running a function on either
  API. The current Chat Completions schema also ignores an unmodelled
  `tool_choice` field, so effective forced selection on the existing connection
  has not been demonstrated.
- The current Chat Completions schema ignores `reasoning_effort`; it supports
  the capability-gated boolean `think` on passthrough requests. Responses rejects
  unknown `reasoning_effort` or `reasoning` fields with HTTP 422. The report's
  configured `medium`/`max`/`none` settings therefore do not prove that reasoning
  effort reached either model on the existing connection.
- The reportedly accepted “20K-token” request does not establish a soft limit:
  its exact payload, tokenizer, and request branch were not supplied. Audrey
  enforces a 16,000-token `cl100k_base` admission bound for client-function,
  hydrated-file, and stored/chained requests; an ordinary text-only request
  without those features follows a different branch.
- Opt-in `store: true` and `previous_response_id` are now implemented, and the
  owner-scoped retrieval/deletion and restart gate passed on October 6. Chaining
  can reduce client request bytes and history bookkeeping. Audrey rebuilds the
  full prior context for model inference, so chaining alone does not demonstrate
  lower input-token usage or inference cost. No provider caching or billing
  comparison was measured.

**What would settle the remaining claims:** retain actual request bodies and
raw Responses events for streamed function calls, the prompt-limit probe, and
reasoning/selection controls; compare repeated equivalent workloads before
claiming relative speed or savings. Changes to limits or the bot's provider
configuration require a concrete workload need, rather than these timings.

- **Smaller Qwen router alternatives.** Ollama's official Qwen 3.8 page
  publishes only 27B variants, so there is no smaller Qwen 3.8 parameter model.
  The related Qwen 3.5 family publishes `qwen3.5:2b` at 2.7 GB and
  `qwen3.5:0.8b` at 1.0 GB; neither has been measured on Audrey. The user
  retained `qwen3.5:4b` on 2026-10-03, so no smaller-router evaluation is
  scheduled. `[official Ollama model pages, checked 2026-10-03]`
  ▶ *If reopened:* require repeated production-arm parse, accuracy, confidence,
  latency, cold-load, residency, and 23 model-reached-case evidence.

- **`nemotron-3.5-lightning` quality.** Scored 2/5 and then 5/5 on the hard
  suite in the same window. ⚠️ Both are **n=1 arms with no seed**, so they
  settle nothing and neither number should be quoted. An earlier attribution of
  the gap to hardware memory corruption was **withdrawn** — inference ran in
  VRAM, a sibling model scored 5/5 in the same window, and the harness sets no
  sampler options, which explains it without any hardware theory.
  ▶ *Closes with:* both suites at `--repeat 5`.
- ⛔ **[CLOSED 2026-10-01] Whether `qwen3.8:27b-mtp-q8_0`'s latency tail is memory
  pressure.**
  ⛔ **LARGELY RETIRED 2026-08-25** by the thinking-OFF arm: warm max falls from
  100.8s to 14.3s and the cold load completes in 40.7s, so the extreme tail was
  variable reasoning length, not swapping. The residual +6-59% is a clean
  generation-rate difference consistent with 1.71x the weights to stream.
  ✅ **`ollama ps` read 2026-10-01**, during ai-sec's eval of `qwen3.8:q8-32k`
  (the same weights at `num_ctx 32768`): **100% GPU** across both cards, with
  `nomic-embed-text` resident beside it. These weights fit on the GPUs even with
  a 32k window. ⚠️ It cannot show what other models resident during the August
  sweep did to placement then. `[ollama ps + nvidia-smi, 2026-10-01]`
  Original entry follows for the record.
- **[superseded] Whether `qwen3.8:27b-mtp-q8_0`'s latency tail is memory pressure.** The
  bimodal shape (median fine, max 100.8s, 63s spread on one prompt) fits a
  29 GB model spanning two 24 GB cards with the embedder pinned resident, but
  **`ollama ps` was never captured during the run**, so the GPU/CPU split is
  unknown and the alternative — ordinary thinking-length variance — is not
  excluded. ⚠️ Until this is settled its latency column cannot be compared
  with the Q4 arms at all, exactly as with `llama4:latest`.
  ▶ *Closes with:* `ollama ps` during a re-run, read before anything else.
- ⛔ **[CLOSED 2026-08-25] Whether MTP's Q4 slowdown is real or verbosity.**
  Moot: the thinking-OFF arm shows MTP is a NULL (mean delta 0.000s) with the
  whole request available to it, so there is no slowdown left to attribute.
  The thinking-ON direction is withdrawn as reasoning-length variance.
  Original entry follows for the record.
- **[superseded] Whether MTP's Q4 slowdown is real or verbosity.** The harness sends no seed
  and no temperature, and the answers file carries no token counts, so "MTP
  generates slower" and "the MTP arm happened to write more" are
  indistinguishable. Several mtp-q4 answers are visibly wordier.
  ▶ *Closes with:* `eval_compare.py`'s mean-answer-length column on the results
  JSON; normalise latency by output tokens before claiming a rate difference.
- ⛔ **[CLOSED 2026-08-25] The whole quant bake-off ran with thinking ON.**
  Re-run with `THINK=off`; see the thinking-OFF block in the per-model section.
  ⚠️ It closed one gap and opened a smaller one: `THINK` forces Audrey-direct,
  so the two arms differ in request path as well as thinking and cannot be
  differenced against each other.
  ▶ *A clean thinking coefficient would close with:* both THINK arms run
  DIRECT, which no run has yet done. Original entry follows for the record.
- **[superseded] The whole quant bake-off ran with thinking ON.** `passthrough.think` is
  `null` (config.yaml), so the field is omitted and qwen3.8's template thinks by
  default. ⚠️ The fast path — where qwen3.8 actually serves — runs
  `think: false`. So NO arm here, including the incumbent baseline, describes
  the production latency profile, and TTFT dominance is partly an artifact of
  the arm rather than of the model.
  ▶ *Closes with:* re-running the three arms with `THINK=off`. That is also the
  only version of this test whose latency numbers would inform a serving
  decision.
- **`ornith-1.5:35b` quality**, and whether it fixes `ornith:latest`'s two
  trust defects (the `most_common` tie-break and the Berlin Wall fabrication).
  ⚠️ `ornith:latest` was deleted 2026-08-19, so a same-run A/B is no longer
  possible — the comparison is against the recorded numbers above.
  ▶ *Closes with:* both suites at `--repeat 5`. Note the `writing-eli5-rewrite`
  prompt changed on 2026-08-19, so that one case is **not** comparable to the
  recorded `ornith:latest` figure; the other twelve are.
- **Whether the THINK=on arm is helping or hurting.** No thinking-off arm has
  been run on the current suites, and the truncation above means the arm may be
  *costing* several models whole sections of their answers.
  ▶ *Closes with:* a `THINK=off` rerun of the same models on both suites. This
  is now the highest-value open measurement in this file.
- ⚠️ **Never sweep the large models together.** `config.yaml` is explicit:
  `laguna-s-2.1` is **96 GB against 48 GB of VRAM**, `llama4` is 67 GB. Even
  though `_expand_sweep` loads each model once, the total still forces eviction
  and reload from disk, and the result is a memory-bandwidth measurement. One
  `--models` value per run for anything that large. Check `ollama ps` after the
  first prompt and read the CPU/GPU split before spending a suite on it.
- **Whether `laguna-xs-2.1` is a mixture of experts.** Its embedding length,
  2,048, matches `ornith-1.5:35b`, which is one, and a dense model that narrow
  would need several hundred layers to reach 33.4B. That is an inference, not a
  reading. ▶ *Closes with:* the expert count in `ollama show -v
  laguna-xs-2.1:latest`.
- ⛔ **[CLOSED 2026-10-01] Whether Q8 reads ai-sec's findings better than Q4. It
  does not, measurably.** One draw each at temperature 0 on the same packets:
  Q8 misread an absent setting as a missing protection in 3 of 18 cases against
  Q4's 4 of 17, counting overstated theme titles (2 against 3 without them).
  The cases differ between runs, so the error moved rather than going away.
  Corrected the same day from "2 of 18 against 3 of 17", which left the titles
  out. Q8 also lost two more Windows narratives to
  ai-sec's authority guard ("below benchmark", "compliance requirements"), and
  took 2,581s for the 42 cases against 1,765s. This matches the 2026-08-25
  bake-off: Q8_0 buys no measurable quality. ⚠️ Q8's determinism at temperature
  0 was not measured separately. `[ai-sec eval-out/model-2026-10-01-080608 vs
  model-2026-09-30-200843, 2026-10-01]`
- **`laguna-s-2.1` / `laguna-xs-2.1`.** Queued for removal from `config.yaml`
  and `pull-models.sh`. A 2026-08-18 thinking-on arm came back within noise of
  thinking-off, but arm delivery was unverifiable at the time (the per-request
  `think` field did not exist yet), so that result proves nothing either way.
- **Disk sizes** for `muse-glimmer:latest`, `ornith:latest`,
  `glm-4.7-flash:q8_0` — never recorded.
- **Whether any measured difference is thinking-related.** Every number above is
  from a `THINK=on` arm. No thinking-off arm has been run on the current suites,
  so nothing here separates model quality from thinking-arm effect.
