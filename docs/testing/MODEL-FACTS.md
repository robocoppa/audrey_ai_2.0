# Model facts — what we have actually measured

A running ledger of what is **definitively known** about each model on this box.
Started 2026-08-19.

## The entry bar

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

- ⚠️ **`scripts/eval_research.py` sets no `seed` and no `temperature`.** Options
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

### `glm-4.7-flash:q8_0`

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

### `qwen3.8:latest`

- **116/125** — ab 57/65, gap 59/60. `[ab/gap-next4, 2026-08-19]`
- **17 GB** — ruled out for the router slot on size, not ability. `[config.yaml]`
- Leads **both** the `code` and `general` fast-path pools. `[config.yaml]`
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

### `qwen3.8:latest`

- **17 GB** — explicitly ruled out for the router slot on size, not on ability.
  `[config.yaml]`
- Leads **both** the `code` and `general` fast-path pools. `[config.yaml]`

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
- ▶ **Holds `deep_panel_local.general` as of 2026-08-19**, displacing
  muse-glimmer on cost rather than accuracy. Registered in `model_registry`
  BELOW muse so the fast path is untouched. ⛔ Deliberately kept OUT of
  `deep_panel_local.code` — see the thinking instability above.
- Not yet measured on the general-quality suite (`eval_prompts_models_ab.json`)
  — the `ab-r5b` run was SIGKILLed before writing an answers file.

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

### Other installed local models

`nemotron-3.5-lightning:latest`, `llama4:latest`, `laguna-s-2.1:latest`,
`laguna-xs-2.1:latest`, `qwen3-vl:32b` (vision), `llava:34b` (vision),
`nomic-embed-text:latest` (embeddings). `[pull-models.sh]`
No quality facts established for any of them on the current suites.

✅ **Inventory drift closed 2026-08-19.** `ornith:latest` and `ornith-1.5:9b`
are both gone from the box and from `config.yaml` + `pull-models.sh`;
`ornith-1.5:35b` is the surviving tag and is allow-listed.
`scripts/check_model_inventory.py` should be clean.

---

## Cloud models

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

## Not established

Open questions, and what would close each.

- **`nemotron-3.5-lightning` quality.** Scored 2/5 and then 5/5 on the hard
  suite in the same window. ⚠️ Both are **n=1 arms with no seed**, so they
  settle nothing and neither number should be quoted. An earlier attribution of
  the gap to hardware memory corruption was **withdrawn** — inference ran in
  VRAM, a sibling model scored 5/5 in the same window, and the harness sets no
  sampler options, which explains it without any hardware theory.
  ▶ *Closes with:* both suites at `--repeat 5`.
- **`ornith-1.5:9b` as router.** Size (6.6 GB vs the incumbent's ~2.5 GB) is the
  open risk, not classification skill. `classify_with_registry` takes no gate
  argument, so `FairLocalGate` never sees the router; under `GPU_CONCURRENCY=1`
  a large router **evicts** the deep worker rather than queueing behind it.
  ▶ *Closes with:* `router_probe.py` in the production arm (`NOTHINK=1
  FORMAT=1`) for parse/accuracy/confidence, **plus** a live turn with the deep
  worker resident to see whether it gets unloaded. The probe alone cannot
  clear it.
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
- **`laguna-s-2.1` / `laguna-xs-2.1`.** Queued for removal from `config.yaml`
  and `pull-models.sh`. A 2026-08-18 thinking-on arm came back within noise of
  thinking-off, but arm delivery was unverifiable at the time (the per-request
  `think` field did not exist yet), so that result proves nothing either way.
- **Disk sizes** for `muse-glimmer:latest`, `ornith:latest`,
  `glm-4.7-flash:q8_0` — never recorded.
- **Whether any measured difference is thinking-related.** Every number above is
  from a `THINK=on` arm. No thinking-off arm has been run on the current suites,
  so nothing here separates model quality from thinking-arm effect.
