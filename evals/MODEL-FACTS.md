# Language model field report

**Results, strengths, and failure modes from practical tests**

Updated November 27, 2026 · Measurements collected July–November 2026

This report compares local and hosted language models for writing, coding,
answering from supplied documents, selecting tools, and interpreting images.
It records what worked, what failed, and how much evidence supports each
conclusion. It is a small practical study, not a general leaderboard.

## The useful findings

- **Muse Glimmer had the strongest repeated local answer results**, but was
  slower than the alternatives with comparable warm timings.
- **Qwen 3.8 combined good executable-code results with reliable tool selection
  in a narrow test.** It still made arithmetic, attribution, and source-handling
  mistakes. Higher precision weights did not establish better quality.
- **Nemotron Lightning was fast with thinking disabled**, but invented numbers
  missing from a source. Its promising code-formatting result does not make it
  safe for factual answers under the same setting.
- **Qwen 3.5 4B remains the best fit for request routing on Tower.** AutoTrust
  JEV matched all 69 model-reached decisions but used 3.7 times the GPU memory
  and took more than twice as long on its first cold request.
- **Qwen3 Embedding 4B ranked document evidence better than Nomic.** A matched
  follow-up confirmed this with both page-based and upload-style chunks; 8B
  added cost without improving earlier headline counts. Nomic ranked one real
  conversation exchange higher on four matched lookups, supporting separate
  choices for documents and history. Gemma's early synthetic advantage narrowed
  on real documents. No model reliably rejected unsupported details.
- **Kimi K3.1 Pro and Hy4 both solved the harder cloud cases.** Each passed all
  six coding assertions and both challenging reasoning tasks. Hy4's observed
  median coding request took about 176s versus 13.4s for Kimi K3.1 Pro.
- **Output budgets and supplied evidence often mattered as much as model
  choice.** Hidden thinking could consume the answer budget; a broken document
  reader could make a capable model appear unable to answer.

## Test setup and terms

Local tests used a host with **two RTX 3090 Ti GPUs, 24 GB each**. Runtime memory
needs exceeding the combined 48 GB require slower main memory or cause failure.
Even models that fit can compete with other loaded models. Hosted-model timings
also include network and provider conditions. Earlier September–October decision
tests used **Ollama 0.35.0–0.35.1**; the October 15–17 router and retrieval
comparisons used **0.40.4**. Ollama serves models locally and through hosted tags.
Exact tested tags are retained because `latest` can change over time.
Immutable weight identifiers were not consistently recorded, which limits
exact reproduction of older runs.

**Thinking** means extra generated reasoning before the visible answer.
**Grounding** means answering from supplied text rather than filling gaps from
memory. **Tool calling** means choosing a function and its arguments; the
application executes it. **Routing** means classifying a request so the system
can choose an appropriate model or workflow.

**Warm** timing measures an already-loaded model; **cold** timing includes
loading. **Time to first text** measures when the first visible answer appears
and may include hidden thinking. **Median/p50** is the middle observation;
**p95** is the 95th percentile. **Tokens** are units of model input/output,
including reasoning where the provider counts it. An output-token limit can
therefore be reached before an answer appears.

Older answer tests generally used default sampling without a fixed random
seed. Later decision probes used temperature zero, a setting intended to reduce
variation. Each section states its case and repetition counts. Repeating a
prompt does not create a new independent example, and temperature zero does
not prove universal repeatability.

**Scores count case-runs passing automated checks**, not verified facts.
Checks cover requested content, format, refusals, and sometimes executable
code. Manual inspection found both false failures and incorrect answers that
passed. Timing comparisons apply to the measured workload and setup.

## 1. Local answers: strengths and weaknesses

### Repeated comparison

The August 19 comparison used **13 general tasks and 12 supplied-text tasks,
five repetitions each**, with thinking enabled and disabled. General tasks
covered reasoning, knowledge, explanations, writing, and code. The table uses
the later paired runs and corrected thinking-on document checks.

| Model | General: on → off | Supplied text: on → off | Combined: on → off |
|---|---:|---:|---:|
| `muse-glimmer:latest` | 64/65 → 63/65 | 60/60 → 60/60 | **124/125 → 123/125** |
| `ornith-1.5:35b` | 64/65 → 48/65 | 59/60 → 58/60 | **123/125 → 106/125** |
| `nemotron-3.5-lightning:latest` | 61/65 → 60/65 | 60/60 → 55/60 | **121/125 → 115/125** |
| `qwen3.8:latest` | 60/65 → 54/65 | 60/60 → 58/60 | **120/125 → 112/125** |
| `glm-4.7-flash:q8_0` | 59/65 → 56/65 | 58/60 → 51/60 | **117/125 → 107/125** |

These are **25 unique prompts per model**, not 125 independent tasks. The
recorded off score includes one known false failure: Nemotron answered
`2140 ms` correctly but a check required `2,140`. Conversely, passing answers
sometimes invented technical behavior or misordered historical events.

### What each model did well—and where it failed

**Muse Glimmer.** Strongest combined results in this comparison and little
change without thinking. It supplied the required alphabetical code tie-break
in 5/5 earlier repetitions and redirected a nonexistent-paper question toward
real adjacent publications in five inspected answers. However, it sometimes
claimed `9.11 > 9.9`, retained jargon in a simplified explanation, and produced
historical chronology errors that passed automated checks. Its high score does
not establish factual reliability beyond the tested material.

**Qwen 3.8.** In a separate file-reading function test, all **5/5 repetitions
in each of three states**—thinking omitted, enabled, and disabled—chose the
correct function. Disabling thinking reduced duration from about **1.5 to 0.8s**
and output tokens from **100–111 to 45**. This supports that narrow tool task.
Prose was less reliable: decimal comparison failed in 3/5 off-state draws; one
history answer invented an official's name, and one conflicting-source answer
incorrectly discarded a measurement. Some answers contained only a recap or
closing line while still passing keyword checks. An earlier short-email task
passed only 2/5 draws, a weakness not captured by its code/tool strengths.

**Nemotron 3.5 Lightning.** Very fast on short supplied-text answers without
thinking, but two missing-fact replies fabricated **`2322 ms`** and
**`p99 latency: 1424 ms`**. Another passing answer turned missing information
about flow control into a claim that no protection existed. It also emitted
placeholder code causing a syntax error. A separate repeated hard-code test
passed **22/25**, including two timeouts and one missing opening code fence
among the failures. Of 23 completed requests, the 22 after the first had
median **6.8s**, mean **19.73s**, and maximum **148.3s**; failed calls are excluded.

For code formatting, a ten-call probe with order reversed partway found fenced
output in **2/5 default-thinking draws versus 5/5 without thinking**. Generated
tokens fell from **3,713–34,463 to 475–695**. This measured formatting, not
execution correctness, and does not override the factual-answer failures.

**Ornith 1.5 35B.** Strong with thinking on, especially when evidence was
supplied. Disabling thinking hurt general answers much more than document
answers: wrong ordering, missing imports, and incomplete explanations appeared.
On one code task, reasoning length ranged from **4,008 to 24,169 characters**
over five draws; the longest still produced an invalid implementation.
Longer reasoning did not guarantee correctness. Its recorded package was
22 GB, with roughly 44s cold loading and 2.6–12.6s warm calls in that workload.

**GLM 4.7 Flash Q8—older version.** With thinking disabled, all five strict
machine-readable JSON-format attempts failed, often adding Markdown code fences or changing
required values. Supplied-text answers sometimes invented flow-control behavior.
Other answers began halfway through the requested explanation. These findings
apply to this local version, not hosted GLM 5.2 or 5.3.
Some replies correctly denied a nonexistent paper but then supplied invented
replacement citations; checking only the initial refusal missed that error.

### Speed and thinking cost

The following supplied-text timings exclude each observed first cold request
and contain **59 warm observations per model per state**.

| Model | Mean total: thinking on → off | Off-state mean time to first text | Off-state median total | First cold request, off |
|---|---:|---:|---:|---:|
| Nemotron Lightning | 5.15s → **0.90s** | 0.32s | 0.90s | 39.0s |
| Qwen 3.8 | 6.25s → **2.29s** | 0.49s | 2.00s | 29.1s |
| Muse Glimmer | 15.17s → **5.71s** | 3.55s | 5.00s | 27.7s |

The ranking is limited to these three measured models. Serial run order and
uncontrolled sampling limit the precision of a causal thinking-cost estimate.
Cold loading is paid again after eviction, not necessarily on every turn.

A separate five-sample-per-state probe capped output at **2,048 tokens**.
Six of ten model/state combinations with default or enabled thinking returned
no visible text; Qwen and Ornith did so in both states. All five models produced
visible answers and no hidden reasoning with thinking disabled. Inspected short
answers devoted **86–98% of generated characters** to reasoning—character
shares, not measured token percentages. Budget exhaustion is distinct from
poor answer quality, and disabling thinking has its own quality tradeoffs.

### Other local models: limited or historical evidence

| Model | Retained result | Main limitation |
|---|---|---|
| `llama4:latest` | 107/125; strict JSON 0/5, duplicate keys in 4/5; word-frequency code 1/5 | No thinking capability in the tested runtime. Its 67 GB package exceeded GPU memory; first-text times were 21–23s, with a 155.9s cold request. Not comparable to fully GPU-resident speed. |
| `laguna-xs-2.1:latest` | Two single supplied-text passes, each 11/12 | Both invented unsupported flow-control behavior. Warm medians 4.2/3.7s, means 6.45/6.98s, maxima 14.9/15.9s. Thinking delivery was unverified. |
| `laguna-s-2.1:latest` | Supplied-text 10/12; hard code 4/5; easier code 6/6; partial general 8/9, one pass each | Several failures were timeouts; one correct refusal was rejected by an old check. Completed calls reached 512.1s. The 96 GB package could not fit the host's GPU memory. |
| `ornith:latest`—older model | Earlier 111/125; alphabetical code tie-break 1/5; decimal comparison 5/5 | Missed code edge cases and invented historical attribution/quotes. Do not transfer these results to Ornith 1.5. |

**Practical lesson:** choose thinking settings by model and task. Formatting,
code execution, source fidelity, and visible-answer completeness need separate
checks; one combined score hides those differences.

## 2. Qwen variants and structured report writing

### Q4, Q8, and multi-token prediction

Quantization stores model weights at lower precision to reduce memory. Q4 and
Q8 here refer to roughly four- and eight-bit variants. Multi-token prediction
(MTP) is a generation optimization; a configured MTP tag alone does not prove
how many speculative tokens the runtime accepts.

The August 25 test used **five executable code tasks × five draws per variant**.

| Variant | Thinking-on passes | Thinking-off passes | Warm off median total |
|---|---:|---:|---:|
| `qwen3.8:latest`—Q4, 17 GB | 24/25 | 24/25 | 3.90s |
| `qwen3.8:27b-mtp-q4_K_M`—17 GB | 25/25 | 23/25 | 3.55s |
| `qwen3.8:27b-mtp-q8_0`—29 GB | 24/25 | 23/25 | 4.70s |

Quality differences were too small to establish superiority. Real failures
included missing expired-entry cleanup and a parser rejecting combined time
units. MTP Q4's paired per-task off-state median differences from the incumbent Q4
were **0/0/0/−0.1/+0.1s**, at 0.1s recording precision: no consistent observed
benefit. A pooled median need not match the average of paired task differences.
Speculative-token acceptance traces and a verified MTP-disabled control were
not retained, so this is a tag comparison rather than an isolated MTP test.

Q8 was 6–59% slower on off-state task medians. Its thinking-on maximum was
**100.8s**, versus **14.3s** warm with thinking off. However, on/off runs also
used different request paths and prompts, so neither that change nor Q4's
13.4→3.90s establishes an isolated thinking effect. Token and placement traces
were insufficient to prove the cause of the earlier tail.

Later observations showed Q8 at a 32K context entirely on both GPUs, alongside
the text embedder. This disproves a necessary CPU-offload claim for that later
setup; it does not reconstruct memory placement during August.

### Security-report narratives: Qwen versus Gemma

A separate workload turned structured security findings into prose. One pass
per model at temperature zero covered **42 pipeline cases**, including checks
that were not model-generated narratives; each narrative file had 36 outcomes.
The models used 32K context, meaning space for about 32,000 input/output tokens.

| Model | Accepted narratives | Total workload time | Absent settings overstated as missing protections |
|---|---:|---:|---:|
| `qwen3.8:32k`—Q4 | 25 | **1,764.65s** | 4/17 |
| `qwen3.8:q8-32k`—Q8 | 24 | 2,581.34s | 3/18 |
| `gemma4:31b-32k`—Q4 | 26 | 3,871.46s | 4/18 |

Error counts include overstated titles as well as body text. The failure is
consequential: an unset option may use a secure default, so “not explicitly
configured” does not mean “unprotected.” Q8 changed where errors occurred
without removing this problem.

Gemma produced the most accepted narratives but returned unparseable structured
output on **2/3 adversarial cases**, taking 368s and 479s. It carried internal
numbered references into **18/26 narratives**, versus **5/25 for Qwen Q4**.
All three pipelines passed their safety/check thresholds; those successes
include rejecting bad output and do not mean every model answer was correct.
Acceptance count alone therefore favors neither factual quality nor speed.

One Qwen 16K/32K context pair over 38 cases finished in **1,021.94/1,018.84s**,
showing no material difference in that pair, not zero context cost generally.
Q4 outputs were repeatable in several quiet-host, temperature-zero tests:
36 outcomes matched over five runs; an earlier 16K test had 22 byte-identical
narratives over five draws. Q8 repeatability was not separately measured.
Shared Q4/Q8 prompt templates do not prove identical reasoning.

Gemma's runtime summary reported 3.5 GB, while GPU monitoring showed roughly
13.5/13.9 GB device usage across the two cards with a small embedder alongside.
Treat download size, runtime estimates, and actual device use as different
measurements.

**Practical lesson:** more precision, more accepted text, and more passing
pipeline checks do not establish more accurate prose. Inspect the substantive
claims and count rejected answers as well as successful ones.

### Checking conditional rules against supplied facts

On October 22, `qwen3.8:latest` (digest `22130167…`, thinking off, temperature
zero) completed six synthetic tax-worksheet cases using a fixed official excerpt.
Each first draft was then assessed by the same model, with one repair allowed if
flagged. **Both versions passed 5/6; the checker improved none and approved the
wrong case.** No repair was triggered.

The failed draft applied a deduction without the required age, blindness and
dependency facts. The rule's conditions were supplied, but the model claimed no
facts were missing; its second pass agreed. Correct cases separated return years
and withholding, recognized a corrected record and duplicate copy, and selected
the supplied filing-status row. Decimal subtraction/rounding used server code,
so these are not model-arithmetic successes.

Checking added six calls, 10,160 input tokens and 870 output tokens. Request time
rose from 43.05s to 62.74s; cold loading was uncontrolled. This small study, with one sample per case,
tests fixed-passage checking, not document retrieval or complete returns.

A [targeted followup](results/2026-10-22-tax-review-eligibility-followup-results.json)
with explicit missing-fact instructions identified age, blindness and dependency
and asked one grouped question. Checking stopped for user input without approval
or a reviewer call. However, the structured applied deduction still contained the
quoted $15,750 rule amount despite prose saying it could not yet apply. The test
harness incorrectly derived a result from it and rejected equivalent field labels.
The single writer call used **853 input and 310 output tokens in 27.24s**;
cold loading was uncontrolled. The harness was corrected to suppress dependent
arithmetic while prerequisites are missing and accept equivalent field names,
without rewriting the raw applied amount.

An [October 23 rerun](results/2026-10-23-tax-review-eligibility-stop-results.json)
with the same model digest and case input again asked for all three missing facts.
The computed result stayed null, and server checking stopped as limited, needing
user input: **no false approval and no reviewer call**. The raw applied-deduction
field still contained $15,750, contradicting the stated unresolved eligibility.
This remains a structured-output defect, not a full accuracy pass. One writer
call used **885 input and 305 output tokens in 4.896s**; cold loading was
uncontrolled, so the faster request does not establish a speed improvement.

**Practical lesson:** the same model can repeat an unsupported assumption, and
recognizing missing facts in prose does not guarantee correct structured values.
Keep source rules, saved user facts and applied arithmetic separate. Further
isolated prompt tuning stops here; the next useful comparison exercises the
actual workflow from source retrieval through user confirmation and calculation.

## 3. Small models for request classification

The routing task assigned requests to **code, reasoning, or general help**.
A wrong “reasoning” label matters because it invokes a more expensive workflow.
Simple keyword and short-request rules handled 13 of a balanced **36-prompt
test**, leaving **23 prompts for the model**.

September used one warm pass per model; October's Clef comparison used three.
Cold tests deliberately unloaded the model first. Memory figures below are
observed runtime residency, not download sizes. Warm timing excludes failures.
The correctness column covers the 23 model-reached prompts; timing summarizes
all returned warm requests on the full 36-prompt fixture.

| Model | Correct model-reached decisions | Warm median / p95 | Cold request | Runtime memory |
|---|---:|---:|---:|---:|
| `tev1:0.8b` | 13/23 | 0.089 / 0.092s | 4.98s | 0.89 GB |
| `tev1:latest`—4B variant | 18/23 | 0.169 / 0.174s | 5.43s | 4.67 GB |
| `nimble:latest` | 22/23 | 0.200 / 0.215s | 15.21s | 8.97 GB |
| `clef:latest` | 69/69 | 0.341 / 0.347s | >20s; 55.13s in broader test | 32.48 GB |
| `clef-flash:latest` | 67/69 | 0.204 / 0.215s | >20s | 12.78 GB |
| `qwen3.5:4b` | 23/23; 69/69 | 0.184 / 0.199s; 0.197 / 0.210s | 11.06s; 10.29s | 4.20 GB |

Tev 0.8B produced **ten** costly false-reasoning labels on model-reached cases;
Tev 4B produced three. Nimble produced one, and a provisional low-confidence
rule would escalate that case rather than avoid the cost. Clef Flash's cold
request and four following sequential requests timed out at 20s; two of those
failures affected model-reached cases. Returned decisions were correct.

Raw labels across all 36 prompts were 22/36 Tev 0.8B, 30/36 Tev 4B, 33/36
Nimble, and 34/36 Qwen in September. October raw results were 108/108 Clef,
104/108 Clef Flash including four timeouts, and 102/108 Qwen. These differ from
the table because some prompts are handled by rules before model inference.

Full Clef's quality was strong, but its runtime memory was **7.7× Qwen's**.
Adding the separately observed 19.64 GB Qwen 32K residency gives about 52.1 GB,
beyond the host's combined GPU capacity. That is memory arithmetic, not a
concurrent-load test. Qwen 3.5 offered the better speed/memory/reliability fit
for this workload; the result is not a universal classifier ranking.

### Nimble v3: faster decisions, weaker routing fit

Two October 15 runs on Ollama **0.40.4** compared `nimble-v3:latest` (9B Q8_0)
with `qwen3.5:4b` (4.7B Q4_K_M), using the same 36 English prompts for three
then five repetitions. Production rules handle 13 prompts before inference;
the comparison below uses the **23 that actually reach the classifier**.
The five-round run provides the timing figures; both runs repeated the same
classification outcomes on that subset.

| Model | Correct model-reached case-runs | Warm median / p95 | Observed model VRAM |
|---|---:|---:|---:|
| `nimble-v3:latest` | 100/115 (87.0%) | 0.074 / 0.076s | 10.53 GB |
| `qwen3.5:4b` | 110/115 (95.7%) | 0.170 / 0.184s | 4.22 GB |

Nimble's strength was warm speed: roughly 95 ms saved per classified request.
It matched Qwen on general requests (8/8 unique cases) and reasoning accuracy
(4/5), with different reasoning misses. Coding was weaker: **8/10 versus
Qwen's 10/10**. Nimble labeled SQL rewriting as reasoning and TypeScript
narrowing as general; the former would select a more expensive workflow under
the tested routing policy. It also missed the database-design reasoning case.
Qwen missed the incomplete train reasoning prompt despite reporting confidence
1.0. These are fixture results, not general production failure rates.

Nimble used about **2.5 times the GPU memory**. Its first cold attempt timed
out at 20 seconds, followed by two early timeouts in the nominal warm set;
its next cold attempt completed in 11.23 seconds. The first full-fixture result
was 92/108 including failures, versus 99/108 for Qwen. Startup failures matter
even though most warm requests were fast. The probe confirms unload-request
acknowledgement, not verified absence from GPU memory; Qwen's second 0.248s
"cold" sample therefore does not establish a reproducible cold-loading time.

The proposed System One uncertainty thresholds flagged 3/23 model-reached cases
per round, versus none under Qwen's existing self-confidence threshold. Two of
Nimble's three flagged cases were correctly labeled. These are **uncertainty
flags, not observed downstream runs**; the confidence scales differ and are not
calibrated probabilities of correctness. Repeating 23 prompts five times does
not create 115 independent questions. No resident model was observed evicted,
but coexistence with a large answering model was not tested.

**Current choice: retain Qwen 3.5 4B for routing.** Nimble's warm speed does not
offset the lower labeled accuracy, extra costly coding misroute, larger memory
footprint and startup failures in this comparison. [Ollama's model page](https://ollama.com/library/nimble-v3)
also specifies Ollama 0.40.4+, a 32K text context, a decision-only API and
**CC BY-NC 4.0 (noncommercial)** licensing. These findings do not transfer to
the earlier Nimble release or to another quantization.

### Confidence can change costs without changing accuracy

An earlier 10-prompt, three-round comparison gave Qwen 3.5 and Ornith 1.5 9B
the same **24/30 correct labels**, with median durations 0.48/0.49s.
At a confidence threshold of 0.95, their reported scores would send **3/30
versus 21/30 requests** to a larger workflow. Those scores were not calibrated
probabilities of correctness. A confidently wrong or timidly correct model can
change downstream cost even when its label accuracy is similar.

An earlier ten-prompt study also illustrates a thinking tradeoff: disabled
thinking gave roughly **0.46s median, 100% valid output, and 80% correct labels**;
thinking variants took **4.75–4.96s**, with **93–100% valid output and 89–90%
correct labels**. Schema settings also varied. These small earlier tests should
not be pooled with the later 36-prompt comparison.

### Clef's broader decision test

Twelve cases contained 29 questions: choices, yes/no judgments, and rubric
scores over text, structured data, and one generated image. Three repetitions
produced **36 requests and 87 answers**. Clef completed all requests, answered
**84/87 correctly**, and matched every answer in **33/36 case-runs**.

Choice was 30/33; yes/no 36/36; rubric scores 18/18. Structured-data cases were
9/9 exact and the image case 3/3. Warm median/p95 was **0.381/0.457s**;
the deliberately cold request took **55.13s**. A legitimate password-change
notification was classified as phishing on all three draws.

This tested a decision interface, not conversational prose or code generation.
Three repetitions of one image are still one visual example.

### AutoTrust JEV: equal routing accuracy, higher memory

Audrey previously tested Jev-style decision models. The September 30 router
run included `tev1:0.8b`, `tev1:latest` (4B), and `nimble:latest` through
Ollama's `/v1/systemone`, which follows TypeSafe's Jev API. The September run
measured Together AI and Bespoke Labs checkpoints. The November 27 run
below is Audrey's first retained AutoTrust benchmark. [Audrey's September router results](results/2026-09-30-systemone-router-probe-results.json) · [Ollama's Jev-style decision API](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)

Ollama serves AutoTrust JEV through the same endpoint for choices, yes/no
probabilities, and rubric scores. `autotrust-jev:9b` is the 11 GB text package.
The `27b-vl` and `35b` variants also accept images. Ollama's launch notes
document Apple Silicon support, and the published variants use MLX packages.
The 11 GB listing is package size. [Ollama model card](https://ollama.com/library/autotrust-jev) · [Ollama tags](https://ollama.com/library/autotrust-jev/tags) · [Ollama launch notes](https://github.com/ollama/ollama/releases/tag/v0.40.15)

On November 27, the operator confirmed one successful Tower decision call
with `autotrust-jev:9b`. It chose `code` for a Python debugging request
against the alternative `general`, with a model-reported probability of
0.99996677. The reply reported 54 input tokens and zero output tokens.
The subsequent run measured GPU residency and router accuracy on Tower's
Linux/NVIDIA setup with Ollama **0.40.16**.

The paired run used the existing 36-prompt fixture, three rounds per model,
and one cold request per model. Audrey's rules bypass the model on 13 of
the prompts. Accuracy and warm latency below use the 23 unique prompts
that reach it. [Full router results](results/2026-11-27-085742-autotrust-jev-router-results.json)

| Model | Correct model-reached decisions | First warm pass median / p95 | Repeat median | Cold request | Observed GPU memory |
|---|---:|---:|---:|---:|---:|
| `autotrust-jev:9b` | 69/69 | 182 / 815 ms | 3.7 ms | 12.95s | 15.52 GB |
| `qwen3.5:4b` | 69/69 | 199 / 207 ms | 196 ms | 5.89s | 4.22 GB |

All 108 warm calls per model returned valid decisions. JEV was 105/108
and Qwen 102/108 across the full fixture. Their misses were on prompts that
Audrey's rules already intercept, so both produce the same 102/108 labels
when the current guards are replayed. Two coding prompts are sent to
reasoning by those guards. Changing the model would leave those two routes
in place. JEV's proposed uncertainty thresholds flagged 0/69 model-reached
cases, and Qwen's threshold flagged 3/69, all repetitions of one correctly
labeled reasoning prompt. These are threshold decisions in the fixture.
The benchmark did not run downstream workflows.

JEV's 3.7 ms median came from exact repeats in rounds two and three, whose
probability distributions matched the first round. The first warm pass
is the relevant comparison for fresh prompts. Ollama reported that both
resident embedders stayed loaded during the probe. Coexistence with a large
answering model remains unmeasured.

**Keep Qwen 3.5 4B for routing.** JEV matched accuracy and had a slightly
lower first-pass median. Qwen had a lower first-pass p95, shorter cold
load and a smaller GPU footprint.

The most promising first use is a bounded judgment, such as whether supplied
evidence supports a claim or whether an observed page change meets a Watch's
criteria. Watch summaries would still need the current text-generation step.
AutoTrust's published held-out results use its own data. Compare the
9B tag with Clef on the text and JSON cases from the existing
[System One decision probe](../scripts/probes/systemone_decision_probe.py).
Add labeled examples from the intended workflow and inspect false positives,
false negatives, and abstentions before using a probability threshold.

### Choosing document/video workflows is a separate task

A historical Qwen 3.5 experiment combined simple rules with model calls to
decide whether to activate document or video analysis. On 42 synthetic prompts
repeated three times, activation precision fell from **70.00% with rules to
63.16% with the hybrid**, despite recovering missed positives. Later revised
rules/fixtures yielded **36/39 correct activations** and **69/72 matching
decisions**; all three mismatches repeated one unrelated request.

Here, activation precision means the fraction of chosen workflows that matched
the proposed expected choice, rather than all decisions including abstentions.

Only 24 decisions in the latter study actually called the model. These were
mixed-language, proposed-label studies, not English production acceptance or
a controlled measure of model improvement. They do not establish performance
on English requests alone. The general lesson is to test unnecessary activation,
not just recovery of missed tasks.

## 4. Hosted models: keep version and task separate

### Kimi K3.1 Pro and Hy4

The November 27 Tower bake-off sent 26 text prompts to each of six Ollama cloud
tags using provider defaults. All 156 streams ended normally. Each prompt ran
once per tag, grouped by model. The timing includes provider and network
latency. The saved answers carry the exact model tags, Ollama tag digests, and
per-request timings. [Original answers](results/2026-11-27-073952-cloud-bakeoff-onbox-answers.md)

| Model | Original checks | After grading fixes | Python assertions | Median total |
|---|---:|---:|---:|---:|
| `kimi-k3.1-pro:cloud` | 25/26 | 26/26 | 6/6 | 10.2s |
| `hy4:cloud` | 25/26 | 25/26 | 6/6 | 23.9s |
| `kimi-k3:cloud` | 25/26 | 26/26 | 6/6 | 6.8s |
| `glm-5.3:cloud` | 25/26 | 25/26 | 6/6 | 6.2s |
| `deepseek-v4-pro:cloud` | 25/26 | 26/26 | 6/6 | 2.0s |
| `minimax-m3.1-flash:cloud` | 26/26 | 26/26 | 6/6 | 3.6s |

Two versions of the checker produce **151/156** and **154/156**. The cloud
tungsten fixture had required the word “German,” though the prompt asked
what the symbol came from. Kimi K3.1 Pro and DeepSeek correctly gave
`wolfram`. The word cap was also counting standalone dashes and a pipe as
words. Kimi K3's email has 120 words under the corrected count. Replaying
the 12 saved answers from the two affected prompts raised those three results.
The remaining flags were Hy4's technical wording in a rewrite for a child
and GLM's 151-word response to a 120-word email brief. [Tungsten reference](https://periodic-table.rsc.org/element/74/tungsten)

- **Kimi K3.1 Pro and Hy4** gave valid prime proofs, calculated the burst
  queues correctly, and kept the cause of conflicting latency measurements
  provisional. Both preserved ordinary return values in the async exercise
  even if a value happened to be an exception object. Hy4 also handled Unicode
  punctuation in the word-frequency exercise.
- **Kimi K3.1 Pro** used simpler language for the photosynthesis rewrite and
  preserved more of the supplied qualifiers in the compression exercise.
  Hy4 kept technical terms in the rewrite and presented invented résumé
  metrics as the sender's experience in the email.
- **Hy4's long coding times** were the largest practical difference. Median
  total time on the six coding requests was about **176s versus 13.4s** for
  Kimi K3.1 Pro. Hy4 took **300.7s** for the TTL cache. Its 26 requests used
  about **28 of the 52 minutes** of summed request time.
- **The fastest baselines had errors the checks missed.** MiniMax reported
  the initial hourly average as 75 requests/s instead of 87.5, although it
  got the backlog and the final choices right. DeepSeek added an unreported
  release build to the conflicting-measurement case. DeepSeek and GLM both
  treated a successfully returned exception object as a failed fetch.

This run favors Kimi K3.1 Pro for general panel work over Hy4 given the
latency gap and comparable correctness on the harder cases. The single pass
leaves room for targeted repeats where the models differed. The automatic
tallies cover the stated checks. Manual reading found additional mistakes.

The Cloud mode configuration uses DeepSeek V4 Pro, GLM 5.3, and Hy4
in that order for drafts. GLM and Hy4 tied on automated checks and GLM
was faster. Kimi K3.1 Pro synthesizes the answer, with GLM 5.3 as its
fallback. The configured order matters when fewer than three draft
slots are available.

### DeepSeek V4 Pro

After a document-reader repair, DeepSeek passed **six comparison/extraction
answers across two tasks × three repeats**. It attributed measurements to the
right files and distinguished application latency from human support response
time. End-to-end median was **5.48s**, range **2.41–11.63s**. Qwen 3.8 passed
the other three answers about facts absent from the documents.

The earlier run was 15/18 overall; its three DeepSeek failures involved missing
reader contents. This is a system failure boundary, not evidence that the model
became better. An earlier research sample also refused to invent historical
claims when retrieval returned no evidence.

Historical structured-output calls exhausted **64K generated tokens with no
usable JSON**. Revised task instructions were followed by **10/10 protocol
checks**. Seven comparable cases improved from 2,446 to 1,565s, but a control
also improved from 24.4 to 19.8s, limiting causal attribution. HTTP 503 overload
errors are provider reliability observations; separate 145/173s worker outliers
show high latency without identifying its cause.

### Cloud code drafts and final answers

In **34 historical task-runs** with the same four-model lineup, the final
writer often retained lines from DeepSeek's draft. The metric below is the
fraction of significant final-code lines also present in each draft.

| Draft model | Mean final-line overlap | Highest-overlap runs, ties included | Mean draft duration |
|---|---:|---:|---:|
| DeepSeek V4 Pro | 0.714 | 20/34 | 9.644s |
| MiniMax M3 | 0.576 | 6/34 | 21.132s |
| Kimi 2.7 Code | 0.515 | 8/34 | 14.379s |
| Local Nemotron Lightning | 0.375 | 3/34 | 25.259s |

Nemotron supplied no code in four runs. Prompts/options varied; final code is
not independent ground truth. Overlap measures resemblance, not standalone
correctness or proof that a draft caused the final answer.

A separate five-task cloud-only code run produced **5/5 executable final
answers**. Kimi 2.7 had the highest overlap in 4/5, GLM 5.2 in 1/5; mean draft
times were **11.6/3.68s**, with mean overlaps **0.802/0.606**. Kimi's draft and
the final answer nevertheless incorrectly claimed that Python's default
`asyncio.gather` cancels remaining tasks after an error. Passing code does not
certify the explanation alongside it.

### GLM 5.2 writing—historical version

Five paired research tasks compared a GLM 5.2 writer with Qwen 3.6. The GLM
arm was faster in **4/5**, by **29–133s**, and manual review favored its headings,
rendered math, tables, and explanatory framing. These are full-system times,
one draw per task/arm, rather than isolated writing-speed measurements.

GLM dropped an unnecessary hedge about tungsten, but hedged a settled vaccine
authorization fact after upstream evidence was mislabeled. Another failure
involved structured-output schemas referencing nested definitions; expanding
those references before generation repaired the interface. Both findings show
that surrounding instructions and evidence representation affect model output.
Neither establishes GLM 5.3's quality.

### Kimi K3 and GLM 5.3: capability evidence

An external client assessment received October 6 reported six successful
isolated checks per model: completion, function selection, using a function
result, streamed text, dispatching a file write, and answering after local
execution. Single-observation durations are retained below.

| Check | Kimi K3 | GLM 5.3 |
|---|---:|---:|
| Basic completion | 1.19s | 1.77s |
| Function selection | 2.04s | 0.45s |
| Answer using function result | 1.84s | 0.61s |
| Streamed text | 2.23s | 0.75s |
| File-write dispatch | 2.11s | 0.47s |
| Answer after local execution | 1.63s | 0.44s |

Prompts and token counts differed; repeats, controlled settings, and raw event
payloads were not supplied. These timings establish neither a speed ranking
nor broad answer quality. Streamed text does not establish streamed function
arguments. Manual cross-model history replay reportedly worked, but an
unavailable-model error followed by manual retry does not test real outages
or automatic prevention of duplicate execution.

Installed metadata on October 7, Ollama 0.35.1, advertised:

| Model | Capabilities | Advertised thinking values | Default |
|---|---|---|---|
| `kimi-k3:cloud` | Text, vision, tools, thinking | `false`, `low`, `high`, `max` | `max` |
| `glm-5.3:cloud` | Text, tools, thinking | `low`, `high`, `max` | `max` |

Metadata describes available controls, not whether every client forwarded them.
Neither advertised `medium`; GLM did not advertise an off value. Repeated prose,
coding, research, and vision comparisons for these versions remain absent.

### Models without sufficient retained measurements

No standalone quality benchmark was located for GLM 5.3 Flash, DeepSeek V4.1
Flash, Nemotron 3 Super/Ultra, MedGemma, Llava 34B, Granite 3.2 Vision, or CLIP
image retrieval. Earlier-version measurements cannot fill those gaps.
Qwen 3.5 397B's September 29 request returned HTTP 410 with reported retirement
on September 25—dated availability evidence, not an answer-quality result.

## 5. Images, summaries, audio, and retrieval

### Qwen3-VL 32B: prompt and budget effects

In one six-frame video-description workload, replacing a screenshot-inventory
prompt with a video-specific prompt plus speech hints changed total time
**291.1→135.7s**, generation time **254.5→98.6s**, and tokens **9,486→3,685**.
Descriptions better covered visible company/title text. Both instructions and
input hints changed, so this is one combined intervention, not an isolated
prompt experiment or general vision benchmark.

Historical probes found that `think:false` and `/no_think` did not materially
reduce hidden thinking: **93–101% of baseline**. Three of six prompt variants
produced no visible answer at 2,048 tokens; the historical budget was increased
to 4,096. Static-scene and cluttered-desk examples used roughly 850 versus
5,700 reasoning characters. Control behavior may differ with newer runtimes.

Separate document-plus-image application checks successfully used both inputs.
A remote PDF/image answer exhausted a 4,096-token limit; its 8,192-token retry
completed with **1,670 output tokens** and the expected document text/logo.
Those receipts identified the application endpoint rather than a concrete
vision model, so they are system/budget evidence, not direct Qwen3-VL scores.

### Short video summaries

One synthetic transcript plus visual-description block, **not an uploaded
video**, was summarized in two or three sentences under a 240-token cap.
Qwen 3.8 and Ornith 1.5 35B produced complete natural summaries; Muse stopped
mid-sentence. DeepSeek V4 Pro and Kimi 2.6 also completed three-sentence outputs.
No model timing was recorded. This supports a narrow summary-format choice,
not general video understanding or a reason to reject Muse for longer writing.

### Whisper transcription

One CPU run using **Whisper small, int8, two cores** processed **565 seconds of
audio in 74 seconds**, about **7.6× faster than real time**. No word-error-rate
or reference-transcript accuracy benchmark was retained. Successful ingestion
and transcription do not establish accuracy.

### Nomic text embeddings

Embeddings represent text numerically to retrieve passages with similar meaning.
One idle probe of `nomic-embed-text:latest` observed **4.18s cold / 0.059s warm**;
the recorded model size was 137M parameters and 323 MB.

In one geology collection, five results for one geology query scored
**0.570–0.590** cosine similarity; results for one unrelated vaccine query scored
**0.487–0.519**. A **0.53 cutoff** separated those examples, but is not universal
across collections or queries.

Another query found its passage through a ten-word paraphrase at **0.796**,
while a six-word verbatim phrase returned no result. The correct passage scored
**0.460**, only slightly above an unrelated passage's **0.431**. Combining
literal-word search with meaning-based search was proposed in response. Those
older observations did not retain a comparative benchmark; the October 15
comparison below supplies one. The small score gap does not establish a
reliable cutoff across queries.

### EmbeddingGemma 2: stronger retrieval in a small test

The October 15 comparison used **32 synthetic English passages, 24 answerable
questions and four separate no-answer diagnostics**, repeated ten times.
These are 28 unique questions, not 280 independent examples. All rankings
were identical across repetitions; all **855 embedding requests** succeeded.

Three formats separated model quality from input formatting: raw Nomic,
Nomic with `search_document:` / `search_query:` prefixes, and Gemma with
`title: none | text:` / `task: search result | query:` prefixes. Actual titles
were omitted for every arm. Meaning-based search used exact cosine ranking;
hybrid search combined it with keyword ranking through reciprocal rank fusion.
No stored vector index or rejection threshold was involved.

| Measurement | EmbeddingGemma 2 | Nomic prefixed | Nomic raw |
|---|---:|---:|---:|
| Correct first result, meaning-based | **24/24** | 21/24 | 21/24 |
| Correct first result, hybrid | **24/24** | 23/24 | 23/24 |
| Correct result within first three | 24/24 | 24/24 | 24/24 |
| Warm query median / p95 | **9.3 / 11.7 ms** | 31.3 / 33.6 ms | 31.5 / 34.2 ms |
| Observed cold query | 3.95 s | 6.71 s | 2.19 s |
| GPU memory while resident | 503 MB | 336 MB | 336 MB |
| Corpus embedding throughput | 69.7 passages/s | 6.5 passages/s* | 100.3 passages/s |

Both Nomic formats misplaced three meaning-based results: starter readiness
versus oven instructions, the adopted Q2 policy versus an earlier Q1 policy,
and a trial **without docking** versus one **with docking**. Hybrid search
corrected the first two. Gemma ranked all three correctly, although its
docking distinction had only a **0.0012 cosine margin**. Both models got all
six code-retrieval questions right. Prefixes gave Nomic no ranking-accuracy improvement
on these cases.

Gemma's warm queries were about **3.4× faster**, with about **168 MB more GPU
memory**. This does not establish faster ingestion. *Prefixed Nomic's first
eight-passage batch took 4.76 seconds and dominated its aggregate; later batches
were much faster.* Only four batches were measured per arm. Cold samples
include loading after unload, without clearing filesystem caches; concurrent
large-model workloads were not tested.

**Unsupported results still need explicit handling.** For an absent 2023
edition, Gemma's nearest hit was the wrong 2025 edition at **0.8364** cosine.
Some correct positive hits scored lower, down to **0.7892**. These diagnostics
measured nearest matches, not rejection accuracy. Existing Nomic cutoffs cannot
be copied to Gemma; source-year filters and reading the actual evidence remain
necessary.

The tested Gemma package was **1.3 GB, Q8_0**, parent tag `740m-q8_0`, digest
`f4981a15b998`; Nomic was **274 MB, F16**, digest `0a109f422b47`.
All actual outputs were finite, approximately unit-length **768-dimensional
vectors**. Gemma's installed metadata instead listed 512 dimensions and a
271M parameter count; use the measured API output when planning an index. No long
context, image, audio or personal-memory retrieval was evaluated.
[Ollama's package page](https://ollama.com/library/embeddinggemma-2) and
[Google's model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en)
describe the model family and retrieval formatting.

**Initial screening recommendation:** validate Gemma on held-out real documents
and calibrate rejection rules before replacing an existing embedder. The completed
follow-up below narrows this recommendation. Its query speed gives
no reason to seek a smaller variant for latency yet. Adoption requires fresh
embeddings in separate indexes: equal dimensions do not make vectors from
different models compatible. The complete comparison, including model digests,
fixture checksum and per-question rankings, is in the
[saved results](results/2026-10-15-083742-embedding-retrieval-results.json).

### Real-document follow-up: Gemma 2 and Qwen3 Embedding

A larger October 15 comparison used **336 passages from four official IRS
publications**, with 60 answerable and 20 unsupported questions. Two whole
publications supplied calibration examples; two different publications supplied
the 30 answerable test questions below. Invented memory/conversation records
and the earlier general/code fixture brought each model's workload to **416
passages and 156 unique queries**, repeated twice. Repetitions check consistency;
they do not double the independent evidence.

All six formats completed. Nomic ran with raw text and its recommended
query/document prefixes; Gemma used its own prefixes; Qwen used an instruction
on queries and raw document text. Meaning-based ranking used exact cosine
similarity; hybrid ranking combined it with keyword search using the same
reciprocal rank fusion for every model. No production index or private records
were touched.

Source review corrected **33 missing alternative answer passages across 27
questions**. Every model was scored against the same repaired labels, without
changing source text, questions or saved rankings. A coding assistant checked
these alternatives after the first run. This is retrospective annotation, not
an independent expert assessment or validation of tax-return accuracy.

| Model | Correct first, dense | Dense hit within three | Correct first, hybrid | Hybrid hit within three |
|---|---:|---:|---:|---:|
| Nomic raw | 12/30 | 22/30 | 16/30 | 27/30 |
| Nomic prefixed | 22/30 | 26/30 | 22/30 | 27/30 |
| Gemma 2 | 22/30 | 28/30 | 23/30 | 27/30 |
| Qwen3 Embedding 0.6B | 25/30 | 29/30 | 23/30 | 29/30 |
| Qwen3 Embedding 4B | 25/30 | 30/30 | 25/30 | 29/30 |
| Qwen3 Embedding 8B | 25/30 | 30/30 | 25/30 | 29/30 |

**Reference retrieval favors Qwen 4B.** Against correctly prefixed Nomic, it
improved three hybrid first results without losing any previously correct first
result, and recovered two more questions within three results. Qwen 8B matched
these totals with greater runtime and storage costs; that does not establish
equal quality on other documents. Gemma's larger advantage in the earlier
synthetic screening narrowed here: one extra hybrid first result, with the
same top-three total as prefixed Nomic.

**Other workloads do not justify replacing every embedder.** On 16 invented
memory/history test questions, dense first-result counts were **14** for raw
Nomic, **14** for Gemma, and **16 / 15 / 16** for Qwen 0.6B / 4B / 8B.
The general/code fixture gave Gemma **24/24** dense first results, versus Nomic
**21/24** and Qwen **21 / 22 / 22**. These small fixtures do not establish
private-memory accuracy or a universal winner.

| Model | Warm query median | Cold query median | Embed 416 passages | Resident GPU memory | Vector width |
|---|---:|---:|---:|---:|---:|
| Nomic raw | 33.1 ms | 2.19 s | 3.8 s | 0.34 GB | 768 |
| Nomic prefixed | 33.5 ms | 2.20 s | 3.9 s | 0.34 GB | 768 |
| Gemma 2 | 9.2 ms | 3.82 s | 10.4 s | 0.50 GB | 768 |
| Qwen 0.6B | 10.8 ms | 3.98 s | 12.6 s | 5.80 GB | 1,024 |
| Qwen 4B | 17.9 ms | 5.72 s | 29.6 s | 8.78 GB | 2,560 |
| Qwen 8B | 26.2 ms | 8.02 s | 43.3 s | 11.01 GB | 4,096 |

Cold medians use three verified unload/load samples. Every Qwen 4B/8B cold
sample exceeded four seconds; Qwen 0.6B's first sample took 14.89 seconds.
All models completed the larger 64-passage, approximately 1,000-token batch
checks without truncation. Larger vectors increase index storage independently
of model residency.

GPU figures include the loaded runtime: **2,048 context for Nomic, 32,768 for
the others**. They are not package sizes or minimum memory requirements; smaller
Qwen contexts were not measured. Nomic's continuation used the same fixtures,
model digests and Ollama version as the candidate run, but ran separately.
Concurrent answering-model load and sustained service capacity were not tested.

**No model passed the strict unsupported-query gate.** Calibration required
retaining at least 90% of answerable questions while rejecting every unsupported
question; none met that target. With thresholds selected only on calibration
data, the reference test retained **27/30** positives for prefixed Nomic and
Gemma, and **29/30** for each Qwen. Related material was admitted for **6/10,
6/10, 5/10, 2/10 and 2/10** unsupported questions, respectively. These are
retrieved passages, not observed fabricated answers. A related passage may omit
the requested year, amount or exception; filters and reading the cited evidence
remain necessary. Completed measurement is not a passed promotion gate.

**Recommendation:** Qwen 4B is the strongest document-retrieval choice in these
tests. Qwen 8B's extra cost bought no improvement on the headline counts; Gemma
remains attractive for fast warm queries but did not establish a broad quality
upgrade. Personal recall needs separate evidence: the invented memory fixture
retained only 6/8 positives at the calibrated cutoff, and the real-history
follow-up below favors Nomic. Qwen also indexes more slowly and uses more GPU
memory. Rebuild an index when changing its model or input format; equal vector
dimensions do not make embedding spaces interchangeable.

**Personal-retrieval follow-up, October 16:** two real stored records were
missed despite correct ownership: a short saved fact scored **0.516** against a
**0.60** cutoff, and a previous arithmetic exchange scored **0.403–0.433** against
**0.62**. The small synthetic calibration did not adequately cover deployed
personal recall. Separate word-matching and ranking defects also caused misses;
repairing those passed exact lookup, a new topic and an absent-topic check, but
an arithmetic paraphrase still failed.

A [matched follow-up](results/2026-10-16-history-embedding-comparison.json)
embedded the same **48 visible conversation chunks** with each model, entirely
in RAM. It labeled the original exchange rather than later questions about it.
Three actual failed search queries and one literal control produced these ranks
for that original exchange (lower is better):

| Query | Nomic raw | Nomic prefixed | Qwen 4B |
|---|---:|---:|---:|
| Paraphrase 1 | 2 | 2 | 4 |
| Paraphrase 2 | 1 | 1 | 4 |
| Paraphrase 3 | 2 | 3 | 7 |
| Literal control | 1 | 1 | 4 |

**Nomic ranked this exchange better in all four comparisons.** Qwen's original
scores were **0.32945, 0.37908, 0.25879 and 0.43342**, all below the then-deployed
**0.62** cutoff. Lowering that cutoff alone would still miss the seventh-ranked
result at a five-result limit. Nomic kept the original within three for every
query; its raw/prefixed ranking differences do not establish a preferred format.
Cosine scores from different models are not comparable quality measures, and no
Nomic admission threshold or unrelated-query rejection was tested in this run.

This is a demonstrated ranking regression on **one positive exchange**, not four
independent facts or a broad personal-memory benchmark. It supports using
different embedders for documents and conversation history rather than assuming
a universal upgrade. It does not justify undoing Qwen's measured document gains
or changing already accepted saved-memory behavior.

### Matched document double-check, October 17

A [matched comparison](results/2026-10-17-document-embedding-comparison.json)
retested Qwen 4B against both Nomic formats in one run. It reused the four
publications and 80 questions above, comparing **336 page-based reference
chunks** with **168 upload-style chunks** of 1,000 tokens and 100-token overlap.
Upload chunks can cross pages. Calibration still used different publications
from the test set. The same 30 answerable test questions appear in each layout;
this tests layout sensitivity, not 60 independent examples or new subject areas.

Each cell gives **correct passage first / within three**, out of 30, before
threshold filtering:

| Layout and retrieval | Qwen 4B | Nomic prefixed | Nomic raw |
|---|---:|---:|---:|
| Page-based, dense | 25 / 30 | 22 / 26 | 12 / 22 |
| Page-based, hybrid | 25 / 29 | 22 / 27 | 16 / 27 |
| Upload-style, dense | 25 / 29 | 14 / 22 | 15 / 22 |
| Upload-style, hybrid | 26 / 28 | 19 / 28 | 20 / 28 |

**Keep Qwen for documents and Nomic for conversation history.** Qwen ranked
upload evidence substantially better than correctly prefixed Nomic: 15 improved
dense ranks versus two worse; nine improved hybrid ranks versus two worse.
Hybrid upload retrieval tied within three, so that gain is earlier evidence,
not higher recall at every depth. Raw Nomic also trailed on first results.

**Unsupported-detail rejection remains unresolved.** No model/layout met the
calibration target. The fallback settings prioritize retaining supported
questions; they are not validated deployment settings. On the ten unsupported
test questions per layout, these settings admitted related but insufficient
material for Qwen **2/10 page-based, 5/10 upload-style**, versus prefixed Nomic
**6/10 and 9/10**. The existing Qwen upload policy admitted **2/10**; do not
replace it with the test's fitted lower cutoff. These are retrieval errors,
not observed fabricated final answers.

All **581 embedding requests completed without errors**. Warm query medians
were **18 ms for Qwen and 33–34 ms for Nomic** under uncontrolled current
residency; no new cold-load or ingestion-speed comparison was made. Both serving
models were observed resident afterward. Ranking used exact in-memory retrieval
before per-file diversity filtering, not a deployed vector-index or answer test.
The reused labels retain their retrospective-review limits. No production
settings or indexes changed; exit 1 denotes the quality findings above.

The full [combined comparison](results/2026-10-15-embedding-documents-combined.json.gz)
and [completed Nomic baselines](results/2026-10-15-140040-embedding-documents-baselines.json.gz)
retain model identities, settings, timings and per-question rankings. The
[original candidate measurements](results/2026-10-15-132341-embedding-documents-results.json.gz)
and [annotation repair](results/2026-10-15-132341-embedding-documents-rescored.json.gz)
preserve the original evidence and correction provenance.

## How to use these results

Match the evidence to the intended job: test the model's tool choices for a
tool-driven application, its source fidelity for document questions, and its
executable output plus explanations for coding. Include loading time and failed
requests when estimating service performance.

Keep these limits visible:

- Passing format or keyword checks does not establish factual correctness.
- Single passes and reused prompts do not establish broad failure rates.
- A larger context, higher precision, or longer reasoning is not automatically
  better; evaluate the failure modes that matter to the workload.
- A missing reader result, rejected output, provider error, and wrong answer
  need separate counts and explanations.
- Reported API capabilities, successful calls, and configured model choices
  do not substitute for quality measurements.

## Sources and traceability

This revision includes the October 15 router and embedding comparisons,
retrospective source-label repairs, October 16 personal-retrieval diagnostics,
the October 17 matched document-layout comparison, the October 22 supplied-
source worksheet check, and the November 27 cloud bake-off and router comparison.
The 48-chunk history
comparison covers one positive exchange and no negative-query gate; its report
contains no source conversation text or user identities. The document follow-up
retains the supplied compact JSON with model identities, fixture hashes, paired
changes and aggregate results; it is not a full per-query trace. No additional
model calls were made during report writing.
Findings come from saved
answers, machine-readable test results, runtime diagnostics, and attributed
client observations. Older test evidence without full controlled settings is
retained with its limitations. The following sources support the sections above;
the report can be read without opening them.

- **Local answers/code:** August 19 saved general and supplied-text on/off
  runs; August 17 code-format probe; August 25 completed quantization runs.
  Primary general runs: `2026-08-19-182113-ab-r5b-onbox-answers.md` and
  `2026-08-19-205355-ab-off-r5b-onbox-answers.md`; supplied-text runs:
  `2026-08-19-155904-gap-r5b-onbox-answers.md` and
  `2026-08-19-213413-gap-off-r5b-onbox-answers.md`.
- **Narrative comparison:** separate security-report results dated September
  30–October 1, run IDs `model-2026-09-30-200843`, `model-2026-10-01-080608`,
  and `model-2026-10-01-142858`; associated result/narrative files.
- **Conditional-rule checking:** [October 22 fixed-passage worksheets](results/2026-10-22-tax-review-fixed-passage-results.json),
  retaining all six initial/final structured outputs, reviewer decisions,
  fixture/model identities and measured costs; the
  [targeted eligibility followup](results/2026-10-22-tax-review-eligibility-followup-results.json)
  retains the subsequent partial improvement and its remaining output/test defects.
- **Classification:** [September results](results/2026-09-30-systemone-router-probe-results.json),
  [October Clef comparison](results/2026-10-03-clef-systemone-router-results.json),
  [Nimble v3 three-round comparison](results/2026-10-15-082632-nimble-v3-router-results.json),
  [Nimble v3 five-round comparison](results/2026-10-15-082843-nimble-v3-router-results.json),
  [November AutoTrust JEV comparison](results/2026-11-27-085742-autotrust-jev-router-results.json),
  [broader Clef decisions](results/2026-10-03-clef-systemone-decision-results.json),
  and [historical workflow-selection study](results/2026-10-06-skill-selection-v3-prospective-full.json).
- **Hosted writing/documents:** [July writer comparison](results/2026-07-14-writer-ab-report.md),
  [September document answers](results/2026-09-28-grounded-documents-reader-fix-answers.md)
  and [results](results/2026-09-28-grounded-documents-reader-fix-results.json).
  Draft-overlap analysis used eight August 17 saved runs with a constant lineup;
  current-version protocol observations came from an external client assessment.
- **Media/retrieval:** [October 15 embedding comparison](results/2026-10-15-083742-embedding-retrieval-results.json),
  dated August runtime receipts, September synthetic summary probe, and
  October operator-supplied capability/API diagnostics.
  Some historical receipts lack retained raw output; these carry less evidence
  than the saved result files.

Some local archives and the separate narrative-study artifacts are not bundled
with this document. Essential measurements are reproduced here; their source
identifiers permit follow-up without implying that all raw data are public.
