# Language model field report

**Results, strengths, and failure modes from practical tests**

Updated October 8, 2026 · Measurements collected July–October 2026

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
- **Qwen 3.5 4B was the best fit for lightweight request classification on the
  tested hardware.** Clef classified well but required much more memory and
  loading time; the smallest Tev model frequently escalated simple requests.
- **Evidence for current Kimi K3 and GLM 5.3 answer quality remains limited.**
  Successful function calls do not establish good writing or reasoning, and
  older Kimi/GLM results cannot be transferred to their successors.
- **Output budgets and supplied evidence often mattered as much as model
  choice.** Hidden thinking could consume the answer budget; a broken document
  reader could make a capable model appear unable to answer.

## Test setup and terms

Local tests used a host with **two RTX 3090 Ti GPUs, 24 GB each**. Runtime memory
needs exceeding the combined 48 GB require slower main memory or cause failure.
Even models that fit can compete with other loaded models. Hosted-model timings
also include network and provider conditions. September–October decision tests
used **Ollama 0.35.0–0.35.1**, software that serves models locally and through hosted tags.
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
literal-word search with meaning-based search was proposed in response; no
comparative retrieval benchmark was retained. The small score gap does not
establish a reliable cutoff across queries.

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

No new model calls were made for this report revision. Findings come from saved
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
- **Classification:** [September results](results/2026-09-30-systemone-router-probe-results.json),
  [October Clef comparison](results/2026-10-03-clef-systemone-router-results.json),
  [broader Clef decisions](results/2026-10-03-clef-systemone-decision-results.json),
  and [historical workflow-selection study](results/2026-10-06-skill-selection-v3-prospective-full.json).
- **Hosted writing/documents:** [July writer comparison](results/2026-07-14-writer-ab-report.md),
  [September document answers](results/2026-09-28-grounded-documents-reader-fix-answers.md)
  and [results](results/2026-09-28-grounded-documents-reader-fix-results.json).
  Draft-overlap analysis used eight August 17 saved runs with a constant lineup;
  current-version protocol observations came from an external client assessment.
- **Media/retrieval:** dated August runtime receipts, September synthetic
  summary probe, and October operator-supplied capability/API diagnostics.
  Some historical receipts lack retained raw output; these carry less evidence
  than the saved result files.

Some local archives and the separate narrative-study artifacts are not bundled
with this document. Essential measurements are reproduced here; their source
identifiers permit follow-up without implying that all raw data are public.
