# eval run — 2026-09-28-grounded-documents-reader-fix-answers

9 cases, 9 passed all applicable checks. Each section is one case (structural header + answer body).

---

## grounded-doc-compare-skill

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:1.0s  total:8.0s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

I've read both documents in full. Here's the comparison.

## The two measurements

**Operations note** (`c3-grounded-operations.md`) — staging API p95 latency: **840 ms**, measured under 250 concurrent sessions. The note explicitly scopes this: "These figures describe the staging exercise only; they are not production forecasts."

**Support note** (`c3-grounded-support.md`) — median first response time: **18 minutes**, alongside a projected peak of 34 tickets per hour. The note explicitly scopes this too: "These are support queue measurements, not application performance measurements."

## Why they are not the same kind of metric

Despite both being "timing" numbers and both appearing in Alder-release notes, these measure fundamentally different things, and each document itself warns against conflating them:

1. **What they measure is different.** The 840 ms is a *machine* metric — how long the API takes to service a request under load. The 18 minutes is a *human/process* metric — how long it takes a support team to begin responding to a customer's ticket. One is a property of the software; the other is a property of staffing and queue workflow.

2. **Different statistical summaries.** p95 (95th percentile) describes the *tail* of a distribution — the slowest 5% of requests — and is meant to capture worst-case latency. The median describes the *center* of a distribution, and is deliberately insensitive to outliers. Averaging, comparing, or blending a percentile with a median mixes two incompatible order statistics.

3. **Different units and scales by ~4 orders of magnitude.** Milliseconds versus minutes (840 ms ≈ 0.84 seconds versus 18 minutes ≈ 1,080 seconds). Collapsing them onto one chart or one "latency" number would be arithmetically meaningless.

4. **Each file's scope limit explicitly rules the other one out.** The operations note states it "does not measure support demand, staffing coverage, customer sentiment, or customer satisfaction." The support note states it "does not report API latency, application error rate, database rollback duration…" The two authors deliberately partitioned the metrics, so reading them as one combined performance picture contradicts what both documents say.

**Bottom line:** the 840 ms p95 is the speed of the *system* under staged load; the 18-minute median is the speed of the *team's* initial reply under a rehearsal queue. They answer different questions and are documented as mutually exclusive in each note, so they should not be aggregated, compared as like-for-like, or treated as a single "response time" metric.

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `list_my_files` ✅1, `get_file_text` ✅2

---

## grounded-doc-single-skill

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:11.6s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

Here are the three facts you asked for, all from **c3-grounded-operations.md**:

- **Release codename:** Alder
- **Planned production date:** October 14, 2026, beginning at 09:30 Mountain Time
- **Unresolved operational risk:** A possible schema lock on the billing ledger during the migration (the team will pause the rollout if the lock lasts longer than 45 seconds)

The file also notes that Release Engineering owns the rollout decision, and that its scope limit means it does not measure support demand, staffing coverage, customer sentiment, or customer satisfaction.

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `list_my_files` ✅1, `get_file_text` ✅1

---

## grounded-doc-absence-skill

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:32.3s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:✅  calibrated:—  grounded:—  no_reasoning_leak:✅

I read both files in full. Here's the answer, kept separate:

**c3-grounded-operations.md**
No customer satisfaction score is recorded. The note explicitly states under "Scope limit" that it *"does not measure support demand, staffing coverage, customer sentiment, or customer satisfaction."* Its metrics are API p95 latency (840 ms), request error rate (0.7%), and rollback rehearsal time (7 min 40 s) — none of which is a satisfaction score.

**c3-grounded-support.md**
No customer satisfaction score is recorded either. Under "Scope limit" it states it *"does not report API latency, application error rate, database rollback duration, customer sentiment, or customer satisfaction."* Its metrics are median first response time (18 minutes) and a projected peak of 34 tickets per hour — again, no satisfaction score.

**Bottom line:** Neither file records a customer satisfaction score. Both explicitly disclaim measuring or reporting it, so there is no score to report and I'm not inferring one.

---
> _Tools used:_
> - **qwen3.8:latest** — `list_my_files` ✅1, `get_file_text` ✅2

---

## grounded-doc-compare-skill#2

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:5.3s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

Both files surfaced the full notes. Here's the comparison.

## The two metrics

**Staging API p95 latency — from `c3-grounded-operations.md`:** **840 ms**, measured with 250 concurrent sessions (with an observed 0.7% error rate).

**Median first response time — from `c3-grounded-support.md`:** **18 minutes**, alongside a projected peak of 34 tickets per hour.

## Why these aren't the same kind of metric

They describe two entirely different systems and time scales, and each note is explicit that the two domains don't overlap:

1. **Different subject of measurement.** The p95 latency is a property of the *software itself* — how long the API takes to return a response under load in a staging exercise. The first response time is a property of the *human support process* — how long it takes a person (or queue) to make initial contact with a customer. One measures machine behavior; the other measures staffing/queue behavior.

2. **Different orders of magnitude.** Milliseconds vs. minutes — they sit roughly three orders of magnitude apart, which is the clearest signal that no one is comparing them as the same thing. A "slow API" at the 840 ms scale has nothing to do with an 18-minute support response.

3. **Different statistical summaries.** p95 is a high-tail percentile (95th percentile: only 5% of requests were slower than 840 ms), which captures worst-acceptable-case behavior under concurrency. "Median" is the midpoint (half of first responses were faster than 18 minutes), capturing typical behavior. Even the summary statistic itself answers a different question.

4. **Each note explicitly walls off the other's territory.** The operations note's *Scope limit* says it "does not measure support demand, staffing coverage…". The support note's *Scope limit* says it "does not report API latency, application error rate, database rollback duration…". So the authors themselves flag that p95 latency belongs to operations and first response time belongs to support — mixing them would conflate two intentionally separate measurement domains.

In short: the p95 is an *application performance* measurement from a load test; the median first response time is a *support queue* measurement from a rehearsal. They happen to coexist under the same "Alder" release, but treating them as the same kind of metric would collapse machine latency into human response time and vice versa.

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `kb_search` ✅2

---

## grounded-doc-single-skill#2

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:2.4s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

Here are the facts, as stated in **c3-grounded-operations.md**:

- **Release codename:** Alder (from "The internal release codename is **Alder**").
- **Planned production date:** October 14, 2026, beginning at 09:30 Mountain Time (from "The production rollout is planned for **October 14, 2026**, beginning at 09:30 Mountain Time").
- **Unresolved operational risk:** A possible schema lock on the billing ledger during the migration — the team will pause the rollout if the lock lasts longer than 45 seconds (from the "Open risk" section).

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `kb_search` ✅1

---

## grounded-doc-absence-skill#2

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:7.3s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:✅  calibrated:—  grounded:—  no_reasoning_leak:✅

Neither file records a customer satisfaction score.

- **c3-grounded-operations.md** — reports staging API p95 latency (840 ms), a 0.7% request error rate, and a 7 min 40 s rollback rehearsal. Its scope limit explicitly states it does **not** measure customer sentiment or customer satisfaction.
- **c3-grounded-support.md** — reports a median first response time of 18 minutes and a projected peak of 34 tickets/hour. Its scope limit likewise states it does **not** report customer sentiment or customer satisfaction.

So there is no customer satisfaction score in either document, and I won't infer one.

---
> _Tools used:_
> - **qwen3.8:latest** — `list_my_files` ✅1, `get_file_text` ✅2

---

## grounded-doc-compare-skill#3

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:5.6s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

I have both documents in full. Here's the comparison.

## The two measurements

| | **c3-grounded-operations.md** | **c3-grounded-support.md** |
|---|---|---|
| **Metric** | Staging API **p95 latency: 840 ms** | Median **first response time: 18 minutes** |
| **Context** | Final staging exercise, 250 concurrent sessions (also reports 0.7% request error rate) | Support queue rehearsal (also projects 34 tickets/hour peak) |
| **What it measures** | How fast the *application* responds, at the 95th percentile | How quickly a *human support agent* gives the *first* reply, at the median |

## Why they are not the same kind of metric

They look superficially alike — both are "how long something takes" — but they differ on three independent axes:

1. **What is being timed.** The operations figure times a software system: the p95 latency of API responses during a staging load test. The support figure times people: how long until a support responder sends a first reply to a ticket. The operations note even states its own figures "are not production forecasts," while the support note describes its number as "support queue measurements, not application performance measurements."

2. **The percentile statistic.** p95 is a *tail* metric — it reports the 95th percentile, so it characterizes the slow end of the distribution (the worst 5% of requests). The support figure is a *median* (50th percentile), characterizing the typical case. They summarize completely different parts of their distributions.

3. **Units and scale.** 840 milliseconds versus 18 minutes — a roughly 1,300× difference — because one is a machine-level latency and the other is a human response-time process.

Each note is also explicit about its scope boundary: the operations note says it "does not measure support demand," and the support note says it "does not report API latency." So neither document treats its number as interchangeable with the other's — and the two metrics should be reported and compared only within their own domains (application performance vs. support responsiveness).

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `list_my_files` ✅1, `get_file_text` ✅2

---

## grounded-doc-single-skill#3

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.1s  total:2.9s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:—  calibrated:—  grounded:—  no_reasoning_leak:✅

Here's what the file **c3-grounded-operations.md** contains, with each fact attributed to it:

- **Release codename:** Alder
- **Planned production date:** October 14, 2026, starting at 09:30 Mountain Time (Release Engineering owns the rollout decision)
- **Unresolved operational risk:** A possible schema lock on the billing ledger during the migration. The team plans to pause the rollout if the lock lasts longer than 45 seconds.

All three facts come from the document's "Release decision" and "Open risk" sections.

---
> _Tools used:_
> - **deepseek-v4-pro:cloud** — `list_my_files` ✅1, `get_file_text` ✅1

---

## grounded-doc-absence-skill#3

- model: `audrey_auto`
- skill: `grounded-document-analysis`
- status: PASS
- route: fast
- latency: route:fast  ttft:0.0s  total:6.4s
- banners: Thinking
- checks: reachable:✅  no_error_marker:✅  has_answer:✅  not_truncated:✅  not_misattributed:✅  no_fiction:—  banners:—  sources:—  url_wellformed:—  route:✅  code_block:—  code_runs:—  contains:✅  within_word_budget:—  names_files:✅  not_contains:—  continuation:—  disclaims:✅  calibrated:—  grounded:—  no_reasoning_leak:✅

**No customer satisfaction score is recorded in either file.**

- **c3-grounded-operations.md** — This is the Alder release *operations* note. It covers staging latency (p95 840 ms), error rate (0.7%), rollback rehearsal time, and the schema-lock risk. Its scope-limit section explicitly states it "does not measure... customer sentiment, or customer satisfaction."
- **c3-grounded-support.md** — This is the Alder release *support* note. It covers support coverage, median first response time (18 minutes), and projected peak tickets (34/hour). Its scope-limit section also explicitly states it "does not report... customer sentiment, or customer satisfaction."

So neither file records a customer satisfaction score, and both name customer satisfaction as out of scope for that document. I won't infer a figure that isn't there.

---
> _Tools used:_
> - **qwen3.8:latest** — `get_file_text` ✅2
