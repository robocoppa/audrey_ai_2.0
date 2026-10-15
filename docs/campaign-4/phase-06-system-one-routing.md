# Campaign 4 Phase 06 — System One router assessment

**Status:** Earlier comparisons are complete; retain `qwen3.5:4b`. The user's
October 15 Nimble v3 comparison is complete: retain Qwen. EmbeddingGemma 2's
first launches stopped at settings validation; the corrected probe is ready to
run. No production model or index changes are part of the tests.

## Decision

Measured `tev1:0.8b`, `tev1:latest`, `nimble:latest`, `clef:latest`, and `clef-flash:latest` did not clear the complete Audrey routing gate. Clef matched incumbent quality on production-reached cases but failed cold latency and residency. Clef Flash failed cold-start reliability. Tev/Nimble comparisons did not establish a replacement that met the full gate.

Model measurements and publishable comparisons belong in [the model report](../../evals/MODEL-FACTS.md), not duplicate phase logs.

## Conditions for reopening

The probe-local `scripts/probes/systemone_router_probe.py` compares Ollama's typed `/v1/systemone` choices with Audrey's real incumbent router. It adds no production client or dependency.

A replacement must preserve label quality, costly-reasoning misroutes, escalation behavior, cold/warm latency, coexistence with active GPU workers, and failure fallback on Audrey's hardware. System One `confidence` measures distribution concentration; it cannot be copied into the incumbent's confidence/escalation threshold without labeled calibration.

Strong keywords, cheap short-prompt routing, explicit tool-name handling, image compatibility, forced virtual choices, complexity ordering, and weak-keyword/general fallbacks stay authoritative. Any future backend switch needs a configuration rollback.

English-only evaluation applies. Skill-selection research is separate under
Phase 03; native automatic selection uses the approved deterministic rules.


## October 15 candidate comparison

Use the existing direct-model probe runner. These are measured comparisons,
not acceptance of a replacement or an automatic configuration change. Published
specifications and eventual measured results belong in the standalone model
report linked above.

- **Nimble v3:** rerun the 36 tracked English router cases against the real
  configured incumbent for three rounds. Report the actual model-reached
  subset separately from deterministic-rule cases, full failures, confidence
  behavior, cold/warm latency and observed residency. Require Ollama 0.40.4+
  before model calls. The current Qwen also generates conversation titles;
  decision-only Nimble cannot replace that role, so capacity decisions must
  account for both models if Qwen remains necessary. This is a reused feasibility
  fixture, not an independent general classifier benchmark.
- **EmbeddingGemma 2:** compare its documented task prefixes against both raw
  Nomic (ordinary deployed text retrieval) and prefixed Nomic (reference search).
  Use the same 32 synthetic documents, 24 positive English queries and four
  separately reported no-answer diagnostics. Measure dense and keyword-fused
  rankings, query latency, corpus embedding throughput, vector validity and
  residency. Each arm embeds its own corpus in memory; no Qdrant collections or
  stored documents are changed. Existing cosine cutoffs are not transferred to
  the candidate. This screens text retrieval only; it does not validate image,
  audio or personal-memory retrieval.

Both reports retain the installed model digests and fixture checksums. Model
load/unload operations affect only tested tags. Run when the IRS indexer and
other GPU work are idle, and run the two comparisons sequentially. The smaller
EmbeddingGemma text-only variant is a follow-up if the pulled default proves too
slow or memory-heavy; it is not another required run before reading the results.

### Tower commands

After committing/pushing on the laptop, on **Tower**:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

No service, UI or eval-image rebuild/restart is required. The existing runner
copies both probe scripts and their fixtures into the running backend. Both
models are already pulled; Nomic and the current router must also be installed.
No upload, personal token, collector run or new production index is needed.

The router comparison is completed; do not rerun it. Two saved comparisons
support retaining the incumbent; measurements are in the model report.

The embedding launches requested ten and five rounds. Both stopped at the old
three-round limit before any model call. The probe now accepts 1–10 rounds,
with a maximum of 1,024 embedding requests checked before model operations.
The default fixture uses 855 requests at ten rounds or 435 at five.

After pulling this harness fix, run only the embedding comparison in the same
Tower checkout:

```bash
scripts/probes/probe-onbox.sh embedding_retrieval_probe.py COPY=embedding_retrieval_cases.json CANDIDATE=embeddinggemma-2:latest ROUNDS=10
```

It self-detaches and uses the same Telegram workflow; allow a few minutes
depending on loading and host conditions. The candidate runs before both
Nomic arms, leaving the incumbent embedder warm. Read the downloaded Telegram
files from `~/Downloads/Telegram Desktop/` on the laptop; the logs also remain
in Tower's `testing-out/probes/` directory. No separate transfer or rerun is
needed when the attachments arrive.

For both comparisons, successful execution means a completed measurement with
all expected model/arm results. Inspect accuracy/ranking metrics and failures
alongside latency and memory before choosing a model. A completed run is not
proof that the candidate won. A setup error stops before comparison; resolve
that error before continuing.
