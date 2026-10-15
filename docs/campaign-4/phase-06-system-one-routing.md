# Campaign 4 Phase 06 — System One router assessment

**Status:** Retain `qwen3.5:4b` for routing; those comparisons are closed.
The deeper embedding comparison is prepared for Tower: EmbeddingGemma 2 and
Qwen3 Embedding 0.6B/4B/8B against both Nomic baselines. The user approved
adopting an improvement and reindexing the existing text stores before growth.
The IRS index has never run; its first build waits for this decision.

## Decision

Measured `tev1:0.8b`, `tev1:latest`, `nimble:latest`, `nimble-v3:latest`,
`clef:latest`, and `clef-flash:latest` did not clear the complete Audrey routing
gate. Clef matched incumbent quality on production-reached cases but failed cold
latency and residency. Clef Flash failed cold-start reliability. Tev/Nimble
comparisons did not establish a replacement that met the full gate.

Model measurements and publishable comparisons belong in [the model report](../../evals/MODEL-FACTS.md), not duplicate phase logs.

## Conditions for reopening

The probe-local `scripts/probes/systemone_router_probe.py` compares Ollama's typed `/v1/systemone` choices with Audrey's real incumbent router. It adds no production client or dependency.

A replacement must preserve label quality, costly-reasoning misroutes, escalation behavior, cold/warm latency, coexistence with active GPU workers, and failure fallback on Audrey's hardware. System One `confidence` measures distribution concentration; it cannot be copied into the incumbent's confidence/escalation threshold without labeled calibration.

Strong keywords, cheap short-prompt routing, explicit tool-name handling, image compatibility, forced virtual choices, complexity ordering, and weak-keyword/general fallbacks stay authoritative. Any future backend switch needs a configuration rollback.

English-only evaluation applies. Skill-selection research is separate under
Phase 03; native automatic selection uses the approved deterministic rules.

## Deeper embedding comparison

The initial Gemma screening is complete; measurements are in the model report.
The follow-up uses the existing `probe-onbox.sh` and Telegram reporting flow,
with no new runner or production indexing during evaluation.

- Four frozen, official English IRS publications contribute 336 page-preserving
  passages and 80 queries: 60 answerable and 20 unsupported. Source PDFs have
  recorded URLs/digests; labels identify reviewed quotes and pages. Publications
  501/502 calibrate thresholds; 527/969 test them without retuning. Entire source
  families stay in one split. These are source-reading cases, not tax advice.
- Invented memory and chat-history records contribute 48 queries; the existing
  general/document/code/reference fixture contributes 28 reused regression cases.
  No private uploads, saved memories or real conversations are read or reported.
- Every candidate gets its documented query/document format and native vector
  width. Both raw Nomic and prefixed Nomic remain baselines; repeated questions
  add timing/stability evidence, not independent quality examples.
- Measure dense/hybrid ranking, positive support and wrong/unsupported admission
  under current thresholds and thresholds fitted only on calibration. The strict
  diagnostic target is at least 90% positive support, zero unsupported admissions;
  report infeasible calibration honestly. Review per-query outcomes before choosing.
- Measure three verified unload/cold loads per arm, warm median/p95, residency,
  production-sized reference batches and two 64-passage batches at approximately
  1,000 tokens each. Frozen stress inputs avoid runtime tokenizer downloads.
  No concurrent answering worker or end-to-end answer quality is measured here.

The full run has six arms and 2,070 embedding requests within a checked 4,096
request cap. All required models must be installed before the first unload or
embedding request. Candidates run sequentially, are unloaded afterward, and raw
Nomic runs last. Do not run another GPU-heavy job alongside it.

### Tower handoff

After committing/pushing on the laptop, on **Tower**:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

No service, UI or eval-image rebuild/restart is required. The existing runner
copies the probe, helper and frozen fixtures into the running backend.
Gemma and Nomic are already installed. Pull the three requested Qwen models
if not already present; stop if a pull fails:

```bash
docker exec ollama ollama pull qwen3-embedding:0.6b
docker exec ollama ollama pull qwen3-embedding:4b
docker exec ollama ollama pull qwen3-embedding:8b
```

Then run once in that same **Tower** checkout:

```bash
scripts/probes/probe-onbox.sh embedding_document_probe.py \
  COPY=embedding_retrieval_probe.py,embedding_real_documents.json,embedding_personal_cases.json,embedding_retrieval_cases.json \
  CANDIDATES=embeddinggemma-2:latest,qwen3-embedding:0.6b,qwen3-embedding:4b,qwen3-embedding:8b \
  ROUNDS=2 BATCH_SIZE=16
```

Allow roughly 10–30 minutes depending on loading and the larger models' batch
cost. It self-detaches, survives SSH disconnection and sends the usual Telegram
result/log attachment. No upload, personal token, IRS indexing or extra PDF
download is required. Read the attachment from the laptop's Telegram Downloads.

`measurement_status: completed` and all six arms comparable prove execution.
Exit 0 also means a candidate cleared the predeclared diagnostic gates; exit 1
can instead mean completed measurements with quality findings. Read both the
measurement status and per-model recommendation, rather than treating a Telegram
verdict as a model winner. Setup errors exit 2 before model work. No store or
configuration switches automatically.

## Adoption and reindexing

Compare supported retrieval, wrong/unsupported hits and the unchanged regression
first, then latency, cold failures, batch costs and GPU use. A document winner
does not automatically win memory/history. Keep Nomic in a workload where a
candidate regresses. Fit thresholds on calibration only; do not tune on holdout
or infer year/ownership policy from a cosine score. Runtime coexistence and a
targeted browser/tool check remain part of migration verification.

If the results establish an improvement, the user's conditional adoption request
authorizes implementing the switch and a recoverable reindex:

1. Add consistent document/query formatting and explicit model/digest/dimension
   identity across ingest, query and rebuild. References currently special-case
   Nomic prefixes; ordinary KB, memory and history currently embed raw text.
   Qwen's native dimensions differ from the current fixed 768; update schemas
   deliberately. Matching dimensions never make two models' vectors compatible.
2. Inventory and snapshot global/user text collections, memories and durable
   history/upload SQLite state. Keep existing originals, IDs, owner metadata,
   sparse vectors, timestamps and deletion/purge boundaries.
3. Re-embed retained text into separate staging collections; verify counts,
   payload hashes and retrieval before cutover. Pause writers during the final
   switch and change backend/sidecar configuration together where applicable.
   BM25 migration and missing-index reconciliation do not rebuild dense vectors.
4. Verify existing file, memory and history search using the selected model;
   retain recovery snapshots through acceptance. Image collections remain CLIP.
5. Build the IRS current-catalog index for the first time with the selected
   model, then resume Phase 20B.2 acceptance. Do not build a Nomic IRS index first
   or re-download the owned archive.
