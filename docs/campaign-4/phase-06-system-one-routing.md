# Campaign 4 Phase 06 — System One router assessment

**Status:** Router assessment is closed; retain `qwen3.5:4b`. All four embedding
candidates completed the deeper test. Nomic's two arms stopped during cold-load
verification; only those baselines need continuation. Candidate measurements are
preserved and corrected labels were rescored offline. Conditional adoption and
reindexing remain approved; the IRS index has never run and still waits.

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

The completed candidates are retained; do not rerun them. The two-arm Nomic
continuation needs at most 690 embedding requests. Model unload is now verified
with bounded polling; unavailable cold samples are disclosed without discarding
valid warm/ranking measurements. Only Nomic is loaded or unloaded in continuation.
Do not run another GPU-heavy job alongside it.

Source review added 33 valid alternative passages across 27 questions. Only labels
and their evidence changed; source/query text and corpus scopes have identical
input fingerprints. Corrected candidate scores are retrospective, with the same
calibration procedure applied solely to corrected calibration labels. Original
and corrected full reports are saved locally; model results are in MODEL-FACTS.

### Nomic-only Tower continuation

After committing/pushing the laptop changes, on **Tower**:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

No service, UI or eval-image rebuild/restart is needed. No model pull, upload,
personal token or IRS index build is required. In that same checkout run:

```bash
scripts/probes/probe-onbox.sh embedding_document_probe.py \
  COPY=embedding_retrieval_probe.py,embedding_real_documents.json,embedding_personal_cases.json,embedding_retrieval_cases.json \
  ARGS=--baselines-only ROUNDS=2 BATCH_SIZE=16
```

Allow roughly 1–3 minutes, depending on model loading. The existing detached
runner sends the result/log through Telegram. Read the attachment on the laptop.
Success: `measurement_status: completed`, exactly `prefixed` and `raw` comparable
arms, and `quality_recommendation.status: baselines_measured`; exit 0 confirms
the missing measurement is complete, not that a model has won. Cold sample status
may be partial/unavailable and must not be presented as verified cold timing.

Use `scripts/analysis/embedding_reports.py` on the laptop to combine the corrected
saved candidate report with this continuation. The helper rejects mismatched
fixtures, query scopes/labels, settings, Ollama version, model digest or format.
It preserves original failures and actual call counts; no candidate rerun or
production change occurs during assembly. Then make the adoption decision.

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
