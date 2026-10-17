# Campaign 4 Phase 06 — System One router assessment

**Status:** Router assessment is closed; retain `qwen3.5:4b`. Qwen's broad
text migration completed October 15. The history-only Nomic restoration and its
paraphrase/absent-calculation browser checks are user-accepted October 17.
Memory and prior lexical history checks stay accepted. Qwen remains for files,
references and memory. The matched document comparison completed October 17
and supports retaining that split; unsupported-detail rejection remains a limit
for every model. The existing-file browser check and first IRS index remain
pending. Do not repeat either migration or accepted checks.

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

## Embedding decision

The existing probe/Telegram workflow measured Gemma 2, Qwen3 Embedding 0.6B/4B/8B,
and both raw and correctly prefixed Nomic. Four official English publications
provided separate calibration/test source families; invented memory/history and
the existing general/code fixture supplied regression checks. Both Nomic arms
completed in a matched continuation; completed candidates were not rerun.
Source-checked alternative-label corrections were applied to all models, with
unchanged source/query inputs and calibration restricted to its original split.
The model report preserves the retrospective-annotation and separate-run limits.

**Use Qwen 4B for files, references and memories; Nomic for chat history.**
The user approved this split October 16, replacing the earlier shared-model
choice after matched history ranking favored Nomic. The October 17 comparison
supports Qwen for both reference and actual upload chunk layouts; 8B added cost
without improving earlier headline counts.
The personal fixtures are small. The real-history comparison covers one
positive exchange, so the split is a targeted repair rather than a universal
quality claim.

All arms, including Nomic, failed the strict zero-unsupported-admissions target.
The comparison establishes a relative retrieval improvement, not tax-answer
accuracy or an ability to infer eligibility from similarity. Source/product/year
scope, page reading and Phase 20E accuracy work remain necessary. Measurements
and complete saved reports live only in [MODEL-FACTS](../../evals/MODEL-FACTS.md).
The document follow-up is complete; other broad candidate studies stay closed.

## Text embedding integration

- `text_embedding` in canonical `config.yaml` owns Qwen's tested model/digest,
  native 2,560 dimensions, raw documents, exact query instruction, residency and
  memory cutoff. `history_embedding` independently pins Nomic's tested digest,
  768 dimensions, `search_document: ` / `search_query: ` prefixes, eight-second
  request timeout and 24-hour residency. Backend, reference indexer and sidecar
  reuse the same profile/format/vector checks. History requires its own valid
  block and cannot silently inherit Qwen. Retired environment model/dimension
  or history-score values cannot override these profiles.
- History startup verifies its compatible collection and warms its own model;
  discovery and the search route require `history_embedding` independently of
  Qwen's `text_embedding`. One model's warmup failure does not mark the other
  unavailable. Readiness retries independently.
- Every production embedding checks model identity, refuses silent truncation
  and validates vector count, width, finite values and norm. Text collections
  carry explicit profile metadata; incompatible collections fail clearly until
  migrated. BM25 works independently of dense width. Image/CLIP search is unchanged.
- Ordinary global/user collections keep their names, point IDs and payloads.
  Memories preserve keys, tags, ownership and timestamps. History rebuilds from
  every live SQLite chunk, including previously unindexed chunks. File tombstones,
  account purge cutoffs and conversation deletion receipts exclude deleted data.
- SQLite source volumes stay read-only. The maintenance job first copies each
  database and any committed WAL to private writable disk, letting SQLite rebuild
  its own SHM there. Verified retained SQLite backups supply history and deletion
  policy; fresh source snapshots enforce resume/cutover checks.
- The maintenance job backs up SQLite and full old vector/payload exports to
  private Tower state, builds separate staged collections, verifies preserved
  payloads/counts/vectors, then copies to the original names. A journal resumes
  interrupted cutover from verified staging. Source changes invalidate an
  unfinished resume. Completed reruns do not replay old backups over newer data.
- Qwen reference cutoff `0.54` and memory `0.60` come from calibration
  questions; they did not pass the absolute quality gate. Ordinary KB starts at
  `0.53` and retains its scoped-file behavior. Nomic history restores its legacy
  floor `0.40` in its own vector space, replacing Qwen history's `0.62`; this is
  not a new calibrated rejection guarantee. Retrieved text must support the
  requested exchange. Reference lexical matches cannot independently pass
  admission. Personal recall keeps its bounded deadlines, startup warming and
  24-hour residency; GPU contention can still cause a clean timeout.

## Tower migrations

The October 15 broad Qwen migration remains complete. Retain its private
`/mnt/user/appdata/runtime/embedding-migration-qwen4/` recovery state; it cannot
be reused for the new history profile.

The approved history-only replacement uses the existing
`scripts/ops/reembed-onbox.sh --scope history` runner with fresh private state at
`/mnt/user/appdata/runtime/history-reembed-nomic/`. It backs up source history
and the old collection, stages every live SQLite chunk with Nomic, verifies
payloads/vectors/profile and source stability, then replaces only the history
collection. Files, global KB, memories, images and original SQLite are not
rewritten. Owner/deletion/purge/retention exclusions remain authoritative.
Interrupted cutover resumes from verified staging; scope/profile/collection
changes refuse journal reuse. The all-data CLI remains available and now honors
both profiles rather than forcing history into Qwen.

The history-only migration and service restart are user-accepted October 17.
Retain private recovery state through the remaining file acceptance; its safe
report and diagnostics remain on Tower. No migration rerun is queued.

### Personal retrieval behavior

Memory search checks whole-key spellings with spaces, underscores or hyphens
before semantic fallback. History search combines relevance-ranked English
word matches from the saved archive with semantic results, including recently
saved chunks awaiting vector indexing. SQLite FTS5 handles word forms such as
calculate/calculations and whole-token matching before result limits. The
derived index backfills existing chunks once and stays synchronized with source
writes. That lexical index requires no vector reindex; the separately approved
Nomic space change does. Request phrasing such as "what did I ask you
to calculate" reduces to the topic word "calculate". Both paths preserve
owner, deletion, purge and date visibility. Keyword history hits identify their match type and have no
cosine score; semantic hits keep their measured score and the selected
history profile's cutoff.
An unavailable embedder produces a clear search error when no literal history
evidence can answer; it must not masquerade as an empty archive.

### History acceptance — complete

The user confirmed October 17 that the scoped migration and the two browser
checks passed: the earlier arithmetic exchange was retrieved through the history
tool, and an absent calculation was not invented. Other accepted history/memory
checks remain closed. The matched history diagnostic is retained in
[MODEL-FACTS](../../evals/MODEL-FACTS.md); its pre-split probe refuses to label
Qwen's old cutoff as the current history policy.

### Document double-check — complete

The October 17 matched comparison completed without request errors and supports
retaining Qwen4 for documents. The same four public English PDFs and corrected
questions were tested in both reference and actual upload chunk layouts, with
separate calibration publications and paired dense/hybrid rankings. This checks
layout sensitivity, not independent new-topic generalization. It wrote no
production data or settings; both configured models were observed resident
afterward. Results and the supplied compact report are in
[MODEL-FACTS](../../evals/MODEL-FACTS.md#matched-document-double-check-october-17).

Every model/layout still failed the strict unsupported-query calibration target.
Exit 1 represents those quality findings, not a failed measurement. Keep the
existing thresholds; the probe's fallback fitted settings are not a validated
deployment policy. Source scope, page reading and Phase 20E remain necessary.
No comparison or migration rerun is queued.

The existing-file `kb_search` browser check and new conversation focus remain
unreported. After closing the file check, continue
[Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md) with the
first current-catalog IRS index and 20C.
