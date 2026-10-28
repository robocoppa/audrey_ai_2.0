# Campaign 4 Phase 06 — routing and embeddings

**Status:** Closed by owner October 27, 2026. Router assessment and both embedding
migrations are complete. Retain Qwen3.5 4B routing, Qwen3 Embedding 4B for
files/references/memory and Nomic for history. Measurements and comparison limits
live in [MODEL-FACTS](../../evals/MODEL-FACTS.md); no migration or comparison rerun
is queued. Closure does not assert an unreported existing-file browser pass.

## Decision

The measured System One candidates did not satisfy the complete quality,
cold/warm latency, residency and failure gate on Audrey's hardware. A
confidence concentration value cannot replace Audrey's escalation threshold
without labeled calibration. Production routing order, explicit model choices,
image handling and fallback behavior remain authoritative. The probe creates
no production dependency. English-only scope applies.

## Embedding decision

Qwen4 improved the measured document/reference layouts; Nomic repaired the
measured history ranking. Small personal fixtures and one positive real-history
exchange do not establish universal quality. Every compared model/layout still
admitted some unsupported queries. Similarity cannot establish factual support,
tax-year applicability or taxpayer eligibility.

## Text embedding integration

- Canonical `text_embedding` pins Qwen's digest, query/document format and native
  2,560-dimensional vectors. `history_embedding` independently pins Nomic's
  digest, 768 dimensions and `search_document: ` / `search_query: ` prefixes.
  Backend, sidecar and reference indexing enforce the same profile contracts.
  History cannot inherit Qwen or use retired environment overrides.
- Production calls verify identity, reject silent truncation and validate vector
  count, width, finite values and norm. Collections carry explicit profiles;
  incompatible spaces fail clearly. BM25 is independent of dense width and
  image/CLIP embeddings are unchanged. Each model warms and recovers separately.
- Files, memories and history preserve names/IDs, payloads, ownership and source
  authority. Tombstones, deletion receipts and purge cutoffs exclude deleted
  material. History rebuilds include live SQLite chunks awaiting indexing.
- Maintenance snapshots source SQLite/WAL read-only, uses private writable
  copies, backs up source/vector state and verifies staged payloads/counts before
  cutover. Journals resume only against matching source/profile/scope. Completed
  reruns cannot overwrite newer data with old backups.
- Qwen reference/memory/ordinary-KB dense floors are `0.54`/`0.60`/`0.53`;
  Nomic history uses its own `0.40` space. These are admission policies, not
  validated zero-false-admission guarantees. Reference lexical matches cannot
  independently pass admission. Contention can still produce a clean timeout.

## Tower migrations

The broad Qwen migration and history-only Nomic restoration are complete. Their
private recovery states belong to their respective profiles:
`/mnt/user/appdata/runtime/embedding-migration-qwen4/` and
`/mnt/user/appdata/runtime/history-reembed-nomic/`. They are recovery data, not
campaign handoffs or interchangeable resume journals. History-only maintenance
preserves files, global KB, memory, images and canonical SQLite.

## Personal retrieval behavior

Memory first checks whole-key spelling variants. History combines owner-scoped
SQLite FTS5 word matches with semantic ranking, including live chunks not yet
vector-indexed. It preserves owner, deletion, purge and date visibility. Keyword
hits identify their match type without a cosine score; semantic hits identify
their profile/score. If literal evidence cannot answer and the embedder is
unavailable, retrieval reports a search error rather than an empty archive.
