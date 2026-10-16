# Campaign 4 Phase 06 — System One router assessment

**Status:** Router assessment is closed; retain `qwen3.5:4b`. The embedding
comparison is complete. The user approved replacing Nomic throughout text
retrieval with Qwen3 Embedding 4B and rebuilding the existing small data set.
The unified Tower migration and service restart completed successfully October
15, 2026. October 16 browser acceptance exposed memory/history lookup misses:
correctly owned records existed but fell below semantic cutoffs. Literal lookup
fixes are built. Memory search is user-accepted October 16. History still
failed on full-question wording; its request-word correction is built, awaiting
browser acceptance. The existing-file check and first IRS index remain pending.
Do not repeat the migration.

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

**Use Qwen 4B for all text embeddings, as approved October 15.** The clearest
measured win is reference retrieval; 8B added cost without improving the headline
counts. The personal fixtures are small and show trade-offs rather than a
universal improvement. The user explicitly chose one model across the small
existing corpus and authorized reindexing it. Preserve that distinction between
the deployment choice and the narrower quality evidence.

All arms, including Nomic, failed the strict zero-unsupported-admissions target.
The comparison establishes a relative retrieval improvement, not tax-answer
accuracy or an ability to infer eligibility from similarity. Source/product/year
scope, page reading and Phase 20E accuracy work remain necessary. Measurements
and complete saved reports live only in [MODEL-FACTS](../../evals/MODEL-FACTS.md).
No further comparison run is queued.

## Unified text integration

- `text_embedding` in canonical `config.yaml` owns the tested model/digest,
  native 2,560 dimensions, raw documents, exact Qwen query instruction, residency
  and personal-retrieval cutoffs. Backend, reference indexer and tools sidecar
  share profile/format/vector checks. Retired memory model/dimension environment
  values cannot override the profile; no Tower config or `.env` edit is needed.
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
- Reference cutoff `0.54`, memory `0.60` and history `0.62` come from calibration
  questions; they did not pass the absolute quality gate. Ordinary KB starts at
  `0.53` and retains its scoped-file behavior. Reference lexical matches rank
  results but cannot independently pass admission. Personal recall gets an
  eight-second inner deadline and ten-second outer deadline, plus startup warming
  and 24-hour residency; GPU contention can still cause a clean timeout.

## Tower migration — complete

The completed migration verified the canonical Qwen4 profile across every
selected collection, retained private backups, left source SQLite/images
unchanged and restarted the application services successfully. The later
personal-retrieval repair requires rebuilding `custom-tools` and restarting
`audrey`; New conversation autofocus also requires rebuilding `audrey-ui`.
No repeated migration or new upload is needed.

Retain `/mnt/user/appdata/runtime/embedding-migration-qwen4/` through acceptance.
The recoverable maintenance implementation is `scripts/ops/reembed-onbox.sh`;
private failure tracebacks stay under that state's `diagnostics/` directory.

### Personal retrieval behavior

Memory search checks whole-key spellings with spaces, underscores or hyphens
before semantic fallback. History search combines literal English keyword
matches from the saved archive with semantic results, including recently saved
chunks awaiting vector indexing. Request phrasing such as "what did I ask you
to calculate" reduces to the topic word "calculate". Both paths preserve
owner, deletion, purge and date visibility. Keyword history hits identify their match type and have no
cosine score; semantic hits keep their measured score and existing cutoff.
An unavailable embedder produces a clear search error when no literal history
evidence can answer; it must not masquerade as an empty archive.

### Remaining browser acceptance

After deploying the repair, open **https://ai.builtryte.xyz**, Tools **Automatic**.
Use the same account and the existing saved records; no upload or new memory
write is needed:

Memory search is accepted: the saved migration test phrase was retrieved
correctly. Keep that check closed.

1. In a new conversation ask `Use chat_history_search to tell me what I
   asked you to calculate`. Confirm the tool finds the earlier calculation,
   **1250 × 1.08 = 1,350**, rather than saying nothing was saved.
2. The existing-file check remains unreported: use `kb_search` to find a known
   fact in an existing Ready file and confirm the expected fact and source.

Retest only the failed or unreported surfaces. Keep a failed run for diagnosis;
do not rerun migration or the model comparison. After acceptance, retire these
temporary browser instructions and continue [Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md)
with the first current-catalog IRS index, then 20C.
