# Campaign 4 Phase 06 — System One router assessment

**Status:** Router assessment is closed; retain `qwen3.5:4b`. The embedding
comparison is complete. The user approved replacing Nomic throughout text
retrieval with Qwen3 Embedding 4B and rebuilding the existing small data set.
The unified Tower migration and service restart completed successfully October
15, 2026. October 16 browser acceptance exposed memory/history lookup misses:
correctly owned records existed but fell below semantic cutoffs. Literal lookup
fixes are built. Memory search is user-accepted October 16. History still
failed on full-question wording. Expanded regressions exposed word-form,
ranking and substring-crowding gaps; those are fixed locally with a ranked
English text index. Tower SQLite support is confirmed; browser acceptance is
pending. The existing-file check and first IRS index remain pending.
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
before semantic fallback. History search combines relevance-ranked English
word matches from the saved archive with semantic results, including recently
saved chunks awaiting vector indexing. SQLite FTS5 handles word forms such as
calculate/calculations and whole-token matching before result limits. The
derived index backfills existing chunks once and stays synchronized with source
writes; no vector reindex is required. Request phrasing such as "what did I ask you
to calculate" reduces to the topic word "calculate". Both paths preserve
owner, deletion, purge and date visibility. Keyword history hits identify their match type and have no
cosine score; semantic hits keep their measured score and existing cutoff.
An unavailable embedder produces a clear search error when no literal history
evidence can answer; it must not masquerade as an empty archive.

### Remaining browser acceptance

On Tower after pull, rebuild `custom-tools` and restart `audrey`, following the
main plan's deploy table. Existing history receives its derived word index
automatically; no UI build, upload or vector migration is needed.

Open **https://ai.builtryte.xyz**, Tools **Automatic**. Use a fresh conversation
for each lookup and confirm successful `chat_history_search` activity. Memory
search is already user-accepted and needs no retest.

1. Ask `Use chat_history_search to tell me what I asked you to calculate`.
   Expect the original **1250 × 1.08 = 1,350** exchange.
2. Ask `Use chat_history_search to find my earlier arithmetic request. What
   expression did I ask you to work out, and what was the answer?` Expect the
   same original exchange. This tests paraphrased retrieval; passing local
   storage tests does not establish this model-quality result.
3. Seed one disposable chat: `For a temporary chat-history test, the fictional
   lighthouse repair budget is 286 dollars and its inspection day is Tuesday.
   Acknowledge only; do not save this as a memory.` Wait for its answer, then in
   a new chat ask `Use chat_history_search to find my recent lighthouse repair
   discussion. What was the budget and inspection day?` Expect **286 dollars,
   Tuesday** without providing those answers in the lookup question.
4. In a new chat, once, ask `Use chat_history_search to check whether we
   previously planned a ZKQ-7149 volcanology seminar. If there is no matching
   plan, say so without inventing one.` Expect no fabricated prior plan.

Optional: archive only the disposable lighthouse seed and repeat its lookup.
Archived conversations remain searchable; deletion has a separate exclusion
contract covered by hermetic checks. Do not delete real history for this test.

The existing-file `kb_search` acceptance remains unreported. Keep any failed
run for diagnosis and do not repeat passed memory checks, migration or model
benchmarks. Retire these temporary instructions after acceptance, then continue
[Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md) with its
first current-catalog IRS index and 20C.
