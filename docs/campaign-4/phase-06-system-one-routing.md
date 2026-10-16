# Campaign 4 Phase 06 — System One router assessment

**Status:** Router assessment is closed; retain `qwen3.5:4b`. The embedding
comparison is complete. The user approved replacing Nomic throughout text
retrieval with Qwen3 Embedding 4B and rebuilding the existing small data set.
Unified configuration and a recoverable maintenance migration are built.
Tower migration recovery is prepared, including read-only WAL source handling.
Migration completion, browser acceptance and the first IRS index remain pending.

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

## One-time Tower migration

Commit/push on the laptop. On **Tower**, build images while the old services are
still running. Do not start the new images before migrating their existing indexes:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose build audrey custom-tools
```

When no answer or upload is in progress, stop the application writers:

```bash
docker compose stop audrey custom-tools media-worker media-fetcher
```

Start the migration from that same Tower checkout:

```bash
nohup bash scripts/ops/reembed-onbox.sh > testing-out/last-text-reembed.log 2>&1 &
```

Chat and uploads are unavailable during this maintenance window. Qdrant and
Ollama stay running. The script automatically restarts backend/tools/media only
after successful verification; the UI needs no rebuild. No new file upload,
model pull, answering-model evaluation, archive collection or personal token is
required. Duration depends on the retained chunk count; allow a long run rather
than treating it as a short smoke. Do not run another GPU-heavy job alongside it.

The existing notification flow sends start/finish messages and the compact report
through Telegram. Download that attachment on the laptop for review. Reports
contain roles/counts/digests, not private text or owner identifiers. Private
backups, exports and the recovery journal stay in
`/mnt/user/appdata/runtime/embedding-migration-qwen4/`; keep them through acceptance.
Success: exit 0, `status: completed`, Qwen4/2,560 in `embedding`, all collections
verified, `source_sqlite_unchanged: true`, and the runner's restart succeeds.
On failure writers stay stopped: retain state and rerun the same command after
diagnosis. Never discard the journal/staging during an unfinished cutover.
The container limits local BLAS/OpenMP to one thread; embedding computation runs
in Ollama. Unexpected failures save a mode-0600 traceback under the private
state's `diagnostics/` directory. The Telegram report carries only the failure
type, safe phase and relative diagnostic filename. Exception text and tracebacks
remain on Tower and should not be attached to public reports.

### Browser acceptance after migration

Open **https://ai.builtryte.xyz**, Tools **Automatic**. No upload is needed:

1. Ask `Use kb_search to find [a fact or topic] in my existing uploaded files`.
   Choose a Ready file you already know. Confirm the expected fact and source;
   the Tools disclosure must show `kb_search`, so a raw file read does not stand
   in for an embedding search.
2. Ask Audrey to remember a harmless new fact, then in a new conversation ask
   her to search saved memories for it. Confirm `memory_store`/`memory_search`
   and the correct value.
3. Ask her to search previous conversations for a known earlier topic. Confirm
   `chat_history_search` and the correct earlier conversation.

If one fails, keep its conversation/run for diagnosis; do not repeat all checks
or rerun the model comparison. After acceptance, retire this temporary migration
handoff and continue [Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md)
with the first current-catalog IRS index, then 20C.
