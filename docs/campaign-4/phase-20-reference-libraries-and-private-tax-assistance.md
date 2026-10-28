# Phase 20 — reference libraries and private tax assistance

**Status:** Closed by owner October 27, 2026. The implemented reference, Private
processing and saved-review workflows remain available. No deployment, browser
acceptance or accuracy run is queued. Closure does not turn the known tax
accuracy failures or incomplete coverage into passes. English requests only.

## Outcome and supported scope

Tax review supports selected official-source review and conditional arithmetic
worksheets in an owned Private conversation/Project. It does not determine
eligibility for deductions or credits, certify full-return coverage, prepare or
file a return. The server projects `scope: evidence_and_arithmetic` and
`eligibility_status: not_determined` on current and historical worksheets.
Completed tasks and checked support assessments do not establish entitlement.

The federal IRS library is the implemented source. State adapters, additional
statutory/regulatory adapters, broad historical search, Private OCR/media and
fine-tuning remain outside implemented scope. Unknown facts, unread material,
source conflicts and coverage gaps stay explicit. Uploaded/indexed material is
not proof that every page was read; a retrieved passage is not an eligibility
assessment. Private records are never training data.

## Reference storage and ownership

The retained archive is under
`/mnt/user/knowledge/reference-libraries/tax/us/federal/irs/`, with `originals/`
and `catalogs/`. Audrey reads through `/datasets/reference-libraries/`;
public indexing state lives under `/mnt/user/appdata/runtime/reference-libraries/`.
The archive is independent of Audrey/Qdrant and is not a general KB watcher input.
Indexes are rebuildable derivatives. Public references and private taxpayer
records have separate inventories, indexes and searches; public updates receive
neither private documents nor personalized queries.

Documents retain official URL, publisher, product/title, checksum, revision and
retrieval metadata, page/section and known applicability/status. Revision and
download dates do not establish a return year. Unknown applicability stays
unknown. Historical originals remain retained, while model access defaults to
current official editions and explicitly requested recent history within the
configured five-year window. A current publication may have an older revision.

## Component contracts

### 20A — archive collection

The collector enumerates official catalogs, validates allowlisted URLs,
redirects, signatures and hashes, and retains changed content as new revisions.
Bounded retries, resource limits and durable SQLite checkpoints support resume
without deleting prior originals. Missing years, ambiguous metadata and failed
sources are reported as gaps. Final report export is not archive authority.
Collection is sufficient; no campaign rerun remains.

### 20B — named libraries

A server-owned registry maps each library/release to authorized roots and its
own text collection/embedding space. Models choose stable document IDs, never
paths or arbitrary collections. Catalog lookup, page reading and content search
have distinct coverage; they never fall back to unrelated/private collections.
Results identify edition, release, exact page/excerpt and extraction limitations.

`reference_lookup` and `read_reference` enforce active catalog membership,
permitted history and revision checks. Unknown/removed IDs or ambiguous historical
years refuse. Reads use bounded isolated PDF extraction; blank/scanned pages do
not gain OCR coverage. Actual returned passages produce citations; catalog
metadata and gap descriptions cannot become passage evidence.

`reference_search` uses the pinned local Qwen3 Embedding 4B profile and Qdrant
keyword/vector fusion with identical scope filters. Publication switches only
after exact document/vector accounting; failed builds preserve the prior release.
Staging identities prevent overwrites of the published vectors. Compatible
checkpoints resume; changed profiles, source hashes or schemas fail clearly.
Inputs never silently truncate. Dense cutoff policy is measured retrieval
admission, not a tax-accuracy guarantee; lexical matches cannot independently
pass admission.

Return-year filtering requires a retained, unambiguous source statement with
page/checksum. Isolated years and revision dates do not establish applicability.
Other documents remain unknown and are excluded from year-filtered search.
Private owner collections and public reference indexing never share content.

The accepted current-catalog publication accounts for 2,407 selected documents:
2,406 indexed, 62,704 verified chunks and one approved extraction gap,
Publication 5902. Its original remains available; search responses disclose the
incomplete content coverage. Exact gap queries return no substitute passage.
Only exact operator-approved failures can publish as gaps; pending documents,
unapproved errors and mismatched vector counts block publication. Retained
prior editions do not imply that a prior-edition content index is published.

### 20C — private processing

Private uses installed verified local models for routing, planning, answers,
summaries, embeddings and supported document processing. It refuses cloud
fallback, remote file/image inputs and external search/fetch. Local-service
failure reports unavailability. Browser transport still follows the configured
route, including Cloudflare, and the server administrator is trusted. The badge
states processing, storage, transport and mode-lock boundaries without adding
an acknowledgement gate. No offline-browser or storage-encryption claim follows.

`ProcessingPolicy` is an immutable server-owned task scope. Nested scopes may
only tighten it; full streaming, child tasks, cancellation and teardown retain
it. Each content call checks an exact trusted endpoint, approved model/digest,
installed metadata and verifiable local weights. Clients disable environment
proxy inheritance and redirects. Missing/changed identities refuse before
content dispatch. The daemon/administrator are trusted; these checks cannot
atomically prevent a concurrent alias swap.

Tool admission checks actual endpoint/path before argument parsing/logging.
Native tools use expiring owner-bound in-process callbacks with no sidecar HTTP
fallback. Supported operations include calculator, CSV/XLSX analysis, owned file
listing/text, Private hybrid document retrieval and IRS lookup/search/page reads.
Web, general memory/history and media tools are unavailable. Logs remove private
contents, arguments and traces; task-local tracing is disabled.

Durable `private_required` restrictions on Projects, conversations, runs, upload
reservations and files cannot be weakened by a client, model, worker result,
Project move/deletion, retry or requeue. The server derives them from canonical
owned state. Only empty ordinary conversations can enter a Private Project;
previously processed content is never relabelled as having been Private.
Admission and case writes recheck policy, owner, permissions and revision.

Protected files use separate owner-scoped `kb_private_text_*` collections and
restricted source/sidecar storage. Private ingestion supports text/Markdown,
CSV, text-bearing PDF, DOCX/XLSX and HTML. Nonblank unextractable PDF pages,
images, audio/video, remote fetching and ordinary media workers refuse. Page
numbering is preserved. Cleanup settles disk/vector/SQLite work before deciding
what to remove; deletion/purge includes protected derivatives. The ordinary
embedding migration refuses protected collections without dedicated support.

Ordinary attachment, Responses, Project, artifact, table, frame, memory and
history paths cannot reuse protected content. Retrieval consults authoritative
upload state before embedding; unavailable authority fails. Protected filename
matches cannot silently substitute ordinary files. Private turns do not enter
ordinary history projection. Owner browser previews/downloads remain available.

Only a new empty standalone draft can switch away from Private: the server
replaces it atomically, preserving unsent text and visibly removing attachments.
After a message/run, Private stays locked. A Private Project retains its mode;
leaving it is explicit. Legacy Local conversations/API choices keep their real
identities and behavior; new native choices offer Private. Optional planner
unavailability may skip decomposition within the same verified local runtime;
policy/model refusals still abort.

### 20D.1 — Private Projects and native model choices

Project creation derives an immutable restriction from Auto/Private selection.
Owned Private chats attach/detach only matching Projects and remain protected
after Project deletion. Dedicated Project uploads share protected context;
ordinary Private chat uploads remain conversation-only unless explicitly linked
by the Tax review grouping flow. File selection requires current ownership,
matching privacy and Ready text content. A link failure leaves the uploaded file
safely in My files.

Preparation revalidates Project revision, instructions and file membership in
the message-save transaction. Concurrent changes require retry before saving the
question. Passages remain bounded untrusted evidence. Failed tool counts/catalog
listings do not establish factual support. Oversized positive result counts are
bounded without changing identity, library or edition; other validation remains
strict and private failure messages expose no arguments.

### Tax review tool and case record

Each conversation owns one durable case in the application database. Its record
contains year/goal, pinned current IRS release, selected source revisions,
current facts, proposals/history, exact retained evidence, decimal calculations,
linked findings and unresolved items. Backup/purge and conversation/account
deletion include it. Changing year or adopting a new library release is an
explicit owner action; retained originals remain and affected evidence,
calculations and worksheets require recheck. A model cannot change the pin.

Starting Tax review outside a Project atomically creates `Tax review <year>` and
groups the case; inside an owned Private Project it reuses that Project. Retrying
setup is idempotent. Existing ungrouped cases offer explicit grouping; merely
opening them creates nothing. Project views read canonical case data. Project
deletion retains conversations, files and case. A saved case survives empty-draft
cleanup and restores Tax review unless the user explicitly chooses Automatic.

`record_user_fact` binds a verbatim excerpt to the current authenticated native
user message. It records what the user stated without repeated confirmation;
model labels, retrieved documents, instructions and old history cannot acquire
that authority. Extracted/inferred/ambiguous values remain proposals, confirmed
or rejected only by owner action. Extracted facts require exact selected-file
checksum/page/text; passage matching does not verify normalization. Corrections
replace the current row and retain bounded history. Meaningful changes stale
dependent work; unchanged retries are no-ops.

Official evidence matches an exact page in the pinned release. Applicability
stays proposed/unknown unless established under its source contract. Evidence,
facts and calculation dependencies must belong to the active case. The server
rejects stale/rejected support, binds decimal arithmetic and derives limited or
needs-input status; it does not let the model certify correctness. A source
listing is selection coverage, not complete reading.

The Toolbox exposes editable current facts, collapsed worksheets, missing-item
questions, retained passages and scope/filing caution. Normal chat starts or
continues review. Saved owner choices send one visible user reply without
losing the next draft/files; stale/failed saves send none. A gap closes only
when its answer supplies the needed information. Compact drawers trap/restore
focus; archived controls are read-only. Pause/Cancel and genuine interrupted
Resume remain available. Internal IDs and coverage machinery stay in tool data.

### Longer-running Private execution

Tasks have one owner, Private conversation and case, with at most one active
task per conversation. Durable states/checkpoints hold work products, usage and
stop reasons, not reasoning traces. Expiring leases/fences reject stale, late or
revoked writes; renewal requires observed progress. Step/time/token/tool limits
retain partial work without granting a fresh budget on resume.

Native admission rechecks current account/groups/model access and pins source
checksums, year/release, Project membership/instructions and runtime/policy
identity. Changed bindings require explicit revalidation; renaming a Project
alone does not invalidate work. Tool and case-write boundaries recheck current
permission and lease authority. Case/checkpoint writes and expected revisions
commit atomically; owner changes invalidate an in-flight stale revision.

A single verified local writer runs in the native fair queue with bounded tool
rounds and active time. Measured generated-token/tool usage is charged before
checkpoint completion, including failed calls. Missing usage cannot become a
successful completion; unknown partial cancellation usage stays unknown.
Generation checks admission and remaining budget before sending content.

Checkpoint/outbox commit and delivery are idempotent. Recovery finishes the
original canonical answer, never another message. Startup replays committed
results and pauses/fences unfinished tasks without launching new model work.
Stop cancels the durable task; explicit resume retains evidence and spent budget.

Each worksheet save checks linked facts/calculations and exact pinned source
passages. One bounded local structured assessment may identify unsupported,
wrong-year, conflicting or incomplete findings; one focused repair/recheck is
allowed within the same budget. Missing facts skip the assessment. Malformed or
truncated results cannot pass. Reports bind the exact worksheet/case revision;
case edits invalidate them. Only a current checked worksheet completes a task.
This is selected-support checking with the writer model, not independent legal
verification or proof that all relevant source conditions were inventoried.

#### 20E.1 — controlled check/repair measurement

The six-case matched diagnostic is concluded. Five cases passed in both arms;
checking improved none and approved a deduction without required eligibility
facts. Followups asked the missing facts, withheld the derived result and stopped
without approval, but the raw applied-deduction field still contradicted the
unresolved state. That case remains failed. The early stop did not exercise the
new reviewer condition inventory. Full identities, outputs and measurement limits
belong in [MODEL-FACTS](../../evals/MODEL-FACTS.md), not another campaign run log.

Writers/checkers inventory source conditions and linked confirmed facts, but
models still select/interpret those conditions. Empty/incomplete inventories can
miss a rule. False/zero are values; null/missing stays unknown. These guardrails
and checker agreement do not establish eligibility. No scalar probe rerun,
model promotion or tax engine is implied by closure.

### Known native accuracy and coverage limits

The native worksheet separated withholding and prior-year wages and produced
the requested cited arithmetic, but its first answer applied a deduction before
eligibility facts were supplied. Later arithmetic and accepted facts/scope UI
do not prove that behavior was corrected. Tax review therefore retains the
selected-source/conditional-arithmetic scope above.

The retained [native comparison assessment](../../evals/results/2026-10-24-tax-review-native-comparison.md)
has incomplete topic/year coverage. Better uncertainty reporting is not a passed
comparison. The answer's budget explanation is not execution telemetry; runtime
usage and its stopping cause were not established. A targeted Form 8911 lookup
identified the requested edition/page/cutoff, and later export acceptance proved
full-passage persistence and delivery. Those passes do not establish broad
coverage, resolve every source conflict or demonstrate general tax accuracy.
Official sources can conflict; preserved originals are not silently edited to
make an answer appear correct. Catalog metadata and fluent prose are not rules.

### 20F — saved review files

Accepted persistence and XLSX delivery retain exact passages, original extraction
spacing, dates and provenance. A completed chat message alone is not proof of a
saved worksheet. Page reads issue a bounded run-bound `source_ref` for the exact
visible excerpt; server resolution rechecks its pin and stores the actual
passage. Forged/expired references require another read. Models cannot retype or
override provenance.

A new worksheet save adds a protected revision-bound chat workbook link; Toolbox
and Project offer Download/Preview. Exports contain Overview, Facts, Evidence,
Calculations and Unresolved items, with literal values, exact citations and the
supported scope. They do not execute formulas/macros or edit case facts.

Preparation snapshots the displayed revision; first download renders locally,
with concurrent requests reusing the immutable result. Old links never change
to newer facts. The separate `tax-review-exports` cache has bounded retention
(30 days, eight snapshots/case, 512 snapshots/128 MiB total), per-snapshot/workbook
limits and two render jobs. Capacity can evict sooner; expired links direct to
the current export. Canonical case data persists independently and a database
backup need not retain every derived snapshot.

Owner/case checks surround rendering; download streams an open handle.
Conversation/account deletion removes exports; Project deletion retains its
conversations. Export failure does not discard canonical evidence. Read-only
[previews](phase-11-native-file-explorer.md#browser-previews) use the same revision
and authorization without model calls. General [generated files](phase-11-native-file-explorer.md#generated-files)
use the permanent My files lifecycle instead of this cache.

## Operator references

Collection: `src/audrey/references/irs_collect.py` and
`scripts/reference/collect-irs-onbox.sh`. Public indexing:
`scripts/reference/index-irs-onbox.sh`. Consult their current help and the
[testing reference](../reference/testing-protocol.md) for a requested future run.
No archive, migration, index or acceptance rerun is part of this closed campaign.

## Official source starting points

- [Current IRS forms, instructions and publications](https://www.irs.gov/forms-instructions-and-publications)
- [Prior-year forms and instructions](https://www.irs.gov/prior-year-forms-and-instructions)
- [Internal Revenue Bulletins](https://www.irs.gov/irb)
- [Tax code, regulations and official guidance](https://www.irs.gov/privacy-disclosure/tax-code-regulations-and-official-guidance)
