# Phase 20 — reference libraries and private tax assistance

Requested October 14, 2026. The archive is sufficient; 20B.1 is deployed and
accepted. **20B.2 published October 18:** 2,406 current-catalog PDFs, 62,704
verified chunks and no pending documents. Publication 1304 recovered.
Publication 5902 is the sole approved content-search gap: its original remains
available and incomplete coverage is explicitly disclosed. **20B retrieval is
accepted October 18:** correct tool calls and matching cited PDF pages/year.
Tax accuracy remains 20E work. **20C.1–3 accepted/deployed:** shared model/tool
policy and durable restrictions. **20C document/calculator and IRS retry accepted October 20:**
verified native runtime, protected document processing and Private in the model
picker. **20D.1 accepted October 20:** protected Projects and native Local
choice consolidation. OCR and media remain unavailable pending a supported case.
Qwen3 Embedding 4B remains the reference model; Nomic history and prior personal
retrieval checks remain accepted.
English content and requests only. Federal coverage can begin independently;
the tax year is collected per case and state coverage is added when requested.

## Outcome

A person creates a tax Project, identifies the tax year and jurisdiction,
uploads their documents, and works with local models using an owned official
reference library. Audrey asks for missing facts, checks document amounts,
calculates verified totals and explains applicable rules with precise citations.
The result is a reviewed worksheet and unresolved-item list for return preparation.
Return filing and a complete custom tax engine are outside this phase.

The reference-library foundation is reusable for other named subjects. Tax is
its first application; existing topic datasets and ordinary chat stay intact.

## Decisions

- Use retrieval augmented generation (RAG): search retained documents and give
  relevant passages to the model. Reuse local embeddings, Qdrant and hybrid
  keyword/vector search. No additional vector service is required.
- Fine-tuning is deferred until measured extraction or workflow failures justify
  it. Updated rules belong in the versioned library. Private records never
  become model-training data.
- Keep archive ownership separate from model access. The broad English archive
  is sufficient and older originals remain offline. The accessible library uses
  current official editions by default. Historical access requires an explicit
  request and is limited to the most recent five years (2022–2026 initially).
  An active current publication can have an older revision date; catalog edition
  age and tax-year applicability are separate facts. Do not index decades of
  historical material into the model library.
- Include official guidance as well as forms: Internal Revenue Bulletins,
  notices, revenue rulings/procedures, corrections and dated IRS guidance.
  Add official federal statutes and Treasury regulations through separate
  source adapters. Record coverage gaps; ownership of many forms does not
  establish possession of every applicable legal authority.
- Publish searchable releases with known provenance and extraction quality.
  Keep historical originals even when default retrieval excludes obsolete,
  duplicate, withdrawn or draft material. Historical questions select the
  edition applicable to that year; "superseded today" does not invalidate an
  edition for its historical tax year.
- Public references and private taxpayer evidence are separate inventories,
  indexes and searches. Public updating never receives user documents or
  personalized search queries.

## User flow

1. Create a Project, choose **Tax assistance**, and set the tax year and
   federal/state jurisdictions. Select a published reference-library release.
2. Explain the situation through a short interview covering income, household,
   business, investments, property and other potentially relevant events.
   Audrey expands the questions when facts indicate another applicable topic.
3. Upload actual forms and statements. Audrey records document coverage,
   flags unread or uncertain material and asks the user to confirm important
   amounts against the original page/box.
4. Search the appropriate official instructions and guidance using those facts,
   including eligibility conditions, exceptions and referenced forms. Open
   the full cited section when a short passage is insufficient.
5. Normalize financial numbers, check duplicate/missing records, reconcile totals
   and use deterministic calculation tools. State the rule, inputs and rounding
   used; missing facts remain unknown.
6. Present confirmed facts, calculations and proposed return items with sources,
   plus unresolved questions and documents still needed. The user reviews the
   worksheet before using it to prepare or check a return.

A document being uploaded or indexed is not evidence that every page was read.
A relevant search result is not a complete eligibility determination.

## Reference storage and ownership

Proposed persistent Tower root:

```text
/mnt/user/knowledge/reference-libraries/tax/us/federal/
  irs/
    originals/       retained source PDFs/HTML and immutable revisions
    catalogs/        official catalog snapshots and download manifests
  legislation/      official statutory source snapshots
  regulations/      official Treasury regulation snapshots
  releases/         published indexing manifests and coverage summaries
```

Audrey sees the existing read-only knowledge mount under
`/datasets/reference-libraries/`. A dedicated collector writes the host archive;
the broad raw archive is not blindly added to the ordinary dataset watcher.
The originals and catalogs remain useful without Audrey or Qdrant and can be
copied to the laptop. Indexes are rebuildable derivatives, not the only copy.

Each document retains its official URL, publisher, title, product number/type,
language, checksum, retrieval date, revision/publication dates, applicable tax
years, jurisdiction, effective dates where known, status and supersession links.
Revision date and download date never substitute for tax-year applicability.
Unknown applicability stays explicit until established from the source.
Chunks retain document identity and page/section; tables retain headers and units.

## Component contracts

The [main build queue](README.md#build-queue) owns order and status. The sections
below define the archive, retrieval, privacy and assistance contracts. General
Tax review availability requires its workflow and targeted accuracy acceptance.
State adapters follow the selected jurisdiction; unsupported state coverage is
shown explicitly and never inferred from federal references.

### 20A — archive collection

Enumerate official catalogs rather than guessing filenames. Prioritize current
English products, then expand available historical revisions and bulletin
coverage in resumable batches. Discover broadly before downloading; report
estimated counts and observed bytes without claiming complete coverage early.
Use conservative request rates, timeouts/retries, disk limits and explicit
progress. Catalog changes or failed downloads cannot erase prior originals.
Allowlisted public sources only; validate redirects, file signatures and hashes.
Changed content at the same URL becomes a new retained revision.
Report ambiguous language/metadata and download failures for review.
A missing historical year is a gap, not permission to substitute today's form.

### 20B — named libraries

A server-owned registry maps a stable library ID, description and published
release to authorized roots and a dedicated text collection in existing Qdrant.
Use one collection per named library and embedding space, with edition/year/status filters;
do not create a collection per form or silently search unrelated subjects.
Apply identical scope to keyword and vector retrieval. Unknown or unavailable
libraries produce an explicit result.

Provide bounded catalog/search/read operations using stable document IDs,
including exact form/publication lookup and access to cited pages/sections.
The model never chooses an arbitrary filesystem path or collection name.
Results identify official URL, title/number, applicable edition, page/section,
library release and extraction limitations. Blank forms remain available through
the catalog; instructions and guidance provide their explanatory context.
Publish a release only after its coverage and extraction checks pass; preserve
the previous usable release during updates. Tax retrieval thresholds are measured
on tax cases rather than assumed from other topic datasets.

#### 20B.1 — catalog and page reader (accepted October 14)

`reference_libraries` in repository configuration registers the owned IRS root.
`reference_lookup` performs exact product-number/title catalog lookup;
`read_reference` reads one PDF text page using its full document ID and optional
current revision checksum. The existing Automatic tool selection exposes both.
Catalog lookup is not document-content search; vector indexing follows in 20B.2.

Both operations enforce active catalog membership and the same server-owned
history window. Current is the default for lookup and reading; explicit
`catalog=prior` or `bulletins` permits recent editions only. Direct old IDs,
old checksums, removed entries and ambiguous historical edition years are refused.
Failed updates cannot silently serve an old retained body as fresh. Edition-year
filters never assert applicability to a tax year; unverified tax-year filtering
returns an explicit error. Verified return-year filtering is provided by
20B.2 content search; catalog metadata remains separate.

The read-only SQLite inventory is authoritative even when final report export
fails. Reads do not load the full archive, use private files or contact a model.
PDF text extraction runs separately with checksum, file/page/text, memory, CPU
and elapsed-time bounds. Page and character pagination preserve precise coverage;
blank/scanned pages need a future OCR/inspection path. Actual page reads produce
official-source citations in the existing Sources disclosure.

#### 20B.2 — published content search (published and retrieval accepted October 18)

`reference_search` searches PDF text using local Qwen3 Embedding 4B and the
existing Qdrant keyword/vector fusion. It selects one published library/catalog release;
both retrievers receive the same document, release, status and optional year
filters. It never falls back to another subject or a private upload collection.
The default build covers current English catalog entries only. Optional recent
history needs its own explicitly selected build and search.

Public derivative state lives at
`/mnt/user/appdata/runtime/reference-libraries/` (`index.sqlite` and job reports),
with a dedicated Qdrant collection for each library/embedding space. The job mounts only the public
archive read-only, repository configuration and that public index state. It
uses the dedicated reference embedding profile; it does not call an answering model,
download more PDFs or receive private uploads or API credentials.

Each bounded extraction verifies the retained PDF checksum and keeps exact PDF
page/character spans. Public PDFs with editing restrictions open only if the
empty password succeeds; nonempty-password protection remains refused in both
indexing and cited page reads. The whole-document worker periodically reopens
the same verified source with the same parser to release cached objects. Its
768 MiB address-space cap stays within the 2 GiB job container; the one-page
reader retains its separate 512 MiB cap. Page/output limits remain enforced;
offline extraction allows at most 180 CPU seconds and 210 elapsed seconds per
PDF. Only stable failure codes enter reports. A document checkpoint commits after its vector writes
complete. Interrupted runs resume verified checkpoints; incomplete remote
checkpoints are rebuilt. Verified saved documents emit resume progress and do
not repeat extraction or embeddings. Before these exact counts, the job creates
missing `document_id`, `release_id` and `source` keyword payload indexes in the
reference collection. Compatible indexes are reused; incompatible schemas or
service failures surface. Payload indexes change neither vectors nor scope
filters. Unchanged published releases are reused after remote verification. Staging points include their release identity, so a failed build
cannot overwrite the previous release's vectors. Publication atomically switches
only after every selected original is accounted for and exact remote chunk counts
match. Unapproved extraction failures block publication; approved exact-document gaps
remain failed checkpoints with disclosed incomplete coverage. Blank/scanned text layers are
counted explicitly as `no_text`, without claiming OCR coverage.

Results identify release, selected catalog, coverage/subset status, document
revision, PDF page and exact excerpt. Candidates are checked against the immutable
release and current permitted catalog before being returned. Missing vectors or
unavailable services produce explicit errors. Only nonempty returned passages
produce page citations in Sources; a snippet is not full-document review.

Return-year filtering is conservative: an unambiguous first-page statement such
as "For use in preparing 2025 Returns" establishes that document's stated return
year, with its quote/page/checksum retained. Other documents stay unknown and are
excluded from year-filtered search. Revision dates and isolated years do not
establish applicability. This metadata does not decide taxpayer eligibility.
The reference embedding profile pins the tested model digest, query/document
format and native 2,560-dimensional vectors. Indexing and query use the same
configuration; releases/collections separate incompatible profiles. Changed
model identity or collection schema fails clearly. Keyword vectors are enabled
independently of dense width. Inputs cannot silently truncate. The same Qwen text profile serves uploads and
memories; chat history independently uses Nomic after Phase 06's accepted scoped
rebuild. Image embeddings remain unchanged.

The dense cutoff `0.54` comes from calibration questions; lexical matches rank
results but cannot independently pass admission. The held-out test still admitted
some related passages without the requested detail. This is an initial measured
retrieval policy, not complete tax accuracy; 20E remains required.
Private tax processing and the assistance workflow remain 20C/20D work.

### 20C — private processing

**Explicit selection, updated October 20:** use **Private** as the native
local-processing choice. After the first Private gate and protected Project
path are ready, replace Local in new native model choices; retain legacy Local
history and API behavior. This uses installed local models, with no new
model download. The selection activates the policy; no sensitive-content
detection, extra classifier or separate privacy tool selector is required.

- Keep routing, answers, summaries, embeddings and supported file processing
  local. Local-service failure reports unavailability instead of using cloud.
- Automatically limit Tools to operations whose complete processing is local.
  Local files, calculations and the owned reference library remain the target
  capabilities; web search/fetch and other external operations are unavailable.
- Persist the choice on the conversation and propagate restrictions to files
  added through private flows and their derived content. Later reuse must not
  silently send protected material to a cloud model.
- Existing Local conversations keep their actual processing status and legacy
  model ID; they are never relabelled Private. Compatibility/API Local and
  direct-model choices retain their meaning. The native chooser consolidation
  follows the verified Private entrypoint and supported Project path.

One enforceable policy covers routing, planning, workers, synthesis, fallbacks,
OCR/vision, summaries, embeddings and tool dispatch. Cloud fallback is refused;
external searches/fetches and remote image/file inputs are blocked for private
requests. The public reference updater operates independently.

Private evidence and derived content retain their processing restrictions when
reused in conversations, project context, memories or history retrieval.
General memory/history must not carry protected tax content into a later cloud
request. Keep owner isolation and avoid private prompt/query/result log snippets.
Private mode provides a precise statement of its scope: local processing on
Tower is separate from browser transport and the server administrator's access.
Verify storage/backup protection; a fully private-network browser route requires
its own acceptance. Cloudflare access is not described as offline processing.

#### 20C.1 — shared policy boundary (accepted)

`ProcessingPolicy` is an immutable server-owned allowlist carried in the current
async task. Nested scopes may only tighten it; child tasks inherit it and exit
restores the parent policy. The scope must cover full async-stream consumption.
The native Private entrypoint below binds this policy from server-owned state.
A client flag cannot construct or weaken it.

Ollama chat, streaming and embedding calls require a trusted exact endpoint,
disabled environment-proxy inheritance, an approved model name/digest, matching
installed metadata and verifiable local weights. Missing/changed identities,
remote aliases or unavailable metadata fail before request content is sent.
Metadata is rechecked for each call. The daemon, administrator and configured
transport are trusted; checks cannot atomically prevent a concurrent model-alias
swap. No automatic cloud fallback can bypass the boundary while scope is active.

Tool dispatch checks the actual endpoint and operation path before parsing or
logging arguments. It refuses redirects and proxy inheritance. Only explicitly
allowed local operations are eligible. The supported document/reference search
uses verified local embeddings; web, memory/history and media operations remain
refused. Owner-bound callbacks below preserve policy across local service logic.
Protected failure messages omit prompt/tool contents.
Ordinary requests retain existing behavior.

#### 20C.2 — durable restrictions (accepted; deployed October 19)

Schema 22 adds an owner-scoped `private_required` restriction to Projects,
conversations and each accepted run. Existing records remain ordinary. New
conversations inherit a private Project's restriction; moves, detachment,
archiving, model changes and Project deletion cannot clear it. Only empty
ordinary conversations can join a private Project: previously processed content
cannot be relabelled as having been private. SQL guards prevent weakening the
restriction or changing a run's snapshot.

Run admission rechecks the expected restriction in the message-save transaction.
Protected server-side admission additionally requires a bound processing policy.
Ordinary admission refuses protected work before model selection, file/context
work or title generation. The native Private entrypoint supplies the bound policy.
Protected Project linking and context reads require that verified policy and
matching durable restrictions. The stored restriction is derived
by the server, never accepted as a caller-supplied privacy flag.

Finished and interrupted private turns do not enter ordinary chat-history
projection. Projection delivery also excludes protected conversations; ordinary
history behavior is preserved. Auxiliary ordinary title updates refuse protected
rows. These safeguards do not yet supply private memory/history retrieval.

#### 20C.3 — file restrictions and reuse boundaries (accepted October 19)

The uploads database stores an immutable `private_required` flag on files,
chunked sessions and single-upload reservations. Existing records remain
ordinary. The file's accepted restriction comes from its owned reservation in
one commit transaction, never a worker result, caller object or vector payload.
Retries cannot relabel it; startup/backfill preserves protected metadata.
Authoritative checks happen before extraction, embeddings or summarization.

Protected uploads remain excluded from ordinary processing: current workers
cannot claim/complete them or reclaim their originals. OCR/media/fetch callbacks
and requeue refuse before content writes or model calls. Public upload entry
points cannot shed an active private policy. The dedicated native path below
handles supported documents with the existing restriction intact.

Ordinary model reads refuse protected files, including native attachments,
Responses file/image inputs, Project links/manifests, file artifacts, table
analysis and page/frame inspection. Dense, keyword and image retrieval exclude
protected IDs using the owned uploads database; unavailable authority reports an
error before embedding. Tool file lists and filename suggestions omit protected
metadata. Exact newest protected filename matches refuse rather than substitute
an older ordinary file. Direct history reads also exclude protected conversations,
including imported IDs. Owner browser previews and downloads remain available.

#### 20C.4 — verified native runtime (built)

A metadata-only factory verifies the server-selected local model roster and
snapshots exact installed digests, including any configured pins. Each execution
rechecks all identities before work, owns proxy-disabled model/tool clients and
compiles the existing graph against a restricted registry. An isolated app-state
view shares gates and ordinary services without replacing global clients.

Native tasks require a runtime matching their durable private restriction.
The policy spans full stream consumption, child-task cancellation, explicit
stream closure and client teardown. Optional titles run afterward in their own
verified scope; an unapproved title model leaves the fallback title. Title writes
require a bound policy and an unchanged owner/revision/private restriction.
Private failures at the manager boundary report safe codes without exception
contents. Private turns remain excluded from ordinary history delivery.

#### 20C.5 — protected documents and local tools (built)

A dedicated native upload route derives protection from the authenticated
conversation and owned quota reservation. Plain text/Markdown, CSV, text-bearing
PDF, DOCX/XLSX and HTML extract locally. Genuine blank PDF pages retain their
numbering; nonblank pages without extractable text require the future OCR path.
Images, audio/video, URL fetching and legacy worker processing remain refused.
PDF summaries and embeddings use verified local models only.

Protected vectors live in separate owner-scoped `kb_private_text_*` collections,
including during staging and failed commits. Source and text sidecars have
restrictive permissions. Cancellation settles disk/vector writes and SQLite
commit before deciding cleanup. Owner browser reads/downloads remain available;
ordinary model readers cannot reuse protected files. Deletion/purge includes
the private collection. Startup checks never reconstruct protected ownership
from vector payloads. The ordinary full embedding-migration CLI refuses private
collections until dedicated protected migration support exists; unrelated
history-only maintenance remains available. No current migration rerun is needed.

Private tools execute through server-bound, expiring owner-scoped callbacks,
with no sidecar HTTP fallback. Calculator, CSV/XLSX analysis, owned file listing,
paged text/summary reading, private hybrid document search and IRS catalog/search/
page reading reuse existing cores. Dense and keyword retrieval share the exact
owned-file selection and private collection. Document/reference query embeddings
use the scoped verified client and pinned profile. Web, general memory/history
and media tools are unavailable. General recall/store guidance is suppressed;
Private conversation context remains in its canonical local history.

Private log records remove contents, arguments, exception traces and stacks
before formatting; child tasks and worker threads retain the boundary. Graph
tracing is disabled task-locally. Model/tool/data clients disable environment
proxies and redirects; failures do not select a cloud fallback.

#### 20C.6 — native Private chats (document/calculator and IRS accepted)

The deployment's canonical `private_processing.enabled` exposes **Private** in
the existing picker. It uses the installed Local panel plus verified router/title,
summary and pinned text-embedding models. Routing configuration and clients are
isolated from ordinary requests. Local and direct-model choices are unchanged.
No sensitive-content detector or separate privacy tool selector is added.

Only a new/empty conversation can become Private. Before its first canonical
message or run, a standalone Private draft may choose another model: the server
atomically replaces that empty draft and returns a new conversation ID. It never
weakens the protected row or any uploaded file. Unsent text is preserved; existing
attachments are removed from the replacement with a visible notice. After the
first message/run, Private remains locked through reload. A Private Project keeps
its protection; explicitly leave that Project before changing processing mode.
Private uses the previously unused Audrey portrait with the dark hat and
sunglasses (`audrey5.png`); legacy Local keeps its existing asset. UI/API permit
assignment only to an owned matching Private Project. Uploads automatically use protected
processing; its picker offers only ready Private documents. Ordinary chat pickers
exclude those documents, and backend read guards enforce the same restriction.
This is local **server processing**: browser traffic still follows the configured
Cloudflare route, and the server administrator remains trusted. It does not claim
an offline connection or verified tax-preparation accuracy.

Optional planning now skips protected provider unavailability and continues
the original question through the same verified Local panel. Model, digest
and transport-policy refusals still abort. A timed-out optional planner
reproduced the Private IRS failure; the October 20 retry passed with lookup,
page reading and the correct 2025 return year.
Private run admission requires the durable restriction even when a caller
submits the Private model ID directly; Project creation derives the same
restriction from the selected workflow. Every admitted Private turn must use the bound runtime.

**Accepted:** the uploaded-document summary/calculator and the Private IRS
retry. These prove those flows, not general tax-answer accuracy. Refresh and
ordinary-picker isolation are included in the new Project gate below.

### 20D.1 — Private Projects and native model choices (accepted October 20)

Project creation offers Auto or Private and derives an immutable restriction on
the server. Responses expose it read-only. Private Project conversations inherit
the Private workflow; explicit ordinary/cloud choices refuse. Existing private
conversations can attach/detach matching Projects and retain protection after
Project deletion. Previously processed ordinary chats/files are never relabelled.

Dedicated Project uploads use verified local ingestion and link protected files
into shared context. Uploads from a chat stay attached to that chat; they do not
silently become shared Project files. File pickers and server-side linking require
matching privacy, ownership and Ready text documents. A completed upload survives
a subsequent Project-link failure safely in My Files.

Preparation reads canonical protected sources and searches the isolated owner
collection through the pinned local embedder. Settings, file membership and
privacy are rechecked against a Project revision in the message-save transaction.
Edits, file removal/pruning or membership changes during preparation require a
retry before saving the question. Retrieved passages remain bounded untrusted
evidence; initial source disclosures identify inspected Project files.

New native choices show Private in place of Local when available. Saved Local
conversations retain their actual model ID/label, including catalog omissions.
Admin model ordering/role grants and `/v1`/Hermes behavior remain unchanged.
No tax-specific Project fields, schema migration or new model is introduced.

Private Project upload/context/reopen/file-isolation acceptance is complete.
Draft selection, deletion navigation, the MiniMax cloud addition, Private
portrait and activity-sorted history are accepted. Private's Tools and skills
control says Automatic; its processing restrictions remain enforced.
No Project upload, indexing or migration needs repeating. Tool activity and
catalog metadata must not be presented to synthesis as proof that factual
claims were verified. Synthesis now receives
successful/failed tool counts and an explicit limitation when reference calls
returned no passage/page evidence. Reference guidance recommends page reads
only when those tools are actually available; it adds no forced call or new
model. Private callbacks cap oversized positive integer result counts, using
the same
search/catalog maximums. Invalid identity, scope, page/year, types and unknown
fields retain normal validation; no count repair changes the requested library
or edition. Native HTTP APIs keep their strict limits. Tool validation feedback
uses fixed messages without arguments or exception contents. The Private content-
search/page-read retry is accepted; this does not establish tax-answer accuracy. Only current-
catalog search is published; retained earlier editions do not imply a prior
search index.
The main plan owns the remaining queue; no temporary smoke document is retained.

### Tax review foundation and remaining Private task contracts

The [main build queue](README.md#build-queue) owns status and order. The contracts
below cover the dedicated tax-review tool and reusable Private task runner,
with tax assistance as the first workflow. The saved case/tool foundation is built;
foreground worksheet review has a limited user-reported numerical pass; broader
accuracy acceptance and the task runner remain queued.
OCR/media remain deferred until a case requires them.

#### Tax review tool and case record

**20D.2 and presentation followup accepted October 21.** Select
Private, choose Tax review in Tools & skills, choose the federal return year and
goal, then save. The year defaults to the previous completed calendar year and
can be changed before or after setup. Settings open in a dialog bounded by the
visible screen and collapse after saving. The compact box retains settings,
Start/Continue and collapsed findings. Essential confirmation/rejection and
missing-information choices appear in the conversation; related facts can be
confirmed together. After the owner saves a choice, a visible native user reply
informs Audrey and starts the next review automatically. Unsent composer text and
attachments remain untouched; a failed or stale save sends no reply. Only the
owner can approve facts. Record IDs, checksums and passing checklist rows stay
internal. Audrey handles official-source verification herself and asks only for
facts or decisions the user can supply. The generic degraded-features badge is
removed throughout chat, including new chat after deletion; actual service/model
outage indicators remain. The user moved on after the targeted presentation
handoff; no repeated presentation check is queued.
Each conversation owns one case; Project membership is derived from its canonical
conversation. A saved case does not activate Tax review automatically after reload.
Automatic Private retains its existing tools and behavior. The additive schema 23
record lives in the application database, participates in normal SQLite backup,
and cascades on conversation deletion or owner purge.

The in-process tool inspects/checks the case, proposes facts, retains exact IRS
passages, runs decimal calculations and records missing items/return dependencies.
Only owner UI actions confirm/reject proposed facts, resolve missing items or
change the year. Extracted
facts require a selected Private file, original checksum, exact page/text and
proposed normalization; a passage check does not validate that normalization.
Official evidence must match an exact page in the pinned current release.
Model-supplied tax-year applicability stays proposed/unknown. Checklists report
coverage, including conflicting/unread material, without certifying tax correctness.

This first case is federal-only and covers selected passages. There is no complete-return certification, state adapter or background agent. Changing the case year retains original source years/passages, resets prior
confirmations and marks calculations/return proposals for recheck. Explicit
re-recording preserves the old record and supersedes it with the new review.
The goal remains fixed at setup. Historical originals are still available through
ordinary reference tools but are not evidence in this current-release case.
No collector/index/model rerun is needed.

**20D.3a — explicit reference updates (built October 21).** A changed or
unavailable published library leaves the saved case readable. The box shows only
the applicable update/reload choice. Work requiring Tax review stays blocked until
the exact current release is explicitly adopted by the owner. A model cannot
change that pin. A stale browser target or case revision is refused; a publication
that changes during the save returns the approved pin with a further update
choice. No unapproved release is silently adopted.

An update preserves personal facts and confirmations, source revisions and exact
original passages. Existing rule evidence, calculations and return proposals need
rechecking. Reopened official passages can explicitly replace stale evidence;
recalculated work uses the fresh evidence IDs. Superseded originals remain in the
bounded case history, including across successive updates. A new library release
does not itself prove that a law changed or that a retained rule applies to the
return year. These state-transition checks do not establish tax accuracy.

**20D.3b — foreground worksheet (built October 21).** Start/Continue review sends
a visible native user turn with explicit Tax review selected. It uses the existing
highest-priority verified local tool model as one case writer, with twelve tool
rounds and a ten-minute foreground deadline. Ordinary Private retains its existing
panel configuration. The full saved goal is supplied as data; inspect can page the
canonical selected-file inventory, including Project references without extracted
facts. A listing establishes selection, not complete reading. Newly chosen unsent
files stay in the composer; send them in a chat turn before reviewing them.

The native run retains normal Stop, browser reconnect and canonical message
persistence. Deadline/cancellation retains committed case entries; another review
continues from that state. Server restart still interrupts execution: restart-
resumable tasks are 20D.4. There is no background worker in this slice.

The model saves a bounded worksheet with a summary, linked findings and the next
question. The server checks all dependency IDs and rejects rejected facts or stale
support. It derives needs-input/evidence-limited status without letting the model
certify tax correctness. Confirmations, missing items and changed year/library
remain explicit user choices. Meaningful case changes invalidate the saved
worksheet; no-op retries do not. Original evidence stays retained. Worksheets are
covered by the existing case backup/purge and deletion rules in schema23.

The compact Tax review box shows the applicable Start/Continue choice and saved
status, with worksheet details collapsed. Expanded findings include retained IRS
passages, recorded calculation results and limitations. Tax-year applicability is
still proposed/unknown under this tool contract; the new worksheet does not turn
that into verified law or imply every document page has been checked.

One [synthetic upload](../../evals/fixtures/tax-review-2025-single.txt) and its
[source-checked baseline](../../evals/cases/tax_review_baseline.json) exercise the
native flow. It separates 2025 wages/withholding from a 2024 comparison, pauses for
fact confirmation, retains official deduction evidence and checks the requested
wages-minus-deduction arithmetic. Expected values stay outside the upload. This
is a manual native-browser baseline using the existing workflow, not a new probe
runner. The user-reported answer passed the requested numerical/separation check:
$50,000 wages minus a $15,750 deduction gave $34,250; $6,000 withholding stayed
separate and $9,000 of prior-year wages was excluded. An official deduction passage
was retained. This limited pass does not independently verify legal applicability,
close the known tax-accuracy cases in this phase or certify a complete return.

**Tax review** appears in the existing Tools and skills control for a Private
conversation/Project. It may be selected explicitly; ordinary Private questions
remain lightweight. The tool is registered with the normal security declarations
and bound in-process to the authenticated owner and Private case. No sidecar HTTP
fallback, new external service or sensitive-content detector is needed.

Typed operations inspect the case, propose facts, record evidence/calculations,
return missing items, produce the next checklist steps and validate a result.
The server controls permitted transitions and required evidence. Model-supplied
IDs cannot select another owner or escape the active Private context. A model
may propose tasks; it cannot run arbitrary code or replace the workflow rules.

Keep a compact durable record of:

- Return year and federal/state scope, task goal, selected file IDs/revisions and
  pinned reference release. Calendar year and PDF revision date do not establish
  the return year. A new edition/release becomes an explicit case update.
- User statements, extracted facts and user-confirmed facts as distinct states.
  Only a user action can confirm a proposal. Unknown stays unknown.
- Document inventory and unread/uncertain pages; each extracted amount carries
  file/checksum, page/box, original text and normalization. Do not require full
  SSNs, bank details or unrelated identifiers for a planning/calculation task.
- Rule claims with official document/revision, stated tax-year applicability,
  exact page/section and supporting passage. Store conflicting evidence too.
- Decimal inputs, expression, rounding and calculator result, plus duplicate and
  missing-record checks. Generic arithmetic never substitutes for a dated tax rule.
- Proposed return items, dependencies, unresolved questions and verification
  status. Document instructions are evidence, never authority to change the task.

Use a versioned tax checklist for planning and coverage. The small router can
classify or suggest subtasks; it does not decide that all tax work is complete.
An explicit request for Publication 17's year should use exact lookup and page
reading. Search misses, missing library coverage and source conflicts remain
unresolved; they do not establish that a rule or obligation does not exist.

#### Longer-running Private execution

**20D.4a built — checkpoint authority.** Additive schema 24 stores tasks in the
application database. Each task belongs to one owner, Private conversation and tax
case; one queued/running/waiting/paused task can exist per conversation. States,
committed stage, linked case findings, usage and explicit stop reasons persist.
Checkpoints contain work products, not reasoning traces. Existing application
backup/restore includes them; conversation/account deletion and owner purge cascade
them without touching another owner.

Claims are serialized across SQLite connections. Each grants a new expiring lease
and fence; stale revisions, expired/replaced leases and late writes after owner
pause/cancel are refused. An identical checkpoint retry is a no-op after a lost
reply. Renewal requires new observed progress. Limits bound steps, active execution
time, generated tokens and tool calls; over-budget results stay partial. Explicit
expired-lease recovery pauses abandoned attempts without launching work.

The repository pins the case revision, tax year, library release, selected-file
bindings, Project instructions/file membership and server-supplied model/policy
digests. Changed bindings block old work; explicit resume with fresh validation
clears a stale checkpoint while retaining spent budget and case evidence. A Project
rename alone does not invalidate work. Active owner, deletion/purge, Private context
and support-ID ownership are checked on reads and mutations.

No worker, task endpoint, startup scheduling or browser task controls are enabled
in this foundation. **20D.4b is next:** construct the verified local runtime, check
live model/file/reference availability, fence case mutations and deliver results
idempotently through the native conversation. A stored digest alone does not prove
local execution or current source availability. Existing foreground review remains
the user-facing path until that integration passes its own acceptance.

The remaining runner uses typed stages with a tax adapter supplying the checklist
and evidence checks. Reuse the existing fair GPU scheduler,
PrivateProcessingRuntime, reference/file handlers and native progress disclosures.
Native runs persist presentation; task checkpoints own recovery. Retired
document-authoring job tables stay retired. Final message delivery must be
idempotent, and a watchdog must pause abandoned attempts with a specific reason.

On restart, recover to a safe checkpoint and revalidate owner/access, cancellation,
Private policy, installed model digests, file revisions and the reference release
before any model/tool call. Changed evidence invalidates dependent findings and
requires their recheck. Revocation/purge/cancel stops subsequent work and prevents
late commits. Saved case/task data and recovery state participate in owner purge
and the existing local backup/restore path.

Start with sequential local stages sharing the existing GPU gate, releasing it
between steps so interactive chat can proceed. Initial operator-configured limits
cover wall time, generated tokens, tool calls and review passes. A provisional
10-minute budget and two verification passes are starting settings to measure,
not accuracy promises; longer limits can be requested for a specific task.
No extra model download or model promotion is part of this plan.

A reviewer reopens the original evidence and checks specific claims, conditions,
amounts and missing topics. Use a different installed local model when useful,
then repair only the failed parts. Agreement between models is not proof.
Stop when checks complete, the user must supply facts, source coverage is
insufficient/conflicting, two rounds yield no new evidence, the budget expires,
or the user cancels. Report the exact stop reason and partial findings.

#### Accuracy and visible user flow

User flow: select Private → open a tax Project → choose Tax review → specify
return year and jurisdiction → upload/select documents → review proposed facts
and the plan → start review. Show one compact progress item such as
“Checking income records · 3 of 8 checks complete,” with expandable current
findings, missing items and Pause/Resume/Cancel. Keep the existing Models,
Tool calls and Sources disclosures. Completion is a cited worksheet and a
clear question list; this phase does not file returns or author office documents.

Federal work can start with the owned IRS library. State-specific conclusions
wait for the requested jurisdiction's official sources. Source updates run
separately using public inputs; a Private case never performs public searches
with taxpayer details. English-only work and the existing historical window stay.

Compare matched single-pass and extended runs using the same files, library
release, model identities and resource accounting. Score field/box extraction,
duplicate/corrected documents, wrong-year exclusion, eligibility conditions,
citation support, decimal arithmetic/rounding, missing-fact handling and false
certainty. Measure time/tokens as costs, not quality. Synthetic/redacted cases
cover W-2/1099 fields, CSV/XLSX reconciliation, unread pages, source conflicts,
missing records, local-service failure, cancellation and restart recovery.
Keep manual native-browser acceptance for uploads, progress and results.

The earlier filing-requirements answer supplies concrete accuracy cases: preserve
age/blindness qualifiers and the January-1 age convention; investigate the owned
Publication 4012 / Rev. Proc. 2025-32 section 4.14 discrepancy before presenting
a definitive threshold. A retrieved table alone does not resolve conflicting
sources. Longer review must demonstrate improvements against these known gaps.

The reported 2025-versus-2024 comparison is another known-answer case: failed
content searches plus successful catalog lookups did not support its claims.
Check the actual cited sections for each year. Personal exemptions were not
restored; the 2025 basic standard deduction is $15,750 single / $31,500 joint;
qualified tips/overtime deductions have limits and conditions; and the maximum
2025 EITC for three or more qualifying children is $8,046, without the claimed
per-child bonuses. These expectations come from
[Publication 1321 (2025), PDF page 4](https://www.irs.gov/pub/irs-access/p1321_accessible.pdf),
[IRS 2025 deduction guidance](https://www.irs.gov/forms-pubs/how-to-update-withholding-to-account-for-tax-law-changes-for-2025)
and [IRS EITC tables](https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/earned-income-and-earned-income-tax-credit-eitc-tables).
A subsequent successful retrieval still repeated the pre-law 2025 deduction
amounts and put the section 30C refueling-credit cutoff a year early. Include
wrong-year changes and pre-amendment values in the same comparison case: the
[IRS termination table](https://www.irs.gov/newsroom/faqs-for-modification-of-sections-25c-25d-25e-30c-30d-45l-45w-and-179d-under-public-law-119-21-139-stat-72-july-4-2025-commonly-known-as-the-one-big-beautiful-bill-obbb)
sets that cutoff at June 30, 2026. A prompt or plumbing fix alone does not close
this accuracy case.

Official design references: [IRS document-gathering guidance](https://www.irs.gov/filing/gather-your-documents)
for the inventory/interview, [Publication 17 overview](https://www.irs.gov/forms-pubs/about-publication-17)
for its role alongside form instructions, and [Taxpayer Advocate guidance on AI tax advice](https://www.taxpayeradvocate.irs.gov/news/tax-tips/is-ai-generated-tax-advice-making-the-grade/2024/06/)
for the need to verify facts and calculations. These inform workflow design;
tax-rule implementations still require the applicable retained official source.

## Deployment and testing

20A's collector is implemented in `audrey.references.irs_collect` with
`scripts/reference/collect-irs-onbox.sh` as its Tower runner. Laptop checks cover
catalog parsing, durable revisions/resume, HTTP and storage bounds, and runner
reporting/cancellation. A live bounded laptop check retained five official PDFs
without failures. The user's bounded Tower check also passed October 14.
20A is accepted and collection is sufficient. The resumed log confirms 26,706
PDFs downloaded or verified. The process was killed after the final recorded
download; its final report has no inventory. The only known source failure is
the corrupt 1976 Form 940, which is outside model access. No download rerun is
required. The library uses durable SQLite commits and originals directly;
final export/report files are not prerequisites.

### 20B.2 deployment and acceptance

The current-catalog release is published. All 2,407 selected documents are
accounted for: 2,406 indexed and the sole approved Publication 5902 extraction
gap. Exact remote verification counted 62,704 chunks; zero documents are pending.
`coverage.subset` is false. `gap_count: 1`, `content_search_gaps` and
`complete_for_selected_documents: false` accurately disclose missing content.
Publication succeeded under that approved contract; no further recovery run is
needed. Both [embedding migrations](phase-06-system-one-routing.md#tower-migrations)
and the personal-retrieval browser checks remain accepted.

Publication 5902's original remains available in the catalog. The report and
every search response identify the missing content and its source revision.
Exact gap searches return no passage or unrelated fallback; gap metadata never
becomes a citation. The operator-only `--allow-extraction-gap DOCUMENT_ID`
requires an exact failed checkpoint in an existing `--release-id`; wildcards,
duplicates and `--rebuild` are refused. Failed documents are retried and indexed
normally if repaired. Unapproved errors, pending work and vector-count mismatches
still block publication. Normal releases without approved gaps require complete
coverage and zero failures.

For future source updates, pull the user's committed changes on **Tower**, in
`/mnt/user/appdata/audrey_ai_2.0`; rebuild `audrey` with `docker compose build audrey`
only if the job image changed. The existing runner is:

```bash
nohup bash scripts/reference/index-irs-onbox.sh > testing-out/last-reference-index.log 2>&1 &
```

It verifies saved checkpoints and reuses unchanged published releases. A failed
build leaves the previous publication intact. The job survives disconnects and
sends Telegram start/finish messages and its stamped JSON report; reports/logs
also remain under `testing-out/reference-indexes/`. Read the downloaded report
on the laptop. Only local embeddings are generated, sharing Ollama resources
with chat. Expected extraction failures use safe codes; unexpected tracebacks
stay in restricted public-index `diagnostics/`, with only class/path in reports.
No archive collection, private upload, personal token or embedding migration is
needed. The runner's public mounts and resource limits remain unchanged.

**Retrieval acceptance is complete.** The user's cited filing-requirements
answer used the correct tools and matching official PDF pages/year. No repeat
browser/indexing check is required. This establishes the retrieval flow; tax
accuracy and enforced private processing have their own acceptance below.

Operator options remain bounded: `--product-number "Publication 17"` or
`--max-documents 5` intentionally publishes a disclosed subset, not full coverage.
Use these only for a requested diagnosis. `--catalog prior` or `bulletins` builds
a separate permitted recent scope; `--rebuild` forces a fresh replacement for
unchanged sources. None widens the configured history window.

### Collector operator reference

Collection is sufficient; do not rerun it for 20B.1. For future source
updates, in `/mnt/user/appdata/audrey_ai_2.0`, build its disposable image:

```bash
docker compose build audrey
```

No running service needs a restart for 20A. The collector uses an isolated
container, public HTTPS and writable archive/report mounts; no GPU, API token,
uploaded file, Ollama or Qdrant is needed.

Originals reside under
`/mnt/user/knowledge/reference-libraries/tax/us/federal/irs/originals/`.

Future current-catalog updates use the same runner:

```bash
nohup bash scripts/reference/collect-irs-onbox.sh --catalogs current > testing-out/last-irs-archive.log 2>&1 &
```

This updates current official editions without repeating historical collection.
It survives a disconnect.
The existing fleet-watchdog settings send a start ping, completion/interruption
ping and stamped JSON report to Telegram. Notification failures warn without
changing the collection result; no personal data is included. Reports/logs also
remain in `testing-out/reference-archives/` on Tower. Read the downloaded
Telegram report on the laptop.

Rerun the same command to resume; no previously retained revision is removed.
The defaults allow 10 GiB of newly downloaded PDF data per run, 256 MiB per PDF
and reserve 1 GiB of free disk. A file exceeding the individual size limit is
reported as a coverage gap and collection continues. The run stops at the total
byte budget or disk reserve. Use the collector's `--help` to adjust limits for
a later batch. Corrupted sources remain flagged rather than weakening validation.
`--refresh` checks verified PDFs conditionally for changed content; changed
catalog revision/posted dates also trigger checks automatically. PDF size/time
or metadata/download failures never turn into claims of complete coverage.
Only a fully enumerated selected catalog with every eligible PDF verified and
no failures receives `status: completed`.

The readable manifest and SQLite inventory live in `irs/catalogs/`.
Tax-year applicability remains unknown until verified from the source in 20B;
downloaded/revision dates are retained as source metadata. The originals stay
outside the existing general KB watcher/index. Other implementation slices
supply their own exact rebuild and targeted acceptance instructions.

## Official source starting points

- [Current IRS forms, instructions and publications](https://www.irs.gov/forms-instructions-and-publications)
- [Prior-year IRS forms and instructions](https://www.irs.gov/prior-year-forms-and-instructions)
- [Internal Revenue Bulletins](https://www.irs.gov/irb)
- [Federal tax code, regulations and official guidance](https://www.irs.gov/privacy-disclosure/tax-code-regulations-and-official-guidance)

The IRS guidance identifies distinct sources of authority and cautions that
current legal text may not match the tax year being researched. Source adapters
and retrieval must preserve those distinctions.
