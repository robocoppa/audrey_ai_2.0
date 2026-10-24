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

1. Choose Private and start **Tax review**, setting the year and goal. The
   case automatically creates a Private project or reuses the current owned
   Private project, with the published IRS release pinned to the case.
2. Explain the situation in chat. Explicit current-message facts are saved
   directly and shown in the Toolbox facts table; the user need not confirm
   them again. Audrey asks only for missing information relevant to the goal.
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

## Private project for a saved Tax review

Starting a case outside a Project atomically creates `Tax review <year>`, groups
the conversation and saves the case. Inside an owned Private Project the case
uses that Project. Retrying the same setup returns the saved case; it does not
create another project or discard saved facts. A saved case survives abandoned
draft cleanup before its first message.

The Project page reads current facts and saved worksheets directly from the
case, including retained passages and linked arithmetic. Open review conversation
returns to the normal chat/Toolbox. Existing upload and Choose from My files
controls store owned Private documents in the Project. Ready documents already
attached to the case conversation are linked when grouping; new attachments sent
with Tax review are linked before the task pins its context. Ordinary Private
chats keep their existing conversation-only attachment behavior.

Existing ungrouped cases offer **Create Private project**. This explicit action
preserves their case and conversation; merely opening a case never creates or
recreates a Project. Deleting a Project retains its conversations, files and case
under the established deletion contract. Files still require current ownership,
Private readiness and retained source bytes; a filename is never authority.

This changes no schema, model, index or embedding profile. The prior editable
facts workflow is accepted. Only the new grouping/project view needs browser
acceptance; the bounded native accuracy work remains separate.

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
foreground worksheet review has a limited user-reported numerical pass. Saved
execution and bounded evidence checks are built; broader accuracy evaluation remains queued.
OCR/media remain deferred until a case requires them.

#### Tax review tool and case record

**20D.2 foundation accepted October 22; facts followup October 23.**
Select Private and Tax review in Tools & skills to open the right-side Toolbox.
Its settings expand inline; choose the federal return year and goal, then save.
The year defaults to the previous completed calendar year. The Toolbox scrolls
independently of chat, with a compact current facts table and collapsed worksheet
findings. Fact names and values remain editable and persist after refresh.
Corrections replace the current row while retaining bounded history internally;
confirmed/rejected prompts and addressed-item history are not another user-facing
timeline. Extracted or uncertain details awaiting review remain clearly separate
from saved accepted facts.

On compact screens the Toolbox is a right-side modal drawer with Escape, close,
backdrop dismissal and restored trigger focus. It starts closed for a saved case;
selecting Tax review opens setup. A small pending-count badge makes details
needing review visible without forcing the drawer over the chat. Desktop uses a
docked rail. A larger right-edge tab is centered vertically, with sideways lettering whose tops
face left. The tab hides while open; closing restores it and keyboard focus. A
narrow edge gutter keeps it clear of chat controls. App scrollbars use thin muted
blue thumbs with transparent tracks and a system-color fallback for high-contrast
settings. Open/closed state survives recovered-run runtime refreshes. Closing the
panel does not start work or discard current selections. Archived controls are
read-only.

No idle Start/Continue action is shown. The user sends a normal chat prompt to
start or continue work; active Pause/Cancel and genuine interruption Resume remain.
A saved owner correction or review choice sends one visible native user reply
and continues the review, preserving unsent composer text and files. A failed or
stale save sends no reply. Direct statements already saved from the current chat
message need no confirmation click. A necessary missing fact is asked in plain
language, with an answer box that submits a normal native user turn. The chat
composer remains available too. Missing-item records keep a concise topic label
and a distinct question explaining the information needed. The Toolbox does not
offer an “I’ve provided this” button that resolves a gap without receiving
information. A saved current-message fact can address the linked gap only when
the statement answers it; partial answers leave remaining information open. Internal IDs, revisions and coverage checklists stay
in tool feedback. Worksheet-save guidance separates an explicit source-stated
year from unresolved catalog applicability; source verification remains Audrey’s
work, not a user decision.

The Toolbox retains a muted filing caution: AI can be wrong; check sources and
figures before filing, and Audrey does not prepare or file a return. Private
shows a subdued non-action status badge near Archive/Delete in the header,
between the title and actions. A hover/focus tooltip carries the processing and
server-storage details, administrator access, Cloudflare transport, tool scope
and mode lock. Touch reveals the same text; Escape/outside interaction dismisses
it. The compact header keeps the badge while Archive/Delete stay in mobile
navigation. Composer disclosure prose is removed. Neither notice adds an
acknowledgement gate. The generic degraded badge stays removed; actual outage
indicators remain. Accepted Toolbox layout/lifecycle behavior stays closed;
only the changed facts and missing-information flow needs browser acceptance.

Each conversation owns one case; Project membership is derived from its canonical
conversation. A saved case does not activate Tax review automatically after reload.
Automatic Private retains its existing tools and behavior. The additive schema 23
record lives in the application database, participates in normal SQLite backup,
and cascades on conversation deletion or owner purge.

The in-process tool inspects/checks the case, saves direct statements, proposes
uncertain facts, retains exact IRS passages, runs decimal calculations and records
missing items/return dependencies. `record_user_fact` saves a verbatim excerpt
from the current authenticated native user message without a second confirmation.
The server binds owner, conversation, run/message identity and original text;
model-supplied labels or a `user_statement` kind alone establish no authority.
Retrieved documents, Project instructions and old assistant/history text cannot
supply this current-message binding. This records what the user stated, not an
independent verification of its truth or a model’s normalized interpretation.

Document-derived, inferred or ambiguous values remain proposals. Only owner UI
actions confirm/reject those proposals, edit facts or change the year. Extracted
facts require a selected Private file, original checksum, exact page/text and
proposed normalization; a passage check does not validate that normalization.
An explicit correction updates the identified fact in place and retains its prior
value/source internally. Dependent calculations and return items require recheck,
and the previous worksheet becomes stale in the same save. Repeating unchanged
information is a no-op, not another active fact or fresh verification.
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

**20D.3b — worksheet construction.** A chat prompt with Tax review selected sends
a visible native user turn. It uses the existing
highest-priority verified local tool model as one case writer, with twelve tool
rounds per attempt and the saved task's ten-minute active-work budget. Ordinary Private retains its existing
panel configuration. The full saved goal is supplied as data; inspect can page the
canonical selected-file inventory, including Project references without extracted
facts. A listing establishes selection, not complete reading. Newly chosen unsent
files stay in the composer; send them in a chat turn before reviewing them.

The native run retains normal Stop, browser reconnect and canonical message
persistence. Deadline/cancellation retains committed case entries; another review
continues from that state. Saved task execution, controls and safe restart recovery
are described below.

The model saves a bounded worksheet with a summary, linked findings and the next
question. The server checks all dependency IDs and rejects rejected facts or stale
support. It derives needs-input/evidence-limited status without letting the model
certify tax correctness. Review of extracted/uncertain facts and changes to
year/library remain explicit user choices; answers in chat can supply missing
facts. Meaningful case changes invalidate the saved worksheet; no-op retries do not. Original evidence stays retained. Worksheets are
covered by the existing case backup/purge and deletion rules in schema23.

The Toolbox keeps saved status and worksheet details collapsed when idle.
Expanded findings include retained IRS passages, recorded calculation results
and limitations. Tax-year applicability is
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
- Direct current-message statements, extracted/inferred proposals and accepted
  facts with their actual authority/source. Direct statements need no repeated
  confirmation; only a user action accepts an uncertain proposal. Unknown stays
  unknown. Corrections retain internal history, not duplicate active table rows.
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

**20D.4b.1 built — bounded step execution.** A server-owned callback runs inside
an expiring task scope, the isolated verified Private runtime and owner-bound
local tool handlers. Each tool boundary checks live admission and lease authority.
Observed progress renews the lease; a watchdog does not invent progress while a
model waits. Token/tool costs are written before checkpoint completion, preventing
crash retries from erasing spent resources. Failed calls are charged; checkpoint
completion does not count them twice. Step deadlines and revoked authority cancel
the callback; inherited callbacks lose case-write authority on scope exit.

Tax-case mutations check the active task's owner, conversation, lease, source
bindings and expected case revision inside the same SQLite transaction. Case
changes and the task's new expected revision commit together, or both roll back.
Owner actions in another request retain their existing authority and invalidate
an in-flight worker's stale case revision. A model cannot turn its scoped write
into a user confirmation. Abandoned attempts retain partial case/checkpoint work
and charge observed resources once, with a specific stop reason.

**20D.4b.2a built — native admission and preparation.** A trusted service reloads
the current account, groups and Private-model grant without replaying a login or
granting token scopes. It requires the active owned Private case and current IRS
publication. Preparation verifies the installed local runtime and binds all selected
conversation/Project documents, original and extracted-text checksums, Project
instructions, tax year and publication. Repeated checks use file-operation locks
and cached filesystem identity; changed files are rehashed. No generation,
embedding or vector search is needed for admission. Project renaming alone does
not invalidate useful work.

Native task pins include input and permission digests. A database permission change
between tool admission and its eventual case write is refused inside the case
transaction. A verified extracted fact can refine the source inventory atomically
without invalidating its own lease. Explicit resume validates current inputs,
clears a changed checkpoint and retains already spent resources. Original checksum
conflicts with saved case provenance remain blocked. Existing internal tasks
without the new native pins cannot be claimed by this native preparer until
explicitly revalidated.

**20D.4b.2b built — native saved reviews.** Explicit native Tax review turns
create or resume a task, bind it to the owned canonical run and use the existing
native run manager and fair GPU queue. They retain the ordinary visible user turn,
stream, tool/model/source disclosures and fact-confirmation feedback. Each attempt
uses the same single local writer. Ordinary Private and compatibility requests do
not enter this task flow. Admission failure leaves a stopped native row and pauses
an unclaimed task, so retry cannot remain stuck behind abandoned queued work.

The compact review row loads its saved status before offering Start/Resume and
polls without overlapping requests. Pause/Cancel stay available while the native
run is active; Resume is explicit and keeps unsent text/files and spent budgets.
Normal Stop cancels the durable task. Budget exhaustion retains partial work and
does not offer Resume with a fresh budget. No extra modal or passing checklist is
added. Duplicate recovered-run/retry controls are suppressed when the task row
owns recovery, keeping controls accessible on short desktop and mobile screens.

A scoped generation hook checks admission immediately before every model content
request and caps the provider's generation limit to remaining task tokens. Exact
terminal generated-token counts are durably recorded before accepting output;
actual stream output renews progress without estimating tokens from characters.
Missing/unmeasurable usage or swallowed provider failures cannot become successful
task completion. Cancellation before provider terminal may leave unknown partial
generation; active-time, tool and step limits remain in force and saved measured
usage is retained.

Checkpoint and bounded native-answer outbox commit in one transaction. Delivery
finishes the original native message, never a new message. Lost acknowledgements,
shutdown after commit and replay are idempotent. Startup replays committed results
before generic run recovery, then fences/pauses unfinished tasks. It never launches
model work automatically. Explicit resume revalidates current authority and inputs
and starts a new native attempt linked to the same task. Already committed source
findings stay saved. The user advanced past the lifecycle handoff October 22;
no repeat lifecycle check is queued.

**20D.5 built — bounded evidence checking and targeted repair.** Each native
worksheet save checks only its linked facts, calculations, return items and
official passages. Deterministic checks catch missing/stale support, conflicting
or wrong-year evidence, rejected facts and unreproducible decimal arithmetic.
Distinct supporting passages are reopened against exact pinned document/revision,
page and offset. This is selected-passage checking, not full document coverage.

One structured read-only assessment uses the existing verified local writer model
and fair GPU gate. It checks the worksheet summary, each finding's conditions,
year and amounts against complete supplied support. Instructions in source text
remain untrusted data. The complete request is bounded to 24 KB and twelve
source reads; larger work stays explicitly limited rather than silently clipping
support. Every finding requires exactly one assessment. Malformed, incomplete or
truncated replies cannot pass. Missing user facts skip this extra model call.

The writer receives specific failed findings through save_review feedback, then
repairs only those claims or their saved support. One recheck is allowed, within
the same twelve-round attempt, ten-minute task time and durable generated-token
budget. Unchanged failed input or exhausted checks stop further verification in
that attempt. No new scheduler, download, cloud fallback or parallel panel is
introduced. Measured unusable output remains a limited worksheet; authority,
cancellation and unknown-usage failures retain the existing task pause boundary.

A strict server-supplied report binds the exact validated worksheet digest and
case revision and commits beside it under existing task fences. Failed findings
are retained as partial work if the process stops before repair. Model and
browser payloads cannot inject assessments or promote source tax-year status.
An assessment does not approve personal facts: those retain the current-message
or explicit owner-action authority defined above. A case edit invalidates the
current assessment. Only a
current checked worksheet can finish a native task; unresolved checks remain
paused or wait for the owner's needed facts. Model agreement does not prove
legal correctness. The initial checker intentionally uses the same installed
writer; a different reviewer requires evidence of benefit in 20E.

The result uses the Toolbox findings/limitations and existing Models/Tools/Sources
disclosures. The verification slice adds no user approval. Its targeted browser
check continues an existing confirmed synthetic worksheet,
checks its linked rules/arithmetic and verifies saved findings after refresh.
No new upload, reindex, source collection or migration is required. Local checks
prove plumbing and boundaries; deployed behavior and answer quality still need
their separate acceptance.

#### 20E.1 — controlled check/repair measurement

`scripts/probes/tax_review_accuracy_probe.py` uses the installed production
`verify_review` and decimal calculator against six frozen synthetic cases in
`evals/cases/tax_review_accuracy.json`. Cases cover the limited 2025 single-filer
worksheet, wrong-year/withholding separation, corrected/duplicate records, missing
eligibility facts, confirmed head-of-household status and cent rounding. Public
Publication 501 Table 6 text includes its age/blindness/dependency qualification;
the excerpt and source section are hashed. This packet is not a live IRS release
or a claim to have reopened an original PDF page.

Both arms share the exact first draft, confirmed inputs, source packet and local
model digest. The extended arm invokes the production checker, then at most one
focused rewrite and recheck. Expected outputs never enter either model request.
Independent structured checks report improvements, regressions and wrong outputs
that the checker accepted. Full synthetic outputs remain available for prose
review; structured scores do not establish prose entailment or legal correctness.
The report records exact code/fixture/model identities, unknown usage explicitly,
wall time, generation costs and execution failures separately from quality.

Use the existing on-box probe wrapper because this measurement imports installed
production code and requires direct local-model control. `FOREGROUND=1 NOTIFY=1`
prints the complete JSON in the launching shell and sends the existing Telegram
log attachment. No PAT, upload, index run or live application data is needed.
The probe/fixture are copied from the checkout at run time. This isolated
process does not share the application's in-process fair queue. The diagnostic
has concluded; no further run is queued for the remaining scalar-field defect.

The October 22 measurement completed all six pairs without execution failures.
Five passed in both arms. The missing-eligibility case incorrectly supplied the
deduction despite absent age/blindness/dependency facts, and the reviewer approved
it; no repair ran. The source explicitly conditions use of its table, so this is
a substantive unsupported-assumption failure. Do not populate missing-item records
from the hidden answer key to make the test pass. The initial transport-only
failure is closed and supplied no model-quality evidence.

The eligibility followup is built. Native and probe writers share the instruction
to inventory relevant source-stated conditions/exclusions before applying a rule.
An omitted/null/empty fact stays unknown; explicit false and zero are values.
Affected amounts stay unresolved and only a necessary grouped question is asked.
Pure source descriptions that do not apply a rule need no personal interview.

New checker replies include a bounded condition inventory per finding, with
met/not-met/unknown status, cited source IDs and exact confirmed fact IDs/JSON
pointers into their values. Server validation refuses missing, null, enclosing
composite objects, stale/foreign-year or unlinked field support. Unknown conditions
make the applied finding insufficient; contradicted conditions make it unsupported.
This uses the same single reviewer call and one repair/recheck limit. It changes
no owner confirmation, source-year authority, database schema, model or library release.
Old assessments retain their original serialized shape and no-op behavior.

The model still identifies the conditions and judges what each value means; empty
or incomplete inventories can miss a rule, so this is not independent legal
verification. The incomplete-profile regression rejects a fabricated age-field
citation. Native saved reviews carry needs-input/limited status when facts are
missing; the generic calculator does not infer eligibility or erase independent
arithmetic merely because another case question is open.

The targeted followups identify age, blindness and dependency and stop checking
as limited/needs_user_input, without approval or a reviewer call. After correcting
the probe's premature subtraction and equivalent-field grading, the October 23
result withholds derived arithmetic and passes every check except the raw applied
deduction. That non-null field still contradicts the explicit unresolved state.
Keep the failure and raw outputs; do not relabel it as a passed accuracy case.
The scalar is specific to the diagnostic's schema; the native worksheet uses
linked findings and proposed records, not this deduction_usd field. The latest
early stop does not exercise the new reviewer condition inventory.

**20E.1 is concluded with that limitation.** Repeated prompt changes to force
one field to null have little bearing on useful native assistance. Keep the
existing source/confirmation/calculation boundaries; do not add a universal tax
rule engine or another model to make this probe pass. A model still interprets
source conditions, so a second assessment is not independent legal verification.

#### 20E.2 — bounded native accuracy check

Use the existing native flow, with two concrete scenarios:

1. One synthetic household worksheet: leave relevant eligibility facts unknown,
   check that Audrey asks only the necessary grouped question and keeps the
   conclusion unresolved, then supply the facts in chat. Direct statements should
   appear in the current facts table without another confirmation. Audrey must
   retain the applicable official passage/year and save the requested limited
   worksheet with code-calculated arithmetic and actual remaining gaps.
2. One previously incorrect tax-year comparison below: verify the substantive
   claims against the retained official passages. Catalog listings and fluent
   prose alone do not establish a rule or its effective year.

**October 23 native receipt:** the final worksheet cited Publication 17 (2025),
applied $15,750 to $50,000 wages for $34,250 and kept $6,000 withholding separate.
However, the first answer already applied that deduction before age, blindness
and dependency were supplied. Final arithmetic therefore does not pass the
missing-eligibility behavior. The transcript also exposed redundant confirmation,
an empty continuation answer and a confusing retained prompt history. It does
not provide a matched token/latency report. Keep the outcome and native eligibility
limitation; do not reopen the concluded scalar probe.

The editable facts workflow above is accepted. Automatic Private project
grouping is the current browser followup; then resume the bounded accuracy decision
and tax-year comparison. The facts UI does not itself fix premature
application of a deduction; decide a narrow native correction or supported-scope
limit from the observed failure.

Record outcomes and available costs once, including failures. Correct a concrete
native defect if demonstrated, or narrow the supported task; do not reopen the isolated
scalar test or expand into another prompt-tuning sweep. Closure is an explicit
decision about the observed task, not every test eventually turning green.
Unanswered facts, unresolved source conflicts and broader return completeness
remain visible limitations. No model promotion or full-return acceptance follows.

#### Accuracy and visible user flow

User flow: select Private → open a tax Project → choose Tax review → specify
return year and jurisdiction → state facts in chat and/or upload documents →
inspect current facts and review extracted/uncertain details → send a review
prompt or answer a genuinely missing question. Show one compact Toolbox progress
item such as “Checking income records · 3 of 8 checks complete,” with expandable current
findings, missing items and Pause/Resume/Cancel. Keep the existing Models,
Tool calls and Sources disclosures. Completion is a cited worksheet and a
clear question list; this phase does not file returns or author office documents.

Federal work can start with the owned IRS library. State-specific conclusions
wait for the requested jurisdiction's official sources. Source updates run
separately using public inputs; a Private case never performs public searches
with taxpayer details. English-only work and the existing historical window stay.

For any later quality comparison justified by a reported task, keep files,
library release, model identities and resource accounting matched. Relevant
dimensions include field/box extraction,
duplicate/corrected documents, wrong-year exclusion, eligibility conditions,
citation support, decimal arithmetic/rounding, missing-fact handling and false
certainty. Measure time/tokens as costs, not quality. Synthetic/redacted cases
cover W-2/1099 fields, CSV/XLSX reconciliation, unread pages, source conflicts,
missing records, local-service failure, cancellation and restart recovery.
Keep manual native-browser acceptance for uploads, progress and results;
this list does not add tests to the two-scenario gate above.

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
