# Phase 20 — reference libraries and private tax assistance

Requested October 14, 2026. The archive is sufficient; 20B.1 is deployed and
accepted. **20B.2 published October 18:** 2,406 current-catalog PDFs, 62,704
verified chunks and no pending documents. Publication 1304 recovered.
Publication 5902 is the sole approved content-search gap: its original remains
available and incomplete coverage is explicitly disclosed. **20B retrieval is
accepted October 18:** correct tool calls and matching cited PDF pages/year.
Tax accuracy remains 20E work. **20C.1–3 accepted/deployed:** shared model/tool
policy and durable restrictions. **20C.4–6 built; browser acceptance next:**
verified native runtime, protected document processing and Private in the model
picker. The supported text/document path is enforced; Projects, OCR and media
remain unavailable until their own protected processing exists.
Qwen3 Embedding 4B remains the reference model; Nomic history and prior personal
retrieval checks remain accepted.
English content and requests only. Federal coverage can begin independently;
state jurisdiction and priority tax years await the user's answer.

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

## Build slices

| Slice | Deliverable | Acceptance |
|---|---|---|
| 20A | Broad official IRS archive collector | Enumerate current/prior catalogs; resumable downloads; preserved revisions and checksums; clear coverage, failed/skipped items and bytes |
| 20B | Named local reference libraries | Register IRS Federal; exact document lookup and scoped hybrid search/read; year/status filtering and page/section citations; originals accessible |
| 20C | Enforced private processing | Local calls throughout, external content egress blocked, private reuse restrictions and sensitive logging handled; unavailable local resources fail clearly |
| 20D | Tax Project and assistance workflow | Explicit years/jurisdictions/library release, fact interview, verified extraction, financial normalization/reconciliation and cited worksheet |
| 20E | Tax accuracy and retrieval evaluation | Known-answer English cases prove amounts, applicable editions, eligibility/missing-fact behavior and privacy boundaries; compare local models on these tasks |

20A can collect public information before private processing is ready.
20B and 20C establish retrieval/privacy prerequisites. General Tax assistance
availability also requires 20D and targeted 20E acceptance.
State source adapters follow the selected jurisdiction; unsupported state
coverage is shown explicitly and never inferred from federal references.

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

**Explicit selection, decided October 19:** add **Private** beside the existing
workflows in the model picker. This uses installed local models, with no new
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
- General Local and direct-model choices keep their current meaning. Private
  carries the end-to-end processing promise once it is implemented and tested.

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
work or title generation. The supported native Private entrypoint below supplies
the bound policy. Private Project file linking remains refused, and standalone
Project-context reads need the active policy. The stored restriction is derived
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

#### 20C.6 — native Private chats (built; manual acceptance pending)

The deployment's canonical `private_processing.enabled` exposes **Private** in
the existing picker. It uses the installed Local panel plus verified router/title,
summary and pinned text-embedding models. Routing configuration and clients are
isolated from ordinary requests. Local and direct-model choices are unchanged.
No sensitive-content detector or separate privacy tool selector is added.

Only a new/empty conversation can become Private. Its restriction persists
through reload and cannot be cleared by selecting another model. UI/API refuse
Project assignment for this first release. Uploads automatically use protected
processing; its picker offers only ready Private documents. Ordinary chat pickers
exclude those documents, and backend read guards enforce the same restriction.
This is local **server processing**: browser traffic still follows the configured
Cloudflare route, and the server administrator remains trusted. It does not claim
an offline connection or verified tax-preparation accuracy.

After normal commit/push/pull, run on **Tower**, in
`/mnt/user/appdata/audrey_ai_2.0`:

```bash
docker compose up -d --build --force-recreate audrey custom-tools audrey-ui
```

Recreate applies canonical configuration. The sidecar rebuild uses the shared
calculator core. No model pull, schema migration, reindex, collector or media
worker rebuild is required. Existing accepted IRS/migration checks stay closed.

**Manual acceptance at `https://ai.builtryte.xyz`:** start a new chat, select
Private, upload a small non-sensitive PDF with selectable text (or TXT/CSV), and
ask for its contents plus calculator `1250 * 1.08` (expected `1350`). Verify
Processing/Ready, local models/tool disclosure, reload/reconnect and the retained
Private choice. Use owned IRS tools to read Publication17 page1 and identify its
stated tax year. In an ordinary new chat, the Private document must be absent
from the attachment picker. A real upload is required for this gate; use synthetic
or public evidence before taxpayer records.

After this acceptance, extend Private Projects and supported worker operations
without weakening the first release's boundaries, then proceed to 20D/20E.

### 20D/20E — verified assistance

Use existing Projects, uploads, calculator, table tools and source disclosures.
Add the smallest case record needed for confirmed facts, source locations,
uncertainties and document coverage. Financial normalization handles currency,
thousands separators and accounting negatives without hiding exclusions.
Audrey asks for missing eligibility facts before making recommendations.
Tax-specific deterministic rules are introduced only with dated official
references and known-answer checks; generic arithmetic is not a tax engine.

Use synthetic/redacted documents for repeatable cases and manual acceptance
with real documents. Cover W-2/1099 box reading, CSV/XLSX totals, wrong-year
retrieval, cross-references, missing records, conflicting facts and local-service
failure. Compare local models on source fidelity and tax decisions separately
from arithmetic; record supported strengths in the existing model report.

The initial filing-requirements answer supplies concrete 20E cases: preserve the
age/blindness qualifiers in dependent charts and the January-1 age convention;
reconcile conflicting official sources before presenting definitive thresholds.
The current Publication 4012 chart and Rev. Proc. 2025-32 section 4.14 disagree
on the single-65+ amount. Retrieval of a cited table does not settle that conflict.

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
