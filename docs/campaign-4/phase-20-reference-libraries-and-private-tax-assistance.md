# Phase 20 — reference libraries and private tax assistance

Requested October 14, 2026. The archive is sufficient; 20B.1 catalog/read is built.
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
Use one collection per named library, with edition/year/status metadata filters;
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

#### 20B.1 — catalog and page reader (built)

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
returns an explicit error until 20B.2 establishes that metadata.

The read-only SQLite inventory is authoritative even when final report export
fails. Reads do not load the full archive, use private files or contact a model.
PDF text extraction runs separately with checksum, file/page/text, memory, CPU
and elapsed-time bounds. Page and character pagination preserve precise coverage;
blank/scanned pages need a future OCR/inspection path. Actual page reads produce
official-source citations in the existing Sources disclosure.

20B.2 adds published releases and hybrid text search over current editions,
keeping the same scope on keyword/vector search and exact reads. Recent history
is separately selected; defaults never mix decades of revisions. Apply verified
tax-year metadata when selecting an edition for a return.

### 20C — private processing

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

### Deploy and accept 20B.1

After the user's commit/push, on Tower in `/mnt/user/appdata/audrey_ai_2.0`:

```bash
git pull
docker compose up -d --build custom-tools audrey
```

No UI or media-worker rebuild, private upload, indexing run or collector rerun is
needed. The backend already mounts `/mnt/user/knowledge` at `/datasets:ro`.

In a new browser conversation, use Auto/Automatic and ask Audrey to locate the
current Publication 17 in the owned IRS library, read PDF page 1 and identify its
stated tax year with a source link. Expect `reference_lookup` and `read_reference`
in Tools plus a page-specific official URL in Sources. A missing text layer is
disclosed. In that same chat ask for the archived 1976 Form 940: the library
must refuse access, with no old document text. A revision date is not presented
as a verified applicable tax year. This proves the catalog/read surface; hybrid
content search and private tax processing remain future slices.

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
