# Campaign 3 Phase 16 - native document tools

**Status:** Slice 16A passed its deployed restart gate on October 4, 2026.
Slice 16B is laptop-complete and awaits its native browser acceptance.

## Goal

Let an authenticated Audrey user ask for a private document to be created,
revised, or rendered while keeping file authority, approval, execution, and
results inside Audrey.

The accepted direction adapts the useful safety patterns from the reviewed
Hermes document-tools plan. Audrey will reuse its existing accounts, My Files
storage, Projects, job ownership, ranged downloads, quotas, and repair
mechanisms. It will not introduce Nextcloud as a second file authority or move
large document bytes through model context or MCP base64 sessions.

The architectural assessment and rejected alternatives are recorded in
[Advanced document tools assessment](../plans/advanced-document-tools-assessment.md).

## Product contract

- Inputs are authenticated, owner-scoped Audrey file IDs or reviewed
  server-owned templates.
- Every successful mutation publishes a new immutable private file version.
  An existing original is never edited in place.
- The browser or model cannot provide an owner, server path, executable,
  LibreOffice filter, or arbitrary network destination.
- Model-proposed writes pause at a native approval boundary tied to the exact
  operation, arguments, input version, and expiry.
- Rejection, cancellation, timeout, crash, or verification failure leaves the
  original and every prior version intact.
- Generated outputs appear in My Files and can be selected by Projects through
  the existing file lifecycle.
- Public links, recipient delivery, formulas, and destructive operations remain
  separate capabilities with their own policy and acceptance gates.

## Slice 16A - approvals and immutable derivations

Add the platform boundary required before a model can request a side effect.

### Storage

Add owner-scoped records for:

- immutable file versions and derivation provenance;
- a document job with input version, requested operation, normalized arguments,
  state, timestamps, attempts, and output version;
- a single-use approval with actor, operation digest, expiry, decision, and
  audit timestamps.

The operation digest covers the normalized tool name, owner-scoped inputs,
expected versions, bounded arguments, and proposed output type. Any argument
or version change requires a new approval.

### Runtime

- Present a plain-language operation summary and bounded change preview.
- Only the authenticated user can approve or reject an interactive request.
- Expired, rejected, already-used, altered, cross-owner, or stale-version
  approvals cannot execute.
- Bots cannot approve their own work. A Bot token receives only administrator
  allowlisted operations and limits.
- Claim, retry, cancellation, and recovery follow Audrey's durable worker lease
  rules and idempotency keys.

### Acceptance

1. Prove owner isolation for input, version, job, approval, and output IDs.
2. Prove the exact approved operation executes at most once.
3. Prove changed arguments, expired approval, rejection, and stale input
   versions execute nothing.
4. Kill and restart Audrey around pending, approved, and running jobs; recover
   without duplicate or partial publication.
5. Confirm approval records and logs exclude document bodies and credentials.

### Slice 16A implementation result

**Completed on the laptop, 2026-10-03.** Additive schema 20 introduces
owner-scoped immutable file versions, document jobs, single-use approvals, and
derivation provenance. The operation digest covers the normalized operation,
input version, bounded JSON arguments, and output MIME type. Idempotency keys
may repeat only the same digest. Approvals have bounded expiry, exact digest
matching, provider-authenticated owner decisions, and explicit rejection,
expiry, cancellation, and consumption states. Bot accounts cannot approve;
bot-created requests default deny unless the internal caller supplies an
administrator-owned operation allowlist.

A durable repository validates current input versions before request,
approval, claim, and publication. It uses exclusive short transactions for
claims, bounded leases and attempts, restart recovery, stale-worker rejection,
and atomic publication of one immutable output version plus provenance. A
repeated identical completion returns the existing output, while a changed
completion is rejected. Approval rows and owner-visible API responses exclude
operation arguments and document bodies; they expose only the bounded summary,
preview, digest, state, and audit times. Native routes list/read owned jobs,
approve or reject them through provider authentication, and cancel active
work. Privacy purge and isolated backup verification include the new records.

The focused Phase 16A suite passes 6/6, the surrounding schema, migration,
history, Projects, administration, and deletion set passes 83/83, and changed
Python files pass Ruff and compilation. The two-stage
`smoke_native_document_approvals.py` gate captures pending, queued, and running
states, restarts Audrey, reclaims an expired lease, blocks the stale worker,
proves single publication, terminates remaining jobs, and deletes all probe
records.

**Live result, October 4, 2026:** Passed. Capture proved schema 20, exact-digest
denial, cross-owner denial, and all three pre-restart states. Verify passed its
persistence, stale-lease, idempotent-publication, rejection, and cancellation
checks. The cleanup ordering was corrected to delete derived output versions
before their parent source; the retained probe then cleaned three jobs and two
versions with no foreign-key violation.

## Slice 16B - private template to DOCX

Ship one useful output before adding general editing.

- Use one reviewed, server-owned DOCX template.
- Accept bounded plain-text fields through a typed operation schema.
- Validate the whole request before queuing work.
- Stage input read-only and publish a new owner-scoped DOCX only after the
  package can be reopened and expected text is read back.
- Record template identity, input version, normalized operation digest,
  output hash, worker version, and verification result.
- Exclude arbitrary templates, images, macros, formulas, external links,
  existing-file overwrite, and sharing.

### Acceptance

Create a document through the native approval flow, download it through the
existing private route, reopen it, and verify its package structure, expected
text, hash, provenance, My Files presentation, Project selection, cancellation,
restart recovery, and second-user denial.

### Slice 16B implementation result

**Completed on the laptop, October 4, 2026.** Audrey now ships one reviewed
server-owned **Project Brief** template. Its typed request accepts a filename,
title, recipient, date, summary, one to eight objectives, and one to eight next
steps. The exact template SHA-256 is included in the approved operation digest,
so replacing the bundled template invalidates an older approval.

The pure-Python worker claims only `template_to_docx` jobs. It renders to a
private staged path, rejects unsafe ZIP paths, macros, ActiveX, embedded files,
images, external relationships, fields, and formulas, then reopens the DOCX and
reads every expected value back before publication. A successful output uses a
deterministic file ID, Audrey's quota reservation and private Qdrant indexing,
the normal My Files lifecycle, one immutable derived version, an output hash,
and a verified worker identity. A restart after My Files commits reopens and
publishes the same file instead of creating a duplicate.

My Files now has a separate **Create a document** panel. The user completes the
Project Brief form, reviews the server summary and exact request fingerprint,
then approves or rejects it. Queued and running work is polled across dialog
reopens; cancellation remains available, and a verified success refreshes the
file explorer automatically. The output can then be downloaded, opened, or
selected in a Project like any other private Ready document.

Sixteen focused backend tests cover rendering and package safety, validation,
owner isolation, approval routing, operation-specific claims, private
publication, provenance, cancellation boundaries, and restart recovery. The
full 3,154-test hermetic backend gate, scoped Ruff, compilation, and diff integrity pass.
Node remains deferred on this laptop, so the added Vitest/API coverage and web
build run as part of the deployed `audrey-ui` gate rather than being claimed
locally.

The remaining native gate is the browser runbook in
`docs/reference/live-smoke-testing.md`. Slice 16C does not start until that
Project Brief flow is accepted.

## Slice 16C - isolated PDF rendering

Add a dedicated **document-worker** modeled on Audrey's media worker:

- non-root process and read-only application image;
- internal Docker network with no egress, Ollama, browser credentials, or user
  tokens;
- private scratch directory and fixed LibreOffice profile per job;
- staged read-only input plus bounded writable output;
- CPU, memory, process, page, output-size, and whole-job time limits;
- atomic publication after format, page-count, known-text, and hash checks.

The worker accepts only a typed DOCX-to-PDF job. It does not accept shell text,
arbitrary command flags, URLs, server paths, or model-selected filters.

### Acceptance

Render the verified DOCX from Slice 16B, confirm a nonempty readable PDF in My
Files, then prove timeout, malformed input, oversized output, worker restart,
cleanup, owner isolation, and network isolation.

## Slice 16D - typed spreadsheet literals

Admit XLSX only after the file parser and worker boundaries are ready.

- Copy an owned workbook to a staged revision.
- Accept bounded cell edits with explicit text, number, boolean, date, and
  blank types.
- Validate every sheet, range, value, count, and expected input version before
  applying any edit.
- Apply the batch copy-on-write and publish only if the workbook reopens and
  actual cell types and values read back correctly.
- Preserve text beginning with an equals sign as text when the request type is
  text.
- Keep formulas, recalculation, macros, external links, charts, and arbitrary
  formatting outside this slice.

### Acceptance

Prove all literal types, all-or-nothing validation, stale-version conflicts,
concurrent edits with one clear winner, formula-like text preservation,
download/read-back, restart recovery, and cross-owner denial.

## Slice 16E - formulas and sharing

These are separate later decisions rather than implied parts of document
editing.

Formula support requires an allowlisted grammar, dependency and resource
limits, recalculation in the isolated worker, and verified calculated
read-back. Sharing requires explicit recipients or public-link policy, expiry,
revocation, audit, and a clear statement of what Audrey can and cannot prevent
after download.

Do not start either capability until Slices 16A through 16D are accepted and a
new narrow plan is approved.

## Delivery order

1. Complete and accept Phase 15 project grounding.
2. Build 16A as the reusable approval and revision boundary.
3. Build one template-to-DOCX operation in 16B.
4. Add isolated PDF rendering in 16C.
5. Add literal-only spreadsheet revisions in 16D.
6. Reassess formulas and sharing as separate product slices.

Phase 16 is complete only when Slices 16A through 16D pass their laptop and
native gates. Slice 16E remains outside completion unless it is separately
approved.
