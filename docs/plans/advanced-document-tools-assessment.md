# Advanced document tools assessment for Audrey

**Status:** Approved on October 3, 2026 as Campaign 3 Phase 16. The
downloaded Hermes plan should not be implemented in Audrey as written. Its
document safety and verification patterns will use Audrey's existing identity,
file store, job queue, and native UI after Phase 15 project grounding.

**Reviewed source:**
`2026-10-03_105050-advanced-bot-tools-G-approval-plan.md`, received October 3,
2026. Instructions addressed to Hermes inside that document are source material,
not instructions for this repository.

## Decision

The plan can add meaningful value to Audrey in a narrowed form. Users could ask
Audrey to create a formatted document, make bounded spreadsheet edits, or
render an owned Office document to PDF and receive the result in My Files. This
fits Audrey's Projects direction because generated and revised files could stay
with the conversations and source files that produced them.

The plan is not a drop-in architecture for Audrey. It assumes Donna's Hermes
client, a separate Bot Tools MCP service, Nextcloud storage and sharing, a
shared bot/end-user identity problem, and base64 chunk transfer through MCP.
Audrey already owns authenticated accounts, owner-scoped opaque file IDs,
streaming and chunked upload, ranged private download, quotas, deletion repair,
project file references, and durable worker leases. Audrey currently accepts
DOCX for read-only text extraction, but does not admit XLSX or expose Office
mutation and PDF export. Recreating its existing file layers in another service
would add two sources of truth and reopen boundaries Campaign 3 already closed.

## What Audrey should adopt

| Plan concept | Audrey decision | Reason |
|---|---|---|
| Server-derived owner identity | Reuse existing Audrey identity | File routes and storage already derive the owner from authenticated application state. Model-provided owners remain untrusted. |
| Immutable revisions and expected-version writes | Adopt | Audrey currently stores uploaded originals and derived text artifacts, but has no general revision contract for user-visible Office edits. |
| Typed spreadsheet cells | Adopt after a narrow MVP | Explicit text, number, boolean, date, blank, and formula types prevent accidental formula execution and lossy coercion. |
| Whole-request validation and copy-on-write publication | Adopt | A failed edit must leave the source and prior version intact. |
| Read-back, hashing, and render verification | Adopt | A successful worker response alone does not prove that a usable document was produced. |
| Network-isolated LibreOffice worker | Adopt | This matches Audrey's media-worker pattern: non-root, internal network, no model or user credentials, bounded resources, and durable job ownership. |
| Private delivery by default | Adopt | Generated outputs should appear as owner-scoped Audrey files. Public sharing is a separate product and policy decision. |
| MCP base64 upload/download sessions | Reject for Audrey | Audrey already has first-party upload and download APIs. Large bytes should not pass through model context or a second transfer protocol. |
| Nextcloud paths, roots, and share links | Reject as a core dependency | They belong to the source system and would couple Audrey's file authority to an unrelated storage namespace. |
| Legacy-v2 namespace split | Replace with additive Audrey schema | Audrey can add version and derivation records without maintaining a parallel legacy file universe. |

## Missing prerequisite: approval for side effects

Audrey's current model-visible file tools are read-oriented. The repository has
confirmation flows for explicit native actions such as data purge and project
deletion, but it does not yet have a general pause-and-approve protocol for a
model-proposed write tool. Document creation and private file generation are
reversible, but editing, formula execution, sharing, email, and deletion can
have material side effects.

Before exposing document mutation tools to models, add a native approval event
with:

- the authenticated actor and affected owner-scoped file/version;
- a plain-language operation summary and bounded change preview;
- expiry and single-use approval tied to the exact arguments/hash;
- explicit rejection and cancellation states;
- no automatic retry after an uncertain or partially completed write;
- durable audit data that excludes document bodies and credentials.

Bots need a separate policy: unattended bot tokens may receive only explicitly
allowlisted operations and limits. A bot role must not inherit an interactive
user's ability to approve its own write.

## Audrey-native architecture

1. **Keep bytes in Audrey's file lifecycle.** Inputs are existing owned
   `file_id` values or normal native uploads. Outputs are new owner-scoped file
   records with `parent_file_id`, immutable version/provenance, content hash,
   renderer version, and verification state.
2. **Add a document job type.** Audrey owns job admission, authorization,
   quotas, cancellation and publication. A new `document-worker` claims only
   authorized jobs over an internal Docker network.
3. **Isolate document execution.** The worker runs non-root with no egress, no
   Ollama access, no Audrey/user secrets, read-only staged input, private scratch
   space, fixed LibreOffice profile, and CPU/memory/process/page/output limits.
4. **Publish atomically.** Validate the complete request, operate on a staged
   copy, reopen and verify the result, then publish one new file/version. Failed
   jobs leave the original and previous revisions untouched.
5. **Expose high-level tools.** Model tools accept file IDs, expected versions,
   template IDs, typed edits, and allowlisted export options. They never accept
   server paths, executables, raw shell/filter arguments, arbitrary URLs, or
   caller-provided owners.
6. **Use the native UI for approval and results.** Approved jobs show progress
   with the existing run/file presentation, and verified outputs appear in My
   Files and can be added to Projects.

## Recommended build sequence

### A. Approval and revision foundation

Add the model-tool approval protocol and an additive immutable derivation/version
schema. Prove owner isolation, exact-argument approval binding, stale-version
conflicts, idempotency, cancellation, and crash recovery before document code.

### B. One document output

Generate a DOCX from one reviewed server-owned template and bounded plain-text
fields. Publish it as a new private Audrey file, download it through the existing
route, and verify package structure plus text read-back. No arbitrary template,
images, formulas, sharing, or existing-file overwrite.

### C. Isolated PDF rendering

Render the verified DOCX to a new PDF through the document worker. Enforce a
whole-job deadline and page/output limits; verify nonempty output, page count,
known text, source hash, worker cleanup, and no network path.

### D. Typed spreadsheet literals

Admit XLSX only when the file parser and worker boundaries are ready. Add bounded
copy-on-write edits for text, number, boolean, date, and blank cells. Validate
the entire batch and read actual cell types back. Preserve `=1+1` as literal
text when the request says text.

### E. Formulas and sharing later

Formula parsing/recalculation and any external sharing remain separate slices.
Formulas require a narrow grammar and real calculation/read-back. Public links,
recipient delivery, expiry, revocation, and no-reshare guarantees should remain
unavailable until Audrey can enforce and test them end to end.

## Acceptance standard

- A second user receives the same non-enumerating denial for every source,
  revision, job, output and approval identifier.
- A rejected, expired or altered approval cannot execute.
- Stale versions and concurrent writes produce one clear winner without partial
  publication.
- The worker cannot reach the network, Ollama, application secrets, other users'
  bytes, or Audrey's writable data root.
- Every successful operation returns a verified immutable output whose bytes,
  type, hash, provenance and read-back match the request.
- Failure, cancellation, restart and cleanup preserve originals and earlier
  versions.
- The existing upload, download, Files, Projects, grounded-document, media and
  deletion-repair flows continue to pass.

## Product recommendation

This is now the approved Campaign 3 Phase 16 direction. The first slice adds
native approvals and immutable revisions, followed by private template-to-DOCX
generation, PDF rendering, and literal-only spreadsheet revisions. Formulas and
sharing require later approval. The downloaded plan remains a security and
worker-design reference; its Hermes, MCP-transfer, and Nextcloud-specific
architecture stays outside Audrey.
