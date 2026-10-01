# Campaign 3 Phase 7 - scanned PDF OCR

**Status:** OCR is complete and live-settled; the PDF Summary/Transcript follow-up is laptop-complete and awaiting deployment.

## Goal

Make image-only PDFs usable anywhere Audrey already accepts an ordinary text
PDF. Upload stays fast, OCR work survives a worker restart, recognized text is
owner scoped, and the original PDF remains available for inspection or
download.

A PDF with a normal text layer keeps the existing synchronous path. Only a PDF
whose parser returns no text enters OCR processing.

## Slice 7A - English scanned PDFs

The upload flow is:

1. Audrey stores and sniffs the PDF using the existing upload and quota gates.
2. The ordinary PDF parser runs first. A nonempty text layer is indexed
   immediately and never reaches OCR.
3. An empty PDF becomes a durable `pending` text row in the existing media job
   queue. The upload request returns instead of holding an HTTP connection open.
4. `media-worker` leases the row, uses Poppler to render one page at a time, and
   uses Tesseract to recognize English text.
5. The worker posts the complete typed result to Audrey with its current lease.
   Audrey writes the OCR sidecar, embeds owner-scoped chunks, and moves the row
   to `ready` only after indexing succeeds.
6. The native document reader and `get_file_text` use the OCR sidecar, so chat,
   the Files viewer, and retrieval see the same derived text.

The original PDF remains the quota and download object. The derived
`<file_id>.ocr.txt` sidecar supplies the complete Transcript text without
replacing the original byte count. A separate owner-scoped summary sidecar
holds the generated document description.

The worker image now includes `poppler-utils`, `tesseract-ocr`, and the explicit
English language pack. The command contracts follow the official
[Tesseract CLI documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)
and Poppler's
[`pdftoppm` manual](https://github.com/davidben/poppler/blob/master/utils/pdftoppm.1).

## Bounds and failure behavior

`config.yaml` owns the OCR settings sent with each claim:

| Setting | Initial value | Purpose |
|---|---:|---|
| `kb.ocr.enabled` | `true` | Kill switch; `false` restores the prior immediate 422 for image-only PDFs. |
| `language` | `eng` | Installed Tesseract language. |
| `dpi` | `200` | Raster resolution, validated between 72 and 600. |
| `max_pages` | `100` | Refuse oversized documents before rendering page one. |
| `max_chars` | `5,000,000` | Bound the recognized text returned and embedded. |
| `timeout_s` | `1,200` | Whole-document CPU budget, below the shared 30-minute lease. |

Pages are rendered and removed one at a time, which prevents a large PDF from
filling the worker scratch directory with a complete raster copy. A corrupt,
blank, over-limit, or timed-out document becomes `failed` with a user-visible
reason. A missing Poppler or Tesseract executable is treated as a broken worker
image: the exception escapes, the lease eventually returns to the queue, and
documents do not spend all attempts on an infrastructure failure.

The result route requires the media service token, the exact active lease, a
text row, and `application/pdf`. Audrey remains the only SQLite and Qdrant
writer. Deletion already removes every sidecar under the file id, so OCR text
uses the existing tombstone and repair lifecycle.

## Laptop verification

- The OCR driver is covered for page-at-a-time rendering, page limits,
  scratch cleanup, missing binaries, and worker failure reporting.
- Upload coverage proves the empty-PDF transition to `pending` and the
  `enabled: false` rollback.
- Lease coverage proves typed OCR settings, service authentication, stale-lease
  rejection, sidecar storage, ready-state completion, and original byte
  accounting.
- Reader and payload coverage proves OCR text reaches the native viewer and
  owner-scoped text collection without being mislabeled as a video artifact.
- The generated live fixture is verified as a one-page image PDF with no text
  layer.
- The full hermetic backend suite passes: 3,043 tests with one existing FastAPI
  deprecation warning. Changed-file Ruff, Python compilation, and diff checks
  pass.
- Frontend packages are present in this checkout, but its shell has no Node or
  npm executable. The browser test/build command could not run locally. The
  frontend change is the upload help sentence only.

## Live evidence and PDF presentation follow-up

The corrected synthetic OCR proof passed on 2026-09-30: the image-only PDF
moved from Pending to Ready in 4.619 seconds, produced one indexed chunk, the
reader returned 101 characters with its page marker and every required word,
and deletion plus repair completed cleanly. The user then manually uploaded a
PDF through Files, confirmed the document was readable, attached a document in
chat, and received an answer grounded in its contents. That upload/chat path is
settled and should not be repeated for the OCR slice.

The follow-up adds a generated two or three sentence summary for every newly
processed PDF. **View text** now opens on **Summary**; **Transcript** shows the
complete selectable or OCR text. Summary generation is fail-soft, so a useful
indexed document remains Ready if its summary model is unavailable.

After deploying `audrey` and `audrey-ui`, test this presentation with one real
PDF in the native browser:

1. Upload a new PDF. Existing Ready PDFs are not backfilled by this slice.
2. Wait for **Ready**, choose **View text**, and confirm **Summary** opens first
   with a natural two or three sentence description.
3. Choose **Transcript** and confirm the complete document text appears.
4. Ask Audrey one question whose answer is in that PDF and confirm the answer is
   grounded in the document.

This one manual flow is the remaining acceptance check for the follow-up. The
already-passed synthetic OCR script remains diagnostic evidence and does not
need to be rerun.

## Completion gate

OCR remains complete. The presentation follow-up completes when a newly
uploaded PDF shows both Summary and Transcript in the deployed Files page and
answers one grounded chat question.
