# Campaign 3 Phase 7 - scanned PDF OCR

**Status:** Complete and live-settled on 2026-09-30.

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
`<file_id>.ocr.txt` sidecar supplies text but does not replace the original byte
count or become a video-style artifact.

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
- The full hermetic backend suite passes: 3,022 tests with one existing FastAPI
  deprecation warning. Changed-file Ruff, Python compilation, YAML parsing, and
  diff checks pass.
- Frontend packages are present in this checkout, but its shell has no Node or
  npm executable. The browser test/build command could not run locally. The
  frontend change is the upload help sentence only.

## Deploy and targeted live smoke

Rebuild `audrey`, `media-worker`, and `audrey-ui`. Rebuilding the worker is
required because Poppler and Tesseract are image packages; restarting the old
image cannot add them.

Run the targeted smoke from the laptop checkout over the working LAN/WARP
route. **You do not need to upload a PDF first.** The script generates a
high-contrast image-only PDF, uploads it, waits for the deployed worker, reads
back the recognized text, deletes the temporary upload, and drains cleanup.

```bash
cd /home/bart/Documents/github/audrey/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_scanned_pdf_ocr.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"` with:

- upload `initial_status: "pending"`;
- processing `final_status: "ready"` and a positive chunk count;
- reader HTTP 200, `page_marker: true`, and all required words;
- cleanup `deleted: true` and `repair_status: "ready"`.

A failure naming a missing executable means the old `media-worker` image is
still running or its rebuild failed. A row-level failure includes the bounded
OCR reason in the script output.

The first live attempt completed queueing, OCR, indexing, reading, deletion,
and repair, but its synthetic all-caps bitmap caused Tesseract to split `TOTAL
FORTY TWO` across lines. The corrected DejaVu Sans fixture then passed: the PDF
moved from Pending to Ready in 4.619 seconds, produced one indexed chunk, the
reader returned 101 characters with its page marker and every required word,
and deletion plus repair completed cleanly.

After the automated smoke passes, the useful browser check is one real scanned
PDF: upload it in **Files**, watch Pending/Processing become Ready, choose
**View text**, and confirm the recognized words match the page. A normal PDF
with selectable text should still become Ready immediately.

## Completion gate

The automated Unraid smoke passed and the user accepted the result on
2026-09-30. Broader languages, handwriting, OCR correction, audio-only files,
and other document formats stay outside this completed slice.
