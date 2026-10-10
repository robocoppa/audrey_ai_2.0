# Campaign 4 Phase 07 — scanned PDF OCR and presentation

**Status:** Complete and accepted.

## Processing contract

PDFs with a usable text layer keep the normal extraction path. Image-only PDFs enter the existing durable media queue and return Pending promptly. The media worker renders one page at a time with Poppler and recognizes English text with Tesseract. Audrey validates the active lease, writes the owner-scoped OCR sidecar, indexes it, and reports Ready only after indexing succeeds.

Audrey remains the SQLite/Qdrant writer. Original bytes remain the quota/download object; the OCR sidecar supplies document text to My Files, `get_file_text`, retrieval, and grounded chat. Deletion uses the existing tombstone/sidecar repair lifecycle.

## Bounds and failures

`kb.ocr` owns enablement, English language `eng`, 200 DPI (admitted range 72–600), 100 pages, 5,000,000 recognized characters, and a 1,200-second whole-document budget within the shared lease. Pages are removed after recognition.

Corrupt, blank, oversized, or timed-out documents show a failed state and reason. Missing worker executables are infrastructure failures and do not burn all file attempts. Disabling OCR restores image-only PDF rejection.

## Document viewer

Newly processed PDFs open on a natural two or three sentence Summary; Transcript exposes their complete extracted/OCR text. Summary generation is fail-soft: a useful indexed document remains Ready if summary generation fails. Existing Ready PDFs are not automatically backfilled.

No further work is scheduled for this phase.
