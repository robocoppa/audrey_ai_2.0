"""Upload-side extraction (Phase 13).

Given a file on disk (already saved by the upload route to
`/data/uploads/<sanitized_user>/<uuid><ext>`), return the extracted text
or raise a typed error. Thin layer over `kb.chunk.load_text` plus:

  - libmagic mime sniffing (trust the bytes, not the client-declared
    Content-Type or filename extension).
  - Explicit empty-extraction error so the route can 422 a scanned PDF
    with a clear message, instead of silently writing zero chunks.

Mime whitelist lives in `ALLOWED_MIMES`; any sniffed mime outside it is
rejected with `UnsupportedMimeError`. The route layer does a cheap
extension pre-check before saving, but the real gate is here — extension
is a hint, sniffing is the truth.

Images take the same sniff path (so we can reject `a.png.exe`-style
tricks) but don't go through `load_text`; the route hands them to the
CLIP embedder directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from audrey.kb.chunk import load_text

log = logging.getLogger(__name__)

# Content types we'll ingest. Keep short and explicit; expand as needed.
ALLOWED_TEXT_MIMES: frozenset[str] = frozenset({
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "text/html",
    "text/x-rst",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
})
ALLOWED_IMAGE_MIMES: frozenset[str] = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff",
})
ALLOWED_MIMES: frozenset[str] = ALLOWED_TEXT_MIMES | ALLOWED_IMAGE_MIMES


class ExtractError(Exception):
    """Base for upload extraction errors."""


class UnsupportedMimeError(ExtractError):
    """Sniffed mime isn't in the allow-list."""


class EmptyExtractionError(ExtractError):
    """File parsed fine but contained no extractable text (likely a scanned PDF)."""


def sniff_mime(path: Path) -> str:
    """Return the sniffed mime type. Falls back to extension-derived guess if libmagic is unavailable."""
    try:
        import magic  # type: ignore

        return magic.from_file(str(path), mime=True) or ""
    except ImportError:
        log.warning("kb.extract: python-magic not installed; falling back to suffix-based mime guess")
        return _guess_from_suffix(path)
    except Exception as e:  # noqa: BLE001
        log.warning("kb.extract: libmagic sniff failed for %s: %s — falling back to suffix", path, e)
        return _guess_from_suffix(path)


def _guess_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain", ".log": "text/plain",
        ".md": "text/markdown", ".rst": "text/x-rst",
        ".csv": "text/csv",
        ".html": "text/html", ".htm": "text/html",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def is_image_mime(mime: str) -> bool:
    return mime in ALLOWED_IMAGE_MIMES


def is_text_mime(mime: str) -> bool:
    return mime in ALLOWED_TEXT_MIMES


def extract_text(path: Path) -> str:
    """Run the appropriate loader for `path`. Raises EmptyExtractionError on no-op extracts.

    The caller is expected to have already validated mime via sniff_mime +
    ALLOWED_TEXT_MIMES. We don't re-sniff here to keep this a pure CPU-bound
    function the route can run in a thread.
    """
    raw = load_text(path)
    if raw is None or not raw.strip():
        raise EmptyExtractionError(
            f"no extractable text from {path.name}; "
            "scanned PDFs without a text layer are not yet supported"
        )
    return raw


__all__ = [
    "ALLOWED_MIMES", "ALLOWED_TEXT_MIMES", "ALLOWED_IMAGE_MIMES",
    "ExtractError", "UnsupportedMimeError", "EmptyExtractionError",
    "sniff_mime", "is_image_mime", "is_text_mime", "extract_text",
]
