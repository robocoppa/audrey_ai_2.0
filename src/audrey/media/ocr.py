"""Bounded CPU OCR for scanned PDFs in the media worker.

The worker renders and recognizes one page at a time so a large PDF cannot
fill its scratch volume with a complete raster copy.  Poppler and Tesseract
are command line dependencies baked into the worker image; keeping the Python
module stdlib-only preserves the worker's deliberately narrow dependency set.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DPI = 200
DEFAULT_LANGUAGE = "eng"
DEFAULT_MAX_CHARS = 5_000_000
DEFAULT_MAX_PAGES = 100
DEFAULT_TIMEOUT_S = 1200.0


class OcrUnavailableError(RuntimeError):
    """The worker image is missing a required OCR executable."""


class OcrFailedError(RuntimeError):
    """The document could not produce a complete, bounded OCR result."""


@dataclass(frozen=True, slots=True)
class OcrDocument:
    text: str
    pages: int
    language: str


def _tail(value: str, limit: int = 500) -> str:
    cleaned = value.strip()
    return cleaned[-limit:] if cleaned else "no diagnostic output"


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    value = deadline - time.monotonic()
    if value <= 0:
        raise OcrFailedError("OCR exceeded its total time budget")
    return value


def _run(command: list[str], *, deadline: float | None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_remaining(deadline),
        )
    except subprocess.TimeoutExpired as exc:
        raise OcrFailedError("OCR exceeded its total time budget") from exc
    except OSError as exc:
        raise OcrUnavailableError(f"could not start {command[0]}: {exc}") from exc


def _binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise OcrUnavailableError(
            f"{name} is not on PATH — rebuild the media-worker image with OCR tools",
        )
    return found


def _pdf_pages(source: Path, *, pdfinfo: str, deadline: float | None) -> int:
    result = _run([pdfinfo, str(source)], deadline=deadline)
    if result.returncode != 0:
        raise OcrFailedError(f"pdfinfo could not read the PDF: {_tail(result.stderr)}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise OcrFailedError("pdfinfo did not report a page count")
    pages = int(match.group(1))
    if pages < 1:
        raise OcrFailedError("the PDF has no pages")
    return pages


def ocr_pdf(
    source: Path,
    work_dir: Path,
    *,
    language: str = DEFAULT_LANGUAGE,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
) -> OcrDocument:
    """Return complete OCR text for one PDF or raise a typed error.

    Page headers make the plain-text artifact navigable without polluting the
    source PDF.  A blank page is omitted; a document whose every page is blank
    fails rather than becoming a ready file with zero searchable content.
    """
    if dpi < 72 or dpi > 600:
        raise ValueError("OCR dpi must be between 72 and 600")
    if max_pages < 1:
        raise ValueError("OCR max_pages must be positive")
    if max_chars < 1:
        raise ValueError("OCR max_chars must be positive")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("OCR timeout_s must be positive or None")
    language = language.strip()
    if not language or len(language) > 100:
        raise ValueError("OCR language must be 1-100 characters")

    pdfinfo = _binary("pdfinfo")
    pdftoppm = _binary("pdftoppm")
    tesseract = _binary("tesseract")
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None

    pages = _pdf_pages(source, pdfinfo=pdfinfo, deadline=deadline)
    if pages > max_pages:
        raise OcrFailedError(
            f"PDF has {pages} pages; OCR is limited to {max_pages} pages",
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[str] = []
    chars = 0
    try:
        for page in range(1, pages + 1):
            prefix = work_dir / f"page-{page:05d}"
            image = prefix.with_suffix(".png")
            rendered = _run(
                [
                    pdftoppm,
                    "-f", str(page),
                    "-l", str(page),
                    "-singlefile",
                    "-r", str(dpi),
                    "-png",
                    str(source),
                    str(prefix),
                ],
                deadline=deadline,
            )
            if rendered.returncode != 0 or not image.is_file():
                raise OcrFailedError(
                    f"could not render PDF page {page}: {_tail(rendered.stderr)}",
                )

            try:
                recognized = _run(
                    [tesseract, str(image), "stdout", "-l", language, "--psm", "3"],
                    deadline=deadline,
                )
            finally:
                image.unlink(missing_ok=True)
            if recognized.returncode != 0:
                raise OcrFailedError(
                    f"could not recognize PDF page {page}: {_tail(recognized.stderr)}",
                )

            text = recognized.stdout.strip()
            if not text:
                continue
            piece = f"--- Page {page} ---\n{text}"
            chars += len(piece) + (2 if pieces else 0)
            if chars > max_chars:
                raise OcrFailedError(
                    f"OCR text exceeds the {max_chars:,}-character limit",
                )
            pieces.append(piece)
    finally:
        if work_dir.exists():
            for child in work_dir.iterdir():
                child.unlink(missing_ok=True)
            work_dir.rmdir()

    if not pieces:
        raise OcrFailedError("OCR found no text in the PDF")
    return OcrDocument(text="\n\n".join(pieces), pages=pages, language=language)


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_TIMEOUT_S",
    "OcrDocument",
    "OcrFailedError",
    "OcrUnavailableError",
    "ocr_pdf",
]
