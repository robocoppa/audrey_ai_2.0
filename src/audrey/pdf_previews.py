"""Local, bounded PDF browser pages, sharing the existing Poppler runner."""
from __future__ import annotations

import asyncio
import io
import re
import stat
import tempfile
from pathlib import Path

from PIL import Image

from audrey.media import visual_render

MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_PAGES = 2000
MAX_PAGE_EDGE = 1600
PREVIEW_TIMEOUT_S = 45
_slots = asyncio.Semaphore(2)


class PDFPreviewError(ValueError):
    """A PDF cannot safely be shown; never expose parser messages to clients."""


class PDFPreviewUnavailableError(PDFPreviewError):
    """The installed renderer is unavailable."""


async def _disk(function, *args):
    # Keep the caller's file lock until any disk worker settles on cancellation.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


def _source_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or not 0 < value.st_size <= MAX_SOURCE_BYTES:
        raise PDFPreviewError("PDF source is unavailable or exceeds the preview limit.")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise PDFPreviewError("Not a PDF source.")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _page_count(info: Path) -> int:
    if info.stat().st_size > 65536:
        raise PDFPreviewError("PDF metadata exceeds the preview limit.")
    text = info.read_text(errors="replace")
    if re.search(r"^Encrypted:\s+yes\b", text, re.M | re.I):
        raise PDFPreviewError("Encrypted PDF preview is unavailable.")
    match = re.search(r"^Pages:\s+(\d+)\s*$", text, re.M)
    if not match or not 0 < int(match.group(1)) <= MAX_PAGES:
        raise PDFPreviewError("PDF page count exceeds the preview limit.")
    return int(match.group(1))


def _page_image(path: Path) -> bytes:
    content = visual_render._read_image(path)
    with Image.open(io.BytesIO(content)) as image:
        if (image.format != "PNG" or min(image.size) < 1
                or max(image.size) > MAX_PAGE_EDGE):
            raise PDFPreviewError("PDF page exceeds the preview pixel limit.")
        image.verify()
    return content


async def pdf_preview(path: Path, *, page: int | None = None) -> dict | bytes:
    """Return safe metadata or one inert PNG, never the PDF's active content.

    The source remains under the caller's operation lock. The shared command
    runner kills and reaps Poppler on cancellation/timeout; every temporary
    output is removed before this function returns or raises.
    """
    if page is not None and (type(page) is not int or not 1 <= page <= MAX_PAGES):
        raise PDFPreviewError("Invalid PDF page.")
    try:
        async with asyncio.timeout(PREVIEW_TIMEOUT_S), _slots:
            before = await _disk(_source_identity, path)
            with tempfile.TemporaryDirectory(prefix="audrey-pdf-preview-") as directory:
                root = Path(directory)
                info = root / "info.txt"
                await visual_render._run("pdfinfo", str(path), output=info)
                count = await _disk(_page_count, info)
                if page is not None:
                    if page > count:
                        raise PDFPreviewError("Invalid PDF page.")
                    await visual_render._run(
                        "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                        "-scale-to", str(MAX_PAGE_EDGE), "-png", str(path), str(root / "page"),
                    )
                    result = await _disk(_page_image, root / "page.png")
                else:
                    result = {"format": "pdf", "kind": "pdf", "page_count": count,
                              "truncated": False}
                if before != await _disk(_source_identity, path):
                    raise PDFPreviewError("PDF source changed during preview.")
                return result
    except visual_render.VisualRenderError as error:
        if "unavailable" in str(error):
            raise PDFPreviewUnavailableError("PDF renderer is unavailable.") from None
        raise PDFPreviewError("PDF could not be rendered.") from None
