"""Bounded rendering of one owned PDF page or video frame."""
from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import tempfile
from pathlib import Path

MAX_IMAGE_BYTES = 8 * 1024 * 1024
RENDER_TIMEOUT_S = 25


class VisualRenderError(ValueError):
    """A source could not produce the requested visual evidence."""


async def _run(*args: str, output: Path | None = None) -> None:
    if shutil.which(args[0]) is None:
        raise VisualRenderError(f"{args[0]} is unavailable")
    with (output.open("wb") if output else contextlib.nullcontext(asyncio.subprocess.DEVNULL)) as stream:
        process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL, stdout=stream,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=RENDER_TIMEOUT_S)
        except BaseException:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
        if process.returncode:
            raise VisualRenderError("The source could not render the requested page or frame.")


def _read_image(path: Path) -> bytes:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise VisualRenderError("The requested page or frame produced no usable bounded image.")
    return path.read_bytes()


async def render_pdf_page(path: Path, page: int) -> tuple[bytes, int]:
    """Return PNG bytes and total page count; page numbers start at one."""
    with tempfile.TemporaryDirectory(prefix="audrey-pdf-page-") as directory:
        root = Path(directory)
        info = root / "info.txt"
        await _run("pdfinfo", str(path), output=info)
        if info.stat().st_size > 65536:
            raise VisualRenderError("PDF metadata exceeds the inspection limit.")
        match = re.search(r"^Pages:\s+(\d+)\s*$", info.read_text(errors="replace"), re.M)
        if not match:
            raise VisualRenderError("Could not determine the PDF page count.")
        total = int(match.group(1))
        if not 1 <= page <= total:
            raise VisualRenderError(f"Page must be between 1 and {total}.")
        await _run(
            "pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
            "-scale-to", "1600", "-png", str(path), str(root / "page"),
        )
        return _read_image(root / "page.png"), total


async def render_video_frame(path: Path, timestamp_s: float) -> bytes:
    """Decode one frame near the requested position without network protocols."""
    with tempfile.TemporaryDirectory(prefix="audrey-video-frame-") as directory:
        image = Path(directory) / "frame.jpg"
        await _run(
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-ss", str(timestamp_s),
            "-i", str(path), "-frames:v", "1",
            "-vf", "scale=1280:1280:force_original_aspect_ratio=decrease",
            "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1", output=image,
        )
        return _read_image(image)
