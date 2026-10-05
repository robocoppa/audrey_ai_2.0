"""Disposable parser for untrusted remote documents and images.

Invoked by absolute script path so the HTTP route package is never imported.
The worker has CPU/address-space limits; the parent also enforces a wall limit.
"""
from __future__ import annotations

import base64
import io
import json
import resource
import sys
from pathlib import Path
from zipfile import ZipFile


def parse(path: Path, kind: str, declared: str, max_chars: int) -> dict:
    import magic

    mime = magic.from_file(str(path), mime=True)
    if kind == "image":
        from PIL import Image, ImageOps

        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Remote image must contain JPEG, PNG, or WEBP bytes.")
        if declared not in {"", "application/octet-stream", mime}:
            raise ValueError("Remote image Content-Type does not match its bytes.")
        with Image.open(path) as source:
            if source.width * source.height > 50_000_000:
                raise OverflowError("Remote image exceeds the pixel limit.")
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            rgba = image.convert("RGBA")
            preview = Image.new("RGB", rgba.size, "white")
            preview.paste(rgba, mask=rgba.getchannel("A"))
            output = io.BytesIO()
            preview.save(output, format="JPEG", quality=85)
            return {"image": base64.b64encode(output.getvalue()).decode("ascii")}

    text_mimes = {"text/plain", "text/markdown", "text/x-markdown", "text/csv", "text/html", "text/x-rst", "application/xhtml+xml"}
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if mime == "application/pdf":
        suffix = ".pdf"
        allowed = {mime}
    elif mime in {docx_mime, "application/zip"}:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(e.file_size for e in entries) > 32 * 1024 * 1024:
                raise OverflowError("Remote DOCX exceeds the expanded archive limit.")
            if "word/document.xml" not in archive.namelist():
                raise ValueError("Remote archive is not a DOCX document.")
        suffix, allowed = ".docx", {docx_mime}
    elif mime in text_mimes:
        suffix = ".html" if mime in {"text/html", "application/xhtml+xml"} or declared in {"text/html", "application/xhtml+xml"} else ".txt"
        allowed = text_mimes
    else:
        raise ValueError("Remote file does not contain a supported document.")
    if declared not in {"", "application/octet-stream", *allowed}:
        raise ValueError("Remote document Content-Type does not match its bytes.")
    from audrey.kb.chunk import load_text

    source = path.with_suffix(suffix)
    path.rename(source)
    text = load_text(source)
    if not text or not text.strip():
        raise ValueError("Remote document has no extractable text; upload scanned PDFs for OCR.")
    if len(text) > max_chars:
        raise OverflowError("Remote document exceeds the character limit.")
    return {"text": text}


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        result = parse(Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
    except (MemoryError, OverflowError):
        result = {"error": "Remote input exceeds the extraction resource limit.", "status": 413}
    except Exception:  # noqa: BLE001 - parser failures must not disclose source or internals
        result = {"error": "Remote input is unsupported, corrupt, or has no extractable text. Upload scanned PDFs for OCR.", "status": 422}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
