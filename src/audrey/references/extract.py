"""Bounded whole-PDF text extraction for the offline public index builder."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
from pathlib import Path

from audrey.references.library import MAX_PAGE_CHARS, MAX_PDF_BYTES, _safe_path

MAX_OUTPUT_BYTES = 32 * 1024 * 1024


def extract(payload: dict) -> None:
    from pypdf import PdfReader

    path = _safe_path(Path(payload["root"]), payload["relative_path"])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size != payload["bytes"] or not 0 < size <= MAX_PDF_BYTES:
            raise ValueError("PDF size does not match its retained revision")
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != payload["sha256"]:
            raise ValueError("PDF checksum does not match its retained revision")
        source.seek(0)
        reader = PdfReader(source, strict=False)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 10000:
            raise ValueError("Unsupported encrypted or oversized PDF")
        written = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            # Never publish silent partial extraction as complete coverage.
            if len(text) > MAX_PAGE_CHARS:
                raise ValueError("PDF page exceeds its text limit")
            line = (
                json.dumps(
                    {"page": number, "total_pages": len(reader.pages), "text": text},
                    ensure_ascii=False,
                ).encode()
                + b"\n"
            )
            written += len(line)
            if written > MAX_OUTPUT_BYTES:
                raise ValueError("PDF text exceeds its output limit")
            sys.stdout.buffer.write(line)


def main() -> int:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES,) * 2)
    try:
        encoded = sys.stdin.buffer.read(8193)
        if len(encoded) > 8192:
            return 2
        extract(json.loads(encoded))
        return 0
    except Exception:  # noqa: BLE001 — parser internals never leave the child
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
