"""Bounded whole-PDF text extraction for the offline public index builder."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import sys
from pathlib import Path

from audrey.references.library import MAX_PAGE_CHARS, MAX_PDF_BYTES, _safe_path

MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_CPU_SECONDS = 180
MAX_WALL_SECONDS = 210
READER_BATCH_PAGES = 16
ERROR_PREFIX = "AUDREY_REFERENCE_EXTRACT_ERROR "
ERROR_MESSAGES = {
    "password_required": "Reference PDF requires a password",
    "size_mismatch": "PDF size does not match its retained revision",
    "checksum_mismatch": "PDF checksum does not match its retained revision",
    "page_count_limit": "Unsupported oversized PDF",
    "page_text_limit": "PDF page exceeds its text limit",
    "output_limit": "PDF text exceeds its output limit",
    "request_limit": "PDF extraction request exceeds its input limit",
    "memory_limit": "PDF extraction exceeded its memory limit",
    "text_decode_error": "PDF text contains an unsupported character mapping",
    "parser_error": "PDF text extraction failed in the parser",
}


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
        # Public IRS PDFs sometimes set editing permissions while allowing anyone
        # to open them. Never request or guess a nonempty password.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("Reference PDF requires a password")
        if not 1 <= len(reader.pages) <= 10000:
            raise ValueError("Unsupported oversized PDF")
        total_pages = len(reader.pages)
        written = 0
        for number in range(1, total_pages + 1):
            if number > 1 and (number - 1) % READER_BATCH_PAGES == 0:
                # Reader object/font caches accumulate across large public PDFs.
                # Reopen the same verified descriptor, without changing the parser
                # or the extracted page text, before those caches hit the memory cap.
                reader.close()
                del reader
                gc.collect()
                source.seek(0)
                reader = PdfReader(source, strict=False)
                if reader.is_encrypted and not reader.decrypt(""):
                    raise ValueError(ERROR_MESSAGES["password_required"])
            text = reader.pages[number - 1].extract_text() or ""
            # Never publish silent partial extraction as complete coverage.
            if len(text) > MAX_PAGE_CHARS:
                raise ValueError("PDF page exceeds its text limit")
            line = (
                json.dumps(
                    {"page": number, "total_pages": total_pages, "text": text},
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
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES,) * 2)
    try:
        encoded = sys.stdin.buffer.read(8193)
        if len(encoded) > 8192:
            raise ValueError(ERROR_MESSAGES["request_limit"])
        extract(json.loads(encoded))
        return 0
    except Exception as error:  # noqa: BLE001 — only stable codes leave the child
        if isinstance(error, MemoryError):
            code = "memory_limit"
        elif isinstance(error, UnicodeDecodeError):
            code = "text_decode_error"
        elif isinstance(error, ValueError):
            code = next(
                (name for name, message in ERROR_MESSAGES.items() if str(error) == message),
                "parser_error",
            )
        else:
            code = "parser_error"
        sys.stderr.write(ERROR_PREFIX + json.dumps({"code": code}) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
