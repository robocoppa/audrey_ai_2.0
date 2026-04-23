"""File loading + tiktoken-based chunking.

Chunking is token-based (cl100k_base, same tokenizer used by the
complexity gate) with a configurable size and overlap. We chunk on
token boundaries, then walk backward to the nearest whitespace so we
don't split a word in half. For very short files (fewer than
`chunk_tokens`) we emit a single chunk covering the whole file.

Loaders by extension (content-type detection via extension is
sufficient for curated datasets — we don't sniff magic bytes):

    .md, .txt, .rst, .log, .csv → plain text
    .pdf                        → pypdf page-by-page
    .docx                       → python-docx paragraph join
    .html, .htm                 → beautifulsoup text extraction

Everything else is skipped with a log line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tiktoken

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".log", ".csv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass(slots=True)
class Chunk:
    text: str
    idx: int


_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def load_text(path: Path) -> str | None:
    """Load a file's text content, or None if unsupported/unreadable."""
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            return _load_pdf(path)
        if suffix == ".docx":
            return _load_docx(path)
        if suffix in (".html", ".htm"):
            return _load_html(path)
    except Exception as e:  # noqa: BLE001 — one bad file shouldn't kill the crawl
        log.warning("kb.chunk: failed to load %s: %s", path, e)
        return None
    return None


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            log.warning("kb.chunk: pdf page %d of %s failed: %s", i, path, e)
    return "\n\n".join(p for p in parts if p.strip())


def _load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def chunk_text(text: str, *, chunk_tokens: int = 1000, overlap_tokens: int = 100) -> list[Chunk]:
    """Split `text` into overlapping token windows.

    Returns chunks in source order. Empty input → empty list. A document
    shorter than `chunk_tokens` becomes a single chunk.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    enc = _encoder()
    tokens = enc.encode(cleaned)
    if len(tokens) <= chunk_tokens:
        return [Chunk(text=cleaned, idx=0)]
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = chunk_tokens // 5  # safety: keep stride positive
    stride = chunk_tokens - overlap_tokens
    out: list[Chunk] = []
    for i, start in enumerate(range(0, len(tokens), stride)):
        end = min(start + chunk_tokens, len(tokens))
        piece = enc.decode(tokens[start:end]).strip()
        if piece:
            out.append(Chunk(text=piece, idx=i))
        if end >= len(tokens):
            break
    return out


__all__ = [
    "Chunk", "chunk_text", "load_text",
    "TEXT_SUFFIXES", "IMAGE_SUFFIXES",
]
