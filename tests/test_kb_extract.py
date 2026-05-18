"""Tests for `extract_text` empty-extraction error wording.

`EmptyExtractionError` is the upload route's signal to 422 with a clear
message. The hint text now branches on suffix so a `.txt` doesn't get
told the file is "a scanned PDF without a text layer".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audrey.kb.extract import EmptyExtractionError, extract_text


def test_extract_text_empty_pdf_mentions_scanned_pdfs(tmp_path: Path, monkeypatch):
    # We don't want a real PDF here — short-circuit load_text to return ""
    # so the empty branch runs deterministically.
    from audrey.kb import extract as extract_mod

    monkeypatch.setattr(extract_mod, "load_text", lambda _: "")
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")  # contents irrelevant; load_text is stubbed

    with pytest.raises(EmptyExtractionError) as exc:
        extract_text(f)
    assert "scanned PDFs" in str(exc.value)
    assert "empty.pdf" in str(exc.value)


@pytest.mark.parametrize("suffix", [".txt", ".md", ".html", ".docx", ".csv"])
def test_extract_text_empty_non_pdf_uses_generic_hint(tmp_path: Path, monkeypatch, suffix):
    from audrey.kb import extract as extract_mod

    monkeypatch.setattr(extract_mod, "load_text", lambda _: "")
    f = tmp_path / f"empty{suffix}"
    f.write_text("", encoding="utf-8")

    with pytest.raises(EmptyExtractionError) as exc:
        extract_text(f)
    msg = str(exc.value)
    assert "scanned PDFs" not in msg
    assert "empty or contained no extractable text" in msg
    assert f"empty{suffix}" in msg


def test_extract_text_returns_loaded_content(tmp_path: Path, monkeypatch):
    # Happy path — non-empty loader output passes through unchanged.
    from audrey.kb import extract as extract_mod

    monkeypatch.setattr(extract_mod, "load_text", lambda _: "hello world")
    f = tmp_path / "x.md"
    f.write_text("hello world", encoding="utf-8")

    assert extract_text(f) == "hello world"


def test_extract_text_treats_whitespace_only_as_empty(tmp_path: Path, monkeypatch):
    # `.strip()` falsiness is the contract — a file containing only newlines
    # is the same as an empty extraction.
    from audrey.kb import extract as extract_mod

    monkeypatch.setattr(extract_mod, "load_text", lambda _: "   \n\n\t")
    f = tmp_path / "blank.txt"
    f.write_text("", encoding="utf-8")

    with pytest.raises(EmptyExtractionError):
        extract_text(f)
