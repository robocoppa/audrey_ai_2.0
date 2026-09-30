"""The OCR live smoke must generate a true image-only PDF fixture."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from pypdf import PdfReader

SMOKE_DIR = Path(__file__).parent / "smoke"
sys.path.insert(0, str(SMOKE_DIR))
import smoke_scanned_pdf_ocr as smoke  # noqa: E402


def test_generated_fixture_has_one_page_and_no_text_layer():
    reader = PdfReader(io.BytesIO(smoke._scanned_pdf()))

    assert len(reader.pages) == 1
    assert not (reader.pages[0].extract_text() or "").strip()
    assert reader.pages[0].images


def test_required_words_are_ocr_friendly_and_meaningful():
    assert smoke._REQUIRED_WORDS == {
        "audrey",
        "scanned",
        "invoice",
        "forty",
        "dollars",
    }
