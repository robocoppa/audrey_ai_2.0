"""Real encrypted-public PDF extraction keeps its identity and resource gates."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter
from test_reference_library import _pdf

from audrey.references import extract as module


def _payload(tmp_path, content):
    source = tmp_path / "reference.pdf"
    source.write_bytes(content)
    return {"root": str(tmp_path), "relative_path": source.name,
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _worker(payload):
    return subprocess.run(
        [sys.executable, "-m", "audrey.references.extract"],
        input=json.dumps(payload).encode(), capture_output=True, timeout=15, check=False,
    )


@pytest.mark.parametrize("password", ["", "required-password"])
def test_bounded_worker_opens_only_blank_password_pdf(tmp_path, password):
    result = _worker(_payload(tmp_path, _pdf("PUBLIC_PDF_TEXT", user_password=password)))
    if password:
        assert result.returncode != 0
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        pages = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(pages) == 2
        assert pages[0]["text"].strip() == "PUBLIC_PDF_TEXT"
        assert pages[1]["page"] == 2


@pytest.mark.parametrize("field,value", [("bytes", 1), ("sha256", "0" * 64)])
def test_public_permissions_pdf_still_requires_retained_identity(tmp_path, field, value):
    payload = _payload(tmp_path, _pdf(user_password=""))
    payload[field] = value
    result = _worker(payload)
    assert result.returncode != 0
    assert not result.stdout


def test_public_permissions_pdf_does_not_bypass_page_text_limit(tmp_path):
    content = _pdf("a" * (module.MAX_PAGE_CHARS + 1), user_password="")
    result = _worker(_payload(tmp_path, content))
    assert result.returncode != 0
    assert not result.stdout


def test_public_permissions_pdf_does_not_bypass_total_output_limit(tmp_path, monkeypatch):
    payload = _payload(tmp_path, _pdf("PUBLIC_PDF_TEXT", user_password=""))
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="output limit"):
        module.extract(payload)


@pytest.mark.parametrize(
    "error,code",
    [
        (MemoryError("PRIVATE_SOURCE_MARKER"), "memory_limit"),
        (UnicodeDecodeError("utf-16-be", b"x", 0, 1, "PRIVATE_SOURCE_MARKER"),
         "text_decode_error"),
        (ValueError(module.ERROR_MESSAGES["password_required"]), "password_required"),
        (ValueError("PRIVATE_SOURCE_MARKER"), "parser_error"),
        (RuntimeError("PRIVATE_SOURCE_MARKER"), "parser_error"),
    ],
)
def test_child_diagnostic_uses_stable_code_without_parser_details(
    monkeypatch, capsys, error, code,
):
    monkeypatch.setattr(module.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(module.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))

    def fail(_):
        raise error

    monkeypatch.setattr(module, "extract", fail)
    assert module.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == module.ERROR_PREFIX + json.dumps({"code": code}) + "\n"
    assert "PRIVATE_SOURCE_MARKER" not in captured.err


def test_reopened_reader_keeps_page_text_and_blank_password_support(tmp_path):
    template = PdfReader(io.BytesIO(_pdf("FIRST_PAGE_TEXT", second_page="SECOND_PAGE_TEXT")))
    writer = PdfWriter()
    count = module.READER_BATCH_PAGES * 2 + 3
    for index in range(count):
        writer.add_page(template.pages[index % 2])
    writer.encrypt("", owner_password="public-pdf-owner")  # noqa: S106
    output = io.BytesIO()
    writer.write(output)
    content = output.getvalue()
    original = PdfReader(io.BytesIO(content))
    expected_text = [page.extract_text() for page in original.pages]
    result = _worker(_payload(tmp_path, content))
    assert result.returncode == 0, result.stderr
    pages = [json.loads(line) for line in result.stdout.splitlines()]
    assert [page["page"] for page in pages] == list(range(1, count + 1))
    assert all(page["total_pages"] == count for page in pages)
    assert [page["text"] for page in pages] == expected_text
