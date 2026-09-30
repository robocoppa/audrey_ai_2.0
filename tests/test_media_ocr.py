"""Scanned-PDF OCR driver and media-worker dispatch contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from audrey.media import ocr, worker
from audrey.media.ocr import OcrDocument, OcrFailedError, OcrUnavailableError

TOKEN = "not-a-real-token"  # noqa: S105


def _completed(command: list[str], *, stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


class TestOcrDriver:
    def test_it_renders_and_recognizes_one_page_at_a_time(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF fixture")
        work = tmp_path / "work"
        commands: list[list[str]] = []
        recognized = iter(["First page text\n", "Second page text\n"])
        monkeypatch.setattr(ocr, "_binary", lambda name: name)

        def fake_run(command: list[str], *, deadline):
            commands.append(command)
            if command[0] == "pdfinfo":
                return _completed(command, stdout="Pages:          2\n")
            if command[0] == "pdftoppm":
                Path(command[-1]).with_suffix(".png").write_bytes(b"png")
                return _completed(command)
            return _completed(command, stdout=next(recognized))

        monkeypatch.setattr(ocr, "_run", fake_run)
        result = ocr.ocr_pdf(source, work, language="eng", dpi=200, max_pages=2)

        assert result == OcrDocument(
            text="--- Page 1 ---\nFirst page text\n\n--- Page 2 ---\nSecond page text",
            pages=2,
            language="eng",
        )
        assert [command[0] for command in commands] == [
            "pdfinfo", "pdftoppm", "tesseract", "pdftoppm", "tesseract",
        ]
        assert not work.exists()

    def test_page_limit_fails_before_any_page_is_rendered(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "large.pdf"
        source.write_bytes(b"%PDF fixture")
        commands: list[list[str]] = []
        monkeypatch.setattr(ocr, "_binary", lambda name: name)

        def fake_run(command: list[str], *, deadline):
            commands.append(command)
            return _completed(command, stdout="Pages: 101\n")

        monkeypatch.setattr(ocr, "_run", fake_run)
        with pytest.raises(OcrFailedError, match="101 pages"):
            ocr.ocr_pdf(source, tmp_path / "work", max_pages=100)
        assert [command[0] for command in commands] == ["pdfinfo"]

    def test_missing_executable_is_an_image_failure_not_a_document_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF fixture")
        monkeypatch.setattr(ocr.shutil, "which", lambda _name: None)
        with pytest.raises(OcrUnavailableError, match="rebuild the media-worker"):
            ocr.ocr_pdf(source, tmp_path / "work")


class _Calls:
    def __init__(self):
        self.posted: list[tuple[str, dict | None]] = []

    def __call__(self, endpoint, path, token, body, **kwargs):
        self.posted.append((path, body))
        return 200, {}


class TestWorkerDispatch:
    def _job(self, source: Path) -> dict:
        return {
            "file_id": "doc1",
            "filename": "scan.pdf",
            "mime": "application/pdf",
            "kind": "text",
            "path": str(source),
            "lease_id": "L1",
            "user": "alice@example.com",
            "bytes": 123,
            "attempts": 1,
            "ocr": {
                "language": "eng",
                "dpi": 240,
                "max_pages": 12,
                "max_chars": 3456,
                "timeout_s": 78,
            },
        }

    def test_scanned_pdf_posts_typed_ocr_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF fixture")
        calls = _Calls()
        seen: dict = {}

        def fake_ocr(path, work_dir, **kwargs):
            seen.update(path=path, work_dir=work_dir, **kwargs)
            return OcrDocument(text="--- Page 1 ---\nInvoice total 42", pages=1, language="eng")

        monkeypatch.setattr(worker, "ocr_pdf", fake_ocr)
        monkeypatch.setattr(worker, "post", calls)
        worker.handle_job(
            self._job(source),
            endpoint="http://audrey:8000",
            token=TOKEN,
            work_dir=tmp_path / "work",
        )

        assert seen["dpi"] == 240
        assert seen["max_pages"] == 12
        assert seen["max_chars"] == 3456
        assert seen["timeout_s"] == 78.0
        assert calls.posted == [
            (
                "/v1/files/doc1/ocr-result",
                {
                    "lease_id": "L1",
                    "text": "--- Page 1 ---\nInvoice total 42",
                    "pages": 1,
                    "language": "eng",
                },
            ),
        ]

    def test_document_failure_is_written_to_the_row(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF fixture")
        calls = _Calls()
        monkeypatch.setattr(
            worker,
            "ocr_pdf",
            lambda *args, **kwargs: (_ for _ in ()).throw(OcrFailedError("too many pages")),
        )
        monkeypatch.setattr(worker, "post", calls)

        worker.handle_job(
            self._job(source), endpoint="http://x", token=TOKEN, work_dir=tmp_path / "w",
        )

        assert calls.posted[0][0] == "/v1/files/doc1/ingest-failed"
        assert calls.posted[0][1]["reason"] == "too many pages"

    def test_missing_ocr_binary_preserves_the_queued_document(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF fixture")
        calls = _Calls()
        monkeypatch.setattr(
            worker,
            "ocr_pdf",
            lambda *args, **kwargs: (_ for _ in ()).throw(OcrUnavailableError("missing")),
        )
        monkeypatch.setattr(worker, "post", calls)

        with pytest.raises(OcrUnavailableError):
            worker.handle_job(
                self._job(source), endpoint="http://x", token=TOKEN,
                work_dir=tmp_path / "w",
            )
        assert calls.posted == []
