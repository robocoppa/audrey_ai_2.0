"""Bounded read-only projections of Audrey-generated files; no external viewer."""
from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path
from typing import BinaryIO

from openpyxl import load_workbook

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
MAX_SHEETS = 8
MAX_ROWS = 200
MAX_COLUMNS = 32
MAX_PARAGRAPHS = 1000


class FilePreviewError(ValueError):
    """Content-free failure; callers must not return parser exception text."""


def _text(value: str) -> tuple[str, bool]:
    return value[:MAX_TEXT_CHARS], len(value) > MAX_TEXT_CHARS


def _table(rows, *, budget: list[int]) -> tuple[list[list], bool]:
    visible, truncated = [], False
    for row in rows:
        if len(visible) >= MAX_ROWS or budget[0] <= 0:
            truncated = True
            break
        cells = []
        for value in row[:MAX_COLUMNS]:
            if value is None or isinstance(value, (bool, int)):
                cell = value
            elif isinstance(value, float) and math.isfinite(value):
                cell = value
            else:
                raw = str(value)
                cell = raw[:min(budget[0], 8000)]
                truncated |= len(cell) < len(raw)
                budget[0] -= len(cell)
            cells.append(cell)
        truncated |= len(row) > MAX_COLUMNS and any(value is not None for value in row[MAX_COLUMNS:])
        visible.append(cells)
    return visible, truncated


def workbook_preview(source: BinaryIO) -> dict:
    """Read values and inert formula text; never evaluate formulas or fetch links."""
    with zipfile.ZipFile(source) as package:
        entries = package.infolist()
        if (len(entries) > 1024 or sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES
                or any(entry.file_size > MAX_EXPANDED_BYTES // 2 for entry in entries)):
            raise FilePreviewError("Workbook is too large to preview.")
    source.seek(0)
    # data_only=False displays formulas as inert text if a retained export ever
    # includes one. It never evaluates them or follows workbook external links.
    book = load_workbook(source, read_only=True, data_only=False, keep_links=False)
    try:
        sheets, budget = [], [MAX_TEXT_CHARS]
        truncated = len(book.worksheets) > MAX_SHEETS
        for sheet in book.worksheets[:MAX_SHEETS]:
            # Do not trust dimensions inside a ZIP: generated write-only files
            # may omit them, and corrupted dimensions must not allocate a grid.
            sheet.reset_dimensions()
            rows, cut = _table(sheet.iter_rows(max_col=MAX_COLUMNS + 1, values_only=True), budget=budget)
            # iter_rows pads to max_col; trimming only trailing blank cells keeps
            # all leading/interior blanks and actual values in their columns.
            for row in rows:
                while row and row[-1] is None:
                    row.pop()
            sheets.append({"name": sheet.title, "rows": rows, "truncated": cut})
            truncated |= cut
        return {"format": "xlsx", "kind": "table", "sheets": sheets, "truncated": truncated,
                "notice": "Showing part of this file. Download it to see all contents." if truncated else ""}
    finally:
        book.close()


def preview_generated_file(path: Path, *, format: str, text_path: Path) -> dict:
    """Use saved renderer text for documents, original cells for spreadsheets."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise FilePreviewError("File is unavailable or too large to preview.")
    if format == "xlsx":
        with path.open("rb") as source:
            return workbook_preview(source)
    if format not in {"txt", "md", "json", "csv", "docx"}:
        raise FilePreviewError("This format cannot be previewed.")
    raw = ""
    if format != "csv":
        if text_path.is_symlink() or not text_path.is_file() or text_path.stat().st_size > MAX_SOURCE_BYTES:
            raise FilePreviewError("Saved document text is unavailable.")
        # Read a bounded stream even if a file changes unexpectedly after stat.
        with text_path.open(encoding="utf-8") as source:
            raw = source.read(MAX_TEXT_CHARS + 1)
    if format == "csv":
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            rows, cut = _table(reader, budget=[MAX_TEXT_CHARS])
        truncated = cut
        result = {"kind": "table", "sheets": [{"name": "Table", "rows": rows, "truncated": cut}]}
    elif format == "docx":
        value, truncated = _text(raw)
        paragraphs = value.split("\n\n")
        truncated |= len(paragraphs) > MAX_PARAGRAPHS
        result = {"kind": "document", "paragraphs": paragraphs[:MAX_PARAGRAPHS]}
    else:
        value, truncated = _text(raw)
        if format == "json" and not truncated:
            # Canonical renderer text is already pretty JSON; reformat only to
            # validate that an unexpectedly damaged sidecar is not called JSON.
            json.loads(value)
        result = {"kind": "text", "text": value}
    return {**result, "format": format, "truncated": truncated,
            "notice": "Showing part of this file. Download it to see all contents." if truncated else ""}
