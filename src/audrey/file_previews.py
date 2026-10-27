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
MAX_ROW_OFFSET = 10_000


class FilePreviewError(ValueError):
    """Content-free failure; callers must not return parser exception text."""


def _options(sheet: int | None, row_offset: int, text_offset: int = 0) -> None:
    if ((sheet is not None and (type(sheet) is not int or not 0 <= sheet < MAX_SHEETS))
            or type(row_offset) is not int or not 0 <= row_offset <= MAX_ROW_OFFSET
            or type(text_offset) is not int or not 0 <= text_offset <= MAX_SOURCE_BYTES):
        raise FilePreviewError("Invalid preview position.")


def _table(rows, *, budget: list[int], row_offset: int = 0) -> dict:
    visible, content_truncated, more = [], False, False
    total_rows, observed_rows = None, 0
    for index, row in enumerate(rows):
        observed_rows = index + 1
        if index < row_offset:
            continue
        if len(visible) >= MAX_ROWS or budget[0] <= 0:
            more = True
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
                content_truncated |= len(cell) < len(raw)
                budget[0] -= len(cell)
            cells.append(cell)
        content_truncated |= len(row) > MAX_COLUMNS and any(value is not None for value in row[MAX_COLUMNS:])
        visible.append(cells)
    else:
        # Only report an exact count after observing EOF, rather than trusting
        # workbook dimensions or forcing a second full scan for each page.
        total_rows = observed_rows
    next_row = row_offset + len(visible) if more else None
    if next_row is not None and next_row > MAX_ROW_OFFSET:
        next_row, content_truncated = None, True
    return {"rows": visible, "row_offset": row_offset, "next_row_offset": next_row,
            "total_rows": total_rows, "content_truncated": content_truncated,
            "truncated": row_offset > 0 or more or content_truncated}


def _notice(truncated: bool) -> str:
    return "Showing part of this file. Download it to see all contents." if truncated else ""


def workbook_preview(source: BinaryIO, *, sheet: int | None = None, row_offset: int = 0) -> dict:
    """Read values and inert formula text; never evaluate formulas or fetch links."""
    _options(sheet, row_offset)
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
        if sheet is not None and sheet >= len(book.worksheets):
            raise FilePreviewError("Worksheet is unavailable.")
        sheets, budget = [], [MAX_TEXT_CHARS]
        content_truncated = len(book.worksheets) > MAX_SHEETS
        truncated = content_truncated
        for index, worksheet in enumerate(book.worksheets[:MAX_SHEETS]):
            if sheet is not None and sheet != index:
                continue
            # Do not trust dimensions inside a ZIP: generated write-only files
            # may omit them, and corrupted dimensions must not allocate a grid.
            worksheet.reset_dimensions()
            page = _table(worksheet.iter_rows(max_col=MAX_COLUMNS + 1, values_only=True),
                          budget=budget, row_offset=row_offset)
            # Trimming trailing blank cells preserves interior column positions.
            for row in page['rows']:
                while row and row[-1] is None:
                    row.pop()
            sheets.append({"name": worksheet.title, "sheet_index": index, **page})
            truncated |= page['truncated']
            content_truncated |= page['content_truncated']
        return {"format": "xlsx", "kind": "table", "sheets": sheets,
                "sheet_count": len(book.worksheets), "truncated": truncated,
                "content_truncated": content_truncated, "notice": _notice(truncated)}
    finally:
        book.close()


def preview_generated_file(path: Path, *, format: str, text_path: Path,
                           sheet: int | None = None, row_offset: int = 0, text_offset: int = 0) -> dict:
    """Use saved renderer text for documents, original cells for spreadsheets."""
    _options(sheet, row_offset, text_offset)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise FilePreviewError("File is unavailable or too large to preview.")
    if format == "xlsx":
        with path.open("rb") as source:
            return workbook_preview(source, sheet=sheet, row_offset=row_offset)
    if format not in {"txt", "md", "json", "csv", "docx"}:
        raise FilePreviewError("This format cannot be previewed.")
    if format == "csv":
        if sheet not in (None, 0):
            raise FilePreviewError("Worksheet is unavailable.")
        with path.open(encoding="utf-8-sig", newline="") as source:
            page = _table(csv.reader(source), budget=[MAX_TEXT_CHARS], row_offset=row_offset)
        return {"format": format, "kind": "table", "sheet_count": 1,
                "sheets": [{"name": "Table", "sheet_index": 0, **page}],
                "truncated": page['truncated'], "content_truncated": page['content_truncated'],
                "notice": _notice(page['truncated'])}
    if text_path.is_symlink() or not text_path.is_file() or text_path.stat().st_size > MAX_SOURCE_BYTES:
        raise FilePreviewError("Saved document text is unavailable.")
    # Read no more than the maximum source size even if a sidecar changes after
    # stat. Character offsets remain exact for UTF-8, including non-ASCII text.
    with text_path.open(encoding="utf-8") as source:
        raw = source.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise FilePreviewError("Saved document text is too large to preview.")
    end = min(text_offset + MAX_TEXT_CHARS, len(raw))
    if format == "docx":
        paragraphs = raw[text_offset:end].split("\n\n")
        if len(paragraphs) > MAX_PARAGRAPHS:
            # Include the separator in this page so the next starts at the next
            # paragraph; returning exact raw text lets clients join pages safely.
            end = text_offset + sum(len(part) for part in paragraphs[:MAX_PARAGRAPHS]) + 2 * MAX_PARAGRAPHS
    value = raw[text_offset:end]
    next_offset = end if end < len(raw) else None
    truncated = text_offset > 0 or next_offset is not None
    result = {"kind": "text", "text": value}
    if format == "docx":
        paragraphs = value.split("\n\n")
        if len(paragraphs) > MAX_PARAGRAPHS and paragraphs[-1] == "":
            paragraphs.pop()
        result = {"kind": "document", "text": value, "paragraphs": paragraphs}
    elif format == "json":
        # Validate the bounded original once even for a middle page; fragments
        # need not be valid standalone JSON and must not be silently reformatted.
        json.loads(raw)
    return {**result, "format": format, "offset": text_offset, "next_offset": next_offset,
            "total_chars": len(raw), "truncated": truncated, "content_truncated": False,
            "notice": _notice(truncated)}
