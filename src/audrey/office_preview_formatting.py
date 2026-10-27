"""Safe, bounded formatting read from retained Office bytes, not model metadata."""
from __future__ import annotations

import math
import re
import zipfile
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import BinaryIO

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl.xml.functions import iterparse

MAX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_COLUMNS = 32
MAX_TABLE_ROWS = 200
MAX_RICH_NODES = 20_000
_HEADER_FILLS = {'16324F', '203D60'}
_BANDED_FILLS = {'F2F6FA'}


def bounded_package(source: BinaryIO) -> None:
    with zipfile.ZipFile(source) as package:
        entries = package.infolist()
        if (len(entries) > 1024 or sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES
                or any(entry.file_size > MAX_EXPANDED_BYTES // 2 for entry in entries)):
            # Imported only while called to keep the limits shared with the
            # public preview exception without a module initialization cycle.
            from audrey.file_previews import FilePreviewError
            raise FilePreviewError('Office file is too large to preview.')
    source.seek(0)


def _formatted_number(value, number_format: str) -> str | None:
    """Project a fixed set of literal formats; never interpret arbitrary formats."""
    formats = {'#,##0': (0, False, ''), '#,##0.00': (2, False, ''),
               '$#,##0.00': (2, False, '$'), '0.00%': (2, True, '')}
    if (number_format not in formats or isinstance(value, bool)
            or not isinstance(value, (int, float))):
        return None
    digits, percent, prefix = formats[number_format]
    try:
        with localcontext() as context:
            context.prec = 400
            number = Decimal(str(value))
            if not number.is_finite():
                return None
            if percent:
                number *= 100
            number = number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
            negative = number.is_signed() and number != 0
            return ('-' if negative else '') + prefix + format(abs(number), f',.{digits}f') + ('%' if percent else '')
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _rgb(color) -> str:
    return str(color.rgb)[-6:].upper() if color is not None and color.type == 'rgb' else ''


def cell_style(source, value) -> dict:
    style = {}
    font = source.font
    if font is not None:
        if font.bold:
            style['bold'] = True
        if font.italic:
            style['italic'] = True
    fill = source.fill
    color = _rgb(fill.fgColor) if fill is not None and fill.patternType == 'solid' else ''
    if color in _HEADER_FILLS:
        style['header'] = True
    elif color in _BANDED_FILLS:
        style['banded'] = True
    display = _formatted_number(value, source.number_format)
    if display is not None:
        style['display'] = display
    return style


def column_widths(source: BinaryIO, worksheet) -> list[float]:
    """Read only column declarations, absent from openpyxl's read-only sheet."""
    widths = []
    source.seek(0)
    with zipfile.ZipFile(source) as package:
        # The package path was resolved by openpyxl, never supplied by a model;
        # ZIP members are read in place, not extracted to filesystem paths.
        with package.open(worksheet._worksheet_path) as data:
            for _, element in iterparse(data, events=('start',)):
                name = element.tag.rsplit('}', 1)[-1]
                if name == 'sheetData':
                    break
                if name != 'col':
                    continue
                try:
                    first, last = int(element.get('min', '0')), int(element.get('max', '0'))
                    width = float(element.get('width', '0'))
                except ValueError:
                    continue
                if not (1 <= first <= last and first <= MAX_COLUMNS and math.isfinite(width) and width > 0):
                    continue
                while len(widths) < min(last, MAX_COLUMNS):
                    widths.append(13)
                for index in range(first - 1, min(last, MAX_COLUMNS)):
                    widths[index] = round(max(8, min(60, width)), 2)
    return widths


def _run_style(run, paragraph) -> dict:
    """Expose semantic emphasis only; never forward arbitrary office styles."""
    style = {'text': run.text}
    for name in ('bold', 'italic'):
        value = getattr(run, name)
        if value is None:
            value = getattr(run.style.font, name)
        if value is None:
            value = getattr(paragraph.style.font, name)
        if value:
            style[name] = True
    if run.style.name.casefold() in {'inlinecode', 'inline code'}:
        style['code'] = True
    return style


def _paragraph_block(paragraph: Paragraph, budget: list[int]) -> dict:
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError('Document formatting exceeds the preview limit.')
    runs = []
    for run in paragraph.runs:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError('Document formatting exceeds the preview limit.')
        if run.text:
            runs.append(_run_style(run, paragraph))
    if ''.join(run['text'] for run in runs) != paragraph.text:
        # Unsupported inline nodes must not silently lose their textual content.
        runs = [{'text': paragraph.text}]
    name = paragraph.style.name.casefold().replace(' ', '')
    heading = re.fullmatch(r'heading([123])', name)
    if heading:
        return {'kind': 'heading', 'level': int(heading[1]), 'runs': runs}
    if name in {'codeblock', 'code'}:
        return {'kind': 'code', 'runs': runs}
    if name in {'quote', 'intensequote'}:
        return {'kind': 'quote', 'runs': runs}
    result = {'kind': 'paragraph', 'runs': runs}
    if name.startswith('listbullet'):
        result['list'] = 'bullet'
    elif name.startswith('listnumber'):
        result['list'] = 'number'
    return result


def _block_text(block: dict) -> str:
    if block['kind'] == 'table':
        return '\n'.join('\t'.join(''.join(run['text'] for run in cell['runs'])
                                    for cell in row) for row in block['rows'])
    return ''.join(run['text'] for run in block['runs'])


def _slice_runs(runs: list[dict], start: int, end: int) -> list[dict]:
    result, position = [], 0
    for run in runs:
        after = position + len(run['text'])
        if position < end and after > start:
            result.append({**run, 'text': run['text'][max(0, start - position):end - position]})
        position = after
    return result


def _numbering_starts(document) -> dict[int, int]:
    starts = {}
    for num in document.part.numbering_part.element[:MAX_RICH_NODES]:
        if num.tag != qn('w:num'):
            continue
        identifier = num.get(qn('w:numId'), '')
        if not identifier.isdecimal() or len(identifier) > 9:
            continue
        for override in num.findall(qn('w:lvlOverride')):
            start = override.find(qn('w:startOverride'))
            if start is None or override.get(qn('w:ilvl')) != '0':
                continue
            value = start.get(qn('w:val'), '')
            if value.isdecimal() and len(value) <= 9:
                starts[int(identifier)] = int(value)
    return starts


def document_blocks(path: Path, raw: str, start: int, end: int) -> tuple[list[dict], bool]:
    """Match actual Office structure to canonical text and slice this section.

    Legacy plain generation retains its exact sidecar text and offsets. Rich
    structure is used only when it maps to that text without dropping anything.
    """
    with path.open('rb') as source:
        bounded_package(source)
        document = Document(source)
    budget, position, blocks = [MAX_RICH_NODES], 0, []
    incomplete, number = False, 0
    numbering_starts = _numbering_starts(document)
    try:
        for ordinal, item in enumerate(document.iter_inner_content()):
            if ordinal >= MAX_RICH_NODES:
                return [], True
            if isinstance(item, Paragraph):
                text = item.text
                # Standard generated list IDs avoid resolving all paragraph/run
                # styles outside this section merely to retain list numbering.
                name = str(item._p.style or '').casefold().replace(' ', '')
                if name.startswith('listnumber'):
                    number += 1
                    properties = item._p.pPr
                    num = properties.numPr if properties is not None else None
                    if num is not None and num.numId is not None:
                        number = numbering_starts.get(num.numId.val, number)
                else:
                    number = 0
            elif isinstance(item, Table):
                number = 0
                if len(item.rows) > MAX_TABLE_ROWS or len(item.columns) > MAX_COLUMNS:
                    return [], True
                text = '\n'.join('\t'.join(cell.text for cell in row.cells) for row in item.rows)
            else:
                return [], True
            if not text:
                continue
            match = raw.find(text, position)
            if match < 0 or raw[position:match].strip():
                return [], True
            after = match + len(text)
            position = after
            if after <= start or match >= end:
                continue
            if isinstance(item, Paragraph):
                block = _paragraph_block(item, budget)
                if block.get('list') == 'number':
                    block['list_start'] = number or 1
                block = {**block, 'runs': _slice_runs(block['runs'], max(0, start - match), end - match)}
            elif match < start or after > end:
                block = {'kind': 'paragraph', 'runs': [{'text': raw[max(start, match):min(end, after)]}]}
                incomplete = True
            else:
                rows = []
                for row in item.rows:
                    cells = []
                    for cell in row.cells:
                        paragraphs = [_paragraph_block(p, budget) for p in cell.paragraphs]
                        runs = []
                        for index, paragraph in enumerate(paragraphs):
                            if index:
                                runs.append({'text': '\n'})
                            runs.extend(paragraph['runs'])
                        cells.append({'runs': runs})
                    rows.append(cells)
                block = {'kind': 'table', 'rows': rows}
                if rows and all(cell['runs'] and all(run.get('bold') for run in cell['runs'])
                                for cell in rows[0]):
                    block['header'] = True
            blocks.append(block)
    except ValueError:
        return [], True
    if raw[position:].strip():
        return [], True
    return blocks, incomplete
