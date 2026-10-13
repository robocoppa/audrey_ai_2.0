"""Read-only spreadsheet content, completeness, exactness and parser limits."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from audrey.kb import table_analysis as tables
from audrey.kb.table_analysis import (
    XLSX_MIME,
    TableAnalysisError,
    TableAnalysisRequest,
    analyze_table,
    extract_table_text,
)


def csv_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


def workbook_file(tmp_path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("defusedxml")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Costs"
    sheet.append(["Category", "Amount"])
    sheet.append(["Supplies", 0.1])
    sheet.append(["Supplies", 0.2])
    sheet.append(["Supplies", "=B2+B3"])
    other = workbook.create_sheet("People")
    other.append(["Name", "Active"])
    other.append(["Anne", True])
    path = tmp_path / "data.xlsx"
    workbook.save(path)
    workbook.close()
    return path


def rewrite_zip(path: Path, transform) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = [(item.filename, archive.read(item.filename)) for item in archive.infolist()]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries:
            archive.writestr(name, transform(name, content))
    path.write_bytes(output.getvalue())


def test_csv_unique_headers_samples_and_full_row_stats(tmp_path):
    path = csv_file(tmp_path, "\ufeffCost,Cost,,Cost (2)\n1,2,x,3\n4,,y,6\n")
    result = analyze_table(path, "text/csv", {"limit": 1})
    assert result["columns"] == ["Cost", "Cost (2)", "column_3", "Cost (2) (2)"]
    assert result["rows"] == [["1", "2", "x", "3"]]
    assert result["total_rows"] == 2 and result["has_more"]
    assert result["full_table_scanned"]
    assert result["column_stats"][1] == {"column": "Cost (2)", "blank_cells": 1, "numeric_cells": 1}


def test_decimal_grouped_totals_disclose_ignored_cells(tmp_path):
    path = csv_file(tmp_path, "Category,Amount\nA,0.1\nA,0.2\nA,\nA,unknown\nB,7.00\n")
    result = analyze_table(path, "text/csv", {
        "action": "aggregate", "group_by": ["Category"],
        "aggregations": [
            {"op": "sum", "column": "Amount"},
            {"op": "mean", "column": "Amount"},
            {"op": "count"}, {"op": "count", "column": "Amount"},
        ],
    })
    measures = result["groups"][0]["aggregations"]
    assert measures[0]["value"] == "0.3"
    assert measures[0]["missing_cells"] == 1 and measures[0]["non_numeric_cells"] == 1
    assert measures[1]["value"] == "0.15"
    assert measures[2]["value"] == 4 and measures[3]["value"] == 3
    assert result["groups_total"] == 2 and not result["has_more"]


def test_filters_pagination_and_projection_preserve_total(tmp_path):
    path = csv_file(tmp_path, "Name,Amount\nAnne,1\nBob,2\nBen,3\nBill,4\n")
    result = analyze_table(path, "text/csv", {
        "action": "rows", "columns": ["Name"],
        "filters": [{"column": "Amount", "op": "gte", "value": "2"},
                    {"column": "Name", "op": "contains", "value": "B"}],
        "offset": 1, "limit": 1,
    })
    assert result["rows"] == [["Ben"]]
    assert result["matched_rows"] == 3 and result["total_rows"] == 4 and result["has_more"]


def test_duplicates_report_groups_and_extra_rows(tmp_path):
    path = csv_file(tmp_path, "Name,Amount\nA,1\nA,1\nA,2\nB,1\n")
    result = analyze_table(path, "text/csv", {"action": "duplicates", "columns": ["Name"]})
    assert result["duplicate_groups"] == 1 and result["extra_duplicate_rows"] == 2
    assert result["duplicates"] == [{"values": ["A"], "occurrences": 3}]


def test_empty_numeric_data_is_null_not_zero(tmp_path):
    path = csv_file(tmp_path, "Amount\nunknown\n \n")
    result = analyze_table(path, "text/csv", {
        "action": "aggregate",
        "aggregations": [{"op": op, "column": "Amount"} for op in ["sum", "mean", "min", "max"]],
    })
    assert all(item["value"] is None for item in result["groups"][0]["aggregations"])


def test_empty_filter_results_still_count_zero(tmp_path):
    path = csv_file(tmp_path, "Name,Amount\nAnne,1\n")
    result = analyze_table(path, "text/csv", {
        "action": "aggregate",
        "filters": [{"column": "Name", "op": "eq", "value": "Bob"}],
        "aggregations": [{"op": "count"}, {"op": "sum", "column": "Amount"}],
    })
    assert result["groups"][0]["aggregations"][0]["value"] == 0
    assert result["groups"][0]["aggregations"][1]["value"] is None


@pytest.mark.parametrize("bad", [
    {"action": "aggregate"}, {"action": "rows", "group_by": ["a"]},
    {"filters": [{"column": "a", "op": "gt", "value": "NaN"}]},
    {"action": "rows", "filters": [{"column": "a", "op": "eq"}]},
    {"limit": 101}, {"offset": -1}, {"sql": "select 1"},
    {"action": "rows", "columns": ["a", "a"]},
])
def test_invalid_request_is_rejected_before_parse(bad):
    with pytest.raises(ValidationError):
        TableAnalysisRequest.model_validate(bad)


@pytest.mark.parametrize("text,error", [
    ("", "empty"), ("a\n\"unterminated", "parsed"),
    ("a\nvalue\0", "null bytes"),
])
def test_invalid_csv(tmp_path, text, error):
    with pytest.raises(TableAnalysisError, match=error):
        analyze_table(csv_file(tmp_path, text), "text/csv", {})


def test_encoding_unknown_column_and_wrong_kind(tmp_path):
    path = csv_file(tmp_path, "a\n1\n")
    with pytest.raises(TableAnalysisError, match="unknown column"):
        analyze_table(path, "text/csv", {"columns": ["b"]})
    with pytest.raises(TableAnalysisError, match="only CSV"):
        analyze_table(path, "application/pdf", {})
    path.write_bytes(b"a\n\xff")
    with pytest.raises(TableAnalysisError, match="UTF-8"):
        analyze_table(path, "text/csv", {})


def test_csv_size_row_column_cell_and_group_limits_are_failures(tmp_path, monkeypatch):
    path = csv_file(tmp_path, "a,b\n1,x\n2,y\n3,z\n")
    for name, value, spec in [
        ("MAX_FILE_BYTES", 2, {}),
        ("MAX_ROWS", 2, {}),
        ("MAX_COLUMNS", 1, {}),
        ("MAX_CELLS", 4, {}),
        ("MAX_CELL_CHARS", 0, {}),
        ("MAX_GROUPS", 2, {"action": "aggregate", "group_by": ["b"], "aggregations": [{"op": "count"}]}),
    ]:
        with monkeypatch.context() as context:
            context.setattr(tables, name, value)
            with pytest.raises(TableAnalysisError, match="limit"):
                analyze_table(path, "text/csv", spec)


def test_display_truncation_is_explicit_and_does_not_change_duplicates(tmp_path):
    path = csv_file(tmp_path, "Name\n" + "x" * 501 + "\n" + "x" * 500 + "y\n")
    result = analyze_table(path, "text/csv", {})
    assert result["display_cells_truncated"]
    assert len(result["rows"][0][0]) == 500
    duplicates = analyze_table(path, "text/csv", {"action": "duplicates"})
    assert duplicates["duplicate_groups"] == 0


def test_excel_sheet_selection_formulas_and_exact_sum(tmp_path):
    path = workbook_file(tmp_path)
    result = analyze_table(path, XLSX_MIME, {
        "action": "aggregate", "aggregations": [{"op": "sum", "column": "Amount"}],
    })
    assert result["sheets"] == ["Costs", "People"]
    assert result["groups"][0]["aggregations"][0]["value"] == "0.3"
    assert result["formula_cells"] == 1 and result["formulas_without_cached_values"] == 1
    assert result["groups"][0]["aggregations"][0]["missing_cells"] == 1
    people = analyze_table(path, XLSX_MIME, {"sheet": "People"})
    assert people["rows"] == [["Anne", "true"]]
    with pytest.raises(TableAnalysisError, match="does not exist"):
        analyze_table(path, XLSX_MIME, {"sheet": "Missing"})


def test_excel_false_dimensions_do_not_hide_rows(tmp_path):
    path = workbook_file(tmp_path)
    rewrite_zip(path, lambda name, data: data.replace(b'ref="A1:B4"', b'ref="A1:A1"') if name.endswith("sheet1.xml") else data)
    assert analyze_table(path, XLSX_MIME, {})["total_rows"] == 3


def test_excel_sparse_huge_cell_rejected(tmp_path):
    path = workbook_file(tmp_path)
    rewrite_zip(path, lambda name, data: data.replace(b'r="B4"', b'r="B999999999"') if name.endswith("sheet1.xml") else data)
    with pytest.raises(TableAnalysisError, match="limits"):
        analyze_table(path, XLSX_MIME, {})


def test_excel_uncompressed_limit_and_invalid_zip(tmp_path, monkeypatch):
    path = workbook_file(tmp_path)
    monkeypatch.setattr(tables, "MAX_UNCOMPRESSED_BYTES", 2)
    with pytest.raises(TableAnalysisError, match="archive limit"):
        analyze_table(path, XLSX_MIME, {})
    path.write_text("not an archive")
    with pytest.raises(TableAnalysisError, match="safely"):
        analyze_table(path, XLSX_MIME, {})


def test_excel_xml_entities_are_blocked(tmp_path):
    path = workbook_file(tmp_path)
    def entity(name, data):
        if name.endswith("sheet1.xml"):
            return b'<!DOCTYPE worksheet [<!ENTITY x "secret">]>' + data.replace(b"Supplies", b"&x;")
        return data
    rewrite_zip(path, entity)
    with pytest.raises(TableAnalysisError, match="safely"):
        analyze_table(path, XLSX_MIME, {})


def test_preview_labels_omissions_and_missing_original(tmp_path):
    path = csv_file(tmp_path, "Name\n" + "x" * 200 + "\n")
    text = extract_table_text(path, "text/csv", 128)
    assert "[Preview only." in text
    path.unlink()
    with pytest.raises(TableAnalysisError, match="unavailable"):
        analyze_table(path, "text/csv", {})


def test_header_and_result_limits_fail_with_clear_remedy(tmp_path, monkeypatch):
    path = csv_file(tmp_path, "x" * 129 + "\n1\n")
    with pytest.raises(TableAnalysisError, match="header exceeds"):
        analyze_table(path, "text/csv", {})
    path = csv_file(tmp_path, "Name\nAnne\n")
    monkeypatch.setattr(tables, "MAX_RESULT_CHARS", 10)
    with pytest.raises(TableAnalysisError, match="smaller limit"):
        analyze_table(path, "text/csv", {})

def test_automatic_byte_bounded_paging_retains_counts_and_next_offset(tmp_path, monkeypatch):
    path = csv_file(tmp_path, "Name\n" + "\n".join(["é" * 100] * 10) + "\n")
    full = analyze_table(path, "text/csv", {"action": "rows", "limit": 10})
    import json
    full_size = len(json.dumps(full, ensure_ascii=False).encode("utf-8"))
    monkeypatch.setattr(tables, "MAX_RESULT_BYTES", full_size - 500)
    page = analyze_table(path, "text/csv", {"action": "rows", "limit": 10})
    assert 0 < page["returned_items"] < 10
    assert page["total_rows"] == 10 and page["full_table_scanned"]
    assert page["next_offset"] == page["returned_items"] and page["has_more"]
    assert len(json.dumps(page, ensure_ascii=False).encode("utf-8")) <= tables.MAX_RESULT_BYTES
    total = page["returned_items"]
    while page["next_offset"] is not None:
        page = analyze_table(path, "text/csv", {
            "action": "rows", "limit": 10, "offset": page["next_offset"],
        })
        total += page["returned_items"]
    assert total == 10


def test_large_decimal_sum_is_exact_beyond_default_context(tmp_path):
    path = csv_file(tmp_path, "Amount\n99999999999999999999999999999999999999999999999999.1\n0.2\n")
    result = analyze_table(path, "text/csv", {
        "action": "aggregate", "aggregations": [{"op": "sum", "column": "Amount"}],
    })
    assert result["groups"][0]["aggregations"][0]["value"] == "99999999999999999999999999999999999999999999999999.3"

def test_wide_table_header_pages_keep_exact_names(tmp_path):
    names = [f"column_{index}_" + "x" * 90 for index in range(20)]
    path = csv_file(tmp_path, ",".join(names) + "\n" + ",".join(["1"] * 20) + "\n")
    seen = []
    offset = 0
    import json
    while offset is not None:
        page = analyze_table(path, "text/csv", {
            "column_offset": offset, "column_limit": 20,
        })
        seen.extend(page["columns"])
        assert page["column_count"] == 20
        assert len(json.dumps(page, ensure_ascii=False).encode()) <= 2500
        offset = page["next_column_offset"]
    assert seen == names
