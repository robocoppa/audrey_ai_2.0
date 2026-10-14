"""Hermetic IRS catalog fixtures follow the inspected official table layouts."""

import pytest

from audrey.references.irs_catalog import (
    CATALOG_URLS,
    CatalogParseError,
    normalize_irs_url,
    parse_catalog_page,
)


def _page(rows, *, catalog="current", next_href=None, summary=None):
    if catalog == "bulletins":
        headers = ["IRB in HTML", "Download PDF", "Title"]
    else:
        headers = ["Product Number", "Title", "Revision Date"]
        if catalog == "current":
            headers.append("Posted Date")
    head = "<thead><tr>" + "".join(f"<th>{name}</th>" for name in headers) + "</tr></thead>"
    body = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    ) + "</tbody>"
    result = "<html><title>Internal Revenue Service</title><table>" + head + body + "</table>"
    if summary is not None:
        result += f'<h4 id="solr-results-summary">{summary}</h4>'
    if next_href is not None:
        result += f'<a href="{next_href}" rel="next">Next</a>'
    return result + "</html>"


def _row(product="Form 1040", title="U.S. Individual Income Tax Return",
         link="/pub/irs-pdf/f1040.pdf", revision="2025", posted="01/29/2026"):
    return [f'<a href="{link}">{product}</a>', title, revision, posted]


def test_current_metadata_and_actual_next_link_preserved():
    html = _page(
        [_row(), _row("Instruction 1040", "Instructions for Form 1040",
                      "/pub/irs-pdf/i1040gi.pdf")],
        next_href="?items_per_page=200&amp;page=1",
        summary="Showing 1 - 2 of 3",
    )
    result = parse_catalog_page(html, CATALOG_URLS["current"], "current")
    assert result.row_count == 2 and result.total_rows == 3
    assert result.next_urls == (
        "https://www.irs.gov/forms-instructions-and-publications?items_per_page=200&page=1",
    )
    assert result.entries[0].source_url == "https://www.irs.gov/pub/irs-pdf/f1040.pdf"
    assert result.entries[0].revision_date == "2025"
    assert result.entries[0].posted_date == "01/29/2026"
    assert result.entries[0].document_kind == "form"
    assert result.entries[1].document_kind == "instruction"


def test_language_metadata_keeps_foreign_and_ambiguous_rows_for_reporting():
    html = _page([
        _row("Publication 1", "Your Rights As A Taxpayer", "/pub/irs-pdf/p1.pdf"),
        _row("Publication 1 (SP)", "Your Rights As A Taxpayer (Spanish Version)",
             "/pub/irs-pdf/p1sp.pdf"),
        _row("Publication 1 (AR)", "Your Rights As A Taxpayer", "/pub/irs-pdf/p1ar.pdf"),
        _row("Publication 1", "Your Rights (Translated Version)", "/pub/irs-pdf/p1xx.pdf"),
        _row("Form 706 (Schedule A)", "Real Estate", "/pub/irs-pdf/f706sa.pdf"),
        _row("Form 433-A (OIC)", "Collection Information Statement", "/pub/irs-pdf/f433aoi.pdf"),
    ])
    result = parse_catalog_page(html, CATALOG_URLS["current"], "current")
    assert [entry.language for entry in result.entries] == [
        "en", "non_english", "non_english", "unknown", "en", "en",
    ]
    assert result.row_count == 6


def test_prior_retains_revision_metadata_without_inventing_tax_year():
    row = _row("Publication 1", "Your Rights As A Taxpayer",
               "/pub/irs-prior/p1--2017.pdf", revision="2017")[:-1]
    result = parse_catalog_page(_page([row], catalog="prior"), CATALOG_URLS["prior"], "prior")
    entry = result.entries[0]
    assert entry.revision_date == "2017" and entry.posted_date == ""
    assert entry.document_kind == "publication"
    assert not hasattr(entry, "tax_year")


def test_bulletin_uses_pdf_column_not_html_link_and_paginates():
    rows = [[
        '<a href="https://www.irs.gov/irb/2026-42_irb">Internal Revenue Bulletin: 2026-42</a>',
        '<a href="/pub/irs-irbs/irb26-42.pdf">irb26-42.pdf</a>',
        "Internal Revenue Bulletin: 2026-42",
    ]]
    result = parse_catalog_page(
        _page(rows, catalog="bulletins", next_href="?page=1"), CATALOG_URLS["bulletins"],
        "bulletins",
    )
    assert result.entries[0].document_kind == "bulletin"
    assert result.entries[0].language == "en"
    assert result.entries[0].source_url == "https://www.irs.gov/pub/irs-irbs/irb26-42.pdf"
    assert result.next_urls == ("https://www.irs.gov/irb?page=1",)
    assert result.total_rows is None


@pytest.mark.parametrize("url", [
    "http://www.irs.gov/pub/irs-pdf/f1040.pdf",
    "https://irs.gov/pub/irs-pdf/f1040.pdf",
    "https://www.irs.gov.evil.example/pub/irs-pdf/f1040.pdf",
    "https://www.irs.gov:443/pub/irs-pdf/f1040.pdf",
    "https://name@www.irs.gov/pub/irs-pdf/f1040.pdf",
    "https://www.irs.gov/pub/irs-pdf/../secret.pdf",
    "https://www.irs.gov/pub/irs-pdf/%2e%2e/secret.pdf",
    "https://www.irs.gov/pub/irs-pdf/f1040.pdf?url=https://evil.example",
    "https://www.irs.gov/pub/irs-pdf/f1040.pdf#page=2",
    "https://www.irs.gov/pub/irs-pdf/nested/f1040.pdf",
    "https://www.irs.gov/pub/irs-pdf/f1040.pdf\\",
    "https://www.irs.gov/irb/2026-42_irb",
    "https://www.irs.gov/forms-instructions-and-publications?page=0&page=1",
    "https://www.irs.gov/forms-instructions-and-publications?page=-1",
    "https://www.irs.gov/forms-instructions-and-publications?page=999999",
    "https://www.irs.gov/forms-instructions-and-publications?items_per_page=1000",
    "https://www.irs.gov/forms-instructions-and-publications?redirect=https://evil.example",
    "https://www.irs.gov/forms-instructions-and-publications?order=unknown",
])
def test_urls_fail_closed(url):
    with pytest.raises(CatalogParseError):
        normalize_irs_url(url)


def test_relative_urls_and_query_order_canonicalized():
    assert normalize_irs_url("/pub/irs-prior/p1--1999.pdf", CATALOG_URLS["prior"]) == (
        "https://www.irs.gov/pub/irs-prior/p1--1999.pdf"
    )
    assert normalize_irs_url("?page=1&items_per_page=200", CATALOG_URLS["current"]) == (
        CATALOG_URLS["current"] + "&page=1"
    )


@pytest.mark.parametrize("link", [
    "https://evil.example/f1040.pdf",
    "/pub/irs-pdf/../../secret.pdf",
    "/not-a-supported-path/f1040.pdf",
    "https://www.irs.gov.evil.example/f1040.html",
])
def test_tampered_document_row_raises(link):
    with pytest.raises(CatalogParseError):
        parse_catalog_page(_page([_row(link=link)]), CATALOG_URLS["current"], "current")


@pytest.mark.parametrize("next_href", [
    "https://evil.example/forms-instructions-and-publications?page=1",
    "/prior-year-forms-and-instructions?items_per_page=200&page=1",
    "?items_per_page=200&page=0",
    "?items_per_page=200&page=2",
    "?page=1",
    "?items_per_page=200&page=1&find=1040",
])
def test_tampered_or_inconsistent_pagination_raises(next_href):
    with pytest.raises(CatalogParseError):
        parse_catalog_page(_page([_row()], next_href=next_href), CATALOG_URLS["current"], "current")


def test_known_non_pdf_format_counted_but_not_misrepresented_as_pdf():
    html = _page([_row("Publication 17 (EPUB)", "Your Federal Income Tax",
                       "/pub/irs-epub/p17.epub")])
    result = parse_catalog_page(html, CATALOG_URLS["current"], "current")
    assert result.row_count == 1 and result.entries == ()


@pytest.mark.parametrize("html", [
    "<html><title>Access Denied</title></html>",
    "<html><title>Just a moment...</title><p>Checking your browser</p></html>",
    "<html><table><thead><tr><th>Document</th></tr></thead><tbody></tbody></table></html>",
    _page([_row()]).replace("Revision Date", "Tax Year"),
    _page([["Form 1040", "Title", "2025", ""]]),
    _page([_row()]).replace("<tbody>", "").replace("</tbody>", ""),
    _page([]),
])
def test_challenge_schema_drift_or_unexpected_empty_raises(html):
    with pytest.raises(CatalogParseError):
        parse_catalog_page(html, CATALOG_URLS["current"], "current")


def test_explicit_zero_results_distinguished_from_missing_table():
    result = parse_catalog_page(
        '<html><div class="view-empty">No results found.</div></html>',
        CATALOG_URLS["current"], "current",
    )
    assert result.entries == () and result.row_count == 0 and result.total_rows == 0


def test_summary_requires_next_page_and_correct_row_count():
    with pytest.raises(CatalogParseError, match="more rows"):
        parse_catalog_page(_page([_row()], summary="Showing 1 - 1 of 2"),
                           CATALOG_URLS["current"], "current")
    with pytest.raises(CatalogParseError, match="contradicts"):
        parse_catalog_page(_page([_row()], summary="Showing 1 - 2 of 2"),
                           CATALOG_URLS["current"], "current")


def test_only_actual_next_link_followed_not_numbered_or_last_or_footer_links():
    html = _page([_row()]) + (
        '<a href="?items_per_page=200&amp;page=20">21</a>'
        '<a href="?items_per_page=200&amp;page=20" rel="last">Last</a>'
        '<a href="/help">Help</a>'
    )
    result = parse_catalog_page(html, CATALOG_URLS["current"], "current")
    assert result.next_urls == ()


def test_page_url_and_catalog_must_match():
    with pytest.raises(CatalogParseError):
        parse_catalog_page(_page([_row()]), CATALOG_URLS["prior"], "current")
    with pytest.raises(CatalogParseError):
        parse_catalog_page(_page([_row()]), CATALOG_URLS["current"], "unknown")
