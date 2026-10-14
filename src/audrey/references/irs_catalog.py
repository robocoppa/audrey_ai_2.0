"""Parse the official IRS PDF catalogs without crawling unrelated website links.

The catalog's revision and posted dates are retained verbatim. A revision date
is not asserted to be the tax year to which a document applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

CATALOG_PATHS = {
    "current": "/forms-instructions-and-publications",
    "prior": "/prior-year-forms-and-instructions",
    "bulletins": "/irb",
}
CATALOG_URLS = {
    "current": "https://www.irs.gov/forms-instructions-and-publications?items_per_page=200",
    "prior": "https://www.irs.gov/prior-year-forms-and-instructions?items_per_page=200",
    "bulletins": "https://www.irs.gov/irb",
}
_PDF_PATH = re.compile(r"/pub/(?:irs-pdf|irs-prior|irs-irbs)/[A-Za-z0-9][A-Za-z0-9_.-]*\.pdf", re.I)
_LANGUAGE_CODES = {
    "ar", "bn", "cn", "cn-s", "cn-t", "fa", "fr", "guj", "ht", "it",
    "ja", "km", "ko", "pa", "pl", "pt", "ru", "sp", "es", "tl", "ur",
    "vie", "vi", "zh", "zh-s", "zh-t", "zh-hans", "zh-hant",
}
_LANGUAGE_NAMES = (
    "arabic", "bengali", "chinese", "farsi", "french", "gujarati",
    "haitian creole", "italian", "japanese", "khmer", "korean", "punjabi",
    "polish", "portuguese", "russian", "spanish", "tagalog", "urdu",
    "vietnamese", "bilingual",
)
_EMPTY_TEXT = re.compile(r"\b(?:no results(?: found)?|no matching (?:results|products)|"
                         r"your search yielded no results)\b", re.I)
_SUMMARY = re.compile(r"Showing\s+([\d,]+)\s*[-–]\s*([\d,]+)\s+of\s+([\d,]+)", re.I)


class CatalogParseError(ValueError):
    """The official catalog no longer matches its supported, bounded format."""


@dataclass(frozen=True)
class CatalogEntry:
    catalog: str
    source_url: str
    product_number: str
    title: str
    revision_date: str = ""
    posted_date: str = ""
    language: str = "en"
    document_kind: str = "other"
    catalog_url: str = ""


@dataclass(frozen=True)
class CatalogPage:
    entries: tuple[CatalogEntry, ...]
    next_urls: tuple[str, ...]
    row_count: int
    total_rows: int | None = None


def normalize_irs_url(url: str, base_url: str | None = None) -> str:
    """Allow only official PDF directories and the three catalog endpoints.

    Query parameters are permitted on catalogs only and are canonicalized.
    Redirect destinations must pass this same validation before being requested.
    """
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise CatalogParseError("Missing or oversized IRS URL")
    if re.search(r"[\s\x00-\x1f\x7f\\]", url):
        raise CatalogParseError("IRS URL contains whitespace, controls or backslashes")
    if base_url is not None:
        base_url = normalize_irs_url(base_url)
        url = urljoin(base_url, url)
    try:
        parsed = urlsplit(url)
        invalid_host = (
            parsed.scheme != "https"
            or parsed.hostname != "www.irs.gov"
            or parsed.netloc != "www.irs.gov"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        )
    except ValueError as exc:
        raise CatalogParseError("Malformed IRS URL") from exc
    if invalid_host or parsed.fragment or "%" in parsed.path or ".." in parsed.path:
        raise CatalogParseError("IRS URL is outside the official allowlist")
    if _PDF_PATH.fullmatch(parsed.path):
        if parsed.query:
            raise CatalogParseError("IRS PDF URLs cannot contain query parameters")
        return urlunsplit(("https", "www.irs.gov", parsed.path, "", ""))
    paths = {path: catalog for catalog, path in CATALOG_PATHS.items()}
    catalog = paths.get(parsed.path)
    if catalog is None:
        raise CatalogParseError("IRS URL is outside the official PDF/catalog paths")
    try:
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise CatalogParseError("Malformed IRS catalog query") from exc
    keys = [key for key, _ in query]
    allowed = {"page", "sort", "order"}
    if catalog != "bulletins":
        allowed |= {"items_per_page", "find"}
    if catalog == "prior":
        allowed |= {"product_number", "date", "title"}
    if len(keys) != len(set(keys)) or not set(keys) <= allowed:
        raise CatalogParseError("Unexpected or repeated IRS catalog query parameter")
    orders = {
        "current": {
            "natural_sort_field", "forms_and_pubs_picklist_title",
            "picklist_revision_date_iso", "posted_date",
        },
        "prior": {
            "natural_sort_field", "prior_year_products_picklist_title",
            "prior_year_products_picklist_revision_date",
        },
        "bulletins": {"IRB in HTML"},
    }
    for key, value in query:
        if len(value) > 256 or re.search(r"[\x00-\x1f\x7f]", value):
            raise CatalogParseError("Invalid IRS catalog query value")
        if key == "page" and (not value.isascii() or not value.isdigit()
                              or int(value) > 100_000):
            raise CatalogParseError("Invalid IRS catalog page")
        if key == "items_per_page" and (
            not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 200
        ):
            raise CatalogParseError("Invalid IRS catalog page size")
        if key == "sort" and value not in {"asc", "desc"}:
            raise CatalogParseError("Invalid IRS catalog sort direction")
        if key == "order" and value not in orders[catalog]:
            raise CatalogParseError("Invalid IRS catalog sort field")
    return urlunsplit(("https", "www.irs.gov", parsed.path, urlencode(sorted(query)), ""))


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def _language(product: str, title: str) -> str:
    """English catalog rows are English unless translation metadata says otherwise."""
    folded = f"{product} {title}".casefold()
    if any(re.search(rf"\b{re.escape(name)}\b", folded) for name in _LANGUAGE_NAMES):
        return "non_english"
    parentheticals = re.findall(r"\(([^()]*)\)", product.casefold())
    if any(value.strip() in _LANGUAGE_CODES for value in parentheticals):
        return "non_english"
    if re.search(r"\b(?:translated|translation|foreign language|multilingual)\b", folded):
        return "unknown"
    # An explicit edition/version label must name English before it is trusted.
    versions = re.findall(r"\(([^()]*\b(?:version|language)[^()]*)\)", folded)
    if versions and any(not re.search(r"\benglish\b", version) for version in versions):
        return "unknown"
    return "en"


def _kind(product: str, catalog: str) -> str:
    if catalog == "bulletins":
        return "bulletin"
    if re.match(r"^instructions?\b", product, re.I):
        return "instruction"
    if re.match(r"^publication\b", product, re.I):
        return "publication"
    if re.match(r"^(?:form|schedule)\b", product, re.I):
        return "form"
    return "other"


def _pdf_link(cell: Tag, page_url: str) -> str | None:
    links = cell.find_all("a", href=True)
    if not links:
        raise CatalogParseError("Catalog product row has no document link")
    pdfs: list[str] = []
    for link in links:
        href = str(link["href"])
        resolved = urljoin(page_url, href)
        try:
            parsed = urlsplit(resolved)
        except ValueError as exc:
            raise CatalogParseError("Malformed catalog document link") from exc
        if parsed.path.casefold().endswith(".pdf"):
            pdfs.append(normalize_irs_url(href, page_url))
            continue
        # IRS also lists EPUB/HTML products. They do not become PDF originals.
        # No other non-PDF destination is silently treated as a recognized row.
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.irs.gov"
            or parsed.fragment
            or parsed.query
            or re.search(r"[\s\x00-\x1f\x7f\\%]", resolved)
            or ".." in parsed.path
            or not (
                re.fullmatch(r"/pub/irs-(?:epub|mobi)/[A-Za-z0-9_.-]+\.(?:epub|mobi)",
                             parsed.path, re.I)
                or re.fullmatch(r"/publications/[A-Za-z0-9-]+", parsed.path)
            )
        ):
            raise CatalogParseError("Catalog document link is outside supported IRS formats")
    if len(set(pdfs)) > 1:
        raise CatalogParseError("Catalog product row contains multiple different PDFs")
    return pdfs[0] if pdfs else None


def _next_pages(soup: BeautifulSoup, page_url: str, catalog: str) -> tuple[str, ...]:
    current_query = dict(parse_qsl(urlsplit(page_url).query, keep_blank_values=True))
    current_page = int(current_query.pop("page", "0"))
    next_urls: list[str] = []
    for anchor in soup.find_all("a", href=True):
        rel = anchor.get("rel", [])
        if "next" not in rel and str(anchor.get("title", "")).casefold() != "go to next page":
            continue
        next_url = normalize_irs_url(str(anchor["href"]), page_url)
        parsed = urlsplit(next_url)
        if parsed.path != CATALOG_PATHS[catalog]:
            raise CatalogParseError("Catalog next page points at another catalog")
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        page = query.pop("page", None)
        if page is None or int(page) != current_page + 1 or query != current_query:
            raise CatalogParseError("Catalog next page changed filters or skipped/repeated a page")
        if next_url not in next_urls:
            next_urls.append(next_url)
    if len(next_urls) > 1:
        raise CatalogParseError("Catalog contains conflicting next-page links")
    return tuple(next_urls)


def parse_catalog_page(html: str, page_url: str, catalog: str) -> CatalogPage:
    """Parse one page; preserve all PDF language metadata for coverage reporting."""
    if catalog not in CATALOG_PATHS:
        raise CatalogParseError("Unknown IRS catalog")
    page_url = normalize_irs_url(page_url)
    if urlsplit(page_url).path != CATALOG_PATHS[catalog]:
        raise CatalogParseError("Page URL does not match the requested catalog")
    soup = BeautifulSoup(html, "html.parser")
    title = _text(soup.title) if soup.title is not None else ""
    if re.search(r"access denied|verify (?:that )?you are human|just a moment|"
                 r"checking your browser", title, re.I):
        raise CatalogParseError("IRS catalog returned an access challenge")
    expected = (
        ["irb in html", "download pdf", "title"] if catalog == "bulletins"
        else ["product number", "title", "revision date"]
             + (["posted date"] if catalog == "current" else [])
    )
    candidates = []
    for table in soup.find_all("table"):
        headers = [_text(th).casefold() for th in table.select("thead th")]
        if len(headers) == len(expected) and set(headers) == set(expected):
            candidates.append((table, headers))
    if len(candidates) != 1:
        if not candidates and _EMPTY_TEXT.search(
            " ".join(_text(tag) for tag in soup.select(".view-empty, #solr-results-summary"))
        ):
            return CatalogPage((), (), 0, 0)
        raise CatalogParseError("IRS catalog table or expected columns are missing/ambiguous")
    table, headers = candidates[0]
    tbody = table.find("tbody")
    if tbody is None:
        raise CatalogParseError("IRS catalog table has no body")
    entries: list[CatalogEntry] = []
    row_count = 0
    for row in tbody.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) == 1 and cells[0].get("colspan") and _EMPTY_TEXT.search(_text(cells[0])):
            continue
        if len(cells) != len(headers) or any(cell.get("colspan") for cell in cells):
            raise CatalogParseError("IRS catalog row does not match its columns")
        values = dict(zip(headers, cells, strict=True))
        row_count += 1
        product = _text(values["irb in html" if catalog == "bulletins" else "product number"])
        title = _text(values["title"])
        if not product or not title:
            raise CatalogParseError("IRS catalog row lacks product number or title")
        source = _pdf_link(values["download pdf" if catalog == "bulletins"
                                  else "product number"], page_url)
        if source is None:
            continue
        entries.append(CatalogEntry(
            catalog=catalog, source_url=source, product_number=product, title=title,
            revision_date=_text(values["revision date"]) if "revision date" in values else "",
            posted_date=_text(values["posted date"]) if "posted date" in values else "",
            language=_language(product, title), document_kind=_kind(product, catalog),
            catalog_url=page_url,
        ))
    summary_tag = soup.select_one("#solr-results-summary")
    summary = _SUMMARY.search(_text(summary_tag)) if summary_tag is not None else None
    total_rows = None
    end = None
    if summary:
        start, end, total_rows = (int(value.replace(",", "")) for value in summary.groups())
        if total_rows == 0:
            if row_count != 0 or start != 0 or end != 0:
                raise CatalogParseError("IRS empty catalog count contradicts its rows")
        elif start < 1 or end < start or end > total_rows or end - start + 1 != row_count:
            raise CatalogParseError("IRS catalog summary contradicts its rows")
    next_urls = _next_pages(soup, page_url, catalog)
    if end is not None and total_rows is not None:
        if end < total_rows and not next_urls:
            raise CatalogParseError("IRS catalog has more rows but no next page")
        if end >= total_rows and next_urls:
            raise CatalogParseError("IRS catalog has a next page after its final row")
    if row_count == 0 and total_rows != 0 and not _EMPTY_TEXT.search(_text(tbody)):
        raise CatalogParseError("IRS catalog is unexpectedly empty")
    return CatalogPage(tuple(entries), next_urls, row_count, total_rows)
