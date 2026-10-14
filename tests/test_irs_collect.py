"""Hermetic collector integration: real archive storage with mocked public HTTP."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from audrey.references import irs_collect
from audrey.references.irs_archive import ArchiveStore
from audrey.references.irs_catalog import CATALOG_URLS, CatalogParseError
from audrey.references.irs_collect import CapacityError, CollectionError, PublicIRSClient, collect

PDF_URL = "https://www.irs.gov/pub/irs-pdf/f1040.pdf"
NEXT_URL = CATALOG_URLS["current"] + "&page=1"
PDF_V1 = b"%PDF-1.4\nfirst published revision\n%%EOF\n"
PDF_V2 = b"%PDF-1.4\ncorrected published revision\n%%EOF\n"


def _catalog(rows=None, *, start=1, total=None, next_page=False):
    if rows is None:
        rows = [("Form 1040", "Income Tax Return", "f1040.pdf")]
    if total is None:
        total = len(rows)
    body = "".join(
        f'<tr><td><a href="/pub/irs-pdf/{filename}">{product}</a></td>'
        f"<td>{title}</td><td>2025</td><td>01/29/2026</td></tr>"
        for product, title, filename in rows
    )
    paging = '<a rel="next" href="?items_per_page=200&amp;page=1">Next</a>' if next_page else ""
    return (
        "<html><title>Internal Revenue Service</title><table><thead><tr>"
        "<th>Product Number</th><th>Title</th><th>Revision Date</th><th>Posted Date</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
        f'<h4 id="solr-results-summary">Showing {start} - {start + len(rows) - 1} '
        f"of {total}</h4>{paging}</html>"
    ).encode()


def _args(root, *extra):
    args = irs_collect.parser().parse_args([
        "--root", str(root), "--catalogs", "current", "--min-free-bytes", "1", *extra,
    ])
    args.catalogs = list(dict.fromkeys(args.catalogs.split(",")))
    return args


@pytest.fixture
def clients():
    active = []

    def create(handler, *, delay=0):
        client = PublicIRSClient(delay=delay, transport=httpx.MockTransport(handler))
        active.append(client)
        return client

    yield create
    for client in active:
        client.close()


class _Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(irs_collect.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(irs_collect.time, "sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("location", [
    "https://evil.example/f1040.pdf",
    "http://www.irs.gov/pub/irs-pdf/f1040.pdf",
    "https://www.irs.gov:8443/pub/irs-pdf/f1040.pdf",
    "https://name@www.irs.gov/pub/irs-pdf/f1040.pdf",
    "/pub/irs-pdf/%2e%2e/secret.pdf",
])
def test_blocked_redirect_never_fetches_destination(clients, location):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": location})

    client = clients(handler)
    with pytest.raises(CatalogParseError):
        with client.response(PDF_URL):
            pytest.fail("Redirect must be rejected before yielding")
    assert requested == [PDF_URL]


def test_redirect_cannot_change_pdf_into_catalog(clients):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": CATALOG_URLS["current"]})

    with pytest.raises(CollectionError, match="resource type"):
        with clients(handler).response(PDF_URL):
            pytest.fail("Resource-type redirect must be rejected")
    assert requested == [PDF_URL]


def test_allowed_redirect_is_requested_and_retains_final_url(clients, tmp_path):
    final = "https://www.irs.gov/pub/irs-prior/f1040--2025.pdf"
    requested = []

    def handler(request):
        requested.append(str(request.url))
        if str(request.url) == PDF_URL:
            return httpx.Response(302, headers={"location": final})
        return httpx.Response(200, content=PDF_V1)

    result = clients(handler).pdf(
        PDF_URL, tmp_path / "download.part", headers={}, max_bytes=1024, min_free_bytes=0,
    )
    assert requested == [PDF_URL, final]
    assert result["final_url"] == final
    assert result["bytes"] == len(PDF_V1)


@pytest.mark.parametrize(("content", "message"), [
    (b"<html>Access denied</html>", "not a PDF"),
    (b"%PDF-1.4\ninterrupted body", "truncated"),
])
def test_html_or_truncated_pdf_refused(clients, tmp_path, content, message):
    client = clients(lambda _: httpx.Response(200, content=content))
    with pytest.raises(CollectionError, match=message):
        client.pdf(PDF_URL, tmp_path / "download.part", headers={},
                   max_bytes=1024, min_free_bytes=0)


def test_declared_pdf_size_refused_before_writing(clients, tmp_path):
    client = clients(lambda _: httpx.Response(
        200, content=PDF_V1, headers={"content-length": "5000"},
    ))
    destination = tmp_path / "download.part"
    with pytest.raises(CapacityError):
        client.pdf(PDF_URL, destination, headers={}, max_bytes=100, min_free_bytes=0)
    assert not destination.exists()


class _Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


def test_unannounced_pdf_size_refused_while_streaming(clients, tmp_path):
    client = clients(lambda _: httpx.Response(
        200, stream=_Chunks([b"%PDF-1.4\n", b"x" * 1000, b"%%EOF"]),
    ))
    with pytest.raises(CapacityError):
        client.pdf(PDF_URL, tmp_path / "download.part", headers={},
                   max_bytes=100, min_free_bytes=0)


def test_catalog_body_bound(clients):
    client = clients(lambda _: httpx.Response(
        200, stream=_Chunks([b"x" * (4 * irs_collect.MIB), b"x"]),
    ))
    with pytest.raises(CollectionError, match="4 MiB"):
        client.catalog(CATALOG_URLS["current"])


def test_pdf_disk_reserve_bound(clients, tmp_path, monkeypatch):
    monkeypatch.setattr(irs_collect.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    client = clients(lambda _: httpx.Response(200, content=PDF_V1))
    with pytest.raises(CapacityError, match="reserve"):
        client.pdf(PDF_URL, tmp_path / "download.part", headers={},
                   max_bytes=1024, min_free_bytes=1)


def test_requests_spaced_even_across_separate_calls(clients, fake_clock):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=PDF_V1)

    client = clients(handler, delay=1.25)
    with client.response(PDF_URL):
        pass
    with client.response(PDF_URL):
        pass
    assert requested == [PDF_URL, PDF_URL]
    assert fake_clock.sleeps == [1.25]


def test_rate_limit_retry_after_bounded_and_retries_limited(clients, fake_clock):
    requested = []
    client = clients(lambda request: (
        requested.append(str(request.url)) or httpx.Response(429, headers={"retry-after": "99999"})
    ))
    with pytest.raises(CollectionError, match="rate-limited"):
        with client.response(PDF_URL):
            pytest.fail("Persistent rate limit cannot yield a response")
    assert len(requested) == 3
    assert fake_clock.sleeps == [60.0, 60.0]


def test_transient_unavailable_response_retried_then_succeeds(clients, fake_clock):
    statuses = iter([503, 200])
    client = clients(lambda _: httpx.Response(next(statuses), content=PDF_V1))
    with client.response(PDF_URL) as response:
        assert response.status_code == 200
    assert fake_clock.sleeps == [2.0]


def test_caller_http_error_not_retried_or_yielded_twice(clients, fake_clock):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=PDF_V1)

    with pytest.raises(httpx.ReadError, match="caller failed"):
        with clients(handler).response(PDF_URL) as response:
            raise httpx.ReadError("caller failed", request=response.request)
    assert requested == [PDF_URL]
    assert fake_clock.sleeps == []


def test_complete_fresh_collection_two_pages_language_reporting_and_json(
    clients, tmp_path, capsys,
):
    requested = []
    first_rows = [
        ("Form 1040", "Income Tax Return", "f1040.pdf"),
        ("Form 1040 (SP)", "Income Tax Return (Spanish Version)", "f1040sp.pdf"),
        ("Publication 1", "Rights (Translated Version)", "p1xx.pdf"),
    ]

    def handler(request):
        requested.append(str(request.url))
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog(first_rows, total=4, next_page=True))
        if str(request.url) == NEXT_URL:
            return httpx.Response(200, content=_catalog(
                [("Publication 17", "Your Federal Income Tax", "p17.pdf")], start=4, total=4,
            ))
        assert str(request.url) in {PDF_URL, "https://www.irs.gov/pub/irs-pdf/p17.pdf"}
        return httpx.Response(200, content=PDF_V1, headers={"etag": '"v1"'})

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path), store, clients(handler))
        assert report["status"] == "completed"
        assert report["archive_complete_for_selected_catalogs"] is True
        assert report["coverage"]["current"] == {
            "pages": 2, "rows_seen": 4, "official_total_rows": 4, "enumeration_complete": True,
        }
        assert report["run"]["downloaded"] == 2
        assert report["inventory"]["discovered"] == 4
        assert report["inventory"]["eligible"] == 2
        assert report["inventory"]["skipped_non_english"] == 1
        assert report["inventory"]["skipped_unknown_language"] == 1
        assert report["remaining_known_documents"] == 0
        assert json.loads(json.dumps(report)) == report
        manifest = json.loads((tmp_path / "catalogs/manifest.json").read_text())
        assert len(manifest["revisions"]) == 2
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["event"] == "catalog_page"
    assert sum(event["event"] == "pdf_progress" for event in events) == 2
    assert "https://www.irs.gov/pub/irs-pdf/f1040sp.pdf" not in requested
    assert "https://www.irs.gov/pub/irs-pdf/p1xx.pdf" not in requested


def test_bounded_pages_partial_then_resume_verifies_and_skips_original(clients, tmp_path):
    pdf_requests = []

    def handler(request):
        url = str(request.url)
        if url == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog(total=2, next_page=True))
        if url == NEXT_URL:
            return httpx.Response(200, content=_catalog(
                [("Publication 17", "Your Federal Income Tax", "p17.pdf")], start=2, total=2,
            ))
        pdf_requests.append(url)
        return httpx.Response(200, content=PDF_V1)

    with ArchiveStore(tmp_path) as store:
        first = collect(_args(tmp_path, "--max-pages", "1"), store, clients(handler))
        assert first["status"] == "partial"
        assert not first["archive_complete_for_selected_catalogs"]
        assert not first["coverage"]["current"]["enumeration_complete"]
        assert first["run"]["downloaded"] == 1
        assert first["limits"] == ["current: requested catalog page limit"]
        second = collect(_args(tmp_path), store, clients(handler))
        assert second["status"] == "completed"
        assert second["run"]["cached"] == 1
        assert second["run"]["downloaded"] == 1
    assert pdf_requests == [PDF_URL, "https://www.irs.gov/pub/irs-pdf/p17.pdf"]


def test_discovery_only_never_downloads_or_claims_finished_archive(clients, tmp_path):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        assert str(request.url) == CATALOG_URLS["current"]
        return httpx.Response(200, content=_catalog())

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--discover-only"), store, clients(handler))
        assert report["status"] == "partial"
        assert report["coverage"]["current"]["enumeration_complete"]
        assert not report["archive_complete_for_selected_catalogs"]
        assert report["run"]["attempted"] == 0
        assert report["remaining_known_documents"] == 1
    assert requested == [CATALOG_URLS["current"]]


def test_refresh_304_uses_validator_and_changed_pdf_retains_prior_original(clients, tmp_path):
    stage = "first"
    conditional_headers = []

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        conditional_headers.append(dict(request.headers))
        if stage == "unchanged":
            return httpx.Response(304, headers={"etag": '"v1"'})
        content = PDF_V1 if stage == "first" else PDF_V2
        return httpx.Response(200, content=content, headers={"etag": f'"{stage}"'})

    with ArchiveStore(tmp_path) as store:
        collect(_args(tmp_path), store, clients(handler))
        first = store.latest_revision(PDF_URL)
        first_path = tmp_path / first["relative_path"]
        stage = "unchanged"
        unchanged = collect(_args(tmp_path, "--refresh"), store, clients(handler))
        assert unchanged["status"] == "completed"
        assert unchanged["run"]["unchanged"] == 1
        assert unchanged["run"]["downloaded"] == 0
        assert conditional_headers[-1]["if-none-match"] == '"first"'
        stage = "changed"
        changed = collect(_args(tmp_path, "--refresh"), store, clients(handler))
        assert changed["status"] == "completed"
        assert changed["run"]["downloaded"] == 1
        assert store.summary()["revisions"] == 2
        latest = store.latest_revision(PDF_URL)
        assert latest["sha256"] != first["sha256"]
        assert (tmp_path / latest["relative_path"]).read_bytes() == PDF_V2
        assert first_path.read_bytes() == PDF_V1


@pytest.mark.parametrize("corrupt_cache", [False, True])
def test_304_without_verified_local_pdf_rejected(clients, tmp_path, corrupt_cache):
    stage = "first" if corrupt_cache else "invalid_304"
    pdf_headers = []

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        pdf_headers.append(dict(request.headers))
        if stage == "first":
            return httpx.Response(200, content=PDF_V1, headers={"etag": '"first"'})
        return httpx.Response(304)

    with ArchiveStore(tmp_path) as store:
        if corrupt_cache:
            collect(_args(tmp_path), store, clients(handler))
            revision = store.latest_revision(PDF_URL)
            (tmp_path / revision["relative_path"]).write_bytes(b"bad cache")
            stage = "invalid_304"
        report = collect(_args(tmp_path, "--refresh"), store, clients(handler))
        assert report["status"] == "findings"
        assert not report["archive_complete_for_selected_catalogs"]
        assert report["run"]["unchanged"] == 0
        assert report["remaining_known_documents"] == 1
        assert "without a verified" in report["failures"][0]["error"]
        assert "if-none-match" not in pdf_headers[-1]
        assert "if-modified-since" not in pdf_headers[-1]
        assert list((tmp_path / ".staging").iterdir()) == []


def test_refresh_download_failure_retains_existing_original(clients, tmp_path):
    failed = False

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        return httpx.Response(404) if failed else httpx.Response(200, content=PDF_V1)

    with ArchiveStore(tmp_path) as store:
        collect(_args(tmp_path), store, clients(handler))
        revision = store.latest_revision(PDF_URL)
        failed = True
        report = collect(_args(tmp_path, "--refresh"), store, clients(handler))
        assert report["status"] == "findings"
        assert report["run"]["downloaded"] == 0
        assert report["inventory"]["revisions"] == 1
        assert store.latest_revision(PDF_URL)["sha256"] == revision["sha256"]
        assert (tmp_path / revision["relative_path"]).read_bytes() == PDF_V1


def test_download_limit_counts_failed_attempts_not_only_successes(clients, tmp_path):
    pdf_requests = []

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog([
                ("Form 1040", "Income Tax Return", "f1040.pdf"),
                ("Publication 17", "Your Federal Income Tax", "p17.pdf"),
            ]))
        pdf_requests.append(str(request.url))
        return httpx.Response(404)

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--max-downloads", "1"), store, clients(handler))
        assert report["status"] == "findings"
        assert report["run"]["attempted"] == 1
        assert report["run"]["downloaded"] == 0
        assert "requested PDF download-attempt limit" in report["limits"]
        assert report["remaining_known_documents"] == 2
    assert pdf_requests == [PDF_URL]


def test_byte_limit_partial_file_cleaned_and_not_archived(clients, tmp_path):
    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        return httpx.Response(200, stream=_Chunks([PDF_V1]))

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--max-new-bytes", "10"), store, clients(handler))
        assert report["status"] == "findings"
        assert report["run"]["new_bytes"] == 0
        assert store.latest_revision(PDF_URL) is None
        assert report["limits"]
        assert list((tmp_path / ".staging").iterdir()) == []


def test_changed_total_does_not_claim_complete_enumeration(clients, tmp_path):
    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog(total=2, next_page=True))
        return httpx.Response(200, content=_catalog(
            [("Publication 17", "Your Federal Income Tax", "p17.pdf")], start=2, total=3,
            next_page=True,
        ))

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--discover-only"), store, clients(handler))
        assert report["status"] == "findings"
        assert not report["coverage"]["current"]["enumeration_complete"]


def test_main_writes_complete_machine_readable_report(clients, tmp_path, monkeypatch, capsys):
    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        return httpx.Response(200, content=PDF_V1)

    client = clients(handler)
    monkeypatch.setattr(irs_collect, "PublicIRSClient", lambda **_: client)
    report_path = tmp_path / "local-report.json"
    monkeypatch.setattr("sys.argv", [
        "irs_collect", "--root", str(tmp_path / "archive"), "--catalogs", "current",
        "--min-free-bytes", "1", "--delay", "0.25", "--report", str(report_path),
    ])
    assert irs_collect.main() == 0
    report = json.loads(report_path.read_text())
    assert report["schema"] == 1 and report["status"] == "completed"
    assert report["exit_code"] == 0
    output = capsys.readouterr().out
    final_json = output[output.rfind("\n{\n") + 1:]
    assert json.loads(final_json) == report


def test_english_notice_pdf_in_official_catalog_is_retained(clients, tmp_path):
    notice_url = "https://www.irs.gov/pub/irs-pdf/n609.pdf"

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog([
                ("Notice 609", "Privacy Act Notice", "n609.pdf"),
            ]))
        assert str(request.url) == notice_url
        return httpx.Response(200, content=PDF_V1)

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path), store, clients(handler))
        assert report["status"] == "completed"
        assert report["inventory"]["eligible"] == 1
        assert store.latest_revision(notice_url) is not None


def test_verified_resume_clears_prior_refresh_failure_without_pdf_call(clients, tmp_path):
    stage = "first"
    pdf_requests = []

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        pdf_requests.append(str(request.url))
        if stage == "failed":
            return httpx.Response(404)
        assert stage == "first"
        return httpx.Response(200, content=PDF_V1)

    with ArchiveStore(tmp_path) as store:
        collect(_args(tmp_path), store, clients(handler))
        stage = "failed"
        failed = collect(_args(tmp_path, "--refresh"), store, clients(handler))
        assert failed["inventory"]["failed"] == 1
        stage = "resume"
        resumed = collect(_args(tmp_path), store, clients(handler))
        assert resumed["status"] == "completed"
        assert resumed["run"]["cached"] == 1
        assert resumed["inventory"]["failed"] == 0
        assert store.documents()[0]["error"] is None
    assert pdf_requests == [PDF_URL, PDF_URL]


def test_changed_catalog_revision_triggers_refresh_without_explicit_flag(clients, tmp_path):
    changed = False
    conditional_headers = []

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            html = _catalog()
            if changed:
                html = html.replace(b"01/29/2026", b"10/14/2026")
            return httpx.Response(200, content=html)
        conditional_headers.append(dict(request.headers))
        return httpx.Response(200, content=PDF_V2 if changed else PDF_V1,
                              headers={"etag": '"changed"' if changed else '"first"'})

    with ArchiveStore(tmp_path) as store:
        collect(_args(tmp_path), store, clients(handler))
        changed = True
        report = collect(_args(tmp_path), store, clients(handler))
        assert report["run"]["downloaded"] == 1
        assert report["run"]["cached"] == 0
        assert conditional_headers[-1]["if-none-match"] == '"first"'
        latest = store.latest_revision(PDF_URL)
        assert (tmp_path / latest["relative_path"]).read_bytes() == PDF_V2


def test_current_products_prioritized_over_bulletins_when_download_bounded(clients, tmp_path):
    bulletin_url = "https://www.irs.gov/pub/irs-irbs/irb26-42.pdf"
    pdf_requests = []
    bulletin_html = (
        "<html><table><thead><tr><th>IRB in HTML</th><th>Download PDF</th><th>Title</th>"
        "</tr></thead><tbody><tr><td>Internal Revenue Bulletin: 2026-42</td>"
        f'<td><a href="{bulletin_url}">irb26-42.pdf</a></td>'
        "<td>Internal Revenue Bulletin: 2026-42</td></tr></tbody></table></html>"
    ).encode()

    def handler(request):
        url = str(request.url)
        if url == CATALOG_URLS["bulletins"]:
            return httpx.Response(200, content=bulletin_html)
        if url == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        pdf_requests.append(url)
        return httpx.Response(200, content=PDF_V1)

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--catalogs", "bulletins,current",
                               "--max-downloads", "1"), store, clients(handler))
        assert report["status"] == "partial"
        assert store.latest_revision(PDF_URL) is not None
        assert store.latest_revision(bulletin_url) is None
    assert pdf_requests == [PDF_URL]


def test_catalog_storage_guard_preserves_disk_reserve_before_snapshot_write(
    clients, tmp_path, monkeypatch,
):
    monkeypatch.setattr(irs_collect.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=_catalog())

    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path, "--discover-only"), store, clients(handler))
        assert report["status"] == "findings"
        assert report["inventory"]["catalog_pages"] == 0
        assert not report["coverage"]["current"]["enumeration_complete"]
        assert list((tmp_path / "catalogs/snapshots").iterdir()) == []


def test_copy_peak_guard_does_not_commit_original_or_leave_staging(
    clients, tmp_path, monkeypatch,
):
    free = 1024 * 1024

    def handler(request):
        if str(request.url) == CATALOG_URLS["current"]:
            return httpx.Response(200, content=_catalog())
        return httpx.Response(200, content=PDF_V1)

    monkeypatch.setattr(irs_collect.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    client = clients(handler)
    original_pdf = client.pdf

    def downloaded(*args, **kwargs):
        nonlocal free
        metadata = original_pdf(*args, **kwargs)
        free = len(PDF_V1)
        return metadata

    monkeypatch.setattr(client, "pdf", downloaded)
    with ArchiveStore(tmp_path) as store:
        report = collect(_args(tmp_path), store, client)
        assert report["status"] == "findings"
        assert report["run"]["downloaded"] == 0
        assert store.latest_revision(PDF_URL) is None
        assert list((tmp_path / ".staging").iterdir()) == []
        assert not list((tmp_path / "originals").rglob("*.pdf"))
