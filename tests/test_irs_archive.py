"""The IRS archive preserves originals and resumes independently of a search index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from audrey.references.irs_archive import ArchiveStore
from audrey.references.irs_catalog import CatalogEntry

SOURCE = "https://www.irs.gov/pub/irs-pdf/f1040.pdf"
CURRENT = "https://www.irs.gov/forms-instructions-and-publications?items_per_page=200"
PRIOR = "https://www.irs.gov/prior-year-forms-and-instructions?items_per_page=200"
PDF_ONE = b"%PDF-1.7\nfirst original\n%%EOF\n"
PDF_TWO = b"%PDF-1.7\nsecond original\n%%EOF\n"


def entry(**changes: str) -> CatalogEntry:
    return replace(CatalogEntry(
        catalog="current", source_url=SOURCE, product_number="Form 1040",
        title="U.S. Individual Income Tax Return", revision_date="2025",
        posted_date="Jan 01, 2026", language="en", document_kind="form",
        catalog_url=CURRENT,
    ), **changes)


def discover(store: ArchiveStore, *entries: CatalogEntry, html: bytes = b"catalog one") -> None:
    store.record_catalog_page("current", CURRENT, html, entries or [entry()])


def stage(store: ArchiveStore, content: bytes = PDF_ONE) -> Path:
    path = store.root / ".staging/incoming.part"
    path.write_bytes(content)
    return path


def test_restart_preserves_catalog_original_checksum_and_inventory(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root) as store:
        discover(store)
        recorded = store.record_download(SOURCE, stage(store), SOURCE, etag='"revision-one"')
        manifest = json.loads(store.export_manifest().read_text())
        assert manifest["revisions"][0]["sha256"] == hashlib.sha256(PDF_ONE).hexdigest()
    with ArchiveStore(root) as store:
        document = store.documents()[0]
        assert document["status"] == "downloaded"
        assert document["latest_revision"] == recorded
        assert document["etag"] == '"revision-one"'
        assert (root / recorded["relative_path"]).read_bytes() == PDF_ONE
        assert store.summary()["downloaded_bytes"] == len(PDF_ONE)
        assert store.catalog_urls("current") == [CURRENT]
    with sqlite3.connect(root / "catalogs/archive.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_changed_body_keeps_both_originals_and_revision_metadata(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        first = store.record_download(SOURCE, stage(store), SOURCE)
        discover(store, entry(revision_date="2026"), html=b"catalog two")
        second = store.record_download(SOURCE, stage(store, PDF_TWO), SOURCE)
        assert second["sha256"] != first["sha256"]
        assert store.latest_revision(SOURCE) == second
        assert (tmp_path / first["relative_path"]).read_bytes() == PDF_ONE
        assert (tmp_path / second["relative_path"]).read_bytes() == PDF_TWO
        assert first["metadata"]["revision_date"] == "2025"
        assert second["metadata"]["revision_date"] == "2026"
        assert store.summary()["revisions"] == 2
        assert store.summary()["revision_bytes"] == len(PDF_ONE) + len(PDF_TWO)


def test_reverted_body_updates_latest_pointer_without_duplicating_content(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        first = store.record_download(SOURCE, stage(store), SOURCE)
        store.record_download(SOURCE, stage(store, PDF_TWO), SOURCE)
        repeated = store.record_download(SOURCE, stage(store), SOURCE)
        assert repeated["id"] == first["id"]
        assert store.latest_revision(SOURCE)["sha256"] == first["sha256"]
        assert store.summary()["revisions"] == 2


def test_failed_refresh_never_discards_a_retained_original(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        original = store.record_download(SOURCE, stage(store), SOURCE)
        store.record_failure(SOURCE, "temporary HTTP failure")
        document = store.documents()[0]
        assert document["status"] == "failed"
        assert document["failure_count"] == 1
        assert document["latest_revision"] == original
        assert (tmp_path / original["relative_path"]).read_bytes() == PDF_ONE
        assert store.record_not_modified(SOURCE) == original
        assert store.documents()[0]["status"] == "downloaded"
        assert store.documents()[0]["error"] is None


def test_all_languages_retained_but_only_known_english_pdf_kinds_eligible(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store, entry(), entry(
            source_url="https://www.irs.gov/pub/irs-pdf/f1040sp.pdf",
            product_number="Form 1040 (sp)", language="non_english",
        ), entry(
            source_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",
            product_number="Publication 17", language="unknown", document_kind="publication",
        ), entry(
            source_url="https://www.irs.gov/pub/irs-pdf/n26-01.pdf",
            product_number="Notice 2026-01", document_kind="unsupported",
        ))
        summary = store.summary()
        assert summary["discovered"] == 4
        assert summary["eligible"] == 1
        assert summary["skipped_non_english"] == 1
        assert summary["skipped_unknown_language"] == 1
        assert summary["skipped_document_kind"] == 1
        assert len(store.documents()) == 4
        for source_url in (
            "https://www.irs.gov/pub/irs-pdf/f1040sp.pdf",
            "https://www.irs.gov/pub/irs-pdf/p17.pdf",
            "https://www.irs.gov/pub/irs-pdf/n26-01.pdf",
        ):
            with pytest.raises(ValueError, match="eligible"):
                store.record_download(source_url, stage(store), source_url)
        assert store.summary()["revisions"] == 0


def test_one_source_retains_multiple_catalog_memberships(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        store.record_catalog_page("prior", PRIOR, b"prior catalog", [
            entry(catalog="prior", catalog_url=PRIOR, revision_date="2024"),
        ])
        assert store.documents()[0]["catalogs"] == ["current", "prior"]
        assert store.documents(["current"])[0]["revision_date"] == "2025"
        assert store.documents(["prior"])[0]["revision_date"] == "2024"
        assert store.documents(["missing"]) == []
        assert store.documents([]) == []
        assert store.summary()["discovered"] == 1
        assert store.summary()["per_catalog"]["prior"]["discovered"] == 1


def test_catalog_snapshots_and_metadata_changes_survive_repeated_content(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        discover(store, entry(language="unknown"), html=b"catalog two")
        assert store.documents()[0]["eligible"] is False
        discover(store)
        assert store.documents()[0]["eligible"] is True
        summary = store.summary()
        assert summary["catalog_pages"] == 1
        assert summary["catalog_snapshots"] == 2
        snapshots = list((tmp_path / "catalogs/snapshots").glob("*.html"))
        assert {path.read_bytes() for path in snapshots} == {b"catalog one", b"catalog two"}
    with sqlite3.connect(tmp_path / "catalogs/archive.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2


def test_catalog_page_preserves_paging_metadata(tmp_path: Path) -> None:
    next_url = CURRENT + "&page=1"
    with ArchiveStore(tmp_path) as store:
        store.record_catalog_page(
            "current", CURRENT, b"catalog", [entry()], [next_url], total_rows=1234,
        )
        page = store.catalog_pages("current")[0]
        assert page["next_urls"] == [next_url]
        assert page["total_rows"] == 1234
        assert (tmp_path / page["snapshot_path"]).read_bytes() == b"catalog"


def test_removed_catalog_entry_does_not_delete_owned_archive(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        original = store.record_download(SOURCE, stage(store), SOURCE)
        store.record_catalog_page("current", CURRENT, b"empty updated catalog", [])
        assert store.latest_revision(SOURCE) == original
        assert len(store.documents()) == 1
        assert (tmp_path / original["relative_path"]).is_file()


def test_product_number_cannot_choose_archive_path(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store, entry(product_number="../../../../../outside/evil"))
        original = store.record_download(SOURCE, stage(store), SOURCE)
        path = tmp_path / original["relative_path"]
        assert path.parent == tmp_path / "originals/current"
        assert path.is_relative_to(tmp_path)
        assert ".." not in path.name
        assert original["sha256"] in path.name


def test_rejects_non_pdf_without_creating_a_revision(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        with pytest.raises(ValueError, match="PDF signature"):
            store.record_download(SOURCE, stage(store, b"<html>denied</html>"), SOURCE)
        assert store.summary()["revisions"] == 0
        assert list((tmp_path / "originals").rglob("*.pdf")) == []


def test_staging_outside_root_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(PDF_ONE)
    with ArchiveStore(tmp_path / "archive") as store:
        discover(store)
        with pytest.raises(ValueError, match="staged inside"):
            store.record_download(SOURCE, outside, SOURCE)
        inside = store.root / "unexpected.pdf"
        inside.write_bytes(PDF_ONE)
        with pytest.raises(ValueError, match=r"beneath \.staging"):
            store.record_download(SOURCE, inside, SOURCE)


def test_staging_symbolic_link_escape_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(PDF_ONE)
    with ArchiveStore(tmp_path / "archive") as store:
        discover(store)
        symlink = store.root / ".staging/link.part"
        symlink.symlink_to(outside)
        with pytest.raises(ValueError, match="Symbolic links"):
            store.record_download(SOURCE, symlink, SOURCE)
        assert outside.read_bytes() == PDF_ONE


def test_original_directory_symbolic_link_escape_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with ArchiveStore(tmp_path / "archive") as store:
        discover(store)
        (store.root / "originals/current").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="Symbolic links"):
            store.record_download(SOURCE, stage(store), SOURCE)
        assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", ["catalogs", "catalogs/archive.sqlite"])
def test_existing_storage_symbolic_links_are_refused(tmp_path: Path, name: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "archive"
    root.mkdir()
    symlink = root / name
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic links"):
        ArchiveStore(root)
    assert list(outside.iterdir()) == []


def test_corrupt_original_is_never_overwritten_silently(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        original = store.record_download(SOURCE, stage(store), SOURCE)
        target = tmp_path / original["relative_path"]
        target.write_bytes(b"unexpected bytes")
        with pytest.raises(ValueError, match="unexpected content"):
            store.record_download(SOURCE, stage(store), SOURCE)
        with pytest.raises(ValueError, match="missing or has changed"):
            store.record_not_modified(SOURCE)
        assert target.read_bytes() == b"unexpected bytes"
        assert store.summary()["revisions"] == 1


def test_not_modified_requires_a_retained_original(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        with pytest.raises(ValueError, match="No archived"):
            store.record_not_modified(SOURCE)


@pytest.mark.parametrize("url", [
    "https://example.com/f1040.pdf",
    "http://www.irs.gov/pub/irs-pdf/f1040.pdf",
    "https://www.irs.gov/pub/irs-pdf/../../outside.pdf",
    "https://www.irs.gov/pub/irs-pdf/f1040.pdf?token=example",
])
def test_source_urls_use_the_shared_official_allowlist(tmp_path: Path, url: str) -> None:
    with ArchiveStore(tmp_path) as store:
        with pytest.raises(ValueError):
            discover(store, entry(source_url=url))
        assert store.documents() == []
        assert list((tmp_path / "catalogs/snapshots").iterdir()) == []


def test_report_names_and_symbolic_link_targets_are_refused(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        for name in ("../outside.json", str(tmp_path / "outside.json"), ".."):
            with pytest.raises(ValueError, match="single file name"):
                store.save_report(name, {"schema": 1})
        outside = tmp_path / "outside.json"
        outside.write_text("original")
        (store.root / "reports/escaped.json").symlink_to(outside)
        with pytest.raises(ValueError, match="Symbolic links"):
            store.save_report("escaped.json", {"schema": 1})
        assert outside.read_text() == "original"
        report = store.save_report("inventory.json", store.summary())
        assert json.loads(report.read_text())["schema"] == 1


def test_failure_for_unknown_source_cannot_create_inventory(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        with pytest.raises(ValueError, match="undiscovered"):
            store.record_failure(SOURCE, "failure")
        assert store.documents() == []


def test_missing_original_can_be_restored_without_adding_a_revision(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        original = store.record_download(SOURCE, stage(store), SOURCE)
        path = tmp_path / original["relative_path"]
        path.unlink()
        restored = store.record_download(SOURCE, stage(store), SOURCE)
        assert restored["id"] == original["id"]
        assert path.read_bytes() == PDF_ONE
        assert store.summary()["revisions"] == 1


def test_interrupted_publication_leaves_no_partial_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ArchiveStore(tmp_path) as store:
        discover(store)
        with monkeypatch.context() as patch:
            def fail_link(*args: object, **kwargs: object) -> None:
                raise OSError("interrupted before publication")
            patch.setattr(os, "link", fail_link)
            with pytest.raises(OSError, match="before publication"):
                store.record_download(SOURCE, stage(store), SOURCE)
        assert list((tmp_path / "originals/current").iterdir()) == []
        assert store.summary()["revisions"] == 0
        original = store.record_download(SOURCE, stage(store), SOURCE)
        assert (tmp_path / original["relative_path"]).read_bytes() == PDF_ONE


def test_official_english_pdf_with_other_kind_is_still_archived(tmp_path: Path) -> None:
    source_url = "https://www.irs.gov/pub/irs-pdf/n26-01.pdf"
    with ArchiveStore(tmp_path) as store:
        discover(store, entry(
            source_url=source_url, product_number="Notice 2026-01", document_kind="other",
        ))
        original = store.record_download(source_url, stage(store), source_url)
        assert store.documents()[0]["eligible"] is True
        assert (tmp_path / original["relative_path"]).read_bytes() == PDF_ONE
        assert store.summary()["skipped_document_kind"] == 0
