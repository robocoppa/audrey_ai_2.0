"""Read-only public reference scope and bounded PDF text-page regressions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from audrey.references import library as module
from audrey.references.irs_archive import ArchiveStore
from audrey.references.irs_catalog import CATALOG_URLS, CatalogEntry
from audrey.references.library import (
    ReferenceLibraryRegistry,
    ReferenceNotFound,
    ReferenceQueryError,
    ReferenceReadError,
    ReferenceUnavailable,
    document_id,
)


def _pdf(text="CURRENT_REFERENCE_MARKER " * 30, *, second_page="SECOND_PAGE_MARKER"):
    writer = PdfWriter()
    for value in [text, second_page]:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        if value:
            stream = DecodedStreamObject()
            escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    import io

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _entry(filename="f1040.pdf", *, catalog="current", product="Form 1040",
           title="U.S. Individual Income Tax Return", revision="2025", language="en"):
    directory = {"current": "irs-pdf", "prior": "irs-prior", "bulletins": "irs-irbs"}[catalog]
    return CatalogEntry(
        catalog=catalog, source_url=f"https://www.irs.gov/pub/{directory}/{filename}",
        product_number=product, title=title, revision_date=revision, posted_date="01/29/2026",
        language=language, document_kind="bulletin" if catalog == "bulletins" else "form",
        catalog_url=CATALOG_URLS[catalog],
    )


def _record(store, entries, *, content=None, snapshot=b"catalog snapshot", download=True):
    catalog = entries[0].catalog
    store.record_catalog_page(catalog, CATALOG_URLS[catalog], snapshot, entries)
    revisions = {}
    if download:
        for entry in entries:
            if entry.language != "en":
                continue
            temporary = store.root / ".staging" / (document_id(entry.source_url) + ".part")
            temporary.write_bytes(content or _pdf())
            revisions[entry.source_url] = store.record_download(
                entry.source_url, temporary, entry.source_url,
            )
            temporary.unlink()
    return revisions


def _registry(root, **values):
    return ReferenceLibraryRegistry({
        "enabled": True, "asof_year": 2026, "history_years": 5,
        "libraries": [{
            "id": "irs-federal", "name": "IRS Federal",
            "description": "Official retained public IRS documents", "root": str(root),
        }], **values,
    })


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "irs"
    current = _entry()
    old_active = _entry("p1.pdf", product="Publication 1", title="Your Rights As A Taxpayer",
                        revision="Sep 2017")
    foreign = _entry("f1040sp.pdf", product="Form 1040 (SP)", language="non_english")
    unknown = _entry("f1040xx.pdf", product="Form 1040 (XX)", language="unknown")
    prior = [
        _entry(f"f1040--{year}.pdf", catalog="prior", revision=str(year))
        for year in [2000, 2021, 2022, 2026, 2027]
    ] + [
        _entry("f1040--unknown.pdf", catalog="prior", revision="Undated"),
        _entry("f1040--ambiguous.pdf", catalog="prior", revision="2022 / 2023"),
    ]
    bulletins = [
        _entry(f"irb{str(year)[2:]}-01.pdf", catalog="bulletins", revision="",
               product=f"Internal Revenue Bulletin: {year}-01", title=f"IRB {year}-01")
        for year in [2000, 2021, 2022, 2026, 2027]
    ]
    with ArchiveStore(root) as store:
        revisions = _record(store, [current, old_active, foreign, unknown])
        revisions.update(_record(store, prior))
        revisions.update(_record(store, bulletins))
    return root, current, old_active, prior, bulletins, revisions


def test_current_default_recent_history_explicit_and_old_active_current_retained(archive):
    root, current, old_active, _, _, _ = archive
    registry = _registry(root)
    result = registry.catalog("irs-federal")
    assert {item["source_url"] for item in result["items"]} == {
        current.source_url, old_active.source_url,
    }
    assert result["catalog"] == "current"
    prior = registry.catalog("irs-federal", catalog="prior")
    assert {item["catalog_edition_year"] for item in prior["items"]} == {2022, 2026}
    bulletins = registry.catalog("irs-federal", catalog="bulletins")
    assert {item["catalog_edition_year"] for item in bulletins["items"]} == {2022, 2026}
    assert all(item["tax_year_applicability"] == "unknown" for item in result["items"])
    old = registry.lookup("irs-federal", document_id(old_active.source_url))
    assert old["catalog_edition_year"] == 2017
    assert old["applicable_tax_years"] is None


def test_direct_lookup_and_read_enforce_same_edition_window(archive):
    root, _, _, prior, bulletins, _ = archive
    registry = _registry(root)
    for entry in [prior[0], prior[1], prior[4], prior[5], prior[6],
                  bulletins[0], bulletins[1], bulletins[4]]:
        for operation in [registry.lookup, registry.read]:
            with pytest.raises(ReferenceNotFound):
                operation("irs-federal", document_id(entry.source_url), catalog=entry.catalog)
    recent = prior[2]
    with pytest.raises(ReferenceNotFound):
        registry.lookup("irs-federal", document_id(recent.source_url))
    with pytest.raises(ReferenceNotFound):
        registry.read("irs-federal", document_id(recent.source_url))
    assert registry.lookup(
        "irs-federal", document_id(recent.source_url), catalog="prior",
    )["catalog_edition_year"] == 2022
    assert "CURRENT_REFERENCE_MARKER" in registry.read(
        "irs-federal", document_id(recent.source_url), catalog="prior", limit=100,
    )["text"]


def test_registry_counts_only_accessible_catalog_metadata_not_owned_old_archive(archive):
    root, _, _, _, _, _ = archive
    listing = _registry(root).list_libraries()
    item = listing["items"][0]
    assert item["accessible_catalog_counts"] == {
        "current": {"catalogued": 2, "retained": 2},
        "prior": {"catalogued": 2, "retained": 2},
        "bulletins": {"catalogued": 2, "retained": 2},
    }
    assert item["historical_edition_window"] == [2022, 2026]
    assert item["default_catalog"] == "current"
    assert item["search_status"] == "not_indexed"
    assert "root" not in item and "path" not in json.dumps(item)


def test_exact_product_title_literal_filters_paging_and_bounds(archive):
    root, current, _, _, _, _ = archive
    registry = _registry(root)
    result = registry.catalog("irs-federal", product_number="form 1040")
    assert [item["source_url"] for item in result["items"]] == [current.source_url]
    result = registry.catalog("irs-federal", query="individual")
    assert [item["source_url"] for item in result["items"]] == [current.source_url]
    assert registry.catalog("irs-federal", query="' OR 1=1 --")["items"] == []
    assert registry.catalog("irs-federal", query="%")["items"] == []
    first = registry.catalog("irs-federal", limit=1)
    assert first["has_more"] and first["next_offset"] == 1
    second = registry.catalog("irs-federal", limit=1, offset=1)
    assert not second["has_more"] and second["next_offset"] is None
    assert first["items"][0]["document_id"] != second["items"][0]["document_id"]


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 21}, {"limit": True}, {"limit": 1.5},
    {"offset": -1}, {"offset": 100001}, {"query": ""}, {"query": "x" * 201},
    {"product_number": ""}, {"catalog": "private"}, {"year": 2000}, {"year": 2025},
])
def test_unsupported_queries_fail_closed(archive, kwargs):
    registry = _registry(archive[0])
    with pytest.raises(ReferenceQueryError):
        registry.catalog("irs-federal", **kwargs)


def test_active_catalog_snapshot_membership_not_ever_seen(tmp_path):
    root = tmp_path / "irs"
    retired = _entry()
    retained = _entry("p1.pdf", product="Publication 1")
    with ArchiveStore(root) as store:
        original = _record(store, [retired], snapshot=b"old current page")[retired.source_url]
        _record(store, [retained], snapshot=b"new current page")
    registry = _registry(root)
    assert [item["source_url"] for item in registry.catalog("irs-federal")["items"]] == [
        retained.source_url,
    ]
    with pytest.raises(ReferenceNotFound):
        registry.lookup("irs-federal", document_id(retired.source_url))
    with pytest.raises(ReferenceNotFound):
        registry.read("irs-federal", document_id(retired.source_url))
    assert (root / original["relative_path"]).is_file()


def test_document_identity_stable_latest_revision_pin_only(tmp_path):
    root = tmp_path / "irs"
    old = _entry(revision="2000")
    latest = _entry(revision="2026")
    with ArchiveStore(root) as store:
        first = _record(store, [old], content=_pdf("OLD_BODY"),
                        snapshot=b"old edition")[old.source_url]
        second = _record(store, [latest], content=_pdf("LATEST_BODY"),
                         snapshot=b"latest edition")[latest.source_url]
    registry = _registry(root)
    identifier = document_id(latest.source_url)
    assert identifier == document_id(old.source_url)
    with pytest.raises(ReferenceNotFound):
        registry.read("irs-federal", identifier, revision_sha256=first["sha256"])
    result = registry.read("irs-federal", identifier, revision_sha256=second["sha256"])
    assert result["text"].strip() == "LATEST_BODY"
    assert result["revision_sha256"] == second["sha256"]


def test_catalog_changed_since_retained_original_disclosed(tmp_path):
    root = tmp_path / "irs"
    old = _entry(revision="2025")
    latest = _entry(revision="2026")
    with ArchiveStore(root) as store:
        _record(store, [old], snapshot=b"downloaded edition")
        _record(store, [latest], snapshot=b"catalog updated", download=False)
    result = _registry(root).read("irs-federal", document_id(old.source_url))
    assert result["revision_date"] == "2026"
    assert result["archived_edition"]["revision_date"] == "2025"
    assert result["catalog_metadata_matches_original"] is False
    assert result["edition_status"] == "catalog_changed_since_download"


def test_real_pdf_selected_page_character_paging_and_precise_provenance(archive):
    root, current, _, _, _, revisions = archive
    registry = _registry(root)
    identifier = document_id(current.source_url)
    first = registry.read("irs-federal", identifier, page=1, limit=100)
    second = registry.read("irs-federal", identifier, page=1, offset=100, limit=100)
    assert first["page"] == 1 and first["total_pages"] == 2
    assert first["next_offset"] == 100
    assert first["text"] + second["text"] == registry.read(
        "irs-federal", identifier, page=1, limit=200,
    )["text"]
    assert first["source_url"] == current.source_url
    assert first["revision_sha256"] == revisions[current.source_url]["sha256"]
    last = registry.read("irs-federal", identifier, page=2)
    assert last["text"].strip() == "SECOND_PAGE_MARKER"
    assert not last["has_more"]
    assert "no OCR" in last["limitations"][0]
    assert "full document" in last["limitations"][1]


def test_blank_scan_discloses_no_text_layer_not_fabricated_content(tmp_path):
    root = tmp_path / "irs"
    entry = _entry()
    with ArchiveStore(root) as store:
        _record(store, [entry], content=_pdf("", second_page=""))
    result = _registry(root).read("irs-federal", document_id(entry.source_url))
    assert result["text"] == ""
    assert result["extraction_status"] == "no_text_layer"
    assert result["text_layer_available"] is False


@pytest.mark.parametrize("kwargs", [
    {"page": 0}, {"page": 10001}, {"page": True}, {"offset": -1},
    {"offset": 200001}, {"limit": 0}, {"limit": 4001}, {"limit": True},
])
def test_page_and_character_bounds(archive, kwargs):
    root, current, *_ = archive
    with pytest.raises(ReferenceQueryError):
        _registry(root).read("irs-federal", document_id(current.source_url), **kwargs)


def test_actual_page_or_offset_outside_pdf_refused(archive):
    root, current, *_ = archive
    registry = _registry(root)
    with pytest.raises(ReferenceQueryError, match="page"):
        registry.read("irs-federal", document_id(current.source_url), page=3)
    with pytest.raises(ReferenceQueryError, match="offset"):
        registry.read("irs-federal", document_id(current.source_url), offset=9000)


def test_missing_unknown_and_outside_original_same_not_found(archive, tmp_path):
    root, current, _, _, _, revisions = archive
    registry = _registry(root)
    identifier = document_id(current.source_url)
    original = root / revisions[current.source_url]["relative_path"]
    original.unlink()
    with pytest.raises(ReferenceNotFound, match="Reference document not found"):
        registry.read("irs-federal", identifier)
    with pytest.raises(ReferenceNotFound, match="Reference document not found"):
        registry.read("irs-federal", "0" * 64)
    with pytest.raises(ReferenceNotFound, match="Reference document not found"):
        registry.read("irs-federal", "../../private-records")
    outside = tmp_path / "private.pdf"
    outside.write_bytes(_pdf("PRIVATE_DATA_MUST_NEVER_READ"))
    with sqlite3.connect(root / "catalogs/archive.sqlite") as connection:
        connection.execute(
            "UPDATE revisions SET relative_path=? WHERE source_url=?",
            ("../../private.pdf", current.source_url),
        )
    connection.close()
    with pytest.raises(ReferenceNotFound, match="Reference document not found"):
        registry.read("irs-federal", identifier)
    assert not registry.lookup("irs-federal", identifier)["original_available"]


@pytest.mark.parametrize("parent_symlink", [False, True])
def test_original_symlink_or_symlink_parent_refused(archive, tmp_path, parent_symlink):
    root, current, _, _, _, revisions = archive
    original = root / revisions[current.source_url]["relative_path"]
    outside = tmp_path / "private.pdf"
    outside.write_bytes(original.read_bytes())
    if parent_symlink:
        directory = original.parent
        for item in directory.iterdir():
            item.unlink()
        directory.rmdir()
        directory.symlink_to(tmp_path, target_is_directory=True)
    else:
        original.unlink()
        original.symlink_to(outside)
    with pytest.raises(ReferenceNotFound):
        _registry(root).read("irs-federal", document_id(current.source_url))


def test_hash_mismatch_never_returns_modified_body(archive):
    root, current, _, _, _, revisions = archive
    original = root / revisions[current.source_url]["relative_path"]
    content = original.read_bytes()
    original.write_bytes(content.replace(b"CURRENT_REFERENCE_MARKER", b"PRIVATE_REFERENCE_MARKER"))
    with pytest.raises(ReferenceNotFound):
        _registry(root).read("irs-federal", document_id(current.source_url))


def test_registry_reads_create_or_modify_no_archive_files(archive):
    root, current, *_ = archive

    def snapshot():
        return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns,
                                             hashlib.sha256(path.read_bytes()).hexdigest())
                for path in root.rglob("*") if path.is_file()}

    before = snapshot()
    registry = _registry(root)
    registry.list_libraries()
    registry.catalog("irs-federal")
    registry.lookup("irs-federal", document_id(current.source_url))
    registry.read("irs-federal", document_id(current.source_url))
    assert snapshot() == before
    assert not (root / "catalogs/archive.sqlite-wal").exists()
    assert not (root / "catalogs/archive.sqlite-shm").exists()


def test_no_creation_for_missing_disabled_or_unregistered_library(tmp_path):
    root = tmp_path / "never-create"
    registry = _registry(root)
    assert registry.list_libraries()["items"][0]["status"] == "unavailable"
    with pytest.raises(ReferenceUnavailable):
        registry.catalog("irs-federal")
    with pytest.raises(ReferenceNotFound):
        registry.catalog("user-uploads")
    assert not root.exists()
    disabled = _registry(root, enabled=False)
    assert disabled.list_libraries() == {"enabled": False, "status": "disabled", "items": []}
    with pytest.raises(ReferenceNotFound):
        disabled.catalog("irs-federal")


@pytest.mark.parametrize("config", [
    {"enabled": "yes"}, {"enabled": True, "libraries": [{}]},
    {"enabled": True, "libraries": [{"id": "../private"}]},
    {"enabled": True, "libraries": "bad"}, {"history_years": 10},
    {"asof_year": True}, {"history_years": 0},
])
def test_invalid_server_configuration_rejected(config):
    with pytest.raises(ReferenceQueryError):
        ReferenceLibraryRegistry(config)


def test_root_symlink_archive_symlink_and_collect_lock_fail_closed(archive, tmp_path):
    root, *_ = archive
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ReferenceUnavailable):
        _registry(link).catalog("irs-federal")
    lock_path = root / ".collect.lock"
    lock_path.touch()
    with lock_path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReferenceUnavailable, match="in progress"):
            _registry(root).catalog("irs-federal")


def test_wall_clock_deadline_and_memory_cpu_failures_return_safe_error(archive, monkeypatch):
    root, current, *_ = archive
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(ReferenceReadError, match="20-second"):
        _registry(root).read("irs-federal", document_id(current.source_url))
    assert calls[0]["timeout"] == 20
    assert calls[0]["stderr"] == subprocess.DEVNULL


def test_worker_exit_failure_returns_safe_resource_error(archive, monkeypatch):
    root, current, *_ = archive

    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode=-9)

    monkeypatch.setattr(module.subprocess, "run", failed)
    with pytest.raises(ReferenceReadError, match="resource limits"):
        _registry(root).read("irs-federal", document_id(current.source_url))


def test_parent_rejects_oversized_worker_output(archive, monkeypatch):
    root, current, *_ = archive

    def oversized(*args, **kwargs):
        kwargs["stdout"].write(b"x" * 32769)
        return subprocess.CompletedProcess(args[0], returncode=0)

    monkeypatch.setattr(module.subprocess, "run", oversized)
    with pytest.raises(ReferenceReadError, match="resource limits"):
        _registry(root).read("irs-federal", document_id(current.source_url))


def test_child_page_extraction_limit_disclosed(archive, monkeypatch):
    root, current, _, _, _, revisions = archive
    revision = revisions[current.source_url]
    text = "x" * (module.MAX_PAGE_CHARS + 100)
    reader = type("FakeReader", (), {"is_encrypted": False, "pages": [
        type("FakePage", (), {"extract_text": lambda _: text})(),
    ]})()
    monkeypatch.setattr("pypdf.PdfReader", lambda *_, **__: reader)
    result = module._reader_worker({
        "root": str(root), "relative_path": revision["relative_path"],
        "bytes": revision["bytes"], "sha256": revision["sha256"],
        "page": 1, "offset": 199999, "limit": 4000,
    })
    assert result["text"] == "x"
    assert result["total_chars"] == 200100
    assert result["available_chars"] == 200000
    assert result["page_text_truncated"]
    assert result["next_offset"] is None


def test_crash_retained_wal_readonly_reads_commits_without_mutating_files(
    tmp_path,
):
    root = tmp_path / "irs"
    entry = _entry()
    with ArchiveStore(root) as store:
        _record(store, [entry])
    database = root / "catalogs/archive.sqlite"
    script = (
        "import sqlite3,os,sys;"
        "db=sqlite3.connect(sys.argv[1]);"
        "db.execute('PRAGMA journal_mode=WAL');"
        "db.execute(\"UPDATE observations SET title='COMMITTED_CRASH_WAL_TITLE'\");"
        "db.commit();os._exit(0)"
    )
    subprocess.run([module.sys.executable, "-c", script, str(database)], check=True)
    paths = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
    assert all(path.is_file() for path in paths)
    assert paths[1].stat().st_size > 0
    before = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
              for path in paths}
    modes = {path: path.stat().st_mode for path in paths}
    for path in paths:
        path.chmod(0o444)
    try:
        result = _registry(root).catalog("irs-federal")
        assert result["items"][0]["title"] == "COMMITTED_CRASH_WAL_TITLE"
        after = {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                 for path in paths}
        assert after == before
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def test_nonempty_wal_without_shm_refused_without_sidecar_creation(archive):
    root, *_ = archive
    wal = root / "catalogs/archive.sqlite-wal"
    wal.write_bytes(b"uncheckpointed state")
    with pytest.raises(ReferenceUnavailable):
        _registry(root).catalog("irs-federal")
    assert not (root / "catalogs/archive.sqlite-shm").exists()


def test_writable_shm_refused_to_preserve_readonly_contract(archive):
    root, *_ = archive
    (root / "catalogs/archive.sqlite-wal").write_bytes(b"uncheckpointed state")
    (root / "catalogs/archive.sqlite-shm").write_bytes(b"shared memory")
    with pytest.raises(ReferenceUnavailable, match="read-only mount"):
        _registry(root).catalog("irs-federal")

def test_title_query_accepts_bounded_200_characters(archive):
    registry = _registry(archive[0])
    assert registry.catalog("irs-federal", query="x" * 200)["items"] == []


def test_failed_current_refresh_cannot_return_previous_original_as_current(archive):
    root, current, _, _, _, revisions = archive
    original = root / revisions[current.source_url]["relative_path"]
    before = original.read_bytes()
    with ArchiveStore(root) as store:
        store.record_failure(current.source_url, "Official source could not be refreshed")
    registry = _registry(root)
    item = registry.lookup("irs-federal", document_id(current.source_url))
    assert item["status"] == "failed"
    assert item["original_available"]
    with pytest.raises(ReferenceUnavailable, match="update failed"):
        registry.read("irs-federal", document_id(current.source_url))
    assert original.read_bytes() == before


def test_recent_catalog_metadata_cannot_expose_old_retained_historical_body(tmp_path):
    root = tmp_path / "irs"
    old = _entry("f1040--2025.pdf", catalog="prior", revision="2010")
    newest = _entry("f1040--2025.pdf", catalog="prior", revision="2025")
    with ArchiveStore(root) as store:
        retained = _record(store, [old], content=_pdf("OLD_BODY"),
                           snapshot=b"older edition")[old.source_url]
        _record(store, [newest], snapshot=b"updated catalog edition", download=False)
    registry = _registry(root)
    for operation in [registry.lookup, registry.read]:
        with pytest.raises(ReferenceNotFound):
            operation("irs-federal", document_id(newest.source_url), catalog="prior")
    assert (root / retained["relative_path"]).is_file()
