"""Publication remains atomic across partial builds, failures and resumed checkpoints."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from audrey.references.releases import (
    ReleaseError,
    ReleaseStateError,
    ReleaseStore,
    reference_collection_name,
)


def metadata(number=17, *, sha=None, catalog="current"):
    source = f"https://www.irs.gov/pub/irs-pdf/p{number}.pdf"
    return {
        "library_id": "irs-federal",
        "document_id": hashlib.sha256(source.encode()).hexdigest(),
        "revision_sha256": sha or hashlib.sha256(f"body-{number}".encode()).hexdigest(),
        "source_url": source,
        "product_number": f"Publication {number}",
        "title": f"Official guidance {number}",
        "catalog": catalog,
        "language": "en",
        "catalog_edition_year": 2026,
        "applicable_tax_years": None,
    }


def begin(store, docs=None, *, fingerprint="a" * 64, coverage=None, release_id=None):
    return store.begin(
        "irs-federal",
        "current",
        fingerprint=fingerprint,
        provenance={"asof_year": 2026, "history_years": 5, "embedding_model": "local-embed"},
        coverage=coverage or {"scope": "subset", "source_documents": 2407},
        documents=[metadata()] if docs is None else docs,
        release_id=release_id,
    )


def checkpoint(store, release, doc=None, *, status="indexed", chunks=None, error=""):
    doc = doc or metadata()
    if chunks is None:
        chunks = (
            [
                {
                    "document_id": doc["document_id"],
                    "chunk_index": 0,
                    "page": 1,
                    "offset": 10,
                    "text": "Official reference text.",
                }
            ]
            if status == "indexed"
            else []
        )
    store.put_document(
        release["id"],
        doc["document_id"],
        metadata=doc,
        page_count=0 if status == "failed" else 2,
        status=status,
        chunks=chunks,
        error=error,
    )


def test_durable_release_document_chunk_and_publication(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    checkpoint(store, release)
    published = store.publish(release["id"], verified_chunks=1)
    assert published["status"] == "published"
    assert published["coverage"] == {
        "scope": "subset",
        "source_documents": 2407,
        "selected_documents": 1,
        "complete_for_selected_documents": True,
    }
    assert published["collection"] == reference_collection_name("irs-federal")
    reopened = ReleaseStore(tmp_path)
    assert reopened.published("irs-federal") == published
    doc = reopened.document(release["id"], metadata()["document_id"])
    assert doc["status"] == "indexed" and doc["page_count"] == 2 and doc["chunk_count"] == 1
    assert doc["metadata"]["revision_sha256"] == metadata()["revision_sha256"]
    chunk = reopened.chunk(release["id"], doc["document_id"], 0)
    assert chunk["page"] == 1 and chunk["offset"] == 10 and chunk["char_end"] == 34
    assert list(reopened.chunks(release["id"])) == [chunk]


def test_partial_or_failed_replacement_never_replaces_active_release(tmp_path):
    store = ReleaseStore(tmp_path)
    original = begin(store)
    checkpoint(store, original)
    store.publish(original["id"], verified_chunks=1)
    newer = begin(store, docs=[metadata(17), metadata(18)], fingerprint="b" * 64)
    checkpoint(store, newer)
    with pytest.raises(ReleaseStateError, match="incomplete"):
        store.publish(newer["id"], verified_chunks=1)
    assert store.published("irs-federal")["id"] == original["id"]
    checkpoint(
        store, newer, metadata(18), status="failed", error="Public PDF text could not be read"
    )
    with pytest.raises(ReleaseStateError, match="incomplete"):
        store.publish(newer["id"], verified_chunks=1)
    store.fail(newer["id"], "Embedding service unavailable")
    assert store.latest_build("irs-federal")["status"] == "failed"
    assert store.published("irs-federal")["id"] == original["id"]


def test_exact_remote_chunk_count_required(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    checkpoint(store, release)
    for count in (0, 2):
        with pytest.raises(ReleaseStateError, match="verification is incomplete"):
            store.publish(release["id"], verified_chunks=count)
    assert store.published("irs-federal") is None
    assert store.publish(release["id"], verified_chunks=1)["status"] == "published"


def test_no_text_document_is_accounted_for_without_remote_points(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    checkpoint(store, release, status="no_text")
    published = store.publish(release["id"], verified_chunks=0)
    assert published["no_text_documents"] == 1
    assert published["indexed_documents"] == 0
    assert published["completed_documents"] == 1


def test_failed_release_resume_preserves_successful_checkpoints(tmp_path):
    store = ReleaseStore(tmp_path)
    docs = [metadata(17), metadata(18)]
    release = begin(store, docs)
    checkpoint(store, release, docs[0])
    checkpoint(store, release, docs[1], status="failed", error="Temporary extraction failure")
    store.fail(release["id"], "Temporary service failure")
    resumed = begin(ReleaseStore(tmp_path), docs)
    assert resumed["id"] == release["id"] and resumed["status"] == "building"
    assert resumed["indexed_documents"] == 1 and resumed["failed_documents"] == 1
    assert store.document(release["id"], docs[0]["document_id"])["chunk_count"] == 1
    checkpoint(store, resumed, docs[1])
    assert store.publish(resumed["id"], verified_chunks=2)["completed_documents"] == 2


def test_explicit_resume_requires_matching_scope_policy_and_unpublished_state(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    assert begin(store, release_id=release["id"])["id"] == release["id"]
    with pytest.raises(ReleaseStateError, match="cannot resume"):
        begin(store, fingerprint="b" * 64, release_id=release["id"])
    checkpoint(store, release)
    store.publish(release["id"], verified_chunks=1)
    with pytest.raises(ReleaseStateError, match="cannot resume"):
        begin(store, release_id=release["id"])


def test_resume_cannot_swap_selected_original_revision_or_document(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    for docs in ([metadata(18)], [metadata(17, sha="c" * 64)], [metadata(17), metadata(18)]):
        with pytest.raises(ReleaseStateError, match="changed"):
            begin(store, docs)
    assert store.release(release["id"])["expected_documents"] == 1


def test_unpublished_release_does_not_appear_as_active(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    assert store.published("irs-federal") is None
    assert store.latest_build("irs-federal")["id"] == release["id"]


def test_published_release_is_immutable_and_cannot_republish_over_a_newer_one(tmp_path):
    store = ReleaseStore(tmp_path)
    first = begin(store)
    checkpoint(store, first)
    store.publish(first["id"], verified_chunks=1)
    second = begin(store, fingerprint="b" * 64)
    checkpoint(store, second)
    store.publish(second["id"], verified_chunks=1)
    for operation in (
        lambda: checkpoint(store, first),
        lambda: store.fail(first["id"], "Cannot mutate a published release"),
        lambda: store.publish(first["id"], verified_chunks=1),
    ):
        with pytest.raises(ReleaseStateError):
            operation()
    assert store.published("irs-federal")["id"] == second["id"]


def test_document_transaction_rolls_back_partial_chunk_generator(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)

    def failed_chunks():
        yield {"chunk_index": 0, "page": 1, "offset": 0, "text": "First valid chunk"}
        raise RuntimeError("Extraction stream interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        checkpoint(store, release, chunks=failed_chunks())
    assert store.document(release["id"], metadata()["document_id"])["status"] == "pending"
    assert list(store.chunks(release["id"])) == []
    checkpoint(store, release)
    assert store.release(release["id"])["chunk_count"] == 1


@pytest.mark.parametrize(
    "bad_chunk",
    [
        {"chunk_index": 1, "page": 1, "offset": 0, "text": "Wrong index"},
        {"chunk_index": 0, "page": 3, "offset": 0, "text": "Outside PDF"},
        {"chunk_index": 0, "page": 1, "offset": 200000, "text": "Outside bounded page"},
        {"chunk_index": 0, "page": 1, "offset": 0, "text": ""},
        {"chunk_index": 0, "page": 1, "offset": 0, "text": "x" * 8001},
        {"chunk_index": 0, "page": 1, "offset": 0, "text": "Wrong owner", "document_id": "c" * 64},
    ],
)
def test_invalid_chunk_cannot_complete_checkpoint(tmp_path, bad_chunk):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    with pytest.raises(ReleaseStateError):
        checkpoint(store, release, chunks=[bad_chunk])
    assert store.document(release["id"], metadata()["document_id"])["status"] == "pending"


def test_original_checksum_cannot_change_at_checkpoint(tmp_path):
    store = ReleaseStore(tmp_path)
    release = begin(store)
    with pytest.raises(ReleaseStateError, match="original identity"):
        checkpoint(store, release, metadata(sha="b" * 64))
    assert store.release(release["id"])["pending_documents"] == 1


def test_private_asset_fields_are_not_persisted_and_year_evidence_is_retained(tmp_path):
    store = ReleaseStore(tmp_path)
    doc = {
        **metadata(),
        "_asset_path": Path("/private/uploads/source.pdf"),
        "root": "/private",
        "applicable_tax_years": [2025],
        "applicability_basis": "explicit_return_year",
        "verification_page": 1,
        "verification_quote": "For use in preparing 2025 Returns",
        "verification_sha256": metadata()["revision_sha256"],
    }
    release = begin(store, [doc])
    checkpoint(store, release, doc)
    stored = store.document(release["id"], doc["document_id"])
    assert "_asset_path" not in stored and "root" not in stored
    assert stored["applicable_tax_years"] == [2025]
    assert stored["metadata"]["verification_sha256"] == doc["revision_sha256"]


def test_iterator_can_write_while_yielding_without_holding_a_read_snapshot(tmp_path):
    store = ReleaseStore(tmp_path)
    docs = [metadata(i) for i in range(200)]
    release = begin(store, (item for item in docs))
    seen = []
    for doc in store.documents(release["id"]):
        seen.append(doc["document_id"])
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(checkpoint, store, release, doc["metadata"]).result(timeout=5)
    assert len(set(seen)) == 200
    assert store.publish(release["id"], verified_chunks=200)["indexed_documents"] == 200
    assert len(list(store.chunks(release["id"]))) == 200


def test_two_store_instances_can_read_write_from_worker_threads(tmp_path):
    store = ReleaseStore(tmp_path)
    other = ReleaseStore(tmp_path)
    docs = [metadata(i) for i in range(8)]
    release = begin(store, docs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(checkpoint, other, release, item) for item in docs]
        for future in futures:
            future.result(timeout=5)
    assert store.publish(release["id"], verified_chunks=8)["completed_documents"] == 8


@pytest.mark.parametrize(
    "identifier", ["../private", "/data/private", "IRS", "irs/federal", "a" * 65]
)
def test_collection_names_never_accept_paths_or_unregistered_name_shapes(identifier):
    with pytest.raises(ReleaseStateError):
        reference_collection_name(identifier)


@pytest.mark.parametrize("target", ["index.sqlite", "index.sqlite-wal", "index.sqlite-shm"])
def test_state_sidecar_symlinks_are_refused(tmp_path, target):
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    root = tmp_path / "index"
    root.mkdir()
    (root / target).symlink_to(outside)
    with pytest.raises(ReleaseStateError, match="Symbolic links"):
        ReleaseStore(root)
    assert outside.read_text() == "unchanged"


def test_ancestor_symlink_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ReleaseStateError, match="Symbolic links"):
        ReleaseStore(tmp_path / "alias/nested")
    assert list(outside.iterdir()) == []


def test_duplicate_selected_documents_rollback_initial_release(tmp_path):
    store = ReleaseStore(tmp_path)
    with pytest.raises(ReleaseError):
        begin(store, [metadata(), metadata()])
    assert store.latest_build("irs-federal") is None


def test_empty_or_non_english_selection_is_rejected(tmp_path):
    store = ReleaseStore(tmp_path)
    with pytest.raises(ReleaseStateError, match="selected documents"):
        begin(store, [])
    doc = {**metadata(), "language": "non_english"}
    with pytest.raises(ReleaseStateError, match="English"):
        begin(store, [doc])
    assert store.latest_build("irs-federal") is None
