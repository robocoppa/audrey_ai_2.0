"""An approved source gap can publish without hiding failures or replaying good work."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from test_reference_indexing import services as _services

from audrey.references import indexing
from audrey.references.library import ReferenceReadError, ReferenceUnavailable

services = _services


async def _failed_build(services, monkeypatch, *, all_fail=False):
    registry, releases, qdrant, embedder, _, _ = services
    selected = list(registry.iter_index_documents("irs-federal"))
    original = indexing._extract
    failed_asset = selected[0]["_asset"]

    def extraction(asset):
        if all_fail or asset == failed_asset:
            raise ReferenceReadError("PDF text contains an unsupported character mapping")
        return original(asset)

    worker = Mock(side_effect=extraction)
    monkeypatch.setattr(indexing, "_extract", worker)
    report = await indexing.build_reference_index(registry, releases, qdrant, embedder)
    assert report["status"] == "failed"
    return report, selected, worker, original


async def test_exact_approved_gap_resumes_saved_release_and_remains_honest(services, monkeypatch):
    registry, releases, qdrant, embedder, _, _ = services
    failed, selected, worker, _ = await _failed_build(services, monkeypatch)
    gap = selected[0]
    embedder.embed_many.reset_mock()
    worker.reset_mock()
    upsert = AsyncMock(wraps=qdrant.upsert_text)
    monkeypatch.setattr(qdrant, "upsert_text", upsert)
    resumed = await indexing.build_reference_index(
        registry, releases, qdrant, embedder, release_id=failed["release_id"],
        allowed_extraction_gaps=(gap["document_id"],),
    )
    assert resumed["status"] == "published", resumed
    assert resumed["release_id"] == failed["release_id"]
    assert resumed["resumed_documents"] == resumed["indexed_documents"] == 1
    assert resumed["failed_documents"] == 1
    assert resumed["pending_documents"] == 0
    assert resumed["coverage"]["complete_for_selected_documents"] is False
    assert resumed["coverage"]["gap_count"] == 1
    listed = resumed["coverage"]["content_search_gaps"]
    assert len(listed) == 1 and listed[0]["document_id"] == gap["document_id"]
    assert listed[0]["product_number"] == gap["product_number"]
    assert listed[0]["original_available"] is True
    assert listed[0]["reason"] == "PDF text contains an unsupported character mapping"
    assert releases.document(resumed["release_id"], gap["document_id"])["status"] == "failed"
    worker.assert_called_once_with(gap["_asset"])
    embedder.embed_many.assert_not_awaited()
    upsert.assert_not_awaited()
    assert registry.catalog("irs-federal", product_number=gap["product_number"])["returned"] == 1
    assert registry.read("irs-federal", gap["document_id"], page=1)["text"]

    # The normal unchanged-publication path must not report a perfect index.
    unchanged = await indexing.build_reference_index(registry, releases, qdrant, embedder)
    assert unchanged["status"] == "published" and unchanged["unchanged"]
    assert unchanged["coverage"] == resumed["coverage"]
    assert unchanged["failed_documents"] == 1
    assert unchanged["resumed_documents"] == 1
    assert unchanged["failures"][0]["document_id"] == gap["document_id"]
    embedder.embed_many.assert_not_awaited()


async def test_another_failed_source_still_blocks_approved_gap_publication(services, monkeypatch):
    registry, releases, qdrant, embedder, _, _ = services
    failed, selected, _, _ = await _failed_build(services, monkeypatch, all_fail=True)
    resumed = await indexing.build_reference_index(
        registry, releases, qdrant, embedder, release_id=failed["release_id"],
        allowed_extraction_gaps=(selected[0]["document_id"],),
    )
    assert resumed["status"] == "failed"
    assert resumed["failed_documents"] == 2
    assert resumed["coverage"]["complete_for_selected_documents"] is False
    assert releases.published("irs-federal") is None
    embedder.embed_many.assert_not_awaited()


async def test_a_recovered_approved_source_is_indexed_without_a_gap(services, monkeypatch):
    registry, releases, qdrant, embedder, _, _ = services
    failed, selected, _, original = await _failed_build(services, monkeypatch)
    monkeypatch.setattr(indexing, "_extract", original)
    embedder.embed_many.reset_mock()
    resumed = await indexing.build_reference_index(
        registry, releases, qdrant, embedder, release_id=failed["release_id"],
        allowed_extraction_gaps=(selected[0]["document_id"],),
    )
    assert resumed["status"] == "published", resumed
    assert resumed["indexed_documents"] == 2 and resumed["resumed_documents"] == 1
    assert resumed["coverage"]["complete_for_selected_documents"] is True
    assert not resumed["coverage"].get("content_search_gaps")
    assert resumed["failed_documents"] == resumed["pending_documents"] == 0
    embedder.embed_many.assert_awaited_once()


@pytest.mark.parametrize("flags", [
    {"allowed_extraction_gaps": ("a" * 64,)},
    {"release_id": "rel_" + "b" * 32, "rebuild": True,
     "allowed_extraction_gaps": ("a" * 64,)},
])
async def test_gap_override_requires_saved_release_before_model_metadata(services, flags):
    registry, releases, qdrant, embedder, _, _ = services
    with pytest.raises(ValueError, match="explicit existing release"):
        await indexing.build_reference_index(registry, releases, qdrant, embedder, **flags)
    embedder.verify_identity.assert_not_awaited()


async def test_gap_override_cannot_name_an_unselected_source(services):
    registry, releases, qdrant, embedder, _, _ = services
    with pytest.raises(ReferenceUnavailable, match="selected release"):
        await indexing.build_reference_index(
            registry, releases, qdrant, embedder, release_id="rel_" + "b" * 32,
            allowed_extraction_gaps=("a" * 64,),
        )
    embedder.verify_identity.assert_not_awaited()
