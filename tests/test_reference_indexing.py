"""Hermetic source-to-publication-to-search checks with actual PDFs and Qdrant."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import QdrantClient
from test_reference_library import _entry, _pdf, _record, _registry
from test_reference_library import archive as reference_archive_fixture

from audrey.kb.qdrant import TEXT_DIM, QdrantKB, SearchScope
from audrey.references.indexing import applicability, build_reference_index, chunks
from audrey.references.irs_archive import ArchiveStore
from audrey.references.library import document_id
from audrey.references.releases import ReleaseStore
from audrey.references.search import ReferenceSearchService

archive = reference_archive_fixture


@pytest.fixture
def services(tmp_path):
    archive_root = tmp_path / "archive"
    entries = [
        _entry("p17.pdf", product="Publication 17"),
        _entry("p18.pdf", product="Publication 18"),
    ]
    text = "For use in preparing 2025 Returns. Gross income filing requirements depend on filing status and age. "
    with ArchiveStore(archive_root) as store:
        _record(
            store, entries, content=_pdf(text, second_page="Gross income threshold depends on age.")
        )
    registry = _registry(archive_root)
    releases = ReleaseStore(tmp_path / "index")
    qdrant = QdrantKB.__new__(QdrantKB)
    qdrant._client = QdrantClient(":memory:")
    qdrant._sparse_cache = {}
    qdrant.text_collection = "private_must_not_be_used"
    qdrant.image_collection = "private_images_must_not_be_used"
    vector = [1.0] + [0.0] * (TEXT_DIM - 1)
    embedder = SimpleNamespace(
        model="nomic-embed-text",
        embed_many=AsyncMock(side_effect=lambda texts: [vector for _ in texts]),
        embed_one=AsyncMock(return_value=vector),
    )
    yield registry, releases, qdrant, embedder, entries, archive_root
    qdrant.close()


async def test_actual_pdf_hybrid_search_has_exact_page_spans_and_year_evidence(services):
    registry, releases, qdrant, embedder, entries, _ = services
    report = await build_reference_index(registry, releases, qdrant, embedder)
    assert report["status"] == "published", report
    assert report["indexed_documents"] == 2 and report["failed_documents"] == 0
    assert report["coverage"]["complete_for_selected_documents"]
    assert not report["coverage"]["subset"]
    assert all(
        text.startswith("search_document: ")
        for call in embedder.embed_many.call_args_list
        for text in call.args[0]
    )
    service = ReferenceSearchService(registry, releases, qdrant, embedder)
    response = await service.search(
        "irs-federal",
        "gross income filing requirements",
        tax_year=2025,
        product_number="Publication 17",
    )
    assert response["returned"] > 0
    for hit in response["results"]:
        assert hit["document_id"] == document_id(entries[0].source_url)
        original = registry.read("irs-federal", hit["document_id"], page=hit["page"], limit=4000)
        assert original["text"][hit["offset"] : hit["char_end"]] == hit["text"]
        assert hit["applicable_tax_years"] == [2025]
    assert (await service.search("irs-federal", "gross income", tax_year=2026))["results"] == []
    assert set(await qdrant.list_collections()) == {"audrey_reference_irs-federal"}


@pytest.mark.parametrize("erase_checkpoint_vectors", [False, True])
async def test_interrupted_replacement_preserves_old_release_and_resumes(
    services, erase_checkpoint_vectors
):
    registry, releases, qdrant, embedder, _, _ = services
    first = await build_reference_index(registry, releases, qdrant, embedder)
    original_upsert = qdrant.upsert_text
    calls = 0

    async def interrupted(points, *, collection):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated outage")
        await original_upsert(points, collection=collection)

    qdrant.upsert_text = interrupted
    failed = await build_reference_index(registry, releases, qdrant, embedder, rebuild=True)
    assert failed["status"] == "failed" and failed["indexed_documents"] == 1
    assert releases.published("irs-federal")["id"] == first["release_id"]
    assert (
        await qdrant.count_text(
            collection="audrey_reference_irs-federal",
            scope=SearchScope(reference_release_id=first["release_id"]),
        )
        == first["chunks"]
    )
    if erase_checkpoint_vectors:
        for _hit_id, payload in await qdrant.scroll_collection("audrey_reference_irs-federal"):
            if payload["release_id"] == failed["release_id"]:
                await qdrant.delete_by_source(
                    payload["source"], collection="audrey_reference_irs-federal"
                )
    qdrant.upsert_text = original_upsert
    resumed = await build_reference_index(
        registry, releases, qdrant, embedder, release_id=failed["release_id"]
    )
    assert resumed["status"] == "published", resumed
    assert resumed["release_id"] == failed["release_id"]
    assert resumed["resumed_documents"] == (0 if erase_checkpoint_vectors else 1)
    assert releases.published("irs-federal")["id"] == failed["release_id"]


async def test_remote_count_mismatch_cannot_publish(services):
    registry, releases, qdrant, embedder, _, _ = services
    qdrant.count_text = AsyncMock(return_value=999)
    report = await build_reference_index(registry, releases, qdrant, embedder)
    assert report["status"] == "failed"
    assert releases.published("irs-federal") is None
    assert report["verified_chunks"] == 999
    assert "chunk count" in report["error"]


async def test_checksum_failure_is_explicit_gap_and_blocks_publication(services):
    registry, releases, qdrant, embedder, _, archive = services
    candidate = next(registry.iter_index_documents("irs-federal"))
    path = archive / candidate["_asset"]["relative_path"]
    body = bytearray(path.read_bytes())
    body[-20] ^= 1
    path.write_bytes(body)
    report = await build_reference_index(registry, releases, qdrant, embedder)
    assert report["status"] == "failed" and report["failed_documents"] == 1
    assert releases.published("irs-federal") is None
    assert "_asset" not in json.dumps(list(releases.documents(report["release_id"])))


async def test_subset_and_blank_text_coverage_are_disclosed(services):
    registry, releases, qdrant, embedder, entries, archive = services
    with ArchiveStore(archive) as store:
        _record(store, entries, content=_pdf("", second_page=""))
    report = await build_reference_index(
        registry, releases, qdrant, embedder, product_numbers=["Publication 17"]
    )
    assert report["status"] == "published", report
    assert report["no_text_documents"] == 1 and report["chunks"] == 0
    assert report["coverage"]["subset"] and report["coverage"]["eligible_total"] == 2
    embedder.embed_many.assert_not_awaited()


def test_enumeration_applies_the_same_recent_edition_window(archive):
    root, current, old_active, _, _, _ = archive
    registry = _registry(root)
    assert {item["source_url"] for item in registry.iter_index_documents("irs-federal")} == {
        current.source_url,
        old_active.source_url,
    }
    assert {
        item["catalog_edition_year"]
        for item in registry.iter_index_documents("irs-federal", catalog="prior")
    } == {2022, 2026}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("For use in preparing 2025 Returns", [2025]),
        ("For use in preparing\n2025 tax Returns", [2025]),
        ("Rev. 2025. Posted in 2026. 2025", None),
        ("For use in preparing 2024 Returns; For use in preparing 2025 Returns", None),
    ],
)
def test_only_explicit_unambiguous_return_year_is_verified(text, expected):
    assert applicability(text, "a" * 64)["applicable_tax_years"] == expected


def test_chunk_spans_preserve_whitespace_and_page_boundaries():
    pages = [{"page": 1, "text": " \nGross income " * 700}, {"page": 2, "text": "Other page"}]
    result = list(chunks(pages))
    for item in result:
        page = pages[item["page"] - 1]["text"]
        assert page[item["offset"] : item["offset"] + len(item["text"])] == item["text"]
    assert result[-1]["page"] == 2


async def test_unchanged_published_release_is_reused_without_embedding(services):
    registry, releases, qdrant, embedder, _, _ = services
    first = await build_reference_index(registry, releases, qdrant, embedder)
    embedder.embed_many.reset_mock()
    repeated = await build_reference_index(registry, releases, qdrant, embedder)
    assert repeated["status"] == "published" and repeated["unchanged"]
    assert repeated["release_id"] == first["release_id"]
    embedder.embed_many.assert_not_awaited()
