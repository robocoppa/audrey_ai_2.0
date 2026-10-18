"""Hermetic source-to-publication-to-search checks with actual PDFs and Qdrant."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import QdrantClient
from test_reference_library import _entry, _pdf, _record, _registry
from test_reference_library import archive as reference_archive_fixture

from audrey.embedding_profile import QWEN_QUERY_PREFIX, EmbeddingProfile
from audrey.kb.qdrant import TEXT_DIM, QdrantKB, SearchScope
from audrey.references import indexing
from audrey.references.embedding import ReferenceEmbedder
from audrey.references.indexing import applicability, build_reference_index, chunks
from audrey.references.irs_archive import ArchiveStore
from audrey.references.library import ReferenceReadError, ReferenceUnavailable, document_id
from audrey.references.releases import ReleaseStore, reference_collection_name
from audrey.references.search import ReferenceSearchService

archive = reference_archive_fixture
_PROFILE = EmbeddingProfile("nomic-embed-text", "a" * 64, TEXT_DIM, "search_document: ", "search_query: ")
_COLLECTION = reference_collection_name("irs-federal", _PROFILE.identity)


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
        model="nomic-embed-text", profile=_PROFILE, verify_identity=AsyncMock(),
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
    assert set(await qdrant.list_collections()) == {_COLLECTION}


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
            collection=_COLLECTION,
            scope=SearchScope(reference_release_id=first["release_id"]),
        )
        == first["chunks"]
    )
    if erase_checkpoint_vectors:
        for _hit_id, payload in await qdrant.scroll_collection(_COLLECTION):
            if payload["release_id"] == failed["release_id"]:
                await qdrant.delete_by_source(
                    payload["source"], collection=_COLLECTION
                )
    qdrant.upsert_text = original_upsert
    resumed = await build_reference_index(
        registry, releases, qdrant, embedder, release_id=failed["release_id"]
    )
    assert resumed["status"] == "published", resumed
    assert resumed["release_id"] == failed["release_id"]
    assert resumed["resumed_documents"] == (0 if erase_checkpoint_vectors else 1)
    assert releases.published("irs-federal")["id"] == failed["release_id"]


async def test_extraction_failure_resumes_only_failed_document(services, monkeypatch):
    registry, releases, qdrant, embedder, _, _ = services
    selected = list(registry.iter_index_documents("irs-federal"))
    failed_id, saved_id = (item["document_id"] for item in selected)
    original_extract = indexing._extract
    calls = 0

    def fail_first_extraction(asset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReferenceReadError("Temporary extraction limit")
        return original_extract(asset)

    monkeypatch.setattr(indexing, "_extract", fail_first_extraction)
    failed = await build_reference_index(registry, releases, qdrant, embedder)
    identifier = failed["release_id"]
    assert failed["status"] == "failed"
    assert failed["failed_documents"] == failed["indexed_documents"] == 1
    assert failed["pending_documents"] == 0
    assert releases.document(identifier, failed_id)["status"] == "failed"
    saved_checkpoint = releases.document(identifier, saved_id)
    assert saved_checkpoint["status"] == "indexed"
    assert releases.published("irs-federal") is None
    saved_points = {
        point.id: point.model_dump()
        for point in qdrant._client.scroll(
            _COLLECTION, limit=100, with_payload=True, with_vectors=True
        )[0]
    }
    assert len(saved_points) == saved_checkpoint["chunk_count"] > 0

    # A fresh vector identifies every write during recovery, so saved vectors
    # must remain byte-for-byte identical rather than being silently regenerated.
    recovery_vector = [0.0, 1.0] + [0.0] * (TEXT_DIM - 2)
    embedder.embed_many = AsyncMock(
        side_effect=lambda texts: [recovery_vector for _ in texts]
    )
    extraction = Mock(wraps=original_extract)
    monkeypatch.setattr(indexing, "_extract", extraction)
    upsert = AsyncMock(wraps=qdrant.upsert_text)
    monkeypatch.setattr(qdrant, "upsert_text", upsert)
    progress = []
    resumed = await build_reference_index(
        registry, releases, qdrant, embedder,
        release_id=identifier, progress=progress.append,
    )
    assert resumed["status"] == "published", resumed
    assert resumed["release_id"] == identifier
    assert resumed["indexed_documents"] == 2
    assert resumed["failed_documents"] == resumed["pending_documents"] == 0
    assert resumed["coverage"]["complete_for_selected_documents"]
    assert resumed["resumed_documents"] == 1
    assert releases.published("irs-federal")["id"] == identifier
    extraction.assert_called_once_with(selected[0]["_asset"])
    embedder.embed_many.assert_awaited_once()
    assert all(
        point.payload["document_id"] == failed_id
        for call in upsert.await_args_list for point in call.args[0]
    )
    assert {
        point.id: point.model_dump()
        for point in qdrant._client.retrieve(
            _COLLECTION, list(saved_points), with_payload=True, with_vectors=True
        )
    } == saved_points
    assert [event for event in progress if event["event"] == "document_resumed"] == [
        {
            "event": "document_resumed", "completed": 2, "selected": 2,
            "document_id": saved_id, "chunks": saved_checkpoint["chunk_count"],
        }
    ]


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


async def test_qwen_reference_space_is_separate_and_old_release_is_retained(services):
    registry, releases, qdrant, nomic, _, _ = services
    original = await build_reference_index(registry, releases, qdrant, nomic)
    model = "qwen3-embedding:4b"
    digest = "b" * 64
    vector = [2.0] + [0.0] * 2559
    ollama = SimpleNamespace(
        tags=AsyncMock(return_value=[{"name": model, "digest": digest}]),
        embed=AsyncMock(side_effect=lambda **kw: [vector for _ in kw["texts"]]),
    )
    profile = EmbeddingProfile(model, digest, 2560, "", QWEN_QUERY_PREFIX)
    qwen = ReferenceEmbedder(ollama, profile)
    replacement = await build_reference_index(registry, releases, qdrant, qwen)
    assert replacement["status"] == "published", replacement
    published = releases.published("irs-federal")
    new_collection = reference_collection_name("irs-federal", profile.identity)
    assert published["collection"] == new_collection
    assert published["provenance"]["embedding_digest"] == digest
    assert published["provenance"]["embedding_dimensions"] == 2560
    assert await qdrant.collection_compatible(new_collection, dim=2560, sparse=True)
    assert set(await qdrant.list_collections()) == {_COLLECTION, new_collection}
    assert await qdrant.count_text(collection=_COLLECTION) == original["chunks"]
    assert not any(
        text.startswith("search_document:")
        for call in ollama.embed.call_args_list for text in call.kwargs["texts"]
    )
    service = ReferenceSearchService(
        registry, releases, qdrant, qwen,
        config={"min_dense_score": .54, "min_term_overlap": None},
    )
    result = await service.search("irs-federal", "gross income filing", tax_year=2025)
    assert result["returned"] > 0
    assert ollama.embed.call_args.kwargs["texts"] == [QWEN_QUERY_PREFIX + "gross income filing"]
    assert all(call.kwargs["truncate"] is False for call in ollama.embed.call_args_list)
    calls = ollama.embed.await_count
    ollama.tags.return_value[0]["digest"] = "c" * 64
    with pytest.raises(ReferenceUnavailable, match=r"identity|digest|unavailable"):
        await service.search("irs-federal", "gross income filing")
    assert ollama.embed.await_count == calls


async def test_incompatible_existing_reference_collection_cannot_replace_publication(services):
    registry, releases, qdrant, nomic, _, _ = services
    original = await build_reference_index(registry, releases, qdrant, nomic)
    profile = EmbeddingProfile("qwen3-embedding:4b", "b" * 64, 2560, "", QWEN_QUERY_PREFIX)
    collection = reference_collection_name("irs-federal", profile.identity)
    await qdrant.ensure_collection(collection, dim=TEXT_DIM)
    embedder = SimpleNamespace(profile=profile, verify_identity=AsyncMock(), embed_many=AsyncMock())
    result = await build_reference_index(registry, releases, qdrant, embedder)
    assert result["status"] == "failed"
    assert "incompatible" in result["error"]
    assert releases.published("irs-federal")["id"] == original["release_id"]
    embedder.embed_many.assert_not_awaited()


async def test_last_batch_model_replacement_prevents_publication(services):
    registry, releases, qdrant, nomic, _, _ = services
    original = await build_reference_index(registry, releases, qdrant, nomic)
    profile = EmbeddingProfile("qwen3-embedding:4b", "b" * 64, 2560, "", QWEN_QUERY_PREFIX)
    client = SimpleNamespace(tags=AsyncMock(return_value=[{"name": profile.model, "digest": profile.digest}]))
    calls = 0
    async def embed(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            client.tags.return_value[0]["digest"] = "c" * 64
        return [[1.0] + [0.0] * 2559 for _ in kwargs["texts"]]
    client.embed = AsyncMock(side_effect=embed)
    report = await build_reference_index(registry, releases, qdrant, ReferenceEmbedder(client, profile))
    assert report["status"] == "failed"
    assert "digest" in report["error"]
    assert releases.published("irs-federal")["id"] == original["release_id"]
