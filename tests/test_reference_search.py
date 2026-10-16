"""Published reference isolation, source validation, year evidence and outages."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from audrey.kb.qdrant import KBHit
from audrey.references.embedding import EmbeddingProfile
from audrey.references.library import ReferenceNotFound, ReferenceQueryError, ReferenceUnavailable
from audrey.references.releases import reference_collection_name
from audrey.references.search import ReferenceSearchService

_DOC = "a" * 64
_SHA = "b" * 64
_TEXT = "Gross income filing requirements depend on filing status and age."
_PROFILE = EmbeddingProfile("nomic-embed-text", "a" * 64, 1, "search_document: ", "search_query: ")
_COLLECTION = reference_collection_name("irs-federal", _PROFILE.identity)


@pytest.fixture
def services():
    metadata = {
        "document_id": _DOC,
        "revision_sha256": _SHA,
        "product_number": "Publication 17",
        "title": "Your Federal Income Tax",
        "source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
        "catalog_edition_year": 2025,
        "applicable_tax_years": [2025],
        "applicability_basis": "explicit_return_year",
    }
    registry = SimpleNamespace(
        first_history_year=2022,
        asof_year=2026,
        history_years=5,
        catalog=Mock(return_value={"items": [metadata], "has_more": False}),
        lookup=Mock(return_value={**metadata, "status": "downloaded", "original_available": True}),
    )
    release = {
        "id": "release-test",
        "library_id": "irs-federal",
        "catalog": "current",
        "collection": _COLLECTION,
        "status": "published",
        "provenance": {
            "asof_year": 2026,
            "history_years": 5,
            **_PROFILE.provenance,
            "embedding_identity": _PROFILE.identity,
        },
        "coverage": {"eligible_total": 2407, "subset": True},
        "expected_documents": 1,
        "indexed_documents": 1,
        "no_text_documents": 0,
        "chunk_count": 1,
    }
    chunk = {"document_id": _DOC, "chunk_index": 0, "page": 4, "offset": 120, "text": _TEXT}
    store = SimpleNamespace(
        published=Mock(return_value=release),
        document=Mock(return_value={"status": "indexed", "metadata": metadata}),
        chunk=Mock(return_value=chunk),
    )
    payload = {
        "library_id": "irs-federal",
        "release_id": "release-test",
        "catalog": "current",
        "status": "active",
        "document_id": _DOC,
        "revision_sha256": _SHA,
        "page": 4,
        "offset": 120,
        "char_end": 120 + len(_TEXT),
        "applicable_tax_years": [2025],
    }
    hit = KBHit(
        score=0.8, source="release-doc", kind="text", chunk_idx=0, text=_TEXT, payload=payload
    )
    qdrant = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        collection_compatible=AsyncMock(return_value=True),
        has_sparse=AsyncMock(return_value=True),
        count_text=AsyncMock(return_value=1),
        search_hybrid=AsyncMock(return_value=([hit], [hit])),
    )
    embedder = SimpleNamespace(model="nomic-embed-text", profile=_PROFILE, embed_one=AsyncMock(return_value=[1.0]))
    service = ReferenceSearchService(registry, store, qdrant, embedder)
    return service, registry, store, qdrant, embedder, hit, metadata


async def test_both_retrievers_receive_one_published_scope_and_nomic_query_prefix(services):
    service, _, _, qdrant, embedder, _, _ = services
    response = await service.search("irs-federal", "gross income filing", tax_year=2025)
    embedder.embed_one.assert_awaited_once_with("search_query: gross income filing")
    args, kwargs = qdrant.search_hybrid.await_args
    assert args == ([1.0], "gross income filing")
    scope = kwargs["scope"]
    assert scope.reference_library_id == "irs-federal"
    assert scope.reference_release_id == "release-test"
    assert scope.reference_catalog == "current"
    assert scope.reference_tax_year == 2025
    assert scope.reference_status == "active"
    assert kwargs["collection"] == _COLLECTION
    assert response["returned"] == 1
    assert response["results"][0]["page"] == 4
    assert response["results"][0]["offset"] == 120
    assert response["coverage"]["subset"]
    assert response["coverage"]["selected_documents"] == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("library_id", "foreign"),
        ("release_id", "unpublished"),
        ("catalog", "prior"),
        ("status", "building"),
        ("revision_sha256", "c" * 64),
        ("document_id", "old-id"),
        ("page", 99),
        ("offset", 99),
        ("char_end", 999),
        ("applicable_tax_years", [1976]),
    ],
)
async def test_out_of_scope_or_tampered_vector_payload_is_never_returned(services, key, value):
    service, _, _, _, _, hit, _ = services
    hit.payload[key] = value
    response = await service.search("irs-federal", "gross income")
    assert response["results"] == []


async def test_vector_text_must_equal_the_immutable_release_chunk(services):
    service, _, _, _, _, hit, _ = services
    hit.text = "injected text not in the indexed PDF"
    assert (await service.search("irs-federal", "gross income"))["results"] == []


@pytest.mark.parametrize("change", ["removed", "revised", "missing", "failed"])
async def test_catalog_liveness_blocks_removed_revised_and_failed_sources(services, change):
    service, registry, _, _, _, _, _ = services
    if change == "removed":
        registry.lookup.side_effect = ReferenceNotFound("Removed edition")
    else:
        current = registry.lookup.return_value
        if change == "revised":
            current["revision_sha256"] = "c" * 64
        elif change == "missing":
            current["original_available"] = False
        else:
            current["status"] = "failed"
    assert (await service.search("irs-federal", "gross income"))["results"] == []


async def test_year_filter_excludes_unknown_applicability_even_if_vector_claims_year(services):
    service, _, _, _, _, _, metadata = services
    metadata["applicable_tax_years"] = None
    response = await service.search("irs-federal", "gross income", tax_year=2025)
    assert response["results"] == []
    assert "unknown-year sources are excluded" in response["notice"]


@pytest.mark.parametrize("year", [1976, 2021, 2027, True])
async def test_year_outside_policy_is_refused_before_any_embedding(services, year):
    service, registry, _, _, embedder, _, _ = services
    with pytest.raises(ReferenceQueryError):
        await service.search("irs-federal", "income", tax_year=year)
    registry.catalog.assert_not_called()
    embedder.embed_one.assert_not_awaited()


async def test_exact_product_miss_returns_no_results_without_embedding(services):
    service, registry, _, _, embedder, _, _ = services
    registry.catalog.return_value = {"items": [], "has_more": False}
    response = await service.search("irs-federal", "income", product_number="Form 940")
    assert response["results"] == []
    embedder.embed_one.assert_not_awaited()


@pytest.mark.parametrize("failure", ["unpublished", "collection", "sparse", "embedding", "search"])
async def test_unavailable_dependency_never_falls_back_to_global_or_private_kb(services, failure):
    service, _, store, qdrant, embedder, _, _ = services
    if failure == "unpublished":
        store.published.return_value = None
    elif failure == "collection":
        qdrant.collection_exists.return_value = False
    elif failure == "sparse":
        qdrant.has_sparse.return_value = False
    elif failure == "embedding":
        embedder.embed_one.side_effect = RuntimeError("offline")
    else:
        qdrant.search_hybrid.side_effect = RuntimeError("offline")
    with pytest.raises(ReferenceUnavailable):
        await service.search("irs-federal", "income")
    assert qdrant.search_hybrid.await_count <= 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("history_years", 10),
        ("asof_year", 2025),
        ("embedding_model", "different"),
        ("embedding_digest", "b" * 64),
        ("embedding_dimensions", 2560),
        ("embedding_identity", "b" * 64),
        ("embedding_truncate", True),
        ("query_prefix", ""),
    ],
)
async def test_changed_release_policy_is_refused_before_embedding(services, key, value):
    service, _, store, _, embedder, _, _ = services
    store.published.return_value["provenance"][key] = value
    with pytest.raises(ReferenceUnavailable):
        await service.search("irs-federal", "income")
    embedder.embed_one.assert_not_awaited()


async def test_weak_unrelated_candidates_do_not_become_tax_evidence(services):
    service, _, _, _, _, hit, _ = services
    hit.score = 0.1
    response = await service.search("irs-federal", "volcanic basalt magma")
    assert response["results"] == []


async def test_explicit_recent_history_is_a_separate_scope(services):
    service, registry, store, _, _, hit, _ = services
    store.published.return_value["catalog"] = "prior"
    hit.payload["catalog"] = "prior"
    response = await service.search("irs-federal", "income", catalog="prior")
    assert response["catalog"] == "prior"
    assert registry.lookup.call_args.kwargs["catalog"] == "prior"


async def test_lost_published_vectors_are_reported_unavailable_before_embedding(services):
    service, _, _, qdrant, embedder, _, _ = services
    qdrant.count_text.return_value = 0
    with pytest.raises(ReferenceUnavailable, match="incomplete"):
        await service.search("irs-federal", "income")
    embedder.embed_one.assert_not_awaited()


async def test_lexical_admission_can_be_disabled_without_disabling_hybrid_ranking(services):
    service, _, _, qdrant, _, hit, _ = services
    service.min_term_overlap = None
    hit.score = .1
    response = await service.search("irs-federal", "Gross income filing requirements")
    assert response["results"] == []
    qdrant.search_hybrid.assert_awaited_once()


async def test_changed_collection_schema_is_refused_before_embedding(services):
    service, _, _, qdrant, embedder, _, _ = services
    qdrant.collection_compatible.return_value = False
    with pytest.raises(ReferenceUnavailable, match="schema"):
        await service.search("irs-federal", "income")
    embedder.embed_one.assert_not_awaited()


async def test_collection_identity_cannot_be_redirected_to_another_space(services):
    service, _, releases, _, embedder, _, _ = services
    releases.published.return_value["collection"] = "other_collection"
    with pytest.raises(ReferenceUnavailable, match="policy"):
        await service.search("irs-federal", "income")
    embedder.embed_one.assert_not_awaited()
