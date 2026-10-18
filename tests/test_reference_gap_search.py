"""Approved extraction gaps stay visible without becoming searchable evidence."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from audrey.kb.qdrant import KBHit
from audrey.references.embedding import EmbeddingProfile
from audrey.references.releases import ReleaseStore
from audrey.references.search import ReferenceSearchService

_TEXT = "Gross income filing requirements depend on filing status and age."
_REASON = "PDF text contains an unsupported character mapping"
_PROFILE = EmbeddingProfile("local-embed", "a" * 64, 1, "", "query: ")


def _document(number):
    source = f"https://www.irs.gov/pub/irs-pdf/p{number}.pdf"
    return {
        "library_id": "irs-federal",
        "document_id": hashlib.sha256(source.encode()).hexdigest(),
        "revision_sha256": hashlib.sha256(f"public-pdf-{number}".encode()).hexdigest(),
        "product_number": f"Publication {number}",
        "title": "IRS website user guide" if number == 5902 else "Income filing guidance",
        "source_url": source,
        "catalog": "current",
        "language": "en",
        "status": "downloaded",
        "original_available": True,
        "catalog_edition_year": 2025,
        "applicable_tax_years": [2025],
        "applicability_basis": "explicit_return_year",
    }


@pytest.fixture
def gapped_search(tmp_path):
    good, gap = _document(17), _document(5902)
    store = ReleaseStore(tmp_path / "release-state")
    release = store.begin(
        "irs-federal",
        "current",
        fingerprint="b" * 64,
        provenance={
            "asof_year": 2026,
            "history_years": 5,
            **_PROFILE.provenance,
            "embedding_identity": _PROFILE.identity,
        },
        coverage={"eligible_total": 2, "subset": False},
        documents=[good, gap],
    )
    store.put_document(
        release["id"], good["document_id"], metadata=good, page_count=1,
        status="indexed", chunks=[{"chunk_index": 0, "page": 1, "offset": 0, "text": _TEXT}],
    )
    store.put_document(
        release["id"], gap["document_id"], metadata=gap, page_count=0,
        status="failed", chunks=[], error=_REASON,
    )
    # The release fixes the selected PDF revision; approval identifies its failed document.
    store.publish(
        release["id"], verified_chunks=1,
        allowed_extraction_gaps=(gap["document_id"],),
    )

    def catalog(_library, *, catalog, product_number, limit):
        assert catalog == "current"
        items = [item for item in (good, gap) if not product_number or item["product_number"] == product_number]
        return {"items": items, "has_more": False}

    registry = SimpleNamespace(
        first_history_year=2022, asof_year=2026, history_years=5,
        catalog=Mock(side_effect=catalog),
        lookup=Mock(side_effect=lambda _library, identifier, **_kwargs: good if identifier == good["document_id"] else gap),
    )
    hit = KBHit(
        score=0.8, source="public-source", kind="text", chunk_idx=0, text=_TEXT,
        payload={
            "library_id": "irs-federal", "release_id": release["id"],
            "catalog": "current", "status": "active", "document_id": good["document_id"],
            "revision_sha256": good["revision_sha256"], "page": 1, "offset": 0,
            "char_end": len(_TEXT), "applicable_tax_years": [2025],
        },
    )
    qdrant = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        collection_compatible=AsyncMock(return_value=True),
        has_sparse=AsyncMock(return_value=True), count_text=AsyncMock(return_value=1),
        search_hybrid=AsyncMock(return_value=([hit], [hit])),
    )
    embedder = SimpleNamespace(profile=_PROFILE, embed_one=AsyncMock(return_value=[1.0]))
    service = ReferenceSearchService(registry, store, qdrant, embedder)
    return service, registry, store, qdrant, embedder, hit, good, gap


def _assert_gap(response, gap):
    coverage = response["coverage"]
    assert coverage["complete_for_selected_documents"] is False
    assert coverage["gap_count"] == 1
    item, = coverage["content_search_gaps"]
    assert item["document_id"] == gap["document_id"]
    assert item["revision_sha256"] == gap["revision_sha256"]
    assert item["product_number"] == "Publication 5902"
    assert item["source_url"] == gap["source_url"]
    assert item["reason"] == _REASON
    assert item["original_available"] is True
    assert "coverage is incomplete" in response["notice"]


async def test_real_published_gap_keeps_genuine_passages_and_discloses_incomplete_coverage(gapped_search):
    service, _, _, _, _, _, good, gap = gapped_search
    response = await service.search("irs-federal", "gross income filing", tax_year=2025)
    assert response["returned"] == 1
    assert response["results"][0]["document_id"] == good["document_id"]
    assert response["results"][0]["text"] == _TEXT
    _assert_gap(response, gap)
    from audrey.pipeline.react import _observed_evidence
    from audrey.tools.dispatch import ToolResult

    sources = _observed_evidence([ToolResult(
        name="reference_search", call_id=None, elapsed_s=0, is_error=False,
        content=json.dumps(response),
    )])
    assert sources == [{
        "title": "Publication 17 · Page 1",
        "url": good["source_url"] + "#page=1", "tool": "reference_search",
    }]


async def test_exact_gap_has_no_embedding_no_fallback_and_no_passage(gapped_search):
    service, registry, _, qdrant, embedder, _, _, gap = gapped_search
    response = await service.search("irs-federal", "website guide", product_number="Publication 5902")
    assert response["returned"] == 0 and response["results"] == []
    assert "requested document is an approved content-search gap" in response["notice"]
    _assert_gap(response, gap)
    embedder.embed_one.assert_not_awaited()
    qdrant.search_hybrid.assert_not_awaited()
    registry.lookup.assert_not_called()
    from audrey.pipeline.react import _observed_evidence
    from audrey.tools.dispatch import ToolResult

    assert _observed_evidence([ToolResult(
        name="reference_search", call_id=None, elapsed_s=0, is_error=False,
        content=json.dumps(response),
    )]) == []


async def test_stale_or_fabricated_gap_vector_never_becomes_evidence(gapped_search):
    service, _, _, _, _, hit, _, gap = gapped_search
    hit.payload.update(document_id=gap["document_id"], revision_sha256=gap["revision_sha256"])
    response = await service.search("irs-federal", "gross income filing")
    assert response["returned"] == 0 and response["results"] == []
    assert "No matching passage" in response["notice"]
    _assert_gap(response, gap)


async def test_unrelated_product_miss_still_discloses_gap_without_broadening(gapped_search):
    service, _, _, qdrant, embedder, _, _, gap = gapped_search
    response = await service.search("irs-federal", "gross income", product_number="Publication 99999")
    assert response["results"] == []
    assert "No permitted document matches" in response["notice"]
    _assert_gap(response, gap)
    embedder.embed_one.assert_not_awaited()
    qdrant.search_hybrid.assert_not_awaited()


async def test_exact_indexed_product_keeps_its_restriction_and_gap_disclosure(gapped_search):
    service, _, _, qdrant, _, _, good, gap = gapped_search
    response = await service.search("irs-federal", "gross income", product_number="Publication 17")
    assert response["returned"] == 1
    assert qdrant.search_hybrid.await_args.kwargs["scope"].reference_document_ids == [good["document_id"]]
    _assert_gap(response, gap)


async def test_bounded_success_retains_gap_notice_and_useful_exact_passage_span(gapped_search):
    from audrey.routes.references import _bounded_search

    service, _, _, _, _, _, good, gap = gapped_search
    response = await service.search("irs-federal", "gross income filing", tax_year=2025)
    response["coverage"]["content_search_gaps"][0]["title"] = (
        "IRS Independent Office of Appeals - Taxpayer Digital Communications User Guide"
    )
    passage = response["results"][0]
    # Stress the response bound with a longer version of the validated public passage;
    # exact publication/source admission is exercised separately above.
    original_text = passage["text"] * 60
    passage.update(text=original_text, char_end=passage["offset"] + len(original_text))
    assert len(json.dumps(response, ensure_ascii=False)) > 3500

    bounded = _bounded_search(response)

    assert len(json.dumps(bounded, ensure_ascii=False)) <= 3500
    assert bounded["notice"] == response["notice"]
    assert bounded["coverage"] == response["coverage"]
    _assert_gap(bounded, gap)
    assert bounded["returned"] == 1
    result, = bounded["results"]
    assert result["document_id"] == good["document_id"]
    assert 100 <= len(result["text"]) < len(original_text)
    assert result["text"] == original_text[:len(result["text"])]
    assert result["char_end"] == result["offset"] + len(result["text"])
    assert result["passage_truncated"] is True
