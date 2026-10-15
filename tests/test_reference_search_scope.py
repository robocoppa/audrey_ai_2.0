"""Reference release filters apply identically to dense, lexical and exact counts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from audrey.kb import qdrant as qdrant_module
from audrey.kb.qdrant import TEXT_DIM, QdrantKB, SearchScope, _as_filter, build_text_point

_COLLECTION = "reference_irs_federal"
_DOCUMENT = "a" * 64
_SCOPE = {
    "reference_library_id": "irs-federal",
    "reference_release_id": "release-current",
    "reference_catalog": "current",
    "reference_document_ids": [_DOCUMENT],
    "reference_tax_year": 2025,
    "reference_status": "indexed",
}


def _vector():
    return [1.0, *([0.0] * (TEXT_DIM - 1))]


@pytest.fixture
def references(monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_: client)
    kb = QdrantKB(host="unused", port=0)
    kb._ensure_named_sync(_COLLECTION, TEXT_DIM)
    base = {
        "library_id": "irs-federal",
        "release_id": "release-current",
        "catalog": "current",
        "document_id": _DOCUMENT,
        "applicable_tax_years": [2025],
        "status": "indexed",
    }
    payloads = [
        base,
        base,
        {**base, "library_id": "other-library"},
        {**base, "release_id": "older-release"},
        {**base, "catalog": "prior"},
        {**base, "document_id": "b" * 64},
        {**base, "applicable_tax_years": [2024]},
        {**base, "applicable_tax_years": []},
        {**base, "status": "failed"},
    ]
    points = [
        build_text_point(
            source=f"reference:{payload['library_id']}:{payload['release_id']}:{index}",
            chunk_idx=0,
            text="shared tax deduction evidence",
            vector=_vector(),
            mtime=0,
            extra=payload,
            sparse=True,
        )
        for index, payload in enumerate(payloads)
    ]
    client.upsert(_COLLECTION, points=points, wait=True)
    yield kb
    kb.close()


def test_empty_reference_document_scope_is_refused():
    with pytest.raises(ValueError, match="reference_document_ids=\\[\\]"):
        SearchScope(reference_library_id="irs-federal", reference_document_ids=[])


def test_invalid_reference_catalog_is_refused():
    with pytest.raises(ValueError, match="Reference catalog"):
        SearchScope(reference_catalog="all")


@pytest.mark.parametrize("field", list(_SCOPE))
def test_each_reference_field_is_a_real_scope(field):
    scope = SearchScope(**{field: _SCOPE[field]})
    assert not scope.is_empty()
    assert _as_filter(scope) is not None


def test_reference_scope_composes_with_existing_private_and_tombstone_clauses():
    scope = SearchScope(
        **_SCOPE,
        file_ids=["kept-file"],
        artifact="transcript",
        user="alice@example.com",
        excluded_file_ids=["deleted-file"],
    )
    result = _as_filter(scope).model_dump(exclude_none=True)
    must = {item["key"]: item["match"] for item in result["must"]}
    assert must == {
        "file_id": {"any": ["kept-file"]},
        "artifact": {"value": "transcript"},
        "user": {"value": "alice@example.com"},
        "library_id": {"value": "irs-federal"},
        "release_id": {"value": "release-current"},
        "catalog": {"value": "current"},
        "document_id": {"any": [_DOCUMENT]},
        "applicable_tax_years": {"value": 2025},
        "status": {"value": "indexed"},
    }
    assert result["must_not"] == [
        {
            "key": "file_id",
            "match": {"any": ["deleted-file"]},
        }
    ]


async def test_dense_and_lexical_search_each_enforce_the_full_reference_conjunction(references):
    dense, lexical = await references.search_hybrid(
        _vector(),
        "shared tax deduction evidence",
        collection=_COLLECTION,
        top_k=20,
        scope=SearchScope(**_SCOPE),
    )
    for hits in (dense, lexical):
        assert len(hits) == 2
        assert {hit.payload["library_id"] for hit in hits} == {"irs-federal"}
        assert {hit.payload["release_id"] for hit in hits} == {"release-current"}
        assert {hit.payload["catalog"] for hit in hits} == {"current"}
        assert {hit.payload["document_id"] for hit in hits} == {_DOCUMENT}
        assert all(hit.payload["applicable_tax_years"] == [2025] for hit in hits)
        assert {hit.payload["status"] for hit in hits} == {"indexed"}


async def test_hybrid_threads_identical_reference_filter_into_both_queries(monkeypatch):
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_: client)
    kb = QdrantKB(host="unused", port=0)
    kb._sparse_cache[_COLLECTION] = True

    await kb.search_hybrid(
        _vector(),
        "tax deduction",
        collection=_COLLECTION,
        scope=SearchScope(**_SCOPE),
    )

    calls = client.query_points.call_args_list
    assert len(calls) == 2
    dense = next(call for call in calls if "using" not in call.kwargs)
    lexical = next(call for call in calls if "using" in call.kwargs)
    assert dense.kwargs["query_filter"] == lexical.kwargs["query_filter"]
    assert dense.kwargs["collection_name"] == lexical.kwargs["collection_name"] == _COLLECTION
    assert {clause.key for clause in dense.kwargs["query_filter"].must} == {
        "library_id",
        "release_id",
        "catalog",
        "document_id",
        "applicable_tax_years",
        "status",
    }


async def test_exact_count_verifies_selected_release_instead_of_collection_total(references):
    assert await references.count_text(collection=_COLLECTION, scope=SearchScope(**_SCOPE)) == 2
    assert await references.count_text(collection=_COLLECTION) == 9
    assert (
        await references.count_text(
            collection=_COLLECTION,
            scope=SearchScope(reference_library_id="irs-federal", reference_release_id="unknown"),
        )
        == 0
    )


async def test_count_failures_surface_without_unscoped_retry(monkeypatch):
    client = MagicMock()
    client.count.side_effect = ConnectionError("Qdrant unavailable")
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_: client)
    kb = QdrantKB(host="unused", port=0)

    with pytest.raises(ConnectionError, match="Qdrant unavailable"):
        await kb.count_text(collection=_COLLECTION, scope=SearchScope(**_SCOPE))

    client.count.assert_called_once()
    assert client.count.call_args.kwargs["exact"] is True
    assert client.count.call_args.kwargs["count_filter"] is not None
