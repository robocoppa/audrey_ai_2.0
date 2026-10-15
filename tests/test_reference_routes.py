"""Service authentication, bounded results and fail-closed reference routes."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from audrey.references.library import (
    ReferenceNotFound,
    ReferenceQueryError,
    ReferenceReadError,
    ReferenceUnavailable,
)
from audrey.routes import references

_SECRET = "reference-test-service-secret"  # noqa: S105
_HEADERS = {"X-Audrey-Service-Token": _SECRET}
_ID = "a" * 64


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(references.router)
    app.state.cfg = SimpleNamespace(env=SimpleNamespace(kb_service_token=_SECRET))
    app.state.reference_libraries = Mock()
    app.state.reference_libraries.catalog.return_value = {
        "library_id": "irs-federal", "items": [], "offset": 0,
        "returned": 0, "has_more": False, "next_offset": None,
    }
    app.state.reference_libraries.read.return_value = {
        "source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
        "text": "Retained source text", "offset": 0, "next_offset": None,
        "has_more": False, "page": 2, "total_chars": 20,
        "text_layer_available": True,
    }
    return app


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://reference-test",
    ) as client:
        yield client


@pytest.mark.parametrize("path,body", [
    ("catalog", {}), ("read", {"document_id": _ID}),
])
async def test_model_routes_require_service_authentication(client, app, path, body):
    for headers in ({}, {"X-Audrey-Service-Token": "wrong"},
                    {"Authorization": "Bearer signed-in-user"}):
        response = await client.post(f"/v1/references/{path}", json=body, headers=headers)
        assert response.status_code == 401
    assert not app.state.reference_libraries.catalog.called
    assert not app.state.reference_libraries.read.called


@pytest.mark.parametrize("field,value", [
    ("root", "/data/uploads"), ("path", "/etc/passwd"),
    ("user", "someone"), ("collection", "kb_text"),
    ("history_years", 100), ("asof_year", 1900),
])
async def test_model_cannot_choose_paths_owners_or_history_policy(client, field, value):
    for path, body in [("catalog", {}), ("read", {"document_id": _ID})]:
        response = await client.post(
            f"/v1/references/{path}", headers=_HEADERS, json={**body, field: value},
        )
        assert response.status_code == 422


async def test_lookup_default_and_explicit_filters_reach_registry(client, app):
    response = await client.post("/v1/references/catalog", headers=_HEADERS, json={})
    assert response.status_code == 200
    app.state.reference_libraries.catalog.assert_called_with(
        "irs-federal", product_number=None, query=None, catalog="current",
        offset=0, limit=5, year=None,
    )
    response = await client.post("/v1/references/catalog", headers=_HEADERS, json={
        "catalog": "prior", "product_number": "Form 1040", "tax_year": 2025,
    })
    assert response.status_code == 200
    assert app.state.reference_libraries.catalog.call_args.kwargs["year"] == 2025


async def test_reader_passes_stable_ids_page_catalog_and_exact_revision(client, app):
    response = await client.post("/v1/references/read", headers=_HEADERS, json={
        "document_id": _ID, "catalog": "prior", "page": 2,
        "revision_sha256": "b" * 64,
    })
    assert response.status_code == 200
    app.state.reference_libraries.read.assert_called_once_with(
        "irs-federal", _ID, catalog="prior", page=2, offset=0, limit=1200,
        revision_sha256="b" * 64,
    )


@pytest.mark.parametrize("exception,status", [
    (ReferenceNotFound("Unknown document"), 404),
    (ReferenceUnavailable("Reference archive is unavailable"), 503),
    (ReferenceQueryError("Tax-year applicability is unknown"), 422),
    (ReferenceReadError("PDF resource limit reached"), 422),
])
async def test_library_failures_do_not_widen_to_other_sources(client, app, exception, status):
    app.state.reference_libraries.catalog.side_effect = exception
    response = await client.post("/v1/references/catalog", headers=_HEADERS, json={})
    assert response.status_code == status
    assert not app.state.reference_libraries.read.called


def test_catalog_budget_keeps_valid_paging_instead_of_clipping_json():
    source = {
        "library_id": "irs-federal", "offset": 10,
        "returned": 5, "has_more": False, "next_offset": None,
        "items": [{"document_id": str(n), "title": "x" * 1000} for n in range(5)],
    }
    result = references._bounded_catalog(source)
    assert len(json.dumps(result, ensure_ascii=False)) <= references._RESULT_CHARS
    assert 0 < result["returned"] < 5
    assert result["next_offset"] == 10 + result["returned"]
    assert result["has_more"]
    assert len(source["items"]) == 5


def test_text_budget_reports_the_actual_next_character_offset():
    source = {
        "text": "\\" * 4000, "offset": 1200, "next_offset": None,
        "has_more": False, "total_chars": 5200,
        "text_layer_available": True, "source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
    }
    result = references._bounded_read(source)
    assert len(json.dumps(result, ensure_ascii=False)) <= references._RESULT_CHARS
    assert result["text"] == source["text"][:len(result["text"])]
    assert result["next_offset"] == 1200 + len(result["text"])
    assert result["has_more"]
    assert result["total_chars"] == 5200


async def test_uninitialized_service_returns_unavailable_without_other_kb(client, app):
    del app.state.reference_libraries
    response = await client.post("/v1/references/catalog", headers=_HEADERS, json={})
    assert response.status_code == 503


def test_public_page_sources_preserve_page_citations_and_exclude_catalog_only():
    from audrey.pipeline.react import _observed_evidence
    from audrey.pipeline.run_observations import _observable_arguments
    from audrey.tools.dispatch import ToolResult

    def result(name, **body):
        return ToolResult(
            name=name, call_id=None, content=json.dumps(body), is_error=False, elapsed_s=0,
        )

    sources = _observed_evidence([
        result("reference_lookup", items=[{
            "source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "title": "Catalog only",
        }]),
        result("read_reference", source_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",
               product_number="Publication 17", page=2, text="Public page text"),
        result("read_reference", source_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",
               product_number="Publication 17", page=3, text="More public page text"),
        result("read_reference", source_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",
               product_number="Publication 17", page=4, text=""),
    ])
    assert [source["url"] for source in sources] == [
        "https://www.irs.gov/pub/irs-pdf/p17.pdf#page=2",
        "https://www.irs.gov/pub/irs-pdf/p17.pdf#page=3",
    ]
    assert "Public page text" not in json.dumps(sources)
    assert _observable_arguments("read_reference", {
        "document_id": _ID, "page": 2, "path": "/private/path",
    }) == {"document_id": _ID, "page": 2, "path": "[redacted]"}


async def test_search_requires_service_auth_and_refuses_model_scope_overrides(client, app):
    app.state.reference_search = SimpleNamespace(search=AsyncMock())
    for headers in ({}, {"Authorization": "Bearer browser-user"}):
        response = await client.post("/v1/references/search", json={"query": "filing"}, headers=headers)
        assert response.status_code == 401
    for key in ("collection", "release_id", "root", "user", "asof_year"):
        response = await client.post("/v1/references/search", headers=_HEADERS,
                                     json={"query": "filing", key: "untrusted"})
        assert response.status_code == 422
    app.state.reference_search.search.assert_not_awaited()


async def test_search_preserves_valid_bounded_json_and_exact_character_spans(client, app):
    result = {
        "release_id": "public-release", "coverage": {"subset": True, "eligible_total": 2407},
        "returned": 5, "results": [{
            "document_id": _ID, "revision_sha256": "b" * 64,
            "source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "product_number": "Publication 17",
            "page": page, "offset": 40, "text": "Evidence text " * 80,
            "char_end": 40 + len("Evidence text " * 80), "passage_truncated": False,
        } for page in range(1, 6)],
    }
    app.state.reference_search = SimpleNamespace(search=AsyncMock(return_value=result))
    response = await client.post("/v1/references/search", headers=_HEADERS,
                                 json={"query": "filing requirements", "tax_year": 2025})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(json.dumps(body, ensure_ascii=False)) <= references._RESULT_CHARS
    assert body["coverage"] == result["coverage"]
    assert body["returned"] == len(body["results"]) > 0
    assert body["result_limit_applied"]
    for item in body["results"]:
        assert item["char_end"] == item["offset"] + len(item["text"])
        assert result["results"][0]["text"].startswith(item["text"])
    assert app.state.reference_search.search.await_args.kwargs["tax_year"] == 2025


async def test_search_without_release_state_returns_unavailable(client, app):
    response = await client.post("/v1/references/search", headers=_HEADERS, json={"query": "filing"})
    assert response.status_code == 503
    app.state.reference_search = SimpleNamespace(search=AsyncMock(side_effect=ReferenceUnavailable("Not published")))
    response = await client.post("/v1/references/search", headers=_HEADERS, json={"query": "filing"})
    assert response.status_code == 503


def test_search_snippets_add_page_sources_without_exposing_query_or_passage_in_events():
    from audrey.pipeline.react import _observed_evidence
    from audrey.pipeline.run_observations import _observable_arguments
    from audrey.tools.dispatch import ToolResult

    result = ToolResult(name="reference_search", call_id=None, elapsed_s=0, is_error=False,
                        content=json.dumps({"results": [
                            {"source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "page": 9,
                             "product_number": "Publication 17", "text": "Quoted public evidence"},
                            {"source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "page": 9,
                             "text": "Duplicate page"},
                            {"source_url": "https://www.irs.gov/pub/irs-pdf/p17.pdf", "page": 10,
                             "text": ""},
                        ]}))
    sources = _observed_evidence([result])
    assert sources == [{"title": "Publication 17 · Page 9",
                        "url": "https://www.irs.gov/pub/irs-pdf/p17.pdf#page=9", "tool": "reference_search"}]
    assert _observable_arguments("reference_search", {"query": "personal filing situation", "tax_year": 2025}) == {
        "query": "[redacted]", "tax_year": 2025,
    }
