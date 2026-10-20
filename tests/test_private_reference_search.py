"""Private callbacks use the published local index and verified local embeddings."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient
from test_reference_library import _entry, _pdf, _record, _registry

from audrey.config import Config, EnvOverrides
from audrey.embedding_profile import EmbeddingProfile
from audrey.identity import Principal
from audrey.kb.qdrant import QdrantKB
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.private_processing import PrivateProcessingRuntime
from audrey.processing_policy import ProcessingPolicy
from audrey.references.indexing import build_reference_index
from audrey.references.irs_archive import ArchiveStore
from audrey.references.releases import ReleaseStore
from audrey.references.search import ReferenceSearchService
from audrey.routes import references
from audrey.routes.inflight import UserInflightRegistry
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.tools.private_dispatch import private_tool_scope
from audrey.tools.private_handlers import make_private_tool_handlers

_MODEL = "local-embed:latest"
_PROFILE = EmbeddingProfile(_MODEL, "a" * 64, 8, "", "query: ")
_SERVER = "http://custom-tools:8001"
_TEXT = "For use in preparing 2025 Returns. Gross income filing requirements depend on filing status and age."


@pytest.fixture
async def private_reference(tmp_path, request):
    root = tmp_path / "irs"
    with ArchiveStore(root) as archive:
        _record(archive, [_entry("p17.pdf", product="Publication 17")], content=_pdf(getattr(request, "param", _TEXT)))
    registry = _registry(root)
    releases = ReleaseStore(tmp_path / "releases")
    qdrant = QdrantKB.__new__(QdrantKB)
    qdrant._client = QdrantClient(":memory:")
    qdrant._sparse_cache = {}
    qdrant.text_collection = "private_files_must_not_be_searched"
    qdrant.image_collection = "private_images_must_not_be_searched"
    vector = [1.0] + [0.0] * 7
    staging_embedder = SimpleNamespace(
        model=_MODEL, profile=_PROFILE, verify_identity=AsyncMock(),
        embed_many=AsyncMock(side_effect=lambda texts: [vector for _ in texts]),
    )
    report = await build_reference_index(registry, releases, qdrant, staging_embedder)
    assert report["status"] == "published", report
    cfg = Config({
        "router": {"model": _MODEL}, "title_generation": {"enabled": False},
        "model_registry": {"general": [{"name": _MODEL, "location": "local"}]},
        "text_embedding": {
            "model": _MODEL, "digest": _PROFILE.digest, "dimensions": _PROFILE.dimensions,
            "document_prefix": _PROFILE.document_prefix, "query_prefix": _PROFILE.query_prefix,
        },
    }, EnvOverrides(_env_file=None))
    tools = ToolRegistry({name: ToolSpec(
        name=name, description="Read the owned public reference library",
        parameters=schema.model_json_schema(), server_url=_SERVER, path=f"/{name}",
    ) for name, schema in (
        ("reference_search", references.SearchRequest),
        ("reference_lookup", references.CatalogRequest),
        ("read_reference", references.ReadRequest),
    )})
    app = SimpleNamespace(state=SimpleNamespace(
        cfg=cfg, registry=ModelRegistry(cfg), health=HealthTracker(),
        gate=FairLocalGate(concurrency=1), inflight=UserInflightRegistry(max_inflight_per_user=3),
        tools=tools, reference_libraries=registry,
        reference_search=ReferenceSearchService(registry, releases, qdrant, staging_embedder),
    ))
    policy = ProcessingPolicy({_MODEL: _PROFILE.digest}, "http://ollama:11434",
                              allowed_tools=frozenset(tools.names()),
                              tool_servers=frozenset({_SERVER}))
    owner = Principal("usr_test", "owner", "test", "subject", "owner@example.com",
                      "Owner", "user", "active", "test")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.host == "ollama"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": _MODEL, "model": _MODEL, "digest": _PROFILE.digest},
            ]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={
                "details": {"format": "gguf"}, "capabilities": ["embedding"],
                "model_info": {"general.architecture": "local", "general.parameter_count": 100},
            })
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["model"] == _MODEL and payload["truncate"] is False
        assert payload["input"] == [_PROFILE.query("gross income filing requirements")]
        return httpx.Response(200, json={"embeddings": [vector]})

    runtime = PrivateProcessingRuntime(policy, ollama_transport=httpx.MockTransport(respond),
                                       tool_transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected sidecar request")))
    try:
        yield runtime, app, owner, requests
    finally:
        qdrant.close()


@pytest.mark.parametrize("catalog", ["current", "prior"])
async def test_private_search_reads_real_published_passages_without_sidecar(private_reference, catalog):
    runtime, app, owner, requests = private_reference
    async with runtime.scope(app) as execution:
        assert execution.state.reference_search.embedder is not app.state.reference_search.embedder
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": "reference_search", "arguments": {
                    "query": "gross income filing requirements", "catalog": catalog,
                    "tax_year": 2025,
                }},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    content = json.loads(result.content)
    embedding_requests = [request for request in requests if request.url.path == "/api/embed"]
    if catalog == "prior":
        # Only current was published. No automatic scope widening or sidecar fallback.
        assert result.is_error and embedding_requests == []
    else:
        assert not result.is_error, result.content
        assert len(embedding_requests) == 1
        assert content["returned"] >= 1
        passage = content["results"][0]
        assert passage["page"] == 1 and _TEXT in passage["text"]
        assert passage["source_url"] == "https://www.irs.gov/pub/irs-pdf/p17.pdf"
        assert content["catalog"] == "current" and content["tax_year"] == 2025
        assert passage["applicable_tax_years"] == [2025]


@pytest.mark.parametrize("requested", [6, 8, 1_000_000])
async def test_private_search_caps_oversized_count_and_keeps_actual_evidence(private_reference, requested):
    runtime, app, owner, requests = private_reference
    arguments = {"query": "gross income filing requirements", "top_k": requested, "tax_year": 2025}
    async with runtime.scope(app) as execution:
        service = execution.state.reference_search
        service.search = AsyncMock(wraps=service.search)
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": "reference_search", "arguments": arguments},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert 0 < body["returned"] <= 5
    assert body["results"][0]["source_url"] == "https://www.irs.gov/pub/irs-pdf/p17.pdf"
    assert _TEXT in body["results"][0]["text"]
    assert body["tax_year"] == 2025 and body["catalog"] == "current"
    assert "capped at 5 passages" in body["notice"]
    assert len(result.content) <= references._RESULT_CHARS
    assert service.search.await_args.kwargs["top_k"] == 5
    assert arguments["top_k"] == requested
    assert len([request for request in requests if request.url.path == "/api/embed"]) == 1


@pytest.mark.parametrize("requested", [21, 100, 1_000_000])
async def test_private_catalog_caps_only_result_count_without_embedding(private_reference, requested):
    runtime, app, owner, requests = private_reference
    registry = app.state.reference_libraries
    registry.catalog = Mock(wraps=registry.catalog)
    arguments = {"query": "Individual", "limit": requested, "catalog": "current"}
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": "reference_lookup", "arguments": arguments},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert 0 < body["returned"] <= 20
    assert body["items"][0]["product_number"] == "Publication 17"
    assert body["items"][0]["source_url"] == "https://www.irs.gov/pub/irs-pdf/p17.pdf"
    assert body["catalog"] == "current" and body["tax_year_applicability"] == "unknown"
    assert "capped at 20 catalog entries" in body["notice"]
    assert len(result.content) <= references._RESULT_CHARS
    assert registry.catalog.call_args.kwargs["limit"] == 20
    assert arguments["limit"] == requested
    assert not any(request.url.path == "/api/embed" for request in requests)


@pytest.mark.parametrize("name,changes", [
    ("reference_search", {"top_k": -8}),
    ("reference_search", {"top_k": "not-a-count"}),
    ("reference_search", {"top_k": "8"}),
    ("reference_search", {"top_k": 8.5}),
    ("reference_search", {"library_id": "INVALID_LIBRARY"}),
    ("reference_search", {"catalog": "outside"}),
    ("reference_search", {"root": "/private/SECRET_REFERENCE_PATH"}),
    ("reference_lookup", {"limit": 0}),
    ("reference_lookup", {"limit": "not-a-count"}),
    ("reference_lookup", {"limit": "100"}),
    ("reference_lookup", {"limit": 20.5}),
    ("reference_lookup", {"user": "SECRET_REFERENCE_OWNER"}),
    ("read_reference", {"document_id": "SECRET_REFERENCE_DOCUMENT", "limit": 5000}),
    ("read_reference", {"document_id": "a" * 64, "page": 0}),
])
async def test_invalid_private_reference_arguments_get_safe_corrective_feedback(private_reference, name, changes, caplog):
    runtime, app, owner, requests = private_reference
    arguments = {"query": "SECRET_REFERENCE_QUERY", **changes} if name != "read_reference" else changes
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": name, "arguments": arguments},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert result.is_error
    body = json.loads(result.content)
    assert body["error"] == "arguments_invalid"
    assert "declared fields" in body["detail"] and "types and limits" in body["detail"]
    assert "SECRET_REFERENCE" not in result.content + caplog.text
    assert not any(request.url.path == "/api/embed" for request in requests)


@pytest.mark.parametrize("schema,args", [
    (references.SearchRequest, {"query": "filing", "top_k": 6}),
    (references.CatalogRequest, {"query": "filing", "limit": 21}),
    (references.ReadRequest, {"document_id": "a" * 64, "limit": 4001}),
])
def test_public_reference_request_schemas_keep_their_strict_bounds(schema, args):
    with pytest.raises(ValidationError):
        schema.model_validate(args)


@pytest.mark.parametrize("private_reference", [_TEXT * 60], indirect=True)
async def test_cap_notice_preserves_existing_result_truncation_and_json_budget(private_reference):
    runtime, app, owner, _ = private_reference
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": "reference_search", "arguments": {
                    "query": "gross income filing requirements", "top_k": 8,
                }},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert body["returned"] > 0
    assert body["result_limit_applied"] is True
    assert "capped at 5 passages" in body["notice"]
    assert len(result.content) <= references._RESULT_CHARS
    for passage in body["results"]:
        assert passage["char_end"] == passage["offset"] + len(passage["text"])


@pytest.mark.parametrize("name,count_field", [("reference_search", "top_k"), ("reference_lookup", "limit")])
async def test_boolean_counts_keep_existing_schema_behavior_without_count_normalization(private_reference, name, count_field):
    runtime, app, owner, _ = private_reference
    query = "gross income filing requirements" if name == "reference_search" else "Individual"
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": name, "arguments": {"query": query, count_field: True}},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert body["returned"] == 1
    assert "capped" not in body.get("notice", "")


async def test_cap_notice_does_not_break_evidence_that_already_fills_result_budget(private_reference):
    runtime, app, owner, _ = private_reference
    async with runtime.scope(app) as execution:
        service = execution.state.reference_search
        search = service.search

        async def metadata_near_budget(**args):
            result = await search(**args)
            result["result_limit_applied"] = False
            result["notice"] += " " * (references._RESULT_CHARS - 1 - references._size(result))
            return result

        service.search = metadata_near_budget
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": "reference_search", "arguments": {
                    "query": "gross income filing requirements", "top_k": 8,
                }},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert body["returned"] > 0 and _TEXT in body["results"][0]["text"]
    assert len(result.content) <= references._RESULT_CHARS


@pytest.mark.parametrize("name,args", [
    ("reference_search", {"top_k": 8, "catalog": "prior"}),
    ("reference_search", {"top_k": 8, "tax_year": 2027}),
    ("reference_search", {"top_k": 8, "library_id": "another-library"}),
    ("reference_lookup", {"limit": 100, "tax_year": 2025}),
    ("reference_lookup", {"limit": 100, "library_id": "another-library"}),
])
async def test_count_caps_do_not_repair_or_widen_reference_scope(private_reference, name, args):
    runtime, app, owner, requests = private_reference
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            result = await dispatch_one(execution.state.archive_http, execution.state.tools, {
                "function": {"name": name, "arguments": {
                    "query": "gross income filing requirements", **args,
                }},
            }, user_id="owner", timeout_s=5, max_result_chars=4000)
    assert result.is_error
    assert not any(request.url.path == "/api/embed" for request in requests)
