"""Private callbacks use the published local index and verified local embeddings."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
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
async def private_reference(tmp_path):
    root = tmp_path / "irs"
    with ArchiveStore(root) as archive:
        _record(archive, [_entry("p17.pdf", product="Publication 17")], content=_pdf(_TEXT))
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
    tools = ToolRegistry({"reference_search": ToolSpec(
        name="reference_search", description="Search the owned public reference index",
        parameters={"type": "object"}, server_url=_SERVER, path="/reference_search",
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
