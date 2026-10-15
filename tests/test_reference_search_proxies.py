"""Public reference search proxy validation and dependency boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from audrey.tools.discovery import ToolRegistry, ToolUserScope, discover_one
from audrey.tools.dispatch import dispatch_one

_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

import app as sidecar  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402


@pytest.fixture
def capabilities(monkeypatch):
    registry = CapabilityRegistry.all_available()
    monkeypatch.setattr(sidecar.app.state, "capabilities", registry)
    return registry


async def _discover():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sidecar.app),
        base_url="http://sidecar",
    ) as client:
        specs = await discover_one(client, "http://sidecar")
    return ToolRegistry(by_name={spec.name: spec for spec in specs})


async def _call(monkeypatch, handler, payload=None):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://audrey",
        headers=sidecar._service_headers("reference-service-marker"),
    ) as backend:
        monkeypatch.setattr(sidecar.app.state, "audrey", backend, raising=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sidecar.app),
            base_url="http://sidecar",
        ) as client:
            return await client.post(
                "/reference_search",
                json={"query": "deduction eligibility"} if payload is None else payload,
            )


async def test_reference_search_real_discovery_has_public_scope_and_search_dependencies(
    capabilities,
):
    registry = await _discover()
    spec = registry.get("reference_search")
    assert spec is not None
    assert spec.user_scope is ToolUserScope.NONE
    assert spec.dependencies == frozenset({"audrey_references", "qdrant", "text_embedding"})
    assert not spec.purge_gated
    assert spec.parameters["properties"]["catalog"]["default"] == "current"
    assert spec.parameters["properties"]["top_k"]["default"] == 3
    assert "query" in spec.parameters["required"]
    assert set(spec.parameters["properties"]).isdisjoint(
        {
            "user",
            "path",
            "root",
            "collection",
            "release_id",
            "history_years",
            "as_of_year",
        }
    )
    assert "read_reference" in spec.description
    assert "not exhaustive" in spec.description
    assert "not tax years" in spec.description


async def test_search_preserves_scope_and_backend_citations_with_service_identity(
    capabilities,
    monkeypatch,
):
    captured = {}
    payload = {
        "library_id": "irs-federal",
        "query": "retirement contribution deduction",
        "catalog": "prior",
        "top_k": 2,
        "product_number": "Publication 590-A",
        "tax_year": 2025,
    }
    result_body = {
        "library_id": "irs-federal",
        "release_id": "published-reference-release",
        "matches": [{"document_id": "a" * 64, "revision_sha256": "b" * 64, "page": 5}],
        "tax_year_applicability": "unknown",
    }

    def handler(request):
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        captured["timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json=result_body)

    result = await _call(monkeypatch, handler, payload)
    assert result.status_code == 200
    assert result.json() == result_body
    assert captured["path"] == "/v1/references/search"
    assert captured["body"] == payload
    assert captured["headers"]["X-Audrey-Service-Token"] == "reference-service-marker"
    assert "Authorization" not in captured["headers"]
    assert captured["timeout"] == 25


@pytest.mark.parametrize("dependency", ["audrey_references", "qdrant", "text_embedding"])
async def test_dependency_failure_hides_search_and_never_calls_fallback(
    capabilities,
    monkeypatch,
    dependency,
):
    capabilities.set_unavailable(dependency, "offline")
    registry = await _discover()
    assert registry.get("reference_search") is None
    if dependency != "audrey_references":
        assert registry.get("reference_lookup") is not None
        assert registry.get("read_reference") is not None

    def unexpected(_request):
        pytest.fail("Unavailable reference search must not call backend/global/private fallbacks")

    result = await _call(monkeypatch, unexpected)
    assert result.status_code == 503
    assert result.json()["detail"]["components"] == [dependency]


@pytest.mark.parametrize(
    "field",
    [
        "user",
        "tags",
        "path",
        "root",
        "collection",
        "release_id",
        "as_of_year",
        "history_years",
    ],
)
async def test_model_cannot_override_private_identity_or_library_publication_policy(
    capabilities,
    monkeypatch,
    field,
):
    def unexpected(_request):
        pytest.fail("Forbidden model-controlled input must not reach the backend")

    result = await _call(monkeypatch, unexpected, {"query": "tax", field: "forged"})
    assert result.status_code == 422


@pytest.mark.parametrize(
    "extra",
    [
        {"query": ""},
        {"query": "x" * 1001},
        {"library_id": "../private"},
        {"catalog": "all"},
        {"top_k": 0},
        {"top_k": 6},
        {"product_number": "x" * 101},
        {"tax_year": 1700},
        {"tax_year": 2300},
    ],
)
async def test_search_request_limits_are_checked_before_proxy(capabilities, monkeypatch, extra):
    def unexpected(_request):
        pytest.fail("Invalid reference query/bounds must not reach the backend")

    result = await _call(monkeypatch, unexpected, {"query": "tax", **extra})
    assert result.status_code == 422


@pytest.mark.parametrize("code", [404, 409, 422, 503])
async def test_backend_missing_release_or_unverified_year_has_no_fallback(
    capabilities,
    monkeypatch,
    code,
):
    requests = []
    detail = {"error": "reference_search_unavailable"}

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(code, json={"detail": detail})

    result = await _call(monkeypatch, handler)
    assert result.status_code == code
    assert result.json() == {"detail": detail}
    assert requests == ["/v1/references/search"]


async def test_dispatch_does_not_add_private_owner_to_public_search(capabilities):
    registry = await _discover()
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"library_id": "irs-federal", "matches": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await dispatch_one(
            client,
            registry,
            {
                "id": "call_reference",
                "function": {"name": "reference_search", "arguments": {"query": "tax"}},
            },
            user_id="alice@example.com",
            max_result_chars=4000,
            timeout_s=30,
        )
    assert not result.is_error
    assert captured == {"query": "tax"}
