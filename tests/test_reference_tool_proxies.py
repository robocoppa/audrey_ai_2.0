"""Hermetic discovery and proxy guards for public reference model tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolUserScope, discover_one
from audrey.tools.dispatch import dispatch_one

_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

import app as sidecar  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402

_DOCUMENT = "5" * 64
_REVISION = "a" * 64
_PAYLOADS = {
    "reference_lookup": {"product_number": "Form 1040"},
    "read_reference": {"document_id": _DOCUMENT},
}
_PATHS = {
    "reference_lookup": "/v1/references/catalog",
    "read_reference": "/v1/references/read",
}


@pytest.fixture
def capabilities(monkeypatch):
    registry = CapabilityRegistry()
    monkeypatch.setattr(sidecar.app.state, "capabilities", registry)
    return registry


async def _discover():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sidecar.app), base_url="http://sidecar",
    ) as client:
        specs = await discover_one(client, "http://sidecar")
    return ToolRegistry(by_name={spec.name: spec for spec in specs})


async def _call(monkeypatch, name, handler, payload=None):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://audrey",
        headers=sidecar._service_headers("internal-service-marker"),
    ) as backend:
        monkeypatch.setattr(sidecar.app.state, "audrey", backend, raising=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sidecar.app), base_url="http://sidecar",
        ) as client:
            return await client.post(
                f"/{name}", json=_PAYLOADS[name] if payload is None else payload,
            )


async def test_real_discovery_has_exact_declared_public_scope_and_no_private_inputs(capabilities):
    registry = await _discover()
    raw_schemas = sidecar.app.openapi()["components"]["schemas"]
    for name in _PAYLOADS:
        declaration = TOOL_DECLARATIONS[name]
        spec = registry.get(name)
        assert spec is not None, name
        assert spec.user_scope is declaration.user_scope is ToolUserScope.NONE
        assert spec.dependencies == declaration.dependencies == frozenset({"audrey_references"})
        assert not spec.purge_gated
        assert not declaration.user_scoped
        assert set(spec.parameters["properties"]).isdisjoint({
            "user", "tags", "path", "root", "filename", "url", "collection",
            "as_of_year", "history_years",
        })
        assert "public" in spec.description and "English IRS" in spec.description
        assert "not tax years" in spec.description or "not tax" in spec.description
    assert raw_schemas["ReferenceLookupRequest"]["additionalProperties"] is False
    assert raw_schemas["ReferenceReadRequest"]["additionalProperties"] is False
    lookup = registry.get("reference_lookup")
    reader = registry.get("read_reference")
    assert lookup.parameters["properties"]["catalog"]["default"] == "current"
    assert lookup.parameters["properties"]["limit"]["default"] == 5
    assert reader.parameters["properties"]["limit"]["default"] == 1200
    assert reader.parameters["properties"]["catalog"]["default"] == "current"
    assert "document_id" in reader.parameters["required"]
    assert "older archive" in lookup.description
    assert "Older archive IDs are refused" in reader.description


@pytest.mark.parametrize("name", list(_PAYLOADS))
async def test_private_backend_and_embedding_outages_do_not_disable_public_reference_tools(
    capabilities, monkeypatch, name,
):
    for component in (
        "qdrant", "text_embedding", "image_embedding", "memory",
        "audrey_kb", "audrey_files", "audrey_history", "chat_archive_source",
    ):
        capabilities.set_unavailable(component, "offline")
    registry = await _discover()
    assert registry.get(name) is not None
    result = await _call(
        monkeypatch, name, lambda _: httpx.Response(200, json={"library_id": "irs-federal"}),
    )
    assert result.status_code == 200


@pytest.mark.parametrize("name", list(_PAYLOADS))
async def test_reference_dependency_failure_prevents_proxy_and_discovery(
    capabilities, monkeypatch, name,
):
    capabilities.set_unavailable("audrey_references", "offline")
    registry = await _discover()
    assert registry.get(name) is None

    def unexpected(_request):
        pytest.fail("Unavailable reference service must not be called")

    result = await _call(monkeypatch, name, unexpected)
    assert result.status_code == 503
    assert result.json()["detail"] == {
        "error": "capability_unavailable", "components": ["audrey_references"],
    }


@pytest.mark.parametrize("name", list(_PAYLOADS))
async def test_proxy_uses_existing_service_identity_and_preserves_public_source_metadata(
    capabilities, monkeypatch, name,
):
    captured = {}
    response = {
        "library_id": "irs-federal", "document_id": _DOCUMENT,
        "revision_sha256": _REVISION,
        "official_url": "https://www.irs.gov/pub/irs-pdf/f1040.pdf",
        "tax_year_applicability": "unknown", "page": 1,
        "text": "Public form instructions.",
    }

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        captured["timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json=response)

    result = await _call(monkeypatch, name, handler)
    assert result.status_code == 200
    assert result.json() == response
    assert captured["path"] == _PATHS[name]
    assert captured["headers"]["X-Audrey-Service-Token"] == "internal-service-marker"
    assert "Authorization" not in captured["headers"]
    assert captured["body"]["library_id"] == "irs-federal"
    assert "user" not in captured["body"]
    assert captured["timeout"] == 25
    assert captured["body"]["limit"] == (5 if name == "reference_lookup" else 1200)


async def test_lookup_can_browse_current_catalog_without_query(capabilities, monkeypatch):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"library_id": "irs-federal", "documents": []})

    result = await _call(monkeypatch, "reference_lookup", handler, payload={})
    assert result.status_code == 200
    assert captured == {
        "library_id": "irs-federal", "product_number": None, "query": None,
        "catalog": "current", "offset": 0, "limit": 5, "tax_year": None,
    }


async def test_reader_passes_exact_revision_page_and_pagination(capabilities, monkeypatch):
    captured = {}
    payload = {
        "library_id": "irs-federal", "document_id": _DOCUMENT,
        "page": 7, "offset": 2400, "limit": 1200, "revision_sha256": _REVISION,
        "catalog": "prior",
    }

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"library_id": "irs-federal", "text": "Page seven."})

    result = await _call(monkeypatch, "read_reference", handler, payload=payload)
    assert result.status_code == 200
    assert captured == payload


@pytest.mark.parametrize("name", list(_PAYLOADS))
@pytest.mark.parametrize("field", [
    "user", "path", "root", "filename", "url", "collection", "as_of_year", "history_years",
])
async def test_private_identity_path_and_history_policy_inputs_are_rejected(
    capabilities, monkeypatch, name, field,
):
    def unexpected(_request):
        pytest.fail("Unsupported inputs must be refused before the backend")

    result = await _call(
        monkeypatch, name, unexpected, payload={**_PAYLOADS[name], field: "forged"},
    )
    assert result.status_code == 422


@pytest.mark.parametrize(("name", "extra"), [
    ("reference_lookup", {"library_id": "../private-files"}),
    ("reference_lookup", {"library_id": "https://example.com"}),
    ("reference_lookup", {"product_number": ""}),
    ("reference_lookup", {"query": "x" * 201}),
    ("reference_lookup", {"catalog": "all"}),
    ("reference_lookup", {"offset": 100001}),
    ("reference_lookup", {"limit": 21}),
    ("read_reference", {"catalog": "all"}),
    ("read_reference", {"document_id": "short-id"}),
    ("read_reference", {"document_id": "https://www.irs.gov/pub/irs-pdf/f1040.pdf"}),
    ("read_reference", {"revision_sha256": "short-hash"}),
    ("read_reference", {"page": 0}),
    ("read_reference", {"page": 10001}),
    ("read_reference", {"offset": -1}),
    ("read_reference", {"offset": 200001}),
    ("read_reference", {"limit": 4001}),
])
async def test_reference_request_bounds_are_enforced_before_proxy(
    capabilities, monkeypatch, name, extra,
):
    def unexpected(_request):
        pytest.fail("Invalid reference IDs/bounds must be refused before the backend")

    result = await _call(
        monkeypatch, name, unexpected, payload={**_PAYLOADS[name], **extra},
    )
    assert result.status_code == 422


@pytest.mark.parametrize("name", list(_PAYLOADS))
async def test_dispatch_does_not_add_private_owner_to_public_reference_tool(capabilities, name):
    registry = await _discover()
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"library_id": "irs-federal"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await dispatch_one(
            client, registry,
            {"id": "call_reference", "function": {"name": name, "arguments": _PAYLOADS[name]}},
            user_id="alice@example.com", max_result_chars=4000, timeout_s=30,
        )
    assert not result.is_error
    assert "user" not in captured
    assert captured == _PAYLOADS[name]


async def test_unsupported_tax_year_is_explicit_backend_error(capabilities, monkeypatch):
    captured = {}
    detail = {"error": "reference_tax_year_unknown", "message": "Applicability is not established."}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(422, json={"detail": detail})

    result = await _call(
        monkeypatch, "reference_lookup", handler,
        payload={"product_number": "Form 1040", "tax_year": 2025},
    )
    assert captured["tax_year"] == 2025
    assert result.status_code == 422
    assert result.json() == {"detail": detail}


@pytest.mark.parametrize("name", list(_PAYLOADS))
@pytest.mark.parametrize("code", [404, 409, 422, 503])
async def test_expected_backend_failure_is_preserved(capabilities, monkeypatch, name, code):
    detail = {"error": "reference_unavailable", "message": "Requested edition is unavailable."}
    result = await _call(
        monkeypatch, name, lambda _: httpx.Response(code, json={"detail": detail}),
    )
    assert result.status_code == code
    assert result.json() == {"detail": detail}


@pytest.mark.parametrize(("failure", "code"), [
    ("timeout", 504), ("network", 502), ("html", 502), ("list", 502), ("null", 502),
])
async def test_transport_and_invalid_success_responses_do_not_leak_details(
    capabilities, monkeypatch, failure, code,
):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("secret-provider-address", request=request)
        if failure == "network":
            raise httpx.ConnectError("secret-provider-address", request=request)
        if failure == "html":
            return httpx.Response(200, text="<html>secret-provider-address</html>")
        return httpx.Response(200, json=["secret-provider-address"] if failure == "list" else None)

    result = await _call(monkeypatch, "read_reference", handler)
    assert result.status_code == code
    assert "secret-provider-address" not in result.text


@pytest.mark.parametrize("body", ["<html>internal-address</html>", ["internal-address"], None])
async def test_malformed_backend_error_returns_safe_generic_detail(
    capabilities, monkeypatch, body,
):
    def handler(_request):
        return httpx.Response(503, text=body) if isinstance(body, str) else httpx.Response(503, json=body)

    result = await _call(monkeypatch, "reference_lookup", handler)
    assert result.status_code == 503
    assert result.json() == {"detail": "Reference read failed."}
    assert "internal-address" not in result.text
