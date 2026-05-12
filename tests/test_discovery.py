"""Tests for tool discovery — turning OpenAPI specs into Ollama tool schemas.

The tool layer didn't have hermetic tests before. This file plus
test_dispatch.py fill that gap. We synthesize OpenAPI documents via
`httpx.MockTransport` and verify the output shape: which endpoints
become tools, how `$ref`s get inlined, how unsupported JSON Schema
keywords get scrubbed, what happens on unreachable servers.

The audit findings this test file pins:
  - `health` endpoints are filtered by tag, not by a `op_id == "health"`
    special case. The special case was redundant (already handled by the
    GET vs POST filter and the `tools` tag check) and got dropped.
  - Tag-based filtering remains the canonical rule.

Pure-function tests over httpx; no real network calls.
"""

from __future__ import annotations

from typing import Any

import httpx

from audrey.tools.discovery import (
    ToolRegistry,
    ToolSpec,
    _resolve_refs,
    _strip_unsupported_keywords,
    discover_all,
    discover_one,
)

# ─── Fixtures ─────────────────────────────────────────────────────────


def _openapi(paths: dict[str, Any], components: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a minimal OpenAPI doc for the mock transport."""
    return {
        "openapi": "3.1.0",
        "info": {"title": "test", "version": "0.0.0"},
        "paths": paths,
        "components": {"schemas": components or {}},
    }


def _post(operation_id: str, *, tags: list[str], schema: dict[str, Any], description: str = "") -> dict[str, Any]:
    """Build one OpenAPI POST operation entry."""
    return {
        "post": {
            "operationId": operation_id,
            "tags": tags,
            "description": description,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": schema}},
            },
            "responses": {"200": {"description": "OK"}},
        }
    }


def _client_with_doc(doc: dict[str, Any]) -> httpx.AsyncClient:
    """An AsyncClient whose /openapi.json returns `doc`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=doc)
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ─── _strip_unsupported_keywords ──────────────────────────────────────


def test_strip_removes_unknown_keywords():
    """JSON Schema keywords not in the allow-list disappear."""
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "format": "email"}},
        "additionalProperties": False,
        "examples": [{"q": "x@y"}],
    }
    out = _strip_unsupported_keywords(schema)
    assert out == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


def test_strip_preserves_property_names():
    """`properties` is a name-keyed map. The scrubber must not filter its
    keys (the names) against the keyword allow-list — only their schema
    bodies get cleaned."""
    schema = {
        "type": "object",
        "properties": {
            # "format" is a property NAME, not a JSON Schema keyword here.
            "format": {"type": "string"},
            "additionalProperties": {"type": "boolean"},
        },
    }
    out = _strip_unsupported_keywords(schema)
    assert set(out["properties"].keys()) == {"format", "additionalProperties"}


# ─── _resolve_refs ────────────────────────────────────────────────────


def test_resolve_refs_inlines_local_schema():
    components = {"Inner": {"type": "string", "description": "leaf"}}
    schema = {"type": "object", "properties": {"x": {"$ref": "#/components/schemas/Inner"}}}
    out = _resolve_refs(schema, components)
    assert out == {
        "type": "object",
        "properties": {"x": {"type": "string", "description": "leaf"}},
    }


def test_resolve_refs_does_not_mutate_components():
    """Resolved refs are deepcopied so sibling endpoints see the original."""
    components = {"Inner": {"type": "string"}}
    schema = {"$ref": "#/components/schemas/Inner"}
    resolved = _resolve_refs(schema, components)
    resolved["mutated"] = True
    assert "mutated" not in components["Inner"]


# ─── discover_one — tag and shape filtering ───────────────────────────


async def test_discover_one_finds_tools_tag():
    doc = _openapi({
        "/web_search": _post(
            "web_search",
            tags=["tools"],
            schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            description="search the web",
        ),
    })
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")
    assert [t.name for t in tools] == ["web_search"]
    assert tools[0].description == "search the web"
    assert tools[0].parameters == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    assert tools[0].server_url == "http://server.test"
    assert tools[0].path == "/web_search"


async def test_discover_one_skips_endpoints_without_tools_tag():
    """The `system`-tagged /health endpoint is filtered out by the tag check.
    No special-case needed for operationId == 'health'."""
    doc = _openapi({
        "/health": {
            "post": {
                "operationId": "health",
                "tags": ["system"],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {}}}}},
                "responses": {"200": {"description": "OK"}},
            }
        },
        "/web_search": _post(
            "web_search",
            tags=["tools"],
            schema={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
    })
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")
    assert [t.name for t in tools] == ["web_search"]


async def test_discover_one_skips_endpoint_with_missing_operation_id():
    """Endpoints without an operationId are unusable; skip them silently."""
    doc = _openapi({
        "/anon": {
            "post": {
                "tags": ["tools"],
                "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "properties": {"x": {"type": "string"}}}}}},
                "responses": {"200": {"description": "OK"}},
            }
        },
    })
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")
    assert tools == []


async def test_discover_one_skips_non_object_schema():
    """The Ollama tool format requires object-with-properties schemas.
    String/array/number top-level schemas get dropped."""
    doc = _openapi({
        "/bare_string": _post(
            "bare_string",
            tags=["tools"],
            schema={"type": "string"},
        ),
    })
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")
    assert tools == []


async def test_discover_one_returns_empty_on_unreachable_server():
    """Network errors are logged and produce an empty list; the rest of
    the registry still loads."""
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = await discover_one(client, "http://broken.test")
    assert tools == []


async def test_discover_one_resolves_refs_in_request_body():
    doc = _openapi(
        paths={
            "/echo": _post(
                "echo",
                tags=["tools"],
                schema={"$ref": "#/components/schemas/EchoRequest"},
            ),
        },
        components={
            "EchoRequest": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        },
    )
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")
    assert tools[0].parameters == {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }


# ─── discover_all + ToolRegistry ──────────────────────────────────────


async def test_registry_to_ollama_tools_shape():
    """End-to-end: discovered tool round-trips through
    `to_ollama_tools()` into the Ollama function-call schema."""
    doc = _openapi({
        "/web_search": _post(
            "web_search",
            tags=["tools"],
            schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            description="search the web",
        ),
    })
    # Patch the module-level AsyncClient by monkeypatching discover_one
    # to use our mocked transport directly.
    async with _client_with_doc(doc) as client:
        tools = await discover_one(client, "http://server.test")

    registry = ToolRegistry()
    for t in tools:
        registry.by_name[t.name] = t

    out = registry.to_ollama_tools()
    assert out == [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }]


async def test_discover_all_empty_server_list_returns_empty_registry():
    registry = await discover_all([])
    assert registry.names() == []


def test_tool_spec_to_ollama_tool_shape():
    """Sanity for the dataclass conversion on its own."""
    spec = ToolSpec(
        name="memory_search",
        description="search memories",
        parameters={"type": "object", "properties": {"user": {"type": "string"}}},
        server_url="http://x",
        path="/memory_search",
    )
    assert spec.to_ollama_tool() == {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "search memories",
            "parameters": {"type": "object", "properties": {"user": {"type": "string"}}},
        },
    }
