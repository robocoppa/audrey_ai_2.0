"""Tool discovery — fetch each tool server's /openapi.json and convert to Ollama-callable tools.

Each tool server (started Phase 2) exposes a FastAPI app with one or more
POST endpoints under the `tools` tag. We hit `/openapi.json` at startup,
then again on demand via `POST /v1/tools/rediscover` if a server changes.

Output shape per discovered tool — matches Ollama's `/api/chat` tool schema:
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "...",
        "parameters": {<inlined JSON Schema>}
      }
    }

We also keep the dispatch metadata (server URL + path) in a separate dict
so `dispatch.py` can route a tool_call back to the right server.

Failure modes:
  - Server unreachable → log warning, skip; the rest still load.
  - Endpoint with no POST or no request body → skip (not a tool, e.g. /health).
  - $ref resolution failure → skip the endpoint, don't poison the registry.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolSpec:
    name: str                          # operation_id, e.g. "web_search"
    description: str                   # endpoint summary or description
    parameters: dict[str, Any]         # inlined JSON Schema for the request body
    server_url: str                    # base URL of the originating server
    path: str                          # POST path on that server, e.g. "/web_search"

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolRegistry:
    by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.by_name.keys())

    def specs(self) -> list[ToolSpec]:
        return list(self.by_name.values())

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        return [s.to_ollama_tool() for s in self.by_name.values()]

    def get(self, name: str) -> ToolSpec | None:
        return self.by_name.get(name)


# ─── OpenAPI → Ollama-tool conversion ─────────────────────────────────

def _resolve_refs(node: Any, components: dict[str, Any]) -> Any:
    """Walk a JSON Schema fragment and inline any $ref → #/components/schemas/...

    Ollama's tool-calling implementation doesn't follow refs — schemas must
    be self-contained. We deepcopy as we go so the components dict stays
    intact for sibling endpoints.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/components/schemas/"):
                raise ValueError(f"Unsupported $ref: {ref}")
            key = ref.removeprefix("#/components/schemas/")
            target = components.get(key)
            if target is None:
                raise ValueError(f"Missing schema: {key}")
            return _resolve_refs(deepcopy(target), components)
        return {k: _resolve_refs(v, components) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, components) for v in node]
    return node


def _strip_unsupported_keywords(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON-Schema keywords that confuse small Ollama models.

    Most coder/general models accept a clean subset: type, properties,
    required, enum, description, items, default, minLength, maxLength,
    minimum, maximum. Drop the rest — they cause silent tool-call failures
    on the smaller routers.
    """
    allowed = {
        "type", "properties", "required", "enum", "description",
        "items", "default", "minLength", "maxLength", "minimum", "maximum",
        "title",
    }

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k in allowed or k == "$defs"}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(schema)


def _build_tool_from_operation(
    *,
    operation_id: str,
    op: dict[str, Any],
    path: str,
    server_url: str,
    components: dict[str, Any],
) -> ToolSpec | None:
    """Convert one OpenAPI POST operation into a ToolSpec, or None if unsuitable."""
    request_body = op.get("requestBody") or {}
    content = request_body.get("content") or {}
    json_schema_wrapper = content.get("application/json") or {}
    raw_schema = json_schema_wrapper.get("schema")
    if not raw_schema:
        log.info("discovery: skip %s (%s): no application/json request body", operation_id, path)
        return None

    try:
        resolved = _resolve_refs(raw_schema, components)
    except ValueError as e:
        log.warning("discovery: skip %s: ref resolution failed: %s", operation_id, e)
        return None

    parameters = _strip_unsupported_keywords(resolved)
    if parameters.get("type") != "object" or not parameters.get("properties"):
        log.info(
            "discovery: skip %s: schema not an object-with-properties (type=%r, keys=%s)",
            operation_id, parameters.get("type"), sorted(parameters.keys()),
        )
        return None

    description = (op.get("description") or op.get("summary") or operation_id).strip()
    return ToolSpec(
        name=operation_id,
        description=description,
        parameters=parameters,
        server_url=server_url.rstrip("/"),
        path=path,
    )


async def discover_one(
    client: httpx.AsyncClient,
    server_url: str,
    *,
    timeout_s: float = 10.0,
) -> list[ToolSpec]:
    """Discover tools from one server. Returns [] on any error (logged)."""
    base = server_url.rstrip("/")
    try:
        r = await client.get(f"{base}/openapi.json", timeout=timeout_s)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("discovery: %s unreachable: %s", base, e)
        return []

    spec = r.json()
    paths = spec.get("paths", {}) or {}
    components = (spec.get("components", {}) or {}).get("schemas", {}) or {}

    tools: list[ToolSpec] = []
    for path, methods in paths.items():
        post = methods.get("post")
        if not post:
            continue
        op_id = post.get("operationId")
        if not op_id or op_id == "health":
            continue
        tags = post.get("tags") or []
        if tags and "tools" not in tags:
            # Server explicitly tagged this as non-tool (e.g. "system").
            continue
        tool = _build_tool_from_operation(
            operation_id=op_id, op=post, path=path,
            server_url=base, components=components,
        )
        if tool is not None:
            tools.append(tool)
    return tools


async def discover_all(server_urls: list[str], *, timeout_s: float = 10.0) -> ToolRegistry:
    """Discover tools across every configured server. Later names win on collision."""
    registry = ToolRegistry()
    if not server_urls:
        log.info("discovery: no tool servers configured, skipping")
        return registry

    async with httpx.AsyncClient() as client:
        for url in server_urls:
            tools = await discover_one(client, url, timeout_s=timeout_s)
            for t in tools:
                if t.name in registry.by_name:
                    log.warning("discovery: duplicate tool %r — %s overrides %s",
                                t.name, t.server_url, registry.by_name[t.name].server_url)
                registry.by_name[t.name] = t
            log.info("discovery: %s -> %d tool(s): %s", url, len(tools), [t.name for t in tools])

    log.info("discovery: total %d tool(s) registered: %s", len(registry.by_name), registry.names())
    return registry


__all__ = ["ToolSpec", "ToolRegistry", "discover_all", "discover_one"]
