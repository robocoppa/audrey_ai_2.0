"""Tool discovery — fetch each tool server's /openapi.json and convert to Ollama-callable tools.

Each tool server exposes a FastAPI app with one or more POST endpoints
under the `tools` tag. We hit `/openapi.json` at startup, then again on
demand via `POST /v1/tools/rediscover` if a server changes.

Output shape per discovered tool — matches Ollama's `/api/chat` tool schema:
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "...",
        "parameters": {<inlined JSON Schema>}
      }
    }

Each runtime policy record carries its dispatch metadata (server URL + path),
identity binding, component dependencies, and current availability. The model
registry is projected from records explicitly marked visible and available.

Failure modes:
  - Server unreachable → log warning, skip; the rest still load.
  - Endpoint with no POST or no request body → skip (not a tool, e.g. /health).
  - $ref resolution failure → skip the endpoint, don't poison the registry.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx

log = logging.getLogger(__name__)


class ToolVisibility(StrEnum):
    """Whether a declared endpoint may be shown to a model."""

    MODEL = "model"
    INTERNAL = "internal"


class ToolUserScope(StrEnum):
    """How the authenticated user is bound into a model tool request."""

    NONE = "none"
    ARGUMENT = "argument"
    TAGS = "tags"


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """Static security and dependency policy for one known tool name."""

    name: str
    visibility: ToolVisibility
    user_scope: ToolUserScope
    dependencies: frozenset[str]
    purge_gated: bool = False

    @property
    def user_scoped(self) -> bool:
        return self.user_scope is not ToolUserScope.NONE


def _declare(
    name: str,
    *,
    user_scope: ToolUserScope = ToolUserScope.NONE,
    dependencies: tuple[str, ...],
    purge_gated: bool = False,
) -> ToolDeclaration:
    return ToolDeclaration(
        name=name,
        visibility=ToolVisibility.MODEL,
        user_scope=user_scope,
        dependencies=frozenset(dependencies),
        purge_gated=purge_gated,
    )


# The sole model-tool security catalogue. Adding a `tags=["tools"]` route is
# insufficient: discovery refuses it until its identity binding and component
# dependencies are declared here. Dispatch consumes the same record, removing
# the former tools-server-route plus user-scope-set two-file invariant.
TOOL_DECLARATIONS: dict[str, ToolDeclaration] = {
    declaration.name: declaration
    for declaration in (
        _declare("web_search", dependencies=("web_search",)),
        _declare("web_fetch", dependencies=("web_fetch",)),
        _declare(
            "kb_search",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("audrey_kb", "qdrant", "text_embedding"),
        ),
        _declare(
            "kb_image_search",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("audrey_kb", "qdrant", "image_embedding"),
        ),
        _declare(
            "memory_store",
            user_scope=ToolUserScope.TAGS,
            dependencies=("memory", "qdrant", "text_embedding"),
        ),
        _declare(
            "memory_recall",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("chat_archive_source", "memory", "qdrant"),
            purge_gated=True,
        ),
        _declare(
            "memory_search",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=(
                "chat_archive_source", "memory", "qdrant", "text_embedding",
            ),
            purge_gated=True,
        ),
        _declare(
            "chat_history_search",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("chat_archive", "qdrant", "text_embedding"),
            purge_gated=True,
        ),
        _declare(
            "list_my_files",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("audrey_files",),
        ),
        _declare(
            "get_file_text",
            user_scope=ToolUserScope.ARGUMENT,
            dependencies=("audrey_files",),
        ),
    )
}


class ToolPolicyError(ValueError):
    """A model-visible endpoint violates the declared capability policy."""


@dataclass(slots=True)
class ToolSpec:
    name: str                          # operation_id, e.g. "web_search"
    description: str                   # endpoint summary or description
    parameters: dict[str, Any]         # inlined JSON Schema for the request body
    server_url: str                    # base URL of the originating server
    path: str                          # POST path on that server, e.g. "/web_search"
    visibility: ToolVisibility = ToolVisibility.MODEL
    user_scope: ToolUserScope = ToolUserScope.NONE
    dependencies: frozenset[str] = field(default_factory=frozenset)
    purge_gated: bool = False
    available: bool = True
    unavailable_reason: str | None = None

    @property
    def user_scoped(self) -> bool:
        return self.user_scope is not ToolUserScope.NONE

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
        return sorted(spec.name for spec in self.specs())

    def specs(self) -> list[ToolSpec]:
        return [
            spec
            for spec in self.by_name.values()
            if spec.visibility is ToolVisibility.MODEL and spec.available
        ]

    def policy_records(self) -> list[ToolSpec]:
        """Return all known records, including unavailable/internal entries."""
        return list(self.by_name.values())

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        return [spec.to_ollama_tool() for spec in self.specs()]

    def get(self, name: str) -> ToolSpec | None:
        spec = self.by_name.get(name)
        if (
            spec is None
            or spec.visibility is not ToolVisibility.MODEL
            or not spec.available
        ):
            return None
        return spec


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

    `properties` and `$defs` map *user-chosen names* → schema bodies, so
    we walk them as a name-keyed map (preserve keys, recurse into values)
    rather than filtering their keys against the keyword allow-list.
    """
    allowed = {
        "type", "properties", "required", "enum", "description",
        "items", "default", "minLength", "maxLength", "minimum", "maximum",
        "title",
    }
    name_keyed = {"properties", "$defs"}

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k not in allowed:
                    continue
                if k in name_keyed and isinstance(v, dict):
                    out[k] = {prop_name: clean(prop_schema) for prop_name, prop_schema in v.items()}
                else:
                    out[k] = clean(v)
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(schema)


def _capability_status(openapi: dict[str, Any]) -> dict[str, tuple[bool, str]]:
    """Validate the optional live component snapshot from a tool server."""
    raw = openapi.get("x-audrey-capabilities", {}) or {}
    if not isinstance(raw, dict):
        raise ToolPolicyError("x-audrey-capabilities is not an object")
    states: dict[str, tuple[bool, str]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ToolPolicyError("capability entries must be named objects")
        available = value.get("available")
        reason = value.get("reason", "")
        if not isinstance(available, bool) or not isinstance(reason, str):
            raise ToolPolicyError(f"{name}: malformed capability state")
        states[name] = (available, reason)
    return states


def _build_tool_from_operation(
    *,
    operation_id: str,
    op: dict[str, Any],
    path: str,
    server_url: str,
    components: dict[str, Any],
    declaration: ToolDeclaration,
    capability_status: dict[str, tuple[bool, str]],
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

    properties = parameters.get("properties") or {}
    has_user = "user" in properties
    has_tags = "tags" in properties
    if declaration.user_scope is ToolUserScope.ARGUMENT and not has_user:
        raise ToolPolicyError(
            f"{operation_id}: policy requires a user argument but schema has none"
        )
    if declaration.user_scope is not ToolUserScope.ARGUMENT and has_user:
        raise ToolPolicyError(
            f"{operation_id}: schema exposes user but policy does not bind it"
        )
    if declaration.user_scope is ToolUserScope.TAGS and not has_tags:
        raise ToolPolicyError(
            f"{operation_id}: tag-scoped policy requires a tags argument"
        )

    description = (op.get("description") or op.get("summary") or operation_id).strip()
    unavailable = sorted(
        dependency
        for dependency in declaration.dependencies
        if dependency in capability_status
        and not capability_status[dependency][0]
    )
    return ToolSpec(
        name=operation_id,
        description=description,
        parameters=parameters,
        server_url=server_url.rstrip("/"),
        path=path,
        visibility=declaration.visibility,
        user_scope=declaration.user_scope,
        dependencies=declaration.dependencies,
        purge_gated=declaration.purge_gated,
        available=not unavailable,
        unavailable_reason=(
            f"dependency_unavailable:{','.join(unavailable)}"
            if unavailable
            else None
        ),
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

    try:
        spec = r.json()
        if not isinstance(spec, dict):
            raise ToolPolicyError("OpenAPI document is not an object")
        paths = spec.get("paths", {}) or {}
        if not isinstance(paths, dict):
            raise ToolPolicyError("OpenAPI paths is not an object")
        component_root = spec.get("components", {}) or {}
        if not isinstance(component_root, dict):
            raise ToolPolicyError("OpenAPI components is not an object")
        components = component_root.get("schemas", {}) or {}
        if not isinstance(components, dict):
            raise ToolPolicyError("OpenAPI component schemas is not an object")
        capability_status = _capability_status(spec)

        tools: list[ToolSpec] = []
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                raise ToolPolicyError(f"{path}: path item is not an object")
            post = methods.get("post")
            if not post:
                continue
            if not isinstance(post, dict):
                raise ToolPolicyError(f"{path}: POST operation is not an object")
            tags = post.get("tags") or []
            if not isinstance(tags, list):
                raise ToolPolicyError(f"{path}: POST tags is not a list")
            if "tools" not in tags:
                # Only an explicit tools tag grants model visibility. Untagged
                # POSTs and internal/system operations are not capabilities.
                continue
            op_id = post.get("operationId")
            if not isinstance(op_id, str) or not op_id:
                raise ToolPolicyError(
                    f"{path}: model-visible POST has no operationId"
                )
            declaration = TOOL_DECLARATIONS.get(op_id)
            if declaration is None:
                raise ToolPolicyError(
                    f"{op_id}: model-visible tool has no policy declaration"
                )
            if declaration.visibility is not ToolVisibility.MODEL:
                raise ToolPolicyError(
                    f"{op_id}: tools tag conflicts with internal policy"
                )
            tool = _build_tool_from_operation(
                operation_id=op_id,
                op=post,
                path=path,
                server_url=base,
                components=components,
                declaration=declaration,
                capability_status=capability_status,
            )
            if tool is not None:
                tools.append(tool)
        return tools
    except Exception as e:  # noqa: BLE001 — isolate one malformed server
        log.warning("discovery: %s rejected: %s", base, e)
        return []


async def discover_all(server_urls: list[str], *, timeout_s: float = 10.0) -> ToolRegistry:
    """Discover tools across every configured server. Later names win on collision.

    Servers are fetched concurrently (`asyncio.gather`), then folded into the
    registry in `server_urls` order — so the collision precedence ("later names
    win") is unchanged from the old sequential loop; only the network fetch now
    overlaps. Per-server exception isolation is retained even if a future
    `discover_one` regression raises instead of returning an empty result.
    """
    registry = ToolRegistry()
    if not server_urls:
        log.info("discovery: no tool servers configured, skipping")
        return registry

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(discover_one(client, url, timeout_s=timeout_s) for url in server_urls),
            return_exceptions=True,
        )
        # Fold in input order — `gather` preserves it, so the zip stays aligned
        # and collision precedence matches the previous sequential behavior.
        for url, result in zip(server_urls, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                log.warning("discovery: %s failed unexpectedly: %s", url, result)
                continue
            tools = result
            for t in tools:
                if t.name in registry.by_name:
                    log.warning("discovery: duplicate tool %r — %s overrides %s",
                                t.name, t.server_url, registry.by_name[t.name].server_url)
                registry.by_name[t.name] = t
            log.info("discovery: %s -> %d tool(s): %s", url, len(tools), [t.name for t in tools])

    log.info("discovery: total %d tool(s) registered: %s", len(registry.by_name), registry.names())
    return registry


__all__ = [
    "TOOL_DECLARATIONS",
    "ToolDeclaration",
    "ToolPolicyError",
    "ToolRegistry",
    "ToolSpec",
    "ToolUserScope",
    "ToolVisibility",
    "discover_all",
    "discover_one",
]
