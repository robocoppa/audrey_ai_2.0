"""Tests for the tool dispatcher — turning one Ollama tool_call into a
ToolResult.

Pairs with test_discovery.py to cover the tool layer. The dispatcher is
the single load-bearing security boundary in the tool path: every
user-scoped tool's `user` argument gets overwritten with the pipeline
user_id before the network call. The other invariants pinned here:

  - The dispatcher never raises. Every failure path returns a
    `ToolResult` with `is_error=True`.
  - JSON-string `arguments` (Ollama sometimes returns this shape)
    parse correctly; the error path echoes the original raw value
    rather than a half-rebound local.
  - Unknown tool, network error, timeout, 4xx/5xx, and result
    truncation each return a structured tool result the model can
    reason against.

Pure-function tests; httpx via MockTransport.
"""

from __future__ import annotations

import json

import httpx

from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import _force_user_tag, dispatch_one, to_tool_message

# ─── Fixtures ─────────────────────────────────────────────────────────


def _registry(specs: list[ToolSpec]) -> ToolRegistry:
    r = ToolRegistry()
    for s in specs:
        r.by_name[s.name] = s
    return r


def _spec(name: str, server: str = "http://server.test", path: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        server_url=server,
        path=path or f"/{name}",
    )


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _call(name: str, args: dict | str | None, call_id: str | None = "call_0") -> dict:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": args},
    }


# ─── Happy path ───────────────────────────────────────────────────────


async def test_dispatch_one_happy_path():
    """A registered tool with a valid call returns ok and a JSON content."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": ["a", "b"]})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry, _call("web_search", {"q": "hello"}),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert seen["url"].endswith("/web_search")
    assert seen["body"] == {"q": "hello"}
    assert result.is_error is False
    assert json.loads(result.content) == {"results": ["a", "b"]}
    assert result.name == "web_search"
    assert result.call_id == "call_0"


# ─── User-scope invariant ─────────────────────────────────────────────


async def test_dispatch_one_overwrites_user_for_scoped_tool():
    """The single load-bearing security move: a user-scoped tool gets
    its `user` argument overwritten with the pipeline user_id, no matter
    what the model supplied."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    registry = _registry([_spec("memory_search")])
    async with _client(handler) as http:
        await dispatch_one(
            http, registry,
            _call("memory_search", {"user": "evil@example.com", "query": "secrets"}),
            max_result_chars=2000, timeout_s=5.0,
            user_id="alice@example.com",
        )
    assert seen["body"]["user"] == "alice@example.com"
    assert seen["body"]["query"] == "secrets"


async def test_dispatch_one_overwrites_memory_store_tags():
    """`memory_store` carries the user identity in a free-form `tags`
    string. The dispatcher strips any existing `user:` token and appends
    the real one."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"key": "x", "value": "y", "tags": seen["body"]["tags"],
                                          "created_at": "", "updated_at": ""})

    registry = _registry([_spec("memory_store")])
    async with _client(handler) as http:
        await dispatch_one(
            http, registry,
            _call("memory_store", {"key": "x", "value": "y", "tags": "user:evil@example.com,topic:secrets"}),
            max_result_chars=2000, timeout_s=5.0,
            user_id="alice@example.com",
        )
    # `tags` is comma-separated and order may vary; check membership.
    tags = set(seen["body"]["tags"].split(","))
    assert "user:alice@example.com" in tags
    assert "topic:secrets" in tags
    assert "user:evil@example.com" not in tags


async def test_dispatch_one_does_not_overwrite_unscoped_tool():
    """Tools not in `_USER_SCOPED_TOOLS` (e.g. `web_search`) leave the
    model's arguments alone — there's no user identity to overwrite for
    an anonymous web search."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        await dispatch_one(
            http, registry,
            _call("web_search", {"q": "btrfs"}),
            max_result_chars=2000, timeout_s=5.0,
            user_id="alice@example.com",
        )
    # `user` was never in the args and should not have been added.
    assert "user" not in seen["body"]


def test_force_user_tag_replaces_existing_user_token():
    out = _force_user_tag("topic:a,user:evil@example.com,topic:b", "alice@example.com")
    parts = set(out.split(","))
    assert "user:alice@example.com" in parts
    assert "topic:a" in parts
    assert "topic:b" in parts
    assert "user:evil@example.com" not in parts


def test_force_user_tag_appends_when_none_present():
    out = _force_user_tag("topic:a", "alice@example.com")
    assert "user:alice@example.com" in out.split(",")
    assert "topic:a" in out.split(",")


# ─── Argument parsing ─────────────────────────────────────────────────


async def test_dispatch_one_accepts_arguments_as_json_string():
    """Ollama sometimes returns `arguments` as a JSON-encoded string
    rather than a dict. The dispatcher parses it transparently."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", '{"q": "hello"}'),  # string, not dict
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is False
    assert seen["body"] == {"q": "hello"}


async def test_dispatch_one_json_string_error_echoes_raw_args():
    """When `arguments` is a string that won't parse, the error payload
    echoes the original string. This pins the `raw_args` refactor: even
    if a future edit rebinds `args` before the except branch, the error
    path will still log/echo the original value."""
    registry = _registry([_spec("web_search")])
    async with _client(lambda _r: httpx.Response(200)) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", "not json {garbage"),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "arguments_not_json"
    assert body["raw"] == "not json {garbage"


async def test_dispatch_one_rejects_non_object_arguments():
    """If `arguments` parses to something other than an object (e.g. a
    list or number), reject it before dispatch."""
    registry = _registry([_spec("web_search")])
    async with _client(lambda _r: httpx.Response(200)) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", "[1, 2, 3]"),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "arguments_not_object"


# ─── Unknown tool, network, timeout, 4xx/5xx ──────────────────────────


async def test_dispatch_one_unknown_tool():
    registry = _registry([_spec("web_search")])
    async with _client(lambda _r: httpx.Response(200)) as http:
        result = await dispatch_one(
            http, registry,
            _call("not_a_tool", {}),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "unknown_tool"
    assert body["tool"] == "not_a_tool"
    assert body["available"] == ["web_search"]


async def test_dispatch_one_timeout_returns_error_result():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {"q": "x"}),
            max_result_chars=2000, timeout_s=0.1,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "timeout"


async def test_dispatch_one_network_error_returns_error_result():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {"q": "x"}),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "network_error"


async def test_dispatch_one_http_4xx_returns_error_result():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="bad args")

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {"q": "x"}),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert result.is_error is True
    body = json.loads(result.content)
    assert body["error"] == "http_422"


# ─── Truncation ───────────────────────────────────────────────────────


async def test_dispatch_one_truncates_oversize_responses():
    """Long responses get cut at `max_result_chars` with a visible
    `…[truncated]` marker so the model knows it didn't see everything."""
    big_blob = {"results": ["x" * 5000]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=big_blob)

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {"q": "x"}),
            max_result_chars=500, timeout_s=5.0,
        )
    assert len(result.content) <= 500
    assert result.content.endswith("…[truncated]")
    assert result.is_error is False


async def test_dispatch_one_does_not_truncate_short_responses():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {"q": "x"}),
            max_result_chars=2000, timeout_s=5.0,
        )
    assert "…[truncated]" not in result.content


# ─── to_tool_message ──────────────────────────────────────────────────


async def test_to_tool_message_includes_call_id_when_present():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"x": 1})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {}, call_id="call_42"),
            max_result_chars=2000, timeout_s=5.0,
        )
    msg = to_tool_message(result)
    assert msg == {
        "role": "tool",
        "name": "web_search",
        "content": result.content,
        "tool_call_id": "call_42",
    }


async def test_to_tool_message_omits_call_id_when_absent():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"x": 1})

    registry = _registry([_spec("web_search")])
    async with _client(handler) as http:
        result = await dispatch_one(
            http, registry,
            _call("web_search", {}, call_id=None),
            max_result_chars=2000, timeout_s=5.0,
        )
    msg = to_tool_message(result)
    assert "tool_call_id" not in msg
