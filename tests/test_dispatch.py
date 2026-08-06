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
import sys
from pathlib import Path

import httpx

from audrey.tools.discovery import ToolRegistry, ToolSpec, discover_one
from audrey.tools.dispatch import (
    _force_user_tag,
    audit_user_scoping,
    dispatch_one,
    to_tool_message,
)

# Same pattern as test_web_fetch.py: put tools-server on the path at import
# time so the module itself can be imported lazily inside the one test that
# needs it, without a `Path` call inside an async function.
_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

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


async def test_dispatch_one_overwrites_user_for_list_my_files():
    """Phase 40's file-listing tool, pinned separately from the others.

    `POST /v1/files/list` on Audrey authenticates with a service token and
    takes its target user from the request body — there is no second gate on
    that route. This overwrite is the entire reason a prompt naming another
    person's address cannot return their file list.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"files": []})

    registry = _registry([_spec("list_my_files")])
    async with _client(handler) as http:
        await dispatch_one(
            http, registry,
            _call("list_my_files", {"user": "victim@example.com"}),
            max_result_chars=2000, timeout_s=5.0,
            user_id="alice@example.com",
        )
    assert seen["body"]["user"] == "alice@example.com"


# ─── Scoping audit (the check the set's own comment asked for) ────────


def _user_spec(name: str) -> ToolSpec:
    """A spec whose request schema declares a `user` property."""
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"user": {"type": "string"}}},
        server_url="http://server.test",
        path=f"/{name}",
    )


def test_audit_flags_a_user_taking_tool_missing_from_the_set():
    """The failure this exists to catch: a new tools-server route with a
    `user` field, shipped without the matching `_USER_SCOPED_TOOLS` entry.
    Nothing else in the system notices — the dispatcher just forwards
    whatever the model wrote."""
    assert audit_user_scoping(_registry([_user_spec("list_their_files")])) == [
        "list_their_files"
    ]


def test_audit_is_quiet_for_a_tool_already_in_the_set():
    assert audit_user_scoping(_registry([_user_spec("list_my_files")])) == []


def test_audit_ignores_tools_that_take_no_user():
    assert audit_user_scoping(_registry([_spec("web_search")])) == []


async def test_every_user_taking_tool_on_the_real_server_is_scoped():
    """The one that would actually have caught it.

    The three above audit synthetic specs, which only prove the function
    works. This runs the **real** tools-server OpenAPI document through the
    **real** discovery path and asserts the set covers everything it finds —
    so adding a `user`-taking route to `tools-server/app.py` and forgetting
    `_USER_SCOPED_TOOLS` fails here, on a laptop, rather than silently in
    production where the symptom is one user reading another's data.
    """
    import app as tools_server  # lazily, so a heavy import can't break collection

    schema = tools_server.app.openapi()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=schema)

    async with _client(handler) as http:
        specs = await discover_one(http, "http://custom-tools:8000")

    registry = _registry(specs)
    assert "list_my_files" in registry.by_name, "phase 40's tool is not being discovered"
    assert audit_user_scoping(registry) == []


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
    assert body["available"] == ["web_search"]
    # The invented name must NOT be echoed back — repeating it in the error text
    # is what got `web_fetch` called a second time in the 2026-07-22 research eval.
    assert "not_a_tool" not in result.content


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
    """Long responses get cut at `max_result_chars`, saying how much was lost.

    This used to assert a bare `…[truncated]`, and that marker turned out to be
    actively harmful: it said a cut had happened and nothing else, so on
    2026-08-05 a model reasonably inferred it should ask again with a bigger
    `top_k` — which returns an identical amount of text, because the cap is on
    the response and not on the query. `tests/test_tool_truncation.py` owns the
    behaviour in full; this pins that the dispatcher applies it.

    One item in the list here, so there is nothing to drop and the character
    path is what runs.
    """
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
    assert "truncated" in result.content
    # The size of the hole, and that retrying will not close it.
    assert "of 5,0" in result.content
    assert "larger top_k" in result.content
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
