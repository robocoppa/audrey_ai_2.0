"""Bounded client-owned function tools and stateless Responses history.

This leaf adapter admits schemas, checks call/result links, and translates
provider calls. It never discovers or executes a server-managed tool.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from fastapi import HTTPException

from audrey.routes.openai.schemas import (
    ResponseCreateRequest,
    ResponseInputFunctionCall,
    ResponseInputFunctionCallOutput,
    ResponseInputMessage,
    ResponseReplayMessage,
)
from audrey.routes.openai.structured_outputs import (
    StructuredOutputError,
    _admit_schema,
    _unique_json_object,
    _validate_value,
)

MAX_TOOL_DEFINITIONS = 16
MAX_TOOL_BYTES = 64 * 1024
MAX_CALL_ARGUMENT_BYTES = 64 * 1024
MAX_GENERATED_CALLS = 8
MAX_REPLAY_ITEMS = 128
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
_TOOL_KEYS = frozenset({"type", "name", "description", "parameters", "strict"})


def _error(message: str, *, status: int = 400, code: str = "responses_client_tools_invalid") -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def has_client_tools(payload: ResponseCreateRequest) -> bool:
    """Identify explicit tool controls or client-owned output history."""
    return (
        payload.tools is not None
        or payload.tool_choice is not None
        or payload.parallel_tool_calls is not None
        or (
            not isinstance(payload.input, str)
            and any(not isinstance(item, ResponseInputMessage) for item in payload.input)
        )
    )


def _json_text(value: Any, *, label: str, status: int = 400) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise _error(f"{label} must contain standard JSON.", status=status) from exc


def _byte_length(text: str, *, label: str, status: int) -> int:
    try:
        return len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise _error(f"{label} must contain valid Unicode.", status=status) from exc


def _reject_constant(_value: str) -> Any:
    raise StructuredOutputError("Non-standard JSON constant.")


def _arguments(value: Any, *, status: int) -> tuple[str, dict[str, Any]]:
    encoded = value if isinstance(value, str) else _json_text(value, label="Function arguments", status=status)
    if _byte_length(encoded, label="Function arguments", status=status) > MAX_CALL_ARGUMENT_BYTES:
        raise _error("Function arguments exceed 64 KiB.", status=413 if status == 400 else status)
    try:
        decoded = json.loads(
            encoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (ValueError, RecursionError) as exc:
        raise _error("Function arguments must be valid JSON without duplicate keys.", status=status) from exc
    if not isinstance(decoded, dict):
        raise _error("Function arguments must decode to a JSON object.", status=status)
    normalized = _json_text(decoded, label="Function arguments", status=status)
    if _byte_length(normalized, label="Function arguments", status=status) > MAX_CALL_ARGUMENT_BYTES:
        raise _error("Function arguments exceed 64 KiB.", status=413 if status == 400 else status)
    return normalized, decoded


def _definitions(payload: ResponseCreateRequest) -> dict[str, dict[str, Any]]:
    tools = payload.tools or []
    if len(tools) > MAX_TOOL_DEFINITIONS:
        raise _error(f"At most {MAX_TOOL_DEFINITIONS} client functions are allowed.")
    if _byte_length(_json_text(tools, label="Client tools"), label="Client tools", status=400) > MAX_TOOL_BYTES:
        raise _error("Client tool definitions exceed 64 KiB.", status=413)
    admitted: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tools):
        if raw.get("type") != "function":
            raise _error("Only client-provided function tools are supported.", code="responses_feature_unsupported")
        unknown = sorted(set(raw) - _TOOL_KEYS)
        if unknown:
            raise _error(f"Client function {index} uses unsupported fields: {', '.join(unknown)}.")
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise _error("Client function names must contain 1–64 letters, digits, underscores, or hyphens.")
        if name in admitted:
            raise _error(f"Duplicate client function name: {name}.")
        description = raw.get("description")
        if description is not None and (not isinstance(description, str) or len(description) > 1024):
            raise _error(f"Client function {name} description must be at most 1,024 characters.")
        strict = raw.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise _error(f"Client function {name} strict must be a boolean.")
        schema = raw.get("parameters", {"type": "object", "properties": {}, "additionalProperties": False})
        if not isinstance(schema, dict):
            raise _error(f"Client function {name} parameters must be a JSON Schema object.")
        try:
            _admit_schema(schema, strict=bool(strict))
        except (StructuredOutputError, RecursionError) as exc:
            raise _error(f"Client function {name} parameters are invalid: {exc}") from exc
        admitted[name] = {"name": name, "parameters": schema, "strict": bool(strict)}
        if description is not None:
            admitted[name]["description"] = description
    return admitted


def _validate_arguments(arguments: dict[str, Any], definition: dict[str, Any], *, status: int) -> None:
    schema = definition["parameters"]
    try:
        _validate_value(arguments, schema, root=schema, path="$", depth=0)
    except (StructuredOutputError, RecursionError) as exc:
        raise _error(f"Function {definition['name']} arguments violate its schema: {exc}", status=status) from exc


def validate_client_tool_request(payload: ResponseCreateRequest) -> None:
    """Admit tool definitions, supported controls, and complete replay groups."""
    choice = payload.tool_choice
    if choice is not None and choice not in ("auto", "none"):
        raise _error("tool_choice supports only 'auto' or 'none'.", code="responses_feature_unsupported")
    definitions = _definitions(payload)
    if isinstance(payload.input, str):
        return
    if len(payload.input) > MAX_REPLAY_ITEMS:
        raise _error(f"At most {MAX_REPLAY_ITEMS} replay input items are allowed.")
    calls: set[str] = set()
    item_ids: set[str] = set()
    pending: set[str] = set()
    last_was_call = False
    for item in payload.input:
        item_id = getattr(item, "id", None)
        if item_id is not None:
            if not _ID.fullmatch(item_id) or item_id in item_ids:
                raise _error("Replayed item ids must be valid and unique.")
            item_ids.add(item_id)
        if isinstance(item, ResponseInputFunctionCall):
            if pending and not last_was_call:
                raise _error("All pending function calls must have results before a new call group.")
            if item.call_id in calls:
                raise _error(f"Duplicate replayed function call_id: {item.call_id}.")
            _, arguments = _arguments(item.arguments, status=400)
            if item.name in definitions:
                _validate_arguments(arguments, definitions[item.name], status=400)
            calls.add(item.call_id)
            pending.add(item.call_id)
            if len(pending) > MAX_GENERATED_CALLS:
                raise _error(f"At most {MAX_GENERATED_CALLS} calls may share a pending group.")
            last_was_call = True
        elif isinstance(item, ResponseInputFunctionCallOutput):
            if item.call_id not in pending:
                raise _error("Function output must match one unique preceding unanswered call_id.")
            pending.remove(item.call_id)
            last_was_call = False
        else:
            if pending:
                raise _error("All pending function calls require results before the next message.")
            last_was_call = False
    if pending:
        raise _error("All pending function calls require results before generation.")


def adapt_tool_history_item(item) -> dict[str, Any]:
    """Translate one typed output item into the shared provider message shape."""
    if isinstance(item, ResponseInputFunctionCall):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": item.call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": item.arguments},
            }],
        }
    if isinstance(item, ResponseInputFunctionCallOutput):
        return {"role": "tool", "content": item.output, "tool_call_id": item.call_id}
    if isinstance(item, ResponseReplayMessage):
        return {"role": "assistant", "content": "\n".join(part.text for part in item.content)}
    raise TypeError("Expected a replayed Responses output item.")


def merge_adjacent_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one assistant turn's text and parallel calls together for Ollama."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if (
            result
            and message.get("role") == "assistant"
            and message.get("tool_calls")
            and message.get("content") is None
            and result[-1].get("role") == "assistant"
        ):
            result[-1] = {**result[-1], "tool_calls": [*(result[-1].get("tool_calls") or []), *message["tool_calls"]]}
        else:
            result.append(dict(message))
    return result


def provider_tools(payload: ResponseCreateRequest) -> list[dict[str, Any]] | None:
    """Convert flat Responses definitions to Ollama's nested function shape."""
    if payload.tool_choice == "none":
        return None
    return [
        {"type": "function", "function": {key: value for key, value in definition.items() if key != "strict"}}
        for definition in _definitions(payload).values()
    ] or None


def validate_generated_calls(raw_calls: Any, payload: ResponseCreateRequest) -> list[dict[str, Any]]:
    """Validate every provider call before exposing any executable item."""
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise _error("Provider function calls must be an array.", status=502)
    if len(raw_calls) > MAX_GENERATED_CALLS:
        raise _error(f"Provider emitted more than {MAX_GENERATED_CALLS} calls.", status=502)
    if raw_calls and payload.tool_choice == "none":
        raise _error("Provider emitted a function call while tool_choice is none.", status=502)
    if payload.parallel_tool_calls is False and len(raw_calls) > 1:
        raise _error("Provider emitted parallel calls while parallel_tool_calls is false.", status=502)
    definitions = _definitions(payload)
    history_call_ids = set() if isinstance(payload.input, str) else {
        item.call_id for item in payload.input if isinstance(item, ResponseInputFunctionCall)
    }
    history_item_ids = set() if isinstance(payload.input, str) else {
        item.id for item in payload.input if getattr(item, "id", None) is not None
    }
    used_call_ids: set[str] = set(history_call_ids)
    used_item_ids: set[str] = set(history_item_ids)
    result = []
    for raw in raw_calls:
        if not isinstance(raw, dict) or (raw.get("type") not in (None, "function")):
            raise _error("Provider emitted an invalid function call.", status=502)
        function = raw.get("function")
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or function["name"] not in definitions
        ):
            raise _error("Provider called an unadvertised function.", status=502)
        definition = definitions[function["name"]]
        arguments, decoded = _arguments(function.get("arguments"), status=502)
        _validate_arguments(decoded, definition, status=502)
        call_id = raw.get("call_id", raw.get("id"))
        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex}"
        if not isinstance(call_id, str) or not _ID.fullmatch(call_id) or call_id in used_call_ids:
            raise _error("Provider function call ids must be valid and unique.", status=502)
        used_call_ids.add(call_id)
        item_id = f"fc_{uuid.uuid4().hex}"
        if item_id in used_item_ids:
            raise _error("Provider output item id collided with replayed history.", status=502)
        used_item_ids.add(item_id)
        result.append({
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": definition["name"],
            "arguments": arguments,
        })
    return result
