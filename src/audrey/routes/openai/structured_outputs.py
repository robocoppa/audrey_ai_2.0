"""Bounded JSON Schema admission and output validation for Responses.

Audrey forwards an admitted schema to Ollama's format field and validates the
final model text again before reporting success. The local validator is a
documented subset of JSON Schema; rejecting an unsupported keyword is safer
than accepting a contract Audrey cannot verify.
"""

from __future__ import annotations

import json
import math
from typing import Any

from audrey.routes.openai.schemas import (
    ResponseCreateRequest,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
)

_MAX_SCHEMA_BYTES = 64 * 1024
_MAX_SCHEMA_DEPTH = 12
_MAX_SCHEMA_NODES = 512
_MAX_PROPERTIES = 256
_MAX_ANY_OF = 16
_ALLOWED_TYPES = frozenset({
    "object", "array", "string", "number", "integer", "boolean", "null",
})
_ANNOTATION_KEYS = frozenset({"title", "description", "default", "examples"})
_ALLOWED_KEYS = _ANNOTATION_KEYS | frozenset({
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "properties",
    "required",
    "type",
})


class StructuredOutputError(ValueError):
    """The requested schema or generated JSON violates Audrey's contract."""


def response_json_schema(request: ResponseCreateRequest) -> dict[str, Any] | None:
    """Return an admitted provider schema, or None for ordinary text."""

    if request.text is None:
        return None
    format_ = request.text.format
    if isinstance(format_, ResponseFormatJSONObject):
        raise StructuredOutputError(
            "text.format.type='json_object' is not supported; use json_schema"
        )
    if not isinstance(format_, ResponseFormatJSONSchema):
        return None
    schema = format_.schema_
    _admit_schema(schema, strict=bool(format_.strict))
    return schema


def structured_output_instruction(request: ResponseCreateRequest) -> str | None:
    """Return a compact provider instruction reinforcing the schema contract."""

    schema = response_json_schema(request)
    if schema is None:
        return None
    format_ = request.text.format
    assert isinstance(format_, ResponseFormatJSONSchema)
    description = (
        f" Purpose: {format_.description.strip()}" if format_.description else ""
    )
    return (
        f"Return only one JSON object matching the {format_.name!r} schema."
        f" Do not use Markdown or add prose.{description}\n"
        f"Schema: {json.dumps(schema, separators=(',', ':'), ensure_ascii=False)}"
    )


def validate_structured_output(text: str, request: ResponseCreateRequest) -> Any:
    """Parse and validate a completed structured response; return decoded JSON."""

    schema = response_json_schema(request)
    if schema is None:
        return None
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"structured output is not valid JSON: {exc.msg}"
        ) from exc
    _validate_value(value, schema, root=schema, path="$", depth=0)
    return value


def _reject_json_constant(value: str) -> Any:
    raise StructuredOutputError(
        f"structured output uses non-standard JSON constant: {value}"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(
                f"structured output repeats object property: {key}"
            )
        result[key] = value
    return result


def response_text_config(request: ResponseCreateRequest) -> dict[str, Any]:
    """Serialize the requested text configuration for the response envelope."""

    if request.text is None:
        return {"format": {"type": "text"}}
    return request.text.model_dump(by_alias=True, exclude_none=True)


def _admit_schema(schema: dict[str, Any], *, strict: bool) -> None:
    try:
        encoded = json.dumps(
            schema,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError("text.format.schema must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_SCHEMA_BYTES:
        raise StructuredOutputError(
            f"text.format.schema exceeds {_MAX_SCHEMA_BYTES} bytes"
        )
    counter = [0]
    _admit_node(schema, root=schema, path="$", depth=0, counter=counter, strict=strict)
    if schema.get("type") != "object":
        raise StructuredOutputError("text.format.schema root type must be 'object'")


def _admit_node(
    node: Any,
    *,
    root: dict[str, Any],
    path: str,
    depth: int,
    counter: list[int],
    strict: bool,
) -> None:
    if not isinstance(node, dict):
        raise StructuredOutputError(f"{path} must be a schema object")
    counter[0] += 1
    if counter[0] > _MAX_SCHEMA_NODES:
        raise StructuredOutputError(
            f"text.format.schema exceeds {_MAX_SCHEMA_NODES} schema nodes"
        )
    if depth > _MAX_SCHEMA_DEPTH:
        raise StructuredOutputError(
            f"text.format.schema exceeds depth {_MAX_SCHEMA_DEPTH}"
        )
    unknown = sorted(set(node) - _ALLOWED_KEYS)
    if unknown:
        raise StructuredOutputError(
            f"{path} uses unsupported JSON Schema keyword(s): {', '.join(unknown)}"
        )

    type_ = node.get("type")
    types: list[str] = []
    if isinstance(type_, str):
        types = [type_]
    elif isinstance(type_, list) and type_ and all(
        isinstance(item, str) for item in type_
    ):
        types = list(type_)
    elif type_ is not None:
        raise StructuredOutputError(f"{path}.type must be a string or non-empty list")
    if any(item not in _ALLOWED_TYPES for item in types):
        raise StructuredOutputError(f"{path}.type contains an unsupported JSON type")
    if len(types) != len(set(types)):
        raise StructuredOutputError(f"{path}.type contains duplicates")

    ref = node.get("$ref")
    if ref is not None:
        _resolve_ref(root, ref, path=path)

    defs = node.get("$defs")
    if defs is not None:
        if not isinstance(defs, dict):
            raise StructuredOutputError(f"{path}.$defs must be an object")
        for name, child in defs.items():
            if not isinstance(name, str) or not name:
                raise StructuredOutputError(
                    f"{path}.$defs keys must be non-empty strings"
                )
            _admit_node(
                child,
                root=root,
                path=f"{path}.$defs.{name}",
                depth=depth + 1,
                counter=counter,
                strict=strict,
            )

    any_of = node.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not 1 <= len(any_of) <= _MAX_ANY_OF:
            raise StructuredOutputError(
                f"{path}.anyOf must contain 1 to {_MAX_ANY_OF} schemas"
            )
        for index, child in enumerate(any_of):
            _admit_node(
                child,
                root=root,
                path=f"{path}.anyOf[{index}]",
                depth=depth + 1,
                counter=counter,
                strict=strict,
            )

    properties = node.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise StructuredOutputError(f"{path}.properties must be an object")
        if len(properties) > _MAX_PROPERTIES:
            raise StructuredOutputError(
                f"{path}.properties exceeds {_MAX_PROPERTIES} entries"
            )
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise StructuredOutputError(
                    f"{path}.properties keys must be non-empty strings"
                )
            _admit_node(
                child,
                root=root,
                path=f"{path}.properties.{name}",
                depth=depth + 1,
                counter=counter,
                strict=strict,
            )
        required = node.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise StructuredOutputError(f"{path}.required must be a string list")
        if len(required) != len(set(required)):
            raise StructuredOutputError(f"{path}.required contains duplicates")
        missing = sorted(set(required) - set(properties))
        if missing:
            raise StructuredOutputError(
                f"{path}.required names unknown properties: {', '.join(missing)}"
            )
        if strict:
            if set(required) != set(properties):
                raise StructuredOutputError(
                    f"{path} strict object schemas must require every property"
                )
            if node.get("additionalProperties") is not False:
                raise StructuredOutputError(
                    f"{path} strict object schemas require additionalProperties=false"
                )
    elif "required" in node:
        raise StructuredOutputError(f"{path}.required needs properties")

    additional = node.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise StructuredOutputError(f"{path}.additionalProperties must be boolean")
    if strict and "object" in types and additional is not False:
        raise StructuredOutputError(
            f"{path} strict object schemas require additionalProperties=false"
        )

    items = node.get("items")
    if items is not None:
        _admit_node(
            items,
            root=root,
            path=f"{path}.items",
            depth=depth + 1,
            counter=counter,
            strict=strict,
        )

    enum = node.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise StructuredOutputError(f"{path}.enum must be a non-empty list")
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        value = node.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise StructuredOutputError(
                f"{path}.{key} must be a non-negative integer"
            )
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        value = node.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise StructuredOutputError(f"{path}.{key} must be a finite number")


def _resolve_ref(root: dict[str, Any], ref: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise StructuredOutputError(f"{path}.$ref must be a local JSON pointer")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise StructuredOutputError(f"{path}.$ref does not resolve: {ref}")
        current = current[part]
    if not isinstance(current, dict):
        raise StructuredOutputError(
            f"{path}.$ref does not resolve to a schema: {ref}"
        )
    return current


def _validate_value(
    value: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    path: str,
    depth: int,
) -> None:
    if depth > _MAX_SCHEMA_DEPTH * 2:
        raise StructuredOutputError(
            f"structured output exceeds validation depth at {path}"
        )
    if "$ref" in schema:
        _validate_value(
            value,
            _resolve_ref(root, schema["$ref"], path=path),
            root=root,
            path=path,
            depth=depth + 1,
        )
    if "anyOf" in schema:
        failures = 0
        for branch in schema["anyOf"]:
            try:
                _validate_value(
                    value, branch, root=root, path=path, depth=depth + 1
                )
                break
            except StructuredOutputError:
                failures += 1
        if failures == len(schema["anyOf"]):
            raise StructuredOutputError(
                f"structured output {path} matches no anyOf branch"
            )

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise StructuredOutputError(
            f"structured output {path} does not match const"
        )
    if "enum" in schema and not any(
        _json_equal(value, item) for item in schema["enum"]
    ):
        raise StructuredOutputError(
            f"structured output {path} is not an allowed enum value"
        )

    expected = schema.get("type")
    types = [expected] if isinstance(expected, str) else list(expected or [])
    if types and not any(_matches_type(value, item) for item in types):
        raise StructuredOutputError(
            f"structured output {path} must have type {' or '.join(types)}"
        )

    if isinstance(value, dict) and ("object" in types or "properties" in schema):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        missing = [name for name in required if name not in value]
        if missing:
            raise StructuredOutputError(
                f"structured output {path} is missing: {', '.join(missing)}"
            )
        extras = set(value) - set(properties)
        if extras and schema.get("additionalProperties") is False:
            raise StructuredOutputError(
                f"structured output {path} has unexpected properties: "
                f"{', '.join(sorted(extras))}"
            )
        _check_length(value, schema, path, "Properties")
        for name, child in properties.items():
            if name in value:
                _validate_value(
                    value[name],
                    child,
                    root=root,
                    path=f"{path}.{name}",
                    depth=depth + 1,
                )
    elif isinstance(value, list) and ("array" in types or "items" in schema):
        _check_length(value, schema, path, "Items")
        items = schema.get("items")
        if items is not None:
            for index, item in enumerate(value):
                _validate_value(
                    item,
                    items,
                    root=root,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                )
    elif isinstance(value, str):
        _check_length(value, schema, path, "Length")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number(value, schema, path)


def _matches_type(value: Any, type_: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }[type_]


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _check_length(
    value: Any,
    schema: dict[str, Any],
    path: str,
    suffix: str,
) -> None:
    minimum = schema.get(f"min{suffix}")
    maximum = schema.get(f"max{suffix}")
    if minimum is not None and len(value) < minimum:
        raise StructuredOutputError(
            f"structured output {path} is shorter than {minimum}"
        )
    if maximum is not None and len(value) > maximum:
        raise StructuredOutputError(
            f"structured output {path} is longer than {maximum}"
        )


def _check_number(
    value: int | float,
    schema: dict[str, Any],
    path: str,
) -> None:
    checks = (
        ("minimum", lambda limit: value >= limit),
        ("maximum", lambda limit: value <= limit),
        ("exclusiveMinimum", lambda limit: value > limit),
        ("exclusiveMaximum", lambda limit: value < limit),
    )
    for key, check in checks:
        if key in schema and not check(schema[key]):
            raise StructuredOutputError(
                f"structured output {path} violates {key}"
            )


__all__ = [
    "StructuredOutputError",
    "response_json_schema",
    "response_text_config",
    "structured_output_instruction",
    "validate_structured_output",
]
