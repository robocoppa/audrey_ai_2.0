"""Safe projection of real ReAct activity onto Audrey run events.

Tool dispatch needs the complete arguments and result body. Browser-facing run
events do not. This boundary keeps useful lifecycle, timing, and source data
while preventing injected identity fields, stored memory, fetched documents,
and provider errors from becoming transient UI payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from audrey.pipeline.run_events import RunEventEmitter
from audrey.tools.dispatch import ToolResult

_REDACTED = "[redacted]"
_MAX_ARGUMENT_STRING = 500
_MAX_ARGUMENT_ITEMS = 20
_MAX_SOURCE_TITLE = 500
_MAX_SOURCE_URL = 2_048
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Values omitted here still reveal their key, but never their content. This is
# intentionally an allow-list: a newly added tool cannot accidentally expose a
# secret-bearing argument before its UI projection has been reviewed.
_VISIBLE_ARGUMENTS: dict[str, frozenset[str]] = {
    "web_search": frozenset({"query", "count"}),
    "web_fetch": frozenset({"url", "max_chars"}),
    "kb_search": frozenset({"query", "top_k", "file_ids"}),
    "kb_image_search": frozenset({"query", "top_k", "file_ids"}),
    "memory_store": frozenset({"key"}),
    "memory_recall": frozenset({"key"}),
    "memory_search": frozenset({"query", "top_k"}),
    "chat_history_search": frozenset({"query", "top_k"}),
    "list_my_files": frozenset(),
    "get_file_text": frozenset({"filename", "artifact", "page", "max_chars"}),
}


class RunEventToolObserver:
    """Emit owner-visible tool lifecycle and source events without raw bodies."""

    def __init__(self, emitter: RunEventEmitter) -> None:
        self._emitter = emitter
        self._seen_source_urls: set[str] = set()

    def started(self, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "?")
        event_call_id = f"tool_{uuid.uuid4().hex}"
        self._emitter.tool_started(event_call_id, name=name)
        self._emitter.tool_arguments(
            event_call_id,
            arguments=_observable_arguments(name, function.get("arguments")),
        )
        return event_call_id

    def finished(
        self,
        event_call_id: str,
        result: ToolResult,
        *,
        sources: Iterable[dict[str, str]] = (),
    ) -> None:
        source_rows = tuple(sources)
        summary = {
            "status": "failed" if result.is_error else "succeeded",
            "elapsedMs": max(0, round(result.elapsed_s * 1_000)),
            "contentBytes": len(result.content.encode("utf-8")),
            "sourceCount": len(source_rows),
        }
        self._emitter.tool_finished(
            event_call_id,
            status="failed" if result.is_error else "succeeded",
            result=summary,
            error=_observable_error_code(result) if result.is_error else "",
        )
        if result.is_error:
            return
        for source in source_rows:
            self._source_observed(source, fallback_type=result.name)

    def interrupted(self, event_call_id: str, *, code: str) -> None:
        self._emitter.tool_finished(
            event_call_id,
            status="failed",
            result={"status": "failed"},
            error=code,
        )

    def _source_observed(
        self,
        source: dict[str, str],
        *,
        fallback_type: str,
    ) -> None:
        raw_url = str(source.get("url") or "").strip()[:_MAX_SOURCE_URL]
        url = str(_observable_url(raw_url))
        if not url or url in self._seen_source_urls:
            return
        self._seen_source_urls.add(url)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        self._emitter.source_observed(
            f"src_{digest}",
            title=str(source.get("title") or "").strip()[:_MAX_SOURCE_TITLE],
            url=url,
            source_type=str(source.get("tool") or fallback_type)[:100],
        )


def _observable_arguments(name: str, raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {"_status": "arguments_not_json"}
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        return {"_status": "arguments_not_object"}

    visible = _VISIBLE_ARGUMENTS.get(name, frozenset())
    projected: dict[str, Any] = {}
    for key, value in raw_arguments.items():
        clean_key = str(key)[:100]
        if clean_key not in visible:
            projected[clean_key] = _REDACTED
        elif clean_key == "url":
            projected[clean_key] = _observable_url(value)
        else:
            projected[clean_key] = _bounded_value(value)
    return projected


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_ARGUMENT_STRING:
            return value
        return value[:_MAX_ARGUMENT_STRING] + "…"
    if depth >= 2:
        return _REDACTED
    if isinstance(value, list | tuple):
        return [
            _bounded_value(item, depth=depth + 1)
            for item in value[:_MAX_ARGUMENT_ITEMS]
        ]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_ARGUMENT_ITEMS]
        }
    return str(type(value).__name__)


def _observable_url(value: Any) -> Any:
    if not isinstance(value, str):
        return _bounded_value(value)
    value = value[:_MAX_ARGUMENT_STRING]
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.split("?", 1)[0].split("#", 1)[0]
    # Credentials, query tokens, and fragments are dispatch details, not UI
    # state. Preserve only the navigational origin/path for the activity card.
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _observable_error_code(result: ToolResult) -> str:
    try:
        body = json.loads(result.content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return "tool_failed"
    value = body.get("error") if isinstance(body, dict) else None
    code = str(value or "")
    return code if _SAFE_ERROR_CODE.fullmatch(code) else "tool_failed"


__all__ = ["RunEventToolObserver"]
