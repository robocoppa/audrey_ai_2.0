"""Revocable native chat file creation bound to one authenticated live run."""
from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from audrey.identity import Principal
from audrey.processing_policy import current_processing_policy
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec, _resolve_refs

_MAX_FILES = 8
_FILE_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


class NativeFileToolUnavailableError(RuntimeError):
    """Content-free refusal when no authorized native creation scope exists."""


@dataclass(slots=True)
class _Binding:
    app: Any
    principal: Principal
    conversation_id: str
    run_id: str
    policy: Any
    files: list[dict]
    active: bool = True
    pending: int = 0


_BINDING: ContextVar[_Binding | None] = ContextVar("audrey_native_files", default=None)


@contextmanager
def native_file_scope(*, app: Any, principal: Principal, conversation_id: str,
                      run_id: str, files: list[dict]) -> Iterator[None]:
    """Bind server authority; inherited callbacks expire on scope exit."""
    if (not isinstance(principal, Principal) or principal.status != "active"
            or not principal.storage_namespace or not conversation_id or not run_id
            or _BINDING.get() is not None):
        raise NativeFileToolUnavailableError("Native file creation is unavailable.")
    binding = _Binding(app, principal, conversation_id, run_id, current_processing_policy(), files)
    token = _BINDING.set(binding)
    try:
        yield
    finally:
        binding.active = False
        _BINDING.reset(token)


def native_file_handler(*, user_id: str | None):
    binding = _BINDING.get()

    def guard() -> None:
        if (binding is None or not binding.active or _BINDING.get() is not binding
                or current_processing_policy() is not binding.policy
                or user_id != binding.principal.storage_namespace):
            raise NativeFileToolUnavailableError("Native file creation is unavailable.")
        if binding.policy is not None:
            binding.policy.assert_tool_allowed("create_file")

    guard()

    async def create(arguments: dict) -> dict:
        guard()
        from audrey.generated_files import create_generated_file

        if len(binding.files) + binding.pending >= _MAX_FILES:
            return {"error": "generated_file_limit", "detail": "At most eight files can be created in one reply."}
        binding.pending += 1
        try:
            result = await create_generated_file(
                binding.app, binding.principal, binding.conversation_id, binding.run_id, arguments,
            )
        finally:
            binding.pending -= 1
        guard()
        if isinstance(result, dict) and result.get("error"):
            return result
        if not _valid_file(result):
            raise NativeFileToolUnavailableError("The created file response is unavailable.")
        if not any(item["file_id"] == result["file_id"] for item in binding.files):
            binding.files.append(dict(result))
        return result

    return create


def _valid_file(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    file_id = item.get("file_id")
    return (isinstance(file_id, str) and _FILE_ID.fullmatch(file_id) is not None
            and isinstance(item.get("filename"), str) and bool(item["filename"])
            and item.get("download_url") == f"/api/files/{file_id}/download")


def create_file_spec(server_url: str = "http://native.invalid") -> ToolSpec:
    from audrey.file_rendering import CreateFileRequest

    schema = CreateFileRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    parameters = _resolve_refs(schema, schema.get("$defs", {}))
    parameters.pop("$defs", None)
    declaration = TOOL_DECLARATIONS["create_file"]
    return ToolSpec(
        name="create_file",
        description=("Create a downloadable file owned by this user: text, Markdown, CSV, JSON, Word DOCX or Excel XLSX. "
                     "Supply filename and structured content; the server saves it in My files and the current Project. "
                     "Do not invent a download link or filesystem path. Repeating identical content in this reply reuses the file. "
                     "Generated content is output, not independent evidence. This does not save or validate a Tax review case."),
        parameters=parameters, server_url=server_url.rstrip("/"), path="/create_file",
        user_scope=declaration.user_scope, dependencies=declaration.dependencies, in_process_only=True,
    )


def with_create_file(tools: ToolRegistry, *, server_url: str = "http://native.invalid") -> ToolRegistry:
    """Add a native side capability without mutating shared discovery."""
    return ToolRegistry({**tools.by_name, "create_file": create_file_spec(server_url)})


def append_generated_file_links(content: str, files: list[dict], *, max_chars: int = 64000) -> str:
    """Persist only server-approved links, even if the model omits them."""
    links, seen = [], set()
    for item in files[:_MAX_FILES]:
        if not _valid_file(item) or item["file_id"] in seen:
            continue
        seen.add(item["file_id"])
        # A raw path or an off-origin URL containing it is not a usable
        # authenticated chat download. Only an existing exact link replaces
        # the server's canonical link.
        if re.search(r"\]\(" + re.escape(item["download_url"]) + r"\)", content):
            continue
        filename = re.sub(r"[\x00-\x1f\x7f]", "", item["filename"])[:240]
        filename = filename.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        links.append(f"[Download {filename}]({item['download_url']})")
    if not links:
        return content
    suffix = "\n\n" + "\n\n".join(links)
    if len(content) + len(suffix) > max_chars:
        marker = "\n\n[Reply shortened to retain file downloads.]"
        content = content[:max(0, max_chars - len(suffix) - len(marker))] + marker
    return content + suffix


__all__ = ["NativeFileToolUnavailableError", "append_generated_file_links", "create_file_spec",
           "native_file_handler", "native_file_scope", "with_create_file"]
