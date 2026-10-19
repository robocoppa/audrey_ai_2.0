"""Inactive-by-default local processing guards shared by model/tool boundaries.

This foundation is deliberately not a user-facing privacy mode. It covers only
calls made in a server-owned scope; durable file/conversation restrictions and
cross-process propagation must be implemented before offering that guarantee.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# These tools neither call another model nor write material into a general
# memory/history index. Model-using tools require explicit policy propagation
# and protected-content reuse rules before they can join this subset.
LOCAL_MODEL_FREE_TOOLS = frozenset({
    "calculate", "analyze_table", "reference_lookup", "read_reference",
    "list_my_files", "get_file_text",
})

_MODEL_PART = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TOOL = re.compile(r"[a-z][a-z0-9_]{0,99}\Z")
_ERRORS = {
    "invalid_processing_policy": "The local processing policy is invalid.",
    "processing_policy_downgrade": "A protected processing scope cannot be weakened.",
    "local_model_not_allowed": "This model is not permitted for local processing.",
    "local_model_not_installed": "The permitted local model could not be verified as installed.",
    "local_model_digest_mismatch": "The installed model does not match the permitted local model.",
    "local_model_metadata_invalid": "The model metadata does not establish local weights.",
    "local_model_remote": "Remote model execution is not permitted in this processing scope.",
    "local_model_endpoint_blocked": "This model endpoint is not permitted for local processing.",
    "local_tool_not_allowed": "This tool is not permitted in this processing scope.",
    "local_tool_endpoint_blocked": "This tool endpoint is not permitted for local processing.",
    "private_processing_unavailable": "Private processing is not available yet for this operation.",
    "private_model_unavailable": "The local model could not be verified for this processing scope.",
    "private_transport_not_allowed": "This transport is not permitted for local processing.",
    "private_tool_not_allowed": "This tool operation is not permitted for local processing.",
}


class ProcessingPolicyError(RuntimeError):
    """A stable, content-free refusal safe to show or record."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERRORS else "invalid_processing_policy"
        super().__init__(_ERRORS[self.code])


def _model_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ProcessingPolicyError("local_model_not_allowed")
    if value.count(":") > 1:
        raise ProcessingPolicyError("local_model_not_allowed")
    name, separator, tag = value.partition(":")
    tag = tag if separator else "latest"
    if (
        not all(_MODEL_PART.fullmatch(part) for part in name.split("/"))
        or not _MODEL_PART.fullmatch(tag)
        or tag == "cloud" or tag.endswith("-cloud")
    ):
        raise ProcessingPolicyError("local_model_not_allowed")
    return f"{name}:{tag}"


def _digest(value: object) -> str:
    if not isinstance(value, str):
        raise ProcessingPolicyError("invalid_processing_policy")
    result = value.removeprefix("sha256:")
    if not _DIGEST.fullmatch(result):
        raise ProcessingPolicyError("invalid_processing_policy")
    return result


def _endpoint(value: object) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 2048
        or any(character.isspace() for character in value)
        or "?" in value or "#" in value or "\\" in value
    ):
        raise ProcessingPolicyError("invalid_processing_policy")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"} or not hostname
            or parsed.username is not None or parsed.password is not None
            or not hostname.isascii()
            or not re.fullmatch(r"[A-Za-z0-9._~/-]*", parsed.path)
            or "//" in parsed.path
            or any(segment in {".", ".."} for segment in parsed.path.split("/"))
        ):
            raise ValueError
        if ":" in hostname:
            hostname = f"[{ipaddress.IPv6Address(hostname).compressed}]"
        elif not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", hostname):
            raise ValueError
        default_port = 80 if parsed.scheme == "http" else 443
        authority = hostname if port in {None, default_port} else f"{hostname}:{port}"
        return urlunsplit((parsed.scheme, authority, parsed.path.rstrip("/"), "", ""))
    except (ValueError, TypeError):
        raise ProcessingPolicyError("invalid_processing_policy") from None


@dataclass(frozen=True, slots=True)
class ProcessingPolicy:
    """Operator-supplied exact local identities and trusted service endpoints.

    Endpoint trust is explicit: this class does not infer a local service from
    private IPs or a model name. An endpoint's network isolation remains an
    operator deployment responsibility. Missing model metadata fails closed.
    """

    local_models: Mapping[str, str]
    ollama_base_url: str
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    tool_servers: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.local_models, Mapping) or not self.local_models:
                raise ProcessingPolicyError("invalid_processing_policy")
            models: dict[str, str] = {}
            for name, digest in self.local_models.items():
                canonical = _model_name(name)
                if canonical in models:
                    raise ProcessingPolicyError("invalid_processing_policy")
                models[canonical] = _digest(digest)
            if (
                not isinstance(self.allowed_tools, (frozenset, set))
                or not isinstance(self.tool_servers, (frozenset, set))
                or any(
                    not isinstance(name, str) or not _TOOL.fullmatch(name)
                    or name not in LOCAL_MODEL_FREE_TOOLS for name in self.allowed_tools
                )
            ):
                raise ProcessingPolicyError("invalid_processing_policy")
            servers = frozenset(_endpoint(server) for server in self.tool_servers)
            if self.allowed_tools and not servers:
                raise ProcessingPolicyError("invalid_processing_policy")
            object.__setattr__(self, "local_models", MappingProxyType(models))
            object.__setattr__(self, "ollama_base_url", _endpoint(self.ollama_base_url))
            object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
            object.__setattr__(self, "tool_servers", servers)
        except ProcessingPolicyError:
            raise ProcessingPolicyError("invalid_processing_policy") from None

    def assert_model_allowed(self, model: str) -> None:
        if _model_name(model) not in self.local_models:
            raise ProcessingPolicyError("local_model_not_allowed")

    def assert_ollama_endpoint(self, base_url: str) -> None:
        try:
            matches = _endpoint(base_url) == self.ollama_base_url
        except ProcessingPolicyError:
            matches = False
        if not matches:
            raise ProcessingPolicyError("local_model_endpoint_blocked")

    def assert_local_model_metadata(
        self,
        model: str,
        *,
        installed_models: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> None:
        self.assert_model_allowed(model)
        canonical = _model_name(model)
        if not isinstance(installed_models, (list, tuple)):
            raise ProcessingPolicyError("local_model_metadata_invalid")
        matches: list[Mapping[str, Any]] = []
        for row in installed_models:
            if not isinstance(row, Mapping):
                raise ProcessingPolicyError("local_model_metadata_invalid")
            names = [row[key] for key in ("name", "model") if key in row]
            try:
                normalized = [_model_name(name) for name in names]
            except ProcessingPolicyError:
                # Unrelated cloud entries are normal in a shared Ollama catalog.
                if model in names or canonical in names:
                    raise ProcessingPolicyError("local_model_metadata_invalid") from None
                continue
            if canonical in normalized:
                if any(name != canonical for name in normalized):
                    raise ProcessingPolicyError("local_model_metadata_invalid")
                matches.append(row)
        if not matches:
            raise ProcessingPolicyError("local_model_not_installed")
        if len(matches) != 1 or not isinstance(metadata, Mapping):
            raise ProcessingPolicyError("local_model_metadata_invalid")
        row = matches[0]
        for candidate in (row, metadata):
            if any(
                key in candidate and candidate[key] is not False and candidate[key] is not None
                for key in ("remote_model", "remote_host")
            ):
                raise ProcessingPolicyError("local_model_remote")
        try:
            actual_digest = _digest(row.get("digest"))
        except ProcessingPolicyError:
            raise ProcessingPolicyError("local_model_metadata_invalid") from None
        if actual_digest != self.local_models[canonical]:
            raise ProcessingPolicyError("local_model_digest_mismatch")
        model_info = metadata.get("model_info")
        details = metadata.get("details")
        if (
            not isinstance(model_info, Mapping) or not model_info
            or not isinstance(model_info.get("general.architecture"), str)
            or not model_info["general.architecture"].strip()
            or type(model_info.get("general.parameter_count")) is not int
            or model_info["general.parameter_count"] <= 0
            or not isinstance(details, Mapping) or details.get("format") != "gguf"
        ):
            raise ProcessingPolicyError("local_model_metadata_invalid")

    def assert_tool_allowed(self, name: str, *, server_url: str | None = None) -> None:
        if not isinstance(name, str) or name not in self.allowed_tools:
            raise ProcessingPolicyError("local_tool_not_allowed")
        if server_url is not None:
            try:
                permitted = _endpoint(server_url) in self.tool_servers
            except ProcessingPolicyError:
                permitted = False
            if not permitted:
                raise ProcessingPolicyError("local_tool_endpoint_blocked")


_CURRENT_POLICY: ContextVar[ProcessingPolicy | None] = ContextVar("audrey_processing_policy", default=None)


def current_processing_policy() -> ProcessingPolicy | None:
    return _CURRENT_POLICY.get()


@contextmanager
def processing_scope(policy: ProcessingPolicy | None) -> Iterator[None]:
    """Bind a task-local policy, allowing nested scopes only to tighten it.

    Keep this scope active throughout asynchronous stream iteration, not just
    while creating its generator. No request flag activates this foundation.
    Metadata verification trusts the operator-managed Ollama service; it is not
    atomic protection against an administrator changing model aliases.
    """
    current = current_processing_policy()
    if policy is not None and not isinstance(policy, ProcessingPolicy):
        raise ProcessingPolicyError("invalid_processing_policy")
    if current is not None and (
        policy is None
        or policy.ollama_base_url != current.ollama_base_url
        or any(current.local_models.get(name) != digest for name, digest in policy.local_models.items())
        or not policy.allowed_tools <= current.allowed_tools
        or not policy.tool_servers <= current.tool_servers
    ):
        raise ProcessingPolicyError("processing_policy_downgrade")
    token = _CURRENT_POLICY.set(policy)
    try:
        yield
    finally:
        _CURRENT_POLICY.reset(token)
