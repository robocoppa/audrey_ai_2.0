"""Server-owned verified runtime for protected native work.

This module does not enable a browser workflow. Internal callers must preserve
stored restrictions and finish file, derived-context and logging enforcement
before offering Private processing to users.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import httpx

from audrey.conversation_titles import ConversationTitleGenerator
from audrey.models.ollama import OllamaClient
from audrey.pipeline.graph import build_graph
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    _model_name,
    processing_scope,
)
from audrey.tools.discovery import ToolRegistry


async def _verify_models(ollama: OllamaClient, policy: ProcessingPolicy) -> None:
    """Read only names/metadata; no prompt, file or embedding input is sent."""
    try:
        installed = await ollama.tags()
        for model in policy.local_models:
            policy.assert_local_model_metadata(
                model, installed_models=installed, metadata=await ollama.show(model),
            )
    except ProcessingPolicyError:
        raise
    except Exception:  # noqa: BLE001 - provider diagnostics must not disclose content
        raise ProcessingPolicyError("private_model_unavailable") from None


async def prepare_processing_policy(
    *,
    ollama_base_url: str,
    model_names: Sequence[str],
    expected_digests: Mapping[str, str] | None = None,
    allowed_tools: frozenset[str] = frozenset(),
    tool_servers: frozenset[str] = frozenset(),
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProcessingPolicy:
    """Verify a server-selected roster and snapshot its installed model digests.

    Callers supply trusted deployment endpoints and operation-specific names,
    never request-supplied provider settings. Optional configured digest pins
    must agree with the installed roster. Registry location hints do not prove
    the presence of local weights. Every later content call verifies again.
    """
    if (
        isinstance(model_names, (str, bytes)) or not isinstance(model_names, Sequence)
        or any(not isinstance(name, str) for name in model_names)
    ):
        raise ProcessingPolicyError("invalid_processing_policy")
    seed = ProcessingPolicy(
        local_models={name: "0" * 64 for name in model_names},
        ollama_base_url=ollama_base_url,
        allowed_tools=allowed_tools,
        tool_servers=tool_servers,
    )
    if len(seed.local_models) != len(model_names):
        raise ProcessingPolicyError("invalid_processing_policy")
    pins: Mapping[str, str] = {}
    if expected_digests is not None:
        if not isinstance(expected_digests, Mapping):
            raise ProcessingPolicyError("invalid_processing_policy")
        if expected_digests:
            pins = ProcessingPolicy(
                local_models=expected_digests, ollama_base_url=seed.ollama_base_url,
            ).local_models
            if not pins.keys() <= seed.local_models.keys():
                raise ProcessingPolicyError("invalid_processing_policy")

    ollama = OllamaClient(seed.ollama_base_url, trust_env=False, transport=transport)
    try:
        try:
            installed = await ollama.tags()
            digests: dict[str, str] = {}
            for model in seed.local_models:
                matches = []
                for row in installed:
                    if not isinstance(row, Mapping):
                        raise ProcessingPolicyError("local_model_metadata_invalid")
                    names = []
                    for key in ("name", "model"):
                        try:
                            names.append(_model_name(row.get(key)))
                        except ProcessingPolicyError:
                            continue
                    if model in names:
                        matches.append(row)
                if not matches:
                    raise ProcessingPolicyError("local_model_not_installed")
                if len(matches) != 1:
                    raise ProcessingPolicyError("local_model_metadata_invalid")
                digests[model] = matches[0].get("digest")
            policy = ProcessingPolicy(
                local_models=digests, ollama_base_url=seed.ollama_base_url,
                allowed_tools=seed.allowed_tools, tool_servers=seed.tool_servers,
            )
            if any(policy.local_models[name] != digest for name, digest in pins.items()):
                raise ProcessingPolicyError("local_model_digest_mismatch")
            # A caller in a protected parent scope cannot prepare a weaker child.
            with processing_scope(policy):
                for model in policy.local_models:
                    policy.assert_local_model_metadata(
                        model, installed_models=installed, metadata=await ollama.show(model),
                    )
            return policy
        except ProcessingPolicyError:
            raise
        except Exception:  # noqa: BLE001 - redact untrusted metadata/transport diagnostics
            raise ProcessingPolicyError("private_model_unavailable") from None
    finally:
        await ollama.aclose()


@dataclass(frozen=True, slots=True)
class _ExecutionState:
    original: Any
    overrides: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.original, name)


@dataclass(frozen=True, slots=True)
class _ExecutionApp:
    original: Any
    state: _ExecutionState

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


@dataclass(frozen=True, slots=True)
class PrivateProcessingRuntime:
    """Exact policy plus owned transports for one complete async operation.

    Each scope owns its clients and graph, so concurrent ordinary/protected
    requests never replace application-wide clients or graph dependencies.
    The policy stays bound until the stream and its cleanup have completed.
    """

    policy: ProcessingPolicy
    ollama_transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)
    tool_transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, ProcessingPolicy):
            raise ProcessingPolicyError("invalid_processing_policy")

    @asynccontextmanager
    async def scope(self, app: Any) -> AsyncIterator[Any]:
        """Yield an isolated execution view only after every local pin verifies."""
        with processing_scope(self.policy):
            async with AsyncExitStack() as stack:
                cfg = app.state.cfg
                timeout = float(cfg.timeouts.get("medium", 180))
                ollama = OllamaClient(
                    self.policy.ollama_base_url, default_timeout_s=timeout,
                    trust_env=False, transport=self.ollama_transport,
                )
                stack.push_async_callback(ollama.aclose)
                http = await stack.enter_async_context(httpx.AsyncClient(
                    timeout=10.0, trust_env=False, follow_redirects=False,
                    transport=self.tool_transport,
                ))
                await _verify_models(ollama, self.policy)
                tools: ToolRegistry = app.state.tools.restrict(self.policy.allowed_tools)
                for spec in tools.specs():
                    self.policy.assert_tool_allowed(spec.name, server_url=spec.server_url)
                    if spec.path != f"/{spec.name}":
                        raise ProcessingPolicyError("private_tool_not_allowed")
                graph = build_graph(
                    cfg, ollama, app.state.registry, app.state.health,
                    app.state.gate, tools, http,
                )
                title = None
                title_cfg = cfg.raw.get("title_generation", {}) or {}
                title_model = str(title_cfg.get("model") or cfg.router.get("model") or "")
                if title_cfg.get("enabled", True) and title_model:
                    try:
                        self.policy.assert_model_allowed(title_model)
                    except ProcessingPolicyError:
                        # An optional label may keep its fallback; it must never
                        # expand the operation's verified model roster.
                        pass
                    else:
                        title = ConversationTitleGenerator(
                            ollama=ollama, registry=app.state.registry,
                            gate=app.state.gate, inflight=app.state.inflight,
                            model=title_model,
                            timeout_s=float(title_cfg.get("timeout_s", 10.0)),
                            max_prompt_chars=int(title_cfg.get("max_prompt_chars", 6000)),
                            max_tokens=int(title_cfg.get("max_tokens", 32)),
                            no_thinking=bool(title_cfg.get("no_thinking", True)),
                        )
                overrides = MappingProxyType({
                    "ollama": ollama, "graph": graph, "archive_http": http,
                    "tools": tools, "conversation_titles": title,
                    "archive_client": None,
                })
                yield _ExecutionApp(app, _ExecutionState(app.state, overrides))


__all__ = ["PrivateProcessingRuntime", "prepare_processing_policy"]
