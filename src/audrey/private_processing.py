"""Server-owned verified runtime for protected native work.

Native Private operations use this runtime for verified local models, private
documents and owned reference tools. Unsupported operations remain unavailable;
ordinary workflow clients and configuration are never mutated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import httpx

from audrey.conversation_titles import ConversationTitleGenerator
from audrey.kb.embed import text_embedder
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.graph import build_graph
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    _model_name,
    processing_scope,
)
from audrey.references.embedding import reference_embedder
from audrey.references.search import ReferenceSearchService
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


def private_configuration(cfg: Any, policy: ProcessingPolicy) -> Any:
    """Copy routing configuration, removing every unverified generation target."""
    isolated = copy(cfg)
    raw = deepcopy(cfg.raw)
    registry = raw.get("model_registry", {})
    def permitted(entry):
        try:
            return _model_name(entry["name"]) in policy.local_models
        except ProcessingPolicyError:
            return False

    raw["model_registry"] = {
        task: [{**entry, "location": "local"} for entry in entries if permitted(entry)]
        for task, entries in registry.items()
    }
    # Local's panel is the source of truth for protected deep work. The generic
    # pool also covers fast-path escalation; no cloud pool can become a fallback.
    if "deep_panel_local" in raw:
        raw["deep_panel"] = deepcopy(raw["deep_panel_local"])
    agentic = raw.setdefault("agentic", {})
    agentic.setdefault("memory", {})["enabled"] = False
    for key in ("debug_context_trace", "debug_research_trace", "debug_panel_drafts"):
        agentic[key] = False
    raw.setdefault("debug", {}).update(log_incoming_payload=False, log_incoming_payload_content=False)
    raw.setdefault("complexity", {})["log_breakdown"] = False
    isolated._merged = raw
    return isolated


async def prepare_native_private_runtime(app: Any) -> PrivateProcessingRuntime:
    """Prepare the explicit native Private workflow from deployment settings."""
    cfg = app.state.cfg
    if (cfg.raw.get("private_processing") or {}).get("enabled") is not True:
        raise ProcessingPolicyError("private_processing_unavailable")
    names = {str(cfg.router.get("model") or "")}
    titles = cfg.raw.get("title_generation") or {}
    if titles.get("enabled", True):
        names.add(str(titles.get("model") or cfg.router.get("model") or ""))
    pools = cfg.raw.get("deep_panel_local") or {}
    for pool in pools.values():
        names.update(pool.get("workers") or [])
        names.update(str(pool.get(key) or "") for key in ("synthesizer", "fallback_synth"))
    summary = ((cfg.raw.get("kb") or {}).get("video") or {}).get("summarise_model")
    if summary:
        names.add(str(summary))
    profile = cfg.raw.get("text_embedding") or {}
    if not pools or not profile.get("model") or not profile.get("digest"):
        raise ProcessingPolicyError("private_processing_unavailable")
    names.add(profile["model"])
    names.discard("")
    from audrey.tools.private_handlers import SUPPORTED_PRIVATE_TOOL_NAMES
    available = frozenset(app.state.tools.names()) & SUPPORTED_PRIVATE_TOOL_NAMES
    specs = app.state.tools.restrict(available).specs()
    policy = await prepare_processing_policy(
        ollama_base_url=cfg.env.ollama_host,
        model_names=sorted(names), expected_digests={profile["model"]: profile["digest"]},
        allowed_tools=available, tool_servers=frozenset(spec.server_url for spec in specs),
    )
    return PrivateProcessingRuntime(policy)


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
                cfg = private_configuration(app.state.cfg, self.policy)
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
                registry = ModelRegistry(cfg)
                graph = build_graph(
                    cfg, ollama, registry, app.state.health,
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
                            ollama=ollama, registry=registry,
                            gate=app.state.gate, inflight=app.state.inflight,
                            model=title_model,
                            timeout_s=float(title_cfg.get("timeout_s", 10.0)),
                            max_prompt_chars=int(title_cfg.get("max_prompt_chars", 6000)),
                            max_tokens=int(title_cfg.get("max_tokens", 32)),
                            no_thinking=bool(title_cfg.get("no_thinking", True)),
                        )
                embedder = text_embedder(ollama, cfg.raw) if "text_embedding" in cfg.raw else None
                reference_search = None
                original_search = getattr(app.state, "reference_search", None)
                if original_search is not None and embedder is not None:
                    reference_search = ReferenceSearchService(
                        original_search.registry, original_search.releases, original_search.qdrant,
                        reference_embedder(ollama, cfg.raw),
                        config={"min_dense_score": original_search.min_dense_score,
                                "min_term_overlap": original_search.min_term_overlap},
                    )
                overrides = MappingProxyType({
                    "cfg": cfg, "registry": registry, "text_embedder": embedder,
                    "reference_search": reference_search,
                    "ollama": ollama, "graph": graph, "archive_http": http,
                    "tools": tools, "conversation_titles": title,
                    "archive_client": None,
                })
                yield _ExecutionApp(app, _ExecutionState(app.state, overrides))


__all__ = ["PrivateProcessingRuntime", "prepare_processing_policy", "prepare_native_private_runtime"]
