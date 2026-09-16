"""Deployment-defined models filtered by Audrey-owned access groups.

The YAML defines what this deployment can run. SQLite policies define who may
see and invoke each entry. Keeping those responsibilities separate lets an
administrator change access without turning the database into a second model
registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal

from audrey.app_state import ApplicationStore, ModelAccessPolicy
from audrey.config import Config
from audrey.identity import Principal
from audrey.models.ollama import OllamaError
from audrey.models.registry import ModelRegistry

ModelKind = Literal["workflow", "direct"]
ModelAudience = Literal["users", "testers", "admins"]
InventorySource = Literal["ollama", "configuration"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServedModel:
    """One stable native model selection and its execution target."""

    id: str
    label: str
    description: str
    kind: ModelKind
    mode: str
    protocol_model: str
    concrete_model: str
    presentation: str
    capabilities: tuple[str, ...]
    enabled: bool
    audience: ModelAudience
    num_ctx: int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelInventory:
    """Runtime model definitions plus the source used to build them."""

    models: tuple[ServedModel, ...]
    source: InventorySource
    warning: str = ""


_WORKFLOWS: tuple[ServedModel, ...] = (
    ServedModel(
        id="auto",
        label="Auto",
        description="Audrey chooses the right path for the request.",
        kind="workflow",
        mode="auto",
        protocol_model="audrey_auto",
        concrete_model="",
        presentation="auto",
        capabilities=("text", "files", "tools"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="fast",
        label="Fast",
        description="A quick answer from Audrey's fast path.",
        kind="workflow",
        mode="fast",
        protocol_model="audrey_fast",
        concrete_model="",
        presentation="fast",
        capabilities=("text", "files", "tools"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="deep",
        label="Deep",
        description="Planning, a model panel, and synthesis.",
        kind="workflow",
        mode="deep",
        protocol_model="audrey_deep",
        concrete_model="",
        presentation="deep",
        capabilities=("text", "files", "tools"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="research",
        label="Research",
        description="Grounded research with verification and sources.",
        kind="workflow",
        mode="research",
        protocol_model="audrey_research",
        concrete_model="",
        presentation="research",
        capabilities=("text", "files", "tools", "web"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="local",
        label="Local",
        description="Audrey's deep workflow using local models only.",
        kind="workflow",
        mode="local",
        protocol_model="audrey_local",
        concrete_model="",
        presentation="local",
        capabilities=("text", "files", "tools"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="cloud",
        label="Cloud",
        description="Audrey's deep workflow using cloud models only.",
        kind="workflow",
        mode="cloud",
        protocol_model="audrey_cloud",
        concrete_model="",
        presentation="cloud",
        capabilities=("text", "files", "tools"),
        enabled=True,
        audience="users",
    ),
    ServedModel(
        id="video",
        label="Video",
        description="Audrey's workflow for video questions.",
        kind="workflow",
        mode="video",
        protocol_model="audrey_video",
        concrete_model="",
        presentation="video",
        capabilities=("text", "files", "tools", "video"),
        enabled=True,
        audience="users",
    ),
)


def configured_models(cfg: Config) -> tuple[ServedModel, ...]:
    """Build the catalog from static workflows and allowed direct models."""

    native = cfg.raw.get("native_models") or {}
    direct_defaults = native.get("direct_defaults") or {}
    overrides = native.get("entries") or {}
    models = list(_WORKFLOWS)
    passthrough = cfg.raw.get("passthrough") or {}
    if not passthrough.get("enabled", False):
        return tuple(models)

    registry = ModelRegistry(cfg)
    for concrete in passthrough.get("allowed_models") or ():
        models.append(
            _direct_model(
                str(concrete),
                direct_defaults=direct_defaults,
                overrides=overrides,
                registry=registry,
            )
        )
    return tuple(models)


async def discover_models(cfg: Config, ollama: Any | None) -> ModelInventory:
    """Prefer Ollama's installed tags over the compatibility allowlist.

    `passthrough.allowed_models` remains the explicit contract for `/v1`
    compatibility clients. The native application is different: it discovers
    the models this Ollama instance can actually serve, then applies Audrey's
    own enabled/audience policies. If discovery is unavailable, the configured
    catalog is retained so Audrey workflows and known recovery controls remain
    usable.
    """

    configured = configured_models(cfg)
    passthrough = cfg.raw.get("passthrough") or {}
    tags = getattr(ollama, "tags", None)
    if not passthrough.get("enabled", False) or not callable(tags):
        return ModelInventory(configured, "configuration")
    try:
        installed = await tags()
    except OllamaError as exc:
        log.warning("native model discovery failed; using configured fallback: %s", exc)
        return ModelInventory(
            configured,
            "configuration",
            "Ollama model discovery is unavailable; showing configured fallback models.",
        )

    native = cfg.raw.get("native_models") or {}
    direct_defaults = native.get("direct_defaults") or {}
    overrides = native.get("entries") or {}
    registry = ModelRegistry(cfg)
    names = _installed_model_names(installed)
    workflows = tuple(model for model in configured if model.kind == "workflow")
    directs = tuple(
        _direct_model(
            concrete,
            direct_defaults=direct_defaults,
            overrides=overrides,
            registry=registry,
        )
        for concrete in names
    )
    warning = "" if names else "Ollama reported no installed models."
    return ModelInventory(workflows + directs, "ollama", warning)


async def catalog_for_principal(
    cfg: Config,
    store: ApplicationStore,
    principal: Principal,
    *,
    include_hidden: bool = False,
    ollama: Any | None = None,
    inventory: tuple[ServedModel, ...] | None = None,
) -> tuple[ServedModel, ...]:
    """Return models this principal may invoke, with database policy overlays."""

    policies = {
        policy.model_id: policy
        for policy in await store.list_model_access_policies()
    }
    result: list[ServedModel] = []
    available = inventory
    if available is None:
        available = (await discover_models(cfg, ollama)).models
    for model in available:
        model = _apply_policy(model, policies.get(model.id))
        if include_hidden and principal.is_admin:
            result.append(model)
        elif model.enabled and _audience_allows(model.audience, principal):
            result.append(model)
    return tuple(result)


async def resolve_model(
    cfg: Config,
    store: ApplicationStore,
    principal: Principal,
    model_id: str,
    *,
    ollama: Any | None = None,
) -> ServedModel | None:
    """Resolve an invokable model without revealing inaccessible entries."""

    model_id = str(model_id).strip()
    return next(
        (
            model
            for model in await catalog_for_principal(
                cfg,
                store,
                principal,
                ollama=ollama,
            )
            if model.id == model_id
        ),
        None,
    )


def _apply_policy(
    model: ServedModel,
    policy: ModelAccessPolicy | None,
) -> ServedModel:
    if policy is None:
        return model
    return replace(model, enabled=policy.enabled, audience=policy.audience)


def _direct_model(
    concrete: str,
    *,
    direct_defaults: dict[str, Any] | Any,
    overrides: dict[str, Any] | Any,
    registry: ModelRegistry,
) -> ServedModel:
    entry = overrides.get(concrete) or {}
    location = registry.location_of(concrete)
    return ServedModel(
        id=f"direct/{concrete}",
        label=str(entry.get("label") or _model_label(concrete)),
        description=str(
            entry.get("description")
            or f"Talk directly to {concrete} without Audrey orchestration."
        ),
        kind="direct",
        mode="direct",
        protocol_model=f"audrey_passthrough/{concrete}",
        concrete_model=concrete,
        presentation=str(entry.get("presentation") or location),
        capabilities=tuple(entry.get("capabilities") or ("text",)),
        enabled=bool(entry.get("enabled", direct_defaults.get("enabled", True))),
        audience=str(entry.get("audience", direct_defaults.get("audience", "admins"))),
        num_ctx=_optional_positive_int(
            entry.get("num_ctx", direct_defaults.get("num_ctx"))
        ),
        max_tokens=_optional_positive_int(
            entry.get("max_tokens", direct_defaults.get("max_tokens"))
        ),
    )


def _installed_model_names(items: list[dict[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("model") or item.get("name")
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if name:
            names.add(name)
    return tuple(sorted(names, key=str.casefold))


def _audience_allows(audience: str, principal: Principal) -> bool:
    if principal.status != "active":
        return False
    if principal.is_admin:
        return True
    if audience == "users":
        return "users" in principal.groups
    if audience == "testers":
        return principal.is_tester
    return False


def _model_label(concrete: str) -> str:
    base = concrete.split(":", 1)[0]
    return base.replace("-", " ").replace("_", " ").title()


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


__all__ = [
    "InventorySource",
    "ModelInventory",
    "ModelAudience",
    "ModelKind",
    "ServedModel",
    "catalog_for_principal",
    "configured_models",
    "discover_models",
    "resolve_model",
]
