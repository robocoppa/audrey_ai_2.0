"""Account-safe capability health and the reserved skills contract."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.readiness import ReadinessStatus

router = APIRouter(tags=["application-capabilities"])


class CapabilityState(BaseModel):
    status: Literal["available", "degraded", "unavailable", "disabled"]


class CapabilitiesResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    generated_at: str
    chat: CapabilityState
    tools: CapabilityState
    knowledge: CapabilityState
    skills: CapabilityState


class SkillSummary(BaseModel):
    id: str
    name: str
    description: str
    version: int
    supported_modes: list[Literal["auto", "fast", "deep"]]
    availability: Literal["available", "degraded"]


class SkillsResponse(BaseModel):
    enabled: bool = False
    status: Literal["disabled", "ready", "degraded", "unavailable"] = "disabled"
    items: list[SkillSummary] = Field(default_factory=list)


def capability_response(snapshot: ReadinessStatus) -> CapabilitiesResponse:
    """Project operational readiness without exposing hosts, queues, or tool names."""

    components = snapshot.components
    chat = (
        "available"
        if components.get("ollama") and components["ollama"].status == "available"
        else "unavailable"
    )
    tool_component = components.get("custom_tools")
    if tool_component is None or tool_component.status == "disabled":
        tools = "disabled"
    elif tool_component.status != "available":
        tools = "unavailable"
    elif (
        snapshot.tools.discovered_count < snapshot.tools.policy_count
        or snapshot.tools.available_count < snapshot.tools.discovered_count
    ):
        tools = "degraded"
    else:
        tools = "available"
    qdrant_available = bool(
        components.get("qdrant") and components["qdrant"].status == "available"
    )
    knowledge = (
        "unavailable" if not qdrant_available or tools in {"unavailable", "disabled"}
        else "degraded" if tools == "degraded"
        else "available"
    )
    skill_snapshot = getattr(snapshot, "skills", None)
    skill_status = getattr(skill_snapshot, "status", "disabled")
    skills = {
        "ready": "available",
        "degraded": "degraded",
        "unavailable": "unavailable",
        "disabled": "disabled",
    }.get(skill_status, "unavailable")
    status: Literal["ready", "degraded", "unavailable"] = (
        "unavailable" if chat == "unavailable"
        else "degraded" if (
            tools in {"degraded", "unavailable"}
            or knowledge != "available"
            or skills in {"degraded", "unavailable"}
        )
        else "ready"
    )
    return CapabilitiesResponse(
        status=status,
        generated_at=snapshot.generated_at,
        chat=CapabilityState(status=chat),
        tools=CapabilityState(status=tools),
        knowledge=CapabilityState(status=knowledge),
        skills=CapabilityState(status=skills),
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def application_capabilities(
    request: Request,
    _: Principal = Depends(require_principal),
) -> CapabilitiesResponse:
    collector = getattr(request.app.state, "readiness", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="capability_health_unavailable")
    return capability_response(await collector.collect())


@router.get("/skills", response_model=SkillsResponse)
async def application_skills(
    request: Request,
    _: Principal = Depends(require_principal),
) -> SkillsResponse:
    """Return safe registry metadata without instructions or filesystem paths."""

    registry = getattr(request.app.state, "skills", None)
    if registry is None:
        return SkillsResponse()
    snapshot = registry.snapshot()
    return SkillsResponse(
        enabled=snapshot.enabled,
        status=snapshot.status,
        items=[
            SkillSummary(
                id=item.id,
                name=item.name,
                description=item.description,
                version=item.version,
                supported_modes=list(item.supported_modes),
                availability=item.availability,
            )
            for item in registry.catalog()
        ],
    )


__all__ = ["CapabilitiesResponse", "SkillsResponse", "capability_response", "router"]
