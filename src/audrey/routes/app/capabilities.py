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
    label: str
    description: str


class SkillsResponse(BaseModel):
    enabled: bool = False
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
    status: Literal["ready", "degraded", "unavailable"] = (
        "unavailable" if chat == "unavailable"
        else "degraded" if tools in {"degraded", "unavailable"} or knowledge != "available"
        else "ready"
    )
    return CapabilitiesResponse(
        status=status,
        generated_at=snapshot.generated_at,
        chat=CapabilityState(status=chat),
        tools=CapabilityState(status=tools),
        knowledge=CapabilityState(status=knowledge),
        skills=CapabilityState(status="disabled"),
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
    _: Principal = Depends(require_principal),
) -> SkillsResponse:
    """Phase 3 owns skill registration, selection, and execution."""

    return SkillsResponse()


__all__ = ["CapabilitiesResponse", "SkillsResponse", "capability_response", "router"]
