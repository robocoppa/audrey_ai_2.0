"""Exact, provider-advertised reasoning controls for compatibility passthrough."""

from __future__ import annotations

from fastapi import HTTPException

from audrey.models.ollama import OllamaClient, OllamaError


def validate_reasoning_request(
    model: str, *, effort: str | None, think: bool | None = None,
) -> None:
    """Reject conflicting or pipeline-owned controls before provider/file work."""
    if effort is not None and think is not None:
        raise HTTPException(status_code=400, detail={
            "error": "reasoning_controls_conflict",
            "message": "Supply reasoning effort or the boolean think control, not both.",
        })
    if effort is not None and not model.startswith("audrey_passthrough/"):
        raise HTTPException(status_code=400, detail={
            "error": "reasoning_unsupported",
            "message": "Reasoning effort requires an audrey_passthrough/<model> request.",
        })


async def resolve_reasoning_effort(
    ollama: OllamaClient, concrete: str, effort: str,
) -> bool | str:
    """Forward only the exact supported name; ``none`` requires explicit off."""
    try:
        values = await ollama.thinking_values(concrete)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail={
            "error": "reasoning_metadata_unavailable",
            "message": "Could not verify the requested model's reasoning controls; retry later.",
        }) from exc
    if effort == "none":
        if any(value is False for value in values):
            return False
    elif any(type(value) is str and value == effort for value in values):
        return effort
    raise HTTPException(status_code=400, detail={
        "error": "reasoning_unsupported",
        "message": (
            f"Model {concrete!r} does not advertise reasoning effort {effort!r}. "
            "Named effort must exactly match an advertised string; none requires an explicit false control."
        ),
        "supported_values": list(values),
    })
