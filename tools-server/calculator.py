"""Sidecar route for the shared bounded local calculator."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

try:
    # The image copies the exact shared core beside this route module.
    from calculator_core import (
        CalculateRequest,
        CalculateResponse,
        CalculationError,
        calculate_value,
    )
except ModuleNotFoundError:
    from audrey.tools.calculator import (
        CalculateRequest,
        CalculateResponse,
        CalculationError,
        calculate_value,
    )

router = APIRouter()


@router.post("/calculate", response_model=CalculateResponse, operation_id="calculate", tags=["tools"])
async def calculate(request: CalculateRequest) -> CalculateResponse:
    """Compute decimal arithmetic, percentages, common unit conversions, or real
    roots of linear/quadratic equations in x. Use this for dependable numerical
    results rather than estimating. Read-only, local, bounded to 40 significant
    digits and magnitudes 1e-100..1e100; unsupported math is reported explicitly.
    """
    try:
        return calculate_value(request)
    except CalculationError as error:
        raise HTTPException(status_code=422, detail={
            "error": "calculation_unsupported", "message": str(error),
        }) from error
