"""Numerical and input-boundary checks for the read-only calculate tool."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from calculator import (  # noqa: E402
    CalculateRequest,
    CalculationError,
    calculate_value,
    router,
)


def evaluate(expression: str) -> str:
    result = calculate_value(CalculateRequest(expression=expression))
    assert result.result is not None
    return result.result


@pytest.mark.parametrize(("expression", "expected"), [
    ("0.1 + 0.2", "0.3"),
    ("(120 * 15%) + 2.5", "20.5"),
    ("-4**2 + (-4)**2", "0"),
    ("2**-3", "0.125"),
    ("1.23456789012345678901234567890123456789 + 0", "1.23456789012345678901234567890123456789"),
    ("1e100", "1" + "0" * 100),
    ("0 / 3", "0"),
])
def test_decimal_arithmetic(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize(("value", "source", "target", "expected"), [
    ("12", "inches", "ft", "1"),
    ("1", "m²", "cm2", "10000"),
    ("1", "acre", "ft2", "43560"),
    ("1", "US gallon", "L", "3.785411784"),
    ("1", "kg", "g", "1000"),
    ("1", "lb", "oz", "16"),
    ("32", "F", "C", "0"),
    ("-40", "C", "F", "-40"),
    ("0", "C", "K", "273.15"),
    ("2", "hours", "seconds", "7200"),
])
def test_common_units(value, source, target, expected):
    result = calculate_value(CalculateRequest(
        operation="convert", value=value, from_unit=source, to_unit=target,
    ))
    assert result.result == expected
    assert result.outcome == "value"


@pytest.mark.parametrize(("equation", "roots", "outcome"), [
    ("2*x + 4 = 12", ["4"], "real_roots"),
    ("x**2 - 5*x + 6 = 0", ["2", "3"], "real_roots"),
    ("(x+2)**2 = 0", ["-2"], "real_roots"),
    ("x*x = 0", ["0"], "real_roots"),
    ("x*x + 1 = 0", [], "no_real_roots"),
    ("x + 1 = x + 1", [], "all_real_numbers"),
    ("x + 1 = x + 2", [], "no_solution"),
])
def test_linear_and_quadratic_solutions(equation, roots, outcome):
    result = calculate_value(CalculateRequest(operation="solve", expression=equation))
    assert result.roots == roots
    assert result.outcome == outcome


def test_quadratic_formula_retains_small_root():
    result = calculate_value(CalculateRequest(
        operation="solve", expression="x**2 + 1e20*x + 1 = 0",
    ))
    assert result.roots is not None
    # A naive (-b + sqrt(discriminant)) formula loses this small root.
    assert Decimal(result.roots[1]) == Decimal("-1e-20")
    assert Decimal(result.roots[0]) == Decimal("-1e20")


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo no')",
    "(1).__class__",
    "[1][0]",
    "True + 1",
    "sum([1, 2])",
    "x + 1",
    "2**101",
    "2**(2**100)",
    "1e101",
    "1e-100 / 10",
    "1 / 0",
    "0**0",
    "4**0.5",
    "1_000",
    "'1'",
    "1 << 2",
    "100 % 3",
    "+".join(["1"] * 40),
    "-" * 26 + "1",
])
def test_code_unsupported_math_and_bounds_are_rejected(expression):
    with pytest.raises(CalculationError):
        evaluate(expression)


@pytest.mark.parametrize("equation", [
    "x**3 = 1", "x*x*x = 1", "1/x = 2", "x**x = 2",
    "sin(x) = 0", "y = 2", "x", "x = 1 = 2",
])
def test_unsupported_equations_are_explicit(equation):
    with pytest.raises(CalculationError):
        calculate_value(CalculateRequest(operation="solve", expression=equation))


@pytest.mark.parametrize(("value", "source", "target"), [
    ("1", "m", "kg"),
    ("1", "month", "seconds"),
    ("-1", "K", "C"),
    ("-460", "F", "C"),
    ("nan", "kg", "g"),
    ("Infinity", "kg", "g"),
])
def test_conversion_validation(value, source, target):
    with pytest.raises(CalculationError):
        calculate_value(CalculateRequest(
            operation="convert", value=value, from_unit=source, to_unit=target,
        ))


@pytest.mark.parametrize("payload", [
    {},
    {"operation": "convert", "value": "1", "from_unit": "m"},
    {"operation": "evaluate", "expression": "1", "value": "2"},
    {"expression": "1", "user": "other"},
    {"expression": "1" * 513},
])
def test_request_schema_enforces_operation_fields_and_size(payload):
    with pytest.raises(ValidationError):
        CalculateRequest.model_validate(payload)


async def test_route_returns_numeric_result_and_safe_error():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        good = await client.post("/calculate", json={"expression": "250 * 12%"})
        assert good.status_code == 200
        assert good.json()["result"] == "30"
        rejected = await client.post("/calculate", json={"expression": "open('/etc/passwd')"})
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["error"] == "calculation_unsupported"
        missing = await client.post("/calculate", json={"operation": "convert"})
        assert missing.status_code == 422
    schema = app.openapi()
    operation = schema["paths"]["/calculate"]["post"]
    assert operation["operationId"] == "calculate"
    assert operation["tags"] == ["tools"]
