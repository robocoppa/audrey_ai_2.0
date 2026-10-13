"""Bounded decimal arithmetic, common unit conversion, and linear/quadratic equations.

Expression syntax is parsed with ast and interpreted by an explicit numeric
visitor. No Python evaluation, imports, attributes, calls, or state are exposed.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

_PRECISION = 40
_MAX_ABS = Decimal("1e100")
_MIN_ABS = Decimal("1e-100")
_MAX_NODES = 96
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_PERCENT = re.compile(r"(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*%")
router = APIRouter()


class CalculateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["evaluate", "convert", "solve"] = Field(
        default="evaluate",
        description="evaluate arithmetic, convert units, or solve a linear/quadratic equation in x.",
    )
    expression: str | None = Field(
        default=None, min_length=1, max_length=512,
        description=(
            "evaluate/solve only: decimal numbers, parentheses, + - * / **. "
            "Numeric percentages such as 15% mean /100. Powers must be integers "
            "between -100 and 100. solve requires one '=' and only variable x, "
            "for example x**2 - 5*x + 6 = 0. No functions or code."
        ),
    )
    value: str | None = Field(
        default=None, min_length=1, max_length=160,
        description="convert only: finite decimal number, as a string to preserve precision.",
    )
    from_unit: str | None = Field(
        default=None, min_length=1, max_length=40,
        description="convert only: common length, area, volume, mass, temperature, or time unit.",
    )
    to_unit: str | None = Field(
        default=None, min_length=1, max_length=40,
        description=(
            "convert only: target unit of the same dimension. Examples: m/ft, "
            "m2/ft2, L/US gallon, kg/lb, C/F/K, seconds/hours. Gallons/quarts are US."
        ),
    )

    @model_validator(mode="after")
    def check_fields(self) -> CalculateRequest:
        if self.operation == "convert":
            if self.expression is not None or any(
                value is None for value in (self.value, self.from_unit, self.to_unit)
            ):
                raise ValueError("convert requires value/from_unit/to_unit and no expression")
        elif self.expression is None or any(
            value is not None for value in (self.value, self.from_unit, self.to_unit)
        ):
            raise ValueError("evaluate/solve require expression and no conversion fields")
        return self


class CalculateResponse(BaseModel):
    operation: Literal["evaluate", "convert", "solve"]
    result: str | None = None
    unit: str | None = None
    roots: list[str] | None = None
    outcome: Literal["value", "real_roots", "no_real_roots", "all_real_numbers", "no_solution"]
    precision_digits: int = _PRECISION


class CalculationError(ValueError):
    """Unsupported input or computation outside the tool's numeric limits."""


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > _MAX_ABS or (value and abs(value) < _MIN_ABS):
        raise CalculationError("Values must be zero or have magnitude between 1e-100 and 1e100")
    return +value


def _decimal(text: str) -> Decimal:
    numeric = text.strip()
    if len(numeric) > 160 or not _NUMBER.fullmatch(numeric.lstrip("+-")):
        raise CalculationError("Use a finite decimal number")
    return _bounded(Decimal(numeric))


def _format(value: Decimal) -> str:
    _bounded(value)
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


# Coefficients are in ascending order: constant, x, x**2.
Polynomial = tuple[Decimal, Decimal, Decimal]
_ZERO = Decimal(0)
_ONE = Decimal(1)


def _constant(value: Decimal) -> Polynomial:
    return (_bounded(value), _ZERO, _ZERO)


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    values = [_ZERO] * 5
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            values[i + j] = _bounded(values[i + j] + _bounded(a * b))
    if values[3] or values[4]:
        raise CalculationError("Only linear or quadratic equations are supported")
    return (values[0], values[1], values[2])


def _parse(expression: str, *, variable: bool) -> Polynomial:
    normalized = _PERCENT.sub(r"(\1/100)", expression.strip())
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise CalculationError("Invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise CalculationError("Expression is too complex (maximum 96 syntax nodes)")

    def visit(node: ast.AST, depth: int = 0) -> Polynomial:
        if depth > 24:
            raise CalculationError("Expression is nested too deeply")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            raw = ast.get_source_segment(normalized, node)
            return _constant(_decimal(raw or ""))
        if variable and isinstance(node, ast.Name) and node.id == "x":
            return (_ZERO, _ONE, _ZERO)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand, depth + 1)
            return tuple(-v for v in result) if isinstance(node.op, ast.USub) else result
        if not isinstance(node, ast.BinOp):
            raise CalculationError("Only decimal arithmetic and the variable x are supported")
        left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            return tuple(_bounded(a + sign * b) for a, b in zip(left, right, strict=True))
        if isinstance(node.op, ast.Mult):
            return _multiply(left, right)
        if isinstance(node.op, ast.Div):
            if right[1] or right[2]:
                raise CalculationError("Division by a variable is unsupported")
            if not right[0]:
                raise CalculationError("Division by zero")
            return tuple(_bounded(value / right[0]) for value in left)
        if isinstance(node.op, ast.Pow):
            if right[1] or right[2] or right[0] != right[0].to_integral_value():
                raise CalculationError("Powers require an integer exponent")
            exponent = int(right[0])
            if abs(exponent) > 100:
                raise CalculationError("Exponent must be between -100 and 100")
            if not left[1] and not left[2]:
                if not left[0] and exponent <= 0:
                    raise CalculationError("Zero cannot have a zero or negative exponent")
                return _constant(_bounded(left[0] ** exponent))
            if exponent not in (0, 1, 2):
                raise CalculationError("Variable powers must be 0, 1, or 2")
            if exponent == 0:
                return _constant(_ONE)
            return left if exponent == 1 else _multiply(left, left)
        raise CalculationError("Supported operators are + - * / **")

    return visit(tree.body)


@dataclass(frozen=True)
class _Unit:
    dimension: str
    name: str
    numerator: Decimal
    denominator: Decimal = _ONE
    offset: Decimal = _ZERO


def _unit_key(name: str) -> str:
    return name.strip().lower().replace("°", "").replace("²", "2").replace("³", "3").replace("^", "")


def _build_units() -> dict[str, _Unit]:
    units: dict[str, _Unit] = {}

    def add(dimension: str, name: str, scale: str, *aliases: str,
            denominator: str = "1", offset: str = "0") -> None:
        unit = _Unit(dimension, name, Decimal(scale), Decimal(denominator), Decimal(offset))
        for alias in (name, *aliases):
            units[_unit_key(alias)] = unit

    add("length", "m", "1", "meter", "meters", "metre", "metres")
    add("length", "mm", ".001", "millimeter", "millimeters")
    add("length", "cm", ".01", "centimeter", "centimeters")
    add("length", "km", "1000", "kilometer", "kilometers")
    add("length", "in", ".0254", "inch", "inches")
    add("length", "ft", ".3048", "foot", "feet")
    add("length", "yd", ".9144", "yard", "yards")
    add("length", "mi", "1609.344", "mile", "miles")
    add("area", "m2", "1", "sq m", "square meter", "square meters")
    add("area", "cm2", ".0001", "sq cm")
    add("area", "km2", "1000000", "sq km")
    add("area", "in2", ".00064516", "sq in", "square inch", "square inches")
    add("area", "ft2", ".09290304", "sq ft", "square foot", "square feet")
    add("area", "acre", "4046.8564224", "acres")
    add("area", "ha", "10000", "hectare", "hectares")
    add("volume", "L", "1", "liter", "liters", "litre", "litres")
    add("volume", "mL", ".001", "milliliter", "milliliters")
    add("volume", "m3", "1000", "cubic meter", "cubic meters")
    add("volume", "cm3", ".001", "cubic centimeter", "cc")
    add("volume", "in3", ".016387064", "cubic inch", "cubic inches")
    add("volume", "ft3", "28.316846592", "cubic foot", "cubic feet")
    add("volume", "US gallon", "3.785411784", "gallon", "gallons", "gal", "us gallons")
    add("volume", "US quart", ".946352946", "quart", "quarts", "qt")
    add("mass", "kg", "1", "kilogram", "kilograms")
    add("mass", "g", ".001", "gram", "grams")
    add("mass", "mg", ".000001", "milligram", "milligrams")
    add("mass", "lb", ".45359237", "pound", "pounds", "lbs")
    add("mass", "oz", ".028349523125", "ounce", "ounces")
    add("mass", "metric tonne", "1000", "tonne", "tonnes")
    add("time", "s", "1", "second", "seconds", "sec")
    add("time", "min", "60", "minute", "minutes")
    add("time", "h", "3600", "hour", "hours", "hr")
    add("time", "day", "86400", "days")
    add("time", "week", "604800", "weeks")
    add("temperature", "K", "1", "kelvin")
    add("temperature", "C", "1", "celsius", offset="273.15")
    add("temperature", "F", "5", "fahrenheit", denominator="9", offset="459.67")
    return units


_UNITS = _build_units()


def calculate_value(request: CalculateRequest) -> CalculateResponse:
    """Calculate within fixed precision and magnitude limits; no side effects."""
    with localcontext() as context:
        context.prec = _PRECISION
        try:
            if request.operation == "convert":
                source = _UNITS.get(_unit_key(request.from_unit or ""))
                target = _UNITS.get(_unit_key(request.to_unit or ""))
                if source is None or target is None:
                    raise CalculationError("Unsupported unit; use common length/area/volume/mass/temperature/time units")
                if source.dimension != target.dimension:
                    raise CalculationError("Source and target units must have the same dimension")
                value = _decimal(request.value or "")
                base = _bounded((value + source.offset) * source.numerator / source.denominator)
                if source.dimension == "temperature" and base < 0:
                    raise CalculationError("Temperature cannot be below absolute zero")
                converted = _bounded(base * target.denominator / target.numerator - target.offset)
                return CalculateResponse(operation="convert", result=_format(converted),
                                         unit=target.name, outcome="value")
            if request.operation == "evaluate":
                result = _parse(request.expression or "", variable=False)[0]
                return CalculateResponse(operation="evaluate", result=_format(result), outcome="value")
            equation = request.expression or ""
            if equation.count("=") != 1:
                raise CalculationError("solve requires one '=' and a linear/quadratic equation in x")
            left_text, right_text = equation.split("=")
            left, right = _parse(left_text, variable=True), _parse(right_text, variable=True)
            c, b, a = (_bounded(lhs - rhs) for lhs, rhs in zip(left, right, strict=True))
            if not a and not b:
                return CalculateResponse(operation="solve", roots=[],
                                         outcome="all_real_numbers" if not c else "no_solution")
            if not a:
                roots = [_format(_bounded(-c / b))]
            else:
                discriminant = _bounded(b * b - 4 * a * c)
                if discriminant < 0:
                    return CalculateResponse(operation="solve", roots=[], outcome="no_real_roots")
                # Stable quadratic formula avoids cancellation for disparate roots.
                if not discriminant:
                    roots = [_format(_bounded(-b / (2 * a)))]
                else:
                    root = discriminant.sqrt()
                    q = -(b + (root if b >= 0 else -root)) / 2
                    values = sorted({_bounded(q / a), _bounded(c / q)})
                    roots = [_format(value) for value in values]
            return CalculateResponse(operation="solve", roots=roots, outcome="real_roots")
        except DecimalException as error:
            raise CalculationError("Calculation is undefined or outside numeric limits") from error


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
