"""Safe math evaluation and basic unit conversion."""

from __future__ import annotations

import ast
import math
import re
from typing import Any


TOOL_DEFINITION = {
    "name": "calculator_tool",
    "description": "Safely evaluate math expressions and basic unit conversions.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression or conversion request like '12 km to m'.",
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}


_MATH_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "floor": math.floor,
    "log": math.log,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

_UNIT_FACTORS = {
    "mm": ("length", 0.001),
    "cm": ("length", 0.01),
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "in": ("length", 0.0254),
    "inch": ("length", 0.0254),
    "ft": ("length", 0.3048),
    "yard": ("length", 0.9144),
    "mile": ("length", 1609.344),
    "mg": ("mass", 0.001),
    "g": ("mass", 1.0),
    "kg": ("mass", 1000.0),
    "lb": ("mass", 453.59237),
    "oz": ("mass", 28.349523125),
    "s": ("time", 1.0),
    "sec": ("time", 1.0),
    "min": ("time", 60.0),
    "hour": ("time", 3600.0),
    "day": ("time", 86400.0),
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    expression = str(params.get("expression", "")).strip()
    if not expression:
        return {"ok": False, "error": "expression is required."}

    conversion = _convert_units(expression)
    if conversion is not None:
        return {"ok": True, **conversion}

    try:
        result = _evaluate_math(expression)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "expression": expression, "result": result}


def _evaluate_math(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _MATH_FUNCTIONS or callable(_MATH_FUNCTIONS[func_name]) is False:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        func = _MATH_FUNCTIONS[func_name]
        return float(func(*args))
    if isinstance(node, ast.Name) and node.id in _MATH_FUNCTIONS and not callable(_MATH_FUNCTIONS[node.id]):
        return float(_MATH_FUNCTIONS[node.id])
    raise ValueError("Unsupported expression.")


def _convert_units(expression: str) -> dict[str, Any] | None:
    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*(?:to|in)\s*([A-Za-z]+)\s*$",
        expression,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    source = match.group(2).lower()
    target = match.group(3).lower()

    if source in {"c", "f", "k"} or target in {"c", "f", "k"}:
        converted = _convert_temperature(value, source, target)
        return {"expression": expression, "result": converted, "unit": target}

    if source not in _UNIT_FACTORS or target not in _UNIT_FACTORS:
        raise ValueError("Unsupported conversion units.")
    source_kind, source_factor = _UNIT_FACTORS[source]
    target_kind, target_factor = _UNIT_FACTORS[target]
    if source_kind != target_kind:
        raise ValueError("Cannot convert between incompatible unit types.")
    base_value = value * source_factor
    converted = base_value / target_factor
    return {"expression": expression, "result": converted, "unit": target}


def _convert_temperature(value: float, source: str, target: str) -> float:
    if source == target:
        return value
    if source == "c":
        celsius = value
    elif source == "f":
        celsius = (value - 32) * 5 / 9
    elif source == "k":
        celsius = value - 273.15
    else:
        raise ValueError("Unsupported temperature source unit.")

    if target == "c":
        return celsius
    if target == "f":
        return (celsius * 9 / 5) + 32
    if target == "k":
        return celsius + 273.15
    raise ValueError("Unsupported temperature target unit.")
