"""Launch local applications by name or executable path."""

from __future__ import annotations

from typing import Any

from services import os_control


TOOL_DEFINITION = {
    "name": "app_launcher_tool",
    "description": "Launch local applications by name or executable path.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Application name or executable path to open.",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    target = str(params.get("target", "")).strip()
    if not target:
        return {"ok": False, "error": "target is required."}
    result = os_control.open_app(target)
    return {"ok": True, **result}
