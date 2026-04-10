"""Screenshot capture and optional vision analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services import os_control
from services.vision_service import analyze_image_path as analyze_vision_image_path


TOOL_DEFINITION = {
    "name": "screenshot_tool",
    "description": "Capture a screenshot and optionally analyze it with the configured live backend, including custom prompts for OCR, UI inspection, or coordinate-style guidance.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["capture", "analyze"],
                "description": "Screenshot action to perform.",
            },
            "path": {
                "type": "string",
                "description": "Optional existing screenshot path to analyze.",
            },
            "prompt": {
                "type": "string",
                "description": "Optional analysis prompt for analyze.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    if action == "capture":
        shot = os_control.screenshot()
        return {"ok": True, **shot}
    if action == "analyze":
        target_path = str(params.get("path", "")).strip()
        if not target_path:
            target_path = str(os_control.screenshot()["path"])
        prompt = str(params.get("prompt", "")).strip() or "Describe what is visible on this screen."
        return analyze_image_path(Path(target_path), prompt)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def analyze_image_path(path: Path, prompt: str) -> dict[str, Any]:
    return analyze_vision_image_path(path, prompt)
