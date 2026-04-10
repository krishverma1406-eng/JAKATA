"""Analyze the current frontend camera frame or another attached live image."""

from __future__ import annotations

from typing import Any

from services.vision_service import analyze_base64_frame


TOOL_DEFINITION = {
    "name": "vision_tool",
    "description": "Analyze the current attached image or live frontend camera frame and answer questions about what is visible.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Question or analysis instruction for the current image, such as 'What is this?' or 'Describe what you see in detail.'",
            },
        },
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    runtime_context = params.get("_runtime_context", {})
    if not isinstance(runtime_context, dict):
        runtime_context = {}
    imgbase64 = str(runtime_context.get("imgbase64", "")).strip()
    if not imgbase64:
        return {
            "ok": False,
            "error": "No live image frame is attached to this request.",
        }
    prompt = str(params.get("prompt", "")).strip() or "Describe what is visible in this image."
    result = analyze_base64_frame(imgbase64, prompt)
    if result.get("ok"):
        result["source"] = "live_frame"
    return result
