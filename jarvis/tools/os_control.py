"""Real computer-use tool powered by pyautogui."""

from __future__ import annotations

from typing import Any

from services import os_control as desktop


TOOL_DEFINITION = {
    "name": "os_control",
    "description": "Control the local desktop with screenshots, coordinate-based clicks, typing, scrolling, hotkeys, dragging, and app launching. Pair with screenshot analysis when screen understanding is required.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "click", "type", "scroll", "hotkey", "drag", "open_app", "close_app"],
                "description": "Desktop action to perform.",
            },
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "end_x": {"type": "integer"},
            "end_y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer"},
            "text": {"type": "string"},
            "amount": {"type": "integer"},
            "keys": {"type": "array", "items": {"type": "string"}},
            "target": {"type": "string"},
            "force": {"type": "boolean"},
            "duration": {"type": "number"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    if action == "screenshot":
        return {"ok": True, **desktop.screenshot()}
    if action == "click":
        return {"ok": True, **desktop.click(
            int(params.get("x", 0) or 0),
            int(params.get("y", 0) or 0),
            button=str(params.get("button", "left") or "left"),
            clicks=max(1, int(params.get("clicks", 1) or 1)),
        )}
    if action == "type":
        return {"ok": True, **desktop.type_text(str(params.get("text", "")))}
    if action == "scroll":
        return {"ok": True, **desktop.scroll(int(params.get("amount", 0) or 0))}
    if action == "hotkey":
        keys = [str(item) for item in params.get("keys", []) if str(item).strip()]
        if not keys:
            return {"ok": False, "error": "keys are required for hotkey action."}
        return {"ok": True, **desktop.hotkey(*keys)}
    if action == "drag":
        return {
            "ok": True,
            **desktop.drag(
                int(params.get("x", 0) or 0),
                int(params.get("y", 0) or 0),
                int(params.get("end_x", 0) or 0),
                int(params.get("end_y", 0) or 0),
                duration=float(params.get("duration", 0.2) or 0.2),
            ),
        }
    if action == "open_app":
        target = str(params.get("target", "")).strip()
        if not target:
            return {"ok": False, "error": "target is required for open_app action."}
        return {"ok": True, **desktop.open_app(target)}
    if action == "close_app":
        target = str(params.get("target", "")).strip()
        if not target:
            return {"ok": False, "error": "target is required for close_app action."}
        result = desktop.close_app(target, force=bool(params.get("force", False)))
        if result.get("verified_closed") is False:
            return {
                "ok": False,
                **result,
                "error": (
                    f"{result.get('target', target)} is still running "
                    f"({result.get('remaining', 0)} process(es) remain)."
                ),
            }
        return {"ok": True, **result}
    return {"ok": False, "error": f"Unsupported action: {action}"}
