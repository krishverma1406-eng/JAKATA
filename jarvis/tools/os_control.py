"""Real computer-use tool powered by pyautogui."""

from __future__ import annotations

from typing import Any

from services import os_control as desktop


TOOL_DEFINITION = {
    "name": "os_control",
    "description": "Generic desktop automation for keyboard, mouse, screenshots, waits, and app launch/close. Supports typing text, single-key presses, hotkeys, key down/up, mouse move/down/up, clicks, scroll, drag, and coordinate inspection. Pair with screenshot analysis when screen understanding is required.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot",
                    "position",
                    "move",
                    "click",
                    "mouse_down",
                    "mouse_up",
                    "type",
                    "press",
                    "key_down",
                    "key_up",
                    "scroll",
                    "hotkey",
                    "drag",
                    "wait",
                    "open_app",
                    "close_app",
                ],
                "description": "Desktop action to perform.",
            },
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "end_x": {"type": "integer"},
            "end_y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer"},
            "text": {"type": "string"},
            "key": {"type": "string"},
            "presses": {"type": "integer"},
            "amount": {"type": "integer"},
            "keys": {"type": "array", "items": {"type": "string"}},
            "target": {"type": "string"},
            "force": {"type": "boolean"},
            "interval": {"type": "number"},
            "seconds": {"type": "number"},
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
    if action == "position":
        return {"ok": True, **desktop.position()}
    if action == "move":
        return {"ok": True, **desktop.move_mouse(
            int(params.get("x", 0) or 0),
            int(params.get("y", 0) or 0),
            duration=float(params.get("duration", 0.0) or 0.0),
        )}
    if action == "click":
        return {"ok": True, **desktop.click(
            int(params.get("x", 0) or 0),
            int(params.get("y", 0) or 0),
            button=str(params.get("button", "left") or "left"),
            clicks=max(1, int(params.get("clicks", 1) or 1)),
        )}
    if action == "mouse_down":
        return {"ok": True, **desktop.mouse_down(button=str(params.get("button", "left") or "left"))}
    if action == "mouse_up":
        return {"ok": True, **desktop.mouse_up(button=str(params.get("button", "left") or "left"))}
    if action == "type":
        return {"ok": True, **desktop.type_text(
            str(params.get("text", "")),
            interval=float(params.get("interval", 0.02) or 0.02),
        )}
    if action == "press":
        key = str(params.get("key", "")).strip()
        if not key:
            return {"ok": False, "error": "key is required for press action."}
        return {"ok": True, **desktop.press_key(
            key,
            presses=max(1, int(params.get("presses", 1) or 1)),
            interval=float(params.get("interval", 0.0) or 0.0),
        )}
    if action == "key_down":
        key = str(params.get("key", "")).strip()
        if not key:
            return {"ok": False, "error": "key is required for key_down action."}
        return {"ok": True, **desktop.key_down(key)}
    if action == "key_up":
        key = str(params.get("key", "")).strip()
        if not key:
            return {"ok": False, "error": "key is required for key_up action."}
        return {"ok": True, **desktop.key_up(key)}
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
    if action == "wait":
        return {"ok": True, **desktop.wait(float(params.get("seconds", 0.0) or 0.0))}
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
