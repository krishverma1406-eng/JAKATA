"""Read and write the system clipboard."""

from __future__ import annotations

from typing import Any


TOOL_DEFINITION = {
    "name": "clipboard_tool",
    "description": "Read from or write text to the system clipboard.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "clear"],
                "description": "Clipboard action to perform.",
            },
            "text": {
                "type": "string",
                "description": "Text to copy when action is write.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    try:
        import pyperclip
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pyperclip is not installed.") from exc

    action = str(params.get("action", "")).strip().lower()
    if action == "read":
        text = pyperclip.paste()
        return {"ok": True, "text": text, "length": len(text)}
    if action == "write":
        text = str(params.get("text", ""))
        pyperclip.copy(text)
        return {"ok": True, "copied": True, "length": len(text)}
    if action == "clear":
        pyperclip.copy("")
        return {"ok": True, "cleared": True}
    return {"ok": False, "error": f"Unsupported action: {action}"}
