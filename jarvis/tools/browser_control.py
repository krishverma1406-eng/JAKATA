"""Persistent browser automation via Amazon Nova Act."""

from __future__ import annotations

from typing import Any

from services.browser_automation import BrowserAutomationError, get_browser_service


TOOL_DEFINITION = {
    "name": "browser_control",
    "description": "Control a browser session through Amazon Nova Act, including navigation, clicks, typing, extraction, screenshots, waits, scrolling, JavaScript evaluation, form filling, and prompt-driven browser tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open",
                    "fetch",
                    "click",
                    "type",
                    "extract",
                    "screenshot",
                    "scroll",
                    "wait",
                    "evaluate",
                    "fill_form",
                    "status",
                    "act",
                ],
                "description": "Browser action to perform.",
            },
            "url": {
                "type": "string",
                "description": "URL to open or fetch.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the target element.",
            },
            "text": {
                "type": "string",
                "description": "Visible text target when no selector is available.",
            },
            "value": {
                "type": "string",
                "description": "Text value for typing or field filling.",
            },
            "fields": {
                "type": "object",
                "description": "For fill_form: map CSS selectors to values, or selector keys to objects containing value/selector/text.",
            },
            "script": {
                "type": "string",
                "description": "JavaScript snippet/expression to evaluate in the page context.",
            },
            "arg": {
                "description": "Optional JSON-serializable argument passed to the evaluated script.",
            },
            "prompt": {
                "type": "string",
                "description": "Optional vision prompt for screenshot analysis.",
            },
            "path": {
                "type": "string",
                "description": "Optional screenshot output path.",
            },
            "new_tab": {
                "type": "boolean",
                "description": "Deprecated. Nova Act keeps one browser session per JARVIS session.",
                "default": False,
            },
            "wait_until": {
                "type": "string",
                "enum": ["domcontentloaded", "load", "networkidle"],
                "description": "Optional load state to wait for after navigation.",
            },
            "wait_for_navigation": {
                "type": "boolean",
                "description": "After clicking, wait for page load.",
                "default": False,
            },
            "clear": {
                "type": "boolean",
                "description": "Whether to clear the field before typing/filling.",
                "default": True,
            },
            "press_enter": {
                "type": "boolean",
                "description": "Press Enter after typing.",
                "default": False,
            },
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction.",
                "default": "down",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll amount in pixels.",
                "default": 900,
            },
            "milliseconds": {
                "type": "integer",
                "description": "For wait without selector/text: sleep duration in milliseconds.",
                "default": 1000,
            },
            "state": {
                "type": "string",
                "enum": ["attached", "detached", "visible", "hidden"],
                "description": "Desired element state for wait.",
                "default": "visible",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Optional action timeout in milliseconds.",
            },
            "exact_text": {
                "type": "boolean",
                "description": "Use exact visible-text matching.",
                "default": False,
            },
            "full_page": {
                "type": "boolean",
                "description": "For screenshots, capture the full page.",
                "default": True,
            },
            "analyze": {
                "type": "boolean",
                "description": "For screenshots, run vision analysis on the captured image.",
                "default": False,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum extracted text length to return.",
            },
            "task": {
                "type": "string",
                "description": "For action=act, a natural-language browser task for Nova Act.",
            },
            "max_steps": {
                "type": "integer",
                "description": "For action=act, maximum browser steps Nova Act should take.",
                "default": 30,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    runtime_context = params.get("_runtime_context", {})
    session_id = ""
    if isinstance(runtime_context, dict):
        session_id = str(runtime_context.get("session_id", "")).strip()
    try:
        return get_browser_service().run(action, params, session_id=session_id or "default")
    except BrowserAutomationError as exc:
        return {"ok": False, "action": action, "error": str(exc)}
