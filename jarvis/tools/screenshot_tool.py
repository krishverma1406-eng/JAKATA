"""Screenshot capture and optional vision analysis."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

from config.settings import SETTINGS
from services import os_control


TOOL_DEFINITION = {
    "name": "screenshot_tool",
    "description": "Capture a screenshot and optionally analyze it with the configured live backend.",
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
        return _analyze_screenshot(Path(target_path), prompt)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _analyze_screenshot(path: Path, prompt: str) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"Screenshot does not exist: {path}"}

    image_bytes = path.read_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    providers = [
        (
            "nvidia",
            SETTINGS.nvidia_api_key.strip(),
            f"{SETTINGS.nvidia_base_url.rstrip('/')}/chat/completions",
            SETTINGS.nvidia_complex_model,
            {"chat_template_kwargs": {"thinking": False}},
        ),
        (
            "openrouter",
            SETTINGS.openrouter_api_key.strip(),
            f"{SETTINGS.openrouter_base_url.rstrip('/')}/chat/completions",
            SETTINGS.openrouter_complex_model,
            {},
        ),
    ]

    errors: list[str] = []
    for provider_name, api_key, url, model, extra_payload in providers:
        if not api_key:
            continue
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.2,
            **extra_payload,
        }
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else "unknown"
            errors.append(f"{provider_name}: HTTP {status}: {body}")
            continue
        except requests.RequestException as exc:
            errors.append(f"{provider_name}: {exc}")
            continue

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            return {"ok": True, "path": str(path), "analysis": content, "provider": provider_name}

    if errors:
        return {"ok": False, "error": " | ".join(errors), "path": str(path)}
    return {"ok": False, "error": "No vision-capable provider is configured.", "path": str(path)}
