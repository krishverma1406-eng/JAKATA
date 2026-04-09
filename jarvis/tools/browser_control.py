"""Basic browser control actions."""

from __future__ import annotations

import re
import webbrowser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TOOL_DEFINITION = {
    "name": "browser_control",
    "description": "Open a URL in the browser or fetch page content.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "fetch"],
                "description": "Whether to open the page or fetch its content.",
            },
            "url": {
                "type": "string",
                "description": "The page URL to open or fetch.",
            },
        },
        "required": ["action", "url"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    url = str(params.get("url", "")).strip()
    if not url:
        return {"ok": False, "error": "Missing URL."}

    if action == "open":
        opened = webbrowser.open(url)
        return {"ok": bool(opened), "action": "open", "url": url}

    if action == "fetch":
        request = Request(url=url, headers={"User-Agent": "Jarvis/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            return {"ok": False, "error": f"HTTP {exc.code}", "url": url}
        except URLError as exc:
            return {"ok": False, "error": str(exc), "url": url}

        plain_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
        return {"ok": True, "action": "fetch", "url": url, "content": plain_text[:4000]}

    return {"ok": False, "error": f"Unsupported action: {action}"}
