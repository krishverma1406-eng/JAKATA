"""Desktop control helpers using pyautogui."""

from __future__ import annotations

import json
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR, DATA_AI_DIR

SCREENSHOT_DIR = BASE_DIR / "data_user" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
APP_ALIASES_FILE = DATA_AI_DIR / "app_aliases.json"


def screenshot() -> dict[str, Any]:
    gui = _load_gui()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
    image = gui.screenshot()
    image.save(target)
    return {"path": str(target), "width": image.width, "height": image.height}


def click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict[str, Any]:
    gui = _load_gui()
    gui.click(x=x, y=y, button=button, clicks=clicks)
    return {"x": x, "y": y, "button": button, "clicks": clicks}


def type_text(text: str, interval: float = 0.02) -> dict[str, Any]:
    gui = _load_gui()
    gui.write(text, interval=interval)
    return {"typed": text, "length": len(text)}


def hotkey(*keys: str) -> dict[str, Any]:
    gui = _load_gui()
    gui.hotkey(*keys)
    return {"keys": list(keys)}


def scroll(amount: int) -> dict[str, Any]:
    gui = _load_gui()
    gui.scroll(amount)
    return {"amount": amount}


def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.2) -> dict[str, Any]:
    gui = _load_gui()
    gui.moveTo(start_x, start_y, duration=0)
    gui.dragTo(end_x, end_y, duration=duration, button="left")
    return {"start": [start_x, start_y], "end": [end_x, end_y], "duration": duration}


def open_app(target: str) -> dict[str, Any]:
    resolved = _resolve_app_target(target)
    target_type = str(resolved.get("target_type", "")).strip()
    launch_target = str(resolved.get("launch_target", "")).strip()
    display_name = str(resolved.get("display_name", "")).strip() or target
    resolved_target = str(resolved.get("resolved", "")).strip() or launch_target

    if not launch_target:
        raise RuntimeError(f"Could not resolve an app target for '{target}'.")

    if sys.platform.startswith("win"):
        if target_type == "command":
            subprocess.Popen([launch_target], shell=False)
        else:
            subprocess.Popen(["cmd", "/c", "start", "", launch_target], shell=False)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", launch_target], shell=False)
    else:
        subprocess.Popen(["xdg-open", launch_target], shell=False)
    return {"opened": display_name, "resolved": resolved_target, "target_type": target_type}


def _load_gui() -> Any:
    try:
        import pyautogui
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pyautogui is not installed.") from exc

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def _resolve_app_target(target: str) -> dict[str, str]:
    cleaned = str(target).strip()
    if not cleaned:
        raise RuntimeError("App target is required.")

    candidate = Path(cleaned)
    if candidate.exists():
        return {
            "launch_target": str(candidate),
            "resolved": str(candidate),
            "target_type": "path",
            "display_name": cleaned,
        }

    aliases = _load_app_aliases()
    web_alias = aliases.get("web", {}).get(cleaned.lower())
    if web_alias:
        return {
            "launch_target": web_alias,
            "resolved": web_alias,
            "target_type": "url",
            "display_name": cleaned,
        }

    app_aliases = aliases.get("apps", {})
    executable = app_aliases.get(cleaned.lower(), cleaned)
    found = shutil.which(executable)
    if found:
        return {
            "launch_target": found,
            "resolved": found,
            "target_type": "command",
            "display_name": cleaned,
        }
    if sys.platform.startswith("win") and cleaned.lower() in app_aliases:
        return {
            "launch_target": executable,
            "resolved": executable,
            "target_type": "shell",
            "display_name": cleaned,
        }

    if cleaned.startswith(("http://", "https://")):
        return {
            "launch_target": cleaned,
            "resolved": cleaned,
            "target_type": "url",
            "display_name": cleaned,
        }

    raise RuntimeError(
        f"Couldn't find an installed app or alias for '{cleaned}'. Add one in {APP_ALIASES_FILE.name} if needed."
    )


def _load_app_aliases() -> dict[str, dict[str, str]]:
    default_aliases = {
        "apps": {
            "vscode": "code",
            "vs code": "code",
            "visual studio code": "code",
            "chrome": "chrome",
            "google chrome": "chrome",
            "edge": "msedge",
            "notepad": "notepad",
            "calculator": "calc",
            "cmd": "cmd",
            "powershell": "powershell",
            "terminal": "wt",
            "spotify": "spotify",
        },
        "web": {
            "instagram": "https://www.instagram.com/",
            "youtube": "https://www.youtube.com/",
            "gmail": "https://mail.google.com/",
            "github": "https://github.com/",
        },
    }
    if not APP_ALIASES_FILE.exists():
        return default_aliases
    try:
        payload = json.loads(APP_ALIASES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_aliases
    if not isinstance(payload, dict):
        return default_aliases

    merged = {"apps": dict(default_aliases["apps"]), "web": dict(default_aliases["web"])}
    for key in ("apps", "web"):
        section = payload.get(key, {})
        if not isinstance(section, dict):
            continue
        for alias, resolved in section.items():
            alias_text = str(alias).strip().lower()
            resolved_text = str(resolved).strip()
            if alias_text and resolved_text:
                merged[key][alias_text] = resolved_text
    return merged
