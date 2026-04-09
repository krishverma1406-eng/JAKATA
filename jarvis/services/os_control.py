"""Desktop control helpers using pyautogui."""

from __future__ import annotations

import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR

SCREENSHOT_DIR = BASE_DIR / "data_user" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


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
    if sys.platform.startswith("win"):
        subprocess.Popen(["cmd", "/c", "start", "", resolved], shell=False)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", resolved], shell=False)
    else:
        subprocess.Popen(["xdg-open", resolved], shell=False)
    return {"opened": target, "resolved": resolved}


def _load_gui() -> Any:
    try:
        import pyautogui
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pyautogui is not installed.") from exc

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def _resolve_app_target(target: str) -> str:
    cleaned = str(target).strip()
    if not cleaned:
        raise RuntimeError("App target is required.")

    candidate = Path(cleaned)
    if candidate.exists():
        return str(candidate)

    aliases = {
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
    }
    executable = aliases.get(cleaned.lower(), cleaned)
    found = shutil.which(executable)
    if found:
        return found
    return cleaned
