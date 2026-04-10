"""Desktop control helpers using pyautogui."""

from __future__ import annotations

import contextlib
import json
import subprocess
import shutil
import sys
import time
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


def close_app(target: str, force: bool = False, timeout: float = 2.0) -> dict[str, Any]:
    identity = _app_identity(target)
    display_name = str(identity.get("display_name", "")).strip() or str(target).strip() or "the app"
    processes = _matching_processes(identity)
    if not processes:
        return {
            "target": display_name,
            "matched": 0,
            "closed": 0,
            "remaining": 0,
            "force": force,
            "verified_closed": True,
        }

    closed_pids: set[int] = set()
    if sys.platform.startswith("win"):
        for process in processes:
            if _request_window_close(process.pid):
                closed_pids.add(process.pid)
    else:
        for process in processes:
            with contextlib.suppress(Exception):
                process.terminate()
                closed_pids.add(int(process.pid))

    deadline = time.time() + max(0.2, float(timeout or 0.0))
    while time.time() < deadline:
        remaining = _matching_processes(identity)
        if not remaining:
            break
        time.sleep(0.1)

    remaining = _matching_processes(identity)
    if remaining and force:
        for process in remaining:
            with contextlib.suppress(Exception):
                process.kill()
                closed_pids.add(int(process.pid))
        kill_deadline = time.time() + 1.5
        while time.time() < kill_deadline:
            remaining = _matching_processes(identity)
            if not remaining:
                break
            time.sleep(0.1)

    remaining = _matching_processes(identity)
    return {
        "target": display_name,
        "matched": len(processes),
        "closed": max(0, len(processes) - len(remaining)),
        "remaining": len(remaining),
        "force": force,
        "verified_closed": not remaining,
    }


def _load_gui() -> Any:
    try:
        import pyautogui
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("pyautogui is not installed.") from exc

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def _load_psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("psutil is not installed.") from exc
    return psutil


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


def _app_identity(target: str) -> dict[str, Any]:
    cleaned = str(target).strip()
    if not cleaned:
        raise RuntimeError("App target is required.")

    aliases = _load_app_aliases()
    executable = aliases.get("apps", {}).get(cleaned.lower(), cleaned)
    process_names = {
        _normalize_process_token(cleaned),
        _normalize_process_token(executable),
        _normalize_process_token(Path(cleaned).stem),
        _normalize_process_token(Path(executable).stem),
    }
    process_names.discard("")

    resolved_target = executable
    with contextlib.suppress(RuntimeError):
        resolved = _resolve_app_target(cleaned)
        resolved_target = str(resolved.get("resolved", "")).strip() or resolved_target
        process_names.add(_normalize_process_token(Path(resolved_target).stem))

    return {
        "display_name": cleaned,
        "resolved": resolved_target,
        "process_names": process_names,
    }


def _normalize_process_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if token.endswith(".exe"):
        token = token[:-4]
    return token


def _matching_processes(identity: dict[str, Any]) -> list[Any]:
    psutil = _load_psutil()
    wanted = {str(item).strip().lower() for item in identity.get("process_names", set()) if str(item).strip()}
    if not wanted:
        return []

    matches: list[Any] = []
    for process in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = _normalize_process_token(str(process.info.get("name", "") or ""))
            exe = _normalize_process_token(Path(str(process.info.get("exe", "") or "")).stem)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if name in wanted or exe in wanted:
            matches.append(process)
    return matches


def _request_window_close(pid: int) -> bool:
    if not sys.platform.startswith("win"):
        return False

    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False

    user32 = ctypes.windll.user32
    WM_CLOSE = 0x0010

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found = {"sent": False}

    def callback(hwnd: int, _lparam: int) -> bool:
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if int(window_pid.value) != int(pid):
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        found["sent"] = True
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return bool(found["sent"])


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
