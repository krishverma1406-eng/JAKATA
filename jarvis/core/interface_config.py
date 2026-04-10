"""Generic data_ai-backed interface configuration loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config.settings import DATA_AI_DIR


class InterfaceConfig:
    """Load interface and conversation mode metadata from data_ai JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DATA_AI_DIR / "conversation_modes.json")
        self._cache: dict[str, Any] | None = None
        self._cache_mtime_ns: int | None = None

    def payload(self) -> dict[str, Any]:
        if not self.path.exists():
            self._cache = {}
            self._cache_mtime_ns = None
            return {}

        mtime_ns = self.path.stat().st_mtime_ns
        if self._cache is not None and self._cache_mtime_ns == mtime_ns:
            return dict(self._cache)

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        self._cache = raw
        self._cache_mtime_ns = mtime_ns
        return dict(raw)

    def default_mode(self) -> str:
        payload = self.payload()
        requested = self.normalize_mode(payload.get("default_mode"))
        available = {item["key"] for item in self.list_modes()}
        if requested and requested in available:
            return requested
        if available:
            return sorted(available)[0]
        return "normal"

    def list_modes(self) -> list[dict[str, Any]]:
        payload = self.payload()
        modes = payload.get("modes", {})
        if not isinstance(modes, dict):
            return []

        items: list[dict[str, Any]] = []
        for key, value in modes.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item["key"] = self.normalize_mode(key)
            item["label"] = str(item.get("label") or key).strip() or key
            item["focus_filter"] = item.get("focus_filter", {}) if isinstance(item.get("focus_filter", {}), dict) else {}
            items.append(item)
        return [item for item in items if item.get("key")]

    def get_mode(self, mode: str | None) -> dict[str, Any]:
        normalized = self.normalize_mode(mode)
        mode_map = {item["key"]: item for item in self.list_modes()}
        selected = mode_map.get(normalized) or mode_map.get(self.default_mode())
        if selected:
            result = dict(selected)
            result["focus_filter"] = dict(selected.get("focus_filter", {}))
            return result
        return {
            "key": "normal",
            "label": "Normal",
            "summary": "",
            "system_prompt": "",
            "show_debug": False,
            "briefing_on_start": False,
            "focus_filter": {},
        }

    def briefing(self) -> dict[str, Any]:
        payload = self.payload()
        briefing = payload.get("briefing", {})
        return dict(briefing) if isinstance(briefing, dict) else {}

    @staticmethod
    def normalize_mode(mode: Any) -> str:
        cleaned = re.sub(r"[^a-z0-9_-]+", "-", str(mode or "").strip().lower()).strip("-")
        return cleaned or "normal"


INTERFACE_CONFIG = InterfaceConfig()
