"""Dynamic tool discovery and execution."""

from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path
from typing import Any, Callable

from config.settings import DATA_AI_DIR, SETTINGS, TOOLS_DIR, Settings


class ToolRegistry:
    """Scan the tools directory and expose definitions plus tool execution."""

    _tool_guideline_cache: dict[str, str] | None = None

    def __init__(
        self,
        tools_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.tools_dir = Path(tools_dir or TOOLS_DIR)
        self.reload_interval = self.settings.tool_reload_seconds
        self._last_scan = 0.0
        self._last_signature: tuple[tuple[str, int], ...] = ()
        self._definitions: dict[str, dict[str, Any]] = {}
        self._executors: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._errors: dict[str, str] = {}
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        signature = self._signature()
        if not force and signature == self._last_signature and (time.time() - self._last_scan) < self.reload_interval:
            return

        self._definitions.clear()
        self._executors.clear()
        self._errors.clear()

        if not self.tools_dir.exists():
            self._last_scan = time.time()
            return

        for path in sorted(self.tools_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            try:
                self._load_tool(path)
            except Exception as exc:
                self._errors[path.stem] = str(exc)

        self._last_scan = time.time()
        self._last_signature = signature

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        self.refresh()
        return [self._definitions[name] for name in sorted(self._definitions)]

    def list_tool_names(self) -> list[str]:
        return [tool["name"] for tool in self.get_tool_definitions()]

    def has_tool(self, name: str) -> bool:
        self.refresh()
        return name in self._executors

    def run_tool(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.refresh()
        if name not in self._executors:
            return {"ok": False, "name": name, "error": f"Tool '{name}' is not registered."}

        try:
            payload = dict(params or {})
            if runtime_context:
                payload["_runtime_context"] = dict(runtime_context)
            result = self._executors[name](payload)
            return {"ok": True, "name": name, "result": result}
        except Exception as exc:
            return {"ok": False, "name": name, "error": str(exc)}

    @property
    def errors(self) -> dict[str, str]:
        self.refresh()
        return dict(self._errors)

    @staticmethod
    def rank_tool_definitions(
        query: str,
        tool_definitions: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        query_text = ToolRegistry._normalize_match_text(query)
        query_tokens = ToolRegistry._meaningful_tokens(query_text)
        if not query_tokens:
            return []

        ranked: list[tuple[dict[str, Any], float]] = []
        for tool_definition in tool_definitions:
            score = ToolRegistry._score_tool_definition(query_text, query_tokens, tool_definition)
            if score > 0:
                ranked.append((tool_definition, score))
        ranked.sort(key=lambda item: (-item[1], item[0]["name"]))
        return ranked

    @staticmethod
    def best_matching_tool_name(
        query: str,
        tool_definitions: list[dict[str, Any]],
        min_score: float = 2.0,
    ) -> str | None:
        ranked = ToolRegistry.rank_tool_definitions(query, tool_definitions)
        if not ranked:
            return None
        best_definition, best_score = ranked[0]
        if best_score < min_score:
            return None
        return str(best_definition["name"])

    def _load_tool(self, path: Path) -> None:
        module_name = f"jarvis_tool_{path.stem}_{path.stat().st_mtime_ns}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        definition = getattr(module, "TOOL_DEFINITION", None)
        execute = getattr(module, "execute", None)
        if not isinstance(definition, dict):
            raise RuntimeError("TOOL_DEFINITION dict is missing.")
        if not callable(execute):
            raise RuntimeError("execute(params) is missing.")

        name = definition.get("name")
        description = definition.get("description")
        parameters = definition.get("parameters")
        if not name or not isinstance(name, str):
            raise RuntimeError("Tool name must be a non-empty string.")
        if not description or not isinstance(description, str):
            raise RuntimeError("Tool description must be a non-empty string.")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise RuntimeError("Tool parameters must be a JSON-schema object.")

        self._definitions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._executors[name] = execute

    @staticmethod
    def _score_tool_definition(
        query_text: str,
        query_tokens: list[str],
        tool_definition: dict[str, Any],
    ) -> float:
        search_text = ToolRegistry._tool_search_text(tool_definition)
        search_tokens = set(ToolRegistry._meaningful_tokens(search_text))
        if not search_tokens:
            return 0.0

        name_tokens = set(ToolRegistry._meaningful_tokens(str(tool_definition.get("name", "")).replace("_", " ")))
        overlap = [token for token in query_tokens if token in search_tokens]
        if not overlap:
            return 0.0

        score = 0.0
        for token in overlap:
            score += 2.5 if token in name_tokens else 1.0

        name_text = str(tool_definition.get("name", "")).lower()
        name_aliases = {
            name_text,
            name_text.replace("_", " "),
            name_text.replace("_tool", "").replace("_", " "),
        }
        for alias in name_aliases:
            alias = alias.strip()
            if alias and alias in query_text:
                score += 3.0

        description = str(tool_definition.get("description", "")).lower()
        if description:
            description_tokens = ToolRegistry._meaningful_tokens(description)
            phrase_hits = 0
            for first, second in zip(description_tokens, description_tokens[1:]):
                phrase = f"{first} {second}"
                if phrase in query_text:
                    phrase_hits += 1
            score += min(phrase_hits, 3) * 0.5

        return score

    @staticmethod
    def _tool_search_text(tool_definition: dict[str, Any]) -> str:
        parts: list[str] = [
            str(tool_definition.get("name", "")),
            str(tool_definition.get("description", "")),
        ]
        guideline_text = ToolRegistry._tool_guideline_text(str(tool_definition.get("name", "")).strip())
        if guideline_text:
            parts.append(guideline_text)
        parameters = tool_definition.get("parameters", {})
        if isinstance(parameters, dict):
            parts.append(str(parameters.get("description", "")))
            properties = parameters.get("properties", {})
            if isinstance(properties, dict):
                for property_name, schema in properties.items():
                    parts.append(str(property_name))
                    if isinstance(schema, dict):
                        parts.append(str(schema.get("description", "")))
                        enum_values = schema.get("enum", [])
                        if isinstance(enum_values, list):
                            parts.extend(str(value) for value in enum_values)
        return ToolRegistry._normalize_match_text(" ".join(parts))

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).lower().replace("_", " ")).strip()

    @staticmethod
    def _meaningful_tokens(value: str) -> list[str]:
        stopwords = {
            "a",
            "about",
            "action",
            "an",
            "and",
            "are",
            "as",
            "asks",
            "at",
            "be",
            "by",
            "can",
            "current",
            "do",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "perform",
            "please",
            "sir",
            "tell",
            "the",
            "this",
            "tool",
            "tools",
            "today",
            "to",
            "user",
            "users",
            "what",
            "with",
            "you",
        }
        tokens = re.findall(r"[a-z0-9]+", value.lower())
        return [token for token in tokens if len(token) > 1 and token not in stopwords]

    @classmethod
    def _tool_guideline_text(cls, tool_name: str) -> str:
        if cls._tool_guideline_cache is None:
            cls._tool_guideline_cache = cls._load_tool_guidelines()
        return cls._tool_guideline_cache.get(tool_name, "")

    @staticmethod
    def _load_tool_guidelines() -> dict[str, str]:
        path = DATA_AI_DIR / "tool_guidelines.md"
        if not path.exists():
            return {}

        guidance: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
            if len(cells) < 3:
                continue
            tool_cell = cells[0]
            if not tool_cell.startswith("`") or tool_cell.startswith("`---"):
                continue
            tool_name = tool_cell.strip("` ").strip()
            if not tool_name:
                continue
            guidance[tool_name] = ToolRegistry._normalize_match_text(" ".join(cells[1:]))
        return guidance

    def _signature(self) -> tuple[tuple[str, int], ...]:
        if not self.tools_dir.exists():
            return ()
        signature: list[tuple[str, int]] = []
        for path in sorted(self.tools_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            signature.append((path.name, path.stat().st_mtime_ns))
        return tuple(signature)
