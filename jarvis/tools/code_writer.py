"""Generate and validate new tools inside the tools folder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from config.settings import DATA_AI_DIR, SETTINGS, TOOLS_DIR
from core.brain import Brain

MAX_RETRIES = 3


TOOL_DEFINITION = {
    "name": "code_writer",
    "description": "Generate or repair a tool file, validate it, dry-run it, and log the result.",
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "File-safe tool name without .py extension.",
            },
            "description": {
                "type": "string",
                "description": "What the tool should do.",
            },
            "code": {
                "type": "string",
                "description": "Optional full source code. When omitted, the model generates it.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow replacing an existing tool file.",
                "default": False,
            },
        },
        "required": ["tool_name", "description"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(params.get("tool_name", "")).strip().replace(".py", "")
    description = str(params.get("description", "")).strip()
    overwrite = bool(params.get("overwrite", False))
    user_code = str(params.get("code", "")).strip()

    if not tool_name or not description:
        return {"ok": False, "error": "tool_name and description are required."}
    if not tool_name.replace("_", "").isalnum():
        return {"ok": False, "error": "tool_name must be alphanumeric or underscore."}

    target = TOOLS_DIR / f"{tool_name}.py"
    if target.exists() and not overwrite:
        return {
            "ok": False,
            "error": "Tool file already exists. Set overwrite=true to replace it.",
            "path": str(target),
        }

    brain = Brain(SETTINGS)
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        source = user_code if user_code and attempt == 1 else _generate_tool_code(brain, tool_name, description, last_error)
        target.write_text(source.rstrip() + "\n", encoding="utf-8")

        validation = _validate_tool_file(target)
        if validation["ok"]:
            _append_self_update_log(tool_name, description, target, attempt)
            return {
                "ok": True,
                "path": str(target),
                "tool_name": tool_name,
                "attempts": attempt,
                "validation": validation,
            }
        last_error = validation["error"]

    return {
        "ok": False,
        "error": f"Tool generation failed after {MAX_RETRIES} attempts. Last error: {last_error}",
        "path": str(target),
    }


def _generate_tool_code(brain: Brain, tool_name: str, description: str, last_error: str) -> str:
    prompt = (
        "Generate a complete Python tool module for this JARVIS system.\n"
        "Return only valid Python source code.\n"
        "Requirements:\n"
        "- Must define TOOL_DEFINITION as a dict.\n"
        "- TOOL_DEFINITION.name must match the requested tool name.\n"
        "- Must define execute(params: dict[str, Any]) -> dict[str, Any].\n"
        "- Use only standard library unless absolutely required.\n"
        "- Return JSON-serializable data.\n"
        "- Add concise docstrings.\n"
        f"Requested tool name: {tool_name}\n"
        f"Requested behavior: {description}\n"
    )
    if last_error:
        prompt += f"Previous validation error to fix: {last_error}\n"

    response = brain.chat(
        messages=[{"role": "user", "content": prompt}],
        task_kind="code",
        tools=[],
    )
    content = response.get("content", "").strip()
    return _strip_code_fences(content)


def _validate_tool_file(path: Path) -> dict[str, Any]:
    try:
        spec = importlib.util.spec_from_file_location(f"generated_{path.stem}", path)
        if spec is None or spec.loader is None:
            return {"ok": False, "error": "Could not load generated module spec."}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        return {"ok": False, "error": f"Import failed: {exc}"}

    definition = getattr(module, "TOOL_DEFINITION", None)
    execute = getattr(module, "execute", None)
    if not isinstance(definition, dict):
        return {"ok": False, "error": "TOOL_DEFINITION dict is missing."}
    if not callable(execute):
        return {"ok": False, "error": "execute(params) is missing."}

    name = definition.get("name")
    params_schema = definition.get("parameters")
    if not isinstance(name, str) or not name:
        return {"ok": False, "error": "TOOL_DEFINITION.name must be a non-empty string."}
    if not isinstance(params_schema, dict):
        return {"ok": False, "error": "TOOL_DEFINITION.parameters must be a dict."}

    dummy_params = _dummy_params_for_schema(params_schema)
    try:
        result = execute(dummy_params)
    except Exception as exc:
        return {"ok": False, "error": f"Dry-run execute() failed: {exc}"}
    if not isinstance(result, dict):
        return {"ok": False, "error": "Dry-run execute() must return a dict."}
    return {"ok": True, "tool_name": name, "dry_run": result}


def _dummy_params_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    dummy: dict[str, Any] = {}
    for field in required:
        field_schema = properties.get(field, {})
        field_type = field_schema.get("type")
        if "enum" in field_schema and field_schema["enum"]:
            dummy[field] = field_schema["enum"][0]
        elif field_type == "string":
            dummy[field] = "example"
        elif field_type == "integer":
            dummy[field] = 1
        elif field_type == "number":
            dummy[field] = 1.0
        elif field_type == "boolean":
            dummy[field] = False
        elif field_type == "array":
            dummy[field] = []
        elif field_type == "object":
            dummy[field] = {}
        else:
            dummy[field] = None
    return dummy


def _append_self_update_log(tool_name: str, description: str, path: Path, attempt: int) -> None:
    log_path = DATA_AI_DIR / "self_update_log.md"
    if not log_path.exists():
        log_path.write_text("## Self Update Log\n\n", encoding="utf-8")
    lines = [
        f"### {tool_name}",
        f"- Description: {description}",
        f"- Path: {path}",
        f"- Attempts: {attempt}",
        "",
    ]
    existing = log_path.read_text(encoding="utf-8")
    if "No self-generated tools have been created yet." in existing:
        existing = existing.replace("No self-generated tools have been created yet.\n", "")
    log_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
