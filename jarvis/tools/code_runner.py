"""Execute Python code snippets in an isolated subprocess."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


TOOL_DEFINITION = {
    "name": "code_runner",
    "description": (
        "Execute Python code and return output. Use for running small scripts, testing functions, "
        "doing data processing, or verifying code output. Not for package installation or system commands."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum execution time.",
                "default": 15,
            },
            "input_data": {
                "type": "string",
                "description": "Optional stdin input for the script.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

_BLOCKED = (
    "import os",
    "import sys",
    "import subprocess",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals()",
    "locals()",
    "__builtins__",
    "importlib",
    "shutil",
    "socket",
)


def execute(params: dict[str, Any]) -> dict[str, Any]:
    code = str(params.get("code", "")).strip()
    if not code:
        return {"ok": False, "error": "code is required."}

    code_lower = code.lower()
    for blocked in _BLOCKED:
        if blocked.lower() in code_lower:
            return {
                "ok": False,
                "error": (
                    f"Blocked: '{blocked}' is not allowed in code_runner. "
                    "Use terminal_tool for system operations."
                ),
            }

    timeout = max(1, min(int(params.get("timeout_seconds", 15) or 15), 30))
    stdin_data = str(params.get("input_data", "")) or None

    with tempfile.TemporaryDirectory(prefix="jarvis-code-runner-") as temp_dir:
        temp_path = Path(temp_dir)
        script_path = temp_path / "snippet.py"
        script_path.write_text(code + "\n", encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_data,
                cwd=str(temp_path),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Code timed out after {timeout}s",
                "timed_out": True,
            }

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    success = result.returncode == 0
    return {
        "ok": success,
        "exit_code": result.returncode,
        "stdout": stdout[:5000] if stdout else "",
        "stderr": stderr[:2000] if stderr else "",
        "timed_out": False,
    }
