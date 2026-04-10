"""Guarded terminal access for PowerShell and cmd workflows."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR

_DEFAULT_TIMEOUT = 20
_MAX_TIMEOUT = 120
_DEFAULT_OUTPUT_LIMIT = 12000
_MAX_OUTPUT_LIMIT = 30000
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DANGEROUS_TOKENS = (
    "remove-item",
    "del ",
    "erase ",
    "rd ",
    "rmdir",
    "format ",
    "diskpart",
    "reg delete",
    "shutdown",
    "restart-computer",
    "stop-computer",
    "clear-disk",
    "cipher /w",
    "bcdedit",
)

TOOL_DEFINITION = {
    "name": "terminal_tool",
    "description": "Run PowerShell or cmd commands with timeouts, inspect command help, resolve commands, and start local processes when terminal access is the right fallback.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "help", "which", "start_process"],
                "description": "Terminal action to perform.",
            },
            "shell": {
                "type": "string",
                "enum": ["powershell", "cmd"],
                "description": "Which shell to use for run or help.",
                "default": "powershell",
            },
            "command": {
                "type": "string",
                "description": "Shell command to execute when action is run.",
            },
            "topic": {
                "type": "string",
                "description": "Command or topic name for help or which.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for command execution. Relative paths resolve from the JARVIS project root.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Command timeout in seconds.",
                "default": 20,
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Maximum stdout or stderr characters to return.",
                "default": 12000,
            },
            "input_text": {
                "type": "string",
                "description": "Optional stdin text passed to the command.",
            },
            "program": {
                "type": "string",
                "description": "Executable, app, document, or script to launch when action is start_process.",
            },
            "arguments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Argument list for start_process.",
            },
            "wait": {
                "type": "boolean",
                "description": "Wait for the started process to finish when action is start_process.",
                "default": False,
            },
            "confirm_dangerous": {
                "type": "boolean",
                "description": "Required before running obviously destructive terminal commands.",
                "default": False,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    if action == "run":
        return _run_command(params)
    if action == "help":
        return _command_help(params)
    if action == "which":
        return _which_command(params)
    if action == "start_process":
        return _start_process(params)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _run_command(params: dict[str, Any]) -> dict[str, Any]:
    shell = _normalize_shell(params.get("shell"))
    command = str(params.get("command", "")).strip()
    if not command:
        return {"ok": False, "error": "command is required for run."}

    dangerous = _detect_danger(command)
    if dangerous and not bool(params.get("confirm_dangerous", False)):
        return {
            "ok": False,
            "blocked": True,
            "error": (
                "This terminal command looks destructive. Re-run with "
                "confirm_dangerous=true only after the user explicitly confirms it."
            ),
            "danger_reason": dangerous,
            "command": command,
            "shell": shell,
        }

    timeout_seconds = _timeout_seconds(params.get("timeout_seconds"))
    max_output_chars = _output_limit(params.get("max_output_chars"))
    cwd = _resolve_cwd(str(params.get("cwd", "")).strip())
    argv = _shell_argv(shell, command)

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
            timeout=timeout_seconds,
            input=str(params.get("input_text", "")) if params.get("input_text") is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_value, stdout_truncated = _truncate_text(stdout, max_output_chars)
        stderr_value, stderr_truncated = _truncate_text(stderr, max_output_chars)
        return {
            "ok": False,
            "shell": shell,
            "command": command,
            "cwd": str(cwd),
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": stdout_value,
            "stderr": stderr_value,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
    except OSError as exc:
        return {"ok": False, "shell": shell, "command": command, "cwd": str(cwd), "error": str(exc)}

    stdout_value, stdout_truncated = _truncate_text(completed.stdout, max_output_chars)
    stderr_value, stderr_truncated = _truncate_text(completed.stderr, max_output_chars)
    return {
        "ok": completed.returncode == 0,
        "shell": shell,
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": stdout_value,
        "stderr": stderr_value,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def _command_help(params: dict[str, Any]) -> dict[str, Any]:
    shell = _normalize_shell(params.get("shell"))
    topic = str(params.get("topic", "")).strip()
    if not topic:
        return {"ok": False, "error": "topic is required for help."}

    if shell == "powershell":
        command = f"Get-Help {topic} -Full"
    else:
        command = f"help {topic}"
    result = _run_command(
        {
            "action": "run",
            "shell": shell,
            "command": command,
            "cwd": params.get("cwd", ""),
            "timeout_seconds": params.get("timeout_seconds", _DEFAULT_TIMEOUT),
            "max_output_chars": params.get("max_output_chars", _DEFAULT_OUTPUT_LIMIT),
        }
    )
    result["topic"] = topic
    return result


def _which_command(params: dict[str, Any]) -> dict[str, Any]:
    topic = str(params.get("topic", "")).strip() or str(params.get("program", "")).strip()
    if not topic:
        return {"ok": False, "error": "topic is required for which."}

    lowered = topic.lower()
    if lowered in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        resolved_shell = _shell_argv("powershell", "$PSVersionTable.PSVersion.ToString()")[0]
        return {"ok": True, "topic": topic, "path": resolved_shell, "command_type": "shell"}
    if lowered in {"cmd", "cmd.exe"}:
        resolved_shell = _shell_argv("cmd", "ver")[0]
        return {"ok": True, "topic": topic, "path": resolved_shell, "command_type": "shell"}

    resolved = shutil.which(topic)
    if resolved:
        return {"ok": True, "topic": topic, "path": resolved, "command_type": "executable"}

    result = _run_command(
        {
            "action": "run",
            "shell": "powershell",
            "command": f"Get-Command {topic} | Select-Object Name, CommandType, Source | Format-List",
            "cwd": params.get("cwd", ""),
            "timeout_seconds": params.get("timeout_seconds", _DEFAULT_TIMEOUT),
            "max_output_chars": params.get("max_output_chars", _DEFAULT_OUTPUT_LIMIT),
        }
    )
    result["topic"] = topic
    return result


def _start_process(params: dict[str, Any]) -> dict[str, Any]:
    program = str(params.get("program", "")).strip()
    if not program:
        return {"ok": False, "error": "program is required for start_process."}

    arguments = [str(item) for item in params.get("arguments", []) if str(item).strip()]
    wait = bool(params.get("wait", False))
    cwd = _resolve_cwd(str(params.get("cwd", "")).strip())

    command = [
        "Start-Process",
        "-FilePath",
        _ps_quote(program),
    ]
    if arguments:
        quoted_arguments = ", ".join(_ps_quote(argument) for argument in arguments)
        command.extend(["-ArgumentList", quoted_arguments])
    command.extend(["-WorkingDirectory", _ps_quote(str(cwd))])
    if wait:
        command.append("-Wait")
    command.append("-PassThru")
    command.extend(["|", "Select-Object", "Id, ProcessName, Path", "|", "Format-List"])

    result = _run_command(
        {
            "action": "run",
            "shell": "powershell",
            "command": " ".join(command),
            "cwd": str(cwd),
            "timeout_seconds": params.get("timeout_seconds", _DEFAULT_TIMEOUT),
            "max_output_chars": params.get("max_output_chars", _DEFAULT_OUTPUT_LIMIT),
        }
    )
    result["program"] = program
    result["arguments"] = arguments
    result["wait"] = wait
    return result


def _normalize_shell(value: Any) -> str:
    shell = str(value or "powershell").strip().lower()
    return "cmd" if shell == "cmd" else "powershell"


def _timeout_seconds(value: Any) -> int:
    return max(1, min(int(value or _DEFAULT_TIMEOUT), _MAX_TIMEOUT))


def _output_limit(value: Any) -> int:
    return max(500, min(int(value or _DEFAULT_OUTPUT_LIMIT), _MAX_OUTPUT_LIMIT))


def _resolve_cwd(value: str) -> Path:
    if not value:
        return BASE_DIR
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
    if not resolved.exists():
        raise OSError(f"Working directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise OSError(f"Working directory is not a directory: {resolved}")
    return resolved


def _shell_argv(shell: str, command: str) -> list[str]:
    if shell == "cmd":
        cmd_executable = shutil.which("cmd.exe") or shutil.which("cmd") or "cmd.exe"
        return [cmd_executable, "/d", "/c", command]
    powershell_executable = (
        shutil.which("powershell.exe")
        or shutil.which("powershell")
        or shutil.which("pwsh.exe")
        or shutil.which("pwsh")
        or "powershell.exe"
    )
    return [
        powershell_executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    value = _strip_ansi(value)
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _detect_danger(command: str) -> str:
    lowered = f" {command.lower()} "
    for token in _DANGEROUS_TOKENS:
        if token in lowered:
            return f"matched token '{token.strip()}'"
    return ""


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", value)
