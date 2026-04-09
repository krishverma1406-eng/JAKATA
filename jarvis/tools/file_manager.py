"""Enhanced file management with recursive search and rich document reads."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR

_WATCH_STATE: dict[str, dict[str, float]] = {}
_TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".sql", ".ps1", ".sh",
    ".csv", ".log", ".xml",
}

TOOL_DEFINITION = {
    "name": "file_manager",
    "description": "Read, write, list, search, summarize, and watch files on disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read", "write", "find", "watch"],
                "description": "Which file action to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path for the action.",
            },
            "content": {
                "type": "string",
                "description": "Content to write when action is write.",
            },
            "pattern": {
                "type": "string",
                "description": "Filename or glob pattern for search.",
            },
            "allow_overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an existing file during write.",
                "default": False,
            },
            "recursive": {
                "type": "boolean",
                "description": "Search/list recursively when applicable.",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return for list/find/watch.",
                "default": 25,
            },
            "summarize": {
                "type": "boolean",
                "description": "Summarize long file content before returning.",
                "default": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    path_value = str(params.get("path", ".")).strip() or "."
    target = _resolve_path(path_value)
    recursive = bool(params.get("recursive", True))
    max_results = max(1, min(int(params.get("max_results", 25) or 25), 200))
    summarize = bool(params.get("summarize", True))

    if action == "list":
        return _list_path(target, recursive=recursive, max_results=max_results)
    if action == "read":
        return _read_file(target, summarize=summarize)
    if action == "write":
        return _write_file(target, params)
    if action == "find":
        return _find_files(target, str(params.get("pattern", "")).strip(), recursive=recursive, max_results=max_results)
    if action == "watch":
        return _watch_path(target, recursive=recursive, max_results=max_results)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _list_path(target: Path, recursive: bool, max_results: int) -> dict[str, Any]:
    if not target.exists():
        return {"ok": False, "error": f"Path does not exist: {target}"}
    if not target.is_dir():
        return {"ok": False, "error": f"Path is not a directory: {target}"}

    items: list[dict[str, Any]] = []
    iterator = target.rglob("*") if recursive else target.iterdir()
    for path in iterator:
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "type": "dir" if path.is_dir() else "file",
                "size": stat.st_size,
            }
        )
        if len(items) >= max_results:
            break

    return {"ok": True, "path": str(target), "items": items}


def _read_file(target: Path, summarize: bool) -> dict[str, Any]:
    if not target.exists():
        return {"ok": False, "error": f"File does not exist: {target}"}
    if target.is_dir():
        return {"ok": False, "error": f"Path is a directory, not a file: {target}"}

    suffix = target.suffix.lower()
    if suffix == ".pdf":
        content = _read_pdf(target)
    elif suffix == ".docx":
        content = _read_docx(target)
    elif suffix in _TEXT_EXTENSIONS or not suffix:
        content = target.read_text(encoding="utf-8", errors="replace")
    else:
        return {
            "ok": True,
            "path": str(target),
            "content": "",
            "summary": f"Binary or unsupported text format: {target.suffix or 'unknown'}",
            "content_truncated": False,
        }

    summary, final_content, truncated = _summarize_if_needed(content, summarize)
    return {
        "ok": True,
        "path": str(target),
        "content": final_content,
        "summary": summary,
        "content_truncated": truncated,
    }


def _write_file(target: Path, params: dict[str, Any]) -> dict[str, Any]:
    content = str(params.get("content", ""))
    allow_overwrite = bool(params.get("allow_overwrite", False))
    if target.exists() and not allow_overwrite:
        return {
            "ok": False,
            "error": "Target file already exists. Set allow_overwrite=true to replace it.",
            "path": str(target),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(target), "bytes_written": len(content.encode("utf-8"))}


def _find_files(target: Path, pattern: str, recursive: bool, max_results: int) -> dict[str, Any]:
    query = pattern or target.name or "*"
    search_root = target if target.exists() and target.is_dir() else target.parent if target.parent.exists() else _resolve_path(".")
    candidates = list(search_root.rglob("*") if recursive else search_root.iterdir())
    matches: list[dict[str, Any]] = []

    for path in candidates:
        if path.name.startswith(".") and path.is_dir():
            continue
        name = path.name.lower()
        query_lower = query.lower()
        score = _match_score(query_lower, name)
        if score <= 0:
            continue
        matches.append(
            {
                "path": str(path),
                "name": path.name,
                "type": "dir" if path.is_dir() else "file",
                "score": round(score, 3),
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {"ok": True, "path": str(search_root), "matches": matches[:max_results]}


def _watch_path(target: Path, recursive: bool, max_results: int) -> dict[str, Any]:
    watch_root = target if target.exists() and target.is_dir() else target.parent if target.exists() else _resolve_path(".")
    snapshot = _snapshot_tree(watch_root, recursive=recursive)
    snapshot_key = str(watch_root)
    previous = _WATCH_STATE.get(snapshot_key)
    _WATCH_STATE[snapshot_key] = snapshot

    if previous is None:
        return {"ok": True, "path": str(watch_root), "initialized": True, "changes": []}

    changes: list[dict[str, Any]] = []
    all_paths = set(previous) | set(snapshot)
    for path in sorted(all_paths):
        if path not in previous:
            changes.append({"path": path, "change": "created"})
        elif path not in snapshot:
            changes.append({"path": path, "change": "deleted"})
        elif previous[path] != snapshot[path]:
            changes.append({"path": path, "change": "modified"})
        if len(changes) >= max_results:
            break
    return {"ok": True, "path": str(watch_root), "initialized": False, "changes": changes}


def _snapshot_tree(root: Path, recursive: bool) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        try:
            snapshot[str(path)] = path.stat().st_mtime
        except OSError:
            continue
    return snapshot


def _match_score(query: str, candidate: str) -> float:
    if not query or query == "*":
        return 1.0
    if query in candidate:
        return 1.0 + (len(query) / max(len(candidate), 1))
    return difflib.SequenceMatcher(None, query, candidate).ratio()


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(part.strip() for part in text_parts if part.strip())


def _read_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    return "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())


def _summarize_if_needed(content: str, summarize: bool) -> tuple[str, str, bool]:
    normalized = content.strip()
    if not normalized:
        return "", "", False
    if not summarize or len(normalized) <= 4000:
        return "", normalized, False

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    head = lines[:8]
    tail = lines[-5:] if len(lines) > 10 else []
    summary_lines = ["Long file summarized for brevity."]
    if head:
        summary_lines.append("Start:")
        summary_lines.extend(f"- {line[:200]}" for line in head[:5])
    if tail:
        summary_lines.append("End:")
        summary_lines.extend(f"- {line[:200]}" for line in tail[:3])
    return "\n".join(summary_lines), normalized[:4000], True


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()
