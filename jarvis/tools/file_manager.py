"""Enhanced file management with richer filesystem actions and content search."""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR

_WATCH_STATE: dict[str, dict[str, float]] = {}
_TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".jsx",
    ".rb",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_READ_CHAR_LIMIT = 4000
_MAX_FILES_TO_SCAN = 500

TOOL_DEFINITION = {
    "name": "file_manager",
    "description": "Read, write, list, search, summarize, copy, move, delete, create, and inspect files or folders on disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "read",
                    "write",
                    "find",
                    "search_text",
                    "watch",
                    "exists",
                    "mkdir",
                    "copy",
                    "move",
                    "delete",
                ],
                "description": "Which file action to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path for the action.",
            },
            "destination": {
                "type": "string",
                "description": "Target path used by copy or move actions.",
            },
            "content": {
                "type": "string",
                "description": "Content to write when action is write.",
            },
            "pattern": {
                "type": "string",
                "description": "Filename or glob pattern for search.",
            },
            "query": {
                "type": "string",
                "description": "Text query for search_text.",
            },
            "allow_overwrite": {
                "type": "boolean",
                "description": "Allow replacing an existing file or folder when applicable.",
                "default": False,
            },
            "recursive": {
                "type": "boolean",
                "description": "Search, list, delete, or watch recursively when applicable.",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return for list, find, search_text, or watch.",
                "default": 25,
            },
            "summarize": {
                "type": "boolean",
                "description": "Summarize long file content before returning.",
                "default": True,
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Use case-sensitive matching for search_text.",
                "default": False,
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Include dot-prefixed files and folders in list/find/search actions.",
                "default": False,
            },
            "confirm_destructive": {
                "type": "boolean",
                "description": "Required for destructive actions like delete or overwrite-replace move targets.",
                "default": False,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    path_value = str(params.get("path", ".")).strip() or "."
    target = _resolve_path(path_value, params)
    recursive = bool(params.get("recursive", True))
    max_results = max(1, min(int(params.get("max_results", 25) or 25), 250))
    summarize = bool(params.get("summarize", True))
    include_hidden = bool(params.get("include_hidden", False))

    if action == "list":
        return _list_path(target, recursive=recursive, max_results=max_results, include_hidden=include_hidden)
    if action == "read":
        return _read_file(target, summarize=summarize)
    if action == "write":
        return _write_file(target, params)
    if action == "find":
        return _find_files(
            target,
            str(params.get("pattern", "")).strip(),
            recursive=recursive,
            max_results=max_results,
            include_hidden=include_hidden,
        )
    if action == "search_text":
        return _search_text(
            target,
            str(params.get("query", "")).strip(),
            pattern=str(params.get("pattern", "")).strip(),
            recursive=recursive,
            max_results=max_results,
            case_sensitive=bool(params.get("case_sensitive", False)),
            include_hidden=include_hidden,
        )
    if action == "watch":
        return _watch_path(target, recursive=recursive, max_results=max_results, include_hidden=include_hidden)
    if action == "exists":
        return _exists_path(target)
    if action == "mkdir":
        return _mkdir_path(target)
    if action == "copy":
        return _copy_path(target, params)
    if action == "move":
        return _move_path(target, params)
    if action == "delete":
        return _delete_path(target, params)
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _list_path(target: Path, recursive: bool, max_results: int, include_hidden: bool) -> dict[str, Any]:
    if not target.exists():
        return {"ok": False, "error": f"Path does not exist: {target}"}
    if not target.is_dir():
        return {"ok": False, "error": f"Path is not a directory: {target}"}

    items: list[dict[str, Any]] = []
    iterator = target.rglob("*") if recursive else target.iterdir()
    for path in iterator:
        if not include_hidden and _is_hidden(path, root=target):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": _relative_to_root(path, target),
                "type": "dir" if path.is_dir() else "file",
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
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
    if truncated:
        final_content += (
            f"\n\n[FILE TRUNCATED - showing first {_READ_CHAR_LIMIT} of {len(content)} chars. "
            "Use read with summarize=False and a specific range if you need more.]"
        )
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


def _find_files(
    target: Path,
    pattern: str,
    recursive: bool,
    max_results: int,
    include_hidden: bool,
) -> dict[str, Any]:
    query = pattern or target.name or "*"
    search_root = _resolve_search_root(target)
    matches: list[dict[str, Any]] = []

    if _looks_like_glob(query):
        iterator = search_root.rglob(query) if recursive else search_root.glob(query)
        for path in iterator:
            if not include_hidden and _is_hidden(path, root=search_root):
                continue
            matches.append(
                {
                    "path": str(path),
                    "relative_path": _relative_to_root(path, search_root),
                    "name": path.name,
                    "type": "dir" if path.is_dir() else "file",
                    "score": 1.0,
                }
            )
            if len(matches) >= max_results:
                break
        return {"ok": True, "path": str(search_root), "matches": matches}

    candidates = list(search_root.rglob("*") if recursive else search_root.iterdir())
    query_lower = query.lower()
    for path in candidates:
        if not include_hidden and _is_hidden(path, root=search_root):
            continue
        name = path.name.lower()
        score = _match_score(query_lower, name)
        if score <= 0:
            continue
        matches.append(
            {
                "path": str(path),
                "relative_path": _relative_to_root(path, search_root),
                "name": path.name,
                "type": "dir" if path.is_dir() else "file",
                "score": round(score, 3),
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    return {"ok": True, "path": str(search_root), "matches": matches[:max_results]}


def _search_text(
    target: Path,
    query: str,
    pattern: str,
    recursive: bool,
    max_results: int,
    case_sensitive: bool,
    include_hidden: bool,
) -> dict[str, Any]:
    if not query:
        return {"ok": False, "error": "query is required for search_text."}

    search_root = _resolve_search_root(target)
    file_iter = _iter_candidate_files(target, search_root, recursive=recursive)
    matches: list[dict[str, Any]] = []
    normalized_query = query if case_sensitive else query.lower()
    files_scanned = 0
    scan_limited = False

    for path in file_iter:
        if files_scanned >= _MAX_FILES_TO_SCAN:
            scan_limited = True
            break
        files_scanned += 1
        if not include_hidden and _is_hidden(path, root=search_root):
            continue
        if pattern and not _path_matches_pattern(path, pattern):
            continue
        if not _is_text_candidate(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for index, raw_line in enumerate(content.splitlines(), start=1):
            haystack = raw_line if case_sensitive else raw_line.lower()
            if normalized_query not in haystack:
                continue
            matches.append(
                {
                    "path": str(path),
                    "relative_path": _relative_to_root(path, search_root),
                    "line_number": index,
                    "line": raw_line.strip()[:300],
                }
            )
            if len(matches) >= max_results:
                return {
                    "ok": True,
                    "path": str(search_root),
                    "query": query,
                    "matches": matches,
                    "files_scanned": files_scanned,
                    "scan_limited": scan_limited,
                }

    return {
        "ok": True,
        "path": str(search_root),
        "query": query,
        "matches": matches,
        "files_scanned": files_scanned,
        "scan_limited": scan_limited,
    }


def _watch_path(target: Path, recursive: bool, max_results: int, include_hidden: bool) -> dict[str, Any]:
    watch_root = target if target.exists() and target.is_dir() else target.parent if target.parent.exists() else _resolve_path(".")
    snapshot = _snapshot_tree(watch_root, recursive=recursive, include_hidden=include_hidden)
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


def _exists_path(target: Path) -> dict[str, Any]:
    exists = target.exists()
    data: dict[str, Any] = {
        "ok": True,
        "path": str(target),
        "exists": exists,
        "is_file": target.is_file() if exists else False,
        "is_dir": target.is_dir() if exists else False,
        "is_symlink": target.is_symlink(),
    }
    if exists:
        try:
            stat = target.stat()
            data["size"] = stat.st_size
            data["modified_at"] = stat.st_mtime
        except OSError:
            pass
    return data


def _mkdir_path(target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(target), "created": True}


def _copy_path(target: Path, params: dict[str, Any]) -> dict[str, Any]:
    destination_value = str(params.get("destination", "")).strip()
    if not destination_value:
        return {"ok": False, "error": "destination is required for copy."}
    if not target.exists():
        return {"ok": False, "error": f"Source path does not exist: {target}"}

    destination = _resolve_path(destination_value, params)
    allow_overwrite = bool(params.get("allow_overwrite", False))

    if destination.exists() and not allow_overwrite:
        return {
            "ok": False,
            "error": "Destination already exists. Set allow_overwrite=true to replace it.",
            "path": str(target),
            "destination": str(destination),
        }

    if target.is_dir():
        shutil.copytree(target, destination, dirs_exist_ok=allow_overwrite)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)

    return {"ok": True, "path": str(target), "destination": str(destination), "copied": True}


def _move_path(target: Path, params: dict[str, Any]) -> dict[str, Any]:
    destination_value = str(params.get("destination", "")).strip()
    if not destination_value:
        return {"ok": False, "error": "destination is required for move."}
    if not target.exists():
        return {"ok": False, "error": f"Source path does not exist: {target}"}

    destination = _resolve_path(destination_value, params)
    allow_overwrite = bool(params.get("allow_overwrite", False))
    confirm_destructive = bool(params.get("confirm_destructive", False))

    if destination.exists():
        if not allow_overwrite:
            return {
                "ok": False,
                "error": "Destination already exists. Set allow_overwrite=true to replace it.",
                "path": str(target),
                "destination": str(destination),
            }
        if not confirm_destructive:
            return {
                "ok": False,
                "error": "Replacing an existing move destination is destructive. Re-run with confirm_destructive=true after user confirmation.",
                "path": str(target),
                "destination": str(destination),
            }
        _delete_existing(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    final_path = Path(shutil.move(str(target), str(destination)))
    return {"ok": True, "path": str(target), "destination": str(final_path), "moved": True}


def _delete_path(target: Path, params: dict[str, Any]) -> dict[str, Any]:
    confirm_destructive = bool(params.get("confirm_destructive", False))
    recursive = bool(params.get("recursive", True))
    if not target.exists():
        return {"ok": False, "error": f"Path does not exist: {target}"}
    if not confirm_destructive:
        return {
            "ok": False,
            "error": "Delete is destructive. Re-run with confirm_destructive=true after user confirmation.",
            "path": str(target),
        }

    if target.is_dir():
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
    else:
        target.unlink()
    return {"ok": True, "path": str(target), "deleted": True}


def _snapshot_tree(root: Path, recursive: bool, include_hidden: bool) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if not include_hidden and _is_hidden(path, root=root):
            continue
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
    if not summarize or len(normalized) <= _READ_CHAR_LIMIT:
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
    return "\n".join(summary_lines), normalized[:_READ_CHAR_LIMIT], True


def _resolve_path(path_value: str, params: dict[str, Any] | None = None) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    runtime_context = params.get("_runtime_context", {}) if isinstance(params, dict) else {}
    if isinstance(runtime_context, dict):
        runtime_cwd = str(runtime_context.get("cwd", "")).strip()
        if runtime_cwd:
            cwd_candidate = (Path(runtime_cwd) / candidate).resolve()
            if cwd_candidate.exists():
                return cwd_candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (BASE_DIR / candidate).resolve()


def _resolve_search_root(target: Path) -> Path:
    if target.exists() and target.is_dir():
        return target
    if target.exists() and target.is_file():
        return target.parent
    if target.parent.exists():
        return target.parent
    return _resolve_path(".")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _looks_like_glob(value: str) -> bool:
    return any(token in value for token in ("*", "?", "["))


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts if part not in (".", ".."))


def _iter_candidate_files(target: Path, search_root: Path, recursive: bool):
    if target.exists() and target.is_file():
        yield target
        return
    iterator = search_root.rglob("*") if recursive else search_root.iterdir()
    for path in iterator:
        if path.is_file():
            yield path


def _path_matches_pattern(path: Path, pattern: str) -> bool:
    if _looks_like_glob(pattern):
        return path.match(pattern)
    return pattern.lower() in path.name.lower()


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTENSIONS or not path.suffix


def _delete_existing(target: Path) -> None:
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
