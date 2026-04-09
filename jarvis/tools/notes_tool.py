"""Intentional markdown note management."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from config.settings import USER_NOTES_DIR


TOOL_DEFINITION = {
    "name": "notes_tool",
    "description": "Create, read, list, search, and delete markdown notes in data_user/notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "list", "search", "delete"],
                "description": "Note action to perform.",
            },
            "title": {
                "type": "string",
                "description": "Note title or file stem.",
            },
            "content": {
                "type": "string",
                "description": "Markdown note content for create.",
            },
            "query": {
                "type": "string",
                "description": "Search query for search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results for list/search.",
                "default": 10,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    USER_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    action = str(params.get("action", "")).strip().lower()
    max_results = max(1, min(int(params.get("max_results", 10) or 10), 50))

    if action == "create":
        title = str(params.get("title", "")).strip()
        content = str(params.get("content", "")).rstrip()
        if not title:
            return {"ok": False, "error": "title is required for create."}
        if not content:
            return {"ok": False, "error": "content is required for create."}
        path = _note_path(title)
        path.write_text(content + "\n", encoding="utf-8")
        return {"ok": True, "path": str(path), "title": title}

    if action == "read":
        title = str(params.get("title", "")).strip()
        if not title:
            return {"ok": False, "error": "title is required for read."}
        path = _find_best_note(title)
        if path is None:
            return {"ok": False, "error": f"Note not found: {title}"}
        return {"ok": True, "path": str(path), "content": path.read_text(encoding="utf-8", errors="replace")}

    if action == "list":
        notes = [
            {"title": path.stem, "path": str(path)}
            for path in sorted(USER_NOTES_DIR.glob("*.md"))[:max_results]
        ]
        return {"ok": True, "notes": notes}

    if action == "search":
        query = str(params.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required for search."}
        matches = _search_notes(query, max_results=max_results)
        return {"ok": True, "query": query, "matches": matches}

    if action == "delete":
        title = str(params.get("title", "")).strip()
        if not title:
            return {"ok": False, "error": "title is required for delete."}
        path = _find_best_note(title)
        if path is None:
            return {"ok": False, "error": f"Note not found: {title}"}
        path.unlink(missing_ok=True)
        return {"ok": True, "deleted": str(path)}

    return {"ok": False, "error": f"Unsupported action: {action}"}


def _note_path(title: str) -> Path:
    safe = "".join(char if char.isalnum() or char in {"-", "_", " "} else "_" for char in title).strip()
    safe = "-".join(safe.split()).lower() or "note"
    return USER_NOTES_DIR / f"{safe}.md"


def _find_best_note(title: str) -> Path | None:
    exact = _note_path(title)
    if exact.exists():
        return exact
    candidates = list(USER_NOTES_DIR.glob("*.md"))
    best: tuple[float, Path] | None = None
    for path in candidates:
        score = difflib.SequenceMatcher(None, title.lower(), path.stem.lower()).ratio()
        if best is None or score > best[0]:
            best = (score, path)
    if best and best[0] >= 0.45:
        return best[1]
    return None


def _search_notes(query: str, max_results: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    query_lower = query.lower()
    for path in USER_NOTES_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8", errors="replace")
        haystack = f"{path.stem}\n{content}".lower()
        score = 0.0
        if query_lower in haystack:
            score = 1.0
        else:
            score = difflib.SequenceMatcher(None, query_lower, haystack[:5000]).ratio()
        if score <= 0.2:
            continue
        snippet = ""
        for line in content.splitlines():
            if query_lower in line.lower():
                snippet = line.strip()
                break
        if not snippet:
            snippet = " ".join(content.split())[:180]
        results.append({"title": path.stem, "path": str(path), "score": round(score, 3), "snippet": snippet})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max_results]
