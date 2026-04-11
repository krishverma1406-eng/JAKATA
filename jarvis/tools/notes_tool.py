"""Intentional markdown note management."""

from __future__ import annotations

import difflib
from datetime import datetime
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
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags to categorize the note.",
            },
            "template": {
                "type": "string",
                "enum": ["meeting", "todo", "research", "idea", "bug"],
                "description": "Use a template for structured notes.",
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

_TEMPLATES = {
    "meeting": "# Meeting Notes\n\n**Date:** {date}\n**Attendees:**\n\n## Agenda\n\n## Notes\n\n## Action Items\n",
    "todo": "# Todo List\n\n**Created:** {date}\n\n## Tasks\n\n- [ ] \n\n## Notes\n",
    "research": "# Research: {title}\n\n**Date:** {date}\n\n## Summary\n\n## Key Findings\n\n## Sources\n",
    "idea": "# Idea: {title}\n\n**Date:** {date}\n\n## Concept\n\n## Why This Matters\n\n## Next Steps\n",
    "bug": "# Bug Report: {title}\n\n**Date:** {date}\n\n## Problem\n\n## Steps to Reproduce\n\n## Expected vs Actual\n\n## Fix\n",
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    USER_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    action = str(params.get("action", "")).strip().lower()
    max_results = max(1, min(int(params.get("max_results", 10) or 10), 50))

    if action == "create":
        title = str(params.get("title", "")).strip()
        content = str(params.get("content", "")).rstrip()
        template = str(params.get("template", "")).strip().lower()
        tags = _normalize_tags(params.get("tags"))
        if not title:
            return {"ok": False, "error": "title is required for create."}
        if template and template not in _TEMPLATES:
            return {"ok": False, "error": f"Unsupported template: {template}"}
        if template and not content:
            content = _TEMPLATES[template].format(
                date=datetime.now().strftime("%Y-%m-%d"),
                title=title,
            ).rstrip()
        if not content:
            return {"ok": False, "error": "content is required for create."}
        path = _note_path(title)
        serialized = _serialize_note_content(title, content, tags=tags, template=template or None)
        path.write_text(serialized + "\n", encoding="utf-8")
        return {
            "ok": True,
            "path": str(path),
            "title": title,
            "tags": tags,
            "template": template or None,
        }

    if action == "read":
        title = str(params.get("title", "")).strip()
        if not title:
            return {"ok": False, "error": "title is required for read."}
        path = _find_best_note(title)
        if path is None:
            return {"ok": False, "error": f"Note not found: {title}"}
        raw_content = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = _extract_note_metadata(raw_content)
        return {
            "ok": True,
            "path": str(path),
            "content": body,
            "raw_content": raw_content,
            "tags": metadata.get("tags", []),
            "template": metadata.get("template"),
        }

    if action == "list":
        notes = []
        for path in sorted(USER_NOTES_DIR.glob("*.md"))[:max_results]:
            metadata, _body = _extract_note_metadata(path.read_text(encoding="utf-8", errors="replace"))
            notes.append(
                {
                    "title": path.stem,
                    "path": str(path),
                    "tags": metadata.get("tags", []),
                    "template": metadata.get("template"),
                }
            )
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
        raw_content = path.read_text(encoding="utf-8", errors="replace")
        metadata, content = _extract_note_metadata(raw_content)
        tags = metadata.get("tags", [])
        template = str(metadata.get("template") or "")
        haystack = f"{path.stem}\n{template}\n{' '.join(tags)}\n{content}".lower()
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
        results.append(
            {
                "title": path.stem,
                "path": str(path),
                "score": round(score, 3),
                "snippet": snippet,
                "tags": tags,
                "template": template or None,
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max_results]


def _normalize_tags(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        tag = str(item).strip()
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(tag)
    return normalized


def _serialize_note_content(
    title: str,
    content: str,
    tags: list[str] | None = None,
    template: str | None = None,
) -> str:
    body = str(content or "").rstrip()
    metadata_lines = [
        "---",
        f"title: {title}",
        f"created_at: {datetime.now().strftime('%Y-%m-%d')}",
    ]
    if template:
        metadata_lines.append(f"template: {template}")
    if tags:
        metadata_lines.append("tags:")
        metadata_lines.extend(f"  - {tag}" for tag in tags)
    metadata_lines.append("---")
    return "\n".join(metadata_lines) + "\n\n" + body if (tags or template) else body


def _extract_note_metadata(content: str) -> tuple[dict[str, Any], str]:
    text = str(content or "")
    if not text.startswith("---\n"):
        return {"tags": [], "template": None}, text

    lines = text.splitlines()
    if len(lines) < 3:
        return {"tags": [], "template": None}, text

    metadata: dict[str, Any] = {"tags": [], "template": None}
    index = 1
    current_key = ""
    while index < len(lines):
        line = lines[index]
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :]).lstrip()
            return metadata, body
        stripped = line.strip()
        if stripped.startswith("- ") and current_key == "tags":
            metadata.setdefault("tags", []).append(stripped[2:].strip())
        elif ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip().lower()
            cleaned_value = value.strip()
            if current_key == "template":
                metadata["template"] = cleaned_value or None
            elif current_key == "tags":
                metadata["tags"] = []
            elif current_key == "title":
                metadata["title"] = cleaned_value
            elif current_key == "created_at":
                metadata["created_at"] = cleaned_value
            else:
                current_key = ""
        index += 1

    return {"tags": [], "template": None}, text
