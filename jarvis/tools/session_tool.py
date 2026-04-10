"""Session naming, listing, and search actions."""

from __future__ import annotations

from typing import Any

from core.memory import Memory

_MEMORY: Memory | None = None


TOOL_DEFINITION = {
    "name": "session_tool",
    "description": "Inspect the current session, rename it, list recent sessions, or search past named sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["current", "rename", "list", "search"],
                "description": "Session action to perform.",
            },
            "name": {
                "type": "string",
                "description": "New display name when action=rename.",
            },
            "query": {
                "type": "string",
                "description": "Search phrase when action=search.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional explicit session id. The current session is used when omitted and available.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum sessions to return.",
                "default": 10,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = Memory()
    memory = _MEMORY

    action = str(params.get("action", "")).strip().lower()
    runtime_context = params.get("_runtime_context", {})
    if not isinstance(runtime_context, dict):
        runtime_context = {}
    session_id = str(params.get("session_id") or runtime_context.get("session_id") or "").strip()
    limit = max(1, min(int(params.get("limit", 10) or 10), 25))

    if action == "current":
        if not session_id:
            return {"ok": False, "error": "No current session is available.", "session": None}
        session = memory.get_session(session_id)
        return {"ok": True, "session": session}

    if action == "rename":
        if not session_id:
            return {"ok": False, "error": "No current session is available for rename.", "session": None}
        name = str(params.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "Missing session name.", "session": None}
        session = memory.rename_session(session_id, name)
        return {"ok": True, "session": session}

    if action == "list":
        return {"ok": True, "sessions": memory.list_sessions(limit=limit)}

    if action == "search":
        query = str(params.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "Missing session search query.", "sessions": []}
        return memory.session_search(query, limit=limit)

    return {"ok": False, "error": f"Unsupported action: {action}"}
