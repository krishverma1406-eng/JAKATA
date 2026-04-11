"""Session naming, listing, and search actions."""

from __future__ import annotations

from typing import Any

from core.memory import Memory

_MEMORY: Memory | None = None


TOOL_DEFINITION = {
    "name": "session_tool",
    "description": "Inspect the current session, rename it, list recent sessions, summarize conversation history, or search past named sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["current", "rename", "list", "search", "summarize"],
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


def _get_memory() -> Memory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = Memory()
    return _MEMORY


def _message_excerpt(text: Any, limit: int = 200) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _paired_turns(messages: list[dict[str, Any]], limit_turns: int = 5) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    pending_user = ""
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = _message_excerpt(message.get("content", ""))
        if not content:
            continue
        if role == "user":
            pending_user = content
            continue
        if role == "assistant" and pending_user:
            turns.append({"user": pending_user, "assistant": content})
            pending_user = ""
    return turns[-max(1, limit_turns) :]


def execute(params: dict[str, Any]) -> dict[str, Any]:
    memory = _get_memory()

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

    if action == "summarize":
        session_limit = max(1, min(limit, 10))
        sessions: list[dict[str, Any]]
        if session_id:
            session = memory.get_session(session_id)
            if session is None:
                return {"ok": False, "error": f"Session not found: {session_id}", "sessions": []}
            sessions = [session]
        else:
            sessions = memory.list_sessions(limit=session_limit)

        results: list[dict[str, Any]] = []
        for session in sessions:
            if not isinstance(session, dict):
                continue
            sid = str(session.get("session_id", "")).strip()
            if not sid:
                continue
            messages = memory.load_session_messages(sid, limit_messages=30)
            turns = _paired_turns(messages, limit_turns=5)
            results.append(
                {
                    "session_name": str(session.get("display_name", "Untitled session")).strip() or "Untitled session",
                    "updated_at": str(session.get("updated_at", "")).strip(),
                    "turn_count": len(turns),
                    "turns": turns,
                }
            )
        return {"ok": True, "sessions": results}

    return {"ok": False, "error": f"Unsupported action: {action}"}
