"""Tool for querying stored JARVIS memory."""

from __future__ import annotations

from typing import Any

from core.memory import Memory

_MEMORY: Memory | None = None


TOOL_DEFINITION = {
    "name": "memory_query",
    "description": "Search stored user memory and prior project context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question or search phrase to match against memory.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    query = str(params.get("query", "")).strip()
    limit = int(params.get("limit", 5) or 5)
    if not query:
        return {"ok": False, "error": "Missing memory query.", "memories": []}

    global _MEMORY
    if _MEMORY is None:
        _MEMORY = Memory()
    memory = _MEMORY
    memories = memory.recall(query, limit)
    return {"ok": True, "query": query, "memories": memories}
