"""Tool for querying stored JARVIS memory."""

from __future__ import annotations

from typing import Any

from core.brain import Brain
from core.memory import Memory

_MEMORY: Memory | None = None
_BRAIN: Brain | None = None


TOOL_DEFINITION = {
    "name": "memory_query",
    "description": "Search stored memory, explicitly remember something, forget stored memories, or look up tracked entities.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "forget", "entity"],
                "description": "Memory action to perform.",
                "default": "search",
            },
            "query": {
                "type": "string",
                "description": "Question or search phrase to match against memory, or the thing to forget / entity to look up.",
            },
            "text": {
                "type": "string",
                "description": "Explicit memory text to store when action=remember.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 5,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "search")).strip().lower() or "search"
    query = str(params.get("query", "")).strip()
    text = str(params.get("text", "")).strip()
    limit = int(params.get("limit", 5) or 5)

    global _MEMORY, _BRAIN
    if _MEMORY is None:
        _MEMORY = Memory()
    if _BRAIN is None:
        _BRAIN = Brain()
    memory = _MEMORY
    brain = _BRAIN

    if action == "remember":
        payload = text or query
        if not payload:
            return {"ok": False, "error": "Missing memory text.", "items": []}
        result = memory.remember(payload, brain=brain)
        return result

    if action == "forget":
        if not query:
            return {"ok": False, "error": "Missing forget query.", "items": []}
        return memory.forget(query, limit=limit)

    if action == "entity":
        if not query:
            return {"ok": False, "error": "Missing entity lookup query.", "entities": []}
        return memory.entity_lookup(query, limit=limit)

    if not query:
        return {"ok": False, "error": "Missing memory query.", "memories": []}
    memories = memory.recall(query, limit)
    return {"ok": True, "query": query, "memories": memories}
