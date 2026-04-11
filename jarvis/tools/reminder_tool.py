"""Schedule and manage future reminders."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from services.reminders import get_reminder_service


TOOL_DEFINITION = {
    "name": "reminder_tool",
    "description": "Create, list, delete, and check scheduled reminders stored in data_user/reminders.json.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "delete", "check_due"],
                "description": "Reminder action to perform.",
            },
            "text": {
                "type": "string",
                "description": "Reminder text for create.",
            },
            "due_at": {
                "type": "string",
                "description": "Reminder due time in ISO 8601 format.",
            },
            "delay_minutes": {
                "type": "integer",
                "description": "Optional relative delay in minutes for create.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Optional relative delay in seconds for create.",
            },
            "recur": {
                "type": "string",
                "enum": ["none", "daily", "weekly", "weekdays"],
                "description": "Recurrence pattern.",
                "default": "none",
            },
            "reminder_id": {
                "type": "string",
                "description": "Reminder id for delete.",
            },
            "include_completed": {
                "type": "boolean",
                "description": "Include delivered reminders in list output.",
                "default": False,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    service = get_reminder_service()

    if action == "create":
        due_at = _resolve_due_at(params)
        reminder = service.add_reminder(
            text=str(params.get("text", "")).strip(),
            due_at=due_at,
            recur=str(params.get("recur", "none")).strip(),
        )
        return {"ok": True, "reminder": reminder}

    if action == "list":
        reminders = service.list_reminders(include_completed=bool(params.get("include_completed", False)))
        return {"ok": True, "reminders": reminders}

    if action == "delete":
        reminder_id = str(params.get("reminder_id", "")).strip()
        if not reminder_id:
            return {"ok": False, "error": "reminder_id is required for delete."}
        result = service.delete_reminder(reminder_id)
        return {"ok": True, "result": result}

    if action == "check_due":
        due = service.pop_due_reminders()
        return {"ok": True, "due_reminders": due}

    return {"ok": False, "error": f"Unsupported action: {action}"}


def _resolve_due_at(params: dict[str, Any]) -> str:
    if str(params.get("due_at", "")).strip():
        return str(params.get("due_at", "")).strip()

    delay_seconds = int(params.get("delay_seconds", 0) or 0)
    delay_minutes = int(params.get("delay_minutes", 0) or 0)
    total_seconds = delay_seconds + (delay_minutes * 60)
    if total_seconds <= 0:
        raise ValueError("Provide due_at or a positive delay_minutes/delay_seconds.")
    return (datetime.now(UTC) + timedelta(seconds=total_seconds)).isoformat()
