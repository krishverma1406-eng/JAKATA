"""Calendar read and create actions."""

from __future__ import annotations

from typing import Any

from services.calendar_service import get_calendar_service


TOOL_DEFINITION = {
    "name": "calendar_tool",
    "description": "Read today's calendar events and create new events through Google Calendar.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["today", "create"],
                "description": "Calendar action to perform.",
            },
            "calendar_id": {
                "type": "string",
                "description": "Optional calendar id. Defaults to primary.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum events to return for today.",
                "default": 10,
            },
            "summary": {
                "type": "string",
                "description": "Event summary for create.",
            },
            "start_time": {
                "type": "string",
                "description": "Event start time in ISO format.",
            },
            "end_time": {
                "type": "string",
                "description": "Event end time in ISO format.",
            },
            "description": {
                "type": "string",
                "description": "Optional event description.",
            },
            "location": {
                "type": "string",
                "description": "Optional event location.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    service = get_calendar_service()
    action = str(params.get("action", "")).strip().lower()
    calendar_id = str(params.get("calendar_id", "")).strip() or "primary"

    if action == "today":
        max_results = max(1, min(int(params.get("max_results", 10) or 10), 25))
        events = service.today_events(calendar_id=calendar_id, max_results=max_results)
        return {"ok": True, "events": events, "summary_lines": _summaries(events)}

    if action == "create":
        summary = str(params.get("summary", "")).strip()
        start_time = str(params.get("start_time", "")).strip()
        end_time = str(params.get("end_time", "")).strip()
        if not summary or not start_time or not end_time:
            return {"ok": False, "error": "summary, start_time, and end_time are required for create."}
        event = service.create_event(
            summary=summary,
            start_time=start_time,
            end_time=end_time,
            description=str(params.get("description", "")).strip(),
            location=str(params.get("location", "")).strip(),
            calendar_id=calendar_id,
        )
        return {"ok": True, "event": event}

    return {"ok": False, "error": f"Unsupported action: {action}"}


def _summaries(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in events:
        summary = str(item.get("summary", "")).strip() or "(untitled event)"
        start = str(item.get("start", "")).strip()
        end = str(item.get("end", "")).strip()
        location = str(item.get("location", "")).strip()
        line = summary
        if start:
            line += f" | starts {start}"
        if end:
            line += f" | ends {end}"
        if location:
            line += f" | {location}"
        lines.append(line)
    return lines
