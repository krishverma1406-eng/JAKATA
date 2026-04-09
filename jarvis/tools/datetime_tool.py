"""Current time and date lookup tool."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


TOOL_DEFINITION = {
    "name": "datetime_tool",
    "description": "Get the current date, time, weekday, and timezone.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional IANA timezone like Asia/Calcutta or UTC.",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
}


def execute(params: dict) -> dict:
    timezone_name = params.get("timezone")
    if timezone_name:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except Exception as exc:
            return {"ok": False, "error": f"Invalid timezone '{timezone_name}': {exc}"}
    else:
        now = datetime.now().astimezone()
        timezone_name = str(now.tzinfo)

    return {
        "ok": True,
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": timezone_name,
    }
