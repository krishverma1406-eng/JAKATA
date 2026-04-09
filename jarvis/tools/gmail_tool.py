"""Gmail inbox and send actions."""

from __future__ import annotations

from typing import Any

from services.gmail_service import get_gmail_service


TOOL_DEFINITION = {
    "name": "gmail_tool",
    "description": "Read unread Gmail messages, search inbox, and send email through the Gmail API.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["unread", "search", "send"],
                "description": "Gmail action to perform.",
            },
            "query": {
                "type": "string",
                "description": "Gmail search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum messages to return.",
                "default": 10,
            },
            "to": {
                "type": "string",
                "description": "Recipient email address for send.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject for send.",
            },
            "body": {
                "type": "string",
                "description": "Email body for send.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def execute(params: dict[str, Any]) -> dict[str, Any]:
    service = get_gmail_service()
    action = str(params.get("action", "")).strip().lower()
    max_results = max(1, min(int(params.get("max_results", 10) or 10), 25))

    if action == "unread":
        messages = service.unread(max_results=max_results)
        return {"ok": True, "messages": messages, "summary_lines": _summaries(messages)}
    if action == "search":
        query = str(params.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required for search."}
        messages = service.search(query, max_results=max_results)
        return {"ok": True, "messages": messages, "summary_lines": _summaries(messages)}
    if action == "send":
        to = str(params.get("to", "")).strip()
        subject = str(params.get("subject", "")).strip()
        body = str(params.get("body", "")).strip()
        if not to or not subject or not body:
            return {"ok": False, "error": "to, subject, and body are required for send."}
        sent = service.send(to=to, subject=subject, body=body)
        return {"ok": True, "sent": sent}
    return {"ok": False, "error": f"Unsupported action: {action}"}


def _summaries(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in messages:
        sender = str(item.get("from", "")).strip() or "Unknown sender"
        subject = str(item.get("subject", "")).strip() or "(no subject)"
        date = str(item.get("date", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        summary = f"{sender} | {subject}"
        if date:
            summary += f" | {date}"
        if snippet:
            summary += f" | {snippet[:140]}"
        lines.append(summary)
    return lines
