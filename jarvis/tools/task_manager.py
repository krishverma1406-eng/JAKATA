"""Structured task management with status tracking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import DATA_USER_DIR


TASKS_FILE = DATA_USER_DIR / "tasks.json"

TOOL_DEFINITION = {
    "name": "task_manager",
    "description": (
        "Create, update, list, complete, delete, and search structured tasks with status and priority. "
        "Use for project tasks, to-do items, and work tracking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "update", "complete", "delete", "search"],
            },
            "title": {"type": "string", "description": "Task title."},
            "description": {"type": "string", "description": "Task details."},
            "project": {"type": "string", "description": "Project this task belongs to."},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "default": "medium",
            },
            "due_date": {"type": "string", "description": "Due date ISO format."},
            "task_id": {"type": "string", "description": "Task ID for update, complete, or delete."},
            "status": {
                "type": "string",
                "enum": ["todo", "in_progress", "done", "blocked"],
            },
            "query": {"type": "string", "description": "Search query."},
            "filter_project": {"type": "string", "description": "Filter by project name."},
            "filter_status": {"type": "string", "description": "Filter by status."},
            "max_results": {"type": "integer", "default": 20},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

_PRIORITY_VALUES = {"low", "medium", "high", "urgent"}
_STATUS_VALUES = {"todo", "in_progress", "done", "blocked"}


def _load() -> dict[str, Any]:
    if not TASKS_FILE.exists():
        TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {"tasks": [], "updated_at": ""}
    try:
        payload = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": [], "updated_at": ""}
    if not isinstance(payload, dict):
        return {"tasks": [], "updated_at": ""}
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []
    payload["tasks"] = tasks
    payload["updated_at"] = str(payload.get("updated_at", ""))
    return payload


def _save(payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now().isoformat()
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TASKS_FILE.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).strip().lower()
    payload = _load()
    tasks = payload.get("tasks", [])

    if action == "create":
        title = str(params.get("title", "")).strip()
        if not title:
            return {"ok": False, "error": "title is required."}
        task = {
            "id": f"task_{uuid.uuid4().hex[:8]}",
            "title": title,
            "description": str(params.get("description", "")).strip(),
            "project": str(params.get("project", "")).strip() or "General",
            "priority": _normalize_priority(params.get("priority")),
            "status": "todo",
            "due_date": str(params.get("due_date", "")).strip() or None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        tasks.append(task)
        payload["tasks"] = tasks
        _save(payload)
        return {"ok": True, "task": task}

    if action == "list":
        max_results = max(1, min(int(params.get("max_results", 20) or 20), 100))
        filtered = [task for task in tasks if task.get("status") != "done"]
        project_filter = str(params.get("filter_project", "")).strip().lower()
        status_filter = str(params.get("filter_status", "")).strip().lower()
        if project_filter:
            filtered = [task for task in filtered if project_filter in str(task.get("project", "")).lower()]
        if status_filter:
            filtered = [task for task in filtered if str(task.get("status", "")).lower() == status_filter]

        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        filtered.sort(
            key=lambda task: (
                priority_order.get(str(task.get("priority", "medium")).lower(), 2),
                str(task.get("due_date") or ""),
                str(task.get("created_at") or ""),
            )
        )
        return {
            "ok": True,
            "tasks": filtered[:max_results],
            "total": len(filtered),
            "by_project": _group_by_project(filtered),
        }

    if action in {"update", "complete"}:
        task_id = str(params.get("task_id", "")).strip()
        if not task_id:
            return {"ok": False, "error": "task_id is required."}
        task = next((item for item in tasks if str(item.get("id")) == task_id), None)
        if task is None:
            return {"ok": False, "error": f"Task not found: {task_id}"}

        if action == "complete":
            task["status"] = "done"
            task["completed_at"] = datetime.now().isoformat()
        else:
            if "title" in params and str(params.get("title", "")).strip():
                task["title"] = str(params.get("title", "")).strip()
            if "description" in params:
                task["description"] = str(params.get("description", "")).strip()
            if "project" in params and str(params.get("project", "")).strip():
                task["project"] = str(params.get("project", "")).strip()
            if "priority" in params:
                task["priority"] = _normalize_priority(params.get("priority"))
            if "status" in params:
                task["status"] = _normalize_status(params.get("status"))
                if task["status"] == "done" and not task.get("completed_at"):
                    task["completed_at"] = datetime.now().isoformat()
            if "due_date" in params:
                due_date = str(params.get("due_date", "")).strip()
                task["due_date"] = due_date or None

        task["updated_at"] = datetime.now().isoformat()
        payload["tasks"] = tasks
        _save(payload)
        return {"ok": True, "task": task}

    if action == "delete":
        task_id = str(params.get("task_id", "")).strip()
        if not task_id:
            return {"ok": False, "error": "task_id is required."}
        remaining = [task for task in tasks if str(task.get("id")) != task_id]
        if len(remaining) == len(tasks):
            return {"ok": False, "error": f"Task not found: {task_id}"}
        payload["tasks"] = remaining
        _save(payload)
        return {"ok": True, "deleted": task_id}

    if action == "search":
        query = str(params.get("query", "")).strip().lower()
        if not query:
            return {"ok": False, "error": "query is required."}
        matches = [
            task
            for task in tasks
            if query in str(task.get("title", "")).lower()
            or query in str(task.get("description", "")).lower()
            or query in str(task.get("project", "")).lower()
        ]
        return {"ok": True, "matches": matches}

    return {"ok": False, "error": f"Unknown action: {action}"}


def _normalize_priority(value: Any) -> str:
    cleaned = str(value or "medium").strip().lower() or "medium"
    return cleaned if cleaned in _PRIORITY_VALUES else "medium"


def _normalize_status(value: Any) -> str:
    cleaned = str(value or "todo").strip().lower() or "todo"
    return cleaned if cleaned in _STATUS_VALUES else "todo"


def _group_by_project(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for task in tasks:
        project = str(task.get("project", "General")).strip() or "General"
        title = str(task.get("title", "")).strip()
        if not title:
            continue
        groups.setdefault(project, []).append(title)
    return groups
