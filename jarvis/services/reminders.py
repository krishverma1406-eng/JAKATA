"""Reminder storage and in-process scheduling for JARVIS."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from config.settings import SETTINGS, USER_REMINDERS_FILE, Settings

_SERVICES: dict[str, "ReminderService"] = {}
_RECUR_VALUES = {"none", "daily", "weekly", "weekdays"}


class ReminderService:
    """Persist reminders on disk and dispatch due reminders while JARVIS runs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self.path = Path(USER_REMINDERS_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]\n", encoding="utf-8")
        self._lock = threading.Lock()
        self._scheduler: Any | None = None
        self._callback: Callable[[dict[str, Any]], None] | None = None

    def start(self, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        if self._scheduler is not None:
            self._callback = callback or self._callback
            return

        from apscheduler.schedulers.background import BackgroundScheduler

        self._callback = callback
        self._scheduler = BackgroundScheduler(timezone=self.settings.reminder_timezone)
        self._scheduler.add_job(
            self._dispatch_due_reminders,
            trigger="interval",
            seconds=max(5, self.settings.reminder_check_seconds),
            id="jarvis_due_reminder_check",
            replace_existing=True,
        )
        self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler is None:
            return
        self._scheduler.shutdown(wait=False)
        self._scheduler = None

    def add_reminder(
        self,
        text: str,
        due_at: str | datetime,
        recur: str = "none",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reminder_text = str(text).strip()
        if not reminder_text:
            raise ValueError("Reminder text is required.")

        due = self._parse_due_at(due_at)
        reminder = {
            "id": f"reminder_{uuid.uuid4().hex[:10]}",
            "text": reminder_text,
            "due_at": due.astimezone(UTC).isoformat(),
            "created_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "recur": self._normalize_recur(recur),
            "metadata": metadata or {},
        }

        with self._lock:
            reminders = self._read_reminders_unlocked()
            reminders.append(reminder)
            self._write_reminders_unlocked(reminders)
        return self._decorate_reminder(reminder)

    def list_reminders(self, include_completed: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            reminders = self._read_reminders_unlocked()
        if include_completed:
            return [self._decorate_reminder(reminder) for reminder in reminders]
        return [
            self._decorate_reminder(reminder)
            for reminder in reminders
            if reminder.get("status") == "pending"
        ]

    def delete_reminder(self, reminder_id: str) -> dict[str, Any]:
        reminder_id = str(reminder_id).strip()
        if not reminder_id:
            raise ValueError("Reminder id is required.")

        with self._lock:
            reminders = self._read_reminders_unlocked()
            remaining = [reminder for reminder in reminders if reminder.get("id") != reminder_id]
            if len(remaining) == len(reminders):
                raise ValueError(f"Reminder not found: {reminder_id}")
            self._write_reminders_unlocked(remaining)
        return {"id": reminder_id, "deleted": True}

    def due_reminders(self) -> list[dict[str, Any]]:
        with self._lock:
            reminders = self._read_reminders_unlocked()
        now = datetime.now(UTC)
        due: list[dict[str, Any]] = []
        for reminder in reminders:
            if reminder.get("status") != "pending":
                continue
            due_at = self._parse_due_at(reminder.get("due_at", ""))
            if due_at <= now:
                due.append(self._decorate_reminder(reminder))
        return due

    def mark_delivered(self, reminder_ids: list[str]) -> None:
        ids = {item for item in reminder_ids if item}
        if not ids:
            return
        with self._lock:
            reminders = self._read_reminders_unlocked()
            changed = False
            now = datetime.now(UTC)
            for reminder in reminders:
                if reminder.get("id") in ids and reminder.get("status") == "pending":
                    recur = self._normalize_recur(reminder.get("recur", "none"))
                    reminder["last_delivered_at"] = now.isoformat()
                    if recur == "none":
                        reminder["status"] = "delivered"
                        reminder["delivered_at"] = now.isoformat()
                    else:
                        try:
                            current_due = self._parse_due_at(reminder.get("due_at", ""))
                            next_due = self._next_due_at(current_due, recur, now)
                        except Exception:
                            reminder["status"] = "delivered"
                            reminder["delivered_at"] = now.isoformat()
                        else:
                            reminder["status"] = "pending"
                            reminder["due_at"] = next_due.astimezone(UTC).isoformat()
                            reminder.pop("delivered_at", None)
                    changed = True
            if changed:
                self._write_reminders_unlocked(reminders)

    def pop_due_reminders(self) -> list[dict[str, Any]]:
        due = self.due_reminders()
        self.mark_delivered([reminder.get("id", "") for reminder in due])
        return due

    def _dispatch_due_reminders(self) -> None:
        if self._callback is None:
            return
        for reminder in self.pop_due_reminders():
            try:
                self._callback(reminder)
            except Exception:
                continue

    def _read_reminders_unlocked(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = []
        return payload if isinstance(payload, list) else []

    def _write_reminders_unlocked(self, reminders: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(reminders, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def _parse_due_at(self, due_at: str | datetime) -> datetime:
        if isinstance(due_at, datetime):
            parsed = due_at
        else:
            text = str(due_at).strip()
            if not text:
                raise ValueError("due_at is required.")
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _decorate_reminder(self, reminder: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(reminder)
        for key in ("due_at", "created_at", "delivered_at"):
            value = reminder.get(key)
            if not value:
                continue
            try:
                enriched[f"{key}_local"] = self.format_timestamp(value)
            except Exception:
                continue
        return enriched

    def format_timestamp(self, value: str | datetime) -> str:
        parsed = self._parse_due_at(value)
        local_zone = ZoneInfo(self.settings.reminder_timezone)
        local_time = parsed.astimezone(local_zone)
        return local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    @staticmethod
    def _normalize_recur(value: Any) -> str:
        cleaned = str(value or "none").strip().lower() or "none"
        return cleaned if cleaned in _RECUR_VALUES else "none"

    def _next_due_at(self, current_due: datetime, recur: str, now: datetime) -> datetime:
        candidate = current_due.astimezone(UTC)
        normalized = self._normalize_recur(recur)
        if normalized == "none":
            return candidate

        while True:
            if normalized == "daily":
                candidate += timedelta(days=1)
            elif normalized == "weekly":
                candidate += timedelta(weeks=1)
            elif normalized == "weekdays":
                candidate += timedelta(days=1)
                while candidate.weekday() >= 5:
                    candidate += timedelta(days=1)
            if candidate > now:
                return candidate


def get_reminder_service(settings: Settings | None = None) -> ReminderService:
    config = settings or SETTINGS
    key = str(Path(USER_REMINDERS_FILE).resolve())
    service = _SERVICES.get(key)
    if service is None:
        service = ReminderService(config)
        _SERVICES[key] = service
    return service
