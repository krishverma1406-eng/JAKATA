"""Background proactive engine — JARVIS initiates without being asked."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any, Callable

from config.settings import SETTINGS, Settings
from core.brain import Brain
from core.memory import Memory


CHECK_INTERVALS = {
    "reminders": 30,
    "briefing": 300,
    "stale_task": 3600,
}


class ProactiveEngine:
    """Runs background checks and emits proactive messages."""

    def __init__(
        self,
        on_message: Callable[[dict[str, Any]], None],
        session_id: str = "",
        settings: Settings | None = None,
    ) -> None:
        self.on_message = on_message
        self.session_id = session_id
        self.settings = settings or SETTINGS
        self.memory = Memory(self.settings)
        self.brain = Brain(self.settings)
        self._stop = threading.Event()
        self._last_run: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._delivered_ids: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"jarvis-proactive-{self.session_id or 'default'}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for check_name, interval in CHECK_INTERVALS.items():
                last = self._last_run.get(check_name, 0.0)
                if now - last >= interval:
                    self._last_run[check_name] = now
                    try:
                        self._run_check(check_name)
                    except Exception:
                        continue
            self._stop.wait(10)

    def _run_check(self, check_name: str) -> None:
        if check_name == "reminders":
            self._check_reminders()
            return
        if check_name == "briefing":
            self._check_briefing()
            return
        if check_name == "stale_task":
            self._check_stale_tasks()

    def _check_reminders(self) -> None:
        from services.reminders import get_reminder_service

        service = get_reminder_service(self.settings)
        due = service.pop_due_reminders()
        for reminder in due:
            reminder_id = str(reminder.get("id", "")).strip()
            if reminder_id and reminder_id in self._delivered_ids:
                continue
            if reminder_id:
                self._delivered_ids.add(reminder_id)
            text = str(reminder.get("text", "")).strip()
            due_at = str(reminder.get("due_at_local", "")).strip()
            message = self._generate_reminder_message(text, due_at)
            self._push(message, kind="reminder", data=reminder)

    def _check_briefing(self) -> None:
        projects = self.memory.active_project_items(limit=5)
        calendar_lines: list[str] = []
        try:
            from services.calendar_service import get_calendar_service

            events = get_calendar_service().today_events(max_results=5)
            for event in events[:5]:
                if not isinstance(event, dict):
                    continue
                summary = str(event.get("summary", "")).strip() or "(untitled event)"
                start = str(event.get("start", "")).strip()
                line = summary if not start else f"{summary} | {start}"
                calendar_lines.append(line)
        except Exception:
            calendar_lines = []
        if not projects and not calendar_lines:
            return

        prompt = (
            "You are JARVIS, a proactive personal AI. "
            "The user (Krish) is NOT present right now — you're doing a background check.\n\n"
            "Active projects in memory:\n"
            + "\n".join(f"- {project}" for project in projects)
            + "\n\nToday's calendar:\n"
            + ("\n".join(f"- {line}" for line in calendar_lines) if calendar_lines else "- No scheduled calendar events found.")
            + "\n\n"
            "Current time: "
            + datetime.now().strftime("%A %I:%M %p")
            + "\n\n"
            "Decide: Is there ONE specific, useful thing to tell Krish proactively right now?\n"
            "Examples of good proactive messages:\n"
            "- 'One project hasn't been touched in a while — want to pick it back up?'\n"
            "- 'You have an event later today that may affect your schedule.'\n"
            "- 'One reminder is due soon and should probably be handled first.'\n\n"
            "If there's nothing genuinely useful to say, respond with exactly: SKIP\n"
            "If there IS something useful, write ONE short proactive JARVIS message (max 2 sentences). "
            "Sound like JARVIS, not a notification. Be specific about the real item that needs attention. "
            "Do not invent project names, deadlines, or events."
        )

        response = self.brain.chat(
            messages=[{"role": "user", "content": prompt}],
            task_kind="simple",
            system_override="You are JARVIS. Be brief, direct, and genuinely useful.",
        )
        content = str(response.get("content", "")).strip()
        if content and content.upper() != "SKIP" and len(content) > 10:
            cache_key = f"briefing:{content.lower()}"
            if cache_key not in self._delivered_ids:
                self._delivered_ids.add(cache_key)
                self._push(content, kind="briefing")

    def _check_stale_tasks(self) -> None:
        records = self.memory._load_records_payload().get("records", [])
        now = datetime.now(UTC)
        stale: list[tuple[float, str]] = []
        for record in records:
            if not record.get("active") or record.get("demoted"):
                continue
            if record.get("tag") != "PROJECT":
                continue
            last_retrieved = record.get("last_retrieved_at") or record.get("updated_at", "")
            if not last_retrieved:
                continue
            try:
                last_dt = datetime.fromisoformat(str(last_retrieved).replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                hours_ago = (now - last_dt).total_seconds() / 3600
                if hours_ago > 48:
                    stale.append((hours_ago, str(record.get("text", "")).strip()))
            except Exception:
                continue

        if not stale:
            return

        stale.sort(reverse=True)
        oldest_hours, oldest_text = stale[0]
        if not oldest_text:
            return
        days = int(oldest_hours // 24)
        message = (
            f"Sir, '{oldest_text[:80]}' hasn't come up in {days} day{'s' if days != 1 else ''}. "
            f"Want to pick it back up or mark it done?"
        )
        cache_key = f"stale:{oldest_text[:30].lower()}:{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key not in self._delivered_ids:
            self._delivered_ids.add(cache_key)
            self._push(message, kind="stale_task")

    def _generate_reminder_message(self, text: str, due_at: str) -> str:
        if due_at:
            return f"Reminder, sir: {text} (due {due_at})"
        return f"Reminder: {text}"

    def _push(self, message: str, kind: str = "proactive", data: dict[str, Any] | None = None) -> None:
        text = str(message).strip()
        if not text:
            return
        self.on_message(
            {
                "type": "proactive",
                "kind": kind,
                "message": text,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "data": data or {},
            }
        )
