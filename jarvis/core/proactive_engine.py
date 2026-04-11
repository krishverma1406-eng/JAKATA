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
    "briefing": 1800,
    "stale_task": 7200,
    "system_alerts": 120,
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

    def on_session_start(self, session_id: str) -> None:
        """Fire proactive checks when a brand-new session starts."""
        self.session_id = str(session_id or "").strip()
        threading.Thread(
            target=self._fire_session_start_briefing,
            daemon=True,
            name=f"jarvis-session-start-{self.session_id or 'default'}",
        ).start()

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
            return
        if check_name == "system_alerts":
            self._check_system_alerts()

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
        calendar_lines = self._get_today_calendar()
        if not projects and not calendar_lines:
            return

        now = datetime.now()
        hour = now.hour
        if 8 <= hour <= 10:
            briefing_type = "morning"
        elif 13 <= hour <= 15:
            briefing_type = "afternoon"
        elif 18 <= hour <= 20:
            briefing_type = "evening"
        else:
            return

        cache_key = f"briefing_{briefing_type}_{now.strftime('%Y-%m-%d')}"
        if cache_key in self._delivered_ids:
            return

        prompt = (
            f"You are JARVIS giving Krish a {briefing_type} briefing. "
            "Synthesize this into 1-2 short punchy sentences. "
            "Lead with what's most important. Sound like Iron Man's JARVIS.\n\n"
            f"Projects: {', '.join(projects[:3])}\n"
            f"Calendar: {', '.join(calendar_lines[:2]) if calendar_lines else 'nothing scheduled'}\n"
            f"Time: {now.strftime('%I:%M %p')}\n\n"
            "Morning = what to focus on today. "
            "Afternoon = progress check. "
            "Evening = wrap-up plus tomorrow prep. "
            "Be specific. No filler."
        )
        try:
            response = self.brain.chat(
                messages=[{"role": "user", "content": prompt}],
                task_kind="simple",
                system_override="You are JARVIS. One briefing. No padding.",
            )
        except Exception:
            return
        content = str(response.get("content", "")).strip()
        if content and len(content) > 10:
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

    def _check_system_alerts(self) -> None:
        try:
            import psutil
        except ImportError:
            return

        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged and battery.percent < 15:
            bucket = int(battery.percent // 5) * 5
            cache_key = f"battery_low_{bucket}"
            if cache_key in self._delivered_ids:
                return
            self._delivered_ids.add(cache_key)
            self._push(
                (
                    f"Sir, battery is at {battery.percent:.0f}% and not charging. "
                    "Might want to plug in before it cuts out."
                ),
                kind="alert",
            )

    def _get_today_calendar(self) -> list[str]:
        try:
            from services.calendar_service import get_calendar_service

            events = get_calendar_service().today_events(max_results=3)
        except Exception:
            return []

        lines: list[str] = []
        for event in events[:3]:
            if not isinstance(event, dict):
                continue
            summary = str(event.get("summary", "")).strip()
            if not summary:
                continue
            start = str(event.get("start", "")).strip()
            lines.append(f"{summary} at {start}" if start else summary)
        return lines

    def _fire_session_start_briefing(self) -> None:
        projects = self.memory.active_project_items(limit=3)
        try:
            from services.reminders import get_reminder_service

            service = get_reminder_service(self.settings)
            due_reminders = service.pop_due_reminders()
        except Exception:
            due_reminders = []

        for reminder in due_reminders:
            reminder_id = str(reminder.get("id", "")).strip()
            if reminder_id and reminder_id in self._delivered_ids:
                continue
            if reminder_id:
                self._delivered_ids.add(reminder_id)
            text = str(reminder.get("text", "")).strip()
            if text:
                self._push(
                    f"Welcome back, sir. You have a pending reminder: {text}",
                    kind="reminder",
                    data=reminder,
                )

        if not projects:
            return

        payload = self.memory._load_records_payload()
        records = payload.get("records", [])
        if not isinstance(records, list):
            return

        stale_projects: list[str] = []
        now = datetime.now(UTC)
        for record in records:
            if not record.get("active") or record.get("demoted") or record.get("tag") != "PROJECT":
                continue
            last_seen = record.get("last_retrieved_at") or record.get("updated_at", "")
            if not last_seen:
                continue
            try:
                last_dt = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
            except Exception:
                continue
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=UTC)
            if (now - last_dt).total_seconds() > 86400:
                text = str(record.get("text", "")).strip()
                if text:
                    stale_projects.append(text[:80])

        if not stale_projects:
            return

        stale_text = stale_projects[0][:60].rstrip()
        cache_key = f"session_start_stale:{stale_text.lower()}:{now.strftime('%Y-%m-%d')}"
        if cache_key in self._delivered_ids:
            return
        self._delivered_ids.add(cache_key)
        self._push(
            f"Welcome back. Your {stale_text} hasn't come up since yesterday - want to pick it up?",
            kind="stale_task",
        )

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
