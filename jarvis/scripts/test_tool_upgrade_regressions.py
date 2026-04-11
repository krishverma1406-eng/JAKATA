"""Regression coverage for upgraded built-in tools."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from core.agent import Agent
from core.proactive_engine import CHECK_INTERVALS, ProactiveEngine
from services import reminders as reminder_service_module
from services.reminders import ReminderService
from tools import notes_tool, reminder_tool, system_info_tool, web_search


class WebSearchUpgradeRegressionTests(unittest.TestCase):
    def test_post_process_results_deduplicates_domains_and_scores_relevance(self) -> None:
        results = [
            {
                "title": "Python unit testing guide",
                "url": "https://docs.python.org/3/library/unittest.html",
                "snippet": "Use unittest to build python unit test suites.",
            },
            {
                "title": "Another docs page",
                "url": "https://docs.python.org/3/tutorial/index.html",
                "snippet": "General Python tutorial",
            },
            {
                "title": "Weather update",
                "url": "https://weather.example.com/today",
                "snippet": "Rain in Delhi tonight.",
            },
        ]

        processed = web_search._post_process_results(results, "python unit test")

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed[0]["url"], "https://docs.python.org/3/library/unittest.html")
        self.assertTrue(processed[0]["answers_query"])
        self.assertGreater(processed[0]["relevance_score"], processed[1]["relevance_score"])


class NotesToolUpgradeRegressionTests(unittest.TestCase):
    def test_create_template_note_with_tags_and_search_by_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            notes_tool,
            "USER_NOTES_DIR",
            Path(temp_dir),
        ):
            created = notes_tool.execute(
                {
                    "action": "create",
                    "title": "Login Failure",
                    "template": "bug",
                    "tags": ["backend", "urgent"],
                }
            )
            self.assertTrue(created["ok"])
            self.assertEqual(created["template"], "bug")
            self.assertEqual(created["tags"], ["backend", "urgent"])

            listed = notes_tool.execute({"action": "list"})
            self.assertEqual(listed["notes"][0]["tags"], ["backend", "urgent"])
            self.assertEqual(listed["notes"][0]["template"], "bug")

            read_back = notes_tool.execute({"action": "read", "title": "Login Failure"})
            self.assertIn("# Bug Report: Login Failure", read_back["content"])
            self.assertEqual(read_back["tags"], ["backend", "urgent"])

            searched = notes_tool.execute({"action": "search", "query": "backend"})
            self.assertEqual(len(searched["matches"]), 1)
            self.assertEqual(searched["matches"][0]["template"], "bug")


class ReminderUpgradeRegressionTests(unittest.TestCase):
    def test_reminder_tool_passes_recurrence_to_service(self) -> None:
        service = Mock()
        service.add_reminder.return_value = {"id": "reminder_1", "recur": "daily"}

        with patch("tools.reminder_tool.get_reminder_service", return_value=service):
            result = reminder_tool.execute(
                {
                    "action": "create",
                    "text": "Standup",
                    "due_at": "2026-04-11T09:00:00+00:00",
                    "recur": "daily",
                }
            )

        self.assertTrue(result["ok"])
        service.add_reminder.assert_called_once_with(
            text="Standup",
            due_at="2026-04-11T09:00:00+00:00",
            recur="daily",
        )

    def test_daily_recurrence_reschedules_instead_of_delivering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            reminder_service_module,
            "USER_REMINDERS_FILE",
            Path(temp_dir) / "reminders.json",
        ):
            service = ReminderService(settings=types.SimpleNamespace(reminder_timezone="UTC", reminder_check_seconds=5))
            reminder = service.add_reminder(
                text="Daily sync",
                due_at="2026-04-11T09:00:00+00:00",
                recur="daily",
            )

            fixed_now = datetime(2026, 4, 11, 10, 0, 0, tzinfo=UTC)
            class FixedDateTime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

            with patch("services.reminders.datetime", FixedDateTime):
                service.mark_delivered([reminder["id"]])

            stored = service.list_reminders(include_completed=True)[0]
            self.assertEqual(stored["status"], "pending")
            self.assertEqual(stored["recur"], "daily")
            self.assertEqual(stored["due_at"], "2026-04-12T09:00:00+00:00")

    def test_weekdays_recurrence_skips_weekend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            reminder_service_module,
            "USER_REMINDERS_FILE",
            Path(temp_dir) / "reminders.json",
        ):
            service = ReminderService(settings=types.SimpleNamespace(reminder_timezone="UTC", reminder_check_seconds=5))
            reminder = service.add_reminder(
                text="Weekday check-in",
                due_at="2026-04-10T18:00:00+00:00",
                recur="weekdays",
            )

            fixed_now = datetime(2026, 4, 10, 19, 0, 0, tzinfo=UTC)
            class FixedDateTime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

            with patch("services.reminders.datetime", FixedDateTime):
                service.mark_delivered([reminder["id"]])

            stored = service.list_reminders(include_completed=True)[0]
            self.assertEqual(stored["due_at"], "2026-04-13T18:00:00+00:00")


class SystemInfoUpgradeRegressionTests(unittest.TestCase):
    def test_alerts_action_returns_resource_warnings(self) -> None:
        fake_psutil = types.SimpleNamespace(
            cpu_percent=lambda interval=0.5: 92,
            virtual_memory=lambda: types.SimpleNamespace(
                percent=94,
                available=2 * (1024 ** 3),
                used=14 * (1024 ** 3),
                total=16 * (1024 ** 3),
            ),
            sensors_battery=lambda: types.SimpleNamespace(percent=12, power_plugged=False),
            disk_usage=lambda _path: types.SimpleNamespace(percent=93, free=15 * (1024 ** 3)),
        )

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            result = system_info_tool.execute({"action": "alerts"})

        self.assertTrue(result["ok"])
        self.assertFalse(result["all_clear"])
        self.assertEqual({alert["type"] for alert in result["alerts"]}, {"cpu", "ram", "battery", "disk"})

    def test_agent_summarizes_system_alert_results(self) -> None:
        agent = Agent.__new__(Agent)
        summary = Agent._resolve_final_answer(
            agent,
            {"content": ""},
            [
                {
                    "name": "system_info_tool",
                    "result": {
                        "ok": True,
                        "alerts": [{"message": "Battery at 12% - please plug in"}],
                        "all_clear": False,
                    },
                }
            ],
        )
        self.assertIn("Battery at 12%", summary)


class ProactiveAlertUpgradeRegressionTests(unittest.TestCase):
    def test_system_alert_check_is_registered(self) -> None:
        self.assertEqual(CHECK_INTERVALS["system_alerts"], 120)

    def test_low_battery_alert_fires_once_per_bucket(self) -> None:
        pushed: list[dict[str, object]] = []
        engine = ProactiveEngine(on_message=pushed.append, session_id="battery-1")

        fake_psutil = types.SimpleNamespace(
            sensors_battery=lambda: types.SimpleNamespace(percent=14, power_plugged=False)
        )
        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            engine._check_system_alerts()
            engine._check_system_alerts()

        self.assertEqual(len(pushed), 1)
        self.assertEqual(pushed[0]["kind"], "alert")
        self.assertIn("battery is at 14%", str(pushed[0]["message"]).lower())


if __name__ == "__main__":
    unittest.main()
