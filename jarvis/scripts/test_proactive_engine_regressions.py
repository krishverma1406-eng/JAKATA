"""Regression coverage for proactive engine behavior and delivery."""

from __future__ import annotations

import sys
import types
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.proactive_engine import ProactiveEngine
import server


class ProactiveEngineRegressionTests(unittest.TestCase):
    def test_morning_briefing_fires_once_per_day(self) -> None:
        pushed: list[dict[str, object]] = []
        engine = ProactiveEngine(on_message=pushed.append, session_id="s-1")
        engine.memory = Mock()
        engine.memory.active_project_items.return_value = ["Finish dashboard"]
        engine.brain = Mock()
        engine.brain.chat.return_value = {"content": "Good morning, sir. Finish dashboard before the review."}

        fixed_now = datetime(2026, 4, 11, 9, 0, 0)
        with patch("core.proactive_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
            engine._get_today_calendar = Mock(return_value=["Review at 11:00"])
            engine._check_briefing()
            engine._check_briefing()

        self.assertEqual(len(pushed), 1)
        self.assertEqual(pushed[0]["kind"], "briefing")
        engine.brain.chat.assert_called_once()

    def test_off_hours_briefing_skips_model_call(self) -> None:
        pushed: list[dict[str, object]] = []
        engine = ProactiveEngine(on_message=pushed.append, session_id="s-2")
        engine.memory = Mock()
        engine.memory.active_project_items.return_value = ["Finish dashboard"]
        engine.brain = Mock()

        fixed_now = datetime(2026, 4, 11, 23, 15, 0)
        with patch("core.proactive_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
            engine._get_today_calendar = Mock(return_value=["Review at 11:00"])
            engine._check_briefing()

        self.assertEqual(pushed, [])
        engine.brain.chat.assert_not_called()

    def test_session_start_briefing_emits_due_reminder_and_stale_task(self) -> None:
        pushed: list[dict[str, object]] = []
        engine = ProactiveEngine(on_message=pushed.append, session_id="s-3")
        engine.memory = Mock()
        engine.memory.active_project_items.return_value = ["Analytics dashboard"]
        stale_updated = (datetime(2026, 4, 9, 8, 0, 0, tzinfo=UTC)).isoformat()
        engine.memory._load_records_payload.return_value = {
            "records": [
                {
                    "active": True,
                    "demoted": False,
                    "tag": "PROJECT",
                    "text": "Analytics dashboard migration",
                    "updated_at": stale_updated,
                }
            ]
        }

        reminder_service = Mock()
        reminder_service.pop_due_reminders.return_value = [{"id": "r-1", "text": "Ship the release notes"}]

        fixed_now = datetime(2026, 4, 11, 10, 30, 0, tzinfo=UTC)
        with patch("core.proactive_engine.datetime") as mock_datetime, patch(
            "services.reminders.get_reminder_service",
            return_value=reminder_service,
        ):
            mock_datetime.now.return_value = fixed_now
            mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
            engine._fire_session_start_briefing()

        self.assertEqual([item["kind"] for item in pushed], ["reminder", "stale_task"])
        self.assertIn("pending reminder", str(pushed[0]["message"]))
        self.assertIn("hasn't come up since yesterday", str(pushed[1]["message"]))


class ServerProactiveDeliveryRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        with server._SESSIONS_LOCK:
            server._SESSIONS.clear()
            server._PROACTIVE_ENGINES.clear()
        with server._SSE_LOCK:
            server._SSE_QUEUES.clear()
            server._PROACTIVE_BACKLOG.clear()

    def tearDown(self) -> None:
        self.setUp()

    def test_new_session_triggers_engine_session_start(self) -> None:
        fake_engine = Mock()

        class FakeAgent:
            def __init__(self, settings=None) -> None:
                self.settings = settings
                self.session_id = ""
                self.mode = "normal"
                self.session_meta = {"display_name": "New session", "updated_at": "", "turn_count": 0}

            def bind_session(self, sid: str, mode: str | None = None) -> None:
                self.session_id = sid
                if mode:
                    self.mode = mode

            def set_mode(self, mode: str) -> None:
                self.mode = mode

        with patch("server.Agent", FakeAgent), patch(
            "server._get_or_create_proactive_engine",
            return_value=fake_engine,
        ):
            sid, _agent = server._get_agent(None, None)
            _sid_again, _agent_again = server._get_agent(sid, None)

        fake_engine.on_session_start.assert_called_once_with(sid)

    def test_proactive_backlog_preserves_messages_until_stream_connects(self) -> None:
        payload = {"type": "proactive", "kind": "briefing", "message": "Welcome back, sir."}

        server._dispatch_proactive_message("session-9", payload)
        drained = server._drain_proactive_backlog("session-9")
        drained_again = server._drain_proactive_backlog("session-9")

        self.assertEqual(drained, [payload])
        self.assertEqual(drained_again, [])


if __name__ == "__main__":
    unittest.main()
