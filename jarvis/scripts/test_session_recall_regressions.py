"""Regression coverage for session-based recall and summarization."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import Agent
from core.memory import Memory
from tools import session_tool


class SessionRecallRegressionTests(unittest.TestCase):
    def test_recall_short_circuits_to_session_path_for_conversation_queries(self) -> None:
        memory = Memory.__new__(Memory)
        memory.settings = types.SimpleNamespace(memory_top_k=5)
        memory._is_recent_conversation_query = Mock(return_value=True)
        memory._recall_from_sessions = Mock(return_value=["Recent session summary"])
        memory._semantic_retrieval_ready = Mock(side_effect=AssertionError("semantic lookup should be skipped"))

        results = Memory.recall(memory, "what did we talk about yesterday", limit=3)

        self.assertEqual(results, ["Recent session summary"])
        memory._recall_from_sessions.assert_called_once_with("what did we talk about yesterday", 3)
        memory._semantic_retrieval_ready.assert_not_called()

    def test_recall_from_sessions_reads_actual_session_messages(self) -> None:
        memory = Memory.__new__(Memory)
        memory.session_store = Mock()
        memory.session_store.list_sessions.return_value = [
            {
                "session_id": "session-1",
                "display_name": "Dashboard Build",
                "updated_at": "2026-04-10T14:40:00+05:30",
            },
            {
                "session_id": "session-2",
                "display_name": "Test Cleanup",
                "updated_at": "2026-04-09T09:15:00+05:30",
            },
        ]
        memory.session_store.load_messages.side_effect = lambda session_id, limit_messages=20: {
            "session-1": [
                {"role": "user", "content": "Let's build the analytics dashboard."},
                {"role": "assistant", "content": "I wired the analytics dashboard cards and charts."},
                {"role": "user", "content": "Please fix the broken filters too."},
                {"role": "assistant", "content": "I fixed the dashboard filters and verified the state flow."},
            ],
            "session-2": [
                {"role": "user", "content": "Clean up the flaky tests."},
                {"role": "assistant", "content": "I stabilized the flaky tests and removed the stale fixture."},
            ],
        }[session_id]

        results = Memory._recall_from_sessions(memory, "what did we build last session", 4)

        self.assertGreaterEqual(len(results), 1)
        self.assertIn('Dashboard Build', results[0])
        self.assertIn('analytics dashboard', " ".join(results).lower())
        self.assertIn('JARVIS replied', results[0])


class SessionToolRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        session_tool._MEMORY = None

    def test_session_tool_summarize_returns_recent_turn_pairs(self) -> None:
        memory = Mock()
        memory.list_sessions.return_value = [
            {
                "session_id": "session-1",
                "display_name": "Dashboard Build",
                "updated_at": "2026-04-10T14:40:00+05:30",
            }
        ]
        memory.load_session_messages.return_value = [
            {"role": "user", "content": "Let's build the analytics dashboard."},
            {"role": "assistant", "content": "I wired the analytics dashboard cards and charts."},
            {"role": "user", "content": "Please fix the broken filters too."},
            {"role": "assistant", "content": "I fixed the dashboard filters and verified the state flow."},
        ]

        with patch("tools.session_tool.Memory", return_value=memory):
            result = session_tool.execute({"action": "summarize", "limit": 2})

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["sessions"]), 1)
        summary = result["sessions"][0]
        self.assertEqual(summary["session_name"], "Dashboard Build")
        self.assertEqual(summary["turn_count"], 2)
        self.assertEqual(summary["turns"][0]["user"], "Let's build the analytics dashboard.")
        self.assertIn("dashboard filters", summary["turns"][1]["assistant"])

    def test_session_tool_summarize_uses_runtime_session_id_when_present(self) -> None:
        memory = Mock()
        memory.get_session.return_value = {
            "session_id": "session-7",
            "display_name": "Current Work",
            "updated_at": "2026-04-11T11:20:00+05:30",
        }
        memory.load_session_messages.return_value = [
            {"role": "user", "content": "Summarize the current work."},
            {"role": "assistant", "content": "We reviewed the active branch and latest failures."},
        ]

        with patch("tools.session_tool.Memory", return_value=memory):
            result = session_tool.execute(
                {"action": "summarize", "_runtime_context": {"session_id": "session-7"}}
            )

        self.assertTrue(result["ok"])
        memory.get_session.assert_called_once_with("session-7")
        memory.list_sessions.assert_not_called()
        self.assertEqual(result["sessions"][0]["turn_count"], 1)


class SessionToolRoutingRegressionTests(unittest.TestCase):
    def test_conversation_summary_queries_force_session_tool(self) -> None:
        agent = Agent.__new__(Agent)
        tool_definitions = [
            {"name": "session_tool"},
            {"name": "memory_query"},
        ]

        forced = Agent._forced_tool_names(
            agent,
            "what did we talk about yesterday in previous sessions",
            tool_definitions,
            runtime_context=None,
        )

        self.assertEqual(forced, {"session_tool"})


if __name__ == "__main__":
    unittest.main()
