"""Regression coverage for architecture-level provider, fallback, and server flows."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from core.agent import Agent
from core.brain import Brain
import server


class BrainArchitectureRegressionTests(unittest.TestCase):
    def test_chat_skips_nvidia_when_circuit_is_open(self) -> None:
        brain = Brain()
        brain._call_nvidia = Mock(side_effect=AssertionError("nvidia should be skipped"))  # type: ignore[method-assign]
        brain._call_groq = Mock(return_value={"content": "groq response", "tool_calls": []})  # type: ignore[method-assign]
        brain._call_openrouter = Mock(return_value={"content": "openrouter response", "tool_calls": []})  # type: ignore[method-assign]

        with patch.object(brain._circuit, "is_open", side_effect=lambda provider: provider == "nvidia"):
            response = brain.chat(messages=[{"role": "user", "content": "hello"}])

        brain._call_nvidia.assert_not_called()
        brain._call_groq.assert_called_once()
        self.assertEqual(response["content"], "groq response")

    def test_chat_skips_groq_when_its_circuit_is_open(self) -> None:
        brain = Brain()
        brain._call_nvidia = Mock(side_effect=RuntimeError("nvidia unavailable"))  # type: ignore[method-assign]
        brain._call_groq = Mock(side_effect=AssertionError("groq should be skipped"))  # type: ignore[method-assign]
        brain._call_openrouter = Mock(return_value={"content": "openrouter response", "tool_calls": []})  # type: ignore[method-assign]

        with patch.object(brain._circuit, "is_open", side_effect=lambda provider: provider == "groq"):
            response = brain.chat(messages=[{"role": "user", "content": "hello"}])

        brain._call_groq.assert_not_called()
        brain._call_openrouter.assert_called_once()
        self.assertEqual(response["content"], "openrouter response")


class AgentFallbackRegressionTests(unittest.TestCase):
    def test_execute_with_chain_emits_fallback_event_and_uses_timeout_wrapper(self) -> None:
        agent = Agent.__new__(Agent)
        calls: list[tuple[str, int]] = []

        class DummyTools:
            @staticmethod
            def has_tool(name: str) -> bool:
                return name == "web_search"

        def fake_run_tool_with_timeout(
            tool_name: str,
            _arguments: dict[str, object],
            _runtime_context: dict[str, object],
            timeout: int = 0,
        ) -> dict[str, object]:
            calls.append((tool_name, timeout))
            if tool_name == "weather_tool":
                return {"ok": False, "error": "provider down"}
            return {"ok": True, "results": ["fallback worked"]}

        agent.tools = DummyTools()
        agent._run_tool_with_timeout = fake_run_tool_with_timeout  # type: ignore[method-assign]
        agent._tool_error_blocks_fallback = Agent._tool_error_blocks_fallback.__get__(agent, Agent)
        agent._adapt_args_for_fallback = Agent._adapt_args_for_fallback.__get__(agent, Agent)

        events: list[dict[str, object]] = []
        result = Agent._execute_with_chain(
            agent,
            "weather_tool",
            {"location": "Delhi"},
            {},
            event_handler=events.append,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["_fallback_used"])
        self.assertEqual(calls, [("weather_tool", 30), ("web_search", 30)])
        self.assertEqual(events[-1]["type"], "tool_fallback_suggested")
        self.assertEqual(events[-1]["failed_tool"], "weather_tool")
        self.assertEqual(events[-1]["fallback_tools"], ["web_search"])


class ServerMemoryRegressionTests(unittest.TestCase):
    def test_get_memory_records_filters_and_counts_tags(self) -> None:
        records = [
            {"id": "1", "tag": "PROJECT", "text": "Project A", "active": True, "demoted": False, "updated_at": "2026-04-11T10:00:00"},
            {"id": "2", "tag": "FACT", "text": "Fact A", "active": True, "demoted": False, "updated_at": "2026-04-11T09:00:00"},
            {"id": "3", "tag": "PROJECT", "text": "Old Project", "active": True, "demoted": True, "updated_at": "2026-04-10T09:00:00"},
        ]
        fake_memory = types.SimpleNamespace(_load_records_payload=lambda: {"records": copy.deepcopy(records)})

        with patch("server._get_memory", return_value=fake_memory):
            payload = server.get_memory_records(tag="PROJECT", limit=10, include_demoted=False)

        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["records"][0]["id"], "1")
        self.assertEqual(payload["tags"]["PROJECT"], 1)
        self.assertEqual(payload["tags"]["FACT"], 0)

    def test_delete_memory_record_updates_payload_and_rebuilds_memory(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.payload = {
                    "records": [
                        {"id": "1", "tag": "PROJECT", "text": "Project A"},
                        {"id": "2", "tag": "FACT", "text": "Fact A"},
                    ]
                }
                self.saved_payload: dict[str, object] | None = None
                self.rebuilt = False

            def _load_records_payload(self) -> dict[str, object]:
                return copy.deepcopy(self.payload)

            def _now_iso(self) -> str:
                return "2026-04-11T12:00:00"

            def _save_records_payload(self, payload: dict[str, object]) -> None:
                self.saved_payload = payload

            def _rebuild_materialized_memory(self, stored_records_changed: bool = False) -> None:
                self.rebuilt = stored_records_changed

        fake_memory = FakeMemory()
        with patch("server._get_memory", return_value=fake_memory):
            payload = server.delete_memory_record("1")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["deleted"], 1)
        self.assertEqual(payload["items"], ["Project A"])
        self.assertIsNotNone(fake_memory.saved_payload)
        self.assertEqual(fake_memory.saved_payload["records"], [{"id": "2", "tag": "FACT", "text": "Fact A"}])
        self.assertTrue(fake_memory.rebuilt)


class SourceRegressionTests(unittest.TestCase):
    def test_server_and_frontend_wire_fallback_activity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        server_source = (root / "server.py").read_text(encoding="utf-8")
        frontend_source = (root / "frontend" / "script.js").read_text(encoding="utf-8")
        self.assertIn("tool_fallback_suggested", server_source)
        self.assertIn("tool_fallback_suggested", frontend_source)


if __name__ == "__main__":
    unittest.main()
