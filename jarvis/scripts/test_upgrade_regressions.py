"""Regression tests for reliability/performance upgrades."""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from core.brain import Brain, ProviderCircuitBreaker
from core.planner import Planner
from core.session_store import SessionStore
from core.agent import Agent


class UpgradeRegressionTests(unittest.TestCase):
    def test_provider_circuit_breaker_opens_after_threshold(self) -> None:
        breaker = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        with patch("core.brain.time.time", return_value=100.0):
            breaker.record_failure("nvidia")
            breaker.record_failure("nvidia")
            self.assertFalse(breaker.is_open("nvidia"))
            breaker.record_failure("nvidia")
            self.assertTrue(breaker.is_open("nvidia"))

        with patch("core.brain.time.time", return_value=200.0):
            breaker.record_success("nvidia")
            self.assertFalse(breaker.is_open("nvidia"))

    def test_compile_messages_trims_to_token_limit(self) -> None:
        brain = Brain()
        brain._max_input_tokens = 10
        _, compiled = brain._compile_messages(
            messages=[
                {"role": "user", "content": "a" * 60},
                {"role": "assistant", "content": "b" * 60},
                {"role": "user", "content": "short"},
            ],
            system_override=None,
            tool_definitions=[],
        )
        self.assertLessEqual(brain._estimate_tokens(compiled), 10)
        self.assertEqual(compiled[-1]["content"], "short")

    def test_trim_to_token_limit_drops_assistant_tool_pairs_together(self) -> None:
        brain = Brain()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "weather_tool", "arguments": {}}],
            },
            {
                "role": "tool",
                "content": '{"ok": true}',
                "tool_call_id": "call_1",
                "name": "weather_tool",
            },
            {"role": "user", "content": "latest update"},
        ]

        trimmed = brain._trim_to_token_limit(messages, token_limit=1)
        self.assertEqual(trimmed, [{"role": "user", "content": "latest update"}])

    def test_planner_skips_brain_for_single_likely_tool(self) -> None:
        fake_brain = Mock()
        planner = Planner(brain=fake_brain)
        tool_defs = [{"name": "weather_tool"}, {"name": "web_search"}]
        with patch.object(planner, "should_plan", return_value=True):
            plan = planner.create_plan("weather forecast tomorrow and next week", tool_defs)
        self.assertFalse(plan["needs_planning"])
        fake_brain.chat.assert_not_called()

    def test_session_name_stopwords_filter_question_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            name = store._derive_name_from_text("what can you do for me today")
        self.assertNotIn("what", name.lower())
        self.assertNotIn("can", name.lower())

    def test_run_tool_with_timeout_returns_quickly(self) -> None:
        class SlowTools:
            @staticmethod
            def run_tool(_tool_name: str, _arguments: dict, _runtime_context: dict) -> dict:
                time.sleep(0.8)
                return {"ok": True}

        dummy = types.SimpleNamespace(tools=SlowTools())
        started = time.perf_counter()
        result = Agent._run_tool_with_timeout(dummy, "slow_tool", {}, {}, timeout=0.1)
        elapsed = time.perf_counter() - started
        self.assertFalse(result.get("ok", True))
        self.assertIn("timed out", result.get("error", ""))
        self.assertLess(elapsed, 0.5)

    def test_execute_with_chain_skips_fallback_after_timeout(self) -> None:
        class SlowTools:
            calls: list[str] = []

            def has_tool(self, _name: str) -> bool:
                return True

            def run_tool(self, tool_name: str, _arguments: dict, _runtime_context: dict) -> dict:
                self.calls.append(tool_name)
                return {"ok": True, "tool": tool_name}

        dummy = types.SimpleNamespace(
            tools=SlowTools(),
            _adapt_args_for_fallback=lambda *_args, **_kwargs: {},
        )
        dummy._tool_error_blocks_fallback = Agent._tool_error_blocks_fallback.__get__(dummy, type(dummy))
        dummy._run_tool_with_timeout = lambda tool_name, *_args, **_kwargs: (
            {"ok": False, "error": f"{tool_name} timed out after 0.1s"}
            if tool_name == "app_launcher_tool"
            else {"ok": True, "tool": tool_name}
        )
        result = Agent._execute_with_chain(dummy, "app_launcher_tool", {}, {})
        self.assertFalse(result.get("ok", True))
        self.assertEqual(dummy.tools.calls, [])


if __name__ == "__main__":
    unittest.main()
