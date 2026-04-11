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


if __name__ == "__main__":
    unittest.main()
