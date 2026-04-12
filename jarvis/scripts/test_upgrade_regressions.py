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
from core.memory import Memory


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

    def test_execute_with_chain_uses_fallback_after_timeout(self) -> None:
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
        dummy._tool_is_registered = lambda _name: True
        dummy._tool_error_blocks_fallback = Agent._tool_error_blocks_fallback.__get__(dummy, type(dummy))
        dummy._run_tool_with_timeout = lambda tool_name, *_args, **_kwargs: (
            {"ok": False, "error": f"{tool_name} timed out after 0.1s"}
            if tool_name == "app_launcher_tool"
            else {"ok": True, "tool": tool_name}
        )
        result = Agent._execute_with_chain(dummy, "app_launcher_tool", {}, {})
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("tool"), "terminal_tool")
        self.assertTrue(result.get("_fallback_used"))
        self.assertEqual(dummy.tools.calls, [])

    def test_nvidia_thinking_flag_only_added_for_supported_models(self) -> None:
        brain = Brain()
        brain.settings = types.SimpleNamespace(
            nvidia_api_key="test-key",
            nvidia_base_url="https://example.com/v1",
            nvidia_timeout_seconds=30,
        )
        captured: dict[str, object] = {}

        def fake_call_openai_compatible(**kwargs):
            captured.clear()
            captured.update(kwargs)
            return {"ok": True}

        brain._call_openai_compatible = fake_call_openai_compatible  # type: ignore[method-assign]

        with patch.object(brain, "_nvidia_model_for", return_value="meta/llama-3.3-70b-instruct"):
            brain._call_nvidia(["system"], [{"role": "user", "content": "hello"}], [], "simple", None)
        self.assertNotIn("chat_template_kwargs", captured["payload"])

        with patch.object(brain, "_nvidia_model_for", return_value="deepseek-r1"):
            brain._call_nvidia(["system"], [{"role": "user", "content": "hello"}], [], "simple", None)
        self.assertEqual(captured["payload"]["chat_template_kwargs"], {"thinking": False})

    def test_task_kind_does_not_false_trigger_on_plain_api_or_class_language(self) -> None:
        agent = Agent.__new__(Agent)
        simple_plan = {"steps": [], "needs_planning": False}

        self.assertEqual(
            Agent._task_kind(agent, "what api key do i need for the weather tool", simple_plan),
            "simple",
        )
        self.assertEqual(
            Agent._task_kind(agent, "what is my class schedule today", simple_plan),
            "simple",
        )

    def test_task_kind_still_flags_real_code_requests(self) -> None:
        agent = Agent.__new__(Agent)
        simple_plan = {"steps": [], "needs_planning": False}

        self.assertEqual(
            Agent._task_kind(agent, "fix the api error in app.py", simple_plan),
            "code",
        )

    def test_should_answer_code_directly_stays_false_when_request_mentions_running_tests(self) -> None:
        agent = Agent.__new__(Agent)
        agent._explicit_code_writer_request = lambda _message: False

        self.assertFalse(
            Agent._should_answer_code_directly(
                agent,
                "write a python function to check palindrome and run it with test cases",
            )
        )

    def test_recall_from_sessions_accepts_created_at_date_match(self) -> None:
        memory = Memory.__new__(Memory)
        memory.settings = types.SimpleNamespace(memory_top_k=3)
        memory.session_store = types.SimpleNamespace(
            list_sessions=lambda limit=20: [
                {
                    "session_id": "abc",
                    "display_name": "Yesterday",
                    "created_at": "2026-04-11T09:00:00",
                    "updated_at": "2026-04-12T08:00:00",
                }
            ],
            load_messages=lambda _session_id, limit_messages=120: [
                {"role": "user", "content": "what did we discuss yesterday"},
                {"role": "assistant", "content": "We discussed the Yantra code."},
            ],
        )
        memory._session_target_date = lambda _query: "2026-04-11"
        memory._session_turn_summaries = Memory._session_turn_summaries.__get__(memory, Memory)
        memory._session_excerpt = Memory._session_excerpt
        memory._token_overlap_score = lambda _query, _candidate: 1.0
        memory._query_focus_bonus = lambda _query, _candidate: 0.0

        results = Memory._recall_from_sessions(memory, "what did we talk about yesterday", 3)

        self.assertEqual(len(results), 1)
        self.assertIn("Yantra code", results[0])

    def test_frontend_handles_stream_cancelled_events(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "frontend" / "script.js").read_text(encoding="utf-8")
        self.assertIn("if (data.stream_cancelled)", script)
        self.assertIn("stream-placeholder", script)

    def test_memory_fix_script_uses_include_facts_parameter(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "test_memory_fixes.py").read_text(encoding="utf-8")
        self.assertIn("_candidate_pool(include_facts=False)", source)
        self.assertNotIn("include_chunks", source)


if __name__ == "__main__":
    unittest.main()
