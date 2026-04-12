"""Prompt-driven regression coverage for the 25 manual test questions."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from core.agent import Agent


def make_agent() -> Agent:
    agent = Agent.__new__(Agent)
    agent.settings = types.SimpleNamespace(reminder_timezone="Asia/Calcutta")
    agent.memory = Mock()
    agent.brain = Mock()
    agent.last_tool_trace = []
    agent.pending_confirmation = None
    agent._runtime_policy_cache = {}
    agent._runtime_policy_mtime_ns = None
    agent._runtime_policy = lambda: {}
    agent._policy_markers = Agent._policy_markers.__get__(agent, Agent)
    agent._matches_policy = Agent._matches_policy.__get__(agent, Agent)
    agent._runtime_resolution = Agent._runtime_resolution
    agent._format_local_time = Agent._format_local_time.__get__(agent, Agent)
    agent._format_tool_result = Agent._format_tool_result.__get__(agent, Agent)
    agent._latest_tool_arguments = Agent._latest_tool_arguments
    agent._queue_tool_confirmation = Agent._queue_tool_confirmation.__get__(agent, Agent)
    agent._is_session_summary_request = Agent._is_session_summary_request.__get__(agent, Agent)
    agent._tool_is_compatible = Agent._tool_is_compatible.__get__(agent, Agent)
    agent._forced_tool_names = Agent._forced_tool_names.__get__(agent, Agent)
    agent._explicit_code_writer_request = Agent._explicit_code_writer_request.__get__(agent, Agent)
    agent._strip_code_fences = Agent._strip_code_fences
    agent._extract_quoted_text = Agent._extract_quoted_text
    agent._extract_email_address = Agent._extract_email_address
    agent._extract_reply_body = Agent._extract_reply_body
    agent._last_note_title = lambda: ""
    agent._last_successful_tool_result = lambda _tool_name: {}
    return agent


class PromptRegressionTests(unittest.TestCase):
    def test_q01_name_and_location(self) -> None:
        agent = make_agent()
        agent.memory.recall.return_value = [
            "Your name is Krish.",
            "You live in Delhi, originally from Hisar, Haryana.",
        ]

        result = Agent._canonical_profile_response(agent, "what is my name and where do i live", {})

        self.assertEqual(
            result["response"],
            "Your name is Krish. You live in Delhi, originally from Hisar, Haryana.",
        )

    def test_q02_yesterday_session_summary(self) -> None:
        agent = make_agent()
        agent.memory.session_recall.return_value = [
            'In session "Debug" (2026-04-11): You asked: "Fix memory recall" - JARVIS replied: "I fixed the session date filter."'
        ]
        agent.memory.active_project_catalog.side_effect = AssertionError("project catalog should not be used for session recall")

        result = Agent._session_runtime_response(agent, "what did we talk about yesterday", {})

        self.assertIn("closest saved exchanges", result["response"].lower())
        self.assertIn("Fix memory recall", result["response"])
        agent.memory.session_recall.assert_called_once()

    def test_q03_school_lookup(self) -> None:
        agent = make_agent()
        agent.memory.recall.return_value = ["You study at Everest Public School."]

        result = Agent._canonical_profile_response(agent, "what school do i study in", {})

        self.assertEqual(result["response"], "You study at Everest Public School.")

    def test_q04_reminder_prompt_routes_to_reminder_tool_and_formats_time(self) -> None:
        agent = make_agent()
        forced = Agent._forced_tool_names(
            agent,
            "remind me to push yantra code in 10 minutes",
            [{"name": "reminder_tool"}, {"name": "notes_tool"}],
        )
        trace = [{"name": "reminder_tool", "arguments": {"action": "create"}}]
        summary = Agent._format_tool_result(
            agent,
            "reminder_tool",
            {
                "ok": True,
                "reminder": {
                    "text": "push yantra code",
                    "due_at": "2026-04-12T10:30:00+00:00",
                },
            },
            trace,
        )

        self.assertEqual(forced, {"reminder_tool"})
        self.assertIn("Done - I'll remind you to push yantra code at", summary)

    def test_q05_weather_prefers_weather_tool(self) -> None:
        agent = make_agent()

        forced = Agent._forced_tool_names(
            agent,
            "whats the weather like",
            [{"name": "weather_tool"}, {"name": "web_search"}],
        )

        self.assertEqual(forced, {"weather_tool"})

    def test_q06_youtube_search_uses_browser_control(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(return_value={"ok": True})

        result = Agent._browser_runtime_response(agent, "open youtube and search for lo fi music", {})

        self.assertIn("Opened YouTube search results for lo fi music.", result["response"])
        tool_name, arguments, *_ = agent._run_workflow_tool.call_args.args
        self.assertEqual(tool_name, "browser_control")
        self.assertIn("youtube.com/results?search_query=lo+fi+music", arguments["url"])

    def test_q07_calculator_chain(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {"ok": True, "result": 127.05},
                {"ok": True, "result": 10608.675},
            ]
        )

        result = Agent._calculator_runtime_response(
            agent,
            "what is 15% of 847 and convert that to rupees if 1 dollar is 83.5",
            {},
        )

        self.assertEqual(agent._run_workflow_tool.call_count, 2)
        self.assertIn("15% of 847 is 127.05.", result["response"])
        self.assertIn("Rs.10,608.68", result["response"])

    def test_q08_system_slow_checks_overview_and_processes(self) -> None:
        agent = make_agent()
        scripted_results = [
            {"ok": True, "cpu_percent": 72.4, "ram_percent": 81.1},
            {"ok": True, "processes": [{"name": "chrome", "cpu_percent": 41.2, "memory_percent": 12.0}]},
            {"ok": True, "alerts": [{"message": "Background load is elevated."}]},
        ]

        def fake_run_workflow_tool(tool_name, arguments, _runtime_context, **kwargs):
            result = scripted_results.pop(0)
            trace = kwargs.get("tool_trace", [])
            trace.append({"name": tool_name, "arguments": dict(arguments), "result": result})
            return result

        agent._run_workflow_tool = Mock(side_effect=fake_run_workflow_tool)

        result = Agent._system_runtime_response(agent, "my cpu is feeling slow check whats happening", {})

        calls = [call.args[1]["action"] for call in agent._run_workflow_tool.call_args_list]
        self.assertEqual(calls, ["overview", "processes", "alerts"])
        self.assertIn("CPU at 72.4%", result["response"])
        self.assertIn("Top processes: chrome", result["response"])

    def test_q09_palindrome_prompt_runs_code_runner(self) -> None:
        agent = make_agent()
        agent.brain.chat.return_value = {
            "content": (
                "def is_palindrome(s):\n"
                "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
                "for value in ('racecar', 'hello', 'madam'):\n"
                "    print(value, is_palindrome(value))"
            )
        }
        agent._run_workflow_tool = Mock(
            return_value={"ok": True, "stdout": "racecar True\nhello False\nmadam True", "stderr": "", "exit_code": 0}
        )

        result = Agent._code_execution_response(
            agent,
            "write a python function that checks if a string is a palindrome and run it with test cases",
            {},
        )

        self.assertEqual(agent._run_workflow_tool.call_args.args[0], "code_runner")
        self.assertIn("Output: racecar True", result["response"])
        self.assertNotIn("code_writer", result["response"])

    def test_q10_web_search_then_save_note(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {
                    "ok": True,
                    "results": [
                        {"title": "Model A", "url": "https://example.com/a"},
                        {"title": "Model B", "url": "https://example.com/b"},
                    ],
                },
                {"ok": True, "path": "C:/notes/llm-news.md"},
            ]
        )

        result = Agent._note_search_save_response(
            agent,
            "search the web for latest news about llm models released this week and save a note about it",
            {},
        )

        self.assertEqual([call.args[0] for call in agent._run_workflow_tool.call_args_list], ["web_search", "notes_tool"])
        self.assertIn("Found 2 results. Note saved as", result["response"])

    def test_q11_screenshot_then_analyze(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {"ok": True, "path": "C:/tmp/screen.png"},
                {"ok": True, "analysis": "VS Code is open with the JARVIS project."},
            ]
        )

        result = Agent._screenshot_runtime_response(agent, "take a screenshot and tell me whats on my screen", {})

        self.assertEqual([call.args[1]["action"] for call in agent._run_workflow_tool.call_args_list], ["capture", "analyze"])
        self.assertEqual(result["response"], "VS Code is open with the JARVIS project.")

    def test_q12_delete_downloads_requires_confirmation(self) -> None:
        agent = make_agent()

        result = Agent._file_delete_runtime_response(agent, "delete all files in my downloads folder", {})

        self.assertTrue(result["turn_meta"]["confirmation_required"])
        self.assertIn("permanently delete", result["response"])
        self.assertIn("Downloads", result["response"])

    def test_q13_project_list_uses_active_project_catalog(self) -> None:
        agent = make_agent()
        agent.memory.active_project_catalog.return_value = [
            {"name": "Yantra", "detail": "UI bug fixes"},
            {"name": "Listify", "detail": "Shopping list sync"},
        ]

        result = Agent._project_catalog_response(agent, "what are we working on right now, give me the full project list", {})

        self.assertIn("Current active projects:", result["response"])
        self.assertIn("1. Yantra - UI bug fixes", result["response"])
        self.assertIn("2. Listify - Shopping list sync", result["response"])

    def test_q14_session_rename_routes_to_session_tool(self) -> None:
        agent = make_agent()
        forced = Agent._forced_tool_names(agent, 'rename this session to "Jarvis Debug Session"', [{"name": "session_tool"}])
        summary = Agent._format_tool_result(
            agent,
            "session_tool",
            {"ok": True, "session": {"display_name": "Jarvis Debug Session"}},
            [{"name": "session_tool", "arguments": {"action": "rename"}}],
        )

        self.assertEqual(forced, {"session_tool"})
        self.assertEqual(summary, "Session renamed to 'Jarvis Debug Session'.")

    def test_q15_local_music_prompt_is_compatible_with_music_player(self) -> None:
        agent = make_agent()

        compatible = Agent._tool_is_compatible(agent, "music_player", "play something chill locally", runtime_context=None, forced_tools=set())

        self.assertTrue(compatible)

    def test_q16_task_creation_routes_to_task_manager(self) -> None:
        agent = make_agent()
        forced = Agent._forced_tool_names(
            agent,
            "create a task for fixing the navbar bug in yantra with high priority",
            [{"name": "task_manager"}, {"name": "notes_tool"}],
        )
        summary = Agent._format_tool_result(
            agent,
            "task_manager",
            {"ok": True, "task": {"title": "Fix navbar bug", "project": "Yantra", "priority": "high"}},
            [{"name": "task_manager", "arguments": {"action": "create"}}],
        )

        self.assertEqual(forced, {"task_manager"})
        self.assertEqual(summary, "Task created - 'Fix navbar bug' in Yantra (high priority).")

    def test_q17_recent_sessions_about_memory_fixes(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {
                    "ok": True,
                    "sessions": [{"display_name": "Jarvis Debug Session"}, {"display_name": "Yantra Fixes"}],
                },
                {
                    "ok": True,
                    "sessions": [
                        {
                            "display_name": "Jarvis Debug Session",
                            "updated_at": "2026-04-12T09:15:00+05:30",
                            "snippets": ["You asked about memory fixes and session recall."],
                        }
                    ],
                },
            ]
        )

        result = Agent._session_runtime_response(
            agent,
            "list my recent sessions and tell me which one was about memory fixes",
            {},
        )

        self.assertEqual([call.args[1]["action"] for call in agent._run_workflow_tool.call_args_list], ["list", "search"])
        self.assertIn("Recent sessions: Jarvis Debug Session, Yantra Fixes.", result["response"])
        self.assertIn("memory fixes", result["response"].lower())

    def test_q18_unread_email_reply_requires_confirmation(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {
                    "ok": True,
                    "messages": [{"from": "Alice <alice@example.com>", "subject": "Navbar bug"}],
                },
                {
                    "ok": False,
                    "confirmation_required": True,
                    "prompt": 'Send reply to Alice <alice@example.com> about "Navbar bug" with this message: "got it, will look into it"?',
                },
            ]
        )

        result = Agent._gmail_runtime_response(
            agent,
            'check my unread emails and send a reply to the most recent one saying "got it, will look into it"',
            {},
        )

        self.assertEqual([call.args[0] for call in agent._run_workflow_tool.call_args_list], ["gmail_tool", "gmail_tool"])
        self.assertTrue(result["turn_meta"]["confirmation_required"])
        self.assertIn("Send reply to Alice <alice@example.com>", result["response"])

    def test_q19_image_generation_uses_image_gen_tool(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(return_value={"ok": True, "path": "C:/img/jarvis.png", "url": "http://viewer/jarvis"})

        result = Agent._image_generation_response(
            agent,
            "i want to generate an image of jarvis holographic interface from iron man",
            {},
        )

        self.assertEqual(agent._run_workflow_tool.call_args.args[0], "image_gen_tool")
        self.assertEqual(agent._run_workflow_tool.call_args.args[1]["style"], "digital-art")
        self.assertIn("jarvis holographic interface from iron man", agent._run_workflow_tool.call_args.args[1]["prompt"])
        self.assertIn("Image generated and saved at", result["response"])

    def test_q20_new_york_time_routes_to_datetime_tool(self) -> None:
        agent = make_agent()
        forced = Agent._forced_tool_names(
            agent,
            "what time is it in new york right now",
            [{"name": "datetime_tool"}, {"name": "web_search"}],
        )
        summary = Agent._format_tool_result(
            agent,
            "datetime_tool",
            {"ok": True, "time": "09:30 AM", "timezone": "America/New_York"},
            [{"name": "datetime_tool", "arguments": {"timezone": "America/New_York"}}],
        )

        self.assertEqual(forced, {"datetime_tool"})
        self.assertEqual(summary, "It's 09:30 AM in America/New_York.")

    def test_q21_explicit_code_runs_without_prediction(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            return_value={"ok": True, "stdout": "[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]", "stderr": "", "exit_code": 0}
        )

        result = Agent._code_execution_response(agent, "run this code for me: print([x**2 for x in range(10)])", {})

        self.assertFalse(agent.brain.chat.called)
        self.assertEqual(agent._run_workflow_tool.call_args.args[0], "code_runner")
        self.assertIn("[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]", result["response"])

    def test_q22_open_project_folder_in_vscode(self) -> None:
        agent = make_agent()
        agent._run_workflow_tool = Mock(
            side_effect=[
                {"ok": True, "path": "code"},
                {"ok": True, "program": "code", "arguments": ["C:/Users/anime/3D Objects/JAKATA/jarvis"]},
            ]
        )

        result = Agent._vscode_runtime_response(agent, "open my jarvis project folder in vscode", {})

        self.assertEqual([call.args[0] for call in agent._run_workflow_tool.call_args_list], ["terminal_tool", "terminal_tool"])
        self.assertIn("Opened", result["response"])
        self.assertIn("jarvis", result["response"].lower())

    def test_q23_trust_test_returns_three_specific_facts(self) -> None:
        agent = make_agent()
        agent.memory.recall.return_value = [
            "Your name is Krish.",
            "You study at Everest Public School.",
            "You live in Delhi, originally from Hisar, Haryana.",
        ]

        result = Agent._canonical_profile_response(
            agent,
            "are you actually using my memory or just guessing? tell me 3 specific things you know about me",
            {},
        )

        self.assertIn("Your name is Krish.", result["response"])
        self.assertIn("Everest Public School", result["response"])
        self.assertIn("Delhi", result["response"])

    def test_q24_jarvis_tool_request_marks_code_writer_intent(self) -> None:
        agent = make_agent()
        forced = Agent._forced_tool_names(
            agent,
            "write a tool that monitors clipboard and prints when it changes, save it as a jarvis tool",
            [{"name": "code_writer"}, {"name": "code_runner"}],
        )

        self.assertTrue(Agent._explicit_code_writer_request(agent, "write a tool that monitors clipboard and prints when it changes, save it as a jarvis tool"))
        self.assertEqual(forced, {"code_writer"})

    def test_q25_offline_capabilities_response(self) -> None:
        agent = make_agent()

        result = Agent._offline_capability_response(agent, "my wifi is down, what can you still do for me", {})

        self.assertIn("Without internet, I can still handle", result["response"])
        self.assertIn("local file work", result["response"])
        self.assertIn("web search", result["response"])


if __name__ == "__main__":
    unittest.main()
