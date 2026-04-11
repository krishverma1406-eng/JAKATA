"""Deterministic regression checks for tool fallback chaining."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import Agent


class DummyBrain:
    """Return scripted tool calls and record what the agent exposed each turn."""

    def __init__(self, scripted_responses: list[dict[str, Any]]) -> None:
        self._responses = list(scripted_responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_kind: str,
        stream_handler: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "tools": [tool["name"] for tool in tools],
                "task_kind": task_kind,
            }
        )
        if not self._responses:
            return {"content": "No scripted response.", "tool_calls": []}
        response = self._responses.pop(0)
        response.setdefault("content", "")
        response.setdefault("tool_calls", [])
        return response


class DummyPlanner:
    def create_plan(self, user_message: str, tool_definitions: list[dict[str, Any]]) -> dict[str, Any]:
        return {"needs_planning": False, "steps": []}

    def render_plan(self, plan: dict[str, Any]) -> str:
        return "No explicit plan required."

    def next_matching_step_index(self, plan: dict[str, Any], tool_name: str) -> None:
        return None

    def mark_step_completed(self, plan: dict[str, Any], tool_name: str | None = None, step_index: int | None = None) -> None:
        return None


class DummyMemory:
    def remember(self, *args: Any, **kwargs: Any) -> None:
        return None

    def note_response(self, *args: Any, **kwargs: Any) -> None:
        return None

    def brief_for_ui(self, session_id: str) -> str:
        return ""


class DummyTools:
    def __init__(self, definitions: list[dict[str, Any]], results: dict[str, list[dict[str, Any]]]) -> None:
        self._definitions = definitions
        self._results = {name: list(items) for name, items in results.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def refresh(self) -> None:
        return None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return list(self._definitions)

    def run_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((name, arguments))
        queue = self._results.get(name, [])
        if queue:
            return queue.pop(0)
        return {"ok": True, "tool": name}


TOOL_DEFINITIONS = [
    {"name": "app_launcher_tool", "description": "Launch local apps."},
    {"name": "os_control", "description": "Desktop automation and open app actions."},
    {"name": "browser_control", "description": "Open and control browser pages."},
    {"name": "web_search", "description": "Search the web."},
    {"name": "weather_tool", "description": "Weather lookup."},
    {"name": "gmail_tool", "description": "Gmail access."},
]


def _make_agent(brain: DummyBrain, tools: DummyTools) -> Agent:
    agent = Agent.__new__(Agent)
    agent.settings = type("Settings", (), {"agent_max_iterations": 6})()
    agent.mode = "normal"
    agent.session_id = "test-session"
    agent.session_meta = {"display_name": "Test Session"}
    agent.brain = brain
    agent.memory = DummyMemory()
    agent.planner = DummyPlanner()
    agent.tools = tools
    agent.interface = type("Interface", (), {"get_mode": lambda self, mode: {"show_debug": False}})()
    agent._load_memory_context = lambda user_message: []
    agent._build_messages = (
        lambda user_message, memory_context, plan, plan_note=None, mode_config=None: [
            {"role": "user", "content": user_message}
        ]
    )
    agent._task_kind = lambda user_message, plan: "chat"
    agent._finalize_response = lambda user_message, final_answer, messages, tool_trace, turn_meta=None: final_answer
    agent._select_tool_definitions = lambda user_message, all_tool_definitions, plan, memory_context, runtime_context=None: list(all_tool_definitions)
    return agent


def test_app_launcher_chain() -> None:
    brain = DummyBrain(
        [
            {
                "tool_calls": [
                    {"id": "1", "name": "app_launcher_tool", "arguments": {"app_name": "VS Code"}},
                ]
            },
            {
                "tool_calls": [
                    {"id": "2", "name": "os_control", "arguments": {"action": "open_app", "target": "code"}},
                ]
            },
            {"content": "VS Code is now open.", "tool_calls": []},
        ]
    )
    tools = DummyTools(
        TOOL_DEFINITIONS,
        {
            "app_launcher_tool": [{"ok": False, "error": "executable not found"}],
            "os_control": [{"ok": True, "opened": "code"}],
        },
    )
    agent = _make_agent(brain, tools)
    events: list[dict[str, Any]] = []

    final_answer = agent.run("Open VS Code.", event_handler=events.append)

    assert final_answer == "VS Code is now open."
    assert [name for name, _ in tools.calls] == ["app_launcher_tool", "os_control"]
    assert brain.calls[1]["tools"][:3] == ["app_launcher_tool", "os_control", "browser_control"]
    latest_system = [m for m in brain.calls[1]["messages"] if m.get("role") == "system"][-1]["content"]
    assert "app_launcher_tool" in latest_system
    assert "os_control, browser_control" in latest_system
    fallback_event = [event for event in events if event.get("type") == "tool_fallback_suggested"][-1]
    assert fallback_event["failed_tool"] == "app_launcher_tool"
    assert fallback_event["fallback_tools"] == ["os_control", "browser_control", "terminal_tool"][: len(fallback_event["fallback_tools"])]


def test_weather_chain() -> None:
    brain = DummyBrain(
        [
            {
                "tool_calls": [
                    {"id": "1", "name": "weather_tool", "arguments": {"location": "Delhi"}},
                ]
            },
            {
                "tool_calls": [
                    {"id": "2", "name": "web_search", "arguments": {"query": "current weather Delhi"}},
                ]
            },
            {"content": "I checked the weather via web search after the weather API failed.", "tool_calls": []},
        ]
    )
    tools = DummyTools(
        TOOL_DEFINITIONS,
        {
            "weather_tool": [{"ok": False, "error": "503 upstream unavailable"}],
            "web_search": [{"ok": True, "results": ["Delhi weather is clear."]}],
        },
    )
    agent = _make_agent(brain, tools)

    final_answer = agent.run("Check Delhi weather.")

    assert "weather via web search" in final_answer
    assert [name for name, _ in tools.calls] == ["weather_tool", "web_search"]
    latest_system = [m for m in brain.calls[1]["messages"] if m.get("role") == "system"][-1]["content"]
    assert "web_search" in latest_system


def test_auth_error_does_not_fallback() -> None:
    brain = DummyBrain(
        [
            {
                "tool_calls": [
                    {"id": "1", "name": "gmail_tool", "arguments": {"action": "list"}},
                ]
            },
            {"content": "Gmail OAuth is not configured.", "tool_calls": []},
        ]
    )
    tools = DummyTools(
        TOOL_DEFINITIONS,
        {
            "gmail_tool": [{"ok": False, "error": "OAuth token missing"}],
        },
    )
    agent = _make_agent(brain, tools)
    events: list[dict[str, Any]] = []

    final_answer = agent.run("Check my inbox.", event_handler=events.append)

    assert final_answer == "Gmail OAuth is not configured."
    assert [name for name, _ in tools.calls] == ["gmail_tool"]
    assert len(brain.calls) == 2
    assert not [event for event in events if event.get("type") == "tool_fallback_suggested"]


def main() -> None:
    tests = [
        ("app_launcher fallback chain", test_app_launcher_chain),
        ("weather fallback chain", test_weather_chain),
        ("auth error stops fallback", test_auth_error_does_not_fallback),
    ]
    print("Tool fallback regression tests")
    print("=" * 36)
    failures = 0
    for label, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {label}")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {label}: {exc}")
    if failures:
        raise SystemExit(1)
    print("=" * 36)
    print("All fallback checks passed.")


if __name__ == "__main__":
    main()
