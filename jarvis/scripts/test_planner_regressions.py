"""Regression checks for planner heuristics and fallback splitting."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.planner import Planner


class DummyBrain:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.settings = type(
            "Settings",
            (),
            {
                "planner_simple_word_limit": 8,
                "planner_complex_word_limit": 18,
            },
        )()
        self.payload = payload or {
            "needs_planning": True,
            "strategy": "Use the simple model for compact JSON planning.",
            "steps": [
                {"step": "Search the web", "tool_name": "web_search", "tool_names": ["web_search"]},
                {"step": "Summarize the result", "tool_name": None, "tool_names": []},
            ],
        }
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_kind: str = "simple",
        response_format: str | None = None,
        system_override: str | None = None,
        stream_handler: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
                "task_kind": task_kind,
                "response_format": response_format,
                "system_override": system_override,
            }
        )
        return {"content": json.dumps(self.payload), "tool_calls": []}


class PlannerRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool_definitions = [
            {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "gmail_tool",
                "description": "Send emails and check inbox messages.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def test_create_plan_uses_simple_task_kind(self) -> None:
        brain = DummyBrain()
        planner = Planner(brain)

        plan = planner.create_plan(
            "Search the web then send an email with the answer carefully for me.",
            self.tool_definitions,
        )

        self.assertTrue(plan["needs_planning"])
        self.assertEqual(brain.calls[-1]["task_kind"], "simple")
        self.assertEqual(brain.calls[-1]["response_format"], "json")

    def test_fallback_steps_keep_before_phrases_intact(self) -> None:
        planner = Planner(DummyBrain())
        cases = {
            "Search the web before answering": ["Search the web before answering"],
            "Before you do that, check the file": ["Before you do that, check the file"],
            "back up the folder before deleting": ["back up the folder before deleting"],
        }

        for task, expected_steps in cases.items():
            with self.subTest(task=task):
                steps = planner._fallback_steps(task, self.tool_definitions)
                self.assertEqual([step["step"] for step in steps], expected_steps)

    def test_short_and_request_does_not_trigger_planning(self) -> None:
        planner = Planner(DummyBrain())

        should_plan = planner.should_plan("open YouTube and play music", [])

        self.assertFalse(should_plan)

    def test_short_multi_tool_request_still_triggers_planning(self) -> None:
        planner = Planner(DummyBrain())

        should_plan = planner.should_plan(
            "search the web and send an email with the results",
            self.tool_definitions,
        )

        self.assertTrue(should_plan)


if __name__ == "__main__":
    unittest.main()
