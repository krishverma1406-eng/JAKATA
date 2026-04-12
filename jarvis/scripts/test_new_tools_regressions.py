"""Regression coverage for code_runner and task_manager."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from core.agent import Agent
from core.tool_registry import ToolRegistry
from tools import code_runner, task_manager


class CodeRunnerRegressionTests(unittest.TestCase):
    def test_executes_python_with_stdin(self) -> None:
        result = code_runner.execute(
            {
                "code": "name = input().strip()\nprint(name.upper())",
                "input_data": "jarvis\n",
                "timeout_seconds": 5,
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "JARVIS")
        self.assertEqual(result["exit_code"], 0)

    def test_blocks_dangerous_imports(self) -> None:
        result = code_runner.execute({"code": "import os\nprint(os.getcwd())"})

        self.assertFalse(result["ok"])
        self.assertIn("Blocked", result["error"])

    def test_times_out_long_running_code(self) -> None:
        result = code_runner.execute({"code": "while True:\n    pass", "timeout_seconds": 1})

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["exit_code"], -1)


class TaskManagerRegressionTests(unittest.TestCase):
    def test_create_list_complete_and_search_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            task_manager,
            "TASKS_FILE",
            Path(temp_dir) / "tasks.json",
        ):
            urgent = task_manager.execute(
                {
                    "action": "create",
                    "title": "Ship session summary",
                    "project": "JARVIS",
                    "priority": "urgent",
                    "description": "Wire summarize action into the session tool",
                }
            )
            low = task_manager.execute(
                {
                    "action": "create",
                    "title": "Tidy docs",
                    "project": "JARVIS",
                    "priority": "low",
                }
            )

            listed = task_manager.execute({"action": "list"})
            self.assertTrue(listed["ok"])
            self.assertEqual(listed["total"], 2)
            self.assertEqual(listed["tasks"][0]["id"], urgent["task"]["id"])
            self.assertEqual(listed["tasks"][1]["id"], low["task"]["id"])
            self.assertEqual(listed["by_project"]["JARVIS"], ["Ship session summary", "Tidy docs"])

            search = task_manager.execute({"action": "search", "query": "session"})
            self.assertTrue(search["ok"])
            self.assertEqual(len(search["matches"]), 1)
            self.assertEqual(search["matches"][0]["id"], urgent["task"]["id"])

            completed = task_manager.execute({"action": "complete", "task_id": urgent["task"]["id"]})
            self.assertTrue(completed["ok"])
            self.assertEqual(completed["task"]["status"], "done")

            active = task_manager.execute({"action": "list"})
            self.assertEqual(active["total"], 1)
            self.assertEqual(active["tasks"][0]["id"], low["task"]["id"])

    def test_update_normalizes_priority_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            task_manager,
            "TASKS_FILE",
            Path(temp_dir) / "tasks.json",
        ):
            created = task_manager.execute({"action": "create", "title": "Review tasks"})
            updated = task_manager.execute(
                {
                    "action": "update",
                    "task_id": created["task"]["id"],
                    "priority": "not-real",
                    "status": "blocked",
                }
            )

            self.assertTrue(updated["ok"])
            self.assertEqual(updated["task"]["priority"], "medium")
            self.assertEqual(updated["task"]["status"], "blocked")


class AgentIntegrationRegressionTests(unittest.TestCase):
    def test_forced_tool_selection_includes_new_tools(self) -> None:
        agent = Agent.__new__(Agent)
        tool_definitions = [{"name": "code_runner"}, {"name": "task_manager"}]

        self.assertEqual(
            Agent._forced_tool_names(agent, "run this code: print(2 + 2)", tool_definitions),
            {"code_runner"},
        )
        self.assertEqual(
            Agent._forced_tool_names(agent, "add task finish the regression suite", tool_definitions),
            {"task_manager"},
        )

    def test_final_answer_summarizes_new_tool_results(self) -> None:
        agent = Agent.__new__(Agent)

        code_summary = Agent._resolve_final_answer(
            agent,
            {"content": ""},
            [{"name": "code_runner", "result": {"ok": True, "stdout": "4", "stderr": "", "exit_code": 0}}],
        )
        task_summary = Agent._resolve_final_answer(
            agent,
            {"content": ""},
            [{"name": "task_manager", "result": {"ok": True, "task": {"title": "Ship tests", "status": "done"}}}],
        )

        self.assertEqual(code_summary, "Output: 4")
        self.assertEqual(task_summary, "Task 'Ship tests' is now done.")


class ToolRegistryRegressionTests(unittest.TestCase):
    def test_registry_discovers_new_tools(self) -> None:
        registry = ToolRegistry()
        names = set(registry.list_tool_names())

        self.assertIn("code_runner", names)
        self.assertIn("task_manager", names)


if __name__ == "__main__":
    unittest.main()
