"""Main ReAct loop for JARVIS."""

from __future__ import annotations

import json
from typing import Any, Callable

from config.settings import SETTINGS, Settings
from core.brain import Brain
from core.memory import Memory
from core.planner import Planner
from core.tool_registry import ToolRegistry


class Agent:
    """Plan, act, observe, and respond."""

    def __init__(
        self,
        settings: Settings | None = None,
        brain: Brain | None = None,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.brain = brain or Brain(self.settings)
        self.tools = tools or ToolRegistry(settings=self.settings)
        self.memory = memory or Memory(self.settings)
        self.planner = planner or Planner(self.brain)
        self.history: list[dict[str, Any]] = []

    def run(self, user_message: str, stream_handler: Callable[[str], None] | None = None) -> str:
        self.tools.refresh()
        memory_context = self._load_memory_context(user_message)
        all_tool_definitions = self.tools.get_tool_definitions()
        plan = self.planner.create_plan(user_message, all_tool_definitions)
        plan_note = self.planner.render_plan(plan)
        tool_definitions = self._select_tool_definitions(
            user_message,
            all_tool_definitions,
            plan,
            memory_context,
        )

        messages = self._build_messages(user_message, memory_context, plan_note)
        tool_trace: list[dict[str, Any]] = []
        task_kind = self._task_kind(user_message, plan)

        for _ in range(self.settings.agent_max_iterations):
            response = self.brain.chat(
                messages=messages,
                tools=tool_definitions,
                task_kind=task_kind,
                stream_handler=stream_handler if not tool_definitions else None,
            )
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                final_answer = response.get("content", "").strip() or "No response generated."
                return self._finalize_response(
                    user_message,
                    final_answer,
                    messages,
                    tool_trace,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                plan_step_index = self.planner.next_matching_step_index(plan, tool_call["name"])
                result = self.tools.run_tool(tool_call["name"], tool_call.get("arguments", {}))
                tool_trace.append(
                    {
                        "name": tool_call["name"],
                        "arguments": tool_call.get("arguments", {}),
                        "result": result,
                    }
                )
                self.planner.mark_step_completed(
                    plan,
                    tool_name=tool_call["name"],
                    step_index=plan_step_index,
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_call["name"],
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(result, ensure_ascii=True, default=str),
                    }
                )

        final_answer = "Stopped after reaching the tool iteration limit."
        return self._finalize_response(
            user_message,
            final_answer,
            messages,
            tool_trace,
        )

    def _build_messages(
        self,
        user_message: str,
        memory_context: list[str],
        plan_note: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if memory_context:
            memory_block = (
                "Relevant memory context:\n"
                + "\n".join(f"- {item}" for item in memory_context)
                + "\n\nUse this memory context when it helps answer the user. "
                + "Answer naturally and concisely instead of repeating the bullets verbatim."
            )
            messages.append({"role": "system", "content": memory_block})

        if plan_note and plan_note != "No explicit plan required.":
            messages.append({"role": "system", "content": plan_note})

        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": user_message})
        return messages

    def _load_memory_context(self, user_message: str) -> list[str]:
        if not self._should_query_memory(user_message):
            return []
        return self.memory.recall(user_message, self.settings.memory_top_k)

    def _task_kind(self, user_message: str, plan: dict[str, Any]) -> str:
        lowered = user_message.lower()
        code_markers = (
            "build a tool",
            "create a tool",
            "write code",
            "generate code",
            "build a python script",
            "create a python script",
            "python script",
            "script",
            "code",
            "coding",
            "program",
            "function",
            "class",
            "module",
            "api",
            "debug",
            "refactor",
            "implement",
            "fix this bug",
            "fix the bug",
            "traceback",
            "stack trace",
            "exception",
            "unit test",
            "test case",
        )
        if any(step.get("tool_name") == "code_writer" for step in plan.get("steps", [])):
            return "code"
        if any(marker in lowered for marker in code_markers):
            return "code"
        if any(extension in lowered for extension in (".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".html", ".css", ".sql", ".sh", ".ps1")):
            return "code"
        if plan.get("needs_planning"):
            return "complex"
        return "simple"

    def _should_query_memory(self, user_message: str) -> bool:
        lowered = user_message.lower()
        memory_markers = (
            "remember",
            "remind me",
            "do you remember",
            "can you remind me",
            "what did i tell you",
            "what do you know about",
            "what was i",
            "what was my",
            "what were we",
            "what was we",
            "what was i working on",
            "working on",
            "told you",
            "about my",
            "my ",
            "i like",
            "i am",
            "who am i",
            "what do i",
            "history",
            "context",
            "notes",
            "earlier",
            "project",
            "preference",
            "last time",
            "previous",
            "before",
        )
        return any(marker in lowered for marker in memory_markers)

    def _should_extract_memory(
        self,
        user_message: str,
        tool_trace: list[dict[str, Any]],
    ) -> bool:
        lowered = user_message.lower()
        tool_names = {str(item.get("name", "")) for item in tool_trace if item.get("name")}
        memory_query_only = bool(tool_names) and tool_names <= {"memory_query"}
        recall_markers = (
            "what do you know about me",
            "who am i",
            "what is my",
            "which school",
            "where do i",
            "what are my",
            "remember about me",
        )
        if memory_query_only or any(marker in lowered for marker in recall_markers):
            return False
        extract_markers = (
            "remember",
            "my ",
            "i like",
            "i am",
            "project",
            "working on",
            "build",
            "preference",
            "always",
            "never",
        )
        if any(marker in lowered for marker in extract_markers):
            return True
        return bool(tool_trace)

    def _select_tool_definitions(
        self,
        user_message: str,
        all_tool_definitions: list[dict[str, Any]],
        plan: dict[str, Any],
        memory_context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        requested_tools: set[str] = set()
        ranked_tools = ToolRegistry.rank_tool_definitions(user_message, all_tool_definitions)
        top_score = ranked_tools[0][1] if ranked_tools else 0.0
        max_selected = 1 if 0 < top_score <= 1.5 else 4
        for tool_definition, score in ranked_tools:
            if score < 1.0:
                break
            if top_score > 0 and score < max(1.0, top_score * 0.6):
                continue
            requested_tools.add(str(tool_definition["name"]))
            if len(requested_tools) >= max_selected:
                break

        for step in plan.get("steps", []):
            tool_name = step.get("tool_name")
            if tool_name:
                requested_tools.add(str(tool_name))

        if memory_context:
            requested_tools.discard("memory_query")

        if not requested_tools:
            return [] if memory_context else all_tool_definitions

        selected_tool_definitions = [
            tool_definition
            for tool_definition in all_tool_definitions
            if tool_definition["name"] in requested_tools
        ]
        if selected_tool_definitions:
            return selected_tool_definitions
        return [] if memory_context else all_tool_definitions

    def _remember_turn(self, user_message: str, assistant_message: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        self.history = self.history[-16:]

    def _finalize_response(
        self,
        user_message: str,
        assistant_message: str,
        messages: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
    ) -> str:
        self._remember_turn(user_message, assistant_message)
        self.memory.persist_conversation(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation=messages + [{"role": "assistant", "content": assistant_message}],
            tool_trace=tool_trace,
            brain=self.brain,
            should_extract=self._should_extract_memory(user_message, tool_trace),
            background=self.settings.background_memory_persistence,
        )
        return assistant_message
