"""Task classification and lightweight planning."""

from __future__ import annotations

import json
import re
from typing import Any

from core.brain import Brain
from core.tool_registry import ToolRegistry


class Planner:
    """Decide when a request needs a multi-step plan and build one."""

    def __init__(self, brain: Brain | None = None) -> None:
        self.brain = brain or Brain()

    def should_plan(self, task: str) -> bool:
        if self._looks_simple(task):
            return False
        return self._looks_complex(task)

    def create_plan(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.should_plan(task):
            return {"needs_planning": False, "steps": []}

        planner_prompt = (
            "Break the user request into ordered execution steps.\n"
            'Return strict JSON shaped as {"steps": [{"step": "short description", '
            '"tool_name": "tool or null"}]}.\n'
            "Only reference tools from the provided tool list. Use null when no tool is necessary."
        )
        tool_list = ", ".join(tool["name"] for tool in tool_definitions) or "no tools available"
        response = self.brain.chat(
            messages=[
                {"role": "user", "content": f"Available tools: {tool_list}"},
                {"role": "user", "content": task},
            ],
            task_kind="planning",
            response_format="json",
            system_override=planner_prompt,
        )
        payload = self._parse_json(response.get("content", ""))
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            steps = []
            for item in payload["steps"]:
                if not isinstance(item, dict):
                    continue
                step_text = str(item.get("step", "")).strip()
                if not step_text:
                    continue
                steps.append(
                    {
                        "step": step_text,
                        "tool_name": item.get("tool_name"),
                        "status": "pending",
                    }
                )
            if steps:
                return {"needs_planning": True, "steps": steps}

        return {"needs_planning": True, "steps": self._fallback_steps(task, tool_definitions)}

    def render_plan(self, plan: dict[str, Any]) -> str:
        steps = plan.get("steps", [])
        if not steps:
            return "No explicit plan required."

        lines = ["Execution plan:"]
        for index, step in enumerate(steps, start=1):
            tool_name = step.get("tool_name") or "no tool specified"
            status = step.get("status", "pending")
            lines.append(f"{index}. {step.get('step', '').strip()} [{tool_name}] ({status})")
        return "\n".join(lines)

    def next_matching_step_index(self, plan: dict[str, Any], tool_name: str) -> int | None:
        for index, step in enumerate(plan.get("steps", [])):
            if step.get("status") != "pending":
                continue
            if step.get("tool_name") == tool_name:
                return index
        return None

    def mark_step_completed(
        self,
        plan: dict[str, Any],
        tool_name: str | None = None,
        step_index: int | None = None,
    ) -> None:
        steps = plan.get("steps", [])
        if step_index is not None and 0 <= step_index < len(steps):
            if steps[step_index].get("status") == "pending":
                steps[step_index]["status"] = "completed"
                return

        if tool_name is None:
            return

        fallback_index = self.next_matching_step_index(plan, tool_name)
        if fallback_index is not None:
            steps[fallback_index]["status"] = "completed"

    def _looks_complex(self, task: str) -> bool:
        lowered = task.lower()
        multi_step_markers = (
            " and ",
            " then ",
            " after ",
            " before ",
            " remind ",
            " also ",
            " while ",
        )
        return (
            len(task.split()) >= self.brain.settings.planner_complex_word_limit
            or any(marker in lowered for marker in multi_step_markers)
        )

    def _looks_simple(self, task: str) -> bool:
        lowered = task.lower().strip()
        word_count = len(lowered.split())
        simple_markers = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "yo",
            "sup",
            "ok",
            "okay",
        }
        return (
            lowered in simple_markers
            or word_count <= self.brain.settings.planner_simple_word_limit
        )

    def _fallback_steps(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fragments = re.split(r"\b(?:and then|then|after that|and|after|before)\b", task, flags=re.IGNORECASE)
        steps = []
        for fragment in fragments:
            cleaned = fragment.strip(" ,.")
            if not cleaned:
                continue
            steps.append(
                {
                    "step": cleaned,
                    "tool_name": self._guess_tool(cleaned, tool_definitions),
                    "status": "pending",
                }
            )
        return steps or [{"step": task.strip(), "tool_name": None, "status": "pending"}]

    def _guess_tool(self, fragment: str, tool_definitions: list[dict[str, Any]]) -> str | None:
        available_tool_names = {tool["name"] for tool in tool_definitions}
        best_name = ToolRegistry.best_matching_tool_name(fragment, tool_definitions, min_score=1.5)
        if best_name in available_tool_names:
            return best_name
        return None

    def _parse_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
