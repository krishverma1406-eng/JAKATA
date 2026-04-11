"""Task classification and lightweight planning."""

from __future__ import annotations

import json
import re
from typing import Any

from core.brain import Brain
from core.intent_helpers import explicit_code_writer_request
from core.tool_registry import ToolRegistry


class Planner:
    """Decide when a request needs a multi-step plan and build one."""

    _SIMPLE_MESSAGES = {
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
    _MULTI_STEP_MARKERS = (
        " and ",
        " then ",
        " after ",
        " after that ",
        " also ",
        " while ",
        " once ",
        " when done ",
        " followed by ",
    )
    _STRONG_SEQUENCE_MARKERS = tuple(marker for marker in _MULTI_STEP_MARKERS if marker != " and ")
    _ACTION_MARKERS = {
        "open",
        "launch",
        "start",
        "play",
        "pause",
        "stop",
        "show",
        "list",
        "read",
        "write",
        "save",
        "find",
        "search",
        "take",
        "capture",
        "analyze",
        "describe",
        "click",
        "type",
        "scroll",
        "drag",
        "remember",
        "remind",
        "rename",
        "send",
        "create",
        "build",
        "fix",
        "debug",
        "summarize",
        "check",
        "review",
        "compare",
    }
    _DIRECT_SINGLE_ACTION_PREFIX = re.compile(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
        r"(?:open|launch|start|play|pause|stop|show|list|read|write|save|find|search|take|capture|analyze|describe|"
        r"click|type|scroll|drag|remember|remind|rename|send|create|build|fix|debug|summarize|check|review|compare)\b"
    )

    def __init__(self, brain: Brain | None = None) -> None:
        self.brain = brain or Brain()

    def should_plan(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> bool:
        if self._looks_simple(task):
            return False
        if self._is_single_action_request(task, tool_definitions or []):
            return False
        return self._looks_complex(task, tool_definitions or [])

    def create_plan(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.should_plan(task, tool_definitions):
            return {"needs_planning": False, "steps": []}

        planner_prompt = (
            "You are the high-level planning layer for a tool-using assistant.\n"
            "Decide the minimum plan needed for a genuinely multi-stage user request.\n"
            'Return strict JSON shaped as {"needs_planning": true, "strategy": "short summary", '
            '"steps": [{"step": "high-level phase", "tool_name": "primary tool or null", '
            '"tool_names": ["optional", "tool", "list"]}]}.\n'
            "Rules:\n"
            "- Only create plans for larger tasks with dependent stages, coordination, or multiple likely tools.\n"
            "- Use 2 to 5 high-level phases, not tiny actions like individual clicks, waits, or keystrokes.\n"
            "- A step may list multiple tools when the assistant may need to chain them.\n"
            "- Only reference tools from the provided tool list.\n"
            "- Use null or an empty list when a step is reasoning-only."
        )
        tool_list = ", ".join(tool["name"] for tool in tool_definitions) or "no tools available"
        response = self.brain.chat(
            messages=[
                {"role": "user", "content": f"Available tools: {tool_list}"},
                {"role": "user", "content": task},
            ],
            task_kind="simple",
            response_format="json",
            system_override=planner_prompt,
        )
        payload = self._parse_json(response.get("content", ""))
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            steps = []
            strategy = str(payload.get("strategy", "")).strip()
            for item in payload["steps"]:
                if not isinstance(item, dict):
                    continue
                step_text = str(item.get("step", "")).strip()
                if not step_text:
                    continue
                tool_names = self._normalize_tool_names(item.get("tool_names"), tool_definitions)
                primary_tool = str(item.get("tool_name", "") or "").strip() or None
                if primary_tool and primary_tool not in tool_names:
                    if primary_tool in {tool["name"] for tool in tool_definitions}:
                        tool_names = [primary_tool, *tool_names]
                steps.append(
                    {
                        "step": step_text,
                        "tool_name": tool_names[0] if tool_names else primary_tool,
                        "tool_names": tool_names,
                        "status": "pending",
                    }
                )
            if steps:
                return {"needs_planning": True, "strategy": strategy, "steps": steps}

        return {
            "needs_planning": True,
            "strategy": "Coordinate the larger task in phases and chain tools only where results are needed.",
            "steps": self._fallback_steps(task, tool_definitions),
        }

    def render_plan(self, plan: dict[str, Any]) -> str:
        steps = plan.get("steps", [])
        if not steps:
            return "No explicit plan required."

        lines = ["Execution plan:"]
        strategy = str(plan.get("strategy", "")).strip()
        if strategy:
            lines.append(f"Strategy: {strategy}")
        for index, step in enumerate(steps, start=1):
            tool_name = ", ".join(self._step_tool_names(step)) or "no tool specified"
            status = step.get("status", "pending")
            lines.append(f"{index}. {step.get('step', '').strip()} [{tool_name}] ({status})")
        return "\n".join(lines)

    def next_matching_step_index(self, plan: dict[str, Any], tool_name: str) -> int | None:
        for index, step in enumerate(plan.get("steps", [])):
            if step.get("status") != "pending":
                continue
            if tool_name in self._step_tool_names(step):
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

    def _looks_complex(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> bool:
        word_count = len(task.split())
        marker_hits = self._multi_step_marker_hits(task)
        strong_marker_hits = self._strong_sequence_marker_hits(task)
        action_count = self._estimated_action_count(task)
        likely_tool_count = len(self._likely_tool_names(task, tool_definitions))
        return (
            strong_marker_hits > 0
            or likely_tool_count >= 2
            or (
                marker_hits > 0
                and word_count >= self.brain.settings.planner_simple_word_limit
            )
            or (
                word_count >= self.brain.settings.planner_complex_word_limit
                and (action_count >= 2 or likely_tool_count >= 1)
            )
        )

    def _looks_simple(self, task: str) -> bool:
        lowered = task.lower().strip()
        word_count = len(lowered.split())
        return (
            lowered in self._SIMPLE_MESSAGES
            or word_count <= self.brain.settings.planner_simple_word_limit
        )

    def _is_single_action_request(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> bool:
        lowered = task.lower().strip()
        if not lowered:
            return True
        if self._strong_sequence_marker_hits(task) > 0:
            return False
        action_count = self._estimated_action_count(task)
        likely_tool_count = len(self._likely_tool_names(task, tool_definitions))
        if likely_tool_count <= 1 and action_count <= 1:
            return True
        return bool(self._DIRECT_SINGLE_ACTION_PREFIX.match(lowered) and action_count <= 1)

    def _fallback_steps(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fragments = re.split(r"\b(?:and then|then|after that|and|after)\b", task, flags=re.IGNORECASE)
        steps = []
        for fragment in fragments:
            cleaned = fragment.strip(" ,.")
            if not cleaned:
                continue
            guessed_tools = self._guess_tools(cleaned, tool_definitions)
            steps.append(
                {
                    "step": cleaned,
                    "tool_name": guessed_tools[0] if guessed_tools else None,
                    "tool_names": guessed_tools,
                    "status": "pending",
                }
            )
        if steps:
            return steps
        guessed_tools = self._guess_tools(task.strip(), tool_definitions)
        return [{
            "step": task.strip(),
            "tool_name": guessed_tools[0] if guessed_tools else None,
            "tool_names": guessed_tools,
            "status": "pending",
        }]

    def _guess_tools(self, fragment: str, tool_definitions: list[dict[str, Any]]) -> list[str]:
        if self._explicit_code_writer_request(fragment):
            allowed = {tool["name"] for tool in tool_definitions}
            return ["code_writer"] if "code_writer" in allowed else []
        ranked = ToolRegistry.rank_tool_definitions(fragment, tool_definitions)
        ranked = [(tool_definition, score) for tool_definition, score in ranked if tool_definition["name"] != "code_writer"]
        if not ranked:
            return []
        top_score = ranked[0][1]
        score_floor = max(1.0, top_score * 0.35)
        budget = 6 if len(str(fragment or "").split()) >= 8 else 4
        selected: list[str] = []
        for tool_definition, score in ranked:
            if score < score_floor:
                break
            selected.append(str(tool_definition["name"]))
            if len(selected) >= budget:
                break
        return selected

    def _explicit_code_writer_request(self, fragment: str) -> bool:
        return explicit_code_writer_request(fragment)

    def _likely_tool_names(
        self,
        task: str,
        tool_definitions: list[dict[str, Any]],
    ) -> list[str]:
        return self._guess_tools(task, tool_definitions)

    def _estimated_action_count(self, task: str) -> int:
        tokens = re.findall(r"[a-z0-9]+", task.lower())
        if not tokens:
            return 0
        action_hits = {token for token in tokens if token in self._ACTION_MARKERS}
        connector_hits = self._multi_step_marker_hits(task)
        return max(len(action_hits), connector_hits + 1 if connector_hits else len(action_hits))

    def _multi_step_marker_hits(self, task: str) -> int:
        return self._strong_sequence_marker_hits(task) + int(self._has_multi_action_and(task))

    def _strong_sequence_marker_hits(self, task: str) -> int:
        lowered = f" {task.lower()} "
        return sum(1 for marker in self._STRONG_SEQUENCE_MARKERS if marker in lowered)

    def _has_multi_action_and(self, task: str) -> bool:
        lowered = f" {task.lower()} "
        if " and " not in lowered:
            return False
        parts = [part.strip() for part in lowered.split(" and ") if part.strip()]
        return len(parts) >= 2 and all(self._fragment_has_action_marker(part) for part in parts)

    def _fragment_has_action_marker(self, fragment: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", fragment.lower())
        return any(token in self._ACTION_MARKERS for token in tokens)

    def _normalize_tool_names(
        self,
        raw_tool_names: Any,
        tool_definitions: list[dict[str, Any]],
    ) -> list[str]:
        allowed = {tool["name"] for tool in tool_definitions}
        if isinstance(raw_tool_names, str):
            raw_tool_names = [raw_tool_names]
        if not isinstance(raw_tool_names, list):
            return []

        normalized: list[str] = []
        for item in raw_tool_names:
            name = str(item or "").strip()
            if not name or name not in allowed or name in normalized:
                continue
            normalized.append(name)
        return normalized

    def _step_tool_names(self, step: dict[str, Any]) -> list[str]:
        seen: list[str] = []
        tool_name = str(step.get("tool_name", "") or "").strip()
        if tool_name:
            seen.append(tool_name)
        tool_names = step.get("tool_names", [])
        if isinstance(tool_names, list):
            for item in tool_names:
                value = str(item or "").strip()
                if value and value not in seen:
                    seen.append(value)
        return seen

    def _parse_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
