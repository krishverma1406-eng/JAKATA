"""Main ReAct loop for JARVIS."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable

from config.settings import SETTINGS, Settings
from core.brain import Brain
from core.interface_config import INTERFACE_CONFIG
from core.intent_helpers import explicit_code_writer_request
from core.memory import Memory
from core.planner import Planner
from core.tool_registry import ToolRegistry


TOOL_FALLBACK_CHAINS: dict[str, list[str]] = {
    "browser_control": ["web_search"],
    "weather_tool": ["web_search"],
    "music_player": ["browser_control"],
    "screenshot_tool": ["os_control"],
    "reminder_tool": ["notes_tool"],
    "app_launcher_tool": ["terminal_tool"],
    "file_manager": ["terminal_tool"],
    "gmail_tool": [],
    "calendar_tool": [],
}

# Fallback hints for tool failures
_FALLBACK_HINTS: dict[str, str] = {
    "web_search": (
        "Web search failed. Report which provider failed if the error names Tavily or Brave, "
        "suggest retrying later, and do not claim that live web results were retrieved."
    ),
    "browser_control": (
        "Nova Act browser failed. If the user needs web content, "
        "try web_search instead. If they need to open an app, try app_launcher_tool."
    ),
    "gmail_tool": (
        "Gmail tool failed. Do NOT retry with gmail_tool. Tell the user what failed "
        "and that they may need to run OAuth setup if not already configured."
    ),
    "calendar_tool": (
        "Calendar tool failed. Do NOT retry with calendar_tool. Tell the user what failed "
        "and that they may need to run OAuth setup if not already configured."
    ),
    "music_player": (
        "VLC music player failed. If the user wants YouTube playback, "
        "use browser_control to open the YouTube URL instead."
    ),
    "screenshot_tool": (
        "Screenshot tool failed. Try os_control with action=screenshot as fallback."
    ),
    "weather_tool": (
        "OpenWeatherMap failed. Try web_search with query "
        "'current weather [location]' as fallback."
    ),
    "reminder_tool": (
        "Reminder storage failed. Create a note with notes_tool as a fallback "
        "and tell the user it's stored as a note, not a timed reminder."
    ),
    "app_launcher_tool": (
        "App launcher failed. Try terminal_tool with action=start_process "
        "or action=run with Start-Process command as fallback."
    ),
    "file_manager": (
        "File manager failed. Try terminal_tool with Get-Content or "
        "Get-ChildItem as fallback for read/list operations."
    ),
    "os_control": (
        "Desktop control failed. Report the exact desktop automation error. "
        "If pyautogui or a desktop dependency is missing, stop and tell the user that setup is required."
    ),
    "terminal_tool": (
        "Terminal fallback failed. Report the exact exit code and stderr. "
        "Do not pretend the command worked."
    ),
    "calculator_tool": (
        "Calculator tool failed. If the math is simple and safe to do directly, "
        "compute it inline and say that the answer was not tool-verified."
    ),
    "vision_tool": (
        "Vision analysis failed. If there is no attached live frame, tell the user to enable the camera panel or send an image. "
        "If the vision provider failed, report that clearly and do not invent what is visible."
    ),
}

_FALLBACK_BLOCKING_SIGNALS = (
    "api key",
    "oauth",
    "not configured",
    "client_secret",
    "client secret",
    "token",
    "credential",
    "401",
    "403",
    "unauthorized",
    "forbidden",
)


def _tool_failed(result: dict[str, Any]) -> bool:
    inner = result.get("result", result)
    return isinstance(inner, dict) and inner.get("ok") is False


def _get_fallback_hint(tool_name: str, error: str) -> str:
    """Get fallback hint for a failed tool based on error type."""
    # Auth/config errors — never retry, never fallback
    error_lower = error.lower()
    if any(signal in error_lower for signal in _FALLBACK_BLOCKING_SIGNALS):
        return (
            f"{tool_name} requires configuration that is not set up. "
            "Tell the user exactly what's missing. Do not attempt any fallback."
        )
    return _FALLBACK_HINTS.get(tool_name, "")


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
        self.interface = INTERFACE_CONFIG
        self.session_id = ""
        self.mode = self.interface.default_mode()
        self.session_meta: dict[str, Any] = {}
        self.last_turn_meta: dict[str, Any] = {}
        self.last_tool_trace: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self._startup_messages_emitted = False
        self.bind_session()

    def bind_session(self, session_id: str | None = None, mode: str | None = None) -> str:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.mode = self.interface.normalize_mode(mode or self.mode)
        self.session_meta = self.memory.ensure_session(self.session_id, mode=self.mode)
        self.mode = self.interface.normalize_mode(self.session_meta.get("mode", self.mode))
        self.history = self.memory.load_session_messages(self.session_id, limit_messages=12)
        self._startup_messages_emitted = False
        return self.session_id

    def set_mode(self, mode: str) -> dict[str, Any]:
        normalized = self.interface.normalize_mode(mode)
        previous_mode = self.mode
        self.mode = normalized
        if self.session_id:
            self.session_meta = self.memory.set_session_mode(self.session_id, normalized)
        else:
            self.bind_session(mode=normalized)
        self.mode = self.interface.normalize_mode(self.session_meta.get("mode", normalized))
        if self.mode != previous_mode:
            self._startup_messages_emitted = False
        return dict(self.session_meta)

    def rename_session(self, new_name: str) -> dict[str, Any]:
        if not self.session_id:
            self.bind_session()
        self.session_meta = self.memory.rename_session(self.session_id, new_name)
        return dict(self.session_meta)

    def startup_messages(self) -> list[str]:
        if self._startup_messages_emitted:
            return []
        mode_config = self.interface.get_mode(self.mode)
        messages: list[str] = []
        if mode_config.get("briefing_on_start") and not self.history:
            briefing = self._build_briefing()
            if briefing:
                messages.append(briefing)
        self._startup_messages_emitted = True
        return messages

    def run(
        self,
        user_message: str,
        stream_handler: Callable[[str], None] | None = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        mode_config = self.interface.get_mode(self.mode)
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
            runtime_context=runtime_context,
        )

        if event_handler is not None and mode_config.get("show_debug"):
            if memory_context:
                event_handler(
                    {
                        "type": "activity",
                        "payload": {
                            "event": "memory_context_loaded",
                            "message": "; ".join(memory_context[:4]),
                        },
                    }
                )
            if plan_note and plan_note != "No explicit plan required.":
                event_handler(
                    {
                        "type": "activity",
                        "payload": {
                            "event": "plan_ready",
                            "message": plan_note,
                        },
                    }
                )

        focus_redirect = self._focus_redirect(user_message, mode_config)
        if focus_redirect:
            if event_handler is not None:
                event_handler(
                    {
                        "type": "activity",
                        "payload": {
                            "event": "focus_redirected",
                            "message": focus_redirect,
                        },
                    }
                )
            return self._finalize_response(
                user_message,
                focus_redirect,
                [],
                [],
                turn_meta={"mode": self.mode, "tool_count": 0},
            )

        messages = self._build_messages(user_message, memory_context, plan, plan_note, mode_config)
        tool_trace: list[dict[str, Any]] = []
        task_kind = self._task_kind(user_message, plan)
        total_started_at = time.perf_counter()
        last_response_meta: dict[str, Any] = {}

        for _ in range(self.settings.agent_max_iterations):
            # Always stream — the model won't stream tool-call turns anyway
            active_stream_handler = stream_handler
            brain_started_at = time.perf_counter()
            response = self.brain.chat(
                messages=messages,
                tools=tool_definitions,
                task_kind=task_kind,
                stream_handler=active_stream_handler,
            )
            response_latency_ms = int((time.perf_counter() - brain_started_at) * 1000)
            tool_calls = response.get("tool_calls", [])
            last_response_meta = {
                "provider": response.get("provider", ""),
                "model": response.get("model", ""),
                "latency_ms": response_latency_ms,
            }

            if event_handler is not None and mode_config.get("show_debug"):
                event_handler(
                    {
                        "type": "activity",
                        "payload": {
                            "event": "provider_selected",
                            "message": (
                                f"{response.get('provider', 'unknown')} | "
                                f"{response.get('model', 'unknown')} | {response_latency_ms} ms"
                            ),
                        },
                    }
                )

            if tool_calls and event_handler is not None and stream_handler is not None:
                event_handler({"type": "stream_cancelled"})

            if not tool_calls:
                final_answer = self._resolve_final_answer(response, tool_trace)
                return self._finalize_response(
                    user_message,
                    final_answer,
                    messages,
                    tool_trace,
                    turn_meta={
                        "mode": self.mode,
                        "provider": response.get("provider", ""),
                        "model": response.get("model", ""),
                        "latency_ms": response_latency_ms,
                        "total_latency_ms": int((time.perf_counter() - total_started_at) * 1000),
                        "tool_count": len(tool_trace),
                        "plan_note": plan_note if plan_note != "No explicit plan required." else "",
                    },
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("arguments", {})
                plan_step_index = self.planner.next_matching_step_index(plan, tool_name)
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "tool_started",
                            "name": tool_name,
                            "arguments": tool_args,
                        }
                    )
                result = self._execute_with_chain(
                    tool_name=tool_name,
                    arguments=tool_args,
                    runtime_context={
                        "session_id": self.session_id,
                        "session_mode": self.mode,
                        "session_name": str(self.session_meta.get("display_name", "")).strip(),
                        **(runtime_context or {}),
                    },
                )
                tool_trace.append(
                    {
                        "name": tool_name,
                        "arguments": tool_args,
                        "result": result,
                        "fallback_used": bool(isinstance(result, dict) and result.get("_fallback_used", False)),
                    }
                )
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "tool_result",
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": result,
                        }
                    )
                self.planner.mark_step_completed(
                    plan,
                    tool_name=tool_name,
                    step_index=plan_step_index,
                )

                tool_content = json.dumps(result, ensure_ascii=True, default=str)
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call.get("id"),
                        "content": tool_content,
                    }
                )

        final_answer = "Stopped after reaching the tool iteration limit."
        return self._finalize_response(
            user_message,
            final_answer,
            messages,
            tool_trace,
            turn_meta={
                "mode": self.mode,
                "provider": last_response_meta.get("provider", ""),
                "model": last_response_meta.get("model", ""),
                "latency_ms": last_response_meta.get("latency_ms"),
                "total_latency_ms": int((time.perf_counter() - total_started_at) * 1000),
                "tool_count": len(tool_trace),
                "plan_note": plan_note if plan_note != "No explicit plan required." else "",
            },
        )

    def _build_messages(
        self,
        user_message: str,
        memory_context: list[str],
        plan: dict[str, Any],
        plan_note: str,
        mode_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        
        # Add session awareness — always show turn count, even on first turn
        turn_count = len([m for m in self.history if m.get("role") == "user"])
        session_name = self.session_meta.get("display_name", "Untitled")
        messages.append(
            {
                "role": "system",
                "content": f"This is turn {turn_count + 1} of the current session '{session_name}'.",
            }
        )
        
        daily_summary = self.memory.get_daily_context_summary(self.brain)

        mode_prompt = str(mode_config.get("system_prompt", "")).strip()
        if mode_prompt:
            messages.append({"role": "system", "content": mode_prompt})

        if daily_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Daily user context summary:\n"
                        f"{daily_summary}\n\n"
                        "Use this as lightweight internal context, not as something to quote directly."
                    ),
                }
            )

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
        if plan.get("needs_planning"):
            messages.append({"role": "system", "content": self._plan_execution_guidance()})

        messages.extend(self.history[-12:])
        messages.append({"role": "user", "content": user_message})
        return messages

    def _plan_execution_guidance(self) -> str:
        return (
            "Use the plan as high-level execution guidance only.\n"
            "Do not create micro-steps or call tools just because they appear in the plan.\n"
            "Choose the next necessary tool based on the latest tool result, and chain tools when one output unlocks the next action.\n"
            "Prefer inspection before action when state is uncertain, especially for browser, desktop, file, and screenshot-driven tasks.\n"
            "Push the task to a completed outcome when a reasonable next tool step can finish it.\n"
            "When the user expresses doubt or asks for confirmation, re-check with the relevant tool instead of answering from vibes.\n"
            "Never expose raw tool JSON or internal traces to the user; summarize the verified result naturally."
        )

    def _load_memory_context(self, user_message: str) -> list[str]:
        if not self._should_query_memory(user_message):
            return []
        return self.memory.recall(user_message, self.settings.memory_top_k)

    def _focus_redirect(self, user_message: str, mode_config: dict[str, Any]) -> str:
        focus_filter = mode_config.get("focus_filter", {})
        if not isinstance(focus_filter, dict) or not focus_filter.get("enabled"):
            return ""

        lowered = user_message.lower().strip()
        if not lowered:
            return ""

        task_markers = [str(item).strip().lower() for item in focus_filter.get("task_markers", []) if str(item).strip()]
        small_talk_markers = [str(item).strip().lower() for item in focus_filter.get("small_talk_markers", []) if str(item).strip()]
        task_hits = sum(1 for marker in task_markers if marker and marker in lowered)
        small_talk_hits = sum(1 for marker in small_talk_markers if marker and marker in lowered)
        if small_talk_hits <= 0:
            return ""
        if task_hits >= small_talk_hits:
            return ""
        return str(focus_filter.get("redirect_message", "")).strip()

    def _build_briefing(self) -> str:
        config = self.interface.briefing()
        intro = str(config.get("intro", "Here is the current session snapshot.")).strip()
        max_items = max(1, int(config.get("max_items_per_section", 5) or 5))

        project_items = self.memory.active_project_items(limit=max_items)
        reminder_lines: list[str] = []
        try:
            from services.reminders import get_reminder_service

            reminders = get_reminder_service(self.settings).list_reminders(include_completed=False)
            for reminder in reminders[:max_items]:
                if not isinstance(reminder, dict):
                    continue
                text = str(reminder.get("text", "")).strip()
                due_at = str(reminder.get("due_at_local") or reminder.get("due_at", "")).strip()
                if text:
                    reminder_lines.append(f"- {text}" + (f" | due {due_at}" if due_at else ""))
        except Exception:
            reminder_lines = []
        calendar_lines: list[str] = []
        try:
            from services.calendar_service import get_calendar_service

            events = get_calendar_service().today_events(max_results=max_items)
            for event in events[:max_items]:
                if not isinstance(event, dict):
                    continue
                summary = str(event.get("summary", "")).strip() or "(untitled event)"
                start = str(event.get("start", "")).strip()
                location = str(event.get("location", "")).strip()
                line = f"- {summary}"
                if start:
                    line += f" | {start}"
                if location:
                    line += f" | {location}"
                calendar_lines.append(line)
        except Exception:
            calendar_lines = []

        raw_data = {
            "projects": project_items[:max_items],
            "reminders": reminder_lines[:max_items],
            "calendar": calendar_lines[:max_items],
            "time": datetime.now().strftime("%A, %I:%M %p"),
        }

        if not raw_data["projects"] and not raw_data["reminders"] and not raw_data["calendar"]:
            return intro

        prompt = (
            "You are JARVIS giving Krish an operational briefing.\n"
            "Synthesize this into 2-4 short punchy sentences.\n"
            "Lead with what needs attention today.\n"
            "Be specific, crisp, and direct.\n"
            "Do not invent project names, deadlines, or events.\n"
            "Data:\n"
            + json.dumps(raw_data, ensure_ascii=False, indent=2)
        )
        try:
            response = self.brain.chat(
                messages=[{"role": "user", "content": prompt}],
                task_kind="simple",
                system_override="You are JARVIS. Briefing only. No filler.",
            )
            content = str(response.get("content", "")).strip()
            if content:
                return content
        except Exception:
            pass

        fallback_sections: list[str] = [intro]
        if project_items:
            fallback_sections.append("Active projects:\n" + "\n".join(f"- {item}" for item in project_items[:max_items]))
        if reminder_lines:
            fallback_sections.append("Pending reminders:\n" + "\n".join(reminder_lines[:max_items]))
        if calendar_lines:
            fallback_sections.append("Today's calendar:\n" + "\n".join(calendar_lines[:max_items]))
        return "\n\n".join(part for part in fallback_sections if part.strip()).strip()

    def _task_kind(self, user_message: str, plan: dict[str, Any]) -> str:
        lowered = user_message.lower()
        code_phrases = (
            "build a tool",
            "create a tool",
            "write code",
            "generate code",
            "build a python script",
            "create a python script",
            "python script",
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
        code_terms = (
            "script",
            "code",
            "coding",
            "program",
            "function",
            "class",
            "module",
            "api",
        )
        code_adjacent = {
            "write",
            "create",
            "build",
            "fix",
            "debug",
            "implement",
            "generate",
            "refactor",
            "test",
            "compile",
            "run",
        }
        lowered_tokens = set(re.findall(r"[a-z0-9_+.:-]+", lowered))
        if any(step.get("tool_name") == "code_writer" for step in plan.get("steps", [])):
            return "code"
        if any(marker in lowered for marker in code_phrases):
            return "code"
        if any(marker in lowered_tokens for marker in code_terms) and bool(lowered_tokens & code_adjacent):
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
            "remember that",
            "forget that",
            "forget what",
            "my ",
            "i like",
            "i am",
            "who am i",
            "who is ",
            "what do i",
            "history",
            "chat so far",
            "conversation so far",
            "our chat",
            "recent chat",
            "recent conversation",
            "what have we done",
            "what we have done",
            "what did we talk",
            "what we talked",
            "what did we build",
            "what was i doing",
            "what have i",
            "recently",
            "context",
            "notes",
            "earlier",
            "project",
            "preference",
            "last time",
            "last session",
            "previous",
            "previously",
            "yesterday",
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
        if "?" in lowered:
            return False
        command_only_pattern = re.compile(
            r"^(open|search|play|list|show|find|tell me|what is|how to)\s+"
            r"(spotify|youtube|chrome|vscode|notepad|calculator|terminal)[\s.!?]*$",
            re.IGNORECASE,
        )
        if command_only_pattern.match(user_message.strip()):
            return False
        extract_markers = (
            "remember ",
            "my name is ",
            "i am ",
            "i'm ",
            "i live ",
            "i study ",
            "my school ",
            "i like ",
            "i prefer ",
            "i want ",
            "i'm working on ",
            "i am working on ",
            "my project ",
        )
        if any(marker in lowered for marker in extract_markers):
            return True
        if "project" in lowered and any(marker in lowered for marker in ("my ", "i am ", "i'm ", "working on ")):
            return True
        return False

    def _select_tool_definitions(
        self,
        user_message: str,
        all_tool_definitions: list[dict[str, Any]],
        plan: dict[str, Any],
        memory_context: list[str] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        lowered = user_message.lower().strip()
        simple_chat_markers = {
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "thanks",
            "thank you",
            "ok",
            "okay",
        }
        if self._should_answer_code_directly(user_message):
            return []
        if self._explicit_code_writer_request(user_message):
            return [
                tool_definition
                for tool_definition in all_tool_definitions
                if tool_definition["name"] == "code_writer"
            ]
        if lowered in simple_chat_markers:
            return []

        requested_tool_names: list[str] = []
        requested_tool_set: set[str] = set()

        def add_tool(name: str) -> None:
            cleaned = str(name or "").strip()
            if cleaned and cleaned not in requested_tool_set:
                requested_tool_set.add(cleaned)
                requested_tool_names.append(cleaned)

        forced_tools = self._forced_tool_names(user_message, all_tool_definitions, runtime_context=runtime_context)
        ranked_tools = ToolRegistry.rank_tool_definitions(user_message, all_tool_definitions)
        top_score = ranked_tools[0][1] if ranked_tools else 0.0
        capability_budget = self._tool_selection_budget(user_message, plan, forced_tools)
        score_floor = max(1.0, top_score * (0.35 if plan.get("needs_planning") else 0.45)) if top_score else 1.0

        for name in forced_tools:
            add_tool(name)

        for step in plan.get("steps", []):
            tool_name = step.get("tool_name")
            if tool_name:
                add_tool(str(tool_name))
            for extra_tool_name in step.get("tool_names", []):
                if extra_tool_name:
                    add_tool(str(extra_tool_name))

        for tool_definition, score in ranked_tools:
            tool_name = str(tool_definition["name"])
            if tool_name == "code_writer" and not self._explicit_code_writer_request(user_message):
                continue
            if score < score_floor and tool_name not in requested_tool_set:
                break
            if score < score_floor and tool_name not in forced_tools:
                continue
            add_tool(tool_name)
            if len(requested_tool_names) >= capability_budget:
                break

        if not self._explicit_code_writer_request(user_message):
            requested_tool_set.discard("code_writer")
            requested_tool_names = [name for name in requested_tool_names if name != "code_writer"]

        requested_tool_names = [
            name
            for name in requested_tool_names
            if self._tool_is_compatible(name, lowered, runtime_context=runtime_context, forced_tools=forced_tools)
        ]
        requested_tool_set = set(requested_tool_names)

        if memory_context and "memory_query" not in forced_tools:
            requested_tool_set.discard("memory_query")
            requested_tool_names = [name for name in requested_tool_names if name != "memory_query"]

        if "vision_tool" in forced_tools and not any(
            marker in lowered for marker in ("browser", "website", "web page", "youtube", "http://", "https://", ".com")
        ):
            requested_tool_set.discard("browser_control")
            requested_tool_names = [name for name in requested_tool_names if name != "browser_control"]

        if "browser_control" in requested_tool_set and any(
            marker in lowered for marker in ("youtube", "youtu.be", "spotify web", "soundcloud")
        ):
            requested_tool_set.discard("music_player")
            requested_tool_names = [name for name in requested_tool_names if name != "music_player"]

        if not requested_tool_names:
            return []

        selected_tool_definitions = [
            tool_definition
            for tool_definition in all_tool_definitions
            if tool_definition["name"] in requested_tool_set
        ]
        if selected_tool_definitions:
            definition_map = {tool_definition["name"]: tool_definition for tool_definition in selected_tool_definitions}
            return [definition_map[name] for name in requested_tool_names if name in definition_map]
        return []

    def _tool_selection_budget(
        self,
        user_message: str,
        plan: dict[str, Any],
        forced_tools: set[str],
    ) -> int:
        lowered = user_message.lower().strip()
        word_count = len(lowered.split())
        if forced_tools:
            return max(6, len(forced_tools) + 3)
        if plan.get("needs_planning"):
            return 8
        return min(4 + (word_count // 5), 7)

    def _tool_is_compatible(
        self,
        tool_name: str,
        lowered: str,
        runtime_context: dict[str, Any] | None = None,
        forced_tools: set[str] | None = None,
    ) -> bool:
        forced_tools = forced_tools or set()
        if tool_name in forced_tools:
            return True

        has_live_image = bool(isinstance(runtime_context, dict) and str(runtime_context.get("imgbase64", "")).strip())

        capability_markers: dict[str, tuple[str, ...]] = {
            "vision_tool": (
                "camera", "image", "photo", "picture", "screenshot", "screen", "look at", "see", "visible",
                "holding", "hand", "analyze", "describe",
            ),
            "memory_query": (
                "remember", "memory", "about me", "who am i", "my school", "school name", "where do i live",
                "what do you know", "last session", "previously", "yesterday",
            ),
            "notes_tool": (
                "save a note", "write a note", "briefing note", "draft a note", "save note", "take a note",
            ),
            "reminder_tool": (
                "remind me", "reminder", "remind", "at 6", "at 7", "tomorrow at", "today at",
            ),
            "calendar_tool": (
                "calendar", "event", "events", "schedule", "meeting", "upcoming",
            ),
            "gmail_tool": (
                "email", "gmail", "inbox", "send mail", "mail",
            ),
            "weather_tool": (
                "weather", "forecast", "temperature", "rain", "umbrella",
            ),
            "browser_control": (
                "browser", "website", "web page", "youtube", "spotify web", "soundcloud", "open ", "search ",
                "click", "go to", "http://", "https://", ".com",
            ),
            "web_search": (
                "latest", "current", "recent", "news", "release notes", "search web", "look up", "find online",
            ),
            "file_manager": (
                "file", "files", "folder", "path", "downloads", "download", "pdf", "docx", "txt",
                "read file", "find file", "open file", "newest pdf",
            ),
            "app_launcher_tool": (
                "launch", "open spotify", "open vscode", "open vs code", "open app", "start app",
            ),
            "screenshot_tool": (
                "screenshot", "screen", "on screen", "what is on screen",
            ),
            "os_control": (
                "click", "desktop", "drag", "hotkey", "type on screen",
            ),
            "clipboard_tool": (
                "clipboard", "copy this", "paste this",
            ),
            "session_tool": (
                "session", "rename this session", "recent sessions", "past session", "current session",
            ),
            "music_player": (
                "local music", "play local", "mp3", "song file",
            ),
        }

        markers = capability_markers.get(tool_name)
        if tool_name == "vision_tool":
            return has_live_image and bool(markers and any(marker in lowered for marker in markers))
        if tool_name == "notes_tool":
            return "note" in lowered and any(verb in lowered for verb in ("save", "write", "draft", "take", "briefing"))
        if markers is None:
            return True
        return any(marker in lowered for marker in markers)

    def _merge_tool_definitions(
        self,
        existing: list[dict[str, Any]],
        additions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tool_definition in [*existing, *additions]:
            name = str(tool_definition.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(tool_definition)
        return merged

    def _tool_error_blocks_fallback(self, error: str) -> bool:
        lowered = str(error or "").lower()
        return any(signal in lowered for signal in _FALLBACK_BLOCKING_SIGNALS)

    def _execute_with_chain(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a tool, then automatically try fallbacks before returning to the model."""
        result = self.tools.run_tool(tool_name, arguments, runtime_context)
        if isinstance(result, dict) and result.get("ok") is True:
            return result

        error_msg = str(result.get("error", "")).lower() if isinstance(result, dict) else ""
        if self._tool_error_blocks_fallback(error_msg):
            return result

        fallbacks = TOOL_FALLBACK_CHAINS.get(tool_name, [])
        for fallback_name in fallbacks:
            if not self.tools.has_tool(fallback_name):
                continue
            fallback_args = self._adapt_args_for_fallback(tool_name, fallback_name, arguments)
            fallback_result = self.tools.run_tool(fallback_name, fallback_args, runtime_context)
            if isinstance(fallback_result, dict) and fallback_result.get("ok") is True:
                fallback_result["_fallback_used"] = True
                fallback_result["_original_tool"] = tool_name
                fallback_result["_original_error"] = str(result.get("error", "")) if isinstance(result, dict) else ""
                fallback_result["_fallback_tool"] = fallback_name
                return fallback_result

        if isinstance(result, dict):
            result["_fallbacks_tried"] = list(fallbacks)
            hint = _get_fallback_hint(tool_name, error_msg)
            if hint:
                result["_fallback_instruction"] = hint
        return result

    def _adapt_args_for_fallback(
        self,
        from_tool: str,
        to_tool: str,
        original_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate arguments from one tool schema to another."""
        if from_tool == "browser_control" and to_tool == "web_search":
            url = str(original_args.get("url", "")).strip()
            query = str(original_args.get("task", "") or original_args.get("prompt", "") or url).strip()
            return {"query": query or url, "max_results": 5}

        if from_tool == "weather_tool" and to_tool == "web_search":
            location = str(original_args.get("location", "")).strip()
            action = str(original_args.get("action", "current")).strip()
            query = f"{'forecast' if action == 'forecast' else 'current weather'} {location}".strip()
            return {"query": query, "max_results": 3}

        if from_tool == "screenshot_tool" and to_tool == "os_control":
            return {"action": "screenshot"}

        if from_tool == "app_launcher_tool" and to_tool == "terminal_tool":
            target = str(original_args.get("target") or original_args.get("app_name") or "").strip()
            return {
                "action": "run",
                "shell": "powershell",
                "command": f"Start-Process '{target}'",
            }

        if from_tool == "file_manager" and to_tool == "terminal_tool":
            action = str(original_args.get("action", "list")).strip()
            path = str(original_args.get("path", ".")).strip()
            pattern = str(original_args.get("pattern", "")).strip()
            commands = {
                "list": f"Get-ChildItem '{path}' | Select-Object Name, Length, LastWriteTime",
                "read": f"Get-Content '{path}'",
                "find": f"Get-ChildItem '{path}' -Recurse -Filter '*{pattern}*'",
            }
            return {
                "action": "run",
                "shell": "powershell",
                "command": commands.get(action, f"Get-ChildItem '{path}'"),
            }

        if from_tool == "reminder_tool" and to_tool == "notes_tool":
            text = str(original_args.get("text", "")).strip()
            due = str(original_args.get("due_at", "")).strip()
            return {
                "action": "create",
                "title": f"reminder-{datetime.now().strftime('%Y%m%d-%H%M')}",
                "content": f"# Reminder\n{text}\n\nDue: {due or 'ASAP'}",
            }

        if from_tool == "music_player" and to_tool == "browser_control":
            track = str(original_args.get("track", "")).strip()
            return {
                "action": "open",
                "url": f"https://www.youtube.com/results?search_query={track.replace(' ', '+')}",
            }

        return dict(original_args or {})

    def _forced_tool_names(
        self,
        user_message: str,
        all_tool_definitions: list[dict[str, Any]],
        runtime_context: dict[str, Any] | None = None,
    ) -> set[str]:
        lowered = user_message.lower().strip()
        available = {str(tool_definition["name"]) for tool_definition in all_tool_definitions}
        forced: set[str] = set()
        has_live_image = bool(isinstance(runtime_context, dict) and str(runtime_context.get("imgbase64", "")).strip())

        if any(
            marker in lowered
            for marker in (
                "session",
                "rename this session",
                "rename session",
                "recent sessions",
                "past session",
                "current session",
            )
        ):
            forced.add("session_tool")

        if any(
            marker in lowered
            for marker in (
                "what time",
                "whats the time",
                "what's the time",
                "current time",
                "time now",
                "today date",
                "current date",
                "what day",
                "today day",
            )
        ):
            forced.add("datetime_tool")

        if any(
            marker in lowered
            for marker in (
                "system info",
                "cpu",
                "ram",
                "battery",
                "disk",
                "storage",
                "running processes",
                "process list",
            )
        ):
            forced.add("system_info_tool")

        if any(marker in lowered for marker in ("weather", "forecast", "temperature")):
            forced.add("weather_tool")

        if any(
            marker in lowered
            for marker in (
                "who am i",
                "what do you know about me",
                "what is my",
                "my school",
                "school name",
                "where do i live",
                "where i live",
                "about me",
                "remember that",
                "forget that",
            )
        ):
            forced.add("memory_query")

        if self._explicit_code_writer_request(user_message):
            forced.add("code_writer")

        if lowered in {"are you sure", "you sure", "really", "really?", "sure?", "confirm that", "double check", "check again"}:
            last_tool_names = [str(item.get("name", "")).strip() for item in self.last_tool_trace if item.get("name")]
            if last_tool_names and last_tool_names[-1] == "memory_query":
                forced.add("memory_query")

        note_verbs = ("save", "write", "draft", "take", "briefing")
        if "note" in lowered and any(verb in lowered for verb in note_verbs):
            forced.add("notes_tool")

        if any(marker in lowered for marker in ("remind me", "reminder")):
            forced.add("reminder_tool")

        if any(marker in lowered for marker in ("calendar", "event", "events", "schedule", "meeting", "upcoming")):
            forced.add("calendar_tool")

        if any(marker in lowered for marker in ("email", "gmail", "inbox", "send mail")):
            forced.add("gmail_tool")

        if any(
            marker in lowered
            for marker in (
                "download",
                "downloads",
                "dowload",
                "file",
                "files",
                "folder",
                "path",
                "pdf",
                "docx",
                "txt",
                "read file",
                "open file",
                "open folder",
                "list files",
                "list folder",
                "find file",
                "save file",
                "write file",
            )
        ):
            forced.add("file_manager")

        if any(
            marker in lowered
            for marker in (
                "youtube",
                "browser",
                "website",
                "web page",
                ".com",
                "http://",
                "https://",
            )
        ) and any(
            marker in lowered
            for marker in (
                "open",
                "search",
                "play",
                "click",
                "type",
                "go to",
            )
        ):
            forced.add("browser_control")

        if has_live_image and any(
            marker in lowered
            for marker in (
                "what is this",
                "what's this",
                "what is what i am holding",
                "what am i holding",
                "what i'm holding",
                "what i am holding",
                "what is in my hand",
                "what's in my hand",
                "what do you see",
                "what can you see",
                "describe this",
                "describe what you see",
                "look at this",
                "what do i look like",
                "who is this",
                "analyze this",
                "analyze image",
                "analyze the image",
                "see this",
                "can you see",
                "describe image",
                "describe the image",
            )
        ):
            forced.add("vision_tool")
        if has_live_image and ("holding" in lowered or "in my hand" in lowered or "my hand" in lowered):
            forced.add("vision_tool")

        if "browser_control" in forced:
            forced.discard("app_launcher_tool")
            if any(marker in lowered for marker in ("youtube", "youtu.be", "spotify web", "soundcloud")):
                forced.discard("music_player")
        if "datetime_tool" in forced:
            forced.discard("system_info_tool")
        if "memory_query" in forced:
            forced.discard("session_tool")

        return forced & available

    def _explicit_code_writer_request(self, user_message: str) -> bool:
        return explicit_code_writer_request(user_message)

    def _should_answer_code_directly(self, user_message: str) -> bool:
        lowered = user_message.lower().strip()
        if self._explicit_code_writer_request(user_message):
            return False

        code_request_markers = (
            "small code",
            "small python",
            "code snippet",
            "small snippet",
            "write code",
            "give me code",
            "show code",
            "python class",
            "python function",
            "javascript function",
            "js function",
            "example code",
            "sample code",
            "just code",
            "only code",
            "only small code",
            "write a class",
            "write a function",
            "make a class",
            "make a function",
        )
        file_or_system_markers = (
            "save",
            "write to file",
            "create file",
            "edit file",
            "update file",
            "modify file",
            "tool",
            "jarvis tool",
            "plugin",
            "scaffold",
            "terminal",
            "run this",
            "execute",
            "folder",
            "project",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".json",
        )
        if any(marker in lowered for marker in code_request_markers) and not any(
            marker in lowered for marker in file_or_system_markers
        ):
            return True
        return False

    def _resolve_final_answer(
        self,
        response: dict[str, Any],
        tool_trace: list[dict[str, Any]],
    ) -> str:
        content = str(response.get("content", "")).strip()
        if content:
            sanitized = self._sanitize_assistant_message(content, tool_trace)
            if sanitized:
                return sanitized
        if not tool_trace:
            return "No response generated."

        failures = [
            item for item in tool_trace
            if isinstance(item.get("result"), dict) and item["result"].get("ok") is False
        ]
        if failures:
            failed = failures[-1]
            result = failed.get("result", {})
            error = str(result.get("error", "")).strip() or "unknown error"
            if failed.get("name") == "vision_tool":
                return f"vision_tool failed: {error}"
            return f"{failed.get('name', 'tool')} failed: {error}"

        last = tool_trace[-1]
        result = last.get("result", {})
        tool_name = str(last.get("name", "")).strip()
        if isinstance(result, dict):
            if tool_name == "memory_query":
                return self._summarize_memory_query_result(result)
            if tool_name == "datetime_tool" and result.get("time"):
                return f"The current time is {result.get('time')}."
            if tool_name == "weather_tool" and result.get("location") and result.get("description"):
                return (
                    f"The weather in {result.get('location')} is {result.get('description')} "
                    f"at {result.get('temperature')} degrees."
                )
            if tool_name == "system_info_tool" and any(key in result for key in ("cpu_percent", "ram_percent", "disk_percent")):
                return (
                    f"CPU {result.get('cpu_percent')}%, RAM {result.get('ram_percent')}%, "
                    f"disk {result.get('disk_percent')}%."
                )
            if tool_name == "file_manager":
                items = result.get("items")
                if isinstance(items, list):
                    return f"I found {len(items)} items in {result.get('path', 'that folder')}."
            if tool_name == "browser_control":
                return "The browser action ran, but I couldn't produce a clean final summary."
            if tool_name == "vision_tool" and result.get("analysis"):
                return str(result.get("analysis")).strip()

        return "The requested action ran, but I couldn't produce a final summary."

    def _sanitize_assistant_message(
        self,
        assistant_message: str,
        tool_trace: list[dict[str, Any]],
    ) -> str:
        cleaned = str(assistant_message or "").strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        raw_tool_markers = (
            "memory_query result:",
            "tool result:",
            '"ok":',
            '"memories":',
            '"items":',
            '"query":',
            '"name": "memory_query"',
        )
        last_result = tool_trace[-1].get("result", {}) if tool_trace else {}
        last_tool_name = str(tool_trace[-1].get("name", "")).strip() if tool_trace else ""

        if any(marker in lowered for marker in raw_tool_markers):
            cleaned_summary = self._rewrite_tool_dump_as_answer(cleaned, tool_trace)
            if cleaned_summary:
                return cleaned_summary
            return ""

        if (
            last_tool_name == "memory_query"
            and isinstance(last_result, dict)
            and last_result.get("ok") is True
            and last_result.get("memories")
            and any(
                marker in lowered
                for marker in (
                    "i don't have",
                    "not confirmed",
                    "can't confirm",
                    "not sure",
                    "i am not sure",
                    "i don't know",
                )
            )
        ):
            return self._summarize_memory_query_result(last_result)

        return cleaned

    def _rewrite_tool_dump_as_answer(
        self,
        assistant_message: str,
        tool_trace: list[dict[str, Any]],
    ) -> str:
        if not tool_trace:
            return ""

        latest = tool_trace[-1]
        tool_name = str(latest.get("name", "")).strip() or "tool"
        tool_result = latest.get("result", {})
        prompt = (
            "Rewrite the assistant draft into a clean final answer for the user.\n"
            "Use the tool result as the source of truth.\n"
            "Do not expose raw JSON, internal traces, or debug wording.\n"
            "If the tool failed, say what failed plainly.\n"
            "If the tool succeeded, summarize the useful result naturally.\n"
            "Return only the cleaned answer.\n\n"
            f"Tool name: {tool_name}\n"
            f"Tool result: {json.dumps(tool_result, ensure_ascii=True, default=str)}\n"
            f"Draft assistant message: {assistant_message}\n"
        )
        try:
            response = self.brain.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                task_kind="simple",
                system_override="You are JARVIS. Return only the cleaned final answer.",
            )
        except Exception:
            return ""
        return str(response.get("content", "")).strip()

    def _summarize_memory_query_result(self, result: dict[str, Any]) -> str:
        if result.get("ok") is False:
            return str(result.get("error", "Memory lookup failed.")).strip() or "Memory lookup failed."

        memories = result.get("memories")
        if isinstance(memories, list):
            cleaned = [str(item).strip() for item in memories if str(item).strip()]
            if not cleaned:
                query = str(result.get("query", "")).strip()
                if query:
                    return f"I couldn't find anything solid in memory for '{query}'."
                return "I couldn't find anything solid in memory."
            if len(cleaned) == 1:
                return cleaned[0]
            return cleaned[0]

        items = result.get("items")
        if isinstance(items, list):
            cleaned_items = [str(item).strip() for item in items if str(item).strip()]
            count = int(result.get("stored", len(cleaned_items)) or len(cleaned_items))
            if cleaned_items:
                return cleaned_items[0]
            return f"Stored {count} memory item{'s' if count != 1 else ''}."

        entities = result.get("entities")
        if isinstance(entities, list) and entities:
            first = entities[0]
            if isinstance(first, dict):
                name = str(first.get("name", "")).strip()
                summary = str(first.get("summary", "")).strip()
                if name and summary:
                    return f"{name}: {summary}"
                if name:
                    return name
            return str(entities[0]).strip()

        return "Memory lookup completed."

    def _remember_turn(self, user_message: str, assistant_message: str) -> None:
        if not (
            len(self.history) >= 2
            and self.history[-2].get("role") == "user"
            and self.history[-2].get("content") == user_message
            and self.history[-1].get("role") == "assistant"
            and self.history[-1].get("content") == assistant_message
        ):
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": assistant_message})
        self.history = self.history[-16:]

    def _finalize_response(
        self,
        user_message: str,
        assistant_message: str,
        messages: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        turn_meta: dict[str, Any] | None = None,
    ) -> str:
        assistant_message = self._sanitize_assistant_message(assistant_message, tool_trace)
        self._remember_turn(user_message, assistant_message)
        self.last_turn_meta = dict(turn_meta or {})
        self.last_tool_trace = [dict(item) for item in tool_trace]
        self.memory.persist_conversation(
            user_message=user_message,
            assistant_message=assistant_message,
            conversation=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            tool_trace=tool_trace,
            session_id=self.session_id,
            brain=self.brain,
            should_extract=self._should_extract_memory(user_message, tool_trace),
            background=self.settings.background_memory_persistence,
            turn_meta=turn_meta,
        )
        latest_session = self.memory.get_session(self.session_id)
        if isinstance(latest_session, dict):
            self.session_meta = latest_session
        return assistant_message
