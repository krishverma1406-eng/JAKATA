"""Main ReAct loop for JARVIS."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from config.settings import SETTINGS, Settings
from core.brain import Brain
from core.interface_config import INTERFACE_CONFIG
from core.memory import Memory
from core.planner import Planner
from core.tool_registry import ToolRegistry


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
}


def _get_fallback_hint(tool_name: str, error: str) -> str:
    """Get fallback hint for a failed tool based on error type."""
    # Auth/config errors — never retry, never fallback
    error_lower = error.lower()
    auth_signals = (
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
    if any(signal in error_lower for signal in auth_signals):
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
                plan_step_index = self.planner.next_matching_step_index(plan, tool_call["name"])
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "tool_started",
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
                result = self.tools.run_tool(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                    runtime_context={
                        "session_id": self.session_id,
                        "session_mode": self.mode,
                        "session_name": str(self.session_meta.get("display_name", "")).strip(),
                    },
                )
                tool_trace.append(
                    {
                        "name": tool_call["name"],
                        "arguments": tool_call.get("arguments", {}),
                        "result": result,
                    }
                )
                if event_handler is not None:
                    event_handler(
                        {
                            "type": "tool_result",
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
                
                # Inject fallback hint if tool failed
                tool_content = json.dumps(result, ensure_ascii=True, default=str)
                if isinstance(result, dict) and result.get("ok") is False:
                    error_msg = str(result.get("error", "unknown error")).strip()
                    fallback_note = _get_fallback_hint(tool_call["name"], error_msg)
                    if fallback_note:
                        tool_content = json.dumps(
                            {
                                **result,
                                "_fallback_instruction": fallback_note,
                            },
                            ensure_ascii=True,
                            default=str,
                        )
                
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_call["name"],
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

        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": user_message})
        return messages

    def _plan_execution_guidance(self) -> str:
        return (
            "Use the plan as high-level execution guidance only.\n"
            "Do not create micro-steps or call tools just because they appear in the plan.\n"
            "Choose the next necessary tool based on the latest tool result, and chain tools when one output unlocks the next action.\n"
            "Prefer inspection before action when state is uncertain, especially for browser, desktop, file, and screenshot-driven tasks."
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
        sections: list[str] = [intro]

        project_items = self.memory.active_project_items(limit=max_items)
        project_heading = str(config.get("projects_heading", "Active projects")).strip() or "Active projects"
        empty_projects = str(config.get("empty_projects", "No active project notes found.")).strip()
        if project_items:
            sections.append(project_heading + ":\n" + "\n".join(f"- {item}" for item in project_items[:max_items]))
        else:
            sections.append(f"{project_heading}:\n- {empty_projects}")

        reminder_heading = str(config.get("reminders_heading", "Pending reminders")).strip() or "Pending reminders"
        empty_reminders = str(config.get("empty_reminders", "No pending reminders.")).strip()
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
        sections.append(
            reminder_heading + ":\n" + ("\n".join(reminder_lines) if reminder_lines else f"- {empty_reminders}")
        )

        calendar_heading = str(config.get("calendar_heading", "Today's calendar")).strip() or "Today's calendar"
        empty_calendar = str(config.get("empty_calendar", "No calendar events scheduled for today.")).strip()
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
        sections.append(
            calendar_heading + ":\n" + ("\n".join(calendar_lines) if calendar_lines else f"- {empty_calendar}")
        )

        return "\n\n".join(part for part in sections if part.strip()).strip()

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
            "recently",
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
        if "?" in lowered:
            return False
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "open ",
                "search ",
                "play ",
                "list ",
                "show ",
                "tell me",
                "what ",
                "who ",
                "where ",
                "when ",
                "why ",
                "how ",
                "can you",
                "could you",
                "please ",
                "pls ",
            )
        ):
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
        return any(marker in lowered for marker in extract_markers)

    def _select_tool_definitions(
        self,
        user_message: str,
        all_tool_definitions: list[dict[str, Any]],
        plan: dict[str, Any],
        memory_context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        requested_tools: set[str] = set()
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
        forced_tools = self._forced_tool_names(user_message, all_tool_definitions)
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
            for extra_tool_name in step.get("tool_names", []):
                if extra_tool_name:
                    requested_tools.add(str(extra_tool_name))

        requested_tools.update(forced_tools)

        if memory_context and "memory_query" not in forced_tools:
            requested_tools.discard("memory_query")

        if "browser_control" in requested_tools and any(
            marker in lowered for marker in ("youtube", "youtu.be", "spotify web", "soundcloud")
        ):
            requested_tools.discard("music_player")

        if lowered in simple_chat_markers:
            return []

        if not requested_tools:
            return []

        selected_tool_definitions = [
            tool_definition
            for tool_definition in all_tool_definitions
            if tool_definition["name"] in requested_tools
        ]
        if selected_tool_definitions:
            return selected_tool_definitions
        return []

    def _forced_tool_names(
        self,
        user_message: str,
        all_tool_definitions: list[dict[str, Any]],
    ) -> set[str]:
        lowered = user_message.lower().strip()
        available = {str(tool_definition["name"]) for tool_definition in all_tool_definitions}
        forced: set[str] = set()

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

        if any(
            marker in lowered
            for marker in (
                "download",
                "downloads",
                "dowload",
                "file",
                "files",
                "folder",
                "read ",
                "list ",
                "find file",
                "save ",
                "write ",
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

        if "browser_control" in forced:
            forced.discard("app_launcher_tool")
            if any(marker in lowered for marker in ("youtube", "youtu.be", "spotify web", "soundcloud")):
                forced.discard("music_player")
        if "datetime_tool" in forced:
            forced.discard("system_info_tool")
        if "memory_query" in forced:
            forced.discard("session_tool")

        return forced & available

    def _resolve_final_answer(
        self,
        response: dict[str, Any],
        tool_trace: list[dict[str, Any]],
    ) -> str:
        content = str(response.get("content", "")).strip()
        if content:
            return content
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
            return f"{failed.get('name', 'tool')} failed: {error}"

        last = tool_trace[-1]
        result = last.get("result", {})
        tool_name = str(last.get("name", "")).strip()
        if isinstance(result, dict):
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

        return "The requested action ran, but I couldn't produce a final summary."

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
        turn_meta: dict[str, Any] | None = None,
    ) -> str:
        self._remember_turn(user_message, assistant_message)
        self.last_turn_meta = dict(turn_meta or {})
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
