"""Main ReAct loop for JARVIS."""

from __future__ import annotations

import json
import re
import time
import uuid
import queue
import threading
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable

from config.settings import BASE_DIR, DATA_AI_DIR, SETTINGS, Settings
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
    "timed out",
    "timeout",
)
_TOOL_CALL_TIMEOUT_SECONDS = 30


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
        self.pending_confirmation: dict[str, Any] | None = None
        self._runtime_policy_cache: dict[str, Any] | None = None
        self._runtime_policy_mtime_ns: int | None = None
        self._startup_messages_emitted = False
        self.bind_session()

    def bind_session(self, session_id: str | None = None, mode: str | None = None) -> str:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.mode = self.interface.normalize_mode(mode or self.mode)
        self.session_meta = self.memory.ensure_session(self.session_id, mode=self.mode)
        self.mode = self.interface.normalize_mode(self.session_meta.get("mode", self.mode))
        self.history = self.memory.load_session_messages(self.session_id, limit_messages=12)
        self.pending_confirmation = None
        self._startup_messages_emitted = False
        return self.session_id

    def _runtime_policy(self) -> dict[str, Any]:
        path = DATA_AI_DIR / "runtime_policy.json"
        if not path.exists():
            self._runtime_policy_cache = {}
            self._runtime_policy_mtime_ns = None
            return {}
        mtime_ns = path.stat().st_mtime_ns
        cached = getattr(self, "_runtime_policy_cache", None)
        cached_mtime = getattr(self, "_runtime_policy_mtime_ns", None)
        if cached is not None and cached_mtime == mtime_ns:
            return dict(cached)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        self._runtime_policy_cache = payload
        self._runtime_policy_mtime_ns = mtime_ns
        return dict(payload)

    def _policy_markers(self, key: str, defaults: tuple[str, ...] = ()) -> tuple[str, ...]:
        payload = self._runtime_policy()
        markers = payload.get("markers", {})
        values = markers.get(key, defaults) if isinstance(markers, dict) else defaults
        if not isinstance(values, list):
            values = list(defaults)
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip().lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return tuple(normalized) if normalized else defaults

    def _matches_policy(self, text: str, key: str, defaults: tuple[str, ...] = ()) -> bool:
        lowered = str(text or "").lower()
        return any(marker in lowered for marker in self._policy_markers(key, defaults))

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
        base_runtime_context = self._base_runtime_context(runtime_context)

        pending_resolution = self._handle_pending_confirmation(
            user_message,
            base_runtime_context,
            event_handler=event_handler,
        )
        if pending_resolution is not None:
            return self._finalize_response(
                user_message,
                pending_resolution["response"],
                [],
                pending_resolution.get("tool_trace", []),
                turn_meta={
                    "mode": self.mode,
                    "tool_count": len(pending_resolution.get("tool_trace", [])),
                    **pending_resolution.get("turn_meta", {}),
                },
            )

        runtime_resolution = self._run_runtime_controller(
            user_message,
            base_runtime_context,
            event_handler=event_handler,
        )
        if runtime_resolution is not None:
            return self._finalize_response(
                user_message,
                runtime_resolution["response"],
                [],
                runtime_resolution.get("tool_trace", []),
                turn_meta={
                    "mode": self.mode,
                    "tool_count": len(runtime_resolution.get("tool_trace", [])),
                    **runtime_resolution.get("turn_meta", {}),
                },
            )

        memory_context = self._load_memory_context(user_message)
        all_tool_definitions = self.tools.get_tool_definitions()
        plan = self.planner.create_plan(user_message, all_tool_definitions)
        plan_note = self.planner.render_plan(plan)
        tool_definitions = self._select_tool_definitions(
            user_message,
            all_tool_definitions,
            plan,
            memory_context,
            runtime_context=base_runtime_context,
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
        seen_calls_this_turn: set[tuple[str, str]] = set()
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
                call_key = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
                if call_key in seen_calls_this_turn:
                    result = {"ok": False, "error": "Duplicate tool call skipped"}
                    tool_trace.append(
                        {
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": result,
                            "fallback_used": False,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call.get("id"),
                            "content": json.dumps(result, ensure_ascii=True, default=str),
                        }
                    )
                    continue
                seen_calls_this_turn.add(call_key)
                confirmation = self._confirmation_payload_for_tool(tool_name, tool_args, user_message)
                if confirmation is not None:
                    prompt = self._queue_tool_confirmation(
                        str(confirmation.get("tool_name", "")).strip(),
                        dict(confirmation.get("arguments", {})),
                        user_message,
                        str(confirmation.get("prompt", "")).strip() or "Please confirm.",
                    )
                    if event_handler is not None:
                        event_handler(
                            {
                                "type": "tool_confirmation_requested",
                                "name": tool_name,
                                "arguments": tool_args,
                                "prompt": prompt,
                            }
                        )
                    return self._finalize_response(
                        user_message,
                        prompt,
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
                    runtime_context=base_runtime_context,
                    event_handler=event_handler,
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

    def _base_runtime_context(self, runtime_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_mode": self.mode,
            "session_name": str(self.session_meta.get("display_name", "")).strip(),
            **(runtime_context or {}),
        }

    @staticmethod
    def _runtime_resolution(
        response: str,
        tool_trace: list[dict[str, Any]] | None = None,
        turn_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "response": response,
            "tool_trace": tool_trace or [],
            "turn_meta": turn_meta or {},
        }

    def _handle_pending_confirmation(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        pending = self.pending_confirmation
        if pending is None:
            return None

        lowered = user_message.strip().lower()
        if self._is_negative_reply(lowered):
            self.pending_confirmation = None
            return self._runtime_resolution("Cancelled.")
        if not self._is_affirmative_reply(lowered):
            prompt = str(pending.get("prompt", "")).strip() or "Please answer yes or no."
            return self._runtime_resolution(f"{prompt} Reply yes or no, or say cancel.")

        self.pending_confirmation = None
        if pending.get("kind") != "tool":
            return self._runtime_resolution("Cancelled.")

        tool_trace: list[dict[str, Any]] = []
        result = self._run_workflow_tool(
            str(pending.get("tool_name", "")).strip(),
            dict(pending.get("arguments", {})),
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        response = self._format_tool_result(
            str(pending.get("tool_name", "")).strip(),
            result,
            tool_trace,
            user_message=str(pending.get("source_request", "")).strip(),
        )
        return self._runtime_resolution(response, tool_trace=tool_trace)

    def _is_affirmative_reply(self, lowered: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]+", " ", str(lowered or "").lower()).strip()
        allowed = set(
            self._policy_markers(
                "confirmation_yes",
                defaults=("yes", "y", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead", "confirm"),
            )
        )
        return normalized in allowed

    def _is_negative_reply(self, lowered: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]+", " ", str(lowered or "").lower()).strip()
        blocked = set(
            self._policy_markers(
                "confirmation_no",
                defaults=("no", "n", "nope", "cancel", "stop", "dont", "don't"),
            )
        )
        return normalized in blocked

    def _run_runtime_controller(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower().strip()
        if not lowered:
            return None

        for handler in (
            self._offline_capability_response,
            self._canonical_profile_response,
            self._project_catalog_response,
            self._session_runtime_response,
            self._system_runtime_response,
            self._file_delete_runtime_response,
            self._note_followup_response,
            self._image_followup_response,
            self._screenshot_runtime_response,
            self._gmail_runtime_response,
            self._browser_runtime_response,
            self._note_search_save_response,
            self._image_generation_response,
            self._vscode_runtime_response,
            self._code_execution_response,
            self._calculator_runtime_response,
        ):
            resolution = handler(user_message, runtime_context, event_handler=event_handler)
            if resolution is not None:
                return resolution
        return None

    def _run_workflow_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        trace = tool_trace if tool_trace is not None else []
        if event_handler is not None:
            event_handler({"type": "tool_started", "name": tool_name, "arguments": arguments})
        result = (
            self._execute_with_chain(tool_name, arguments, runtime_context, event_handler=event_handler)
            if allow_fallback
            else self._run_tool_with_timeout(tool_name, arguments, runtime_context, timeout=_TOOL_CALL_TIMEOUT_SECONDS)
        )
        trace.append(
            {
                "name": tool_name,
                "arguments": dict(arguments),
                "result": result,
                "fallback_used": bool(isinstance(result, dict) and result.get("_fallback_used", False)),
            }
        )
        if event_handler is not None:
            event_handler({"type": "tool_result", "name": tool_name, "arguments": arguments, "result": result})
        return result

    @staticmethod
    def _latest_tool_arguments(tool_name: str, tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
        for item in reversed(tool_trace):
            if str(item.get("name", "")).strip() != tool_name:
                continue
            arguments = item.get("arguments", {})
            if isinstance(arguments, dict):
                return dict(arguments)
        return {}

    def _queue_tool_confirmation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        source_request: str,
        prompt: str,
    ) -> str:
        self.pending_confirmation = {
            "kind": "tool",
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "source_request": source_request,
            "prompt": prompt,
        }
        return prompt

    def _confirmation_payload_for_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        source_request: str,
    ) -> dict[str, Any] | None:
        action = str(arguments.get("action", "")).strip().lower()

        if tool_name == "gmail_tool" and action == "send":
            to_address = str(arguments.get("to", "")).strip()
            subject = str(arguments.get("subject", "")).strip()
            body = str(arguments.get("body", "")).strip()
            if not to_address or not subject or not body:
                return None
            prompt = f'Send reply to {to_address} about "{subject}" with this message: "{body}"?'
            return {
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "prompt": prompt,
                "source_request": source_request,
            }

        if tool_name == "calendar_tool" and action == "create":
            summary = str(arguments.get("summary", "")).strip() or "this event"
            start_time = str(arguments.get("start_time", "")).strip()
            end_time = str(arguments.get("end_time", "")).strip()
            timing = " at the requested time"
            if start_time and end_time:
                timing = f" from {start_time} to {end_time}"
            prompt = f'Create calendar event "{summary}"{timing}?'
            return {
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "prompt": prompt,
                "source_request": source_request,
            }

        if tool_name == "file_manager" and action == "delete":
            target = str(arguments.get("path", "")).strip() or "that path"
            confirmed_args = dict(arguments)
            confirmed_args["confirm_destructive"] = True
            confirmed_args.setdefault("recursive", True)
            prompt = f"That will permanently delete {target}. Are you sure? This can't be undone."
            return {
                "tool_name": tool_name,
                "arguments": confirmed_args,
                "prompt": prompt,
                "source_request": source_request,
            }

        return None

    def _format_local_time(self, iso_value: str) -> str:
        cleaned = str(iso_value or "").strip()
        if not cleaned:
            return ""
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            return cleaned
        if parsed.tzinfo is None:
            return cleaned
        timezone_name = self.settings.reminder_timezone or "Asia/Calcutta"
        try:
            localized = parsed.astimezone(ZoneInfo(timezone_name))
        except Exception:
            localized = parsed
        return localized.strftime("%Y-%m-%d %I:%M %p")

    def _format_tool_result(
        self,
        tool_name: str,
        result: dict[str, Any] | Any,
        tool_trace: list[dict[str, Any]] | None = None,
        *,
        user_message: str = "",
    ) -> str:
        trace = tool_trace or []
        arguments = self._latest_tool_arguments(tool_name, trace)
        if not isinstance(result, dict):
            cleaned = str(result).strip()
            return cleaned or f"{tool_name} completed."

        if result.get("ok") is False:
            error = str(result.get("error", "")).strip() or "unknown error"
            if tool_name == "weather_tool" and "openweathermap_api_key" in error.lower():
                return "Weather API not configured. Set OPENWEATHERMAP_API_KEY."
            if tool_name == "image_gen_tool" and "nvidia_api_key" in error.lower():
                return "Image generation needs NVIDIA_API_KEY configured."
            if tool_name == "screenshot_tool" and "gemini_api_key" in error.lower():
                screenshot_path = str(result.get("path", "")).strip()
                if screenshot_path:
                    return f"Screenshot captured at {screenshot_path}, but vision analysis requires GEMINI_API_KEY."
            if tool_name == "browser_control":
                return f"Browser control failed: {error}"
            if tool_name == "terminal_tool":
                stderr = str(result.get("stderr", "")).strip()
                exit_code = result.get("exit_code")
                if stderr:
                    return f"Terminal command failed with exit code {exit_code}: {stderr}"
                return f"Terminal command failed: {error}"
            if tool_name == "gmail_tool":
                return error
            return f"{tool_name} failed: {error}"

        action = str(arguments.get("action", "")).strip().lower()

        if tool_name == "memory_query":
            return self._summarize_memory_query_result(result)

        if tool_name == "datetime_tool":
            time_value = str(result.get("time", "")).strip()
            timezone_name = str(result.get("timezone", "")).strip()
            if time_value and timezone_name:
                return f"It's {time_value} in {timezone_name}."

        if tool_name == "weather_tool":
            location = str(result.get("location", "")).strip()
            description = str(result.get("description", "")).strip()
            if location and description:
                temperature = result.get("temperature")
                feels_like = result.get("feels_like")
                response = f"{location} is {description}"
                if temperature is not None:
                    response += f" at {float(temperature):.2f}C"
                if feels_like is not None:
                    response += f", feels like {float(feels_like):.2f}C"
                response += "."
                return response

        if tool_name == "reminder_tool":
            reminder = result.get("reminder")
            if isinstance(reminder, dict):
                text = str(reminder.get("text", "")).strip() or "that"
                due_at = self._format_local_time(str(reminder.get("due_at", "")).strip())
                if due_at:
                    return f"Done - I'll remind you to {text} at {due_at}."
                return f"Done - reminder set for {text}."

        if tool_name == "notes_tool":
            title = str(result.get("title", "")).strip() or str(arguments.get("title", "")).strip()
            path = str(result.get("path", "")).strip()
            if path and title and action == "create":
                web_count = 0
                for item in trace:
                    if str(item.get("name", "")).strip() != "web_search":
                        continue
                    item_result = item.get("result", {})
                    results = item_result.get("results", []) if isinstance(item_result, dict) else []
                    if isinstance(results, list):
                        web_count = len(results)
                if web_count:
                    return f"Found {web_count} results. Note saved as '{title}' at {path}."
                return f"Note saved as '{title}' at {path}."
            if path and action == "read":
                content = str(result.get("content", "")).strip()
                if content:
                    return f"Opened note '{title or Path(path).stem}' at {path}.\n\n{content[:800].strip()}"
                return f"Opened note '{title or Path(path).stem}' at {path}."
            matches = result.get("matches", [])
            if isinstance(matches, list):
                cleaned = [str(item.get("title", "")).strip() for item in matches if isinstance(item, dict) and str(item.get("title", "")).strip()]
                if cleaned:
                    return "Matching notes: " + ", ".join(cleaned[:5]) + "."

        if tool_name == "web_search":
            results = result.get("results", [])
            if isinstance(results, list) and results:
                top = results[0] if isinstance(results[0], dict) else {}
                title = str(top.get("title", "")).strip()
                url = str(top.get("url", "")).strip()
                query = str(result.get("query", "")).strip()
                if title and url:
                    return f"Top web result for '{query or user_message}': {title} ({url})."
                if title:
                    return f"Top web result for '{query or user_message}': {title}."

        if tool_name == "browser_control":
            url = str(result.get("url", "")).strip() or str(arguments.get("url", "")).strip()
            task = str(arguments.get("task", "")).strip()
            if url:
                return f"Opened {url}."
            if task:
                return f"Completed browser task: {task}."
            return "Browser action completed."

        if tool_name == "screenshot_tool":
            analysis = str(result.get("analysis", "")).strip()
            if analysis:
                return analysis
            path = str(result.get("path", "")).strip()
            if path:
                return f"Screenshot captured at {path}."

        if tool_name == "system_info_tool":
            overview = {}
            processes: list[dict[str, Any]] = []
            alerts: list[dict[str, Any]] = []
            for item in trace:
                if str(item.get("name", "")).strip() != "system_info_tool":
                    continue
                item_result = item.get("result", {})
                if not isinstance(item_result, dict) or item_result.get("ok") is not True:
                    continue
                if "cpu_percent" in item_result:
                    overview = item_result
                if isinstance(item_result.get("processes"), list):
                    processes = item_result.get("processes", [])
                if isinstance(item_result.get("alerts"), list):
                    alerts = item_result.get("alerts", [])
            if not overview:
                overview = result
            parts: list[str] = []
            cpu_percent = overview.get("cpu_percent")
            ram_percent = overview.get("ram_percent")
            if cpu_percent is not None or ram_percent is not None:
                cpu_text = f"CPU at {float(cpu_percent):.1f}%" if cpu_percent is not None else ""
                ram_text = f"RAM at {float(ram_percent):.1f}%" if ram_percent is not None else ""
                parts.append(", ".join(part for part in (cpu_text, ram_text) if part) + ".")
            if processes:
                top_lines = []
                for item in processes[:3]:
                    name = str(item.get("name", "")).strip() or f"PID {item.get('pid')}"
                    cpu = item.get("cpu_percent")
                    memory = item.get("memory_percent")
                    details = []
                    if cpu is not None:
                        details.append(f"{float(cpu):.1f}% CPU")
                    if memory is not None:
                        details.append(f"{float(memory):.1f}% RAM")
                    top_lines.append(f"{name} ({', '.join(details)})" if details else name)
                if top_lines:
                    parts.append("Top processes: " + ", ".join(top_lines) + ".")
            if alerts:
                messages = [str(item.get("message", "")).strip() for item in alerts if isinstance(item, dict) and str(item.get("message", "")).strip()]
                if messages:
                    parts.append(" ".join(messages[:2]))
            return " ".join(part for part in parts if part).strip() or "System check completed."

        if tool_name == "calculator_tool":
            if "result" in result:
                return f"Result: {result.get('result')}"

        if tool_name == "code_runner":
            stdout = str(result.get("stdout", "")).strip()
            stderr = str(result.get("stderr", "")).strip()
            if stdout:
                return f"Output: {stdout}"
            if stderr:
                return f"Execution failed: {stderr}"
            if result.get("exit_code") == 0:
                return "The code ran successfully with no output."

        if tool_name == "file_manager":
            deleted = str(result.get("deleted", "")).strip()
            if deleted:
                return f"Deleted {deleted}."
            content = str(result.get("content", "")).strip()
            if content:
                path = str(result.get("path", "")).strip()
                return f"Read {path}.\n\n{content[:800].strip()}"
            items = result.get("items", [])
            if isinstance(items, list):
                path = str(result.get("path", "")).strip() or "that folder"
                return f"I found {len(items)} items in {path}."

        if tool_name == "task_manager":
            task = result.get("task")
            if isinstance(task, dict):
                title = str(task.get("title", "")).strip() or "Untitled"
                project = str(task.get("project", "")).strip()
                priority = str(task.get("priority", "")).strip()
                if action == "create":
                    suffix = f" in {project}" if project else ""
                    priority_text = f" ({priority} priority)" if priority else ""
                    return f"Task created - '{title}'{suffix}{priority_text}."
                status = str(task.get("status", "")).strip() or "updated"
                return f"Task '{title}' is now {status}."
            tasks = result.get("tasks", [])
            if isinstance(tasks, list):
                return f"I found {len(tasks)} active tasks."

        if tool_name == "session_tool":
            session = result.get("session")
            sessions = result.get("sessions", [])
            if isinstance(session, dict):
                display_name = str(session.get("display_name", "")).strip() or "Untitled session"
                if action == "rename":
                    return f"Session renamed to '{display_name}'."
                return display_name
            if action == "list" and isinstance(sessions, list):
                names = [str(item.get("display_name", "")).strip() for item in sessions if isinstance(item, dict) and str(item.get("display_name", "")).strip()]
                if names:
                    return "Recent sessions: " + ", ".join(names[:5]) + "."
            if action == "search" and isinstance(sessions, list) and sessions:
                match = sessions[0] if isinstance(sessions[0], dict) else {}
                name = str(match.get("display_name", "")).strip() or "Untitled session"
                snippets = [str(snippet).strip() for snippet in match.get("snippets", []) if str(snippet).strip()]
                if snippets:
                    return f"{name}: {snippets[0]}"
                return name

        if tool_name == "gmail_tool":
            sent = result.get("sent")
            if isinstance(sent, dict) or action == "send":
                to_address = str(arguments.get("to") or result.get("to") or "").strip()
                subject = str(arguments.get("subject") or result.get("subject") or "").strip()
                if not to_address:
                    return "Email sent."
                if not subject:
                    return f"Sent the email to {to_address}."
                return f"Sent the email to {to_address} with subject '{subject}'."
            messages = result.get("messages", [])
            if isinstance(messages, list) and messages:
                latest = messages[0] if isinstance(messages[0], dict) else {}
                sender = str(latest.get("from", "")).strip() or "Unknown sender"
                subject = str(latest.get("subject", "")).strip() or "(no subject)"
                return f"Most recent unread email: {sender} | {subject}."

        if tool_name == "calendar_tool":
            event = result.get("event")
            if isinstance(event, dict):
                summary = str(event.get("summary", "")).strip() or str(arguments.get("summary", "")).strip() or "Event"
                start = str(event.get("start", "")).strip() or str(arguments.get("start_time", "")).strip()
                return f"Calendar event '{summary}' created for {start}."
            events = result.get("events", [])
            if isinstance(events, list):
                return f"You have {len(events)} event(s) today."

        if tool_name == "image_gen_tool":
            path = str(result.get("path", "")).strip()
            url = str(result.get("url", "")).strip()
            if path and url:
                return f"Image generated and saved at {path}. Viewer: {url}"
            if path:
                return f"Image generated and saved at {path}."
            if url:
                return f"Image generated: {url}"

        if tool_name == "terminal_tool":
            program = str(result.get("program", "")).strip()
            if result.get("ok") is True and program:
                arguments_list = result.get("arguments", [])
                args_text = " ".join(str(item).strip() for item in arguments_list if str(item).strip())
                suffix = f" {args_text}" if args_text else ""
                return f"Started {program}{suffix}."

        if tool_name == "app_launcher_tool":
            opened = str(result.get("opened", "")).strip()
            resolved = str(result.get("resolved", "")).strip()
            if opened or resolved:
                return f"Opened {opened or resolved}."

        cleaned = str(result.get("message", "")).strip()
        return cleaned or f"{tool_name} completed."

    def _canonical_profile_response(
        self,
        user_message: str,
        _runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        if not self._matches_policy(
            user_message,
            "profile_queries",
            defaults=(
                "what is my name",
                "where do i live",
                "where i live",
                "what school",
                "study in",
                "study at",
                "who am i",
                "using my memory",
                "specific things you know",
                "know about me",
            ),
        ):
            return None
        facts = self.memory.recall(user_message, limit=5)
        if not facts:
            return self._runtime_resolution("I couldn't find a solid profile answer in memory.")
        if event_handler is not None:
            event_handler(
                {
                    "type": "activity",
                    "payload": {
                        "event": "memory_context_loaded",
                        "message": "; ".join(facts[:4]),
                    },
                }
            )
        return self._runtime_resolution(" ".join(facts[:3]).strip())

    def _project_catalog_response(
        self,
        user_message: str,
        _runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        if not self._matches_policy(
            user_message,
            "project_queries",
            defaults=(
                "active projects",
                "project list",
                "full project list",
                "what are we working on",
                "working on right now",
                "what am i working on",
            ),
        ):
            return None
        catalog = self.memory.active_project_catalog(limit=8)
        if not catalog:
            return self._runtime_resolution("I couldn't find any active project records.")
        lines = []
        for index, item in enumerate(catalog, start=1):
            name = str(item.get("name", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if detail and detail.lower() != name.lower():
                lines.append(f"{index}. {name} - {detail[:120]}")
            else:
                lines.append(f"{index}. {name}")
        if event_handler is not None:
            event_handler(
                {
                    "type": "activity",
                    "payload": {"event": "memory_context_loaded", "message": "; ".join(item["name"] for item in catalog[:4])},
                }
            )
        return self._runtime_resolution("Current active projects:\n" + "\n".join(lines))

    def _session_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if (
            self._matches_policy(user_message, "session_search_queries", defaults=("recent sessions",))
            and "about" in lowered
        ):
            topic_match = re.search(r"about\s+(.+)", user_message, flags=re.IGNORECASE)
            topic = str(topic_match.group(1)).strip(" ?.") if topic_match else ""
            tool_trace: list[dict[str, Any]] = []
            recent = self._run_workflow_tool(
                "session_tool",
                {"action": "list", "limit": 5},
                runtime_context,
                event_handler=event_handler,
                tool_trace=tool_trace,
                allow_fallback=False,
            )
            search = self._run_workflow_tool(
                "session_tool",
                {"action": "search", "query": topic or user_message, "limit": 3},
                runtime_context,
                event_handler=event_handler,
                tool_trace=tool_trace,
                allow_fallback=False,
            )
            sessions = recent.get("sessions", []) if isinstance(recent, dict) else []
            matches = search.get("sessions", []) if isinstance(search, dict) else []
            recent_names = [str(item.get("display_name", "")).strip() for item in sessions if isinstance(item, dict)]
            response_parts: list[str] = []
            if recent_names:
                response_parts.append("Recent sessions: " + ", ".join(recent_names[:5]) + ".")
            if matches:
                match = matches[0]
                name = str(match.get("display_name", "Untitled session")).strip() or "Untitled session"
                updated_at = str(match.get("updated_at", "")).strip()
                snippets = [str(snippet).strip() for snippet in match.get("snippets", []) if str(snippet).strip()]
                snippet_text = f" {snippets[0]}" if snippets else ""
                response_parts.append(f'The closest match for "{topic or user_message}" is "{name}" from {updated_at[:10] or "an earlier session"}.{snippet_text}')
            elif topic:
                response_parts.append(f'I could not find a recent session clearly about "{topic}".')
            if response_parts:
                return self._runtime_resolution(" ".join(response_parts), tool_trace=tool_trace)
            return None

        if not self._is_session_summary_request(user_message):
            return None
        recalls = self.memory.session_recall(user_message, limit=4)
        if not recalls:
            return self._runtime_resolution("I couldn't find a saved session matching that conversation request.")
        return self._runtime_resolution("Here are the closest saved exchanges:\n- " + "\n- ".join(recalls[:3]))

    def _system_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        if not self._matches_policy(
            user_message,
            "system_slow_queries",
            defaults=(
                "cpu is feeling slow",
                "system is slow",
                "computer is slow",
                "slow check whats happening",
                "slow check what's happening",
            ),
        ):
            return None
        tool_trace: list[dict[str, Any]] = []
        overview = self._run_workflow_tool(
            "system_info_tool",
            {"action": "overview"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        processes = self._run_workflow_tool(
            "system_info_tool",
            {"action": "processes", "max_results": 5},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        alerts = self._run_workflow_tool(
            "system_info_tool",
            {"action": "alerts"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        final_result = alerts if isinstance(alerts, dict) and alerts.get("ok") is True else processes
        if isinstance(final_result, dict) and final_result.get("ok") is True:
            return self._runtime_resolution(
                self._format_tool_result("system_info_tool", final_result, tool_trace, user_message=user_message),
                tool_trace=tool_trace,
            )
        failing = processes if isinstance(processes, dict) and processes.get("ok") is False else overview
        return self._runtime_resolution(
            self._format_tool_result("system_info_tool", failing, tool_trace, user_message=user_message),
            tool_trace=tool_trace,
        )

    def _file_delete_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        del runtime_context, event_handler
        if not self._matches_policy(user_message, "file_delete_queries", defaults=("delete", "remove")):
            return None
        lowered = user_message.lower()
        if "file" not in lowered and "folder" not in lowered and "directory" not in lowered:
            return None

        target = ""
        quoted = self._extract_quoted_text(user_message)
        if quoted:
            target = quoted
        elif self._matches_policy(
            user_message,
            "downloads_folder_queries",
            defaults=("downloads folder", "download folder", "my downloads", "downloads"),
        ):
            target = str(Path.home() / "Downloads")

        if not target:
            return self._runtime_resolution("Which file or folder should I delete? I need the exact path before I can ask for confirmation.")

        recursive = "folder" in lowered or "directory" in lowered or "all files" in lowered
        arguments = {
            "action": "delete",
            "path": target,
            "recursive": recursive,
            "confirm_destructive": True,
        }
        prompt = self._queue_tool_confirmation(
            "file_manager",
            arguments,
            user_message,
            f"That will permanently delete {target}. Are you sure? This can't be undone.",
        )
        return self._runtime_resolution(prompt, turn_meta={"confirmation_required": True})

    def _note_followup_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        if not self._matches_policy(
            user_message,
            "note_followup_queries",
            defaults=("open that note", "read that note", "show that note", "open the note"),
        ):
            return None
        note_title = self._last_note_title()
        if not note_title:
            return None
        tool_trace: list[dict[str, Any]] = []
        result = self._run_workflow_tool(
            "notes_tool",
            {"action": "read", "title": note_title},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(result, dict) and result.get("ok"):
            content = str(result.get("content", "")).strip()
            excerpt = content[:800].strip()
            response = f"Opened note '{note_title}' at {result.get('path', '')}."
            if excerpt:
                response += f"\n\n{excerpt}"
            return self._runtime_resolution(response, tool_trace=tool_trace)
        return self._runtime_resolution(self._format_tool_result("notes_tool", result, tool_trace), tool_trace=tool_trace)

    def _image_followup_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(
            user_message,
            "image_followup_queries",
            defaults=("where did you save", "where you saved", "open it", "open for me", "where is the image"),
        ):
            return None
        image_result = self._last_successful_tool_result("image_gen_tool")
        if not image_result:
            return None
        path = str(image_result.get("path", "")).strip()
        url = str(image_result.get("url", "")).strip()
        tool_trace: list[dict[str, Any]] = []
        if "open" in lowered and path and Path(path).exists():
            open_result = self._run_workflow_tool(
                "app_launcher_tool",
                {"target": path},
                runtime_context,
                event_handler=event_handler,
                tool_trace=tool_trace,
                allow_fallback=False,
            )
            if isinstance(open_result, dict) and open_result.get("ok"):
                return self._runtime_resolution(f"Opened the generated image from {path}.", tool_trace=tool_trace)
        response = f"The generated image is saved at {path}."
        if url:
            response += f" Viewer URL: {url}"
        return self._runtime_resolution(response, tool_trace=tool_trace)

    def _screenshot_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if "screenshot" not in lowered or not self._matches_policy(
            user_message,
            "screenshot_describe_queries",
            defaults=(
                "what is on my screen",
                "what's on my screen",
                "tell me whats on my screen",
                "tell me what's on my screen",
                "describe my screen",
            ),
        ):
            return None
        tool_trace: list[dict[str, Any]] = []
        capture = self._run_workflow_tool(
            "screenshot_tool",
            {"action": "capture"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if not isinstance(capture, dict) or capture.get("ok") is not True:
            return self._runtime_resolution(self._format_tool_result("screenshot_tool", capture, tool_trace), tool_trace=tool_trace)
        analyze = self._run_workflow_tool(
            "screenshot_tool",
            {
                "action": "analyze",
                "path": str(capture.get("path", "")).strip(),
                "prompt": (
                    "Describe only what is visibly on the screen. Mention the visible apps, windows, headings, "
                    "and notable text. Do not infer hidden intent, future plans, or facts that are not visible."
                ),
            },
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        return self._runtime_resolution(self._format_tool_result("screenshot_tool", analyze, tool_trace), tool_trace=tool_trace)

    def _gmail_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if "unread" not in lowered or "email" not in lowered:
            return None
        if "reply" not in lowered and "send" not in lowered:
            return None
        tool_trace: list[dict[str, Any]] = []
        unread = self._run_workflow_tool(
            "gmail_tool",
            {"action": "unread", "max_results": 5},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if not isinstance(unread, dict) or unread.get("ok") is not True:
            return self._runtime_resolution(self._format_tool_result("gmail_tool", unread, tool_trace), tool_trace=tool_trace)
        messages = unread.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return self._runtime_resolution("You do not have any unread emails right now.", tool_trace=tool_trace)
        latest = messages[0] if isinstance(messages[0], dict) else {}
        to_address = self._extract_email_address(str(latest.get("from", "")).strip())
        subject = str(latest.get("subject", "")).strip()
        body = self._extract_quoted_text(user_message) or self._extract_reply_body(user_message)
        if not to_address or not subject or not body:
            return self._runtime_resolution("I found the unread email, but I could not build a safe reply from your instruction.", tool_trace=tool_trace)
        sender = str(latest.get("from", "")).strip() or to_address
        response = self._queue_tool_confirmation(
            "gmail_tool",
            {
                "action": "send",
                "to": to_address,
                "subject": f"Re: {subject}",
                "body": body,
            },
            user_message,
            f'Send reply to {sender} about "{subject}" with this message: "{body}"?',
        )
        return self._runtime_resolution(response, tool_trace=tool_trace)

    def _browser_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(user_message, "youtube_service_queries", defaults=("youtube",)):
            return None
        if not self._matches_policy(user_message, "youtube_action_queries", defaults=("open", "search", "play")):
            return None
        query_match = re.search(r"search\s+for\s+(.+)$", user_message, flags=re.IGNORECASE)
        search_query = str(query_match.group(1)).strip(" ?.") if query_match else ""
        url = "https://www.youtube.com/"
        if search_query:
            url = f"https://www.youtube.com/results?search_query={quote_plus(search_query)}"
        tool_trace: list[dict[str, Any]] = []
        browser_result = self._run_workflow_tool(
            "browser_control",
            {"action": "open", "url": url},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(browser_result, dict) and browser_result.get("ok") is True:
            response = f'Opened YouTube{" search results for " + search_query if search_query else ""}.'
            return self._runtime_resolution(response, tool_trace=tool_trace)
        launch_result = self._run_workflow_tool(
            "terminal_tool",
            {"action": "start_process", "program": url, "wait": False},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(launch_result, dict) and launch_result.get("ok") is True:
            response = f'Opened YouTube{" search results for " + search_query if search_query else ""} in your default browser.'
            return self._runtime_resolution(response, tool_trace=tool_trace)
        failing_result = launch_result if isinstance(launch_result, dict) and launch_result.get("ok") is False else browser_result
        failing_tool = "terminal_tool" if failing_result is launch_result else "browser_control"
        return self._runtime_resolution(
            self._format_tool_result(failing_tool, failing_result, tool_trace),
            tool_trace=tool_trace,
        )

    def _note_search_save_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(user_message, "web_note_queries", defaults=("search the web", "save a note")):
            return None
        query = re.sub(r"^search the web for\s+", "", user_message, flags=re.IGNORECASE).strip()
        query = re.sub(r"\s+and save a note about it\.?$", "", query, flags=re.IGNORECASE).strip()
        if not query:
            return None
        tool_trace: list[dict[str, Any]] = []
        search = self._run_workflow_tool(
            "web_search",
            {"query": query, "max_results": 5},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if not isinstance(search, dict) or search.get("ok") is not True:
            return self._runtime_resolution(self._format_tool_result("web_search", search, tool_trace), tool_trace=tool_trace)
        results = [item for item in search.get("results", []) if isinstance(item, dict)]
        if not results:
            return self._runtime_resolution("I did not find any web results worth saving.", tool_trace=tool_trace)
        summary_lines = []
        for item in results[:3]:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            if title:
                summary_lines.append(f"- {title}" + (f" | {url}" if url else ""))
        note_title = f"Web research {datetime.now().strftime('%Y-%m-%d %H%M')}"
        notes = self._run_workflow_tool(
            "notes_tool",
            {
                "action": "create",
                "title": note_title,
                "content": "# Web Research\n\n" + "\n".join(summary_lines),
                "tags": ["web-research"],
                "template": "research",
            },
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(notes, dict) and notes.get("ok") is True:
            return self._runtime_resolution(
                f"Found {len(results)} results. Note saved as '{note_title}' at {notes.get('path', '')}.",
                tool_trace=tool_trace,
            )
        return self._runtime_resolution(self._format_tool_result("notes_tool", notes, tool_trace), tool_trace=tool_trace)

    def _image_generation_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(
            user_message,
            "image_generation_queries",
            defaults=(
                "generate image",
                "generate an image",
                "create image",
                "create an image",
                "make an image",
                "make a picture",
                "draw",
                "visualize",
            ),
        ):
            return None
        prompt = re.sub(
            r"^(?:i want to\s+)?(?:generate|create|make|draw|visualize)\s+(?:an?\s+)?(?:image|picture|illustration)?\s*(?:of\s+)?",
            "",
            user_message.strip(),
            flags=re.IGNORECASE,
        ).strip(" .")
        prompt = prompt or user_message.strip()
        style = "digital-art"
        if "anime" in lowered:
            style = "anime"
        elif "realistic" in lowered:
            style = "realistic"
        elif "cinematic" in lowered:
            style = "cinematic"
        tool_trace: list[dict[str, Any]] = []
        result = self._run_workflow_tool(
            "image_gen_tool",
            {"prompt": prompt, "style": style, "size": "square"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        return self._runtime_resolution(self._format_tool_result("image_gen_tool", result, tool_trace), tool_trace=tool_trace)

    def _vscode_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(user_message, "vscode_queries", defaults=("vscode", "vs code")):
            return None
        if not self._matches_policy(user_message, "vscode_open_queries", defaults=("open",)):
            return None
        if not self._matches_policy(user_message, "vscode_target_queries", defaults=("folder", "project")):
            return None
        target_path = BASE_DIR / "jarvis"
        if not target_path.exists():
            target_path = BASE_DIR
        tool_trace: list[dict[str, Any]] = []
        which_result = self._run_workflow_tool(
            "terminal_tool",
            {"action": "which", "topic": "code"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        program = str(which_result.get("path", "")).strip() if isinstance(which_result, dict) else ""
        if not program:
            program = "code"
        open_result = self._run_workflow_tool(
            "terminal_tool",
            {"action": "start_process", "program": program, "arguments": [str(target_path)], "wait": False},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(open_result, dict) and open_result.get("ok") is True:
            return self._runtime_resolution(f"Opened {target_path} in VS Code.", tool_trace=tool_trace)
        uri_result = self._run_workflow_tool(
            "terminal_tool",
            {"action": "start_process", "program": f"vscode://file/{target_path.as_posix()}", "wait": False},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if isinstance(uri_result, dict) and uri_result.get("ok") is True:
            return self._runtime_resolution(f"Opened {target_path} in VS Code.", tool_trace=tool_trace)
        failing_result = uri_result if isinstance(uri_result, dict) and uri_result.get("ok") is False else open_result
        return self._runtime_resolution(self._format_tool_result("terminal_tool", failing_result, tool_trace), tool_trace=tool_trace)

    def _code_execution_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        explicit_code = ""
        explicit_match = re.search(r"run this code for me:\s*(.+)$", user_message, flags=re.IGNORECASE | re.DOTALL)
        if explicit_match:
            explicit_code = explicit_match.group(1).strip()
        if not explicit_code and not self._matches_policy(
            user_message,
            "code_execution_queries",
            defaults=("run this code", "execute this code", "run it with test cases", "run it with test case"),
        ):
            return None

        code = explicit_code
        if not code:
            code_prompt = (
                "Write only runnable Python code for the user's request.\n"
                "Do not include markdown fences, prompts, or explanation.\n"
                "Include simple test cases if the user asked to run tests.\n\n"
                f"User request: {user_message}"
            )
            drafted = self.brain.chat(
                messages=[{"role": "user", "content": code_prompt}],
                tools=[],
                task_kind="code",
                system_override="You are JARVIS. Return only runnable Python code.",
            )
            code = self._strip_code_fences(str(drafted.get("content", "")).strip())
        if not code:
            return self._runtime_resolution("I couldn't produce runnable Python code for that request.")

        tool_trace: list[dict[str, Any]] = []
        result = self._run_workflow_tool(
            "code_runner",
            {"code": code, "timeout_seconds": 15},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        output = self._format_tool_result("code_runner", result, tool_trace, user_message=user_message)
        response = f"```python\n{code}\n```\n\n{output}".strip()
        return self._runtime_resolution(response, tool_trace=tool_trace)

    def _calculator_runtime_response(
        self,
        user_message: str,
        runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", lowered)
        if not percent_match:
            return None
        percent = float(percent_match.group(1))
        value = float(percent_match.group(2))
        tool_trace: list[dict[str, Any]] = []
        step_one = self._run_workflow_tool(
            "calculator_tool",
            {"expression": f"{value} * {percent / 100}"},
            runtime_context,
            event_handler=event_handler,
            tool_trace=tool_trace,
            allow_fallback=False,
        )
        if not isinstance(step_one, dict) or step_one.get("ok") is not True:
            return self._runtime_resolution(self._format_tool_result("calculator_tool", step_one, tool_trace), tool_trace=tool_trace)
        first_result = float(step_one.get("result", 0.0) or 0.0)
        response = f"{percent:g}% of {value:g} is {first_result:.2f}."
        rate_match = re.search(r"1\s*dollar\s*(?:is|=)\s*(\d+(?:\.\d+)?)", lowered)
        if rate_match:
            rate = float(rate_match.group(1))
            step_two = self._run_workflow_tool(
                "calculator_tool",
                {"expression": f"{first_result} * {rate}"},
                runtime_context,
                event_handler=event_handler,
                tool_trace=tool_trace,
                allow_fallback=False,
            )
            if isinstance(step_two, dict) and step_two.get("ok") is True:
                converted = (Decimal(str(first_result)) * Decimal(str(rate))).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )
                response += f" At Rs.{rate:g} per dollar, that's Rs.{converted:,.2f}."
        return self._runtime_resolution(response, tool_trace=tool_trace)

    def _offline_capability_response(
        self,
        user_message: str,
        _runtime_context: dict[str, Any],
        *,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        lowered = user_message.lower()
        if not self._matches_policy(user_message, "offline_capability_queries", defaults=("wifi is down", "without internet")):
            return None
        if not self._matches_policy(user_message, "offline_capability_action_queries", defaults=("what can you still do", "what can you do")):
            return None
        local_capabilities = [
            "system inspection",
            "local file work",
            "notes",
            "reminders",
            "task tracking",
            "calculator",
            "Python code execution",
            "local music playback",
            "desktop automation",
        ]
        online_capabilities = [
            "web search",
            "Gmail",
            "calendar",
            "weather",
            "browser automation",
            "image generation",
            "live provider chat",
        ]
        response = (
            "Without internet, I can still handle "
            + ", ".join(local_capabilities[:-1])
            + f", and {local_capabilities[-1]}. "
            + "Internet-dependent features are "
            + ", ".join(online_capabilities[:-1])
            + f", and {online_capabilities[-1]}. This reply is using a live model right now, but the local tools keep working offline."
        )
        if event_handler is not None:
            event_handler({"type": "activity", "payload": {"event": "offline_capabilities", "message": response}})
        return self._runtime_resolution(response)

    def _last_note_title(self) -> str:
        for trace_item in reversed(self.last_tool_trace):
            if str(trace_item.get("name", "")).strip() != "notes_tool":
                continue
            result = trace_item.get("result", {})
            title = str(result.get("title", "")).strip()
            if title:
                return title
        return ""

    def _last_successful_tool_result(self, tool_name: str) -> dict[str, Any]:
        for trace_item in reversed(self.last_tool_trace):
            if str(trace_item.get("name", "")).strip() != tool_name:
                continue
            result = trace_item.get("result", {})
            if isinstance(result, dict) and result.get("ok") is True:
                return result
        return {}

    @staticmethod
    def _extract_email_address(value: str) -> str:
        match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", str(value))
        return str(match.group(1)).strip() if match else ""

    @staticmethod
    def _extract_quoted_text(value: str) -> str:
        match = re.search(r'"([^"]+)"', str(value))
        return str(match.group(1)).strip() if match else ""

    @staticmethod
    def _extract_reply_body(value: str) -> str:
        match = re.search(r"saying\s+(.+)$", str(value), flags=re.IGNORECASE)
        if not match:
            return ""
        return str(match.group(1)).strip(" .")

    @staticmethod
    def _strip_code_fences(value: str) -> str:
        cleaned = str(value or "").strip()
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

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

        user_name = ""
        try:
            user_name = str(self.memory.profile_fields().get("name", "")).strip()
        except Exception:
            user_name = ""
        briefing_target = user_name or "the user"
        prompt = (
            f"You are JARVIS giving {briefing_target} an operational briefing.\n"
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
        code_nouns = {
            "script",
            "code",
            "coding",
            "program",
            "function",
            "class",
            "module",
            "api",
        }
        code_intent_verbs = {
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
            "edit",
            "update",
            "add",
            "remove",
            "change",
        }
        execution_markers = {
            "traceback",
            "exception",
            "error",
            "failing",
            "failure",
            "bug",
            "test",
            "tests",
            "unit",
            "stack",
        }
        lowered_tokens = set(re.findall(r"[a-z0-9_+.:-]+", lowered))
        if any(step.get("tool_name") == "code_writer" for step in plan.get("steps", [])):
            return "code"
        if any(marker in lowered for marker in code_phrases):
            return "code"
        has_code_noun = bool(lowered_tokens & code_nouns)
        has_code_intent = bool(lowered_tokens & code_intent_verbs)
        has_execution_marker = bool(lowered_tokens & execution_markers)
        if has_code_noun and (has_code_intent or has_execution_marker):
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

    def _is_session_summary_request(self, user_message: str) -> bool:
        return self._matches_policy(
            user_message,
            "session_summary_queries",
            defaults=(
                "what did we talk",
                "what we talked",
                "what have we discussed",
                "what did we discuss",
                "summarize our",
                "our chat",
                "chat so far",
                "conversation so far",
                "recent chat",
                "recent conversation",
                "previous session",
                "previous sessions",
                "last session",
                "what were we",
                "what did we build",
                "what was i doing",
                "yesterday",
                "earlier today",
            ),
        )

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
            "task_manager": (
                "task", "todo", "to-do", "backlog", "mark done", "complete task",
                "blocked", "in progress", "task list", "create task", "new task",
            ),
            "reminder_tool": (
                "remind me", "reminder", "remind", "at 6", "at 7", "tomorrow at", "today at",
            ),
            "system_info_tool": (
                "system info", "cpu", "ram", "battery", "disk", "storage", "processes",
                "system alert", "system alerts", "low battery", "all clear",
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
            "image_gen_tool": (
                "draw", "image", "picture", "visualize", "illustration", "concept art",
                "mockup", "poster", "cover art", "generate image", "create image",
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
            "code_runner": (
                "run python", "execute code", "test snippet", "verify output", "run this code",
                "execute this code", "stdin", "python snippet",
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

    def _tool_is_registered(self, tool_name: str) -> bool:
        has_tool = getattr(self.tools, "has_tool", None)
        if callable(has_tool):
            try:
                return bool(has_tool(tool_name))
            except Exception:
                return False

        get_tool_definitions = getattr(self.tools, "get_tool_definitions", None)
        if callable(get_tool_definitions):
            try:
                return any(
                    str(tool_definition.get("name", "")).strip() == tool_name
                    for tool_definition in get_tool_definitions()
                    if isinstance(tool_definition, dict)
                )
            except Exception:
                return False

        return False

    def _run_tool_with_timeout(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_context: dict[str, Any],
        timeout: int = _TOOL_CALL_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

        def _invoke_tool() -> None:
            try:
                payload = self.tools.run_tool(tool_name, arguments, runtime_context)
            except Exception as exc:  # pragma: no cover - exercised via caller path
                payload = {"ok": False, "error": str(exc)}
            try:
                result_queue.put_nowait(payload)
            except queue.Full:
                return

        worker = threading.Thread(target=_invoke_tool, name=f"tool:{tool_name}", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            return {"ok": False, "error": f"{tool_name} timed out after {timeout}s"}

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return {"ok": False, "error": f"{tool_name} returned no result"}

    def _execute_with_chain(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        runtime_context: dict[str, Any],
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a tool, then automatically try fallbacks before returning to the model."""
        result = self._run_tool_with_timeout(
            tool_name,
            arguments,
            runtime_context,
            timeout=_TOOL_CALL_TIMEOUT_SECONDS,
        )
        if isinstance(result, dict) and result.get("ok") is True:
            return result

        error_msg = str(result.get("error", "")).lower() if isinstance(result, dict) else ""
        if self._tool_error_blocks_fallback(error_msg):
            return result

        fallbacks = [
            fallback_name
            for fallback_name in TOOL_FALLBACK_CHAINS.get(tool_name, [])
            if self._tool_is_registered(fallback_name)
        ]
        if fallbacks and event_handler is not None:
            event_handler(
                {
                    "type": "tool_fallback_suggested",
                    "failed_tool": tool_name,
                    "fallback_tools": list(fallbacks),
                    "error": str(result.get("error", "")) if isinstance(result, dict) else "",
                }
            )
        for fallback_name in fallbacks:
            fallback_args = self._adapt_args_for_fallback(tool_name, fallback_name, arguments)
            fallback_result = self._run_tool_with_timeout(
                fallback_name,
                fallback_args,
                runtime_context,
                timeout=_TOOL_CALL_TIMEOUT_SECONDS,
            )
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
        session_summary_request = self._is_session_summary_request(user_message)

        if session_summary_request or any(
            marker in lowered
            for marker in (
                "session",
                "rename this session",
                "rename session",
                "recent sessions",
                "past session",
                "current session",
                "our chat",
                "what was discussed",
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
                "system alert",
                "system alerts",
                "low battery",
                "all clear",
                "anything wrong with system",
            )
        ):
            forced.add("system_info_tool")

        if any(marker in lowered for marker in ("weather", "forecast", "temperature")):
            forced.add("weather_tool")

        if any(
            marker in lowered
            for marker in (
                "generate image",
                "create image",
                "make an image",
                "make a picture",
                "draw ",
                "draw me",
                "draw",
                "visualize",
                "illustration",
                "concept art",
                "mockup",
            )
        ):
            forced.add("image_gen_tool")

        if any(
            marker in lowered
            for marker in (
                "run python",
                "execute python",
                "run this code",
                "execute this",
                "execute this code",
                "test this code",
                "test this snippet",
                "verify code output",
                "run this script",
            )
        ):
            forced.add("code_runner")

        if any(
            marker in lowered
            for marker in (
                "add task",
                "create task",
                "new task",
                "track this task",
                "todo",
                "to-do",
                "task list",
                "mark task",
                "complete task",
                "blocked task",
                "in progress task",
                "backlog",
            )
        ):
            forced.add("task_manager")

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
        if session_summary_request:
            forced.discard("memory_query")
        elif "memory_query" in forced:
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
            "execute it",
            "folder",
            "project",
            "test case",
            "test cases",
            "run it",
            "run with",
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
        if tool_trace:
            last = tool_trace[-1]
            formatted = self._format_tool_result(
                str(last.get("name", "")).strip(),
                last.get("result", {}),
                tool_trace,
            )
            if formatted:
                return formatted
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
            if tool_name == "system_info_tool" and isinstance(result.get("alerts"), list):
                alerts = [str(item.get("message", "")).strip() for item in result.get("alerts", []) if isinstance(item, dict)]
                if alerts:
                    return " ".join(alerts[:3])
                if result.get("all_clear"):
                    return "No active system alerts."
            if tool_name == "file_manager":
                items = result.get("items")
                if isinstance(items, list):
                    return f"I found {len(items)} items in {result.get('path', 'that folder')}."
            if tool_name == "task_manager":
                task = result.get("task")
                if isinstance(task, dict):
                    title = str(task.get("title", "Untitled")).strip() or "Untitled"
                    status = str(task.get("status", "updated")).strip() or "updated"
                    return f"Task '{title}' is now {status}."
                tasks = result.get("tasks")
                if isinstance(tasks, list):
                    count = len(tasks)
                    suffix = "s" if count != 1 else ""
                    return f"I found {count} active task{suffix}."
                deleted = str(result.get("deleted", "")).strip()
                if deleted:
                    return f"Deleted task {deleted}."
            if tool_name == "code_runner":
                stdout = str(result.get("stdout", "")).strip()
                stderr = str(result.get("stderr", "")).strip()
                if stdout:
                    return stdout
                if stderr:
                    return stderr
                if result.get("exit_code") == 0:
                    return "The code ran successfully with no output."
            if tool_name == "browser_control":
                return "The browser action ran, but I couldn't produce a clean final summary."
            if tool_name == "image_gen_tool" and (result.get("url") or result.get("path")):
                return "The image is ready. It will open in the viewer."
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
            "toolcall>",
            "toolcall>{",
            "memory_query result:",
            "tool result:",
            '"ok":',
            '"memories":',
            '"items":',
            '"query":',
            '"name": "memory_query"',
            "duplicate tool call skipped",
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
        return self._format_tool_result(tool_name, tool_result, tool_trace)

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
            return " ".join(cleaned[:3])

        items = result.get("items")
        if isinstance(items, list):
            cleaned_items = [str(item).strip() for item in items if str(item).strip()]
            count = int(result.get("stored", len(cleaned_items)) or len(cleaned_items))
            if cleaned_items:
                return " ".join(cleaned_items[:3])
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
