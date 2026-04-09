"""LLM routing and provider adapters for JARVIS."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests

from config.settings import DATA_AI_DIR, SETTINGS, Settings


class Brain:
    """Route chat requests through live providers and normalize the response."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._prompt_files = (
            DATA_AI_DIR / "system_prompt.md",
            DATA_AI_DIR / "behavior_rules.md",
            DATA_AI_DIR / "tool_guidelines.md",
            DATA_AI_DIR / "capabilities.md",
        )
        self._prompt_cache_value = ""
        self._prompt_cache_state: tuple[tuple[str, int], ...] = ()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_kind: str = "simple",
        response_format: str | None = None,
        system_override: str | None = None,
        stream_handler: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Call live providers and normalize their response."""

        system_prompt, compiled_messages = self._compile_messages(
            messages,
            system_override,
            include_tooling=bool(tools),
        )

        heuristic_response = self._offline_tool_response(compiled_messages, tools or [])
        if heuristic_response is not None:
            return heuristic_response

        provider_errors: list[str] = []
        try:
            return self._call_nvidia(
                system_prompt,
                compiled_messages,
                tools or [],
                task_kind,
                response_format,
                stream_handler,
            )
        except Exception as exc:  # pragma: no cover - depends on auth/network
            provider_errors.append(f"nvidia: {exc}")

        try:
            return self._call_openrouter(
                system_prompt,
                compiled_messages,
                tools or [],
                task_kind,
                response_format,
                stream_handler,
            )
        except Exception as exc:  # pragma: no cover - depends on auth/network
            provider_errors.append(f"openrouter: {exc}")

        return self._offline_response(compiled_messages, tools or [], provider_errors)

    def build_system_prompt(self) -> str:
        cache_state = self._prompt_cache_key()
        if cache_state == self._prompt_cache_state and self._prompt_cache_value:
            return self._prompt_cache_value

        sections: list[str] = []
        for prompt_file in self._prompt_files[:2]:
            if not prompt_file.exists():
                continue
            text = prompt_file.read_text(encoding="utf-8").strip()
            if text:
                sections.append(text)
        self._prompt_cache_value = "\n\n".join(sections).strip()
        self._prompt_cache_state = cache_state
        return self._prompt_cache_value

    def _compile_messages(
        self,
        messages: list[dict[str, Any]],
        system_override: str | None,
        include_tooling: bool,
    ) -> tuple[str, list[dict[str, Any]]]:
        include_tooling = include_tooling or any(message.get("role") == "tool" for message in messages)
        system_parts = [system_override or self.build_system_prompt_for_context(include_tooling)]
        compiled_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                system_parts.append(self._stringify_content(message.get("content", "")))
                continue

            normalized = dict(message)
            normalized["content"] = self._stringify_content(message.get("content", ""))
            compiled_messages.append(normalized)

        system_prompt = "\n\n".join(part for part in system_parts if part).strip()
        return system_prompt, compiled_messages

    def build_system_prompt_for_context(self, include_tooling: bool) -> str:
        if not include_tooling:
            return self.build_system_prompt()

        sections: list[str] = []
        for prompt_file in self._prompt_files:
            if not prompt_file.exists():
                continue
            text = prompt_file.read_text(encoding="utf-8").strip()
            if text:
                sections.append(text)
        return "\n\n".join(sections).strip()

    def _call_nvidia(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_kind: str,
        response_format: str | None,
        stream_handler: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        api_key = self.settings.nvidia_api_key.strip()
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set.")

        payload: dict[str, Any] = {
            "model": self._nvidia_model_for(task_kind),
            "messages": self._to_openai_messages(system_prompt, messages),
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 4096,
            "chat_template_kwargs": {"thinking": False},
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool_definition}
                for tool_definition in tools
            ]
            payload["tool_choice"] = "auto"
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions"
        if stream_handler is not None:
            streamed = self._post_json_stream_requests(url, payload, headers, self.settings.nvidia_timeout_seconds, stream_handler)
            return {
                "provider": "nvidia",
                "model": streamed.get("model") or payload["model"],
                "content": self._stringify_content(streamed.get("content", "")),
                "tool_calls": streamed.get("tool_calls", []),
                "raw": streamed.get("raw", {}),
            }

        raw = self._post_json_requests(url, payload, headers, self.settings.nvidia_timeout_seconds)
        message = raw["choices"][0]["message"]
        tool_calls = [
            {
                "id": call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": call["function"]["name"],
                "arguments": self._safe_json(call["function"].get("arguments", "{}")),
            }
            for call in message.get("tool_calls", [])
        ]
        content = self._stringify_content(message.get("content", ""))
        embedded_tool_calls, cleaned_content = self._extract_embedded_tool_calls(content)
        if embedded_tool_calls:
            tool_calls.extend(embedded_tool_calls)
            content = cleaned_content
        return {
            "provider": "nvidia",
            "model": raw.get("model", payload["model"]),
            "content": content,
            "tool_calls": tool_calls,
            "raw": raw,
        }

    def _call_openrouter(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_kind: str,
        response_format: str | None,
        stream_handler: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        api_key = self.settings.openrouter_api_key.strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        payload: dict[str, Any] = {
            "model": self._openrouter_model_for(task_kind),
            "messages": self._to_openai_messages(system_prompt, messages),
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool_definition}
                for tool_definition in tools
            ]
            payload["tool_choice"] = "auto"
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/akyourowngames/JAKATA",
            "X-OpenRouter-Title": "JARVIS",
        }
        return self._call_openai_compatible(
            url=f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers=headers,
            timeout=self.settings.openrouter_timeout_seconds,
            provider="openrouter",
            stream_handler=stream_handler,
        )

    def _call_openai_compatible(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
        provider: str,
        stream_handler: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if stream_handler is not None:
            streamed = self._post_json_stream(url, payload, headers, timeout, stream_handler)
            return {
                "provider": provider,
                "model": streamed.get("model") or payload["model"],
                "content": self._stringify_content(streamed.get("content", "")),
                "tool_calls": streamed.get("tool_calls", []),
                "raw": streamed.get("raw", {}),
            }

        raw = self._post_json(url, payload, headers, timeout)
        message = raw["choices"][0]["message"]
        tool_calls = [
            {
                "id": call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": call["function"]["name"],
                "arguments": self._safe_json(call["function"].get("arguments", "{}")),
            }
            for call in message.get("tool_calls", [])
        ]
        content = self._stringify_content(message.get("content", ""))
        embedded_tool_calls, cleaned_content = self._extract_embedded_tool_calls(content)
        if embedded_tool_calls:
            tool_calls.extend(embedded_tool_calls)
            content = cleaned_content
        return {
            "provider": provider,
            "model": raw.get("model", payload["model"]),
            "content": content,
            "tool_calls": tool_calls,
            "raw": raw,
        }

    def _offline_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        last_message = messages[-1] if messages else {}
        if last_message.get("role") == "tool":
            tool_name = last_message.get("name", "tool")
            tool_payload = self._safe_json(last_message.get("content", ""))
            return {
                "provider": "offline",
                "model": "offline-router",
                "content": self._summarize_tool_result(tool_name, tool_payload),
                "tool_calls": [],
                "raw": {"reason": "offline tool summary"},
            }

        heuristic_response = self._offline_tool_response(messages, tools)
        if heuristic_response is not None:
            return heuristic_response

        content = "Live providers failed, so JARVIS is running in offline mode."
        error_note = f" Provider errors: {' | '.join(errors)}" if errors else ""
        return {
            "provider": "offline",
            "model": "offline-router",
            "content": f"{content}{error_note}",
            "tool_calls": [],
            "raw": {"errors": errors},
        }

    def _offline_tool_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        last_message = messages[-1] if messages else {}
        if last_message.get("role") == "tool":
            return None

        last_user_message = next(
            (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        message_lower = str(last_user_message).lower()
        tool_names = {tool["name"] for tool in tools}

        def build_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {
                "provider": "offline",
                "model": "offline-router",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": name,
                        "arguments": arguments,
                    }
                ],
                "raw": {"reason": "offline heuristic"},
            }

        if "reminder_tool" in tool_names:
            reminder_args = self._parse_reminder_request(last_user_message)
            if reminder_args is not None:
                return build_tool_call("reminder_tool", reminder_args)
        if "calculator_tool" in tool_names:
            calculator_args = self._parse_calculator_request(last_user_message)
            if calculator_args is not None:
                return build_tool_call("calculator_tool", calculator_args)
        if "system_info_tool" in tool_names:
            system_action = self._parse_system_info_request(last_user_message)
            if system_action is not None:
                return build_tool_call("system_info_tool", {"action": system_action})
        if "weather_tool" in tool_names:
            weather_args = self._parse_weather_request(last_user_message)
            if weather_args is not None:
                return build_tool_call("weather_tool", weather_args)
        if "gmail_tool" in tool_names:
            gmail_args = self._parse_gmail_request(last_user_message)
            if gmail_args is not None:
                return build_tool_call("gmail_tool", gmail_args)
        if "app_launcher_tool" in tool_names:
            launcher_args = self._parse_app_launcher_request(last_user_message)
            if launcher_args is not None:
                return build_tool_call("app_launcher_tool", launcher_args)
        if "screenshot_tool" in tool_names:
            screenshot_args = self._parse_screenshot_request(last_user_message)
            if screenshot_args is not None:
                return build_tool_call("screenshot_tool", screenshot_args)
        if "clipboard_tool" in tool_names:
            clipboard_args = self._parse_clipboard_request(last_user_message)
            if clipboard_args is not None:
                return build_tool_call("clipboard_tool", clipboard_args)
        if "calendar_tool" in tool_names:
            calendar_args = self._parse_calendar_request(last_user_message)
            if calendar_args is not None:
                return build_tool_call("calendar_tool", calendar_args)
        if "datetime_tool" in tool_names and any(word in message_lower for word in ("time", "date", "day")):
            return build_tool_call("datetime_tool", {})
        if "file_manager" in tool_names and any(
            word in message_lower for word in ("read file", "open file", "save", "write", "find file", "list files")
        ):
            return build_tool_call("file_manager", {"action": "list", "path": "."})
        if "memory_query" in tool_names and any(
            word in message_lower
            for word in (
                "remember",
                "my",
                "project",
                "preference",
                "who am i",
                "about me",
                "know about me",
                "github username",
                "school do i",
                "which class",
                "grade am i",
            )
        ):
            return build_tool_call("memory_query", {"query": last_user_message, "limit": 5})
        return None

    def _to_openai_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message.get("role", "user")
            if role == "assistant" and message.get("tool_calls"):
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(tool_call.get("arguments", {})),
                                },
                            }
                            for tool_call in message.get("tool_calls", [])
                        ],
                    }
                )
                continue

            openai_message = {"role": role, "content": message.get("content", "")}
            if role == "tool":
                openai_message["tool_call_id"] = message.get("tool_call_id")
                openai_message["name"] = message.get("name")
            openai_messages.append(openai_message)

        return openai_messages

    def _openrouter_model_for(self, task_kind: str) -> str:
        if task_kind == "memory":
            return self.settings.openrouter_memory_model
        if task_kind == "simple":
            return self.settings.openrouter_simple_model
        if task_kind == "code":
            return self.settings.openrouter_code_model
        return self.settings.openrouter_complex_model

    def _nvidia_model_for(self, task_kind: str) -> str:
        if task_kind == "memory":
            return self.settings.nvidia_memory_model
        if task_kind == "simple":
            return self.settings.nvidia_simple_model
        if task_kind == "code":
            return self.settings.nvidia_code_model
        return self.settings.nvidia_complex_model

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - depends on network/provider
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - depends on network/provider
            raise RuntimeError(str(exc)) from exc

    def _post_json_stream(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
        stream_handler: Callable[[str], None],
    ) -> dict[str, Any]:
        streamed_payload = dict(payload)
        streamed_payload["stream"] = True
        request = Request(
            url=url,
            data=json.dumps(streamed_payload).encode("utf-8"),
            headers={**headers, "Accept": "text/event-stream"},
            method="POST",
        )
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        model = streamed_payload.get("model")
        raw_chunks: list[dict[str, Any]] = []

        try:
            with urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    raw_chunks.append(chunk)
                    model = chunk.get("model", model)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content_delta = delta.get("content")
                    if isinstance(content_delta, str) and content_delta:
                        content_parts.append(content_delta)
                        stream_handler(content_delta)
                    for tool_delta in delta.get("tool_calls", []) or []:
                        index = int(tool_delta.get("index", 0) or 0)
                        entry = tool_calls_by_index.setdefault(
                            index,
                            {
                                "id": "",
                                "name": "",
                                "arguments_text": "",
                            },
                        )
                        if tool_delta.get("id"):
                            entry["id"] = tool_delta["id"]
                        function_delta = tool_delta.get("function") or {}
                        if function_delta.get("name"):
                            entry["name"] = function_delta["name"]
                        if function_delta.get("arguments"):
                            entry["arguments_text"] += function_delta["arguments"]
        except HTTPError as exc:  # pragma: no cover - depends on network/provider
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - depends on network/provider
            raise RuntimeError(str(exc)) from exc

    def _post_json_requests(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:  # pragma: no cover - depends on network/provider
            body = exc.response.text if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(f"HTTP {status}: {body}") from exc
        except requests.RequestException as exc:  # pragma: no cover - depends on network/provider
            raise RuntimeError(str(exc)) from exc

    def _post_json_stream_requests(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
        stream_handler: Callable[[str], None],
    ) -> dict[str, Any]:
        streamed_payload = dict(payload)
        streamed_payload["stream"] = True
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        model = streamed_payload.get("model")
        raw_chunks: list[dict[str, Any]] = []

        try:
            with requests.post(
                url,
                headers={**headers, "Accept": "text/event-stream"},
                json=streamed_payload,
                timeout=timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    raw_chunks.append(chunk)
                    model = chunk.get("model", model)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content_delta = delta.get("content")
                    if isinstance(content_delta, str) and content_delta:
                        content_parts.append(content_delta)
                        stream_handler(content_delta)
                    for tool_delta in delta.get("tool_calls", []) or []:
                        index = int(tool_delta.get("index", 0) or 0)
                        entry = tool_calls_by_index.setdefault(
                            index,
                            {
                                "id": "",
                                "name": "",
                                "arguments_text": "",
                            },
                        )
                        if tool_delta.get("id"):
                            entry["id"] = tool_delta["id"]
                        function_delta = tool_delta.get("function") or {}
                        if function_delta.get("name"):
                            entry["name"] = function_delta["name"]
                        if function_delta.get("arguments"):
                            entry["arguments_text"] += function_delta["arguments"]
        except requests.HTTPError as exc:  # pragma: no cover - depends on network/provider
            body = exc.response.text if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(f"HTTP {status}: {body}") from exc
        except requests.RequestException as exc:  # pragma: no cover - depends on network/provider
            raise RuntimeError(str(exc)) from exc

        tool_calls = [
            {
                "id": tool_call["id"] or f"call_{uuid.uuid4().hex[:8]}",
                "name": tool_call["name"],
                "arguments": self._safe_json(tool_call["arguments_text"] or "{}"),
            }
            for _, tool_call in sorted(tool_calls_by_index.items())
            if tool_call["name"]
        ]
        content = "".join(content_parts)
        embedded_tool_calls, cleaned_content = self._extract_embedded_tool_calls(content)
        if embedded_tool_calls:
            tool_calls.extend(embedded_tool_calls)
            content = cleaned_content
        return {
            "model": model,
            "content": content,
            "tool_calls": tool_calls,
            "raw": {"chunks": raw_chunks},
        }

        tool_calls = [
            {
                "id": tool_call["id"] or f"call_{uuid.uuid4().hex[:8]}",
                "name": tool_call["name"],
                "arguments": self._safe_json(tool_call["arguments_text"] or "{}"),
            }
            for _, tool_call in sorted(tool_calls_by_index.items())
            if tool_call["name"]
        ]
        content = "".join(content_parts)
        embedded_tool_calls, cleaned_content = self._extract_embedded_tool_calls(content)
        if embedded_tool_calls:
            tool_calls.extend(embedded_tool_calls)
            content = cleaned_content
        return {
            "model": model,
            "content": content,
            "tool_calls": tool_calls,
            "raw": {"chunks": raw_chunks},
        }

    def _safe_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_embedded_tool_calls(self, content: str) -> tuple[list[dict[str, Any]], str]:
        if not content:
            return [], content

        patterns = (
            r"<TOOLCALL>\s*(.*?)\s*</TOOLCALL>",
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            r"```tool_call\s*(.*?)\s*```",
        )
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            parsed = self._safe_json(match.group(1))
            tool_calls = self._normalize_embedded_tool_calls(parsed)
            if not tool_calls:
                continue
            cleaned = (content[: match.start()] + content[match.end() :]).strip()
            return tool_calls, cleaned
        return [], content

    def _normalize_embedded_tool_calls(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("name")
                or item.get("tool_name")
                or item.get("function")
                or item.get("tool")
            )
            arguments = (
                item.get("arguments")
                or item.get("args")
                or item.get("parameters")
                or {}
            )
            if isinstance(arguments, str):
                parsed_arguments = self._safe_json(arguments)
                if isinstance(parsed_arguments, dict):
                    arguments = parsed_arguments
            if not isinstance(arguments, dict):
                arguments = {}
            if not name:
                continue
            normalized.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": str(name),
                    "arguments": arguments,
                }
            )
        return normalized

    def _parse_reminder_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower().strip()
        if "reminder" not in lowered and "remind me" not in lowered:
            return None
        if any(word in lowered for word in ("list reminder", "show reminder", "my reminder", "pending reminder")):
            return {"action": "list"}
        match = re.search(
            r"(?:set\s+a\s+reminder|remind me)(?:\s+in)?\s+(\d+)\s+(second|seconds|minute|minutes|hour|hours)\s+(?:to|about)?\s*(.+)",
            lowered,
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            text = match.group(3).strip(" .")
            args: dict[str, Any] = {"action": "create", "text": text}
            if unit.startswith("second"):
                args["delay_seconds"] = amount
            elif unit.startswith("minute"):
                args["delay_minutes"] = amount
            elif unit.startswith("hour"):
                args["delay_minutes"] = amount * 60
            return args
        return None

    def _parse_calculator_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower().strip()
        prefixes = ("calculate ", "solve ", "what is ", "convert ")
        expression = message.strip()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                expression = message[len(prefix):].strip(" ?")
                break
        if re.search(r"\d", expression) and any(token in expression for token in ("+", "-", "*", "/", "^", " to ", " in ", "sqrt", "sin", "cos", "tan", "log")):
            return {"expression": expression.replace("^", "**")}
        return None

    def _parse_system_info_request(self, message: str) -> str | None:
        lowered = message.lower()
        if any(word in lowered for word in ("cpu", "ram", "system", "battery", "disk")):
            if "process" in lowered:
                return "processes"
            if "battery" in lowered:
                return "battery"
            if "disk" in lowered:
                return "disk"
            return "overview"
        return None

    def _parse_weather_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower()
        if "weather" not in lowered and "forecast" not in lowered and "temperature" not in lowered:
            return None
        action = "forecast" if "forecast" in lowered else "current"
        location_match = re.search(r"\b(?:in|for)\s+([A-Za-z ,]+)$", message.strip(), flags=re.IGNORECASE)
        args: dict[str, Any] = {"action": action}
        if location_match:
            args["location"] = location_match.group(1).strip(" .")
        return args

    def _parse_gmail_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower().strip()
        if "email" not in lowered and "gmail" not in lowered and "inbox" not in lowered and "mail" not in lowered:
            return None

        send_match = re.search(
            r"send an? email to\s+([^\s]+@[^\s]+)\s+with subject\s+(.+?)\s+and body\s+(.+)$",
            message.strip(),
            flags=re.IGNORECASE,
        )
        if send_match:
            return {
                "action": "send",
                "to": send_match.group(1).strip(" ."),
                "subject": send_match.group(2).strip(" ."),
                "body": send_match.group(3).strip(),
            }
        if "unread" in lowered:
            return {"action": "unread", "max_results": 5}
        search_match = re.search(r"(?:search|find).*(?:email|gmail|inbox).*(?:for)\s+(.+)$", message.strip(), flags=re.IGNORECASE)
        if search_match:
            return {"action": "search", "query": search_match.group(1).strip(" ."), "max_results": 10}
        return None

    def _parse_app_launcher_request(self, message: str) -> dict[str, Any] | None:
        cleaned = message.strip()
        match = re.match(r"^(?:open|launch|start)\s+(.+?)[.!?]?\s*$", cleaned, flags=re.IGNORECASE)
        if not match:
            return None
        target = match.group(1).strip()
        lowered_target = target.lower()
        if any(token in lowered_target for token in ("http://", "https://", "www.", ".com", "browser", "website", "page", "file ")):
            return None
        return {"target": target}

    def _parse_screenshot_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower()
        if "screenshot" not in lowered and "screen" not in lowered:
            return None
        if any(marker in lowered for marker in ("what is visible", "what's visible", "what is on my screen", "what's on my screen", "describe", "analyze", "tell me what is visible")):
            return {"action": "analyze", "prompt": "Describe what is visible on this screen."}
        if any(marker in lowered for marker in ("take a screenshot", "capture the screen", "capture screenshot")):
            return {"action": "capture"}
        return None

    def _parse_clipboard_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower().strip()
        if "clipboard" not in lowered:
            return None
        if any(marker in lowered for marker in ("clear", "empty")):
            return {"action": "clear"}
        write_match = re.search(r"(?:copy|write).*(?:clipboard)[: ]+(.+)$", message.strip(), flags=re.IGNORECASE)
        if write_match:
            return {"action": "write", "text": write_match.group(1).strip()}
        if any(marker in lowered for marker in ("read", "summarize", "what is in", "what's in")):
            return {"action": "read"}
        return None

    def _parse_calendar_request(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower()
        if "calendar" not in lowered and "event" not in lowered:
            return None
        if any(marker in lowered for marker in ("today", "what's on my calendar", "what is on my calendar", "next events")):
            return {"action": "today", "max_results": 10}
        return None

    def _summarize_tool_result(self, tool_name: str, payload: Any) -> str:
        data = payload if isinstance(payload, dict) else {}
        if not isinstance(data, dict):
            return f"{tool_name} result: {self._stringify_content(payload)}"

        if data.get("ok") is False:
            error = str(data.get("error", "")).strip() or self._stringify_content(data)
            return f"{tool_name} failed: {error}"

        if tool_name == "calculator_tool":
            expression = str(data.get("expression", "")).strip()
            result = data.get("result")
            unit = str(data.get("unit", "")).strip()
            if expression and result is not None:
                suffix = f" {unit}" if unit else ""
                return f"{expression} = {result}{suffix}".strip()

        if tool_name == "app_launcher_tool":
            resolved = str(data.get("resolved", "")).strip()
            opened = str(data.get("opened", "")).strip() or "the app"
            if resolved:
                return f"I opened {opened}."
            return f"I opened {opened}."

        if tool_name == "gmail_tool":
            if data.get("sent"):
                sent = data.get("sent") if isinstance(data.get("sent"), dict) else {}
                to = str(sent.get("to", "")).strip()
                subject = str(sent.get("subject", "")).strip()
                if to and subject:
                    return f"I sent the email to {to} with subject '{subject}'."
                return "The email was sent."
            summary_lines = data.get("summary_lines")
            if isinstance(summary_lines, list) and summary_lines:
                lines = [str(item).strip() for item in summary_lines[:5] if str(item).strip()]
                if lines:
                    return "Here are the most relevant emails:\n- " + "\n- ".join(lines)
            messages = data.get("messages")
            if isinstance(messages, list) and not messages:
                return "I couldn't find any matching emails."

        if tool_name == "calendar_tool":
            summary_lines = data.get("summary_lines")
            if isinstance(summary_lines, list) and summary_lines:
                lines = [str(item).strip() for item in summary_lines[:5] if str(item).strip()]
                if lines:
                    return "Here are your calendar events:\n- " + "\n- ".join(lines)
            if isinstance(data.get("events"), list) and not data.get("events"):
                return "You have no calendar events for today."
            if data.get("event"):
                event = data.get("event") if isinstance(data.get("event"), dict) else {}
                summary = str(event.get("summary", "")).strip() or "the event"
                return f"I created the calendar event '{summary}'."

        if tool_name == "screenshot_tool":
            path = str(data.get("path", "")).strip()
            analysis = str(data.get("analysis", "")).strip()
            if analysis:
                return analysis
            if path:
                return f"I saved the screenshot to {path}."

        if tool_name == "clipboard_tool":
            if "text" in data:
                text = str(data.get("text", "")).strip()
                if text:
                    return f"Your clipboard contains: {text}"
                return "Your clipboard is empty."
            if data.get("copied"):
                return "I copied the text to your clipboard."
            if data.get("cleared"):
                return "I cleared your clipboard."

        if tool_name == "reminder_tool":
            reminder = data.get("reminder") if isinstance(data.get("reminder"), dict) else {}
            if reminder:
                text = str(reminder.get("text", "")).strip()
                due_at = str(reminder.get("due_at", "")).strip()
                if text and due_at:
                    return f"Reminder set for {due_at}: {text}"
            reminders = data.get("reminders")
            if isinstance(reminders, list):
                if not reminders:
                    return "You have no pending reminders."
                lines = []
                for item in reminders[:5]:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    due_at = str(item.get("due_at", "")).strip()
                    if text and due_at:
                        lines.append(f"{text} | due {due_at}")
                if lines:
                    return "Here are your reminders:\n- " + "\n- ".join(lines)

        if tool_name == "system_info_tool":
            if all(key in data for key in ("cpu_percent", "ram_percent", "disk_percent")):
                cpu = data.get("cpu_percent")
                ram = data.get("ram_percent")
                disk = data.get("disk_percent")
                battery = data.get("battery_percent")
                battery_text = f", battery {battery}%" if battery is not None else ""
                return f"System status: CPU {cpu}%, RAM {ram}%, disk {disk}%{battery_text}."

        return f"{tool_name} result: {self._stringify_content(payload)}"

    def _prompt_cache_key(self) -> tuple[tuple[str, int], ...]:
        state: list[tuple[str, int]] = []
        for prompt_file in self._prompt_files:
            if not prompt_file.exists():
                continue
            stat = prompt_file.stat()
            state.append((str(prompt_file), stat.st_mtime_ns))
        return tuple(state)

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=True, default=str)
        return str(content)
