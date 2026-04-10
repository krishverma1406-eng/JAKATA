"""CLI and web entry point for JARVIS."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config.settings import SETTINGS
from core.agent import Agent
from services.github_auth import COPILOT_TOKEN_PATH, GITHUB_TOKEN_PATH, login_via_device_flow
from services.reminders import get_reminder_service


CONSOLE = Console(highlight=False, soft_wrap=True)


def _speak_in_background(agent: Agent, response: str) -> None:
    if not agent.settings.tts_enabled or not response:
        return

    def _worker() -> None:
        try:
            from services.tts import speak_sync

            speak_sync(response, agent.settings)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


class CliRenderer:
    """Rich terminal rendering for JARVIS."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.console = CONSOLE

    def line(self, text: str = "") -> None:
        self.console.print(text)

    def error(self, message: str) -> None:
        self.console.print(Text(f"JARVIS error: {message}", style="bold red"))

    def info(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def tool(self, message: str) -> None:
        self.console.print(Text(message, style="yellow"))

    def assistant_prefix(self) -> None:
        self.console.print(Text("JARVIS: ", style="bold green"), end="")

    def assistant_chunk(self, chunk: str) -> None:
        self.console.print(Text(chunk, style="green"), end="")

    def user_echo(self, prompt: str) -> None:
        self.console.print(Text(f"You: {prompt}", style="bold cyan"))

    def memory_context(self, items: list[str]) -> None:
        if not items:
            return
        body = "\n".join(f"- {item}" for item in items[:5])
        self.console.print(Panel(body, title="Memory Context", border_style="grey50", style="grey50"))

    def session_banner(self) -> None:
        session_name = str(self.agent.session_meta.get("display_name", "Untitled session")).strip() or "Untitled session"
        mode_config = self.agent.interface.get_mode(self.agent.mode)
        mode_label = str(mode_config.get("label", self.agent.mode)).strip() or self.agent.mode.title()
        self.console.print(
            Panel(
                Text(f"{session_name}\nMode: {mode_label}", style="white"),
                title="Session",
                border_style="blue",
            )
        )

    def briefing(self, text: str) -> None:
        self.console.print(Panel(Text(text, style="white"), title="Briefing", border_style="cyan"))

    def modes_table(self, modes: list[dict[str, Any]]) -> None:
        table = Table(title="Available Modes", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Label", style="white")
        table.add_column("Summary", style="dim")
        for mode in modes:
            table.add_row(
                str(mode.get("key", "")),
                str(mode.get("label", "")),
                str(mode.get("summary", "")),
            )
        self.console.print(table)

    def sessions_table(self, sessions: list[dict[str, Any]], title: str = "Sessions") -> None:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Mode", style="white")
        table.add_column("Updated", style="dim")
        table.add_column("Turns", justify="right")
        for session in sessions:
            table.add_row(
                str(session.get("display_name", "")),
                str(session.get("mode", "")),
                str(session.get("updated_at", "")),
                str(session.get("turn_count", 0)),
            )
        self.console.print(table)

    def input(self) -> str:
        return self.console.input("[bold cyan]You:[/bold cyan] ")


def _notify_reminder(
    agent: Agent,
    reminder: dict[str, object],
    renderer: CliRenderer | None = None,
    print_lock: threading.Lock | None = None,
) -> None:
    text = str(reminder.get("text", "")).strip()
    due_at = str(reminder.get("due_at_local") or reminder.get("due_at", "")).strip()
    message = f"Reminder: {text}"
    if due_at:
        message += f" (due {due_at})"

    def _print() -> None:
        if renderer is None:
            CONSOLE.print(Text(message, style="bold magenta"))
        else:
            renderer.console.print(Text(message, style="bold magenta"))

    if print_lock is None:
        _print()
    else:
        with print_lock:
            _print()
    _speak_in_background(agent, text)


def _initialize_reminders(
    agent: Agent,
    renderer: CliRenderer | None = None,
    print_lock: threading.Lock | None = None,
) -> None:
    try:
        service = get_reminder_service(agent.settings)
        for reminder in service.pop_due_reminders():
            _notify_reminder(agent, reminder, renderer=renderer, print_lock=print_lock)
        service.start(lambda reminder: _notify_reminder(agent, reminder, renderer=renderer, print_lock=print_lock))
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS CLI")
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt")
    parser.add_argument("--github-login", action="store_true", help="Authorize GitHub/Copilot device flow")
    parser.add_argument("--voice", action="store_true", help="Enable push-to-talk in the interactive session")
    parser.add_argument("--wake", action="store_true", help="Enable wake-word listening in the interactive session")
    parser.add_argument("--web", action="store_true", help="Start the FastAPI frontend server")
    parser.add_argument("--mode", default="", help="Conversation mode key from data_ai/conversation_modes.json")
    args = parser.parse_args()

    if args.web:
        import uvicorn

        uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
        return

    if args.github_login:
        try:
            token_data = login_via_device_flow()
        except Exception as exc:
            CONSOLE.print(Text(f"GitHub login failed: {exc}", style="bold red"))
            return
        CONSOLE.print(Text(f"GitHub device token saved to {GITHUB_TOKEN_PATH}", style="green"))
        CONSOLE.print(Text(f"Copilot token cache saved to {COPILOT_TOKEN_PATH}", style="green"))
        CONSOLE.print(Text(f"Copilot base URL: {token_data.get('copilot_base_url', '')}", style="dim"))
        token_prefix = str(token_data.get("copilot_token", ""))[:4]
        if token_prefix:
            CONSOLE.print(Text(f"Token prefix: {token_prefix}", style="dim"))
        return

    session_settings = replace(
        SETTINGS,
        tts_enabled=SETTINGS.tts_enabled or args.voice or args.wake,
        stt_enabled=SETTINGS.stt_enabled or args.voice or args.wake,
        wake_word_enabled=SETTINGS.wake_word_enabled or args.wake,
    )
    agent = Agent(settings=session_settings)
    if args.mode.strip():
        agent.set_mode(args.mode.strip())
    renderer = CliRenderer(agent)

    prompt = " ".join(args.prompt).strip()
    if prompt:
        _initialize_reminders(agent, renderer=renderer)
        for startup in agent.startup_messages():
            renderer.briefing(startup)
        _run_agent_stream(agent, prompt, renderer=renderer)
        return

    _run_combined_mode(agent, renderer=renderer)


def _run_combined_mode(agent: Agent, renderer: CliRenderer) -> None:
    from services.audio_feedback import play_activation_sound, play_response_sound
    from services.stt import SpeechToText
    from services.wake_word import WakeWordDetector

    print_lock = threading.Lock()
    process_lock = threading.Lock()
    stop_event = threading.Event()
    stt = SpeechToText(agent.settings)
    _initialize_reminders(agent, renderer=renderer, print_lock=print_lock)

    def _safe_print(fn: callable) -> None:
        with print_lock:
            fn()

    def _handle_prompt(prompt: str) -> None:
        if not prompt.strip():
            _safe_print(lambda: renderer.info("JARVIS: I didn't catch that."))
            return
        _safe_print(lambda: renderer.user_echo(prompt))
        play_response_sound(agent.settings)
        _run_agent_stream(agent, prompt, renderer=renderer, print_lock=print_lock)

    def _run_push_to_talk() -> None:
        try:
            play_activation_sound(agent.settings)
            transcript = stt.listen_once()
        except KeyboardInterrupt:
            _safe_print(lambda: renderer.info("JARVIS: Voice capture cancelled."))
            return
        except Exception as exc:
            _safe_print(lambda: renderer.error(f"Voice capture failed: {exc}"))
            return
        _handle_prompt(transcript)

    def _wake_loop() -> None:
        detector = WakeWordDetector(agent.settings)

        while not stop_event.is_set():
            try:
                detector.wait_for_wake_word()
            except Exception as exc:
                _safe_print(lambda: renderer.error(f"Wake word failed: {exc}"))
                return
            if stop_event.is_set():
                return
            if not process_lock.acquire(blocking=False):
                continue
            try:
                play_activation_sound(agent.settings)
                _safe_print(lambda: renderer.info("JARVIS: Listening..."))
                transcript = stt.listen_until_silence()
                if transcript.lower().strip() in {"goodbye jarvis", "exit jarvis", "stop jarvis"}:
                    _safe_print(lambda: renderer.line("[green]JARVIS: Goodbye.[/green]"))
                    stop_event.set()
                    return
                _handle_prompt(transcript)
            except Exception as exc:
                _safe_print(lambda: renderer.error(f"Voice capture failed: {exc}"))
            finally:
                process_lock.release()

    renderer.session_banner()
    for startup in agent.startup_messages():
        renderer.briefing(startup)

    feature_notes = ["typing"]
    if agent.settings.stt_enabled:
        feature_notes.append("empty Enter = push-to-talk")
    if agent.settings.wake_word_enabled:
        feature_notes.append("wake word in background")
    renderer.info(f"JARVIS is ready. Type 'exit' to quit. Modes: {', '.join(feature_notes)}.")
    renderer.info("Commands: /modes, /mode <key>, /name <session name>, /session, /sessions [query], /briefing, /new")

    wake_thread: threading.Thread | None = None
    if agent.settings.wake_word_enabled:
        wake_thread = threading.Thread(target=_wake_loop, daemon=True)
        wake_thread.start()

    while not stop_event.is_set():
        try:
            user_input = renderer.input().strip()
        except (EOFError, KeyboardInterrupt):
            renderer.info("\nJARVIS: Shutting down.")
            stop_event.set()
            break

        if user_input.lower() in {"exit", "quit"}:
            renderer.line("[green]JARVIS: Goodbye.[/green]")
            stop_event.set()
            break

        if user_input.startswith("/"):
            handled = _handle_cli_command(agent, user_input, renderer)
            if handled == "quit":
                stop_event.set()
                break
            continue

        if not user_input:
            if not agent.settings.stt_enabled:
                continue
            if not process_lock.acquire(blocking=False):
                continue
            try:
                _run_push_to_talk()
            finally:
                process_lock.release()
            continue

        if not process_lock.acquire(blocking=False):
            continue
        try:
            _run_agent_stream(agent, user_input, renderer=renderer, print_lock=print_lock)
        finally:
            process_lock.release()

    if wake_thread is not None and wake_thread.is_alive():
        stop_event.set()


def _handle_cli_command(agent: Agent, command: str, renderer: CliRenderer) -> str | None:
    parts = command.strip().split(maxsplit=1)
    keyword = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if keyword == "/modes":
        renderer.modes_table(agent.interface.list_modes())
        return "handled"
    if keyword == "/mode":
        if not arg:
            renderer.error("Usage: /mode <mode-key>")
            return "handled"
        agent.set_mode(arg)
        renderer.session_banner()
        for startup in agent.startup_messages():
            renderer.briefing(startup)
        return "handled"
    if keyword == "/name":
        if not arg:
            renderer.error("Usage: /name <session name>")
            return "handled"
        agent.rename_session(arg)
        renderer.session_banner()
        return "handled"
    if keyword == "/session":
        renderer.session_banner()
        return "handled"
    if keyword == "/sessions":
        if arg:
            result = agent.memory.session_search(arg, limit=8)
            sessions = result.get("sessions", []) if isinstance(result, dict) else []
            renderer.sessions_table(sessions, title=f"Sessions matching: {arg}")
        else:
            renderer.sessions_table(agent.memory.list_sessions(limit=12))
        return "handled"
    if keyword == "/briefing":
        briefing = agent._build_briefing()
        if briefing:
            renderer.briefing(briefing)
        else:
            renderer.info("No briefing data is available.")
        return "handled"
    if keyword == "/new":
        current_mode = agent.mode
        agent.bind_session(mode=current_mode)
        renderer.session_banner()
        for startup in agent.startup_messages():
            renderer.briefing(startup)
        return "handled"
    if keyword == "/exit":
        renderer.line("[green]JARVIS: Goodbye.[/green]")
        return "quit"
    renderer.error("Unknown command.")
    return "handled"


def _run_agent_stream(
    agent: Agent,
    prompt: str,
    renderer: CliRenderer,
    print_lock: threading.Lock | None = None,
) -> None:
    stream_state = {"printed": False}
    mode_config = agent.interface.get_mode(agent.mode)

    def _safe(fn: callable) -> None:
        if print_lock is None:
            fn()
            return
        with print_lock:
            fn()

    def _stream_to_stdout(chunk: str) -> None:
        if chunk:
            stream_state["printed"] = True
        _safe(lambda: renderer.assistant_chunk(chunk))

    def _event_handler(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "tool_started":
            if not mode_config.get("show_debug"):
                return
            name = str(event.get("name", "")).strip()
            arguments = json.dumps(event.get("arguments", {}), ensure_ascii=False, default=str)
            _safe(lambda: renderer.tool(f"TOOL > {name} {arguments}"))
            return
        if event_type == "tool_result" and mode_config.get("show_debug"):
            name = str(event.get("name", "")).strip()
            result = event.get("result", {})
            status = "ok" if isinstance(result, dict) and result.get("ok", True) else "error"
            _safe(lambda: renderer.tool(f"TOOL < {name} [{status}]"))
            return
        if event_type != "activity" or not mode_config.get("show_debug"):
            return
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            return
        event_name = str(payload.get("event", "")).strip()
        message = str(payload.get("message", "")).strip()
        if event_name == "memory_context_loaded":
            items = [item.strip() for item in message.split(";") if item.strip()]
            _safe(lambda: renderer.memory_context(items))
            return
        if message:
            style = "dim"
            if event_name in {"plan_ready", "provider_selected"}:
                style = "yellow"
            if event_name == "focus_redirected":
                style = "bold yellow"
            _safe(lambda: renderer.console.print(Text(message, style=style)))

    _safe(renderer.assistant_prefix)
    response = agent.run(prompt, stream_handler=_stream_to_stdout, event_handler=_event_handler)
    if not stream_state["printed"] and response:
        _safe(lambda: renderer.assistant_chunk(response))
    if not response.endswith("\n"):
        _safe(lambda: renderer.console.print())

    turn_meta = agent.last_turn_meta
    if mode_config.get("show_debug") and turn_meta:
        debug_bits = []
        provider = str(turn_meta.get("provider", "")).strip()
        model = str(turn_meta.get("model", "")).strip()
        latency_ms = turn_meta.get("total_latency_ms")
        if provider:
            debug_bits.append(provider)
        if model:
            debug_bits.append(model)
        if latency_ms not in (None, ""):
            debug_bits.append(f"{latency_ms} ms")
        if debug_bits:
            _safe(lambda: renderer.console.print(Text(" | ".join(debug_bits), style="dim")))
    _speak_in_background(agent, response)


if __name__ == "__main__":
    main()
