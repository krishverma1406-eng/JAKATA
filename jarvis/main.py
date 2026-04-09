"""CLI entry point for JARVIS."""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace

from config.settings import SETTINGS
from core.agent import Agent
from services.github_auth import COPILOT_TOKEN_PATH, GITHUB_TOKEN_PATH, login_via_device_flow
from services.reminders import get_reminder_service


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


def _notify_reminder(agent: Agent, reminder: dict[str, object], print_lock: threading.Lock | None = None) -> None:
    text = str(reminder.get("text", "")).strip()
    due_at = str(reminder.get("due_at", "")).strip()
    message = f"JARVIS reminder: {text}"
    if due_at:
        message += f" (due {due_at})"
    if print_lock is None:
        print(message, flush=True)
    else:
        with print_lock:
            print(message, flush=True)
    _speak_in_background(agent, text)


def _initialize_reminders(agent: Agent, print_lock: threading.Lock | None = None) -> None:
    try:
        service = get_reminder_service(agent.settings)
        for reminder in service.pop_due_reminders():
            _notify_reminder(agent, reminder, print_lock=print_lock)
        service.start(lambda reminder: _notify_reminder(agent, reminder, print_lock=print_lock))
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS CLI")
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt")
    parser.add_argument("--github-login", action="store_true", help="Authorize GitHub/Copilot device flow")
    parser.add_argument("--voice", action="store_true", help="Enable push-to-talk in the interactive session")
    parser.add_argument("--wake", action="store_true", help="Enable wake-word listening in the interactive session")
    args = parser.parse_args()

    if args.github_login:
        try:
            token_data = login_via_device_flow()
        except Exception as exc:
            print(f"GitHub login failed: {exc}")
            return
        print(f"GitHub device token saved to {GITHUB_TOKEN_PATH}")
        print(f"Copilot token cache saved to {COPILOT_TOKEN_PATH}")
        print(f"Copilot base URL: {token_data.get('copilot_base_url', '')}")
        token_prefix = str(token_data.get("copilot_token", ""))[:4]
        if token_prefix:
            print(f"Token prefix: {token_prefix}")
        return

    session_settings = replace(
        SETTINGS,
        tts_enabled=SETTINGS.tts_enabled or args.voice or args.wake,
        stt_enabled=SETTINGS.stt_enabled or args.voice or args.wake,
        wake_word_enabled=SETTINGS.wake_word_enabled or args.wake,
    )
    agent = Agent(settings=session_settings)
    stream_state = {"printed": False}

    def _stream_to_stdout(chunk: str) -> None:
        if chunk:
            stream_state["printed"] = True
        print(chunk, end="", flush=True)

    prompt = " ".join(args.prompt).strip()
    if prompt:
        _initialize_reminders(agent)
        print("JARVIS: ", end="", flush=True)
        response = agent.run(prompt, stream_handler=_stream_to_stdout)
        if not stream_state["printed"] and response:
            print(response, end="", flush=True)
        if not response.endswith("\n"):
            print()
        _speak_in_background(agent, response)
        return

    _run_combined_mode(agent)


def _run_combined_mode(agent: Agent) -> None:
    from services.audio_feedback import play_activation_sound, play_response_sound
    from services.stt import SpeechToText
    from services.wake_word import WakeWordDetector

    print_lock = threading.Lock()
    process_lock = threading.Lock()
    stop_event = threading.Event()
    stt = SpeechToText(agent.settings)
    _initialize_reminders(agent, print_lock=print_lock)

    def _safe_print(message: str = "", *, end: str = "\n") -> None:
        with print_lock:
            print(message, end=end, flush=True)

    def _handle_prompt(prompt: str, source: str) -> None:
        if not prompt.strip():
            _safe_print("JARVIS: I didn't catch that.")
            return
        _safe_print(f"You: {prompt}")
        play_response_sound(agent.settings)
        _run_agent_stream(agent, prompt, print_lock=print_lock)

    def _run_push_to_talk() -> None:
        try:
            play_activation_sound(agent.settings)
            transcript = stt.listen_once()
        except Exception as exc:
            _safe_print(f"JARVIS: Voice capture failed: {exc}")
            return
        _handle_prompt(transcript, "voice")

    def _wake_loop() -> None:
        detector = WakeWordDetector(agent.settings)

        while not stop_event.is_set():
            try:
                detector.wait_for_wake_word()
            except Exception as exc:
                _safe_print(f"JARVIS: Wake word failed: {exc}")
                return
            if stop_event.is_set():
                return
            if not process_lock.acquire(blocking=False):
                continue
            try:
                play_activation_sound(agent.settings)
                _safe_print("JARVIS: Listening...")
                transcript = stt.listen_until_silence()
                if transcript.lower().strip() in {"goodbye jarvis", "exit jarvis", "stop jarvis"}:
                    _safe_print("JARVIS: Goodbye.")
                    stop_event.set()
                    return
                _handle_prompt(transcript, "wake")
            except Exception as exc:
                _safe_print(f"JARVIS: Voice capture failed: {exc}")
            finally:
                process_lock.release()

    feature_notes = ["typing"]
    if agent.settings.stt_enabled:
        feature_notes.append("empty Enter = push-to-talk")
    if agent.settings.wake_word_enabled:
        feature_notes.append("wake word in background")
    _safe_print(f"JARVIS is ready. Type 'exit' to quit. Modes: {', '.join(feature_notes)}.")

    wake_thread: threading.Thread | None = None
    if agent.settings.wake_word_enabled:
        wake_thread = threading.Thread(target=_wake_loop, daemon=True)
        wake_thread.start()

    while not stop_event.is_set():
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _safe_print("\nJARVIS: Shutting down.")
            stop_event.set()
            break

        if user_input.lower() in {"exit", "quit"}:
            _safe_print("JARVIS: Goodbye.")
            stop_event.set()
            break

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
            _run_agent_stream(agent, user_input, print_lock=print_lock)
        finally:
            process_lock.release()

    if wake_thread is not None and wake_thread.is_alive():
        stop_event.set()


def _run_agent_stream(agent: Agent, prompt: str, print_lock: threading.Lock | None = None) -> None:
    stream_state = {"printed": False}

    def _stream_to_stdout(chunk: str) -> None:
        if chunk:
            stream_state["printed"] = True
        if print_lock is None:
            print(chunk, end="", flush=True)
            return
        with print_lock:
            print(chunk, end="", flush=True)

    if print_lock is None:
        print("JARVIS: ", end="", flush=True)
    else:
        with print_lock:
            print("JARVIS: ", end="", flush=True)
    response = agent.run(prompt, stream_handler=_stream_to_stdout)
    if not stream_state["printed"] and response:
        if print_lock is None:
            print(response, end="", flush=True)
        else:
            with print_lock:
                print(response, end="", flush=True)
    if not response.endswith("\n"):
        if print_lock is None:
            print()
        else:
            with print_lock:
                print()
    _speak_in_background(agent, response)


if __name__ == "__main__":
    main()
