"""Text-to-speech service backed by Edge TTS."""

from __future__ import annotations

import asyncio
import base64
import random
import re
import tempfile
import threading
from pathlib import Path

from config.settings import SETTINGS, Settings

_MIN_SPOKEN_SENTENCES = 2
_MAX_SPOKEN_SENTENCES = 4
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")
_SCREEN_CLOSERS = (
    "The rest is on screen, sir.",
    "Check the result on screen, sir.",
    "The remaining details are on your screen, sir.",
    "You can read the rest on screen, sir.",
    "The full response is on screen, sir.",
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BREAK_RE.split(text) if part.strip()]


def _prepare_spoken_text(text: str) -> str:
    payload = _normalize_text(text)
    if not payload:
        return ""

    sentences = _split_sentences(payload)
    if len(sentences) <= _MIN_SPOKEN_SENTENCES:
        return payload

    spoken_count = random.randint(
        _MIN_SPOKEN_SENTENCES,
        min(_MAX_SPOKEN_SENTENCES, len(sentences)),
    )
    spoken_preview = " ".join(sentences[:spoken_count]).strip()
    if spoken_count >= len(sentences):
        return spoken_preview

    closer = random.choice(_SCREEN_CLOSERS)
    return f"{spoken_preview} {closer}".strip()


async def save_to_file(
    text: str,
    path: str | Path,
    settings: Settings | None = None,
) -> Path:
    payload = _normalize_text(text)
    if not payload:
        raise ValueError("Cannot synthesize empty text.")

    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("edge-tts is not installed.") from exc

    config = settings or SETTINGS
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    communicator = edge_tts.Communicate(
        payload,
        voice=config.tts_voice,
        rate=config.tts_rate,
        volume=config.tts_volume,
    )
    await communicator.save(str(target))
    return target


async def speak(text: str, settings: Settings | None = None) -> bool:
    config = settings or SETTINGS
    payload = _prepare_spoken_text(text)
    if not config.tts_enabled or not payload:
        return False

    try:
        from playsound import playsound
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("playsound is not installed.") from exc

    temp_path = Path(tempfile.gettempdir()) / f"jarvis_tts_{threading.get_ident()}.mp3"
    try:
        await save_to_file(payload, temp_path, config)
        playsound(str(temp_path))
        return True
    finally:
        temp_path.unlink(missing_ok=True)


async def synthesize_base64(text: str, settings: Settings | None = None) -> str | None:
    config = settings or SETTINGS
    payload = _prepare_spoken_text(text)
    if not config.tts_enabled or not payload:
        return None

    temp_path = Path(tempfile.gettempdir()) / f"jarvis_tts_b64_{threading.get_ident()}.mp3"
    try:
        await save_to_file(payload, temp_path, config)
        return base64.b64encode(temp_path.read_bytes()).decode("ascii")
    finally:
        temp_path.unlink(missing_ok=True)


def speak_sync(text: str, settings: Settings | None = None) -> bool:
    result: dict[str, bool] = {"spoken": False}
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result["spoken"] = asyncio.run(speak(text, settings))
        except BaseException as exc:  # pragma: no cover - passthrough wrapper
            error.append(exc)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join()

    if error:
        raise error[0]
    return result["spoken"]


def synthesize_base64_sync(text: str, settings: Settings | None = None) -> str | None:
    result: dict[str, str | None] = {"audio": None}
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result["audio"] = asyncio.run(synthesize_base64(text, settings))
        except BaseException as exc:  # pragma: no cover - passthrough wrapper
            error.append(exc)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join()

    if error:
        raise error[0]
    return result["audio"]
