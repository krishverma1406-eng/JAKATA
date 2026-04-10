"""TTS service — disabled. Speech output has been removed."""

from __future__ import annotations


def speak_sync(text: str, settings=None) -> bool:  # noqa: ARG001
    return False


def synthesize_base64_sync(text: str, settings=None) -> str | None:  # noqa: ARG001
    return None


async def speak(text: str, settings=None) -> bool:  # noqa: ARG001
    return False


async def synthesize_base64(text: str, settings=None) -> str | None:  # noqa: ARG001
    return None


async def save_to_file(text: str, path, settings=None):  # noqa: ARG001
    raise RuntimeError("TTS is disabled.")
