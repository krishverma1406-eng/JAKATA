"""Small audio cues for activation and response events."""

from __future__ import annotations

from config.settings import SETTINGS, Settings


def play_activation_sound(settings: Settings | None = None) -> None:
    _play_tone_sequence(((880, 90), (1240, 120)), settings or SETTINGS)


def play_response_sound(settings: Settings | None = None) -> None:
    _play_tone_sequence(((660, 80), (880, 100)), settings or SETTINGS)


def _play_tone_sequence(sequence: tuple[tuple[int, int], ...], settings: Settings) -> None:
    if not settings.audio_feedback_enabled:
        return
    try:
        import winsound
    except ImportError:  # pragma: no cover - non-Windows fallback
        return

    for frequency, duration_ms in sequence:
        winsound.Beep(frequency, duration_ms)
