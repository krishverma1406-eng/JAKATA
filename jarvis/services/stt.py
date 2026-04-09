"""Local speech-to-text helpers powered by faster-whisper."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from config.settings import SETTINGS, Settings


class SpeechToText:
    """Record microphone audio and transcribe it with a local Whisper model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._model: Any | None = None

    def record_until_enter(self) -> np.ndarray:
        """Start recording immediately and stop when the user presses Enter."""

        if not self.settings.stt_enabled:
            raise RuntimeError("STT is disabled in settings.")

        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("sounddevice is not installed.") from exc

        chunks: list[np.ndarray] = []
        stop_event = threading.Event()

        def callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                return
            if stop_event.is_set():
                raise sd.CallbackStop()
            chunks.append(indata.copy())

        with sd.InputStream(
            samplerate=self.settings.stt_sample_rate,
            channels=1,
            dtype="int16",
            device=self._input_device(),
            callback=callback,
        ):
            input("Listening... press Enter again to stop recording.")
            stop_event.set()
            time.sleep(0.1)

        return self._combine_chunks(chunks)

    def record_until_silence(self) -> np.ndarray:
        """Record until speech ends and trailing silence is detected."""

        if not self.settings.stt_enabled:
            raise RuntimeError("STT is disabled in settings.")

        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("sounddevice is not installed.") from exc

        sample_rate = self.settings.stt_sample_rate
        block_size = int(sample_rate * 0.1)
        max_blocks = max(1, int(self.settings.stt_max_record_seconds * 10))
        silence_blocks_needed = max(1, int(self.settings.stt_silence_duration_seconds * 10))
        speech_detected = False
        silent_blocks = 0
        blocks: list[np.ndarray] = []

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_size,
            device=self._input_device(),
        ) as stream:
            for _ in range(max_blocks):
                block, overflowed = stream.read(block_size)
                if overflowed:
                    continue
                mono = np.squeeze(block.copy())
                blocks.append(mono)
                rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2))) if mono.size else 0.0
                if rms >= self.settings.stt_silence_threshold:
                    speech_detected = True
                    silent_blocks = 0
                elif speech_detected:
                    silent_blocks += 1
                    if silent_blocks >= silence_blocks_needed:
                        break

        return self._combine_chunks(blocks)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        model = self._get_model()
        normalized_audio = audio.astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            normalized_audio,
            beam_size=1,
            vad_filter=True,
            language="en",
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

    def listen_once(self) -> str:
        return self.transcribe(self.record_until_enter())

    def listen_until_silence(self) -> str:
        return self.transcribe(self.record_until_silence())

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("faster-whisper is not installed.") from exc

        self._model = WhisperModel(
            self.settings.stt_model,
            device="cpu",
            compute_type="int8",
        )
        return self._model

    def _combine_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        flattened = [np.squeeze(chunk).astype(np.int16) for chunk in chunks if np.size(chunk)]
        if not flattened:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(flattened)

    def _input_device(self) -> int | str | None:
        raw = self.settings.stt_input_device.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return raw
