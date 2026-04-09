"""Wake-word detection backed by openWakeWord."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import SETTINGS, Settings


class WakeWordDetector:
    """Block until the configured wake word is detected on the microphone."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        self._model: Any | None = None

    def wait_for_wake_word(self) -> None:
        if not self.settings.wake_word_enabled:
            raise RuntimeError("Wake word is disabled in settings.")

        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("sounddevice is not installed.") from exc

        model = self._get_model()
        frame_length = 1280  # 80 ms of 16 kHz audio, recommended by openWakeWord
        model_key = self._prediction_model_key(model)
        threshold = self.settings.wake_word_threshold
        input_device = self._input_device(sd)

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=frame_length,
            dtype="int16",
            channels=1,
            device=input_device,
        ) as stream:
            while True:
                frame, overflowed = stream.read(frame_length)
                if overflowed:
                    continue
                pcm = np.frombuffer(frame, dtype=np.int16)
                scores = model.predict(pcm)
                score = float(scores.get(model_key, 0.0))
                if score >= threshold:
                    return

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError("openwakeword is not installed.") from exc

        try:
            explicit_model_path = self._resolve_model_path()
            inference_framework, explicit_model_path = self._resolve_runtime_model_path(openwakeword, explicit_model_path)
            self._model = Model(
                wakeword_models=[str(explicit_model_path)],
                inference_framework=inference_framework,
                vad_threshold=self.settings.wake_word_vad_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"openWakeWord initialization failed: {exc}") from exc
        return self._model

    def _wakeword_name(self) -> str:
        keyword = self.settings.wake_word_keyword.strip().lower().replace("_", " ")
        if keyword in {"jarvis", "hey jarvis"}:
            return "hey jarvis"
        if keyword in {"alexa", "hey mycroft", "mycroft", "hey rhasspy", "rhasspy"}:
            return keyword if keyword.startswith("hey ") else f"hey {keyword}"
        return keyword

    def _model_key(self) -> str:
        return self._wakeword_name().replace(" ", "_")

    def _download_name(self) -> str:
        return f"{self._model_key()}_v0.1"

    def _resolve_model_path(self) -> Path:
        configured = self.settings.wake_word_model_path.strip()
        if configured:
            return Path(configured).expanduser()

        workspace_candidate = Path(__file__).resolve().parents[2] / f"{self._download_name()}.tflite"
        if workspace_candidate.exists():
            return workspace_candidate

        return self._downloaded_model_path(self._download_name(), "onnx")

    def _downloaded_model_path(self, model_name: str, inference_framework: str) -> Path:
        import openwakeword

        extension = ".onnx" if inference_framework == "onnx" else ".tflite"
        models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        return models_dir / f"{model_name}{extension}"

    def _prediction_model_key(self, model: Any) -> str:
        keys = list(getattr(model, "models", {}).keys())
        if not keys:
            return self._model_key()

        preferred = self._model_key()
        for key in keys:
            if key == preferred:
                return key
        for key in keys:
            if key.startswith(preferred):
                return key
        return keys[0]

    def _resolve_runtime_model_path(self, openwakeword: Any, model_path: Path) -> tuple[str, Path]:
        suffix = model_path.suffix.lower()
        if suffix == ".onnx":
            if not model_path.exists():
                model_name = model_path.stem
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    openwakeword.utils.download_models([model_name])
            return "onnx", model_path

        if suffix == ".tflite":
            if self._has_tflite_runtime():
                return "tflite", model_path

            onnx_path = model_path.with_suffix(".onnx")
            if not onnx_path.exists():
                model_name = self._download_name()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    openwakeword.utils.download_models([model_name])
                downloaded = self._downloaded_model_path(model_name, "onnx")
                if downloaded.exists():
                    return "onnx", downloaded
            return "onnx", onnx_path if onnx_path.exists() else self._downloaded_model_path(self._download_name(), "onnx")

        model_name = self._download_name()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            openwakeword.utils.download_models([model_name])
        return "onnx", self._downloaded_model_path(model_name, "onnx")

    def _has_tflite_runtime(self) -> bool:
        try:
            import tflite_runtime.interpreter  # type: ignore # pragma: no cover
        except ImportError:
            return False
        return True

    def _input_device(self, sounddevice_module: Any) -> int | str:
        raw = self.settings.wake_word_input_device.strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                return raw
        default_device = sounddevice_module.default.device
        if isinstance(default_device, (list, tuple)):
            return default_device[0]
        return default_device
