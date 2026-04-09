"""Project configuration and filesystem paths for JARVIS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TOOLS_DIR = BASE_DIR / "tools"
CORE_DIR = BASE_DIR / "core"
DATA_AI_DIR = BASE_DIR / "data_ai"
DATA_USER_DIR = BASE_DIR / "data_user"
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_LOGS_DIR = MEMORY_DIR / "logs"
LEGACY_VECTOR_DIR = MEMORY_DIR / "vector_db"
USER_CHUNKS_DIR = DATA_USER_DIR / "chunks"
USER_VECTOR_DIR = DATA_USER_DIR / "vector_index"
USER_NOTES_DIR = DATA_USER_DIR / "notes"
USER_REMINDERS_FILE = DATA_USER_DIR / "reminders.json"
SCREENSHOTS_DIR = DATA_USER_DIR / "screenshots"
GOOGLE_TOKEN_DIR = CONFIG_DIR / "google"
ENV_FILE = CONFIG_DIR / ".env"

load_dotenv(ENV_FILE)


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or default


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    user_name: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_simple_model: str
    nvidia_memory_model: str
    nvidia_complex_model: str
    nvidia_code_model: str
    nvidia_timeout_seconds: int
    copilot_base_url: str
    copilot_simple_model: str
    copilot_memory_model: str
    copilot_complex_model: str
    copilot_code_model: str
    copilot_timeout_seconds: int
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_simple_model: str
    openrouter_memory_model: str
    openrouter_complex_model: str
    openrouter_code_model: str
    openrouter_timeout_seconds: int
    stt_enabled: bool
    stt_model: str
    stt_sample_rate: int
    stt_input_device: str
    stt_silence_threshold: float
    stt_silence_duration_seconds: float
    stt_max_record_seconds: int
    wake_word_enabled: bool
    wake_word_keyword: str
    wake_word_model_path: str
    wake_word_input_device: str
    wake_word_threshold: float
    wake_word_vad_threshold: float
    audio_feedback_enabled: bool
    tool_reload_seconds: int
    agent_max_iterations: int
    memory_top_k: int
    memory_embedding_model: str
    planner_simple_word_limit: int
    planner_complex_word_limit: int
    background_memory_persistence: bool
    tts_enabled: bool
    tts_voice: str
    tts_rate: str
    tts_volume: str
    weather_api_key: str
    weather_default_units: str
    reminder_check_seconds: int
    reminder_timezone: str
    gmail_client_secret_file: str
    gmail_token_file: str
    calendar_client_secret_file: str
    calendar_token_file: str


SETTINGS = Settings(
    user_name=_getenv("JARVIS_USER_NAME", "anime") or "anime",
    nvidia_api_key=_getenv("NVIDIA_API_KEY", "") or "",
    nvidia_base_url=_getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    or "https://integrate.api.nvidia.com/v1",
    nvidia_simple_model=_getenv("NVIDIA_SIMPLE_MODEL", "moonshotai/kimi-k2.5") or "moonshotai/kimi-k2.5",
    nvidia_memory_model=_getenv("NVIDIA_MEMORY_MODEL", "moonshotai/kimi-k2.5") or "moonshotai/kimi-k2.5",
    nvidia_complex_model=_getenv("NVIDIA_COMPLEX_MODEL", "moonshotai/kimi-k2.5") or "moonshotai/kimi-k2.5",
    nvidia_code_model=_getenv("NVIDIA_CODE_MODEL", "moonshotai/kimi-k2.5") or "moonshotai/kimi-k2.5",
    nvidia_timeout_seconds=int(_getenv("NVIDIA_TIMEOUT_SECONDS", "120") or "120"),
    copilot_base_url=_getenv("COPILOT_BASE_URL", "https://api.individual.githubcopilot.com")
    or "https://api.individual.githubcopilot.com",
    copilot_simple_model=_getenv("COPILOT_SIMPLE_MODEL", "gpt-4o") or "gpt-4o",
    copilot_memory_model=_getenv("COPILOT_MEMORY_MODEL", "gpt-4o") or "gpt-4o",
    copilot_complex_model=_getenv("COPILOT_COMPLEX_MODEL", "gpt-4o") or "gpt-4o",
    copilot_code_model=_getenv("COPILOT_CODE_MODEL", "gpt-4o") or "gpt-4o",
    copilot_timeout_seconds=int(_getenv("COPILOT_TIMEOUT_SECONDS", "120") or "120"),
    openrouter_api_key=_getenv("OPENROUTER_API_KEY", "") or "",
    openrouter_base_url=_getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    or "https://openrouter.ai/api/v1",
    openrouter_simple_model=_getenv("OPENROUTER_SIMPLE_MODEL", "openrouter/free") or "openrouter/free",
    openrouter_memory_model=_getenv("OPENROUTER_MEMORY_MODEL", "openrouter/free") or "openrouter/free",
    openrouter_complex_model=_getenv("OPENROUTER_COMPLEX_MODEL", "openrouter/free") or "openrouter/free",
    openrouter_code_model=_getenv("OPENROUTER_CODE_MODEL", "openrouter/free") or "openrouter/free",
    openrouter_timeout_seconds=int(_getenv("OPENROUTER_TIMEOUT_SECONDS", "120") or "120"),
    stt_enabled=_parse_bool(
        _getenv("STT_ENABLED", "true"),
        True,
    ),
    stt_model=_getenv("STT_MODEL", "tiny") or "tiny",
    stt_sample_rate=int(_getenv("STT_SAMPLE_RATE", "16000") or "16000"),
    stt_input_device=_getenv("STT_INPUT_DEVICE", "") or "",
    stt_silence_threshold=float(_getenv("STT_SILENCE_THRESHOLD", "0.015") or "0.015"),
    stt_silence_duration_seconds=float(_getenv("STT_SILENCE_DURATION_SECONDS", "1.5") or "1.5"),
    stt_max_record_seconds=int(_getenv("STT_MAX_RECORD_SECONDS", "20") or "20"),
    wake_word_enabled=_parse_bool(
        _getenv("WAKE_WORD_ENABLED", "false"),
        False,
    ),
    wake_word_keyword=_getenv("WAKE_WORD_KEYWORD", "jarvis") or "jarvis",
    wake_word_model_path=_getenv("WAKE_WORD_MODEL_PATH", "") or "",
    wake_word_input_device=_getenv("WAKE_WORD_INPUT_DEVICE", "") or "",
    wake_word_threshold=float(_getenv("WAKE_WORD_THRESHOLD", "0.5") or "0.5"),
    wake_word_vad_threshold=float(_getenv("WAKE_WORD_VAD_THRESHOLD", "0.0") or "0.0"),
    audio_feedback_enabled=_parse_bool(
        _getenv("AUDIO_FEEDBACK_ENABLED", "true"),
        True,
    ),
    tool_reload_seconds=int(_getenv("TOOL_RELOAD_SECONDS", "300") or "300"),
    agent_max_iterations=int(_getenv("AGENT_MAX_ITERATIONS", "8") or "8"),
    memory_top_k=int(_getenv("MEMORY_TOP_K", "5") or "5"),
    memory_embedding_model=_getenv(
        "MEMORY_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    or "sentence-transformers/all-MiniLM-L6-v2",
    planner_simple_word_limit=int(_getenv("PLANNER_SIMPLE_WORD_LIMIT", "8") or "8"),
    planner_complex_word_limit=int(_getenv("PLANNER_COMPLEX_WORD_LIMIT", "18") or "18"),
    background_memory_persistence=_parse_bool(
        _getenv("BACKGROUND_MEMORY_PERSISTENCE", "true"),
        True,
    ),
    tts_enabled=_parse_bool(
        _getenv("TTS_ENABLED", "true"),
        True,
    ),
    tts_voice=_getenv("TTS_VOICE", "en-US-GuyNeural") or "en-US-GuyNeural",
    tts_rate=_getenv("TTS_RATE", "+0%") or "+0%",
    tts_volume=_getenv("TTS_VOLUME", "+0%") or "+0%",
    weather_api_key=_getenv("OPENWEATHERMAP_API_KEY", "") or "",
    weather_default_units=_getenv("WEATHER_UNITS", "metric") or "metric",
    reminder_check_seconds=int(_getenv("REMINDER_CHECK_SECONDS", "30") or "30"),
    reminder_timezone=_getenv("REMINDER_TIMEZONE", "Asia/Calcutta") or "Asia/Calcutta",
    gmail_client_secret_file=_getenv(
        "GMAIL_CLIENT_SECRET_FILE",
        str(GOOGLE_TOKEN_DIR / "gmail_client_secret.json"),
    ) or str(GOOGLE_TOKEN_DIR / "gmail_client_secret.json"),
    gmail_token_file=_getenv(
        "GMAIL_TOKEN_FILE",
        str(GOOGLE_TOKEN_DIR / "gmail_token.json"),
    ) or str(GOOGLE_TOKEN_DIR / "gmail_token.json"),
    calendar_client_secret_file=_getenv(
        "CALENDAR_CLIENT_SECRET_FILE",
        str(GOOGLE_TOKEN_DIR / "calendar_client_secret.json"),
    ) or str(GOOGLE_TOKEN_DIR / "calendar_client_secret.json"),
    calendar_token_file=_getenv(
        "CALENDAR_TOKEN_FILE",
        str(GOOGLE_TOKEN_DIR / "calendar_token.json"),
    ) or str(GOOGLE_TOKEN_DIR / "calendar_token.json"),
)


def ensure_directories() -> None:
    """Create required project directories if they do not exist yet."""

    for directory in (
        MEMORY_LOGS_DIR,
        LEGACY_VECTOR_DIR,
        USER_CHUNKS_DIR,
        USER_VECTOR_DIR,
        USER_NOTES_DIR,
        SCREENSHOTS_DIR,
        GOOGLE_TOKEN_DIR,
        DATA_AI_DIR,
        DATA_USER_DIR,
        TOOLS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
