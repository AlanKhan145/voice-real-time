"""
Application configuration loaded from environment variables or .env file.
All values have sensible defaults for local CPU development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # ── ASR ────────────────────────────────────────────────────────────────
    language: str = "vi"          # source language code (vi, en, …)
    model_size: str = "base"      # tiny | base | small | medium | large-v3
    device: str = "cpu"           # cpu | cuda
    compute_type: str = "int8"    # int8 | float16 | float32

    # ── Realtime transcription ─────────────────────────────────────────────
    realtime_model_size: str = "tiny"
    realtime_processing_pause: float = 0.2   # seconds between realtime updates

    # ── VAD / silence detection ────────────────────────────────────────────
    silero_sensitivity: float = 0.4
    webrtc_sensitivity: int = 3
    post_speech_silence_duration: float = 0.7   # seconds of silence → end of sentence
    min_length_of_recording: float = 0.5
    min_gap_between_recordings: float = 0.01

    # ── Speaker attribution ────────────────────────────────────────────────
    speaker_enabled: bool = True
    speaker_provider: str = "local"            # local | noop
    speaker_similarity_threshold: float = 0.75

    # ── Translation ────────────────────────────────────────────────────────
    translation_enabled: bool = True
    translation_provider: str = "argos"        # argos | noop
    target_language: str = "en"               # target translation language

    # ── Session ────────────────────────────────────────────────────────────
    session_id: Optional[str] = None          # auto-generated if None

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ── Worker concurrency ─────────────────────────────────────────────────
    worker_max_threads: int = 4

    # ── Audio device ───────────────────────────────────────────────────────
    # None  → dùng input device mặc định của hệ điều hành
    # string→ tên thiết bị hoặc index (chuỗi số) cho sounddevice
    input_device: Optional[str] = None

    @classmethod
    def from_env(cls) -> Config:
        """Load config, optionally reading .env file first."""
        _load_dotenv()
        return cls(
            language=os.getenv("LANGUAGE", "vi"),
            model_size=os.getenv("MODEL_SIZE", "base"),
            device=os.getenv("DEVICE", "cpu"),
            compute_type=os.getenv("COMPUTE_TYPE", "int8"),
            realtime_model_size=os.getenv("REALTIME_MODEL_SIZE", "tiny"),
            realtime_processing_pause=_float("REALTIME_PROCESSING_PAUSE", 0.2),
            silero_sensitivity=_float("SILERO_SENSITIVITY", 0.4),
            webrtc_sensitivity=_int("WEBRTC_SENSITIVITY", 3),
            post_speech_silence_duration=_float("POST_SPEECH_SILENCE_DURATION", 0.7),
            min_length_of_recording=_float("MIN_LENGTH_OF_RECORDING", 0.5),
            min_gap_between_recordings=_float("MIN_GAP_BETWEEN_RECORDINGS", 0.01),
            speaker_enabled=_bool("SPEAKER_ENABLED", True),
            speaker_provider=os.getenv("SPEAKER_PROVIDER", "local"),
            speaker_similarity_threshold=_float("SPEAKER_SIMILARITY_THRESHOLD", 0.75),
            translation_enabled=_bool("TRANSLATION_ENABLED", True),
            translation_provider=os.getenv("TRANSLATION_PROVIDER", "argos"),
            target_language=os.getenv("TARGET_LANGUAGE", "en"),
            session_id=os.getenv("SESSION_ID"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            worker_max_threads=_int("WORKER_MAX_THREADS", 4),
            input_device=os.getenv("INPUT_DEVICE") or None,
        )


def _load_dotenv() -> None:
    """Minimal .env loader — avoids adding python-dotenv as a hard dependency."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    with env_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
