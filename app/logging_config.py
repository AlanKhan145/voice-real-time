"""
Structured logging setup.
Call setup_logging(config) once at startup.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("faster_whisper", "torch", "RealtimeSTT", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
