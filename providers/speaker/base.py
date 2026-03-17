"""
SpeakerProvider abstract base class.

Any backend (resemblyzer, pyannote, cloud API, etc.) must implement this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.models import AudioSpanRef


@dataclass
class SpeakerResult:
    speaker_id: str
    confidence: float
    provider: str


class SpeakerProvider(ABC):
    """
    Contract:
      identify(audio_span_ref) -> SpeakerResult

    The call may be synchronous (blocking).  The worker will run it in a
    thread-pool executor so it never blocks the event loop.
    """

    @abstractmethod
    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        ...

    def shutdown(self) -> None:
        """Optional cleanup hook called at program exit."""
