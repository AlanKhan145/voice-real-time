"""
Core data models for realtime-subtitle.
All domain objects live here; no business logic.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class WorkerStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class EventType(str, Enum):
    PARTIAL_UPDATED = "partial_updated"
    STABILIZED_UPDATED = "stabilized_updated"
    SENTENCE_FINALIZED = "sentence_finalized"
    SPEAKER_COMPLETED = "speaker_completed"
    TRANSLATION_COMPLETED = "translation_completed"
    SEGMENT_ENRICHED = "segment_enriched"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class AudioSpanRef:
    """Reference to the raw PCM audio that corresponds to a segment."""

    session_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    audio_bytes: Optional[bytes] = None  # raw 16-bit PCM @ 16 kHz mono
    sample_rate: int = 16000


@dataclass
class Segment:
    session_id: str
    segment_id: str
    start_ms: int
    end_ms: int

    source_text_partial: str = ""
    source_text_final: str = ""
    is_final: bool = False

    speaker_status: WorkerStatus = WorkerStatus.PENDING
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0

    translation_status: WorkerStatus = WorkerStatus.PENDING
    translated_text: Optional[str] = None

    audio_span_ref: Optional[AudioSpanRef] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def touch(self) -> None:
        self.updated_at = datetime.now()

    @property
    def is_enriched(self) -> bool:
        return (
            self.speaker_status in (WorkerStatus.DONE, WorkerStatus.FAILED)
            and self.translation_status in (WorkerStatus.DONE, WorkerStatus.FAILED)
        )


# ---------------------------------------------------------------------------
# Events – plain dataclasses, no inheritance overhead
# ---------------------------------------------------------------------------


@dataclass
class PartialUpdatedEvent:
    event_type: EventType = field(default=EventType.PARTIAL_UPDATED, init=False)
    session_id: str = ""
    segment_id: str = ""
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StabilizedUpdatedEvent:
    event_type: EventType = field(default=EventType.STABILIZED_UPDATED, init=False)
    session_id: str = ""
    segment_id: str = ""
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentenceFinalizedEvent:
    event_type: EventType = field(default=EventType.SENTENCE_FINALIZED, init=False)
    session_id: str = ""
    segment_id: str = ""
    start_ms: int = 0
    end_ms: int = 0
    source_text_final: str = ""
    audio_span_ref: Optional[AudioSpanRef] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpeakerCompletedEvent:
    event_type: EventType = field(default=EventType.SPEAKER_COMPLETED, init=False)
    session_id: str = ""
    segment_id: str = ""
    speaker_id: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TranslationCompletedEvent:
    event_type: EventType = field(default=EventType.TRANSLATION_COMPLETED, init=False)
    session_id: str = ""
    segment_id: str = ""
    translated_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SegmentEnrichedEvent:
    event_type: EventType = field(default=EventType.SEGMENT_ENRICHED, init=False)
    session_id: str = ""
    segment_id: str = ""
    segment: Optional[Segment] = None
    timestamp: datetime = field(default_factory=datetime.now)


AnyEvent = (
    PartialUpdatedEvent
    | StabilizedUpdatedEvent
    | SentenceFinalizedEvent
    | SpeakerCompletedEvent
    | TranslationCompletedEvent
    | SegmentEnrichedEvent
)


def make_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:8]}"


def make_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:8]}"
