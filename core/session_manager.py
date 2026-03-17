"""
SessionManager — in-memory store for all Segment objects in a session.

Design:
  - Thread-safe via a single asyncio.Lock (all access is within the event loop)
  - Segments are keyed by segment_id
  - Provides helpers to create, retrieve, and list segments
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.models import (
    AudioSpanRef,
    Segment,
    SentenceFinalizedEvent,
    WorkerStatus,
    make_segment_id,
)

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._segments: dict[str, Segment] = {}
        self._lock = asyncio.Lock()
        self._ordered_ids: list[str] = []  # insertion order

    # ------------------------------------------------------------------
    # Segment lifecycle
    # ------------------------------------------------------------------

    async def create_from_event(self, event: SentenceFinalizedEvent) -> Segment:
        """Create and store a new Segment from a SentenceFinalizedEvent."""
        async with self._lock:
            seg = Segment(
                session_id=event.session_id,
                segment_id=event.segment_id,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                source_text_final=event.source_text_final,
                is_final=True,
                audio_span_ref=event.audio_span_ref,
                speaker_status=WorkerStatus.PENDING,
                translation_status=WorkerStatus.PENDING,
            )
            self._segments[seg.segment_id] = seg
            self._ordered_ids.append(seg.segment_id)
            logger.debug("Segment created: %s", seg.segment_id)
            return seg

    async def get(self, segment_id: str) -> Optional[Segment]:
        async with self._lock:
            return self._segments.get(segment_id)

    async def patch_speaker(
        self,
        segment_id: str,
        speaker_id: str,
        confidence: float,
    ) -> Optional[Segment]:
        async with self._lock:
            seg = self._segments.get(segment_id)
            if seg is None:
                logger.warning("patch_speaker: segment %s not found", segment_id)
                return None
            seg.speaker_id = speaker_id
            seg.speaker_confidence = confidence
            seg.speaker_status = WorkerStatus.DONE
            seg.touch()
            logger.debug("Segment %s: speaker patched → %s", segment_id, speaker_id)
            return seg

    async def patch_speaker_failed(self, segment_id: str, reason: str) -> Optional[Segment]:
        async with self._lock:
            seg = self._segments.get(segment_id)
            if seg is None:
                return None
            seg.speaker_status = WorkerStatus.FAILED
            seg.touch()
            logger.warning("Segment %s: speaker failed – %s", segment_id, reason)
            return seg

    async def patch_translation(
        self,
        segment_id: str,
        translated_text: str,
    ) -> Optional[Segment]:
        async with self._lock:
            seg = self._segments.get(segment_id)
            if seg is None:
                logger.warning("patch_translation: segment %s not found", segment_id)
                return None
            seg.translated_text = translated_text
            seg.translation_status = WorkerStatus.DONE
            seg.touch()
            logger.debug("Segment %s: translation patched", segment_id)
            return seg

    async def patch_translation_failed(self, segment_id: str, reason: str) -> Optional[Segment]:
        async with self._lock:
            seg = self._segments.get(segment_id)
            if seg is None:
                return None
            seg.translation_status = WorkerStatus.FAILED
            seg.touch()
            logger.warning("Segment %s: translation failed – %s", segment_id, reason)
            return seg

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def all_segments(self) -> list[Segment]:
        async with self._lock:
            return [self._segments[sid] for sid in self._ordered_ids]

    async def count(self) -> int:
        async with self._lock:
            return len(self._segments)

    def snapshot(self) -> list[Segment]:
        """Synchronous snapshot (for shutdown/logging)."""
        return [self._segments[sid] for sid in self._ordered_ids]
