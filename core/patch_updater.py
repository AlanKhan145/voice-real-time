"""
PatchUpdater — merges worker results back into segments.

Subscribes to:
  - SpeakerCompletedEvent → patch speaker fields
  - TranslationCompletedEvent → patch translation fields

After each patch, checks whether the segment is fully enriched.
If so, publishes SegmentEnrichedEvent.
"""
from __future__ import annotations

import logging

from core.event_bus import EventBus
from core.models import (
    EventType,
    SegmentEnrichedEvent,
    SpeakerCompletedEvent,
    TranslationCompletedEvent,
)
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class PatchUpdater:
    def __init__(self, event_bus: EventBus, session_manager: SessionManager) -> None:
        self._bus = event_bus
        self._sm = session_manager

    def register(self) -> None:
        self._bus.subscribe(EventType.SPEAKER_COMPLETED, self.on_speaker_completed)
        self._bus.subscribe(EventType.TRANSLATION_COMPLETED, self.on_translation_completed)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def on_speaker_completed(self, event: SpeakerCompletedEvent) -> None:
        seg = await self._sm.patch_speaker(
            segment_id=event.segment_id,
            speaker_id=event.speaker_id,
            confidence=event.confidence,
        )
        if seg is None:
            return
        if seg.is_enriched:
            await self._bus.publish(
                SegmentEnrichedEvent(
                    session_id=seg.session_id,
                    segment_id=seg.segment_id,
                    segment=seg,
                )
            )

    async def on_translation_completed(self, event: TranslationCompletedEvent) -> None:
        seg = await self._sm.patch_translation(
            segment_id=event.segment_id,
            translated_text=event.translated_text,
        )
        if seg is None:
            return
        if seg.is_enriched:
            await self._bus.publish(
                SegmentEnrichedEvent(
                    session_id=seg.session_id,
                    segment_id=seg.segment_id,
                    segment=seg,
                )
            )
