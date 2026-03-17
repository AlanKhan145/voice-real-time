"""
End-to-end enrichment flow test with fake providers.

Simulates:
  sentence_finalized
    → SpeakerWorker  (fake fast provider)
    → TranslationWorker (fake fast provider)
  → PatchUpdater → segment_enriched
"""
from __future__ import annotations

import asyncio
import pytest

from core.event_bus import EventBus
from core.models import (
    AudioSpanRef,
    EventType,
    SentenceFinalizedEvent,
    SegmentEnrichedEvent,
)
from core.patch_updater import PatchUpdater
from core.session_manager import SessionManager
from providers.speaker.base import SpeakerProvider, SpeakerResult
from providers.translation.base import TranslationProvider, TranslationResult
from workers.speaker_worker import SpeakerWorker
from workers.translation_worker import TranslationWorker


# ── Fake providers ─────────────────────────────────────────────────────────

class FakeSpeakerProvider(SpeakerProvider):
    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        return SpeakerResult(speaker_id="speaker_1", confidence=0.95, provider="fake")


class FakeTranslationProvider(TranslationProvider):
    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return TranslationResult(
            translated_text=f"[TRANSLATED] {text}",
            source_lang=source_lang,
            target_lang=target_lang,
            provider="fake",
        )


# ── Test ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_enrichment_flow():
    loop = asyncio.get_event_loop()
    session_id = "test_session"

    bus = EventBus(loop)
    sm = SessionManager(session_id)

    speaker_worker = SpeakerWorker(bus, FakeSpeakerProvider(), max_workers=1)
    translation_worker = TranslationWorker(
        bus, FakeTranslationProvider(), source_lang="vi", target_lang="en", max_workers=1
    )
    patch_updater = PatchUpdater(bus, sm)

    enriched_events: list[SegmentEnrichedEvent] = []

    async def capture_enriched(event: SegmentEnrichedEvent):
        enriched_events.append(event)

    # Wire up
    async def on_finalized(event):
        await sm.create_from_event(event)

    bus.subscribe(EventType.SENTENCE_FINALIZED, on_finalized)
    speaker_worker.register()
    translation_worker.register()
    patch_updater.register()
    bus.subscribe(EventType.SEGMENT_ENRICHED, capture_enriched)

    bus_task = asyncio.create_task(bus.run())

    # Fire the event
    span = AudioSpanRef(
        session_id=session_id,
        segment_id="seg_abc",
        start_ms=0,
        end_ms=1500,
        audio_bytes=b"\x00\x01" * 800,
    )
    await bus.publish(SentenceFinalizedEvent(
        session_id=session_id,
        segment_id="seg_abc",
        start_ms=0,
        end_ms=1500,
        source_text_final="Xin chào thế giới.",
        audio_span_ref=span,
    ))

    # Wait for workers (they run in thread pools)
    for _ in range(30):
        await asyncio.sleep(0.1)
        if enriched_events:
            break

    assert len(enriched_events) == 1, "Expected exactly one segment_enriched event"
    seg = enriched_events[0].segment
    assert seg is not None
    assert seg.speaker_id == "speaker_1"
    assert seg.translated_text == "[TRANSLATED] Xin chào thế giới."
    assert seg.is_enriched

    speaker_worker.shutdown()
    translation_worker.shutdown()
    await bus.shutdown()
    await bus_task


@pytest.mark.asyncio
async def test_multiple_segments_independent():
    """Three segments should each be enriched independently."""
    loop = asyncio.get_event_loop()
    session_id = "multi_session"

    bus = EventBus(loop)
    sm = SessionManager(session_id)

    speaker_worker = SpeakerWorker(bus, FakeSpeakerProvider(), max_workers=2)
    translation_worker = TranslationWorker(
        bus, FakeTranslationProvider(), source_lang="vi", target_lang="en", max_workers=2
    )
    patch_updater = PatchUpdater(bus, sm)
    enriched: list[SegmentEnrichedEvent] = []

    async def on_finalized(event):
        await sm.create_from_event(event)

    bus.subscribe(EventType.SENTENCE_FINALIZED, on_finalized)
    speaker_worker.register()
    translation_worker.register()
    patch_updater.register()
    bus.subscribe(EventType.SEGMENT_ENRICHED, lambda e: enriched.append(e))

    bus_task = asyncio.create_task(bus.run())

    for i in range(3):
        span = AudioSpanRef(
            session_id=session_id,
            segment_id=f"seg_{i:03d}",
            start_ms=i * 2000,
            end_ms=(i + 1) * 2000,
            audio_bytes=b"\x00\x01" * 400,
        )
        await bus.publish(SentenceFinalizedEvent(
            session_id=session_id,
            segment_id=f"seg_{i:03d}",
            start_ms=i * 2000,
            end_ms=(i + 1) * 2000,
            source_text_final=f"Câu số {i}.",
            audio_span_ref=span,
        ))
        await asyncio.sleep(0.01)

    # Wait for all to complete
    for _ in range(50):
        await asyncio.sleep(0.1)
        if len(enriched) >= 3:
            break

    assert len(enriched) == 3
    ids = {e.segment_id for e in enriched}
    assert ids == {"seg_000", "seg_001", "seg_002"}

    speaker_worker.shutdown()
    translation_worker.shutdown()
    await bus.shutdown()
    await bus_task
