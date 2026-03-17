"""Tests for PatchUpdater."""
from __future__ import annotations

import asyncio
import pytest

from core.event_bus import EventBus
from core.models import (
    EventType,
    SegmentEnrichedEvent,
    SentenceFinalizedEvent,
    SpeakerCompletedEvent,
    TranslationCompletedEvent,
)
from core.patch_updater import PatchUpdater
from core.session_manager import SessionManager


async def _setup(session_id: str = "sess"):
    loop = asyncio.get_event_loop()
    bus = EventBus(loop)
    sm = SessionManager(session_id)
    updater = PatchUpdater(bus, sm)
    updater.register()
    return bus, sm, updater


@pytest.mark.asyncio
async def test_patch_updater_enriches_segment():
    bus, sm, _ = await _setup()
    enriched_events = []

    async def capture_enriched(event):
        enriched_events.append(event)

    bus.subscribe(EventType.SEGMENT_ENRICHED, capture_enriched)

    # Seed session with a segment
    ev_finalized = SentenceFinalizedEvent(
        session_id="sess",
        segment_id="seg_001",
        start_ms=0,
        end_ms=500,
        source_text_final="Test sentence.",
    )
    await sm.create_from_event(ev_finalized)

    bus_task = asyncio.create_task(bus.run())

    # Simulate speaker result
    await bus.publish(SpeakerCompletedEvent(
        session_id="sess", segment_id="seg_001",
        speaker_id="speaker_1", confidence=0.9,
    ))
    await asyncio.sleep(0.05)

    # Not yet enriched (translation still pending)
    assert len(enriched_events) == 0

    # Simulate translation result
    await bus.publish(TranslationCompletedEvent(
        session_id="sess", segment_id="seg_001",
        translated_text="Test sentence.",
    ))
    await asyncio.sleep(0.05)

    # Now enriched
    assert len(enriched_events) == 1
    seg = enriched_events[0].segment
    assert seg is not None
    assert seg.speaker_id == "speaker_1"
    assert seg.translated_text == "Test sentence."
    assert seg.is_enriched

    await bus.shutdown()
    await bus_task


@pytest.mark.asyncio
async def test_patch_updater_order_independent():
    """Translation completing before speaker should still enrich correctly."""
    bus, sm, _ = await _setup()
    enriched_events = []

    async def capture(event):
        enriched_events.append(event)

    bus.subscribe(EventType.SEGMENT_ENRICHED, capture)

    ev_finalized = SentenceFinalizedEvent(
        session_id="sess",
        segment_id="seg_002",
        start_ms=0,
        end_ms=800,
        source_text_final="Another sentence.",
    )
    await sm.create_from_event(ev_finalized)

    bus_task = asyncio.create_task(bus.run())

    # Translation first
    await bus.publish(TranslationCompletedEvent(
        session_id="sess", segment_id="seg_002",
        translated_text="Another sentence.",
    ))
    await asyncio.sleep(0.05)
    assert len(enriched_events) == 0

    # Speaker second
    await bus.publish(SpeakerCompletedEvent(
        session_id="sess", segment_id="seg_002",
        speaker_id="speaker_2", confidence=0.7,
    ))
    await asyncio.sleep(0.05)
    assert len(enriched_events) == 1

    await bus.shutdown()
    await bus_task
