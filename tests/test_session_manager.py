"""Tests for SessionManager."""
from __future__ import annotations

import asyncio
import pytest

from core.models import SentenceFinalizedEvent, WorkerStatus
from core.session_manager import SessionManager


@pytest.fixture
def session():
    return SessionManager(session_id="test_session")


def _make_event(segment_id: str, text: str = "Hello world.") -> SentenceFinalizedEvent:
    return SentenceFinalizedEvent(
        session_id="test_session",
        segment_id=segment_id,
        start_ms=0,
        end_ms=1000,
        source_text_final=text,
    )


@pytest.mark.asyncio
async def test_create_segment(session):
    ev = _make_event("seg_001")
    seg = await session.create_from_event(ev)
    assert seg.segment_id == "seg_001"
    assert seg.source_text_final == "Hello world."
    assert seg.is_final is True
    assert seg.speaker_status == WorkerStatus.PENDING
    assert seg.translation_status == WorkerStatus.PENDING


@pytest.mark.asyncio
async def test_patch_speaker(session):
    ev = _make_event("seg_002")
    await session.create_from_event(ev)

    seg = await session.patch_speaker("seg_002", "speaker_1", 0.9)
    assert seg is not None
    assert seg.speaker_id == "speaker_1"
    assert seg.speaker_confidence == 0.9
    assert seg.speaker_status == WorkerStatus.DONE


@pytest.mark.asyncio
async def test_patch_translation(session):
    ev = _make_event("seg_003")
    await session.create_from_event(ev)

    seg = await session.patch_translation("seg_003", "Bonjour le monde.")
    assert seg is not None
    assert seg.translated_text == "Bonjour le monde."
    assert seg.translation_status == WorkerStatus.DONE


@pytest.mark.asyncio
async def test_is_enriched_after_both_patches(session):
    ev = _make_event("seg_004")
    await session.create_from_event(ev)

    seg = await session.get("seg_004")
    assert seg is not None
    assert not seg.is_enriched

    await session.patch_speaker("seg_004", "speaker_1", 0.8)
    seg = await session.get("seg_004")
    assert not seg.is_enriched  # translation still pending

    await session.patch_translation("seg_004", "Hello.")
    seg = await session.get("seg_004")
    assert seg.is_enriched


@pytest.mark.asyncio
async def test_patch_unknown_segment_returns_none(session):
    result = await session.patch_speaker("nonexistent", "speaker_1", 0.5)
    assert result is None


@pytest.mark.asyncio
async def test_all_segments_ordered(session):
    for i in range(3):
        await session.create_from_event(_make_event(f"seg_{i:03d}"))
    segs = await session.all_segments()
    assert [s.segment_id for s in segs] == ["seg_000", "seg_001", "seg_002"]


@pytest.mark.asyncio
async def test_speaker_failed(session):
    await session.create_from_event(_make_event("seg_005"))
    seg = await session.patch_speaker_failed("seg_005", "no audio")
    assert seg.speaker_status == WorkerStatus.FAILED


@pytest.mark.asyncio
async def test_is_enriched_with_failed_status(session):
    await session.create_from_event(_make_event("seg_006"))
    await session.patch_speaker_failed("seg_006", "error")
    await session.patch_translation_failed("seg_006", "error")
    seg = await session.get("seg_006")
    assert seg.is_enriched  # FAILED counts as terminal state
