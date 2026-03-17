"""
SpeakerWorker — runs speaker identification for every finalized sentence.

- Subscribes to: sentence_finalized
- Publishes:      speaker_completed  (or logs failure and marks FAILED)
- Runs provider.identify() in a thread-pool so the event loop is never blocked
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from core.event_bus import EventBus
from core.models import (
    EventType,
    SentenceFinalizedEvent,
    SpeakerCompletedEvent,
)
from providers.speaker.base import SpeakerProvider

logger = logging.getLogger(__name__)


class SpeakerWorker:
    def __init__(
        self,
        event_bus: EventBus,
        provider: SpeakerProvider,
        max_workers: int = 2,
    ) -> None:
        self._bus = event_bus
        self._provider = provider
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="speaker-worker"
        )

    def register(self) -> None:
        self._bus.subscribe(EventType.SENTENCE_FINALIZED, self.on_sentence_finalized)

    async def on_sentence_finalized(self, event: SentenceFinalizedEvent) -> None:
        """Non-blocking: schedule identification in thread pool."""
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            self._executor,
            self._run_identification,
            event,
            loop,
        )

    def _run_identification(
        self,
        event: SentenceFinalizedEvent,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Executes in thread pool — must not touch the event loop directly."""
        segment_id = event.segment_id
        audio_span = event.audio_span_ref

        if audio_span is None:
            logger.warning("[speaker] No audio span for segment %s – skipping", segment_id)
            return

        try:
            result = self._provider.identify(audio_span)
            ev = SpeakerCompletedEvent(
                session_id=event.session_id,
                segment_id=segment_id,
                speaker_id=result.speaker_id,
                confidence=result.confidence,
            )
            asyncio.run_coroutine_threadsafe(self._bus.publish(ev), loop)
            logger.debug(
                "[speaker] %s → %s (%.0f%%) via %s",
                segment_id,
                result.speaker_id,
                result.confidence * 100,
                result.provider,
            )
        except Exception as exc:
            logger.error("[speaker] identification failed for %s: %s", segment_id, exc)
            # Still publish a completed event with fallback values so PatchUpdater
            # can mark the segment done and proceed to enriched.
            ev = SpeakerCompletedEvent(
                session_id=event.session_id,
                segment_id=segment_id,
                speaker_id="speaker_unknown",
                confidence=0.0,
            )
            asyncio.run_coroutine_threadsafe(self._bus.publish(ev), loop)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._provider.shutdown()
        logger.info("SpeakerWorker shut down")
