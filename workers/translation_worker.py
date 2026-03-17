"""
TranslationWorker — translates source_text_final for every finalized sentence.

- Subscribes to: sentence_finalized
- Publishes:      translation_completed
- Runs provider.translate() in a thread-pool so the event loop is never blocked
- Only translates from source_text_final (never partial text)
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from core.event_bus import EventBus
from core.models import (
    EventType,
    SentenceFinalizedEvent,
    TranslationCompletedEvent,
)
from providers.translation.base import TranslationProvider

logger = logging.getLogger(__name__)


class TranslationWorker:
    def __init__(
        self,
        event_bus: EventBus,
        provider: TranslationProvider,
        source_lang: str,
        target_lang: str,
        max_workers: int = 2,
    ) -> None:
        self._bus = event_bus
        self._provider = provider
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="translation-worker"
        )

    def register(self) -> None:
        self._bus.subscribe(EventType.SENTENCE_FINALIZED, self.on_sentence_finalized)

    async def on_sentence_finalized(self, event: SentenceFinalizedEvent) -> None:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            self._executor,
            self._run_translation,
            event,
            loop,
        )

    def _run_translation(
        self,
        event: SentenceFinalizedEvent,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        text = event.source_text_final.strip()
        if not text:
            return

        try:
            result = self._provider.translate(text, self._source_lang, self._target_lang)
            ev = TranslationCompletedEvent(
                session_id=event.session_id,
                segment_id=event.segment_id,
                translated_text=result.translated_text,
            )
            asyncio.run_coroutine_threadsafe(self._bus.publish(ev), loop)
            logger.debug(
                "[translation] %s → %r via %s",
                event.segment_id,
                result.translated_text[:60],
                result.provider,
            )
        except Exception as exc:
            logger.error("[translation] failed for %s: %s", event.segment_id, exc)
            ev = TranslationCompletedEvent(
                session_id=event.session_id,
                segment_id=event.segment_id,
                translated_text=f"[translation failed: {exc}]",
            )
            asyncio.run_coroutine_threadsafe(self._bus.publish(ev), loop)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._provider.shutdown()
        logger.info("TranslationWorker shut down")
