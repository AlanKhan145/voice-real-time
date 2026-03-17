"""
Internal event bus built on asyncio.Queue.

Design contract:
  - publish()         -> async, called from within the asyncio event loop
  - publish_threadsafe() -> sync, called from any thread; bridges to the loop
  - subscribe()       -> register an async callback for a given EventType
  - run()             -> dispatcher coroutine; run once inside the event loop

The interface is intentionally thin so it can be replaced by a Redis Streams
or Kafka adapter later without touching any subscriber code.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from core.models import AnyEvent, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[AnyEvent], Awaitable[None]]


class EventBus:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[AnyEvent | None] = asyncio.Queue()
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__qualname__, event_type)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event: AnyEvent) -> None:
        await self._queue.put(event)

    def publish_threadsafe(self, event: AnyEvent) -> None:
        """Thread-safe publish from non-async context (e.g. RealtimeSTT thread)."""
        asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)

    # ------------------------------------------------------------------
    # Dispatcher loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        logger.info("EventBus dispatcher started")
        while self._running:
            event = await self._queue.get()
            if event is None:  # sentinel for shutdown
                break
            await self._dispatch(event)
            self._queue.task_done()
        logger.info("EventBus dispatcher stopped")

    async def _dispatch(self, event: AnyEvent) -> None:
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised for event %s",
                    handler.__qualname__,
                    event.event_type,
                )

    async def shutdown(self) -> None:
        self._running = False
        await self._queue.put(None)  # unblock run()
