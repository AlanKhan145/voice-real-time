"""Tests for EventBus."""
from __future__ import annotations

import asyncio
import pytest

from core.event_bus import EventBus
from core.models import EventType, PartialUpdatedEvent, SentenceFinalizedEvent


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def bus(loop):
    return EventBus(loop)


@pytest.mark.asyncio
async def test_subscribe_and_receive():
    loop = asyncio.get_event_loop()
    bus = EventBus(loop)
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(EventType.PARTIAL_UPDATED, handler)

    ev = PartialUpdatedEvent(session_id="s1", segment_id="seg1", text="hello")
    await bus.publish(ev)

    # Run dispatcher for one iteration
    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    await bus.shutdown()
    await task

    assert len(received) == 1
    assert received[0].text == "hello"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    loop = asyncio.get_event_loop()
    bus = EventBus(loop)
    log = []

    async def handler_a(event):
        log.append(("A", event.text))

    async def handler_b(event):
        log.append(("B", event.text))

    bus.subscribe(EventType.PARTIAL_UPDATED, handler_a)
    bus.subscribe(EventType.PARTIAL_UPDATED, handler_b)

    ev = PartialUpdatedEvent(session_id="s1", segment_id="seg1", text="hi")
    await bus.publish(ev)

    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    await bus.shutdown()
    await task

    assert ("A", "hi") in log
    assert ("B", "hi") in log


@pytest.mark.asyncio
async def test_publish_threadsafe():
    import threading

    loop = asyncio.get_event_loop()
    bus = EventBus(loop)
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(EventType.PARTIAL_UPDATED, handler)

    task = asyncio.create_task(bus.run())

    def publish_from_thread():
        ev = PartialUpdatedEvent(session_id="s1", segment_id="seg1", text="from_thread")
        bus.publish_threadsafe(ev)

    t = threading.Thread(target=publish_from_thread)
    t.start()
    t.join()
    await asyncio.sleep(0.1)
    await bus.shutdown()
    await task

    assert any(e.text == "from_thread" for e in received)


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_bus():
    loop = asyncio.get_event_loop()
    bus = EventBus(loop)
    received = []

    async def bad_handler(event):
        raise ValueError("intentional")

    async def good_handler(event):
        received.append(event)

    bus.subscribe(EventType.PARTIAL_UPDATED, bad_handler)
    bus.subscribe(EventType.PARTIAL_UPDATED, good_handler)

    ev = PartialUpdatedEvent(session_id="s1", segment_id="seg1", text="test")
    await bus.publish(ev)

    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    await bus.shutdown()
    await task

    assert len(received) == 1
