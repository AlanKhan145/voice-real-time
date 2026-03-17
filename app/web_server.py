"""
Web server — trang web đơn giản để thử module phụ đề realtime.

Chạy: python -m app.main --web
Mở: http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

from app.config import Config

logger = logging.getLogger(__name__)

app = FastAPI(title="realtime-subtitle")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Trang chủ."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>realtime-subtitle</h1><p>Static files not found.</p>")


# WebSocket: một client active tại một thời điểm
_active_ws: Optional[WebSocket] = None
_pipeline_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None


async def _run_pipeline(ws: WebSocket, config: Config, stop: asyncio.Event) -> None:
    """Chạy pipeline và gửi events qua WebSocket."""
    from core.event_bus import EventBus
    from core.models import EventType, make_session_id
    from core.session_manager import SessionManager
    from core.patch_updater import PatchUpdater
    from asr.sounddevice_recorder import SoundDeviceRecorder
    from workers.speaker_worker import SpeakerWorker
    from workers.translation_worker import TranslationWorker

    loop = asyncio.get_running_loop()
    session_id = make_session_id()

    event_bus = EventBus(loop)
    session_manager = SessionManager(session_id)
    speaker_provider = _build_speaker_provider(config)
    translation_provider = _build_translation_provider(config)

    speaker_worker = SpeakerWorker(
        event_bus=event_bus,
        provider=speaker_provider,
        max_workers=config.worker_max_threads // 2 or 1,
    )
    translation_worker = TranslationWorker(
        event_bus=event_bus,
        provider=translation_provider,
        source_lang=config.language,
        target_lang=config.target_language,
        max_workers=config.worker_max_threads // 2 or 1,
    )
    patch_updater = PatchUpdater(event_bus, session_manager)
    recorder = SoundDeviceRecorder(
        config=config,
        event_bus=event_bus,
        session_id=session_id,
        loop=loop,
    )

    async def _on_sentence_finalized(event):
        await session_manager.create_from_event(event)

    async def _send_partial(event):
        await ws.send_json({"type": "partial", "text": event.text})

    async def _send_enriched(event):
        if event.segment is None:
            return
        seg = event.segment
        await ws.send_json({
            "type": "enriched",
            "speaker": seg.speaker_id or "unknown",
            "subtitle": seg.source_text_final or "",
            "translation": seg.translated_text or "",
        })

    event_bus.subscribe(EventType.PARTIAL_UPDATED, _send_partial)
    event_bus.subscribe(EventType.SENTENCE_FINALIZED, _on_sentence_finalized)
    speaker_worker.register()
    translation_worker.register()
    patch_updater.register()
    event_bus.subscribe(EventType.SEGMENT_ENRICHED, _send_enriched)

    await ws.send_json({"type": "started", "session_id": session_id})

    recorder.start()

    bus_task = asyncio.create_task(event_bus.run())
    stop_task = asyncio.create_task(stop.wait())

    await asyncio.wait(
        [bus_task, stop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    recorder.stop()
    speaker_worker.shutdown()
    translation_worker.shutdown()
    await asyncio.sleep(1.0)
    await event_bus.shutdown()
    try:
        await asyncio.wait_for(bus_task, timeout=2.0)
    except asyncio.TimeoutError:
        pass


def _build_speaker_provider(config: Config):
    from providers.speaker.local_provider import LocalSpeakerProvider, NoopSpeakerProvider
    if not config.speaker_enabled or config.speaker_provider == "noop":
        return NoopSpeakerProvider()
    return LocalSpeakerProvider(similarity_threshold=config.speaker_similarity_threshold)


def _build_translation_provider(config: Config):
    from providers.translation.local_provider import create_translation_provider, NoopTranslationProvider
    if not config.translation_enabled or config.translation_provider == "noop":
        return NoopTranslationProvider()
    return create_translation_provider(config.translation_provider)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket: bấm Start → bắt đầu pipeline, nhận events."""
    global _active_ws, _pipeline_task, _stop_event

    await websocket.accept()

    if _active_ws is not None:
        await websocket.send_json({"type": "error", "message": "Đã có phiên đang chạy."})
        await websocket.close()
        return

    _active_ws = websocket
    _stop_event = asyncio.Event()
    config = Config.from_env()

    async def run_and_cleanup():
        try:
            await _run_pipeline(websocket, config, _stop_event)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.exception("Pipeline error")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            global _active_ws, _pipeline_task
            _active_ws = None
            _pipeline_task = None

    _pipeline_task = asyncio.create_task(run_and_cleanup())

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg) if msg else {}
            if data.get("action") == "stop":
                if _stop_event:
                    _stop_event.set()
                break
    except WebSocketDisconnect:
        if _stop_event:
            _stop_event.set()
    except Exception:
        pass
    finally:
        _active_ws = None
        if _pipeline_task:
            try:
                await asyncio.wait_for(_pipeline_task, timeout=5.0)
            except asyncio.TimeoutError:
                _pipeline_task.cancel()
                try:
                    await _pipeline_task
                except asyncio.CancelledError:
                    pass
