"""
realtime-subtitle — entry point.

Run:
    python -m app.main
    python -m app.main --language vi --target-language en --model-size base

Environment variables (or .env file) override all flags.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Must happen before any project imports that might configure logging.
from app.logging_config import setup_logging
from app.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing (env vars always win; CLI flags set env vars for Config)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="realtime-subtitle: local speech → subtitle → speaker → translation"
    )
    p.add_argument("--language", default=None, help="Source language (default: vi)")
    p.add_argument("--target-language", default=None, help="Target translation language (default: en)")
    p.add_argument("--model-size", default=None, help="Whisper model size (default: base)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Compute device")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-speaker", action="store_true", help="Disable speaker attribution")
    p.add_argument("--no-translation", action="store_true", help="Disable translation")
    p.add_argument("--speaker-provider", default=None, choices=["local", "noop"])
    p.add_argument("--translation-provider", default=None, choices=["argos", "noop"])
    return p.parse_args()


def _apply_args_to_env(args: argparse.Namespace) -> None:
    import os
    if args.language:
        os.environ.setdefault("LANGUAGE", args.language)
    if args.target_language:
        os.environ.setdefault("TARGET_LANGUAGE", args.target_language)
    if args.model_size:
        os.environ.setdefault("MODEL_SIZE", args.model_size)
    if args.device:
        os.environ.setdefault("DEVICE", args.device)
    if args.log_level:
        os.environ.setdefault("LOG_LEVEL", args.log_level)
    if args.no_speaker:
        os.environ["SPEAKER_ENABLED"] = "false"
    if args.no_translation:
        os.environ["TRANSLATION_ENABLED"] = "false"
    if args.speaker_provider:
        os.environ.setdefault("SPEAKER_PROVIDER", args.speaker_provider)
    if args.translation_provider:
        os.environ.setdefault("TRANSLATION_PROVIDER", args.translation_provider)


# ─────────────────────────────────────────────────────────────────────────────
# Provider factories
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main async pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run(config: Config) -> None:
    from core.event_bus import EventBus
    from core.models import EventType, make_session_id
    from core.session_manager import SessionManager
    from core.patch_updater import PatchUpdater
    from asr.sounddevice_recorder import SoundDeviceRecorder
    from workers.speaker_worker import SpeakerWorker
    from workers.translation_worker import TranslationWorker
    from cli.console_renderer import ConsoleRenderer

    loop = asyncio.get_running_loop()
    session_id = config.session_id or make_session_id()

    # ── Core infrastructure ────────────────────────────────────────────────
    event_bus = EventBus(loop)
    session_manager = SessionManager(session_id)
    renderer = ConsoleRenderer()

    # ── Providers ─────────────────────────────────────────────────────────
    speaker_provider = _build_speaker_provider(config)
    translation_provider = _build_translation_provider(config)

    # ── Workers ───────────────────────────────────────────────────────────
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

    # ── Recorder ──────────────────────────────────────────────────────────
    recorder = SoundDeviceRecorder(
        config=config,
        event_bus=event_bus,
        session_id=session_id,
        loop=loop,
    )

    # ── Wire SessionManager into the event bus ────────────────────────────
    async def _on_sentence_finalized(event):
        await session_manager.create_from_event(event)

    # ── Subscribe all handlers ────────────────────────────────────────────
    event_bus.subscribe(EventType.PARTIAL_UPDATED, renderer.on_partial)
    event_bus.subscribe(EventType.STABILIZED_UPDATED, renderer.on_stabilized)
    event_bus.subscribe(EventType.SENTENCE_FINALIZED, renderer.on_final)
    event_bus.subscribe(EventType.SENTENCE_FINALIZED, _on_sentence_finalized)

    speaker_worker.register()
    translation_worker.register()
    patch_updater.register()

    event_bus.subscribe(EventType.SPEAKER_COMPLETED, renderer.on_speaker_completed)
    event_bus.subscribe(EventType.TRANSLATION_COMPLETED, renderer.on_translation_completed)
    event_bus.subscribe(EventType.SEGMENT_ENRICHED, renderer.on_segment_enriched)

    # ── Graceful shutdown ─────────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def _handle_signal():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    # ── Start ─────────────────────────────────────────────────────────────
    renderer.print_banner(session_id)
    logger.info("Starting recorder  language=%s  model=%s  device=%s",
                config.language, config.model_size, config.device)

    recorder.start()
    renderer.print_listening()

    # Run event bus dispatcher and wait for shutdown signal concurrently
    bus_task = asyncio.create_task(event_bus.run())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    done, pending = await asyncio.wait(
        [bus_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # ── Teardown ──────────────────────────────────────────────────────────
    logger.info("Initiating graceful shutdown…")
    recorder.stop()
    speaker_worker.shutdown()
    translation_worker.shutdown()

    # Give workers a moment to flush in-flight jobs
    await asyncio.sleep(1.5)

    await event_bus.shutdown()
    # Let bus drain
    try:
        await asyncio.wait_for(bus_task, timeout=3.0)
    except asyncio.TimeoutError:
        pass

    for t in pending:
        t.cancel()

    # Print session summary
    segs = session_manager.snapshot()
    renderer.print_shutdown()
    logger.info("Session ended. Total segments: %d", len(segs))
    for seg in segs:
        logger.info(
            "  [%s] speaker=%s  translation=%s  text=%r",
            seg.segment_id,
            seg.speaker_id,
            seg.translated_text,
            seg.source_text_final[:60],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _apply_args_to_env(args)
    config = Config.from_env()
    setup_logging(config.log_level, config.log_file)

    logger.debug("Config: %s", config)

    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
