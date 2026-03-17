"""
RealtimeRecorder — wraps RealtimeSTT's AudioToTextRecorder.

Responsibilities:
  - Open the microphone and run the VAD+ASR pipeline
  - Emit PartialUpdatedEvent on every realtime update (hot path)
  - Emit StabilizedUpdatedEvent when text stabilizes
  - On sentence end: create a SentenceFinalizedEvent and publish it
  - Maintain a rolling audio buffer so Speaker worker can receive PCM data

The recorder runs in its own daemon thread; all events are published
thread-safely into the asyncio event bus.
"""
from __future__ import annotations

import asyncio
import array
import logging
import threading
import time
from typing import Callable, Optional

from app.config import Config
from asr.sentence_boundary import SentenceBoundaryDetector
from asr.stabilizer import TextStabilizer
from core.event_bus import EventBus
from core.models import (
    AudioSpanRef,
    PartialUpdatedEvent,
    SentenceFinalizedEvent,
    StabilizedUpdatedEvent,
    make_segment_id,
)

logger = logging.getLogger(__name__)

# Rolling audio buffer: keep last N seconds of audio
_AUDIO_BUFFER_SECONDS = 30
_SAMPLE_RATE = 16000
_CHANNELS = 1


class RealtimeRecorder:
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._session_id = session_id
        self._loop = loop

        self._stabilizer = TextStabilizer()
        self._boundary = SentenceBoundaryDetector()

        self._recorder = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Audio buffer (ring buffer of raw 16-bit PCM signed shorts)
        _max_samples = _AUDIO_BUFFER_SECONDS * _SAMPLE_RATE
        self._audio_lock = threading.Lock()
        self._audio_buffer: list[bytes] = []  # list of chunk byte-strings
        self._audio_buffer_total_samples = 0
        self._max_buffer_samples = _max_samples

        # Timing
        self._recording_start_ms: int = 0
        self._current_segment_id: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="recorder-thread", daemon=True
        )
        self._thread.start()
        logger.info("RealtimeRecorder thread started")

    def stop(self) -> None:
        self._running = False
        if self._recorder is not None:
            try:
                self._recorder.stop()
                self._recorder.shutdown()
            except Exception:
                logger.debug("Recorder shutdown exception", exc_info=True)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("RealtimeRecorder stopped")

    # ------------------------------------------------------------------
    # Internal – recorder thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._recorder = self._build_recorder()
        except Exception:
            logger.exception("Failed to initialise RealtimeSTT recorder")
            return

        logger.info("Recorder initialised, entering listen loop")
        self._bus.publish_threadsafe(
            PartialUpdatedEvent(
                session_id=self._session_id,
                segment_id="",
                text="[recorder ready – listening …]",
            )
        )

        while self._running:
            try:
                # .text() blocks until VAD detects end of speech,
                # then returns the final transcription for that sentence.
                self._recorder.text(self._on_sentence_final)
            except Exception as exc:
                if self._running:
                    logger.error("Recorder loop error: %s", exc)
                    time.sleep(0.5)

    def _build_recorder(self):
        from RealtimeSTT import AudioToTextRecorder  # type: ignore[import]

        params = dict(
            spinner=False,
            use_microphone=True,
            model=self._config.model_size,
            language=self._config.language if self._config.language != "auto" else "",
            device=self._config.device,
            compute_type=self._config.compute_type,
            silero_sensitivity=self._config.silero_sensitivity,
            webrtc_sensitivity=self._config.webrtc_sensitivity,
            post_speech_silence_duration=self._config.post_speech_silence_duration,
            min_length_of_recording=self._config.min_length_of_recording,
            min_gap_between_recordings=self._config.min_gap_between_recordings,
            enable_realtime_transcription=True,
            realtime_model_type=self._config.realtime_model_size,
            realtime_processing_pause=self._config.realtime_processing_pause,
            on_realtime_transcription_update=self._on_partial_raw,
            on_realtime_transcription_stabilized=self._on_stabilized_raw,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
        )
        return AudioToTextRecorder(**params)

    # ------------------------------------------------------------------
    # RealtimeSTT callbacks (called from recorder's internal threads)
    # ------------------------------------------------------------------

    def _on_recording_start(self) -> None:
        self._recording_start_ms = _now_ms()
        self._current_segment_id = make_segment_id()
        self._stabilizer.reset()
        self._boundary.reset()
        logger.debug("Recording started – segment %s", self._current_segment_id)

    def _on_recording_stop(self) -> None:
        logger.debug("Recording stopped – segment %s", self._current_segment_id)

    def _on_partial_raw(self, text: str) -> None:
        should_emit, text = self._stabilizer.update(text)
        if not should_emit:
            return
        event = PartialUpdatedEvent(
            session_id=self._session_id,
            segment_id=self._current_segment_id,
            text=text,
        )
        self._bus.publish_threadsafe(event)

    def _on_stabilized_raw(self, text: str) -> None:
        self._boundary.update_stabilized(text)
        event = StabilizedUpdatedEvent(
            session_id=self._session_id,
            segment_id=self._current_segment_id,
            text=text,
        )
        self._bus.publish_threadsafe(event)

    def _on_sentence_final(self, raw_text: str) -> None:
        """Called by RealtimeSTT when VAD signals end of speech."""
        if not raw_text or not raw_text.strip():
            return

        if not self._boundary.is_ready_to_finalize(raw_text):
            logger.debug("Boundary check rejected: %r", raw_text)
            return

        final_text = SentenceBoundaryDetector.normalize_final(raw_text)
        end_ms = _now_ms()

        # Snapshot audio buffer for speaker worker
        audio_span = self._snapshot_audio(
            start_ms=self._recording_start_ms, end_ms=end_ms
        )

        event = SentenceFinalizedEvent(
            session_id=self._session_id,
            segment_id=self._current_segment_id,
            start_ms=self._recording_start_ms,
            end_ms=end_ms,
            source_text_final=final_text,
            audio_span_ref=audio_span,
        )
        self._bus.publish_threadsafe(event)
        logger.info("Sentence finalized [%s]: %s", self._current_segment_id, final_text)

    # ------------------------------------------------------------------
    # Audio buffering helpers
    # ------------------------------------------------------------------

    def _snapshot_audio(self, start_ms: int, end_ms: int) -> AudioSpanRef:
        """Collect audio bytes from the internal buffer for this segment."""
        with self._audio_lock:
            audio_bytes = b"".join(self._audio_buffer)

        return AudioSpanRef(
            session_id=self._session_id,
            segment_id=self._current_segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_bytes=audio_bytes if audio_bytes else None,
            sample_rate=_SAMPLE_RATE,
        )

    def _append_audio_chunk(self, chunk: bytes) -> None:
        """Add raw PCM chunk to rolling buffer (thread-safe)."""
        if not chunk:
            return
        num_samples = len(chunk) // 2  # 16-bit = 2 bytes per sample
        with self._audio_lock:
            self._audio_buffer.append(chunk)
            self._audio_buffer_total_samples += num_samples
            # Trim old chunks if over limit
            while self._audio_buffer_total_samples > self._max_buffer_samples and self._audio_buffer:
                removed = self._audio_buffer.pop(0)
                self._audio_buffer_total_samples -= len(removed) // 2


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
