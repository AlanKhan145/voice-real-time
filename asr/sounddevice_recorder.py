"""
SoundDeviceRecorder — realtime ASR dùng sounddevice + faster-whisper + webrtcvad.

Thay thế RealtimeSTT (PyAudio) để tránh lỗi build trên Linux.
- sounddevice: ghi âm từ microphone (có sẵn wheel, không cần compile)
- webrtcvad: phát hiện voice activity (pure Python)
- faster-whisper: transcription
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

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

_SAMPLE_RATE = 16000
_CHANNELS = 1
_FRAME_DURATION_MS = 20  # webrtcvad yêu cầu 10, 20 hoặc 30
_FRAME_BYTES = int(_SAMPLE_RATE * _FRAME_DURATION_MS / 1000) * 2  # 16-bit
_AUDIO_BUFFER_SECONDS = 30


@dataclass
class _TranscribeTask:
    audio_bytes: bytes
    start_ms: int
    end_ms: int
    segment_id: str
    is_final: bool


class SoundDeviceRecorder:
    """
    Recorder dùng sounddevice + faster-whisper + webrtcvad.
    Không phụ thuộc PyAudio/RealtimeSTT.
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        session_id: str,
        loop,
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._session_id = session_id
        self._loop = loop

        self._stabilizer = TextStabilizer()
        self._boundary = SentenceBoundaryDetector()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # Audio buffer cho speaker worker
        _max_samples = _AUDIO_BUFFER_SECONDS * _SAMPLE_RATE
        self._audio_lock = threading.Lock()
        self._audio_buffer: list[bytes] = []
        self._audio_buffer_total_samples = 0
        self._max_buffer_samples = _max_samples

        self._recording_start_ms: int = 0
        self._current_segment_id: str = ""

        # Models (lazy load)
        self._model: Optional[WhisperModel] = None
        self._realtime_model: Optional[WhisperModel] = None
        self._model_lock = threading.Lock()

        # VAD
        self._vad = webrtcvad.Vad(config.webrtc_sensitivity)

        # Trạng thái ghi âm
        self._utterance_buffer: list[bytes] = []
        self._in_speech = False
        self._silence_frames = 0
        self._last_partial_time = 0.0

        # Queue + worker cho transcription (tránh block callback)
        self._transcribe_queue: queue.Queue[Optional[_TranscribeTask]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_run, name="transcribe-worker", daemon=True
        )
        self._worker_thread.start()
        self._thread = threading.Thread(
            target=self._run, name="sounddevice-recorder", daemon=True
        )
        self._thread.start()
        logger.info("SoundDeviceRecorder started")

    def stop(self) -> None:
        self._running = False
        self._transcribe_queue.put(None)  # unblock worker
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("Stream close exception", exc_info=True)
            self._stream = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("SoundDeviceRecorder stopped")

    def _get_model(self) -> WhisperModel:
        """Lazy load model chính cho final transcription."""
        with self._model_lock:
            if self._model is None:
                self._model = WhisperModel(
                    self._config.model_size,
                    device=self._config.device,
                    compute_type=self._config.compute_type,
                )
        return self._model

    def _get_realtime_model(self) -> WhisperModel:
        """Lazy load model nhỏ cho partial (realtime)."""
        with self._model_lock:
            if self._realtime_model is None:
                self._realtime_model = WhisperModel(
                    self._config.realtime_model_size,
                    device=self._config.device,
                    compute_type=self._config.compute_type,
                )
        return self._realtime_model

    def _run(self) -> None:
        try:
            self._bus.publish_threadsafe(
                PartialUpdatedEvent(
                    session_id=self._session_id,
                    segment_id="",
                    text="[recorder ready – listening …]",
                )
            )

            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="int16",
                blocksize=_FRAME_BYTES // 2,  # số samples
                callback=self._audio_callback,
            ) as stream:
                self._stream = stream
                logger.info("Microphone opened, entering VAD loop")

                while self._running:
                    time.sleep(_FRAME_DURATION_MS / 1000.0)

        except Exception:
            logger.exception("SoundDeviceRecorder failed")
        finally:
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Callback từ sounddevice - gọi mỗi block."""
        if status:
            logger.warning("Audio status: %s", status)
        if not self._running:
            return

        frame_bytes = indata.tobytes()
        self._append_audio_chunk(frame_bytes)

        is_speech = self._vad.is_speech(frame_bytes, _SAMPLE_RATE)

        if is_speech:
            self._utterance_buffer.append(frame_bytes)
            self._silence_frames = 0
            if not self._in_speech:
                self._in_speech = True
                self._recording_start_ms = _now_ms()
                self._current_segment_id = make_segment_id()
                self._stabilizer.reset()
                self._boundary.reset()
                logger.debug("Speech started – segment %s", self._current_segment_id)
            else:
                # Đang nói - queue partial transcription
                now = time.monotonic()
                if now - self._last_partial_time >= self._config.realtime_processing_pause:
                    self._last_partial_time = now
                    audio_bytes = b"".join(self._utterance_buffer)
                    if len(audio_bytes) >= 16000:  # >= 0.5s
                        self._transcribe_queue.put(
                            _TranscribeTask(
                                audio_bytes=audio_bytes,
                                start_ms=self._recording_start_ms,
                                end_ms=_now_ms(),
                                segment_id=self._current_segment_id,
                                is_final=False,
                            )
                        )
        else:
            if self._in_speech:
                self._silence_frames += 1
                self._utterance_buffer.append(frame_bytes)

                silence_frames_needed = int(
                    self._config.post_speech_silence_duration * 1000 / _FRAME_DURATION_MS
                )
                min_frames = int(
                    self._config.min_length_of_recording * 1000 / _FRAME_DURATION_MS
                )

                if self._silence_frames >= silence_frames_needed:
                    if len(self._utterance_buffer) >= min_frames:
                        audio_bytes = b"".join(self._utterance_buffer)
                        self._transcribe_queue.put(
                            _TranscribeTask(
                                audio_bytes=audio_bytes,
                                start_ms=self._recording_start_ms,
                                end_ms=_now_ms(),
                                segment_id=self._current_segment_id,
                                is_final=True,
                            )
                        )
                    self._in_speech = False
                    self._utterance_buffer = []
                    self._silence_frames = 0

    def _worker_run(self) -> None:
        """Worker thread: lấy task từ queue, transcribe, emit event."""
        while self._running:
            try:
                task = self._transcribe_queue.get(timeout=0.5)
                if task is None:
                    break
                if task.is_final:
                    self._process_final(task)
                else:
                    self._process_partial(task)
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Transcribe worker error")

    def _process_partial(self, task: _TranscribeTask) -> None:
        """Transcribe với model nhỏ -> partial + stabilized."""
        try:
            audio_np = np.frombuffer(task.audio_bytes, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            model = self._get_realtime_model()
            segments, _ = model.transcribe(
                audio_float,
                language=self._config.language if self._config.language != "auto" else None,
                vad_filter=False,
                beam_size=1,
            )
            text = " ".join(s.text for s in segments).strip()
            if not text:
                return
            should_emit, text = self._stabilizer.update(text)
            if should_emit:
                self._bus.publish_threadsafe(
                    PartialUpdatedEvent(
                        session_id=self._session_id,
                        segment_id=task.segment_id,
                        text=text,
                    )
                )
            self._boundary.update_stabilized(text)
            self._bus.publish_threadsafe(
                StabilizedUpdatedEvent(
                    session_id=self._session_id,
                    segment_id=task.segment_id,
                    text=text,
                )
            )
        except Exception:
            logger.debug("Partial transcription error", exc_info=True)

    def _process_final(self, task: _TranscribeTask) -> None:
        """Transcribe với model chính -> final."""
        try:
            audio_np = np.frombuffer(task.audio_bytes, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            model = self._get_model()
            segments, _ = model.transcribe(
                audio_float,
                language=self._config.language if self._config.language != "auto" else None,
                vad_filter=False,
                beam_size=5,
            )
            raw_text = " ".join(s.text for s in segments).strip()
        except Exception:
            logger.exception("Final transcription failed")
            raw_text = ""

        if not raw_text or not raw_text.strip():
            return

        if not self._boundary.is_ready_to_finalize(raw_text):
            logger.debug("Boundary check rejected: %r", raw_text)
            return

        final_text = SentenceBoundaryDetector.normalize_final(raw_text)
        audio_span = self._snapshot_audio(
            start_ms=task.start_ms, end_ms=task.end_ms
        )

        event = SentenceFinalizedEvent(
            session_id=self._session_id,
            segment_id=task.segment_id,
            start_ms=task.start_ms,
            end_ms=task.end_ms,
            source_text_final=final_text,
            audio_span_ref=audio_span,
        )
        self._bus.publish_threadsafe(event)
        logger.info("Sentence finalized [%s]: %s", task.segment_id, final_text)

    def _append_audio_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        num_samples = len(chunk) // 2
        with self._audio_lock:
            self._audio_buffer.append(chunk)
            self._audio_buffer_total_samples += num_samples
            while (
                self._audio_buffer_total_samples > self._max_buffer_samples
                and self._audio_buffer
            ):
                removed = self._audio_buffer.pop(0)
                self._audio_buffer_total_samples -= len(removed) // 2

    def _snapshot_audio(self, start_ms: int, end_ms: int) -> AudioSpanRef:
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


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
