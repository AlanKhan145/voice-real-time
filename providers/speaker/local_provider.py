"""
LocalSpeakerProvider — production-quality local speaker diarization.

Strategy (tiered, most capable first):
  1. resemblyzer  — deep speaker embeddings, tracks speakers across session
  2. energy/gap heuristic — ultra-light fallback when resemblyzer unavailable

Both strategies share the same SpeakerProvider interface.  The provider
automatically selects the best available backend at init time.
"""
from __future__ import annotations

import logging
import struct
import threading
from typing import Optional

import numpy as np

from core.models import AudioSpanRef
from providers.speaker.base import SpeakerProvider, SpeakerResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Resemblyzer backend
# ─────────────────────────────────────────────────────────────────────────────

class ResemblyzerBackend:
    """
    Uses resemblyzer's VoiceEncoder to produce speaker embeddings, then does
    nearest-neighbour matching to assign speaker labels within the session.
    """

    SIMILARITY_THRESHOLD: float = 0.75

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import]

        self._encoder = VoiceEncoder(device="cpu")
        self._preprocess = preprocess_wav
        self._threshold = similarity_threshold
        self._speaker_embeddings: dict[str, np.ndarray] = {}  # speaker_id → embedding
        self._speaker_counter = 0
        self._lock = threading.Lock()
        logger.info("ResemblyzerBackend initialised")

    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        audio = _pcm_bytes_to_float32(audio_span_ref.audio_bytes, audio_span_ref.sample_rate)
        if audio is None or len(audio) < audio_span_ref.sample_rate * 0.5:
            return self._fallback_label("too_short")

        try:
            wav = self._preprocess(audio, source_sr=audio_span_ref.sample_rate)
            embedding = self._encoder.embed_utterance(wav)
        except Exception as exc:
            logger.warning("Resemblyzer embed error: %s", exc)
            return self._fallback_label("embed_error")

        with self._lock:
            speaker_id, confidence = self._match_or_register(embedding)

        return SpeakerResult(
            speaker_id=speaker_id,
            confidence=confidence,
            provider="resemblyzer",
        )

    def _match_or_register(self, emb: np.ndarray) -> tuple[str, float]:
        best_id: Optional[str] = None
        best_sim: float = -1.0

        for sid, stored_emb in self._speaker_embeddings.items():
            sim = float(np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        if best_id is not None and best_sim >= self._threshold:
            # Update running mean for the matched speaker
            self._speaker_embeddings[best_id] = (
                0.9 * self._speaker_embeddings[best_id] + 0.1 * emb
            )
            return best_id, best_sim

        # New speaker
        self._speaker_counter += 1
        new_id = f"speaker_{self._speaker_counter}"
        self._speaker_embeddings[new_id] = emb
        logger.info("New speaker registered: %s", new_id)
        return new_id, 1.0

    @staticmethod
    def _fallback_label(reason: str) -> SpeakerResult:
        return SpeakerResult(speaker_id="speaker_unknown", confidence=0.0, provider=f"resemblyzer_fallback:{reason}")

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Energy/gap heuristic (zero extra deps)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicBackend:
    """
    When no embedding library is available.

    Heuristic: track average RMS energy per labelled speaker.
    Different speakers tend to have noticeably different microphone distances
    and vocal intensities.  Not reliable, but produces stable labelling for
    demos and tests.

    Falls back gracefully to sequential labels when audio is absent.
    """

    ENERGY_THRESHOLD: float = 0.15  # normalized RMS difference ratio

    def __init__(self, energy_threshold: float = ENERGY_THRESHOLD) -> None:
        self._threshold = energy_threshold
        self._speaker_energies: dict[str, float] = {}
        self._speaker_counter = 0
        self._last_segment_gap_ms: int = 0
        self._last_end_ms: int = 0
        self._lock = threading.Lock()
        logger.info("HeuristicBackend initialised (no resemblyzer)")

    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        rms = _compute_rms(audio_span_ref.audio_bytes)

        with self._lock:
            gap_ms = max(0, audio_span_ref.start_ms - self._last_end_ms)
            self._last_end_ms = audio_span_ref.end_ms
            speaker_id = self._assign(rms, gap_ms)

        return SpeakerResult(
            speaker_id=speaker_id,
            confidence=0.5,
            provider="heuristic",
        )

    def _assign(self, rms: float, gap_ms: int) -> str:
        if not self._speaker_energies:
            self._speaker_counter += 1
            sid = f"speaker_{self._speaker_counter}"
            self._speaker_energies[sid] = rms
            return sid

        # Large gap: possibly different speaker
        if gap_ms > 3000:
            best_id = self._closest_energy(rms)
            if best_id is None:
                self._speaker_counter += 1
                best_id = f"speaker_{self._speaker_counter}"
                self._speaker_energies[best_id] = rms
            return best_id

        best_id = self._closest_energy(rms)
        if best_id is None:
            self._speaker_counter += 1
            best_id = f"speaker_{self._speaker_counter}"
            self._speaker_energies[best_id] = rms
        else:
            # Update running mean
            self._speaker_energies[best_id] = (
                0.8 * self._speaker_energies[best_id] + 0.2 * rms
            )
        return best_id

    def _closest_energy(self, rms: float) -> Optional[str]:
        best_id = None
        best_diff = float("inf")
        for sid, avg_rms in self._speaker_energies.items():
            ref = max(avg_rms, rms, 1e-8)
            diff = abs(avg_rms - rms) / ref
            if diff < best_diff:
                best_diff = diff
                best_id = sid
        if best_diff > self._threshold:
            return None  # too different → new speaker
        return best_id

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Public facade
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpeakerProvider(SpeakerProvider):
    """
    Selects the best available backend automatically.
    Inject `similarity_threshold` to tune resemblyzer matching.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self._backend = self._pick_backend(similarity_threshold)

    @staticmethod
    def _pick_backend(similarity_threshold: float):
        try:
            import resemblyzer  # type: ignore[import]  # noqa: F401
            return ResemblyzerBackend(similarity_threshold=similarity_threshold)
        except ImportError:
            logger.info("resemblyzer not installed; using heuristic speaker backend")
            return HeuristicBackend()

    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        return self._backend.identify(audio_span_ref)

    def shutdown(self) -> None:
        self._backend.shutdown()


class NoopSpeakerProvider(SpeakerProvider):
    """Placeholder that always returns speaker_unknown. Useful in tests."""

    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        return SpeakerResult(speaker_id="speaker_unknown", confidence=1.0, provider="noop")


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pcm_bytes_to_float32(
    data: Optional[bytes], sample_rate: int
) -> Optional[np.ndarray]:
    if not data:
        return None
    try:
        n = len(data) // 2
        samples = struct.unpack(f"<{n}h", data[:n * 2])
        arr = np.array(samples, dtype=np.float32) / 32768.0
        return arr
    except Exception as exc:
        logger.debug("PCM decode error: %s", exc)
        return None


def _compute_rms(data: Optional[bytes]) -> float:
    if not data:
        return 0.0
    arr = _pcm_bytes_to_float32(data, 16000)
    if arr is None or len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))
