"""
Sentence boundary helper.

RealtimeSTT's VAD already handles most segmentation via silence detection.
This module adds a soft-rule layer on top to:
  1. Avoid finalizing mid-sentence when there's a short mid-sentence pause.
  2. Detect sentence-ending punctuation in stabilized text to signal readiness.

Usage:
    detector = SentenceBoundaryDetector()
    ready = detector.check_stabilized(text)   # True when text looks complete
    detector.reset()                           # after a final is emitted
"""
from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)

# Punctuation that strongly signals a sentence end
_END_PUNCT_RE = re.compile(r"[.!?。！？…]+\s*$")

# Short filler words / mid-sentence fragments that should NOT trigger finalize
_SHORT_FRAGMENT_MIN_WORDS = 2


class SentenceBoundaryDetector:
    """
    Lightweight rule-based detector layered on top of VAD silence.

    RealtimeSTT already fires on_recording_complete when silence is detected.
    We additionally check:
      - minimum word count (avoid single-word spurious finals)
      - trailing punctuation in stabilized text as a positive signal
    """

    def __init__(self, min_words: int = _SHORT_FRAGMENT_MIN_WORDS) -> None:
        self._min_words = min_words
        self._last_stabilized: str = ""
        self._stabilized_ts: float = 0.0

    def reset(self) -> None:
        self._last_stabilized = ""
        self._stabilized_ts = 0.0

    def update_stabilized(self, text: str) -> None:
        self._last_stabilized = text
        self._stabilized_ts = time.monotonic()

    def is_ready_to_finalize(self, candidate_text: str) -> bool:
        """
        Called with the VAD-triggered candidate text.
        Returns True if it looks like a complete sentence worth finalizing.
        """
        text = candidate_text.strip()
        if not text:
            return False
        words = text.split()
        if len(words) < self._min_words:
            logger.debug("Skipping finalize – too short (%d words): %r", len(words), text)
            return False
        return True

    @staticmethod
    def has_end_punctuation(text: str) -> bool:
        return bool(_END_PUNCT_RE.search(text.rstrip()))

    @staticmethod
    def normalize_final(text: str) -> str:
        """
        Light normalization before storing as source_text_final.
        - Capitalize first character
        - Ensure single trailing period if no other end punct
        """
        text = text.strip()
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if not _END_PUNCT_RE.search(text):
            text = text + "."
        return text
