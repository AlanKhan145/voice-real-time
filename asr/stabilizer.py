"""
Text stabilizer — reduces console flicker by only emitting a new partial
when the text has changed beyond a noise threshold.

The core idea: if the new text is just the old text with minor tail edits
(common in streaming ASR), debounce it. Only forward if:
  1. The new text length grew by at least MIN_GROW chars, OR
  2. The new text prefix diverges from the previous (re-transcription).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Tune these for your preferred balance of responsiveness vs. flicker
MIN_GROW = 3          # minimum new characters before emitting update
DEBOUNCE_SEC = 0.1    # minimum time between emits


class TextStabilizer:
    def __init__(
        self,
        min_grow: int = MIN_GROW,
        debounce_sec: float = DEBOUNCE_SEC,
    ) -> None:
        self._min_grow = min_grow
        self._debounce_sec = debounce_sec
        self._last_text: str = ""
        self._last_emit_ts: float = 0.0

    def reset(self) -> None:
        self._last_text = ""
        self._last_emit_ts = 0.0

    def should_emit(self, new_text: str) -> bool:
        """Return True if the new text is worth forwarding to the renderer."""
        if not new_text.strip():
            return False

        now = time.monotonic()
        if now - self._last_emit_ts < self._debounce_sec:
            return False

        # Grew enough?
        grew = len(new_text) - len(self._last_text)
        if grew >= self._min_grow:
            return True

        # Prefix diverged (re-transcription of earlier words)?
        common_prefix = _common_prefix_len(self._last_text, new_text)
        if common_prefix < len(self._last_text) * 0.85:
            return True

        return False

    def accept(self, text: str) -> None:
        """Record that we emitted this text."""
        self._last_text = text
        self._last_emit_ts = time.monotonic()

    def update(self, text: str) -> tuple[bool, str]:
        """Convenience: check and accept in one call. Returns (emit, text)."""
        if self.should_emit(text):
            self.accept(text)
            return True, text
        return False, text


def _common_prefix_len(a: str, b: str) -> int:
    i = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        i += 1
    return i
