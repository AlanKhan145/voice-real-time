"""
Console renderer — subscribes to EventBus events and prints to stdout.

All output goes to stdout; logs go to stderr (via logging).
This keeps stdout clean for piping/redirecting subtitles if needed.
"""
from __future__ import annotations

import sys
import logging
from typing import Optional

from core.models import (
    PartialUpdatedEvent,
    StabilizedUpdatedEvent,
    SentenceFinalizedEvent,
    SpeakerCompletedEvent,
    TranslationCompletedEvent,
    SegmentEnrichedEvent,
    Segment,
)

logger = logging.getLogger(__name__)

# ANSI colours (disabled automatically if not a TTY)
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


CYAN = "36"
GREEN = "32"
YELLOW = "33"
MAGENTA = "35"
BLUE = "34"
BOLD = "1"
DIM = "2"


class ConsoleRenderer:
    """Handles all pretty-printing of pipeline events to stdout."""

    def __init__(self) -> None:
        self._last_partial_len = 0  # for in-place overwrite

    # ------------------------------------------------------------------
    # Event handlers (async, registered with EventBus.subscribe)
    # ------------------------------------------------------------------

    async def on_partial(self, event: PartialUpdatedEvent) -> None:
        text = event.text.strip()
        if not text:
            return
        # Overwrite the current line in-place to avoid flicker
        line = _c(DIM, f"[partial] {text}")
        if _USE_COLOR:
            # carriage return, clear to end of line
            sys.stdout.write(f"\r{line}\033[K")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r[partial] {text}")
            sys.stdout.flush()
        self._last_partial_len = len(text)

    async def on_stabilized(self, event: StabilizedUpdatedEvent) -> None:
        text = event.text.strip()
        if not text:
            return
        # Print on a new line (stabilized supersedes partial)
        self._newline_if_needed()
        print(_c(CYAN, f"[stabilized] {text}"), flush=True)

    async def on_final(self, event: SentenceFinalizedEvent) -> None:
        self._newline_if_needed()
        seg_id = _c(BOLD, event.segment_id)
        print(_c(GREEN, f"[final][{seg_id}] {event.source_text_final}"), flush=True)

    async def on_speaker_completed(self, event: SpeakerCompletedEvent) -> None:
        seg_id = _c(BOLD, event.segment_id)
        conf = f"{event.confidence:.0%}"
        print(
            _c(YELLOW, f"[speaker][{seg_id}] {event.speaker_id}  (conf {conf})"),
            flush=True,
        )

    async def on_translation_completed(self, event: TranslationCompletedEvent) -> None:
        seg_id = _c(BOLD, event.segment_id)
        print(
            _c(BLUE, f"[translation][{seg_id}] {event.translated_text}"),
            flush=True,
        )

    async def on_segment_enriched(self, event: SegmentEnrichedEvent) -> None:
        seg = event.segment
        if seg is None:
            return
        self._print_enriched(seg)

    # ------------------------------------------------------------------
    # Startup/shutdown messages
    # ------------------------------------------------------------------

    @staticmethod
    def print_banner(session_id: str) -> None:
        print("─" * 60, flush=True)
        print(_c(BOLD, " realtime-subtitle  |  local speech pipeline"), flush=True)
        print(f" session: {session_id}", flush=True)
        print("─" * 60, flush=True)

    @staticmethod
    def print_listening() -> None:
        print(_c(MAGENTA, "[recorder] listening …"), flush=True)

    @staticmethod
    def print_shutdown() -> None:
        print("\n" + _c(BOLD, "[system] graceful shutdown complete."), flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _newline_if_needed(self) -> None:
        """Ensure we start on a fresh line after an in-place partial line."""
        if self._last_partial_len > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_partial_len = 0

    @staticmethod
    def _print_enriched(seg: Segment) -> None:
        speaker = seg.speaker_id or "unknown"
        translation = seg.translated_text or "(no translation)"
        print(
            _c(MAGENTA, f"[enriched][{seg.segment_id}][{speaker}]"),
            flush=True,
        )
        print(f"  SRC : {seg.source_text_final}", flush=True)
        print(f"  TRX : {translation}", flush=True)
        print("", flush=True)
