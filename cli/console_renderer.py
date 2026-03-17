"""
Console renderer — subscribes to EventBus events and prints to stdout.

Thiết kế giao diện:
- Phụ đề hiển thị realtime (partial)
- Khi enriched: phụ đề màu theo người nói, bản dịch hiển thị bên dưới
"""
from __future__ import annotations

import re
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


# Màu cho từng người nói (speaker_1, speaker_2, ...)
SPEAKER_COLORS = ["36", "32", "33", "35", "34", "31"]  # cyan, green, yellow, magenta, blue, red
BOLD = "1"
DIM = "2"


def _speaker_color(speaker_id: Optional[str]) -> str:
    """Màu ANSI theo speaker_id."""
    if not speaker_id:
        return DIM
    m = re.match(r"speaker_(\d+)", speaker_id or "", re.I)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(SPEAKER_COLORS):
            return SPEAKER_COLORS[idx]
    return DIM


class ConsoleRenderer:
    """Giao diện hiển thị phụ đề: partial realtime, enriched block với màu speaker + dịch."""

    def __init__(self) -> None:
        self._last_partial_len = 0  # for in-place overwrite

    # ------------------------------------------------------------------
    # Event handlers (async, registered with EventBus.subscribe)
    # ------------------------------------------------------------------

    async def on_partial(self, event: PartialUpdatedEvent) -> None:
        """Phụ đề đang nhận dạng — cập nhật tại chỗ."""
        text = event.text.strip()
        if not text:
            return
        line = _c(DIM, f"  {text}")
        if _USE_COLOR:
            sys.stdout.write(f"\r{line}\033[K")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r  {text}")
            sys.stdout.flush()
        self._last_partial_len = len(text)

    async def on_stabilized(self, event: StabilizedUpdatedEvent) -> None:
        """Text ổn định — giữ partial, không in thêm."""
        pass

    async def on_final(self, event: SentenceFinalizedEvent) -> None:
        """Câu hoàn chỉnh — chờ enriched để in block (không in riêng)."""
        pass

    async def on_speaker_completed(self, event: SpeakerCompletedEvent) -> None:
        """Speaker xong — xử lý trong enriched."""
        pass

    async def on_translation_completed(self, event: TranslationCompletedEvent) -> None:
        """Dịch xong — xử lý trong enriched."""
        pass

    async def on_segment_enriched(self, event: SegmentEnrichedEvent) -> None:
        """Segment đủ speaker + dịch — in block: phụ đề (màu speaker) + dịch bên dưới."""
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
        print(_c(BOLD, " realtime-subtitle  |  phụ đề + dịch realtime"), flush=True)
        print(f" session: {session_id}", flush=True)
        print("─" * 60, flush=True)

    @staticmethod
    def print_listening() -> None:
        print(_c(DIM, "  [đang nghe…]"), flush=True)

    @staticmethod
    def print_shutdown() -> None:
        print("\n" + _c(BOLD, "[kết thúc]"), flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _newline_if_needed(self) -> None:
        """Xuống dòng sau partial in-place."""
        if self._last_partial_len > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_partial_len = 0

    def _print_enriched(self, seg: Segment) -> None:
        """In block: phụ đề (màu speaker) + dịch bên dưới."""
        self._newline_if_needed()

        speaker = seg.speaker_id or "unknown"
        color = _speaker_color(speaker)
        subtitle = seg.source_text_final or ""
        translation = seg.translated_text or ""

        # Dòng 1: phụ đề (màu theo người nói)
        label = f"[{speaker}] "
        line1 = _c(color, label + subtitle)
        print(line1, flush=True)

        # Dòng 2: bản dịch (nếu có và khác phụ đề), thụt vào cho đẹp
        if translation and translation.strip() != subtitle.strip():
            indent = " " * len(label)
            line2 = _c(DIM, indent + translation)
            print(line2, flush=True)

        print(flush=True)
