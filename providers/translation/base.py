"""
TranslationProvider abstract base class.

Swap backends by implementing this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str


class TranslationProvider(ABC):
    """
    Contract:
      translate(text, source_lang, target_lang) -> TranslationResult

    The call is synchronous and blocking.  Workers run it in a thread-pool.
    """

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        ...

    def is_available(self) -> bool:
        """Optional health-check; used at startup to warn if backend is missing."""
        return True

    def shutdown(self) -> None:
        """Optional cleanup hook."""
