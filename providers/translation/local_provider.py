"""
LocalTranslationProvider — offline translation via argostranslate.

Tier strategy:
  1. argostranslate (ctranslate2 based, fully local, no cloud) — preferred
  2. NoopTranslationProvider — stub that returns the original text,
     clearly labelled so you notice it in output

argostranslate auto-downloads language packages on first use.
To pre-download offline:
    python -c "
    import argostranslate.package as pkg
    pkg.update_package_index()
    available = pkg.get_available_packages()
    pkg.install_from_path(
        next(p for p in available if p.from_code=='vi' and p.to_code=='en').download()
    )
    "
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from providers.translation.base import TranslationProvider, TranslationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: argostranslate
# ─────────────────────────────────────────────────────────────────────────────

class ArgosTranslationProvider(TranslationProvider):
    """
    Uses argostranslate for fully offline translation.
    Lazy-loads and caches translation pipelines per (src, tgt) pair.
    """

    def __init__(self) -> None:
        self._pipelines: dict[tuple[str, str], object] = {}
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import argostranslate.translate  # type: ignore[import]  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if source_lang == target_lang:
            return TranslationResult(
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                provider="argos:same_lang",
            )

        cleaned = _normalize_source(text)
        pipeline = self._get_pipeline(source_lang, target_lang)

        if pipeline is None:
            logger.warning(
                "argostranslate: no package for %s→%s, falling back to noop",
                source_lang,
                target_lang,
            )
            return TranslationResult(
                translated_text=f"[no package {source_lang}→{target_lang}] {text}",
                source_lang=source_lang,
                target_lang=target_lang,
                provider="argos:no_package",
            )

        try:
            translated = pipeline.translate(cleaned)
            return TranslationResult(
                translated_text=_normalize_translated(translated),
                source_lang=source_lang,
                target_lang=target_lang,
                provider="argos",
            )
        except Exception as exc:
            logger.error("argostranslate error: %s", exc)
            return TranslationResult(
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                provider="argos:error",
            )

    def _get_pipeline(self, source_lang: str, target_lang: str):
        key = (source_lang, target_lang)
        if key in self._pipelines:
            return self._pipelines[key]

        try:
            from argostranslate import translate as argtrans  # type: ignore[import]

            installed = argtrans.get_installed_languages()
            src_lang_obj = next((l for l in installed if l.code == source_lang), None)
            tgt_lang_obj = next((l for l in installed if l.code == target_lang), None)

            if src_lang_obj is None or tgt_lang_obj is None:
                logger.warning(
                    "argostranslate: package %s→%s not installed. "
                    "Run: python scripts/install_argos_packages.py",
                    source_lang,
                    target_lang,
                )
                self._pipelines[key] = None
                return None

            translation = src_lang_obj.get_translation(tgt_lang_obj)
            self._pipelines[key] = translation
            logger.info("argostranslate pipeline ready: %s→%s", source_lang, target_lang)
            return translation

        except Exception as exc:
            logger.error("argostranslate pipeline init error: %s", exc)
            self._pipelines[key] = None
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Noop stub
# ─────────────────────────────────────────────────────────────────────────────

class NoopTranslationProvider(TranslationProvider):
    """
    Returns the source text unchanged with a clear label.
    Use when no translation backend is available.
    """

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        return TranslationResult(
            translated_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            provider="noop",
        )

    def is_available(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_translation_provider(provider_name: str) -> TranslationProvider:
    if provider_name == "noop":
        return NoopTranslationProvider()

    # Default: try argos, fall back to noop
    argos = ArgosTranslationProvider()
    if argos.is_available():
        return argos

    logger.warning(
        "argostranslate not installed. Using noop translation provider.\n"
        "  Install: pip install argostranslate\n"
        "  Then run: python scripts/install_argos_packages.py"
    )
    return NoopTranslationProvider()


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_source(text: str) -> str:
    """Light cleanup before sending to translator."""
    # Remove trailing periods added by sentence boundary (translator adds its own)
    text = text.strip().rstrip(".")
    return text


def _normalize_translated(text: str) -> str:
    """Ensure translated output reads naturally."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    # Ensure sentence ends with punctuation
    if text and text[-1] not in ".!?":
        text += "."
    return text
