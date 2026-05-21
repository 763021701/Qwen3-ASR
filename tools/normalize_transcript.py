#!/usr/bin/env python3
# coding=utf-8
"""
Transcript normalization per docs/normalize_label.md (conservative defaults).

Used by prepare scripts before wrapping with language None<asr_text>...
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

_ASR_TEXT_TAG = "<asr_text>"
_ZW_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_ANNOTATION_RE = re.compile(
    r"\[[^\]]*\]|\([^)]*\)|<[^>]+>|\{[^}]+\}|\b(?:speaker|spk)\s*\d+\b",
    re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds)\b",
    re.IGNORECASE,
)

Locale = Literal["english", "cantonese"]


def extract_transcript_body(text: str) -> str:
    """Strip Qwen3 training prefix if present."""
    s = (text or "").strip()
    if not s:
        return ""
    if _ASR_TEXT_TAG in s:
        return s.split(_ASR_TEXT_TAG, 1)[1].strip()
    return s


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch) not in ("Cc", "Cf"))


def _remove_unicode_punctuation(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def _global_preprocess(s: str) -> str:
    s = _nfc(s.strip())
    s = _ZW_RE.sub("", s)
    s = _ANNOTATION_RE.sub(" ", s)
    s = _TIMESTAMP_RE.sub(" ", s)
    s = _remove_control_chars(s)
    s = " ".join(s.split())
    return s


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2A6DF
        or 0xF900 <= o <= 0xFAFF
    )


def _fullwidth_to_halfwidth(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        elif o == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_english(text: str) -> str:
    """ENGLISH section + GLOBAL RULES (conservative: digits kept, no apostrophe)."""
    s = _global_preprocess(text)
    s = s.replace("'", "").replace("'", "").replace("`", "")
    s = s.lower()
    s = _remove_unicode_punctuation(s)
    s = "".join(ch if (ch.isascii() and (ch.isalnum() or ch.isspace())) else " " for ch in s)
    s = " ".join(s.split())
    return s


def normalize_cantonese(text: str) -> str:
    """CANTONESE / code-switch section + GLOBAL RULES (original script, digits kept)."""
    s = _global_preprocess(text)
    s = _fullwidth_to_halfwidth(s)
    s = _remove_unicode_punctuation(s)

    chars: list[str] = []
    for ch in s:
        if ch.isspace():
            chars.append(" ")
        elif ch.isdigit() or (ch.isascii() and ch.isalpha()):
            chars.append(ch.lower())
        elif _is_cjk(ch):
            chars.append(ch)
        else:
            chars.append(" ")

    s = "".join(chars)
    s = re.sub(r"(?<=[\u4e00-\u9fff\u3400-\u4dbf])\s+(?=[\u4e00-\u9fff\u3400-\u4dbf])", "", s)
    s = " ".join(s.split())
    return s


def normalize_transcript(text: str, locale: Locale) -> str:
    body = extract_transcript_body(text)
    if not body:
        return ""
    if locale == "english":
        return normalize_english(body)
    if locale == "cantonese":
        return normalize_cantonese(body)
    raise ValueError(f"Unknown locale: {locale!r}")
