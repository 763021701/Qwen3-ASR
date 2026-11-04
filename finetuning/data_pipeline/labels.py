# coding=utf-8
"""Label construction and deterministic sample seeds for the data pipeline."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

_ASR_TEXT_TAG = "<asr_text>"


def stable_seed(*parts: Any) -> int:
    """Return a stable 63-bit seed from arbitrary parts (unlike Python hash())."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def make_target(text: str, language: Optional[str], template: str) -> str:
    """Build the supervised target from a raw transcript and a source template."""
    value = (text or "").strip()
    if template.strip().lower() == "raw":
        if not value:
            raise ValueError("Raw transcript is empty.")
        return value
    if not value:
        raise ValueError("Transcript is empty; use text_template='raw' for nospeech rows.")
    if _ASR_TEXT_TAG in value:
        raise ValueError(
            "Transcript already contains '<asr_text>'; use text_template='raw' instead."
        )
    try:
        return template.format(text=value, language=language or "None")
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in text_template: {exc}") from exc
